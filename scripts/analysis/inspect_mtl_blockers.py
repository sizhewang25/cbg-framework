"""Detect VPs that block MTL intersection / annulus regions.

For every FALLBACK target with error EMPTY_REGION / NO_INTERSECTION across the
combos under `--run-dir`, replays each VP's LTD constraint and counts the VPs
whose predicted `[inner, outer]` band does not contain the true VP→target
distance. Such a VP is a necessary-but-not-sufficient cause of the empty
region; high frequency + consistency across LTD families flags a probe whose
location metadata or persistent route inflation makes it unfit for CBG-style
constraints, not a calibration artifact of one model.

Skips weighted MTLs (`planar_annulus_weighted`): by design they tolerate
disagreement, so the "blocker" framing doesn't apply there.

Per-slice mode: pass `--slice fold_K` + `--inputs-dir .../<fold_K>/`.
Merged-fold mode: omit `--slice`, pass `--inputs-dir` at the parent dir whose
children are `fold_*/eval_observations.parquet` (same convention as
`inspect_cbg_vs_shortest_ping.py`).

Outputs:
  - <out-json>: structured per-combo blocker counts (+ cross-combo aggregate)
  - <out-md>:   narrative summary with top-N lists per combo
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq

from scripts.analysis._v2_io import (
    discover_combos,
    group_combos_by_id,
    load_run_json,
)
from scripts.benchmark.v2.checkpoint import load_ltd_checkpoint

# Import all LTD classes so (a) pickle.load can resolve them when reading
# fit_checkpoint.pkl and (b) the registry is populated for the stateless-LTD
# fallback path (`.stateless` marker → construct from LTD_REGISTRY).
import scripts.framework.v2.ltd.bounded_spline  # noqa: F401
import scripts.framework.v2.ltd.low_envelope  # noqa: F401
import scripts.framework.v2.ltd.normal_dist  # noqa: F401
import scripts.framework.v2.ltd.speed_of_internet  # noqa: F401
from scripts.framework.v2.registry import LTD_REGISTRY
from scripts.framework.v2.types import Coord, Latency, VpId
from scripts.libs.cbg.rtt_model import haversine_distance

logger = logging.getLogger(__name__)

_BLOCKER_MTLS = {"spherical_circle", "planar_circle", "planar_annulus"}
_BLOCKER_ERRORS = {"EMPTY_REGION", "NO_INTERSECTION"}

# LTD whose disk is calibrated to fiber-speed physics, not to a per-VP fit.
# Any blocker under this LTD is a strong probe-mislocation candidate: a correct
# probe on a real fiber path cannot have its disk exclude the truth (RTT ≥ 3d/c
# ⇒ disk = RTT·c/3 ≥ d). So the union of blockers across combos using this LTD
# is the physics-justified suspect list — no further threshold needed.
_PHYSICS_LTD = "speed_of_internet"


def _load_eval_targets(eval_parquet: Path) -> dict[str, tuple]:
    """Return `{target_id: ((true_lat, true_lon), [(vp_id, vp_lat, vp_lon, rtt), ...])}`."""
    df = pq.read_table(eval_parquet).to_pandas()
    out: dict[str, tuple] = {}
    for tid, grp in df.groupby("target_id", sort=False):
        truth = (float(grp["target_lat"].iloc[0]), float(grp["target_lon"].iloc[0]))
        obs = list(
            zip(
                grp["vp_id"].astype(str).tolist(),
                grp["vp_lat"].astype(float).tolist(),
                grp["vp_lon"].astype(float).tolist(),
                grp["latency_ms"].astype(float).tolist(),
            )
        )
        out[str(tid)] = (truth, obs)
    return out


def _load_eval_targets_by_fold(inputs_dir: Path) -> dict[str, dict[str, tuple]]:
    """Per-fold targets dict; detects fold-direct vs parent-dir layout."""
    direct = inputs_dir / "eval_observations.parquet"
    if direct.exists():
        return {inputs_dir.name: _load_eval_targets(direct)}
    paths = sorted(inputs_dir.glob("*/eval_observations.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No eval_observations.parquet at {inputs_dir} or under "
            f"{inputs_dir}/*/. Pass --inputs-dir pointing at a fold input dir "
            "or at its parent (merged-fold mode)."
        )
    return {p.parent.name: _load_eval_targets(p) for p in paths}


def _is_blocker(
    ltd_obj,
    vp_id: str,
    vp_lat: float,
    vp_lon: float,
    rtt: float,
    true_lat: float,
    true_lon: float,
) -> Optional[bool]:
    """True iff VP's predicted `[inner, outer]` does NOT bracket the truth.

    None when the VP has no fitted submodel (it contributes no constraint and
    therefore can't be a blocker).
    """
    try:
        result = ltd_obj.predict(
            VpId(vp_id), Coord(lat=vp_lat, lon=vp_lon), Latency(rtt)
        )
    except Exception:
        return None
    if not result.success or result.tg_distance is None:
        return None
    inner = result.tg_distance.lower_km
    outer = result.tg_distance.upper_km
    true_d = float(
        haversine_distance(vp_lat, vp_lon, true_lat, true_lon)
    )
    return not (inner <= true_d <= outer)


def analyze(
    run_dir: Path,
    source: Optional[str],
    setup: Optional[str],
    slice_: Optional[str],
    inputs_dir: Path,
    combos_filter: Optional[list[str]],
    errors_filter: set[str],
) -> dict:
    combo_dirs = discover_combos(
        run_dir, source=source, slice_=slice_, combos=combos_filter
    )
    if setup is not None:
        combo_dirs = [d for d in combo_dirs if setup in d.parts]
    if not combo_dirs:
        raise FileNotFoundError(
            f"No combo dirs under {run_dir} matching "
            f"source={source!r} setup={setup!r} slice={slice_!r}"
        )

    # Restrict to combos whose MTL belongs to the blocker-relevant set; also
    # record the LTD name + kwargs so we can construct a stateless LTD from
    # the registry when no checkpoint pickle was saved.
    eligible: list[Path] = []
    mtl_by_combo: dict[str, str] = {}
    ltd_meta_by_combo: dict[str, tuple[str, dict]] = {}
    for d in combo_dirs:
        try:
            meta = load_run_json(d)
        except FileNotFoundError:
            continue
        mtl = meta.get("mtl")
        if mtl in _BLOCKER_MTLS:
            eligible.append(d)
            mtl_by_combo[d.name] = mtl
            ltd_meta_by_combo[d.name] = (
                meta.get("ltd"),
                meta.get("ltd_kwargs") or {},
            )

    fold_targets = _load_eval_targets_by_fold(inputs_dir)
    grouped = group_combos_by_id(eligible)

    per_combo: dict[str, dict] = {}
    cross = Counter()

    for combo_id, dirs in grouped.items():
        n_fb = 0
        n_skipped_unmatched = 0
        blockers: Counter = Counter()
        for d in dirs:
            fold = d.parent.name
            tgt = pq.read_table(d / "targets.parquet").to_pandas()
            fb = tgt[
                (tgt["status"] == "FALLBACK")
                & tgt["error"].isin(errors_filter)
            ]
            if fb.empty:
                continue
            try:
                ltd_obj = load_ltd_checkpoint(d)
            except FileNotFoundError:
                logger.warning("No checkpoint or stateless marker at %s", d)
                continue
            if ltd_obj is None:
                # Stateless LTD — reconstruct from the registry using run.json
                # kwargs (closed-form models like speed_of_internet).
                ltd_name, ltd_kwargs = ltd_meta_by_combo[combo_id]
                ltd_cls = LTD_REGISTRY.get(ltd_name)
                if ltd_cls is None:
                    logger.warning(
                        "Stateless LTD %r at %s not in registry; skipping",
                        ltd_name, d,
                    )
                    continue
                ltd_obj = ltd_cls(**ltd_kwargs)
            fold_dict = fold_targets.get(fold, {})
            if not fold_dict:
                n_skipped_unmatched += int(len(fb))
                continue
            for tid in fb["target_id"].astype(str):
                target = fold_dict.get(tid)
                if target is None:
                    n_skipped_unmatched += 1
                    continue
                n_fb += 1
                (t_lat, t_lon), obs = target
                for vp_id, v_lat, v_lon, rtt in obs:
                    flag = _is_blocker(
                        ltd_obj, vp_id, v_lat, v_lon, rtt, t_lat, t_lon
                    )
                    if flag:
                        blockers[vp_id] += 1
        per_combo[combo_id] = {
            "mtl": mtl_by_combo[combo_id],
            "total_fallback_targets": n_fb,
            "skipped_unmatched_targets": n_skipped_unmatched,
            "blockers": dict(blockers),
        }
        cross.update(blockers)

    # Suspect list: probes that block at least one FALLBACK target under any
    # combo whose LTD is the speed-of-internet bound. Tracks per-combo evidence
    # for each suspect so a reader can see which combo(s) flagged it.
    suspect_evidence: dict[str, dict[str, int]] = {}
    suspect_combos = sorted(
        cid for cid, (ltd, _) in ltd_meta_by_combo.items()
        if ltd == _PHYSICS_LTD
    )
    for combo_id in suspect_combos:
        for vp, count in per_combo[combo_id]["blockers"].items():
            suspect_evidence.setdefault(vp, {})[combo_id] = count
    suspect_probes = sorted(
        (
            {
                "vp_id": vp,
                "total_blocks": sum(by_combo.values()),
                "by_combo": by_combo,
            }
            for vp, by_combo in suspect_evidence.items()
        ),
        key=lambda r: (-r["total_blocks"], r["vp_id"]),
    )

    return {
        "run_dir": str(run_dir),
        "source": source,
        "setup": setup,
        "slice": slice_,
        "errors_scanned": sorted(errors_filter),
        "mtl_families_scanned": sorted(_BLOCKER_MTLS),
        "physics_ltd": _PHYSICS_LTD,
        "suspect_source_combos": suspect_combos,
        "suspect_probes": suspect_probes,
        "per_combo": per_combo,
        "cross_combo_aggregate": dict(cross),
    }


def render_markdown(data: dict, top_n: int) -> str:
    lines: list[str] = []
    scope = data.get("slice") or "merged across folds"
    lines.append("# MTL blockers — per-probe non-bracketing tally")
    lines.append("")
    lines.append(f"- run dir: `{data['run_dir']}`")
    lines.append(f"- source: `{data.get('source')}`, setup: `{data.get('setup')}`")
    lines.append(f"- scope: **{scope}**")
    lines.append(
        f"- errors scanned: {', '.join(f'`{e}`' for e in data['errors_scanned'])}"
    )
    lines.append(
        f"- MTL families scanned: {', '.join(f'`{m}`' for m in data['mtl_families_scanned'])} "
        f"(weighted MTLs intentionally skipped)"
    )
    lines.append("")
    lines.append(
        "A VP is counted as a *blocker* for a FALLBACK target when its predicted "
        "`[inner, outer]` LTD band does not contain the true VP→target distance. "
        "Necessary-but-not-sufficient: when an MTL region is empty, at least one "
        "VP must fail to bracket the truth, but more than one may share blame. "
        "High percentages + consistency across multiple combos flag genuinely "
        "problematic probes (mislocated, or persistent route inflation), not "
        "calibration artifacts of one LTD."
    )
    lines.append("")

    # --- Suspect probes (headline) -------------------------------------------
    suspects = data.get("suspect_probes", [])
    src_combos = data.get("suspect_source_combos", [])
    physics_ltd = data.get("physics_ltd", _PHYSICS_LTD)
    lines.append("## Suspect probes (physics-justified prune list)")
    lines.append("")
    lines.append(
        f"Union of blockers across combos whose LTD is `{physics_ltd}` — the "
        "fiber-calibrated disk. A correct probe on a real fiber path cannot "
        "have its disk exclude the truth (RTT ≥ 3·d/c ⇒ disk ≥ d), so any "
        "blocker here is a strong mislocation candidate. No frequency "
        "threshold applied: the physics already does the filtering."
    )
    lines.append("")
    if src_combos:
        lines.append(
            "Source combos: " + ", ".join(f"`{c}`" for c in src_combos)
        )
    else:
        lines.append(
            f"_No combo with `{physics_ltd}` LTD was scanned — the suspect "
            "list is empty by construction._"
        )
    lines.append("")
    if not suspects:
        lines.append("**No suspect probes found.**")
        lines.append("")
    else:
        lines.append(f"**{len(suspects)} suspect probe(s):**")
        lines.append("")
        lines.append("| Rank | Probe IP | Total blocks | Per-combo evidence |")
        lines.append("|---:|:---|---:|:---|")
        for i, rec in enumerate(suspects, 1):
            ev = ", ".join(
                f"`{c}`: {n}" for c, n in sorted(rec["by_combo"].items())
            )
            lines.append(
                f"| {i} | `{rec['vp_id']}` | {rec['total_blocks']} | {ev} |"
            )
        lines.append("")

    per_combo = data["per_combo"]
    if not per_combo:
        lines.append("No FALLBACK targets in scope — nothing to report.")
        return "\n".join(lines) + "\n"

    lines.append("## Top blockers per combo")
    lines.append("")
    for combo_id in sorted(per_combo):
        rec = per_combo[combo_id]
        n_fb = rec["total_fallback_targets"]
        lines.append(f"### `{combo_id}` (MTL: `{rec['mtl']}`)")
        lines.append("")
        lines.append(f"FALLBACK targets scanned: **{n_fb}**")
        if rec["skipped_unmatched_targets"]:
            lines.append(
                f"  · {rec['skipped_unmatched_targets']} skipped (no matching "
                "eval_observations row)"
            )
        if n_fb == 0:
            lines.append("")
            lines.append("No FALLBACK targets — nothing blocked.")
            lines.append("")
            continue
        ranked = sorted(
            rec["blockers"].items(), key=lambda kv: (-kv[1], kv[0])
        )[:top_n]
        if not ranked:
            lines.append("")
            lines.append("No non-bracketing VPs found — region collapse from "
                         "polygonization noise.")
            lines.append("")
            continue
        lines.append("")
        lines.append("| Rank | Probe IP | Blocks | % of FALLBACK |")
        lines.append("|---:|:---|---:|---:|")
        for i, (vp, c) in enumerate(ranked, 1):
            pct = 100.0 * c / n_fb if n_fb else 0.0
            lines.append(f"| {i} | `{vp}` | {c} | {pct:.0f}% |")
        lines.append("")

    cross = data["cross_combo_aggregate"]
    if cross:
        lines.append("## Cross-combo aggregate (top {n})".format(n=top_n))
        lines.append("")
        lines.append(
            "Sum across all scanned combos. Probes ranked high *here* are "
            "blockers under multiple LTD families — strong evidence the probe "
            "metadata / path is the issue, not the model."
        )
        lines.append("")
        lines.append("| Rank | Probe IP | Total non-bracket-events |")
        lines.append("|---:|:---|---:|")
        for i, (vp, c) in enumerate(
            sorted(cross.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n], 1
        ):
            lines.append(f"| {i} | `{vp}` | {c} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Benchmark run dir (parent of <source>/<setup>/<fold>/<combo>/).")
    p.add_argument("--source", default=None)
    p.add_argument("--setup", default=None)
    p.add_argument("--slice", dest="slice_", default=None,
                   help="Per-slice mode. Omit for merged across folds.")
    p.add_argument("--inputs-dir", type=Path, required=True,
                   help="Fold input dir or its parent (merged mode).")
    p.add_argument("--combos", nargs="*", default=None,
                   help="Restrict to these combo_ids.")
    p.add_argument("--errors", nargs="*", default=sorted(_BLOCKER_ERRORS),
                   help="MTL error codes to treat as FALLBACK-with-blockers.")
    p.add_argument("--top-n", type=int, default=20,
                   help="Top-N blockers per combo in the markdown.")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-md", type=Path, required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data = analyze(
        run_dir=args.run_dir,
        source=args.source,
        setup=args.setup,
        slice_=args.slice_,
        inputs_dir=args.inputs_dir,
        combos_filter=args.combos,
        errors_filter=set(args.errors),
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(data, indent=2, sort_keys=True))

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(data, top_n=args.top_n))

    print(f"Wrote:\n  {args.out_json}\n  {args.out_md}")


if __name__ == "__main__":
    main()
