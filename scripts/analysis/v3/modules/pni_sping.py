"""Does Shortest-Ping fail where the target sits far from an interconnect?

§8.1's explanation of the baseline's near-perfect classification is a three-link
chain: traffic weighting keeps targets *at* PNI locations, min-RTT then ranks
VPs by proximity to the serving site where the target co-locates, and nearest-
answer snapping turns that VP's coordinate into the right class. The third link
is exact rather than approximate — under nearest-seed snapping the baseline is
correct precisely when its VP shares an answer region with the target — so the
whole chain reduces to a claim about **co-location**.

## Why the successes cannot test it

When the target sits at a site, `d(sping VP, PNI)` and `d(sping VP, TG)` are the
same number, so "the shortest-ping VP is near the interconnect" and "the
shortest-ping VP is near the target" are the *same observation*. The mechanism
is unfalsifiable exactly where it works, and under §7.3's own premise that is
most of the population. Any statistic pooled over all targets is therefore
dominated by the cases that cannot discriminate.

The two hypotheses only separate where the target is **far** from every listed
site. There the naive account predicts the baseline still wins (the closest VP
is still closest), while the PNI account predicts it loses, because min-RTT is
then ranking VPs by proximity to a site the target is not at. So the test is a
trend in the **failures**:

    does Shortest-Ping's error rate rise with `tg_to_nearest_pni_km`?

and the headline is a continuous rank correlation over targets, computed from
`target_nodes.csv`'s distance and never from the bins, which exist only so a
table can be printed.

## Two controls, because the obvious confound is real

`tg_to_nearest_pni_km` is correlated with "rural", which is correlated with VP
sparsity, seed margin and cell size. A rising baseline error rate could simply
be "targets far from carrier hotels are far from everything".

So **every method is scored over the same strata**, not just the baseline: a
trend present in all of them is target difficulty rather than a Shortest-Ping
mechanism, and the reader can see which it is on one table. And
`sping_failure_strata.csv` carries `tg_seed_margin_km`, `tg_seed_nearest_vp_km`
and the §8.2 taxonomy shares per stratum, so the confound sits on the same page
as the claim rather than in a reviewer's question.

## No distance threshold

The co-location rate is exact answer-region membership
(`proximity.has_proximate_sping_vp`), not a kilometre cutoff, and
`tg_to_nearest_pni_km` stays a continuous explanatory variable.
`sping_error_ecdf.csv` gives the full distribution of `d(sping VP, TG)` split by
whether the baseline was right, which is the threshold-free form of "the answer
VP is close to the target".

Correctness comes from `diagram.common.membership.build_membership` and nothing
else, so fallbacks count as failures (§7.2) and every method is scored over one
target set. `has_proximate_sping_vp` is the same quantity as `shortest_ping`
top-1 correctness *by construction*, so their disagreement count is a pipeline
self-check: it is reported rather than raised, following `proximity`'s stance
that a violation is a statement about the data and a stack trace would hide it.

Command: `breakdown-sping-pni`. Writes `sping_*` files into
`outputs/analysis/v3/<run_id>/target-cls-accuracy/<grid>-<resolution>/`, a
directory it shares with `classify` and `breakdown-accuracy` — which is why its
JSON is `sping_pni_manifest.json` and not `meta.json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules import pni
from scripts.analysis.v3.modules.bipartite import (
    CDF_QUANTILES,
    cdf_column,
    quantile_bins,
    spearman,
)
from scripts.analysis.v3.modules.breakdown import taxonomy_of
from scripts.analysis.v3.modules.classify import DEFAULT_TOPN, SHORTEST_PING
from scripts.analysis.v3.modules.diagram.common.labels import label_for
from scripts.analysis.v3.modules.diagram.common.membership import (
    available_methods,
    build_membership,
)
from scripts.analysis.v3.modules.grid import (
    DEFAULT_GRID,
    GRID_HELP,
    RESOLUTION_HELP,
    SWEEP_HELP,
    resolve_cli_grid,
)
from scripts.analysis.v3.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    MissingArtifactError,
    RunPaths,
    discover_runs,
    resolve_run,
)
from scripts.analysis.v3.modules.proximity import TAXONOMY, load_proximity

ACCURACY_CSV = "sping_accuracy_by_pni_distance.csv"
STRATA_CSV = "sping_failure_strata.csv"
ECDF_CSV = "sping_error_ecdf.csv"
MANIFEST_JSON = "sping_pni_manifest.json"

#: Quantile strata for `tg_to_nearest_pni_km`. Presentation only — the reported
#: trend is the continuous rank correlation, which never reads these. Quantile
#: rather than fixed km because a hand-picked cut on the explanatory variable
#: would be choosing where to find the result.
_N_PNI_DISTANCE_BINS = 4

#: Tolerance on the two independently-carried copies of `d(sping VP, TG)`.
#: `pni-graph` rounds at `_KM_DIGITS`, `target-proximity` does not, so half of
#: 10**-3 is the largest disagreement that is pure rounding.
_SPING_KM_TOL = 1e-3

#: The ECDF populations. Split by outcome because the claim is about the
#: *distribution* of the answer VP's distance, and pooling the two hides exactly
#: the contrast that makes it a claim.
ECDF_POPULATIONS = ("all", "shortest_ping_correct", "shortest_ping_wrong")

_MECHANISM_NOTE = (
    "When the target sits at a site, d(sping VP, PNI) and d(sping VP, TG) are "
    "the same number, so the PNI account and the naive 'closest VP is closest' "
    "account make identical predictions. The mechanism is invisible in the "
    "successes; only the far-from-PNI stratum can discriminate, which is why "
    "the reported statistic is a trend in the errors."
)
_CONTROL_NOTE = (
    "Every method is scored over the same strata. A trend present in all of "
    "them is target difficulty -- tg_to_nearest_pni_km correlates with VP "
    "sparsity and seed margin -- not a Shortest-Ping mechanism."
)
_BIN_NOTE = (
    "error_rate_trend is a rank correlation over targets, computed from "
    "tg_to_nearest_pni_km directly. It does not read the bins, so perturbing "
    "the bin edges cannot move it; the bins exist to print a table."
)
_TAUTOLOGY_NOTE = (
    "true by construction, not by measurement: under nearest-seed snapping the "
    "baseline predicts its VP's coordinate, and has_proximate_sping_vp asks "
    "whether that VP's nearest seed is the target's. A pipeline self-check."
)


@dataclass(frozen=True)
class SpingPniBreakdown:
    """The three tables and the manifest, ready to write."""

    accuracy: pd.DataFrame
    strata: pd.DataFrame
    ecdf: pd.DataFrame
    manifest: dict

    def write(self, out_dir: Path) -> Path:
        """Write with `sping_` names into a directory this module does not own."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.accuracy.to_csv(out_dir / ACCURACY_CSV, index=False)
        self.strata.to_csv(out_dir / STRATA_CSV, index=False)
        self.ecdf.to_csv(out_dir / ECDF_CSV, index=False)
        (out_dir / MANIFEST_JSON).write_text(json.dumps(self.manifest, indent=2) + "\n")
        return out_dir


def join_targets(labels: pd.DataFrame, target_nodes: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Proximity labels plus the PNI graph's per-target distances.

    A left join on the proximity frame, so the population stays the one the
    classification was scored over. Targets the PNI graph does not carry keep
    their row with a NaN distance and are counted rather than dropped: a silent
    change of denominator is the failure mode this whole module is guarding
    against elsewhere.
    """
    want = ["target_id", "tg_to_nearest_pni_km", "tg_nearest_pni_id"]
    if "sping_vp_to_tg_km" in target_nodes.columns:
        want.append("sping_vp_to_tg_km")
    right = target_nodes[[c for c in want if c in target_nodes.columns]].copy()
    right = right.rename(columns={"sping_vp_to_tg_km": "pni_sping_vp_to_tg_km"})
    joined = labels.merge(right, on="target_id", how="left", validate="one_to_one")
    missing = int(joined["tg_to_nearest_pni_km"].isna().sum())
    return joined, missing


def accuracy_rows(
    membership: dict[int, pd.DataFrame], joined: pd.DataFrame, edges_km: list[float]
) -> pd.DataFrame:
    """One row per (method, top_n, stratum), with a pooled stratum per method."""
    strat = joined.set_index("target_id")["tg_to_nearest_pni_km_stratum"]
    rows = []
    for top_n, frame in sorted(membership.items()):
        for method in frame.columns:
            hits = frame[method]
            bins = strat.reindex(hits.index)
            populations = [("pooled", "pooled", np.nan, np.nan, hits)]
            for b in sorted(bins.dropna().unique()):
                sub = hits[bins == b]
                populations.append(
                    (
                        int(b),
                        "pni_distance_bin",
                        round(float(edges_km[int(b)]), 3),
                        round(float(edges_km[int(b) + 1]), 3),
                        sub,
                    )
                )
            for stratum, kind, lo, hi, sub in populations:
                n = int(len(sub))
                n_correct = int(sub.sum())
                rows.append(
                    {
                        "method": method,
                        "method_label": label_for(method),
                        "top_n": int(top_n),
                        "stratum": stratum,
                        "stratum_kind": kind,
                        "tg_to_nearest_pni_km_lo": lo,
                        "tg_to_nearest_pni_km_hi": hi,
                        "n_targets": n,
                        "n_correct": n_correct,
                        "accuracy": round(n_correct / n, 6) if n else np.nan,
                        "error_rate": round(1 - n_correct / n, 6) if n else np.nan,
                        "share_of_targets": round(n / len(hits), 6) if len(hits) else np.nan,
                        # The pooled baseline cell at top-1 equals the answer-region
                        # co-location rate by construction; see _TAUTOLOGY_NOTE.
                        "is_tautological": bool(
                            method == SHORTEST_PING and top_n == 1 and stratum == "pooled"
                        ),
                        "zero_variance": bool(n and (n_correct == 0 or n_correct == n)),
                    }
                )
    return pd.DataFrame(rows)


def strata_rows(joined: pd.DataFrame, taxonomy: pd.Series, edges_km: list[float]) -> pd.DataFrame:
    """One row per stratum of method-free target properties: the confound panel."""
    tax = taxonomy.reindex(joined["target_id"].to_numpy())
    frame = joined.assign(_term=tax.to_numpy())

    def block(sub: pd.DataFrame, *, stratum, kind, lo, hi) -> dict:
        n = int(len(sub))
        row = {
            "stratum": stratum,
            "stratum_kind": kind,
            "tg_to_nearest_pni_km_lo": lo,
            "tg_to_nearest_pni_km_hi": hi,
            "n_targets": n,
            "share_of_targets": round(n / len(frame), 6) if len(frame) else np.nan,
            "tg_to_nearest_pni_km_p50": _p(sub["tg_to_nearest_pni_km"], 50),
            "answer_region_colocation_rate": _mean(sub["has_proximate_sping_vp"]),
            "sping_vp_to_tg_km_p50": _p(sub["sping_vp_to_tg_km"], 50),
            "sping_vp_to_tg_km_p90": _p(sub["sping_vp_to_tg_km"], 90),
            "tg_seed_margin_km_p50": _p(sub["tg_seed_margin_km"], 50),
            "tg_seed_nearest_vp_km_p50": _p(sub["tg_seed_nearest_vp_km"], 50),
            "has_proximate_vp_share": _mean(sub["has_proximate_vp"]),
            "has_proximate_sping_vp_share": _mean(sub["has_proximate_sping_vp"]),
            "n_measured_vps_p50": _p(sub["n_measured_vps"], 50),
        }
        for term in TAXONOMY:
            row[f"{term}_share"] = round(float((sub["_term"] == term).mean()), 6) if n else np.nan
        return row

    rows = [block(frame, stratum="pooled", kind="pooled", lo=np.nan, hi=np.nan)]
    for b in sorted(frame["tg_to_nearest_pni_km_stratum"].dropna().unique()):
        rows.append(
            block(
                frame[frame["tg_to_nearest_pni_km_stratum"] == b],
                stratum=int(b),
                kind="pni_distance_bin",
                lo=round(float(edges_km[int(b)]), 3),
                hi=round(float(edges_km[int(b) + 1]), 3),
            )
        )
    return pd.DataFrame(rows)


def ecdf_rows(joined: pd.DataFrame, correct: pd.Series) -> pd.DataFrame:
    """`d(sping VP, TG)` as an ECDF, split by whether the baseline was right."""
    hit = correct.reindex(joined["target_id"].to_numpy()).to_numpy(dtype=bool)
    pops = {
        "all": joined,
        "shortest_ping_correct": joined[hit],
        "shortest_ping_wrong": joined[~hit],
    }
    frames = []
    for name in ECDF_POPULATIONS:
        sub = pops[name]
        frames.append(
            pd.DataFrame(
                {
                    "population": name,
                    "n_targets": int(len(sub)),
                    "quantile": CDF_QUANTILES,
                    "sping_vp_to_tg_km": cdf_column(sub["sping_vp_to_tg_km"]),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _p(values, q: int) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return round(float(np.percentile(v, q)), 3) if v.size else float("nan")


def _mean(values) -> float:
    v = np.asarray(values, dtype=bool)
    return round(float(v.mean()), 6) if v.size else float("nan")


def build_breakdown(
    labels: pd.DataFrame,
    target_nodes: pd.DataFrame,
    membership: dict[int, pd.DataFrame],
    *,
    run_id: str | None = None,
    grid_meta: dict | None = None,
    cls_dir: Path | None = None,
    prox_dir: Path | None = None,
    pni_graph_dir: Path | None = None,
) -> SpingPniBreakdown:
    """Cross every method's correctness with distance-to-nearest-interconnect."""
    joined, n_missing = join_targets(labels, target_nodes)
    strat, edges_km = quantile_bins(joined["tg_to_nearest_pni_km"], n_bins=_N_PNI_DISTANCE_BINS)
    joined["tg_to_nearest_pni_km_stratum"] = strat.to_numpy()

    taxonomy = taxonomy_of(labels)
    accuracy = accuracy_rows(membership, joined, edges_km)
    strata = strata_rows(joined, taxonomy, edges_km)

    top1 = membership[min(membership)]
    baseline = (
        top1[SHORTEST_PING]
        if SHORTEST_PING in top1.columns
        else pd.Series(False, index=top1.index)
    )
    ecdf = ecdf_rows(joined, baseline)

    manifest = {
        "run_id": run_id,
        "grid": grid_meta or {},
        "inputs": {
            "classification": str(cls_dir) if cls_dir else None,
            "proximity": str(prox_dir) if prox_dir else None,
            "pni_graph": str(pni_graph_dir) if pni_graph_dir else None,
        },
        "methods": list(top1.columns),
        "topn_reported": sorted(int(n) for n in membership),
        "n_targets": int(len(joined)),
        "n_targets_without_a_pni_row": n_missing,
        "strata": {
            "n_bins_requested": _N_PNI_DISTANCE_BINS,
            "edges_km": [round(float(e), 3) for e in edges_km],
            "binning": "bipartite.quantile_bins over tg_to_nearest_pni_km",
            "note": _BIN_NOTE,
        },
        "mechanism": _mechanism_block(membership, joined),
        "colocation": _colocation_block(joined, baseline),
        "checks": _checks_block(joined),
        "correctness_policy": (
            "membership comes from diagram.common.membership.build_membership, so "
            "fallbacks count as failures (7.2) and every method is scored over "
            "the same target set"
        ),
        "no_km_threshold_policy": (
            "co-location is exact answer-region membership, not a kilometre "
            "cutoff, and tg_to_nearest_pni_km stays continuous. The only "
            "kilometre numbers here are reported distributions."
        ),
        "tautological_cell": {
            "method": SHORTEST_PING,
            "top_n": 1,
            "stratum": "pooled",
            "note": _TAUTOLOGY_NOTE,
        },
    }
    return SpingPniBreakdown(accuracy=accuracy, strata=strata, ecdf=ecdf, manifest=manifest)


def _mechanism_block(membership: dict[int, pd.DataFrame], joined: pd.DataFrame) -> dict:
    """The trend, per method, as a rank correlation that never reads the bins."""
    km = joined.set_index("target_id")["tg_to_nearest_pni_km"]
    out: dict = {"note": _MECHANISM_NOTE, "control": _CONTROL_NOTE, "error_rate_trend": {}}
    for top_n, frame in sorted(membership.items()):
        block = {}
        for method in frame.columns:
            wrong = (~frame[method]).astype(float)
            d = km.reindex(wrong.index)
            usable = np.isfinite(d.to_numpy(dtype=float))
            block[method] = {
                "spearman_wrong_vs_tg_to_nearest_pni_km": round(
                    spearman(wrong.to_numpy()[usable], d.to_numpy(dtype=float)[usable]), 6
                ),
                "n_targets": int(usable.sum()),
                "n_wrong": int(wrong.sum()),
            }
        out["error_rate_trend"][f"top_{top_n}"] = block
    return out


def _colocation_block(joined: pd.DataFrame, baseline: pd.Series) -> dict:
    """The exact link-3 statistic, plus its tautology self-check."""
    flag = joined.set_index("target_id")["has_proximate_sping_vp"].astype(bool)
    hit = baseline.reindex(flag.index).astype(bool)
    return {
        "answer_region_colocation_rate": _mean(flag),
        "source": "target-proximity/target_labels.csv has_proximate_sping_vp",
        "self_check": {
            "shortest_ping_top1_accuracy": _mean(hit),
            "n_disagreements_with_shortest_ping_membership": int((flag != hit).sum()),
            "note": (
                _TAUTOLOGY_NOTE
                + " A nonzero disagreement means proximity's nearest-seed rule and "
                "classify's have drifted; it is counted, not raised, so the number "
                "reaches the reader instead of a stack trace."
            ),
        },
    }


def _checks_block(joined: pd.DataFrame) -> dict:
    """Agreement between the two independently-carried copies of one distance."""
    out = {
        "sping_km_tolerance": _SPING_KM_TOL,
        "note": (
            "d(sping VP, TG) is carried by both pni-graph/target_nodes.csv and "
            "target-proximity/target_labels.csv, and both trace to "
            "io.load_sping_vp. This module reads proximity's; the count below is "
            "how often the two disagree by more than rounding."
        ),
    }
    if "pni_sping_vp_to_tg_km" in joined.columns:
        a = joined["sping_vp_to_tg_km"].to_numpy(dtype=float)
        b = joined["pni_sping_vp_to_tg_km"].to_numpy(dtype=float)
        both = np.isfinite(a) & np.isfinite(b)
        diff = np.abs(a[both] - b[both])
        out["n_sping_vp_to_tg_km_disagreements"] = int((diff > _SPING_KM_TOL).sum())
        out["max_abs_sping_vp_to_tg_km_disagreement"] = (
            round(float(diff.max()), 9) if diff.size else 0.0
        )
        out["n_compared"] = int(both.sum())
    else:
        out["n_sping_vp_to_tg_km_disagreements"] = None
        out["note"] += " pni-graph carried no sping_vp_to_tg_km, so no check ran."
    return out


def build_for_run(
    run: RunPaths,
    *,
    analysis_root: Path | None = None,
    grid: str = DEFAULT_GRID,
    resolution: int | None = None,
    methods: list[str] | None = None,
    ns: tuple[int, ...] = DEFAULT_TOPN,
    pni_graph: Path | None = None,
) -> tuple[SpingPniBreakdown, Path]:
    """Resolve the three inputs at one quantization, build, and say where to write."""
    cls_dir = run.cls_accuracy_dir(root=analysis_root, grid=grid, resolution=resolution)
    prox_dir = run.proximity_dir(root=analysis_root, grid=grid, resolution=resolution)
    graph_dir = Path(pni_graph) if pni_graph else run.pni_graph_dir(root=analysis_root)

    prox = load_proximity(prox_dir)
    graph = pni.load_pni_graph(graph_dir)
    chosen = methods or available_methods(cls_dir)
    membership = {n: build_membership(cls_dir, chosen, top_n=n) for n in ns}

    result = build_breakdown(
        prox.labels,
        graph.target_nodes,
        membership,
        run_id=run.run_id,
        grid_meta=prox.meta.get("grid", {}),
        cls_dir=cls_dir,
        prox_dir=prox_dir,
        pni_graph_dir=graph_dir,
    )
    return result, cls_dir


def load_breakdown(path: Path) -> SpingPniBreakdown:
    """Read back a written artifact, or say which command writes it."""
    path = Path(path)
    missing = [f for f in (ACCURACY_CSV, STRATA_CSV, ECDF_CSV) if not (path / f).exists()]
    if missing:
        raise MissingArtifactError(
            f"{path} is missing {missing}; run `cli breakdown-sping-pni --run-id <run>` first"
        )
    manifest = path / MANIFEST_JSON
    return SpingPniBreakdown(
        accuracy=pd.read_csv(path / ACCURACY_CSV),
        strata=pd.read_csv(path / STRATA_CSV),
        ecdf=pd.read_csv(path / ECDF_CSV),
        manifest=json.loads(manifest.read_text()) if manifest.exists() else {},
    )


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("breakdown-sping-pni")
    def breakdown_sping_pni_cmd(
        run_id: str = typer.Option(
            None, help="Run to break down. Omit with --all-runs to do every run."
        ),
        all_runs: bool = typer.Option(
            False, "--all-runs", help="Break down every run under --outputs-root."
        ),
        grid: str = typer.Option(DEFAULT_GRID, "--grid", help=GRID_HELP),
        resolution: list[int] = typer.Option([], "--resolution", "-r", help=RESOLUTION_HELP),
        sweep: bool = typer.Option(False, "--sweep", help=SWEEP_HELP),
        method: list[str] = typer.Option(
            None, "--method", help="Restrict to these method ids (repeatable)."
        ),
        topn: str = typer.Option(
            ",".join(str(n) for n in DEFAULT_TOPN),
            help="Comma-separated Ns to report. The PNI strata stay the same "
            "regardless: they describe the dataset, not the scoring.",
        ),
        pni_graph: Path = typer.Option(
            None, "--pni-graph", help="A pni-graph/ directory to read instead of this run's own."
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """Cross each method's correctness with distance to the nearest PNI (§8.1).

        Writes sping_accuracy_by_pni_distance.csv, sping_failure_strata.csv,
        sping_error_ecdf.csv and sping_pni_manifest.json into
        target-cls-accuracy/<grid>-<resolution>/. Needs `classify`,
        `build-proximity` and `build-pni-graph` to have run first.
        """
        if all_runs == (run_id is not None):
            raise typer.BadParameter("pass exactly one of --run-id or --all-runs")
        if all_runs and pni_graph is not None:
            raise typer.BadParameter("--pni-graph cannot be combined with --all-runs")

        g, resolutions = resolve_cli_grid(grid, resolution, sweep=sweep)
        ns = tuple(int(n) for n in topn.split(",") if n.strip())
        runs = discover_runs(outputs_root) if all_runs else [resolve_run(run_id, outputs_root)]
        for run in runs:
            for want_res in resolutions:
                result, out_dir = build_for_run(
                    run,
                    analysis_root=analysis_root,
                    grid=g.name,
                    resolution=want_res,
                    methods=method or None,
                    ns=ns,
                    pni_graph=pni_graph,
                )
                result.write(out_dir)
                trend = result.manifest["mechanism"]["error_rate_trend"]["top_1"]
                sp = trend.get(SHORTEST_PING, {})
                typer.echo(
                    f"{run.run_id}: {result.manifest['n_targets']} targets; "
                    f"shortest-ping error trend vs d(TG,nearest PNI) rho="
                    f"{sp.get('spearman_wrong_vs_tg_to_nearest_pni_km', float('nan')):.3f} "
                    f"-> {out_dir}"
                )
