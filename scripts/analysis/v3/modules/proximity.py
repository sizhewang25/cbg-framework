"""Per-target VP proximity labels: why a target was, or was not, geolocatable.

The stratification layer §8.1 crosses every method's correctness against. Where
`build-bipartite-graph` describes the dataset's geometry with RTT deliberately
excluded, this module lets RTT back in for exactly one purpose — identifying the
**shortest-ping VP** — and asks a different question: for each target, was there
a vantage point in a position to answer it, and was that the vantage point the
baseline actually picked?

## Every distance is measured VP -> seed, never VP -> target

This is the design's load-bearing decision and the one that is easy to get
wrong, because the convenient quantity and the meaningful one differ.

Classification labels a coordinate by its **nearest seed**. So the guarantee
worth having is about the *class*, not about the target:

    d(VP, S) < margin(S)  =>  S is that VP's nearest seed

where `margin(S)` is half the distance from `S` to the nearest other seed. The
proof is one line: for any other seed `S'`, the triangle inequality gives
`d(VP, S') >= d(S, S') - d(VP, S) > 2*margin - margin = margin > d(VP, S)`.

Measure to the *target* instead and the implication fails, by exactly the
target's own `cell_offset_km` (p50 16-20 km at `h3-4`) — which is the same order
as `margin` itself. v2's `eval_source` measures to the target, so its
`closest_vp_km` / `has_vp_proximity` are **not** what is recomputed here; see
[SCHEMA.md](../SCHEMA.md) for the column-by-column mapping.

## The ladder is a diamond, not a chain

Four booleans on two axes, both in **top-1 context**:

                    has_proximate_vp
                   /                \\
    has_discriminative_vp        has_proximate_sping_vp
                   \\                /
                  has_discriminative_sping_vp

* **argmin axis** (`has_proximate_*`) — the classifier's own rule: `tg_seed` is
  this VP's nearest seed, i.e. rank 0.
* **half-gap axis** (`has_discriminative_*`) — the strict-guarantee rule above.
  Tighter: it implies rank 0 but is not implied by it.
* **left column** — an existence claim over every *measured* VP.
* **right column** — a claim about the one VP the baseline designated.

`4 => 2 => 1` and `4 => 3 => 1`, but 2 and 3 are incomparable: geography-strength
and routing-strength are separate axes that meet at the top. That is why the
3-level `proximity_label` of v2 is dropped rather than carried — a total order
cannot express an incomparable pair.

§8.2's taxonomy is the diamond's argmin chain, cut twice, and needs no column of
its own:

| term | flags | what the target requires |
| --- | --- | --- |
| `geometry_only` | `~has_proximate_vp` | no measured VP resolves the class; only multilateration can |
| `selection_miss` | `has_proximate_vp & ~has_proximate_sping_vp` | a VP resolves it, the baseline picked another — the CBG opportunity |
| `selection_hit` | `has_proximate_sping_vp` | the baseline's own VP resolves it |

`geometry_only` is a ceiling on **Shortest-Ping, not on CBG**, and the name says
so on purpose. On as02 its 40 targets are exactly two seeds whose nearest
measured VP sits 92 km and 153 km out with `tg_seed_best_rank == 1` throughout —
the true seed is always the runner-up. Shortest-Ping and Vanilla get 0 of 40;
Octant-Hull gets 39. Calling that stratum a "structural failure" would file the
case that most favours CBG under a heading that says CBG cannot win it.

## `has_proximate_sping_vp` is tautologically Shortest-Ping's top-1 correctness

By construction, and deliberately so. The baseline predicts its VP's own
coordinate and classification is argmin-over-seeds, so the flag and the score
are the same computation on the same input. It is kept because it is the
*self-check*: if the two ever disagree, the rank rule here and `classify`'s have
drifted. Consumers must annotate that cell rather than report it as a finding.
At top-3 the same cell is informative, since the flag stays top-1.

The shortest-ping VP's **identity and coordinate come from `eval_source`**, not
from re-minimizing RTT here, precisely so the tautology holds exactly: it is the
VP `classify` scores the baseline on, and a second derivation would break ties
differently. The RTT-derived agreement is recorded in `meta.json` as a
diagnostic instead.

## Observed only

Rows describe **measured** VPs. No latent (whole-roster) companion columns are
emitted: the latent/observed pairing already has an owner in `bipartite.py`
(`measured_nearest_vp_ratio_per_target`), and two copies of one fact in two
directories are free to disagree. Geographic opportunity alone is also not the
question — RTT and peering decide whether proximity is *usable*.

Command: `build-proximity`. Writes to
`outputs/analysis/v3/<run_id>/target-proximity/<grid>-<resolution>/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules import io
from scripts.analysis.v3.modules.answer_space import (
    AnswerSpace,
    elementwise_km,
    load_answer_space,
    pairwise_km,
)
from scripts.analysis.v3.modules.bipartite import describe_p90, resolve_source_csv
from scripts.analysis.v3.modules.grid import (
    DEFAULT_GRID,
    GRID_HELP,
    RESOLUTION_HELP,
    SWEEP_HELP,
    get_grid,
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

LABELS_CSV = "target_labels.csv"
META_JSON = "meta.json"

#: The four flags, ordered top-of-diamond first. Consumers iterate this rather
#: than hard-coding names, so adding an axis is one edit.
FLAGS = (
    "has_proximate_vp",
    "has_discriminative_vp",
    "has_proximate_sping_vp",
    "has_discriminative_sping_vp",
)

#: The diamond's implications, as `(antecedent, consequent)` pairs. Pinned by a
#: test and re-checked at write time: a violation means the rank rule and the
#: half-gap rule have stopped agreeing, which invalidates every stratum below.
IMPLICATIONS = (
    ("has_discriminative_vp", "has_proximate_vp"),
    ("has_discriminative_sping_vp", "has_proximate_sping_vp"),
    ("has_proximate_sping_vp", "has_proximate_vp"),
    ("has_discriminative_sping_vp", "has_discriminative_vp"),
)

#: Rank ties resolve in the true seed's favour, matching `classify`'s rule so
#: the two layers agree on a VP sitting exactly between two seeds.
_TIE_EPS = 1e-9

_TAUTOLOGY_NOTE = (
    "has_proximate_sping_vp is Shortest-Ping's top-1 correctness by "
    "construction (same VP coordinate, same argmin-over-seeds rule). Report it "
    "as a pipeline self-check, never as a result. At top-3 it is informative, "
    "because the flag stays top-1."
)

_SEED_DISTANCE_NOTE = (
    "Every flag thresholds a VP-to-SEED distance. VP-to-target distances "
    "(closest_vp_to_tg_km, sping_vp_to_tg_km) are carried for context and are "
    "deliberately not what the flags test: a min over VP-to-target selects a "
    "different VP than a min over VP-to-seed, and only the latter implies the "
    "VP's nearest seed is the target's."
)

#: §8.2's three terms, named for what the target *requires* rather than for a
#: verdict. `geometry_only` replaced an earlier `structural_failure`: measured
#: on as02, Octant-Hull answers 39 of the 40 targets in that stratum correctly
#: while Shortest-Ping and Vanilla answer none, so reading it as "unanswerable"
#: is wrong — it is unanswerable *by proximity*, which is the regime CBG exists
#: for. The v2 vocabulary (`NO_PROXIMITY` / `HAS_*_USED_PROXIMITY`) is
#: deliberately not reused: those are cluster-keyed and VP-to-target, and a
#: near-identical spelling would invite mixing the two runs' numbers.
TAXONOMY = ("geometry_only", "selection_miss", "selection_hit")

_TAXONOMY_NOTE = (
    "the diamond's argmin chain, cut twice; no separate column is emitted. "
    "geometry_only = no measured VP resolves the class, so only multilateration "
    "can (a ceiling on Shortest-Ping, NOT on CBG). selection_miss = a VP "
    "resolves it but the baseline picked another -- the CBG opportunity. "
    "selection_hit = the baseline's own VP resolves it."
)

_OBSERVED_NOTE = (
    "Observed only: every column is over MEASURED (VP, target) edges. The "
    "latent/observed pair lives in bipartite-graph/, which owns it."
)


@dataclass(frozen=True)
class ProximityLabels:
    """The per-target label table plus the diagnostics describing it."""

    labels: pd.DataFrame
    meta: dict

    def write(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.labels.to_csv(out_dir / LABELS_CSV, index=False)
        (out_dir / META_JSON).write_text(json.dumps(self.meta, indent=2) + "\n")
        return out_dir


def seed_rank_matrix(d_vp_seed: np.ndarray) -> np.ndarray:
    """`R[v, k]` = how many seeds sit strictly closer to VP `v` than seed `k`.

    The same "strictly closer, ties to the true seed" rule `classify` applies to
    predictions, applied here to VPs. `V x K` is small on every run (134 x 22 at
    most), so the full matrix is materialized rather than argpartitioned.
    """
    return (d_vp_seed[:, None, :] < d_vp_seed[:, :, None] - _TIE_EPS).sum(axis=2)


def _argmin_by_group(
    values: np.ndarray, group_codes: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-group minimum of `values` and the row index attaining it.

    A stable lexsort rather than a `groupby.idxmin`, so ties resolve on row
    order — deterministic given a deterministic edge order, which the canonical
    CSV's dedup provides.
    """
    order = np.lexsort((values, group_codes))
    first = np.ones(order.size, dtype=bool)
    first[1:] = group_codes[order][1:] != group_codes[order][:-1]
    idx = np.full(n_groups, -1, dtype=int)
    best = np.full(n_groups, np.nan)
    hit = order[first]
    idx[group_codes[hit]] = hit
    best[group_codes[hit]] = values[hit]
    return best, idx


def build_proximity(
    space: AnswerSpace,
    vps: pd.DataFrame,
    obs: pd.DataFrame,
    sping: pd.DataFrame,
    *,
    context: pd.DataFrame | None = None,
    source_label: str | None = None,
    source_csv: str | None = None,
) -> ProximityLabels:
    """Label every answer-space target with the proximity diamond.

    `obs` is the canonical CSV as `eval_source.load_canonical_csv` returns it —
    one row per observation, its NaN and non-positive-RTT filter already
    applied, so this describes the graph the benchmark ran on.

    `sping` is `io.load_sping_vp`'s frame: the diamond's right chain, and a
    **required** input rather than something derived here, so this module and
    `classify` cannot disagree about which VP the baseline picked.

    `context` is optional and supplies carried columns that no flag depends on
    (`min_inflation`). Absent, they come back NaN — the labels stand without
    them, which is what keeps the required input set to "the answer space, the
    VPs, the edges, and the baseline's choice".

    Node sets are pinned to the run's own inputs exactly as `build_bipartite`
    pins them: VPs are `vps.csv`'s roster, targets are the answer space's
    `assignments`. Edges naming anything outside those are dropped and counted.
    A target with no surviving edge keeps a row — it is part of the denominator
    every §8.1 rate is taken over — with NaN scalars and all four flags False.
    """
    seeds = space.seeds
    targets = space.assignments.reset_index(drop=True)
    vps = vps.drop_duplicates("vp_id").reset_index(drop=True)

    # --- edges, reconciled against the run's node sets --------------------
    edges = (
        obs.groupby(["target_id", "vp_id"], sort=True)["rtt_ms"]
        .min()
        .reset_index()
    )
    known_vp = edges["vp_id"].isin(set(vps["vp_id"]))
    known_tg = edges["target_id"].isin(set(targets["target_id"]))
    dropped = {
        "edges_naming_an_unknown_vp": int((~known_vp).sum()),
        "edges_naming_an_unknown_target": int((known_vp & ~known_tg).sum()),
    }
    edges = edges.loc[known_vp & known_tg].reset_index(drop=True)
    if edges.empty:
        raise ValueError(
            "no edge survives the join against vps.csv and the answer space; "
            "the CSV and the run do not describe the same graph"
        )

    # --- VP x seed geometry, computed once --------------------------------
    d_vp_seed = pairwise_km(
        vps["vp_lat"].to_numpy(dtype=float),
        vps["vp_lon"].to_numpy(dtype=float),
        seeds["seed_lat"].to_numpy(dtype=float),
        seeds["seed_lon"].to_numpy(dtype=float),
    )
    rank_vp_seed = seed_rank_matrix(d_vp_seed)
    seed_pos = {int(s): i for i, s in enumerate(seeds["seed_id"].to_numpy())}

    tg_pos = pd.Index(targets["target_id"])
    vp_pos = pd.Index(vps["vp_id"])
    e_tg = tg_pos.get_indexer(edges["target_id"]).astype(int)
    e_vp = vp_pos.get_indexer(edges["vp_id"]).astype(int)

    tg_seed_col = np.array(
        [seed_pos[int(s)] for s in targets["seed_id"].to_numpy()], dtype=int
    )
    # Each edge, scored against *its own target's* seed.
    e_seed_col = tg_seed_col[e_tg]
    e_d_to_seed = d_vp_seed[e_vp, e_seed_col]
    e_rank_of_seed = rank_vp_seed[e_vp, e_seed_col]

    n_tg = len(targets)
    nearest_km, nearest_row = _argmin_by_group(e_d_to_seed, e_tg, n_tg)
    best_rank, best_rank_row = _argmin_by_group(
        e_rank_of_seed.astype(float), e_tg, n_tg
    )

    # Context: closest measured VP by distance to the **target**, which selects
    # a different VP than the seed-keyed minimum above whenever the target sits
    # off-centre in its cell. Carried, never thresholded.
    e_d_to_tg = elementwise_km(
        vps["vp_lat"].to_numpy(dtype=float)[e_vp],
        vps["vp_lon"].to_numpy(dtype=float)[e_vp],
        targets["target_lat"].to_numpy(dtype=float)[e_tg],
        targets["target_lon"].to_numpy(dtype=float)[e_tg],
    )
    closest_to_tg_km, _ = _argmin_by_group(e_d_to_tg, e_tg, n_tg)

    vp_ids = vps["vp_id"].to_numpy()

    def _vp_at(edge_rows: np.ndarray) -> np.ndarray:
        """VP id of the winning *edge*, or "" where the target has no edge.

        `_argmin_by_group` indexes the edge list, not the VP roster, so the hop
        through `e_vp` is required — indexing `vp_ids` with an edge row is an
        out-of-bounds read the moment the edge count exceeds the roster size.
        """
        out = np.full(n_tg, "", dtype=object)
        ok = edge_rows >= 0
        out[ok] = vp_ids[e_vp[edge_rows[ok]]]
        return out

    # --- the shortest-ping VP, as classify resolved it --------------------
    sping = _sping_frame(sping, targets, vps, seeds, seed_pos)

    margin = (
        targets["seed_id"]
        .map(seeds.set_index("seed_id")["margin_km"])
        .to_numpy(dtype=float)
    )

    labels = pd.DataFrame(
        {
            "target_id": targets["target_id"].to_numpy(),
            "tg_seed_id": targets["seed_id"].to_numpy(),
            "tg_seed_margin_km": np.round(margin, 3),
            "tg_seed_best_rank": np.where(
                np.isfinite(best_rank), best_rank, -1
            ).astype(int),
            "tg_seed_best_rank_vp_id": _vp_at(best_rank_row),
            "tg_seed_nearest_vp_km": np.round(nearest_km, 3),
            "tg_seed_nearest_vp_id": _vp_at(nearest_row),
            "sping_vp_id": sping["vp_id"],
            "sping_vp_tg_seed_rank": sping["rank"],
            "sping_vp_to_tg_seed_km": np.round(sping["to_seed_km"], 3),
        }
    )

    labels["has_proximate_vp"] = labels["tg_seed_best_rank"].to_numpy() == 0
    labels["has_discriminative_vp"] = nearest_km < margin
    labels["has_proximate_sping_vp"] = sping["rank"] == 0
    labels["has_discriminative_sping_vp"] = sping["to_seed_km"] < margin

    labels["n_measured_vps"] = np.bincount(e_tg, minlength=n_tg)
    labels["closest_vp_to_tg_km"] = np.round(closest_to_tg_km, 3)
    labels["sping_vp_to_tg_km"] = np.round(sping["to_tg_km"], 3)
    labels["min_inflation"] = _carry(context, targets["target_id"], "min_inflation")

    violations = _check_implications(labels)
    meta = _meta(
        labels,
        space=space,
        n_vps=len(vps),
        dropped=dropped,
        n_obs=int(len(obs)),
        sping_diag=sping["diagnostic"],
        violations=violations,
        source_label=source_label,
        source_csv=source_csv,
    )
    return ProximityLabels(labels=labels, meta=meta)


def _carry(
    context: pd.DataFrame | None, target_id: pd.Series, column: str
) -> np.ndarray:
    """One optional context column, aligned to the answer space's target order.

    Carried rather than recomputed where v2 owns the definition —
    `min_inflation` needs `eval_source`'s theoretical-slope constant, and
    forking it here would put two constants behind one name. Missing input or
    missing column both yield NaN: no flag reads these, so their absence
    degrades the table rather than invalidating it.
    """
    if context is None or column not in context.columns:
        return np.full(len(target_id), np.nan)
    src = context.drop_duplicates("target_id").set_index("target_id")[column]
    return pd.Series(target_id.to_numpy()).map(src).to_numpy(dtype=float)


def _sping_frame(
    sping: pd.DataFrame,
    targets: pd.DataFrame,
    vps: pd.DataFrame,
    seeds: pd.DataFrame,
    seed_pos: dict[int, int],
) -> dict:
    """The right chain: the baseline's VP, its `tg_seed` rank and seed distance.

    `sping` is `io.load_sping_vp`'s frame — the **same** resolution `classify`
    scores the Shortest-Ping baseline on, which is what makes
    `has_proximate_sping_vp` an exact self-check rather than an approximate one.
    See that function for why the VP is read from `eval_source` instead of being
    re-minimized over RTT here.

    Its **coordinate**, not the roster's, drives the distances, for the same
    reason. A VP absent from the roster therefore still yields a rank — computed
    from its own coordinate against the seeds — while `sping_vp_id` is reported
    verbatim. That is the honest reading: the baseline did make a prediction
    there, and hiding it would understate the baseline rather than the dataset.
    """
    ev = sping.set_index("target_id")
    tid = pd.Series(targets["target_id"].to_numpy())
    lat = tid.map(ev["sping_vp_lat"]).to_numpy(dtype=float)
    lon = tid.map(ev["sping_vp_lon"]).to_numpy(dtype=float)
    vp_id = tid.map(ev["sping_vp_id"]).fillna("").astype(str).to_numpy()

    has = np.isfinite(lat) & np.isfinite(lon)
    d = pairwise_km(
        np.where(has, lat, 0.0),
        np.where(has, lon, 0.0),
        seeds["seed_lat"].to_numpy(dtype=float),
        seeds["seed_lon"].to_numpy(dtype=float),
    )
    rank_all = seed_rank_matrix(d)
    col = np.array([seed_pos[int(s)] for s in targets["seed_id"].to_numpy()], dtype=int)
    rows = np.arange(len(tid))
    to_seed = np.where(has, d[rows, col], np.nan)
    rank = np.where(has, rank_all[rows, col], -1).astype(int)

    to_tg = tid.map(ev["sping_vp_to_tg_km"]).to_numpy(dtype=float)
    # `sping_vp_to_tg_km` is optional upstream; recompute where it is absent so
    # the context column is never silently empty on a run that carries the VP.
    gap = has & ~np.isfinite(to_tg)
    if gap.any():
        to_tg = to_tg.copy()
        to_tg[gap] = elementwise_km(
            lat[gap],
            lon[gap],
            targets["target_lat"].to_numpy(dtype=float)[gap],
            targets["target_lon"].to_numpy(dtype=float)[gap],
        )
    return {
        "vp_id": vp_id,
        "rank": rank,
        "to_seed_km": to_seed,
        "to_tg_km": to_tg,
        "diagnostic": {
            "source": "io.load_sping_vp (eval_source shortest_ping_vp_*)",
            "n_targets_without_a_sping_vp": int((~has).sum()),
            "n_sping_vps_outside_the_roster": int(
                (~pd.Series(vp_id).isin(set(vps["vp_id"])) & has).sum()
            ),
            "n_sping_vp_to_tg_km_recomputed": int(gap.sum()),
            "note": (
                "the same reader classify scores the baseline on, so the flag "
                "and the score cannot disagree on a tie"
            ),
        },
    }


def _check_implications(labels: pd.DataFrame) -> dict:
    """Count rows where the diamond's nesting fails. Zero on a healthy run.

    Reported rather than asserted: a violation is a real geometric statement
    about the answer space (a half-gap that no longer bounds the Voronoi cell),
    and silently raising would hide it behind a stack trace.
    """
    out = {}
    for ante, cons in IMPLICATIONS:
        bad = int((labels[ante] & ~labels[cons]).sum())
        if bad:
            out[f"{ante} without {cons}"] = bad
    return out


def _meta(
    labels: pd.DataFrame,
    *,
    space: AnswerSpace,
    n_vps: int,
    dropped: dict,
    n_obs: int,
    sping_diag: dict,
    violations: dict,
    source_label: str | None,
    source_csv: str | None,
) -> dict:
    """Base rates, the four-flag cross-tab, and the zero-variance warnings.

    Every rate ships with its cell count. A flag that is constant on a run is a
    fact about that run's VP fleet, not an absence of effect, and a bare 0.0 or
    1.0 reads as the latter — `zero_variance` names it explicitly so a consumer
    cannot report "no separation" where the correct statement is "no contrast to
    separate on".
    """
    n = len(labels)
    rates = {}
    zero_variance = []
    for flag in FLAGS:
        k = int(labels[flag].sum())
        rates[flag] = {"n_true": k, "n_false": n - k, "share": round(k / n, 4) if n else None}
        if n and k in (0, n):
            zero_variance.append(flag)

    cross = (
        labels.groupby(list(FLAGS), dropna=False)
        .size()
        .reset_index(name="n_targets")
        .sort_values("n_targets", ascending=False)
    )
    return {
        "source": source_label,
        "source_csv": source_csv,
        "grid": {
            "scheme": str(space.seeds["grid_scheme"].iloc[0]),
            "resolution": int(space.seeds["grid_resolution"].iloc[0]),
        },
        "n_targets": n,
        "n_seeds": space.n_seeds,
        "n_vps": n_vps,
        "n_obs": n_obs,
        "edges_dropped": dropped,
        "top_n_context": 1,
        "flags": rates,
        "zero_variance": zero_variance,
        "cross_tab": cross.to_dict(orient="records"),
        "implication_violations": violations,
        "argmin_vs_half_gap": {
            "n_proximate_but_not_discriminative_vp": int(
                (labels["has_proximate_vp"] & ~labels["has_discriminative_vp"]).sum()
            ),
            "n_proximate_but_not_discriminative_sping_vp": int(
                (
                    labels["has_proximate_sping_vp"]
                    & ~labels["has_discriminative_sping_vp"]
                ).sum()
            ),
            "note": (
                "the half-gap rule is strictly tighter than the argmin rule, so "
                "these counts are the price of the guarantee, not a disagreement"
            ),
        },
        "section_8_2_taxonomy": {
            "geometry_only": int((~labels["has_proximate_vp"]).sum()),
            "selection_miss": int(
                (labels["has_proximate_vp"] & ~labels["has_proximate_sping_vp"]).sum()
            ),
            "selection_hit": int(labels["has_proximate_sping_vp"].sum()),
            "note": _TAXONOMY_NOTE,
        },
        "distances": {
            "tg_seed_margin_km": describe_p90(labels["tg_seed_margin_km"]),
            "tg_seed_nearest_vp_km": describe_p90(labels["tg_seed_nearest_vp_km"]),
            "sping_vp_to_tg_seed_km": describe_p90(labels["sping_vp_to_tg_seed_km"]),
            "closest_vp_to_tg_km": describe_p90(labels["closest_vp_to_tg_km"]),
        },
        "degree": {"n_measured_vps": describe_p90(labels["n_measured_vps"])},
        "shortest_ping_vp": sping_diag,
        "notes": {
            "seed_keyed": _SEED_DISTANCE_NOTE,
            "observed_only": _OBSERVED_NOTE,
            "tautology": _TAUTOLOGY_NOTE,
        },
    }


def load_proximity(path: Path) -> ProximityLabels:
    """Read back what `write` produced."""
    path = Path(path)
    labels_p = path / LABELS_CSV
    if not labels_p.exists():
        raise MissingArtifactError(
            f"{path} has no {LABELS_CSV}; run `cli build-proximity` first"
        )
    meta_p = path / META_JSON
    return ProximityLabels(
        labels=pd.read_csv(labels_p),
        meta=json.loads(meta_p.read_text()) if meta_p.exists() else {},
    )


def build_for_run(
    run: RunPaths,
    *,
    analysis_root: Path | None = None,
    grid=DEFAULT_GRID,
    resolution: int | None = None,
    answer_space: Path | None = None,
    source_csv: Path | None = None,
) -> tuple[ProximityLabels, Path]:
    """`build_proximity` for one run. Returns the labels and their output dir."""
    from scripts.benchmark.v2.eval_source import load_canonical_csv

    g = get_grid(grid) if isinstance(grid, str) else grid
    res = g.DEFAULT_RESOLUTION if resolution is None else g.validate_resolution(resolution)
    space_dir = answer_space or run.answer_space_dir(
        root=analysis_root, grid=g.name, resolution=res
    )
    space = load_answer_space(space_dir)

    csv_path = resolve_source_csv(run, source_csv)
    result = build_proximity(
        space,
        io.load_vps(run),
        load_canonical_csv(Path(csv_path)),
        io.load_sping_vp(run),
        context=io.load_eval_per_target(run),
        source_label=f"{run.run_id}/{run.source}/{run.setup}",
        source_csv=str(csv_path),
    )
    out_dir = run.proximity_dir(
        root=analysis_root,
        grid=str(space.seeds["grid_scheme"].iloc[0]),
        resolution=int(space.seeds["grid_resolution"].iloc[0]),
    )
    return result, out_dir


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("build-proximity")
    def build_proximity_cmd(
        run_id: str = typer.Option(
            None, help="Run to label. Omit with --all-runs to do every run."
        ),
        all_runs: bool = typer.Option(
            False, "--all-runs", help="Label every run under --outputs-root."
        ),
        answer_space: Path = typer.Option(
            None,
            help="Answer-space dir (from build-answer-space). Defaults to this "
                 "run's target-answer-space/<grid>-<resolution>/ under "
                 "--analysis-root.",
        ),
        source_csv: Path = typer.Option(
            None,
            help="Canonical (vp_id, target_id, rtt_ms) CSV. Defaults to the path "
                 "the run's eval_stats.json records.",
        ),
        grid: str = typer.Option(DEFAULT_GRID, "--grid", help=GRID_HELP),
        resolution: list[int] = typer.Option(
            [], "--resolution", "-r", help=RESOLUTION_HELP
        ),
        sweep: bool = typer.Option(False, "--sweep", help=SWEEP_HELP),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """Label each target with the four-flag VP proximity diamond.

        Writes target_labels.csv + meta.json into
        target-proximity/<grid>-<resolution>/. Every flag thresholds a
        VP-to-seed distance, in top-1 context.
        """
        if all_runs == (run_id is not None):
            raise typer.BadParameter("pass exactly one of --run-id or --all-runs")
        if all_runs and answer_space is not None:
            raise typer.BadParameter("--answer-space cannot be combined with --all-runs")
        if all_runs and source_csv is not None:
            raise typer.BadParameter("--source-csv cannot be combined with --all-runs")

        g, resolutions = resolve_cli_grid(grid, resolution, sweep=sweep)
        runs = discover_runs(outputs_root) if all_runs else [resolve_run(run_id, outputs_root)]
        jobs = (
            [(r, None) for r in runs]
            if answer_space is not None
            else [(r, x) for r in runs for x in resolutions]
        )
        for run, want_res in jobs:
            result, out_dir = build_for_run(
                run,
                analysis_root=analysis_root,
                grid=g,
                resolution=want_res,
                answer_space=answer_space,
                source_csv=source_csv,
            )
            result.write(out_dir)
            f = result.meta["flags"]
            warn = (
                f" · no variance: {', '.join(result.meta['zero_variance'])}"
                if result.meta["zero_variance"]
                else ""
            )
            typer.echo(
                f"{run.run_id}: {result.meta['n_targets']} targets · "
                + " · ".join(
                    f"{k.removeprefix('has_')}={f[k]['n_true']}" for k in FLAGS
                )
                + f"{warn} -> {out_dir}"
            )
