"""The dataset as a bipartite graph: VP nodes, target nodes, measured edges.

Second step of the v3 pipeline, after `build-answer-space`. That command read
only the target side; this one reads the **VPs** and the **measured (VP, target)
edges**, and computes paper §7.3's metric list against the *grid* answer space
the accuracy numbers are actually scored on.

**No RTT enters and no variant runs.** RTT is used for exactly one thing — the
canonical CSV's own row filter (`rtt_ms > 0`, i.e. "was this pair measured at
all") — and is then dropped. That is what makes these numbers the fixed
reference the RTT-dependent characterization of §8.2 and the per-variant results
of §8.3 are read against, rather than another result competing with them.

**Every VP-to-target distance is reported as a pair.** §7.3 asks for two values
for each: the *latent* one over all VP x target pairs, describing where the
infrastructure sits, and the *observed* one over measured edges, describing what
the dataset can actually deliver. An edge set is an artifact of the campaign
rather than a property of the deployment, so reporting one half without the
other is what lets sampling bias be misread as an algorithmic result. The two
are therefore **nested** in `meta.json` (`length_km.{observed,latent}`,
`nearest_vp_km.{observed,latent}`) rather than sitting as siblings, so neither
can be read with the other out of view.

The pairing applies to distances and **not to degree**: a target's latent degree
is the VP count for every target, so it carries nothing, and `meta.json` says so
rather than emitting a constant column.

`measurement_efficiency` is what the pair exists to support — nearest-measured-VP
over nearest-VP per target, `>= 1` by construction. It is the one number that
separates "the VP set is badly placed" (a large latent nearest-VP distance) from
"the VP set is fine but the campaign allocated probes badly" (a ratio above 1),
and only the second is fixable by reallocating measurement.

**Angular geometry stays observed-only**, matching §7.3's own scoping of it to
measured neighbours: the arrangement term describes the constraints a variant
actually receives, and a latent bearing set no method ever sees would not be
that.

> Out of scope, per the run decision: §7.3's optional appendix (Clark-Evans,
> anisotropy, nearest-neighbour CV, degree assortativity) and §8.2's VP coverage
> ceiling.

**Scale.** The latent half is `|VP| x |targets|`, which at the paper's deployment
setting is not materializable. The per-target nearest-VP distance is therefore
**exact at any target count** — `nearest_across_km` chunks the cross matrix and
reduces with a per-chunk `min`, so only `|VP| x chunk` floats are resident. Only
the latent *distribution*, which needs the values rather than their minima, falls
back to a deterministic target subsample, and says so in `meta.json` when it does.

**Why not read `eval_dataset/<basename>_dataset_stats.json`,** which SCHEMA.md
says has "the §7.3 metric list, mostly precomputed"? Because its answer-space
half is keyed to the benchmark's older radius-capped `clusters/` space, not to
the grid, and the two are not comparable (README, "The answer space"). Its
grid-free blocks *are* comparable, and the run's `eval_stats.json` is used as a
cross-check in the tests rather than as an input.

Command: `build-bipartite-graph`. Writes to
`outputs/analysis/v3/<run_id>/bipartite-graph/<grid>-<resolution>/`.
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
    _describe,
    elementwise_km,
    load_answer_space,
    pairwise_km,
)
from scripts.analysis.v3.modules.grid import (
    DEFAULT_GRID,
    GRID_HELP,
    RESOLUTION_HELP,
    SWEEP_HELP,
    Grid,
    get_grid,
    resolve_cli_grid,
)
from scripts.analysis.v3.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    REPO_ROOT,
    MissingArtifactError,
    RunPaths,
    discover_runs,
    resolve_run,
)

VP_NODES_CSV = "vp_nodes.csv"
TARGET_NODES_CSV = "target_nodes.csv"
EDGE_SEGMENTS_CSV = "edge_segments.csv"
EDGE_LENGTH_CDF_CSV = "edge_length_cdf.csv"
PAIRWISE_CDF_CSV = "pairwise_distance_cdf.csv"
META_JSON = "meta.json"

#: Quantiles written to the two CDF artifacts. §7.3 wants "the full pairwise
#: distance CDF" as the figure behind the diameter and p95 scalars; 101 points is
#: enough to draw it and small enough to read, and it fixes a grid so two
#: datasets' CDFs can be differenced row by row.
_CDF_QUANTILES = np.round(np.linspace(0.0, 1.0, 101), 3)

#: Above this node count a full pairwise matrix is subsampled rather than built.
#: The four runs here are 53-134 VPs and 78-458 targets, so this never fires; it
#: exists because the paper's deployment setting is million-IP scale and an
#: O(n^2) matrix must fail into a documented approximation rather than into the
#: OOM killer.
_MAX_PAIRWISE_NODES = 5_000

#: Seed for that subsample, so a capped run is still reproducible.
_PAIRWISE_SEED = 20260903

#: Targets per chunk of the VP x target cross matrix. The latent nearest-VP
#: distance is exact at any target count because the reduction is a per-chunk
#: `min` over VPs, so only `|VP| x chunk` floats are ever resident. That matters:
#: the paper's deployment setting is million-IP scale, where the full cross
#: matrix is not materializable but its row minima are.
_CROSS_CHUNK_TARGETS = 4096

#: Latent VP-target pair distances retained for the *distribution*. Unlike the
#: row minima above, a distribution needs the values themselves, so past this
#: many pairs the target side is subsampled (deterministically) and said so.
#: At 134 VPs that is ~37k targets before it fires; the four runs here are
#: 53k-61k pairs, three orders of magnitude clear of it.
_MAX_CROSS_PAIRS = 5_000_000

#: Relative slack around a measurement-efficiency of exactly 1.0, sized from the
#: residue rather than guessed.
#:
#: The observed minimum comes from a groupby over per-edge `elementwise_km`
#: (an `einsum` row product) and the latent one from `pairwise_km`'s matrix
#: product, so the same pair reaches the same metre by two code paths that
#: differ in the last bits. Measured on as03, twenty targets came out
#: 2.36e-9 km (2.4 micrometres, 1.24e-9 relative) apart, and for every one of
#: them the latent-nearest VP *did* carry an edge — so a 1e-9 threshold reported
#: 438 of 458 targets as having missed their nearest VP when the true answer is
#: all 458.
#:
#: 1e-6 relative is ~800x that residue and still ~1000x tighter than the
#: smallest excess a genuine miss could produce, since a real alternative VP is
#: kilometres away rather than micrometres.
_EFFICIENCY_TOL = 1e-6

#: Decimal places coordinates are rounded to before flow segments are collapsed.
#: 5 dp is ~1 m, finer than any input here (the canonical CSVs carry <= 6 dp) and
#: far below the ~45 km cell the answer space merges at, so this cannot merge two
#: places that the rest of the pipeline keeps apart.
_SEGMENT_COORD_DP = 5

_MEASUREMENT_EFFICIENCY_NOTE = (
    "One value per target: nearest-measured-VP km over nearest-VP km. 1.0 "
    "means the campaign measured the geometrically nearest VP; larger means it did not. "
    "Separates 'the VP set is badly placed' (a large latent nearest-VP "
    "distance) from 'the VP set is fine but the campaign allocated probes "
    "badly' (a ratio above 1), and only the second is fixable by reallocating "
    "measurement. Undefined where a VP sits exactly on the target but carries "
    "no edge (the ratio is infinite); those targets are excluded, so `n` here is "
    "below the target count by exactly that many."
)

DISPERSION_NOTE = (
    "Extent says how far apart the set reaches; this says whether it is spread "
    "or stacked inside that reach, at this file's own resolution. "
    "effective_count is the number of distinct places the set resolves to (the "
    "occupied-cell count of §7.3) and occupancy_ratio is effective_count / "
    "count -- 1.0 means every node is its own place, low means many share one. "
    "For §7.3's multi-scale concentration curve, build the whole ladder with "
    "--sweep and read one rung per directory: reporting coarser rungs inside a "
    "finer rung's file would duplicate them across directories and let two "
    "copies disagree."
)

_LATENT_OBSERVED_NOTE = (
    "observed is over measured edges (what the dataset can deliver); latent is "
    "over all VP x target pairs (where the infrastructure sits). An edge set is "
    "an artifact of the campaign, so reporting one without the other is what "
    "lets sampling bias be misread as an algorithmic result (paper §7.3)."
)


# ---- distribution blocks ----------------------------------------------------


#: Percentiles every distribution block carries. p5/p25/p50/p75/p95 is
#: `answer_space._describe`'s set; p90 is added because §7.3 specifies degree,
#: edge length and max angular gap as median/IQR/p90 rather than as means, all
#: three being heavily skewed. Sorted, since a percentile block that does not
#: read monotonically invites a misread.
_PERCENTILES = (5, 25, 50, 75, 90, 95)

#: Decimals for a ratio. `_describe`'s 3 is right for kilometres and destroys a
#: ratio that lives just above 1: at 3 dp, as03's twenty targets that missed
#: their nearest VP all round to exactly 1.000, so the block reported a perfect
#: campaign while `n_targets_measuring_their_nearest_vp` reported 438 of 458.
_RATIO_DIGITS = 6


def describe_p90(values, *, digits: int = 3) -> dict:
    """`answer_space._describe`'s block shape, plus p90, at a chosen precision.

    Computed here rather than by delegating to `_describe` and patching, because
    the ratio blocks need more than its hard-coded 3 decimals. Agreement with
    `_describe` on the five percentiles it does emit is pinned by a test, so this
    stays a widening rather than a fork.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "min": round(float(v.min()), digits),
        "max": round(float(v.max()), digits),
        "mean": round(float(v.mean()), digits),
        "percentiles": {
            f"p{p}": round(float(np.percentile(v, p)), digits) for p in _PERCENTILES
        },
    }


def _cdf_column(values, *, digits: int = 3) -> np.ndarray:
    """`_CDF_QUANTILES` of `values`, or all-NaN when there is nothing to rank."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.full(_CDF_QUANTILES.size, np.nan)
    return np.round(np.quantile(v, _CDF_QUANTILES), digits)


# ---- angular geometry -------------------------------------------------------
# §7.3's "arrangement term". Two VPs on opposite sides of a target constrain it
# better than five clustered in one metro, so this is the covariate §8.3 expects
# to dominate the bare degree count.


def bearings_deg(tlat: float, tlon: float, vlats, vlons) -> np.ndarray:
    """Initial great-circle bearing (deg, `[0, 360)`) from a target to each VP."""
    tlat_r, tlon_r = np.radians(float(tlat)), np.radians(float(tlon))
    vlat_r = np.radians(np.asarray(vlats, dtype=float))
    vlon_r = np.radians(np.asarray(vlons, dtype=float))
    dlon = vlon_r - tlon_r
    x = np.sin(dlon) * np.cos(vlat_r)
    y = np.cos(tlat_r) * np.sin(vlat_r) - np.sin(tlat_r) * np.cos(vlat_r) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def angular_features(bearings) -> tuple[float, float]:
    """`(max_gap_deg, circular_variance)` of a set of bearings.

    `max_gap_deg` is the largest wedge containing no VP, wrap-around included:
    VPs at 10/40/75 deg leave a 295 deg gap and barely constrain the target,
    while 20/140/260 leaves 120 deg and brackets it. A single VP leaves the whole
    turn, hence 360. `circular_variance` is `1 - |mean unit vector|` -- 0 when
    every landmark lies in one direction, approaching 1 when well surrounded --
    and is the smoother of the two, which is why §7.3 names it as the form the
    §8.3 regressions use.

    Matches `scripts/analysis/partvp/extract_features.py`'s private
    `_angular_features`. Duplicated rather than imported because that module
    reads `eval_observations.parquet`, which does not exist anywhere under
    `outputs/benchmark/v2/`; `test_bipartite.py` pins the two against each other
    so they cannot drift.
    """
    b = np.sort(np.asarray(bearings, dtype=float))
    n = b.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return 360.0, 0.0
    max_gap = float(max(np.diff(b).max(), 360.0 - (b[-1] - b[0])))
    ang = np.radians(b)
    r = float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean()))
    return max_gap, 1.0 - r


# ---- node-set geometry ------------------------------------------------------


def _pairwise_distances(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, dict]:
    """Upper-triangle pairwise great-circle km, plus a note if it was subsampled."""
    n = lat.size
    note: dict = {}
    if n > _MAX_PAIRWISE_NODES:
        rng = np.random.default_rng(_PAIRWISE_SEED)
        keep = np.sort(rng.choice(n, size=_MAX_PAIRWISE_NODES, replace=False))
        lat, lon = lat[keep], lon[keep]
        note = {
            "pairwise_subsampled_from": int(n),
            "pairwise_subsample_size": _MAX_PAIRWISE_NODES,
            "pairwise_subsample_seed": _PAIRWISE_SEED,
        }
    if lat.size < 2:
        return np.empty(0, dtype=float), note
    d = pairwise_km(lat, lon)
    return d[np.triu_indices_from(d, k=1)], note


def nearest_across_km(
    a_lat: np.ndarray,
    a_lon: np.ndarray,
    b_lat: np.ndarray,
    b_lon: np.ndarray,
    *,
    chunk: int = _CROSS_CHUNK_TARGETS,
) -> tuple[np.ndarray, np.ndarray]:
    """For each node in A, `(km, index)` of the nearest node in B — over **all** of B.

    This is the latent half of §7.3's pair: it ignores the edge set entirely and
    describes where the infrastructure sits. Chunked over A so the resident
    footprint is `|B| x chunk` rather than `|A| x |B|`, which keeps the answer
    exact at target counts where the full matrix is not materializable.

    Empty B gives all-NaN and index -1 rather than raising: a run with no VPs is
    degenerate but the caller reports it rather than crashing on it.
    """
    n = int(a_lat.size)
    if b_lat.size == 0 or n == 0:
        return np.full(n, np.nan), np.full(n, -1, dtype=int)
    best = np.empty(n, dtype=float)
    idx = np.empty(n, dtype=int)
    for start in range(0, n, max(int(chunk), 1)):
        stop = min(start + max(int(chunk), 1), n)
        d = pairwise_km(a_lat[start:stop], a_lon[start:stop], b_lat, b_lon)
        j = d.argmin(axis=1)
        best[start:stop] = d[np.arange(stop - start), j]
        idx[start:stop] = j
    return best, idx


def _latent_pair_distances(
    tg_lat: np.ndarray,
    tg_lon: np.ndarray,
    vp_lat: np.ndarray,
    vp_lon: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Every VP-to-target great-circle distance, measured or not, plus a note.

    The latent counterpart of the observed edge-length distribution. Subsampled
    on the target side past `_MAX_CROSS_PAIRS`, since a distribution needs the
    values and not just their minima; the note records that so a percentile is
    never read as exact when it is not.
    """
    n_t, n_v = int(tg_lat.size), int(vp_lat.size)
    note: dict = {}
    if n_t == 0 or n_v == 0:
        return np.empty(0, dtype=float), note
    if n_t * n_v > _MAX_CROSS_PAIRS:
        keep_n = max(_MAX_CROSS_PAIRS // n_v, 1)
        rng = np.random.default_rng(_PAIRWISE_SEED)
        keep = np.sort(rng.choice(n_t, size=keep_n, replace=False))
        tg_lat, tg_lon = tg_lat[keep], tg_lon[keep]
        note = {
            "latent_subsampled_from_targets": n_t,
            "latent_subsample_targets": int(keep_n),
            "latent_subsample_seed": _PAIRWISE_SEED,
        }
    out = [
        pairwise_km(
            tg_lat[s : s + _CROSS_CHUNK_TARGETS],
            tg_lon[s : s + _CROSS_CHUNK_TARGETS],
            vp_lat,
            vp_lon,
        ).ravel()
        for s in range(0, tg_lat.size, _CROSS_CHUNK_TARGETS)
    ]
    return np.concatenate(out), note


def measurement_efficiency(
    nearest_measured_km: np.ndarray, nearest_latent_km: np.ndarray
) -> tuple[np.ndarray, int]:
    """Per-target `measured / latent`, and the count where it is undefined.

    §7.3 calls this measurement efficiency, which is the name kept here; it is
    **emitted** as `measured_nearest_vp_ratio` (column) and
    `measured_nearest_vp_ratio_per_target` (meta block), which say what is
    divided by what without needing the paper open.

    Three cases, because the degenerate ones carry different meanings:

    * `latent > 0` — the ordinary ratio, `>= 1` by construction since the
      measured minimum is taken over a subset of the latent one. Values within
      `_EFFICIENCY_TOL` of 1.0 are **snapped to exactly 1.0**, because the two
      minima are computed by different reductions and disagree in the last bits;
      snapping here rather than at each call site is what makes `ratio == 1.0`
      usable as the "campaign got the closest VP" test with no tolerance
      repeated downstream.
    * both zero — a VP sits exactly on the target *and* was measured, so the
      campaign did as well as possible: 1.0, not `0/0`.
    * `latent == 0` with a positive or missing measured value — a VP sits on the
      target and carries no edge. The ratio is infinite, which no percentile can
      hold and JSON cannot encode, so it is left NaN and **counted** instead.
      Silently coercing it to 1.0 would report the worst allocation failure the
      metric can express as the best.
    """
    measured = np.asarray(nearest_measured_km, dtype=float)
    latent = np.asarray(nearest_latent_km, dtype=float)
    ratio = np.full(measured.shape, np.nan)
    both_zero = (latent == 0.0) & (measured == 0.0)
    ratio[both_zero] = 1.0
    usable = latent > 0.0
    ratio[usable] = measured[usable] / latent[usable]
    snap = np.isfinite(ratio) & (np.abs(ratio - 1.0) <= _EFFICIENCY_TOL)
    ratio[snap] = 1.0
    undefined = int(((latent == 0.0) & ~both_zero).sum())
    return ratio, undefined


def _node_block(
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    grid: Grid,
    resolution: int,
    asns: pd.Series | None,
    pairwise: np.ndarray,
    pairwise_note: dict,
) -> dict:
    """The §7.3 node-set block for one side: extent, then dispersion.

    `pairwise` is passed in rather than computed here because the CDF artifact
    needs the same vector: at the `_MAX_PAIRWISE_NODES` cap an O(n^2) matrix is
    200 MB, so building it twice is not a rounding error, and a second
    subsampled draw would also have to be pinned to the same seed to agree with
    the first.

    The occupied-cell rungs come from `Grid.occupied_cell_hierarchy`, which
    **re-bins from the coordinates** at every rung rather than coarsening cell
    ids. That is required, not an optimization: H3 is aperture-7 and hexagons
    cannot tile hexagons, so a parent id is exact as an index but is not a
    geometric container, and the two routes genuinely disagree on 1-8% of nodes
    at res 3 (measured, `answer_space` meta's `parent_lineage_disagreements`).
    """
    pw, pw_note = pairwise, pairwise_note
    n = int(lat.size)
    n_cells = int(np.unique(grid.cell_ids(lat, lon, resolution)).size)
    block: dict = {
        "count": n,
        "asn_count": None if asns is None else int(asns.dropna().nunique()),
        # A diameter is a maximum, so one near-antipodal node sets it
        # single-handedly; §7.3 requires p95 printed beside it for that reason.
        "geographic_diameter_km": round(float(pw.max()), 3) if pw.size else None,
        "pairwise_p95_km": round(float(np.percentile(pw, 95)), 3) if pw.size else None,
        "pairwise_km": _describe(pw),
        "dispersion": {
            "effective_count": n_cells,
            "occupancy_ratio": round(n_cells / n, 4) if n else None,
            "note": DISPERSION_NOTE,
        },
    }
    block.update(pw_note)
    return block


# ---- edge input -------------------------------------------------------------


def resolve_source_csv(run: RunPaths, override: Path | None = None) -> Path:
    """Path to the run's canonical `(vp_id, target_id, rtt_ms)` CSV.

    The v3 unified configs deliberately do not carry it: `benchmark: {}` on every
    operator run, whose canonical CSV was reconstructed from the run outputs. The
    run itself records it, though — `eval_source/<basename>_eval_stats.json` has a
    `csv` key, relative to the repo root — so that is read rather than
    reconstructed from `run_id`, which does not track the CSV stem.

    Falls back to globbing `datasets/**/<eval_basename>.csv` for a run whose
    `eval_*` predates that key.
    """
    if override is not None:
        p = Path(override)
        if not p.exists():
            raise MissingArtifactError(f"--source-csv {p} does not exist")
        return p

    try:
        recorded = io.load_eval_stats(run).get("csv")
    except MissingArtifactError:
        recorded = None
    if recorded:
        p = Path(recorded)
        p = p if p.is_absolute() else REPO_ROOT / p
        if p.exists():
            return p

    hits = sorted((REPO_ROOT / "datasets").rglob(f"{run.eval_basename}.csv"))
    if hits:
        return hits[0]
    raise MissingArtifactError(
        f"cannot locate the canonical edge CSV for {run.run_id}: "
        f"{run.eval_basename}.csv is not under datasets/ and "
        f"eval_stats.json records "
        f"{recorded!r}. Pass --source-csv <path>."
    )


def load_edges(csv_path: Path) -> tuple[pd.DataFrame, int]:
    """One row per measured `(vp_id, target_id)` pair, RTT dropped.

    Returns `(edges, n_obs)` — the deduplicated edge set and the observation
    count it came from, since a dataset can measure the same pair repeatedly and
    §7.3's density is over pairs.

    Reads through `benchmark.v2.eval_source.load_canonical_csv`, which owns the
    schema (`_REQUIRED`), lowercases the header, and drops NaN rows and
    `rtt_ms <= 0` exactly as `GenericCSVSource` does at materialize time — so the
    graph described here is the graph the benchmark ran on. Deliberately *not*
    `eval_source.build_pairs`: its `gc_km` / `inflation` / `rtt_rank_norm`
    columns are RTT-derived and belong to §8.

    Raises if one id carries more than one coordinate, which would make every
    distance below ambiguous.
    """
    from scripts.benchmark.v2.eval_source import load_canonical_csv

    df = load_canonical_csv(Path(csv_path))
    n_obs = int(len(df))

    cols = ["vp_id", "vp_lat", "vp_lon", "target_id", "target_lat", "target_lon"]
    if "target_asn" in df.columns:
        cols.append("target_asn")
    edges = df[cols].drop_duplicates(["vp_id", "target_id"]).reset_index(drop=True)

    for id_col, lat_col, lon_col in (
        ("vp_id", "vp_lat", "vp_lon"),
        ("target_id", "target_lat", "target_lon"),
    ):
        n_coords = edges.groupby(id_col)[[lat_col, lon_col]].nunique().max(axis=1)
        bad = n_coords[n_coords > 1]
        if not bad.empty:
            raise ValueError(
                f"{csv_path}: {len(bad)} {id_col}(s) carry more than one coordinate "
                f"(e.g. {bad.index[:3].tolist()}); every distance here would be ambiguous"
            )
    return edges, n_obs


# ---- the graph --------------------------------------------------------------


@dataclass(frozen=True)
class BipartiteGraph:
    """Per-node geometry, the two CDFs, and the §7.3 metric block."""

    vp_nodes: pd.DataFrame
    target_nodes: pd.DataFrame
    edge_segments: pd.DataFrame
    edge_length_cdf: pd.DataFrame
    pairwise_distance_cdf: pd.DataFrame
    meta: dict

    @property
    def n_vps(self) -> int:
        return len(self.vp_nodes)

    @property
    def n_targets(self) -> int:
        return len(self.target_nodes)

    def write(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.vp_nodes.to_csv(out_dir / VP_NODES_CSV, index=False)
        self.target_nodes.to_csv(out_dir / TARGET_NODES_CSV, index=False)
        self.edge_segments.to_csv(out_dir / EDGE_SEGMENTS_CSV, index=False)
        self.edge_length_cdf.to_csv(out_dir / EDGE_LENGTH_CDF_CSV, index=False)
        self.pairwise_distance_cdf.to_csv(out_dir / PAIRWISE_CDF_CSV, index=False)
        (out_dir / META_JSON).write_text(json.dumps(self.meta, indent=2) + "\n")
        return out_dir


def _connected_components(edges: pd.DataFrame, vp_ids, target_ids) -> dict:
    """Components of the bipartite graph, over **all** nodes.

    §7.3 asks for this because traffic can concentrate regionally, so the edge
    set may fall apart into pieces that no single calibration spans. Isolated
    nodes count as their own component, which is the graph-theoretic answer and
    also the useful one: a VP that measured nothing is disconnected from the
    experiment, and hiding it in the largest component's share would overstate
    how joined-up the campaign was. Both numbers are therefore reported.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n_v, n_t = len(vp_ids), len(target_ids)
    if n_v + n_t == 0:
        return {"n_components": 0}
    # int64 explicitly: pandas hands back int8 codes for a set of <= 127
    # categories, and `n_v + t` then overflows silently — 53 + 77 wrapped to
    # -126 on as7018 and coo_matrix rejected the negative column. The join
    # upstream guarantees no -1 codes, so that is asserted rather than assumed.
    v = np.asarray(pd.Categorical(edges["vp_id"], categories=list(vp_ids)).codes, dtype=np.int64)
    t = np.asarray(pd.Categorical(edges["target_id"], categories=list(target_ids)).codes, dtype=np.int64)
    if v.min(initial=0) < 0 or t.min(initial=0) < 0:
        raise ValueError("edge endpoint outside the node sets; the join upstream failed")
    adj = coo_matrix(
        (np.ones(len(edges)), (v, n_v + t)), shape=(n_v + n_t, n_v + n_t)
    )
    n_comp, labels = connected_components(adj, directed=False)
    sizes = np.bincount(labels, minlength=n_comp)
    return {
        "n_components": int(n_comp),
        "n_isolated_nodes": int((sizes == 1).sum()),
        "largest_component_nodes": int(sizes.max()),
        "largest_component_node_share": round(float(sizes.max() / sizes.sum()), 6),
        "component_size_nodes": _describe(sizes.astype(float)),
    }


def build_bipartite(
    space: AnswerSpace,
    vps: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    n_obs: int | None = None,
    source_label: str | None = None,
    source_csv: str | None = None,
) -> BipartiteGraph:
    """Compute §7.3's observed-only geometry over one run's bipartite graph.

    The node sets are pinned to the run's own inputs rather than inferred from
    the edge list: VPs are `vps.csv`'s roster and targets are the answer space's
    `assignments`. Edges naming anything outside those are dropped and counted,
    which is what makes `density` read against the same denominator the benchmark
    used (`|E| / (|VP roster| x |answer-space targets|)`) instead of against
    whatever the CSV happened to contain.
    """
    grid = get_grid(str(space.seeds["grid_scheme"].iloc[0]))
    resolution = int(space.seeds["grid_resolution"].iloc[0])

    vps = vps.drop_duplicates("vp_id").reset_index(drop=True)
    targets = space.assignments.reset_index(drop=True)

    n_edges_raw = len(edges)
    known_vp = edges["vp_id"].isin(set(vps["vp_id"]))
    known_tg = edges["target_id"].isin(set(targets["target_id"]))
    dropped_vp = int((~known_vp).sum())
    dropped_tg = int((known_vp & ~known_tg).sum())
    edges = edges.loc[known_vp & known_tg].reset_index(drop=True)
    if edges.empty:
        raise ValueError(
            "no edge survives the join against vps.csv and the answer space; "
            "the CSV and the run do not describe the same graph"
        )

    # --- edge geometry ---------------------------------------------------
    # Edge length is the only place the two node sets meet, so it is computed
    # once and every observed distance below is derived from it.
    #
    # Kept **unrounded**. Rounding here to 3 dp made `measurement_efficiency`
    # divide a rounded numerator by an unrounded denominator, which put 160 of
    # as01's 399 targets a few 1e-4 *below* 1.0 — a value the ratio cannot take,
    # since the measured minimum is over a subset of the latent one. Rounding is
    # applied at the artifact boundary instead, where it is presentation.
    edges = edges.copy()
    edges["length_km"] = _edge_lengths(edges)

    # --- VP nodes ---------------------------------------------------------
    vp_lat = vps["vp_lat"].to_numpy(dtype=float)
    vp_lon = vps["vp_lon"].to_numpy(dtype=float)
    vp_pw, vp_pw_note = _pairwise_distances(vp_lat, vp_lon)
    vp_block = _node_block(
        vp_lat,
        vp_lon,
        grid=grid,
        resolution=resolution,
        asns=vps["vp_asn"] if "vp_asn" in vps.columns else None,
        pairwise=vp_pw,
        pairwise_note=vp_pw_note,
    )
    by_vp = edges.groupby("vp_id")
    vp_nodes = vps.copy()
    vp_nodes["cell_id"] = grid.cell_ids(vp_lat, vp_lon, resolution)
    # Named for the side it points at: there is no VP-to-VP edge, so a bare
    # "degree" invites reading this as one.
    vp_nodes["degree_to_target"] = (
        vp_nodes["vp_id"].map(by_vp.size()).fillna(0).astype(int)
    )
    vp_nodes["nearest_measured_target_km"] = (
        vp_nodes["vp_id"].map(by_vp["length_km"].min()).round(3)
    )
    vp_block["degree_to_target"] = describe_p90(
        vp_nodes["degree_to_target"].to_numpy(dtype=float)
    )
    vp_block["degree_to_target"]["note"] = (
        "targets this VP measured -- §7.3's measurement effort spent. Observed "
        "by definition: the latent value is the target count for every VP, so "
        "it carries nothing and is not emitted."
    )
    vp_block["n_with_no_edge"] = int((vp_nodes["degree_to_target"] == 0).sum())

    # --- target nodes -----------------------------------------------------
    tg_lat = targets["target_lat"].to_numpy(dtype=float)
    tg_lon = targets["target_lon"].to_numpy(dtype=float)
    tg_asns = (
        edges.drop_duplicates("target_id").set_index("target_id")["target_asn"]
        if "target_asn" in edges.columns
        else None
    )
    tg_pw, tg_pw_note = _pairwise_distances(tg_lat, tg_lon)
    tg_block = _node_block(
        tg_lat,
        tg_lon,
        grid=grid,
        resolution=resolution,
        asns=tg_asns,
        pairwise=tg_pw,
        pairwise_note=tg_pw_note,
    )
    by_tg = edges.groupby("target_id")
    nearest_idx = by_tg["length_km"].idxmin()

    target_nodes = targets.loc[
        :, ["target_id", "target_lat", "target_lon", "cell_id", "seed_id"]
    ].copy()
    target_nodes["degree_to_vp"] = (
        target_nodes["target_id"].map(by_tg.size()).fillna(0).astype(int)
    )
    measured_km = target_nodes["target_id"].map(by_tg["length_km"].min())
    target_nodes["nearest_measured_vp_km"] = measured_km.round(3)
    target_nodes["nearest_measured_vp_id"] = target_nodes["target_id"].map(
        edges.loc[nearest_idx].set_index("target_id")["vp_id"]
    )

    # --- the latent half --------------------------------------------------
    # Over all VP x target pairs, ignoring the edge set: where the
    # infrastructure sits, as against what the campaign measured.
    tg_latent_km, tg_latent_idx = nearest_across_km(tg_lat, tg_lon, vp_lat, vp_lon)
    target_nodes["nearest_vp_km"] = np.round(tg_latent_km, 3)
    target_nodes["nearest_vp_id"] = np.where(
        tg_latent_idx >= 0, vps["vp_id"].to_numpy()[tg_latent_idx], None
    )
    eff, _n_eff_undefined = measurement_efficiency(
        measured_km.to_numpy(dtype=float), tg_latent_km
    )
    target_nodes["measured_nearest_vp_ratio"] = np.round(eff, 6)
    # Defined on **distance**, not on id equality. Co-located VPs are common
    # (as01's VP nearest-neighbour p50 is 0.0 km), so `argmin` and any other
    # nearest-VP implementation break ties differently: comparing ids against
    # `eval_source`'s BallTree answer agreed on only 259 of 399 targets while the
    # distances agreed to 4e-4 km. What the operator is asking is whether the
    # campaign measured a VP as close as the closest one, and that is the ratio.
    # Exact, because `measurement_efficiency` has already snapped the residue.
    # Defined on **distance** and not on id equality: co-located VPs are the
    # normal case (as01's VP nearest-neighbour p50 is 0.0 km), so `argmin` and
    # any other nearest-VP implementation break ties differently — comparing ids
    # against `eval_source`'s BallTree answer agreed on only 259 of as01's 399
    # targets while the distances agreed to 4e-4 km. What the operator is asking
    # is whether the campaign measured a VP *as close as* the closest one.
    target_nodes["nearest_vp_is_measured"] = eff <= 1.0

    latent_pairs, latent_note = _latent_pair_distances(tg_lat, tg_lon, vp_lat, vp_lon)

    gaps, circ = _angular_per_target(edges, target_nodes["target_id"])
    target_nodes["max_angular_gap_deg"] = np.round(gaps, 3)
    target_nodes["circular_variance"] = np.round(circ, 6)

    tg_block["degree_to_vp"] = describe_p90(
        target_nodes["degree_to_vp"].to_numpy(dtype=float)
    )
    tg_block["degree_to_vp"]["note"] = (
        "VPs that measured this target -- §7.3's constraints available. Observed "
        "by definition: the latent value is the VP count for every target, so it "
        "carries nothing and is not emitted."
    )
    tg_block["n_with_no_edge"] = int((target_nodes["degree_to_vp"] == 0).sum())
    tg_block["n_seeds"] = int(space.n_seeds)

    # --- meta -------------------------------------------------------------
    denom = len(vps) * len(targets)
    meta = {
        "source": source_label,
        "grid": grid.describe(resolution),
        "scope": {
            "rtt": "not used beyond the canonical CSV's own rtt_ms > 0 row filter",
            "distances": "observed only (measured edges); no latent all-pairs half",
        },
        "inputs": {
            "source_csv": source_csv,
            "n_observations": n_obs,
            "n_edges_before_join": int(n_edges_raw),
            "n_edges_dropped_vp_not_in_roster": dropped_vp,
            "n_edges_dropped_target_not_in_answer_space": dropped_tg,
        },
        "nodes": {"vps": vp_block, "targets": tg_block},
        "edges": {
            "n_edges": int(len(edges)),
            # Measurement completeness: what share of the possible (VP, target)
            # pairs the campaign actually measured. Not the same thing as
            # measured_nearest_vp_ratio_per_target below -- this is a count ratio
            # over all pairs, that is a distance ratio at the minimum only, and
            # a campaign can score well on either while failing the other.
            "edge_density": round(len(edges) / denom, 6) if denom else None,
            "connected_components": _connected_components(
                edges, vps["vp_id"], targets["target_id"]
            ),
            # §7.3 reports every VP-to-target distance as a **pair**, so the
            # two halves are nested rather than siblings: neither can be read
            # without the other in view.
            "length_km": {
                "observed": describe_p90(edges["length_km"].to_numpy(dtype=float)),
                "latent": describe_p90(latent_pairs),
                "n_latent_pairs": int(denom),
                "note": _LATENT_OBSERVED_NOTE,
                **latent_note,
            },
            # §7.3's dominant scalar predictor of region size: the smallest disk
            # does most of the constraining.
            "nearest_vp_km": {
                "observed": describe_p90(
                    target_nodes["nearest_measured_vp_km"].to_numpy(dtype=float)
                ),
                "latent": describe_p90(tg_latent_km),
                "note": _LATENT_OBSERVED_NOTE,
            },
            "measured_nearest_vp_ratio_per_target": {
                **describe_p90(eff, digits=_RATIO_DIGITS),
                "note": _MEASUREMENT_EFFICIENCY_NOTE,
            },
        },
        "angular": {
            "note": "over measured neighbours only; the arrangement term of §7.3",
            "max_angular_gap_deg": describe_p90(gaps),
            "circular_variance": describe_p90(circ),
        },
    }

    segments = edge_segments(edges)
    meta["edges"]["n_distinct_geometry_edge"] = int(len(segments))
    meta["edges"]["n_distinct_geometry_edge_note"] = (
        "distinct (VP coord, target coord) geometries, NOT a second edge count: "
        "all n_edges edges are distinct edges, but many share a line because "
        "several target IPs sit at one facility. This is what the flow map "
        "draws -- see edge_segments()."
    )

    edge_cdf = pd.DataFrame(
        {
            "quantile": _CDF_QUANTILES,
            "observed_edge_km": _cdf_column(edges["length_km"]),
            "latent_pair_km": _cdf_column(latent_pairs),
            "measured_nearest_vp_ratio": _cdf_column(eff, digits=_RATIO_DIGITS),
        }
    )
    pairwise_cdf = pd.DataFrame(
        {
            "quantile": _CDF_QUANTILES,
            "vp_pairwise_km": _cdf_column(vp_pw),
            "target_pairwise_km": _cdf_column(tg_pw),
        }
    )
    return BipartiteGraph(
        vp_nodes=vp_nodes,
        target_nodes=target_nodes,
        edge_segments=segments,
        edge_length_cdf=edge_cdf,
        pairwise_distance_cdf=pairwise_cdf,
        meta=meta,
    )


def _edge_lengths(edges: pd.DataFrame) -> np.ndarray:
    """Great-circle km per edge, paired rather than as a matrix."""
    return elementwise_km(
        edges["vp_lat"].to_numpy(dtype=float),
        edges["vp_lon"].to_numpy(dtype=float),
        edges["target_lat"].to_numpy(dtype=float),
        edges["target_lon"].to_numpy(dtype=float),
    )


def edge_segments(edges: pd.DataFrame) -> pd.DataFrame:
    """Geometrically distinct flow lines, with the edge count each stands for.

    The flow map's data, and a fact about these datasets worth writing down. as01
    has 53,262 edges but only 20 distinct target coordinates, so almost every
    edge is an exact overplot of another: drawing one line per row would blend
    alpha into a density reading that reports duplicate IPs at one facility as
    heavy traffic. Collapsing to distinct `(VP coord, target coord)` pairs and
    carrying `n_edges` as multiplicity keeps the ink honest, and `n_edges` vs
    `len(segments)` is itself the diagnostic.
    """
    cols = ["vp_lat", "vp_lon", "target_lat", "target_lon"]
    rounded = edges[cols].round(_SEGMENT_COORD_DP)
    seg = (
        edges.assign(**{c: rounded[c] for c in cols})
        .groupby(cols, as_index=False)
        .agg(n_edges=("length_km", "size"), length_km=("length_km", "first"))
        .sort_values(cols, ignore_index=True)
    )
    return seg


def _angular_per_target(
    edges: pd.DataFrame, target_order: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """`(max_angular_gap_deg, circular_variance)` per target, in `target_order`.

    A target with no measured edge gets NaN on both, not 360/0: 360 means "one
    VP, so the whole turn is empty", and conflating it with "no VP at all" would
    put an unmeasured target in the same bucket as the worst-arranged measured
    one.
    """
    gaps: dict[str, float] = {}
    circ: dict[str, float] = {}
    for tid, sub in edges.groupby("target_id"):
        b = bearings_deg(
            sub["target_lat"].iloc[0],
            sub["target_lon"].iloc[0],
            sub["vp_lat"].to_numpy(),
            sub["vp_lon"].to_numpy(),
        )
        gaps[tid], circ[tid] = angular_features(b)
    return (
        target_order.map(gaps).to_numpy(dtype=float),
        target_order.map(circ).to_numpy(dtype=float),
    )


def build_for_run(
    run: RunPaths,
    *,
    analysis_root: Path | None = None,
    grid: Grid | str = DEFAULT_GRID,
    resolution: int | None = None,
    answer_space: Path | None = None,
    source_csv: Path | None = None,
) -> tuple[BipartiteGraph, Path]:
    """`build_bipartite` for one run. Returns the graph and its output dir."""
    g = get_grid(grid) if isinstance(grid, str) else grid
    res = g.DEFAULT_RESOLUTION if resolution is None else g.validate_resolution(resolution)
    space_dir = answer_space or run.answer_space_dir(
        root=analysis_root, grid=g.name, resolution=res
    )
    space = load_answer_space(space_dir)

    csv_path = resolve_source_csv(run, source_csv)
    edges, n_obs = load_edges(csv_path)
    graph = build_bipartite(
        space,
        io.load_vps(run),
        edges,
        n_obs=n_obs,
        source_label=f"{run.run_id}/{run.source}/{run.setup}",
        source_csv=str(csv_path),
    )
    out_dir = run.bipartite_dir(
        root=analysis_root,
        grid=str(space.seeds["grid_scheme"].iloc[0]),
        resolution=int(space.seeds["grid_resolution"].iloc[0]),
    )
    return graph, out_dir


def load_bipartite(path: Path) -> BipartiteGraph:
    """Read back what `write` produced, restoring the grid-specific id dtype."""
    path = Path(path)
    vp_path, tg_path = path / VP_NODES_CSV, path / TARGET_NODES_CSV
    if not vp_path.exists() or not tg_path.exists():
        raise MissingArtifactError(
            f"{path} has no {VP_NODES_CSV}/{TARGET_NODES_CSV}; run `cli build-bipartite-graph`"
        )
    meta = json.loads((path / META_JSON).read_text())
    grid = get_grid(str(meta["grid"]["scheme"]))
    vp_nodes = pd.read_csv(vp_path)
    tg_nodes = pd.read_csv(tg_path)
    for frame in (vp_nodes, tg_nodes):
        frame["cell_id"] = grid.coerce_cell_ids(frame["cell_id"])
    return BipartiteGraph(
        vp_nodes=vp_nodes,
        target_nodes=tg_nodes,
        edge_segments=pd.read_csv(path / EDGE_SEGMENTS_CSV),
        edge_length_cdf=pd.read_csv(path / EDGE_LENGTH_CDF_CSV),
        pairwise_distance_cdf=pd.read_csv(path / PAIRWISE_CDF_CSV),
        meta=meta,
    )


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("build-bipartite-graph")
    def build_bipartite_cmd(
        run_id: str = typer.Option(
            None, help="Run to describe. Omit with --all-runs to do every run."
        ),
        all_runs: bool = typer.Option(
            False, "--all-runs", help="Describe every run under --outputs-root."
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
        """VP nodes, target nodes and measured edges — the §7.3 dataset geometry.

        Writes vp_nodes.csv, target_nodes.csv, the two CDFs and meta.json into
        bipartite-graph/<grid>-<resolution>/.
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
            graph, out_dir = build_for_run(
                run,
                analysis_root=analysis_root,
                grid=g,
                resolution=want_res,
                answer_space=answer_space,
                source_csv=source_csv,
            )
            graph.write(out_dir)
            e = graph.meta["edges"]
            vp_disp = graph.meta["nodes"]["vps"]["dispersion"]
            typer.echo(
                f"{run.run_id}: {graph.n_vps} VPs in "
                f"{vp_disp['effective_count']} places "
                f"(occupancy {vp_disp['occupancy_ratio']:.2f}) · "
                f"{graph.n_targets} targets · {e['n_edges']:,} edges "
                f"(density {e['edge_density']:.3f}, "
                f"{e['connected_components']['n_components']} component(s)) -> {out_dir}"
            )
