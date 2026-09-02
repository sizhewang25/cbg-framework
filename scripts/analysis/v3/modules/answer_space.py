"""Build the target answer space: grid quantizer -> seeds -> Voronoi (§7.3/§7.4).

Three steps, exactly as the paper states them:

1. Quantize every ground-truth target onto a grid — H3 `res=4` (~45 km) by
   default, HEALPix `nside=128` (~51 km) for the paper's original setting; see
   `grid.py`. Cell membership is an equivalence relation whose only job is to
   merge points close enough to count as one place. Which tessellation does the
   merging is a *parameter*: steps 2 and 3 are pure spherical geometry.
2. Each occupied cell contributes one **seed**, the spherical centroid of the
   targets inside it — so the answer space is K *real locations*, not K grid
   squares.
3. A coordinate is labelled by its nearest seed. That one rule scores ground
   truth and predictions alike, and error distance is measured to the seed.

The grid's cost is recorded rather than argued away: grid lines fall where the
grid falls, so a facility group straddling one yields two seeds and two classes,
and the Voronoi step cannot undo a split the grid already made. `meta.json`
carries the straddle diagnostic for that.

Command: `build-answer-space`. Writes to
`outputs/analysis/v3/<run_id>/target-answer-space/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules import io
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
    MissingArtifactError,
    RunPaths,
    discover_runs,
    resolve_run,
)
from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM

SWEEP_CSV_TEMPLATE = "grid_sweep.{grid}.csv"
SEEDS_CSV = "seeds.csv"
ASSIGNMENTS_CSV = "assignments.csv"
SEED_MESH_CSV = "seed_mesh_km.csv"
META_JSON = "meta.json"


# ---- geometry helpers -------------------------------------------------------


def _unit_vectors(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    return np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


def spherical_centroid(lat_deg, lon_deg) -> tuple[float, float]:
    """Centroid of points on the sphere: normalized mean of unit vectors.

    Averaging lat/lon directly is wrong near the dateline and at high latitude;
    this is the projection-free form, matching §7.4's "centroid of the targets
    inside the cell".

    Lives here rather than in a grid module because it is not grid math — it is
    the same unit-vector mean as `_unit_vectors` above, and both grids need it
    identically.
    """
    v = _unit_vectors(np.atleast_1d(lat_deg), np.atleast_1d(lon_deg)).mean(axis=0)
    norm = float(np.sqrt(v @ v))
    if norm == 0.0:
        # Antipodal cancellation — impossible within one cell, but a mean of
        # zero has no direction so there is no centroid to return.
        raise ValueError("degenerate point set: unit vectors cancel to zero")
    vx, vy, vz = v
    return (
        float(np.degrees(np.arcsin(vz / norm))),
        float(np.degrees(np.arctan2(vy, vx))),
    )


def pairwise_km(
    lat_a, lon_a, lat_b=None, lon_b=None
) -> np.ndarray:
    """Great-circle distance matrix in km, via unit-vector dot products.

    Chord-length form rather than haversine so a single matrix multiply covers
    all pairs; equivalent to within floating point.
    """
    a = _unit_vectors(lat_a, lon_a)
    b = a if lat_b is None else _unit_vectors(lat_b, lon_b)
    cos = np.clip(a @ b.T, -1.0, 1.0)
    return EARTH_RADIUS_KM * np.arccos(cos)


def elementwise_km(lat_a, lon_a, lat_b, lon_b) -> np.ndarray:
    """Great-circle distance for *paired* coordinates: `out[i] = d(a[i], b[i])`.

    The row-wise companion to `pairwise_km`, which builds the full matrix. Used
    where only the diagonal is wanted — e.g. prediction-to-its-own-target error
    — so the cost stays O(n) instead of O(n^2).
    """
    a = _unit_vectors(lat_a, lon_a)
    b = _unit_vectors(lat_b, lon_b)
    cos = np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0)
    return EARTH_RADIUS_KM * np.arccos(cos)


def _describe(values: np.ndarray) -> dict:
    """Compact distribution summary; `None` when there is nothing to describe."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    pct = {f"p{p}": round(float(np.percentile(v, p)), 3) for p in (5, 25, 50, 75, 95)}
    return {
        "n": int(v.size),
        "min": round(float(v.min()), 3),
        "max": round(float(v.max()), 3),
        "mean": round(float(v.mean()), 3),
        "percentiles": pct,
    }


def _delaunay_degree(lat: np.ndarray, lon: np.ndarray) -> np.ndarray | None:
    """Spherical Delaunay degree per seed, as the convex hull of unit vectors.

    Computing the hull of the unit vectors *is* the spherical Delaunay
    triangulation, so no projection enters (§7.4). Returns None when there are
    too few seeds (or they are coplanar) for a hull to exist.
    """
    if lat.size < 4:
        return None
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(_unit_vectors(lat, lon))
    except Exception:
        return None
    neighbours: list[set[int]] = [set() for _ in range(lat.size)]
    for simplex in hull.simplices:
        for i in simplex:
            for j in simplex:
                if i != j:
                    neighbours[i].add(int(j))
    return np.array([len(s) for s in neighbours], dtype=int)


# ---- the answer space -------------------------------------------------------


@dataclass(frozen=True)
class AnswerSpace:
    """Seeds, target assignments, and the geometry describing both."""

    seeds: pd.DataFrame
    assignments: pd.DataFrame
    seed_mesh_km: pd.DataFrame
    meta: dict

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)

    @property
    def seed_ids(self) -> list[int]:
        return self.seeds["seed_id"].tolist()

    def write(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.seeds.to_csv(out_dir / SEEDS_CSV, index=False)
        self.assignments.to_csv(out_dir / ASSIGNMENTS_CSV, index=False)
        self.seed_mesh_km.to_csv(out_dir / SEED_MESH_CSV, index=True)
        (out_dir / META_JSON).write_text(json.dumps(self.meta, indent=2) + "\n")
        return out_dir


def build_answer_space(
    targets: pd.DataFrame,
    *,
    grid: Grid | str = DEFAULT_GRID,
    resolution: int | None = None,
    source_label: str | None = None,
) -> AnswerSpace:
    """Quantize `targets` and derive the seed set plus its geometry.

    `targets` needs `target_id`, `target_lat`, `target_lon`. Duplicate
    `target_id` is an error; duplicate *coordinates* are fine and expected
    (distinct IPs at one facility), and they collapse into one seed.

    `grid` takes either a `Grid` or a name; `resolution` defaults to whatever
    that grid considers its own default, so callers never have to pair a grid
    with a number that belongs to a different one.
    """
    grid = get_grid(grid) if isinstance(grid, str) else grid
    resolution = grid.validate_resolution(
        grid.DEFAULT_RESOLUTION if resolution is None else resolution
    )
    required = {"target_id", "target_lat", "target_lon"}
    missing = required - set(targets.columns)
    if missing:
        raise ValueError(f"targets is missing columns: {sorted(missing)}")
    if targets["target_id"].duplicated().any():
        dups = targets.loc[targets["target_id"].duplicated(), "target_id"].unique()[:5]
        raise ValueError(f"duplicate target_id in targets (e.g. {dups.tolist()})")
    if targets.empty:
        raise ValueError("targets is empty; there is no answer space to build")

    t = targets.loc[:, ["target_id", "target_lat", "target_lon"]].copy()
    t["cell_id"] = grid.cell_ids(t["target_lat"], t["target_lon"], resolution)

    # Seed per occupied cell, ordered by cell id so seed_id is deterministic
    # and independent of input row order. Sorting is lexicographic for H3's
    # string ids and numeric for HEALPix's ints; either way it is a total order
    # on the ids, which is all determinism needs.
    seed_rows: list[dict] = []
    for rank, (cell, grp) in enumerate(t.groupby("cell_id", sort=True)):
        clat, clon = spherical_centroid(grp["target_lat"], grp["target_lon"])
        if len(grp) > 1:
            d = pairwise_km(grp["target_lat"].to_numpy(), grp["target_lon"].to_numpy())
            spread = float(d.max())
        else:
            spread = 0.0
        seed_rows.append(
            {
                "seed_id": rank,
                "grid_scheme": grid.name,
                "grid_resolution": resolution,
                "cell_id": cell,
                "centroid_lat": clat,
                "centroid_lon": clon,
                "n_targets": int(len(grp)),
                "intra_seed_spread_km": round(spread, 3),
            }
        )
    seeds = pd.DataFrame(seed_rows)

    # Assign every target to its cell's seed. Note this is cell membership, not
    # nearest-seed — for ground truth the two can differ, and the paper's rule
    # is that the *cell* defines a target's class while nearest-seed labels
    # arbitrary coordinates (predictions). Both are recorded so the gap is
    # visible rather than assumed to be empty.
    cell_to_seed = dict(zip(seeds["cell_id"], seeds["seed_id"]))
    t["seed_id"] = t["cell_id"].map(cell_to_seed).astype(int)

    seed_lat = seeds["centroid_lat"].to_numpy()
    seed_lon = seeds["centroid_lon"].to_numpy()
    d_to_all = pairwise_km(
        t["target_lat"].to_numpy(), t["target_lon"].to_numpy(), seed_lat, seed_lon
    )
    t["dist_to_seed_km"] = np.round(
        d_to_all[np.arange(len(t)), t["seed_id"].to_numpy()], 3
    )
    nearest_seed = d_to_all.argmin(axis=1)
    t["nearest_seed_id"] = nearest_seed
    t["cell_seed_is_nearest"] = nearest_seed == t["seed_id"].to_numpy()

    # Seed-to-seed geometry: the margin at which coordinate error flips a label.
    mesh = pairwise_km(seed_lat, seed_lon)
    K = len(seeds)
    if K > 1:
        off = mesh + np.diag(np.full(K, np.inf))
        nearest_km = off.min(axis=1)
        nearest_id = off.argmin(axis=1)
    else:
        nearest_km = np.array([np.inf])
        nearest_id = np.array([-1])
    seeds["nearest_seed_id"] = nearest_id
    seeds["nearest_seed_km"] = np.round(nearest_km, 3)
    # The boundary between two adjacent classes bisects the geodesic joining
    # their seeds, so half the per-seed minimum is the margin at the seed.
    seeds["margin_km"] = np.round(nearest_km / 2.0, 3)
    seeds["radius_km"] = [
        round(
            float(
                t.loc[t["seed_id"] == sid, "dist_to_seed_km"].max()
                if (t["seed_id"] == sid).any()
                else 0.0
            ),
            3,
        )
        for sid in seeds["seed_id"]
    ]
    deg = _delaunay_degree(seed_lat, seed_lon)
    if deg is not None:
        seeds["delaunay_degree"] = deg

    mesh_df = pd.DataFrame(
        np.round(mesh, 3), index=seeds["seed_id"], columns=seeds["seed_id"]
    )
    mesh_df.index.name = "seed_id"

    pitch = grid.nominal_cell_km(resolution)
    n_close_pairs = (
        int(((mesh < pitch) & (mesh > 0)).sum() // 2) if K > 1 else 0
    )
    meta = {
        "source": source_label,
        "grid": grid.describe(resolution),
        "grid_diagnostics": grid.occupancy_diagnostics(
            t["cell_id"].to_numpy(),
            t["target_lat"].to_numpy(),
            t["target_lon"].to_numpy(),
            resolution,
        ),
        "n_targets": int(len(t)),
        "n_unique_target_coords": int(
            t.loc[:, ["target_lat", "target_lon"]].drop_duplicates().shape[0]
        ),
        "n_seeds": K,
        "n_singleton_seeds": int((seeds["n_targets"] == 1).sum()),
        # Rungs at or *coarser* than the grid actually built. Walking a fixed
        # hierarchy regardless of the resolution in use, as an earlier version
        # did, reported occupancy for grids that were never built — at the
        # coarsest rung, every count came from finer grids than the answer space.
        "occupied_cells_by_resolution": {
            str(n): c
            for n, c in grid.occupied_cell_hierarchy(
                t["target_lat"], t["target_lon"], grid.coarsening_ladder(resolution)
            ).items()
        },
        "targets_per_seed": _describe(seeds["n_targets"].to_numpy()),
        "intra_seed_spread_km": _describe(seeds["intra_seed_spread_km"].to_numpy()),
        "dist_to_seed_km": _describe(t["dist_to_seed_km"].to_numpy()),
        "nearest_seed_km": _describe(nearest_km),
        "margin_km": _describe(nearest_km / 2.0),
        "seed_pairwise_km": _describe(mesh[np.triu_indices(K, k=1)]) if K > 1 else {"n": 0},
        "straddle_diagnostic": {
            "note": (
                "Seed pairs closer than one cell pitch are candidate grid-line "
                "splits: one facility group quantized into two classes. Voronoi "
                "cannot undo a split the grid already made (§7.3, §10)."
            ),
            "n_seed_pairs_within_one_cell_pitch": n_close_pairs,
            "n_targets_whose_cell_seed_is_not_nearest": int(
                (~t["cell_seed_is_nearest"]).sum()
            ),
        },
    }
    if deg is not None:
        meta["delaunay_degree"] = _describe(deg.astype(float))

    assignments = t.loc[
        :,
        [
            "target_id",
            "target_lat",
            "target_lon",
            "cell_id",
            "seed_id",
            "dist_to_seed_km",
            "nearest_seed_id",
            "cell_seed_is_nearest",
        ],
    ].reset_index(drop=True)

    return AnswerSpace(
        seeds=seeds, assignments=assignments, seed_mesh_km=mesh_df, meta=meta
    )


def build_for_run(
    run: RunPaths, *, grid: Grid | str = DEFAULT_GRID, resolution: int | None = None
) -> AnswerSpace:
    """Answer space over a run's full ground-truth target set.

    Reads `targets.csv`, which is fold-independent: K-fold splits targets, so
    the union over folds is exactly this file. Building the space once per run
    (rather than per fold) is what makes fold-pooled scoring coherent.
    """
    targets = io.load_targets(run)
    return build_answer_space(
        targets,
        grid=grid,
        resolution=resolution,
        source_label=f"{run.run_id}/{run.source}/{run.setup}",
    )


def load_answer_space(path: Path) -> AnswerSpace:
    """Read back a written answer space (the explicit input to scoring)."""
    path = Path(path)
    seeds_p, assign_p = path / SEEDS_CSV, path / ASSIGNMENTS_CSV
    if not seeds_p.exists() or not assign_p.exists():
        raise MissingArtifactError(
            f"{path} is not an answer space ({SEEDS_CSV} / {ASSIGNMENTS_CSV} missing); "
            f"run `build-answer-space` first"
        )
    mesh_p, meta_p = path / SEED_MESH_CSV, path / META_JSON
    mesh = (
        pd.read_csv(mesh_p, index_col="seed_id")
        if mesh_p.exists()
        else pd.DataFrame()
    )
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    seeds = pd.read_csv(seeds_p)
    assignments = pd.read_csv(assign_p)
    # `cell_id` is grid-specific in dtype (HEALPix int64, H3 hex string) and CSV
    # is untyped, so pandas' inference has to be corrected by the scheme that
    # wrote it. Without this a HEALPix id could come back as a string and stop
    # matching `assignments`.
    if "grid_scheme" in seeds.columns and not seeds.empty:
        grid = get_grid(str(seeds["grid_scheme"].iloc[0]))
        seeds["cell_id"] = grid.coerce_cell_ids(seeds["cell_id"])
        if "cell_id" in assignments.columns:
            assignments["cell_id"] = grid.coerce_cell_ids(assignments["cell_id"])
    return AnswerSpace(
        seeds=seeds,
        assignments=assignments,
        seed_mesh_km=mesh,
        meta=meta,
    )


def sweep_row(space: "AnswerSpace") -> dict:
    """One line of `grid_sweep.<grid>.csv`: how the partition changes with scale.

    The point of the sweep is the trade-off, so each row pairs what coarsening
    buys (fewer classes, fewer straddle candidates) against what it costs
    (targets pulled further from their own seed, which floors error distance).
    """
    m = space.meta
    spread = m["intra_seed_spread_km"]
    return {
        "grid": m["grid"]["scheme"],
        "resolution": m["grid"]["resolution"],
        "cell_km": round(m["grid"]["nominal_cell_km"], 1),
        "cell_area_km2": m["grid"]["cell_area_km2"],
        "n_targets": m["n_targets"],
        "n_classes": m["n_seeds"],
        "n_singleton_classes": m["n_singleton_seeds"],
        "straddle_pairs_within_one_pitch": m["straddle_diagnostic"][
            "n_seed_pairs_within_one_cell_pitch"
        ],
        "n_cell_seed_not_nearest": m["straddle_diagnostic"][
            "n_targets_whose_cell_seed_is_not_nearest"
        ],
        "intra_seed_spread_km_mean": spread["mean"],
        "intra_seed_spread_km_max": spread["max"],
        "nearest_seed_km_p50": m["nearest_seed_km"]["percentiles"]["p50"],
    }


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("build-answer-space")
    def build_answer_space_cmd(
        run_id: str = typer.Option(
            None, help="Run to build for. Omit with --all-runs to do every run."
        ),
        all_runs: bool = typer.Option(
            False, "--all-runs", help="Build for every run under --outputs-root."
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
        """Quantize a run's targets onto a grid and emit the seed answer space.

        Writes seeds.csv, assignments.csv, seed_mesh_km.csv and meta.json to
        <analysis_root>/<run_id>/target-answer-space/<grid>-<resolution>/, one
        directory per requested resolution. Building more than one also writes
        grid_sweep.<grid>.csv one level up, comparing the partitions side by
        side.
        """
        if all_runs == (run_id is not None):
            raise typer.BadParameter("pass exactly one of --run-id or --all-runs")
        g, resolutions = resolve_cli_grid(grid, resolution, sweep=sweep)

        runs = discover_runs(outputs_root) if all_runs else [resolve_run(run_id, outputs_root)]
        for run in runs:
            rows = []
            for res in resolutions:
                space = build_for_run(run, grid=g, resolution=res)
                out = space.write(
                    run.answer_space_dir(root=analysis_root, grid=g.name, resolution=res)
                )
                rows.append(sweep_row(space))
                m = space.meta
                typer.echo(
                    f"{run.run_id}: {m['n_targets']} targets -> K={m['n_seeds']} seeds "
                    f"({m['n_singleton_seeds']} singleton) at "
                    f"{g.name} {g.resolution_arg}={res} "
                    f"[{m['grid']['nominal_cell_km']:.1f} km] -> {out}"
                )
            if len(rows) > 1:
                sweep_dir = run.analysis_dir("target-answer-space", root=analysis_root)
                # Named per grid: with the flat <grid>-<res>/ layout both grids'
                # sweeps land in this one directory, so a fixed filename would let
                # an h3 sweep silently clobber a healpix one.
                sweep_csv = sweep_dir / SWEEP_CSV_TEMPLATE.format(grid=g.name)
                pd.DataFrame(rows).to_csv(sweep_csv, index=False)
                typer.echo(f"{run.run_id}: sweep -> {sweep_csv}")
