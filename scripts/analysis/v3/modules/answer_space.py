"""Build the target answer space: grid quantizer -> seeds -> Voronoi (§7.3/§7.4).

Three steps, exactly as the paper states them:

1. Quantize every ground-truth target onto a grid — H3 `res=4` (~45 km) by
   default, HEALPix `nside=128` (~51 km) for the paper's original setting; see
   `grid.py`. Cell membership is an equivalence relation whose only job is to
   merge points close enough to count as one place. Which tessellation does the
   merging is a *parameter*: the grid supplies cell membership and cell centre,
   and step 3 is pure spherical geometry on top of them.
2. Each occupied cell contributes one **seed**, at the cell's own centre. The
   grid decides which targets are one place, so it decides where that place is
   too: a seed never depends on which targets happened to land in the cell, and
   the same cell yields the same seed in every run.
3. A coordinate is labelled by its nearest seed. Ground truth is labelled by
   cell membership, which is the same rule wherever a cell contains its own
   centre's Voronoi region.

The grid's cost is declared rather than argued away: choosing a grid at a
resolution *is* the granularity claim, and `meta.json` carries `cell_offset_km`
— how far each target sits from the centre standing in for it — as the measured
size of that claim.

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


def geodesic_points(
    lat_a: float, lon_a: float, lat_b: float, lon_b: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """`n` points evenly spaced along the great circle from a to b, inclusive.

    Spherical linear interpolation of the two unit vectors, so the path is the
    actual geodesic rather than a straight line in lat/lon — which at these
    longitudes would bow hundreds of kilometres off it.
    """
    ra, rb = np.radians([lat_a, lat_b]), np.radians([lon_a, lon_b])
    a = np.array([np.cos(ra[0]) * np.cos(rb[0]), np.cos(ra[0]) * np.sin(rb[0]), np.sin(ra[0])])
    b = np.array([np.cos(ra[1]) * np.cos(rb[1]), np.cos(ra[1]) * np.sin(rb[1]), np.sin(ra[1])])
    omega = float(np.arccos(np.clip(a @ b, -1.0, 1.0)))
    t = np.linspace(0.0, 1.0, n)[:, None]
    if omega < 1e-9:
        p = np.repeat(a[None, :], n, axis=0)
    else:
        p = (np.sin((1 - t) * omega) * a + np.sin(t * omega) * b) / np.sin(omega)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return np.degrees(np.arcsin(np.clip(p[:, 2], -1, 1))), np.degrees(
        np.arctan2(p[:, 1], p[:, 0])
    )


#: Above this many seeds the O(K^2) walk stops being cheap: measured 0.08s at
#: K=27 and 15.6s at K=120, superlinear because crowding shrinks the sampling
#: step as well as adding pairs. `h3-4` — the grid we run — sits at K=18-27, so
#: the guard only ever binds on a diagnostic sweep at a finer resolution, where
#: the degree column is reported as NaN with the reason in `meta.json`.
MAX_SEEDS_FOR_ADJACENCY = 128


def seed_crossing_matrix(seeds: pd.DataFrame, *, max_samples: int = 4096) -> np.ndarray:
    """`C[i, j]` = Voronoi class boundaries crossed on the geodesic from seed i to j.

    **This is the relation the confusion question actually wants**: how many
    class boundaries lie between the right answer and the given one. `C == 1` is
    "the estimate slipped across exactly one line", which implies the two cells
    share a boundary; `C == 3` is three cells away. Distance rank cannot say
    this — a seed can be the second-nearest without sharing any boundary, and a
    genuine neighbour can be the twentieth-nearest.

    **`C == 1` is not the same as "shares a boundary", and the gap is large.**
    The implication runs one way only: a shared edge need not lie on the segment
    joining two seeds, so a genuinely adjacent pair can walk through a third cell.
    Measured at `h3-4`, mean degree under `C == 1` is 2.7-2.8 against 4.8-5.3 for
    true (projected 2-D Delaunay) adjacency — roughly 40% of adjacent pairs are
    not one crossing apart. Read `seeds_crossed == 1` as "one boundary on the
    direct path", which is the question the confusion table is asking, and not as
    a claim about the cells' full topology.

    Sampling is stepped at a quarter of the tightest margin in the answer space,
    so a cell cannot be stepped over. Under-sampling could only ever *undercount*
    crossings, and `max_samples` caps the work on very long paths; the cap is
    recorded in the manifest when it binds.
    """
    lat = seeds["seed_lat"].to_numpy(dtype=float)
    lon = seeds["seed_lon"].to_numpy(dtype=float)
    K = len(seeds)
    out = np.zeros((K, K), dtype=int)
    if K < 2:
        return out

    margins = seeds["margin_km"].to_numpy(dtype=float)
    margins = margins[np.isfinite(margins) & (margins > 0)]
    step_km = max(1.0, float(margins.min()) / 4.0) if margins.size else 1.0
    mesh = pairwise_km(lat, lon)

    for i in range(K):
        for j in range(i + 1, K):
            n = int(np.clip(mesh[i, j] / step_km + 2, 32, max_samples))
            plat, plon = geodesic_points(lat[i], lon[i], lat[j], lon[j], n)
            owner = pairwise_km(plat, plon, lat, lon).argmin(axis=1)
            crossings = int((owner[1:] != owner[:-1]).sum())
            out[i, j] = out[j, i] = crossings
    return out


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
    #
    # One batched `cell_centers` call rather than one per group: HEALPix answers
    # the whole array in a single vectorized call, and asking per cell would let
    # a scalar-shaped implementation pass unnoticed.
    counts = t.groupby("cell_id", sort=True).size()
    cells = counts.index.to_numpy()
    centers = grid.cell_centers(cells, resolution)
    seeds = pd.DataFrame(
        {
            "seed_id": np.arange(len(cells)),
            "grid_scheme": grid.name,
            "grid_resolution": resolution,
            "cell_id": cells,
            # Deliberately unrounded: rounding would move the seed off the exact
            # centre and could break `cell_ids(seed) == cell_id` at a boundary.
            "seed_lat": centers[:, 0],
            "seed_lon": centers[:, 1],
            "n_targets": counts.to_numpy().astype(int),
        }
    )

    # Assign every target to its cell's seed: cell membership *is* the class.
    # Predictions are labelled by nearest seed instead (`classify.py`), which is
    # the same rule wherever a cell contains its own centre's Voronoi region.
    cell_to_seed = dict(zip(seeds["cell_id"], seeds["seed_id"]))
    t["seed_id"] = t["cell_id"].map(cell_to_seed).astype(int)

    seed_lat = seeds["seed_lat"].to_numpy()
    seed_lon = seeds["seed_lon"].to_numpy()
    d_to_all = pairwise_km(
        t["target_lat"].to_numpy(), t["target_lon"].to_numpy(), seed_lat, seed_lon
    )
    # The quantization offset, and the whole cost of the cell-centre choice:
    # how far a target sits from the centre standing in for it. Bounded by the
    # cell, so the grid and its resolution declare it up front.
    t["cell_offset_km"] = np.round(
        d_to_all[np.arange(len(t)), t["seed_id"].to_numpy()], 3
    )

    # Seed-to-seed geometry: the margin at which coordinate error flips a label.
    mesh = pairwise_km(seed_lat, seed_lon)
    K = len(seeds)
    if K > 1:
        nearest_km = (mesh + np.diag(np.full(K, np.inf))).min(axis=1)
    else:
        nearest_km = np.array([np.inf])
    seeds["nearest_seed_km"] = np.round(nearest_km, 3)
    # The boundary between two adjacent classes bisects the geodesic joining
    # their seeds, so half the per-seed minimum is the margin at the seed.
    seeds["margin_km"] = np.round(nearest_km / 2.0, 3)
    # §7.4's per-seed degree: how many classes an answer here is confusable
    # with. Defined as "one class boundary away on the direct path" -- see
    # `seed_crossing_matrix` for why that is the right cut and not merely a
    # tighter one.
    crossings = (
        seed_crossing_matrix(seeds) if 1 < K <= MAX_SEEDS_FOR_ADJACENCY else None
    )
    if crossings is not None:
        deg = (crossings == 1).sum(axis=1)
        seeds["class_adjacency_degree"] = deg
        adjacency_edge_km = mesh[np.triu(crossings == 1, k=1)]
    else:
        deg = None
        adjacency_edge_km = None

    mesh_df = pd.DataFrame(
        np.round(mesh, 3), index=seeds["seed_id"], columns=seeds["seed_id"]
    )
    mesh_df.index.name = "seed_id"

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
        "cell_offset_km": _describe(t["cell_offset_km"].to_numpy()),
        "nearest_seed_km": _describe(nearest_km),
        "margin_km": _describe(nearest_km / 2.0),
        "seed_pairwise_km": _describe(mesh[np.triu_indices(K, k=1)]) if K > 1 else {"n": 0},
    }
    if deg is not None:
        meta["class_adjacency_degree"] = _describe(deg.astype(float))
        # §7.4's "Delaunay edge lengths, as a distribution": the separations of
        # the pairs that actually border each other. Its per-seed minimum is
        # `nearest_seed_km`, whose half is the margin.
        meta["adjacency_edge_km"] = _describe(adjacency_edge_km)
    elif K > 1:
        meta["class_adjacency_degree"] = {
            "n": 0,
            "note": (
                f"not computed: K={K} exceeds MAX_SEEDS_FOR_ADJACENCY="
                f"{MAX_SEEDS_FOR_ADJACENCY}. The per-pair geodesic walk is "
                f"O(K^2) and superlinear in K (0.08s at K=27, 15.6s at K=120). "
                f"h3-4, the grid we run, sits at K=18-27."
            ),
        }

    assignments = t.loc[
        :,
        [
            "target_id",
            "target_lat",
            "target_lon",
            "cell_id",
            "seed_id",
            "cell_offset_km",
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
    if "centroid_lat" in seeds.columns:
        # Written when seeds were the spherical centroid of a cell's targets.
        # Every seed coordinate, hence every class boundary, differs; loading it
        # beside current code would silently mix two answer spaces.
        raise MissingArtifactError(
            f"{seeds_p} was built with target-centroid seeds (it has a "
            f"`centroid_lat` column); seeds are now cell centres. Rebuild with "
            f"`build-answer-space`."
        )
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
    buys (fewer classes) against what it costs (`cell_offset_km`: targets sit
    further from the cell centre standing in for them).
    """
    m = space.meta
    offset = m["cell_offset_km"]
    return {
        "grid": m["grid"]["scheme"],
        "resolution": m["grid"]["resolution"],
        "cell_km": round(m["grid"]["nominal_cell_km"], 1),
        "cell_area_km2": m["grid"]["cell_area_km2"],
        "n_targets": m["n_targets"],
        "n_classes": m["n_seeds"],
        "n_singleton_classes": m["n_singleton_seeds"],
        "cell_offset_km_p50": offset["percentiles"]["p50"],
        "cell_offset_km_max": offset["max"],
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
