"""Build the target answer space: HEALPix quantizer -> seeds -> Voronoi (§7.3/§7.4).

Three steps, exactly as the paper states them:

1. Quantize every ground-truth target onto an equal-area HEALPix grid
   (`nside=128`, ~51 km). Cell membership is an equivalence relation whose only
   job is to merge points close enough to count as one place.
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

from scripts.analysis.v3.modules import healpix as hx
from scripts.analysis.v3.modules import io
from scripts.analysis.v3.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    MissingArtifactError,
    RunPaths,
    discover_runs,
    resolve_run,
)
from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM

SWEEP_CSV = "nside_sweep.csv"
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
    nside: int = hx.DEFAULT_NSIDE,
    source_label: str | None = None,
) -> AnswerSpace:
    """Quantize `targets` and derive the seed set plus its geometry.

    `targets` needs `target_id`, `target_lat`, `target_lon`. Duplicate
    `target_id` is an error; duplicate *coordinates* are fine and expected
    (distinct IPs at one facility), and they collapse into one seed.
    """
    nside = hx.validate_nside(nside)
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
    t["healpix_pix"] = hx.ang2pix(t["target_lat"], t["target_lon"], nside)

    # Seed per occupied cell, ordered by cell id so seed_id is deterministic
    # and independent of input row order.
    seed_rows: list[dict] = []
    spread_by_pix: dict[int, float] = {}
    for rank, (pix, grp) in enumerate(t.groupby("healpix_pix", sort=True)):
        clat, clon = hx.spherical_centroid(grp["target_lat"], grp["target_lon"])
        if len(grp) > 1:
            d = pairwise_km(grp["target_lat"].to_numpy(), grp["target_lon"].to_numpy())
            spread = float(d.max())
        else:
            spread = 0.0
        spread_by_pix[int(pix)] = spread
        seed_rows.append(
            {
                "seed_id": rank,
                "healpix_nside": nside,
                "healpix_pix": int(pix),
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
    pix_to_seed = dict(zip(seeds["healpix_pix"], seeds["seed_id"]))
    t["seed_id"] = t["healpix_pix"].map(pix_to_seed).astype(int)

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

    pitch = hx.nominal_cell_km(nside)
    n_close_pairs = (
        int(((mesh < pitch) & (mesh > 0)).sum() // 2) if K > 1 else 0
    )
    meta = {
        "source": source_label,
        "grid": {
            "scheme": "healpix",
            "order": "nested",
            "nside": nside,
            "npix": hx.npix(nside),
            "pixel_area_km2": round(hx.pixel_area_km2(nside), 3),
            "nominal_cell_km": round(pitch, 3),
        },
        "n_targets": int(len(t)),
        "n_unique_target_coords": int(
            t.loc[:, ["target_lat", "target_lon"]].drop_duplicates().shape[0]
        ),
        "n_seeds": K,
        "n_singleton_seeds": int((seeds["n_targets"] == 1).sum()),
        "occupied_cells_by_nside": {
            str(n): c
            for n, c in hx.occupied_cell_hierarchy(
                t["target_lat"], t["target_lon"]
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
            "healpix_pix",
            "seed_id",
            "dist_to_seed_km",
            "nearest_seed_id",
            "cell_seed_is_nearest",
        ],
    ].reset_index(drop=True)

    return AnswerSpace(
        seeds=seeds, assignments=assignments, seed_mesh_km=mesh_df, meta=meta
    )


def build_for_run(run: RunPaths, *, nside: int = hx.DEFAULT_NSIDE) -> AnswerSpace:
    """Answer space over a run's full ground-truth target set.

    Reads `targets.csv`, which is fold-independent: K-fold splits targets, so
    the union over folds is exactly this file. Building the space once per run
    (rather than per fold) is what makes fold-pooled scoring coherent.
    """
    targets = io.load_targets(run)
    return build_answer_space(
        targets, nside=nside, source_label=f"{run.run_id}/{run.source}/{run.setup}"
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
    return AnswerSpace(
        seeds=pd.read_csv(seeds_p),
        assignments=pd.read_csv(assign_p),
        seed_mesh_km=mesh,
        meta=meta,
    )


def sweep_row(space: "AnswerSpace") -> dict:
    """One line of `nside_sweep.csv`: how the partition changes with the grid.

    The point of the sweep is the trade-off, so each row pairs what coarsening
    buys (fewer classes, fewer straddle candidates) against what it costs
    (targets pulled further from their own seed, which floors error distance).
    """
    m = space.meta
    spread = m["intra_seed_spread_km"]
    return {
        "nside": m["grid"]["nside"],
        "cell_km": round(m["grid"]["nominal_cell_km"], 1),
        "cell_area_km2": m["grid"]["pixel_area_km2"],
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
        nside: list[int] = typer.Option(
            [hx.DEFAULT_NSIDE],
            "--nside",
            help="HEALPix nside (power of two, repeatable). 128 = ~51 km, the "
                 "paper §7.3 setting; 64 = ~102 km (metro); 32 = ~204 km (region).",
        ),
        sweep: bool = typer.Option(
            False,
            "--sweep",
            help=f"Shorthand for the full hierarchy {list(hx.NSIDE_HIERARCHY)}.",
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """Quantize a run's targets onto HEALPix and emit the seed answer space.

        Writes seeds.csv, assignments.csv, seed_mesh_km.csv and meta.json to
        <analysis_root>/<run_id>/target-answer-space/nside-<x>/, one directory
        per requested nside. Building more than one also writes
        nside_sweep.csv one level up, comparing the partitions side by side.
        """
        if all_runs == (run_id is not None):
            raise typer.BadParameter("pass exactly one of --run-id or --all-runs")
        nsides = list(hx.NSIDE_HIERARCHY) if sweep else list(dict.fromkeys(nside))

        runs = discover_runs(outputs_root) if all_runs else [resolve_run(run_id, outputs_root)]
        for run in runs:
            rows = []
            for ns in nsides:
                space = build_for_run(run, nside=ns)
                out = space.write(run.answer_space_dir(root=analysis_root, nside=ns))
                rows.append(sweep_row(space))
                m = space.meta
                typer.echo(
                    f"{run.run_id}: {m['n_targets']} targets -> K={m['n_seeds']} seeds "
                    f"({m['n_singleton_seeds']} singleton) at nside={ns} "
                    f"[{m['grid']['nominal_cell_km']:.1f} km] -> {out}"
                )
            if len(rows) > 1:
                sweep_dir = run.analysis_dir("target-answer-space", root=analysis_root)
                pd.DataFrame(rows).to_csv(sweep_dir / SWEEP_CSV, index=False)
                typer.echo(f"{run.run_id}: sweep -> {sweep_dir / SWEEP_CSV}")
