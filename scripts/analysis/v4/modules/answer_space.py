"""The answer space: targets quantized to HEALPix cells, one class per cell.

A prediction is scored as a *classification* over places, not as a coordinate,
because the question an operator asks is "which site?" rather than "how many
km?". The grid is the quantizer that decides which targets are one place — and
it also decides *where* that place is, since a class is seeded at its **cell
centre** and not at the centroid of whichever targets happened to land in it.
A seed therefore never moves when the target set changes.

## Built over the whole ladder, not at one resolution

v4's reason for existing is that accuracy is a *curve* over cell size, so the
answer space is built at every rung of `NSIDE_LADDER` (128 -> 16, i.e. 50.9 km
-> 407 km). Each rung is a complete, self-describing artifact in its own
directory; `grid_sweep.healpix.csv` one level above joins them so the class
count can be read as a curve without opening four files.

Because NESTED ids coarsen by bit shift, every rung comes from **one**
`ang2pix` pass at the finest nside. The rungs are therefore exactly nested: a
class at nside 64 is the union of the nside-128 classes inside it, with no
boundary straddling. That is the property H3 could not provide and the reason
this package is not a parameterisation of v3.

## Adjacency is exact here

Two classes are adjacent when their cells are HEALPix ring-1 neighbours. v3 had
to approximate this by sampling the great-circle path between seeds and counting
boundary crossings, and its own docstring records the cost: mean degree 2.7-2.8
against 4.8-5.3 for true adjacency, so roughly 40% of adjacent pairs were
missed. HEALPix hands us the neighbour set directly, so `class_adjacency_degree`
is the real degree rather than a lower bound.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.v4.modules import healpix as H
from scripts.analysis.v4.modules.paths import RunPaths

EARTH_RADIUS_KM = 6371.0

SEEDS_CSV = "seeds.csv"
ASSIGNMENTS_CSV = "assignments.csv"
SEED_MESH_CSV = "seed_mesh_km.csv"
META_JSON = "meta.json"

#: Written one level above the rung directories, and only when more than one
#: rung was built -- a single-rung sweep has no curve to plot.
SWEEP_CSV = "grid_sweep.healpix.csv"

_TARGET_COLUMNS = ("target_id", "target_lat", "target_lon")


def _unit_vectors(lat_deg, lon_deg) -> np.ndarray:
    lat = np.radians(np.asarray(lat_deg, dtype=float).ravel())
    lon = np.radians(np.asarray(lon_deg, dtype=float).ravel())
    cos_lat = np.cos(lat)
    return np.column_stack([cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)])


def pairwise_km(lat_a, lon_a, lat_b=None, lon_b=None) -> np.ndarray:
    """Great-circle distance matrix in km, via unit-vector dot products.

    Chord form rather than haversine so one matrix multiply covers every pair;
    equivalent to within floating point.
    """
    a = _unit_vectors(lat_a, lon_a)
    b = a if lat_b is None else _unit_vectors(lat_b, lon_b)
    return EARTH_RADIUS_KM * np.arccos(np.clip(a @ b.T, -1.0, 1.0))


def elementwise_km(lat_a, lon_a, lat_b, lon_b) -> np.ndarray:
    """Paired distance: `out[i] = d(a[i], b[i])`. The diagonal, without the
    matrix — used for target-to-its-own-cell-centre offsets."""
    a = _unit_vectors(lat_a, lon_a)
    b = _unit_vectors(lat_b, lon_b)
    return EARTH_RADIUS_KM * np.arccos(np.clip(np.sum(a * b, axis=1), -1.0, 1.0))


def _describe(values) -> dict:
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    qs = [5, 25, 50, 75, 95]
    return {
        "n": int(v.size),
        "min": round(float(v.min()), 3),
        "max": round(float(v.max()), 3),
        "mean": round(float(v.mean()), 3),
        "percentiles": {
            f"p{q}": round(float(np.percentile(v, q)), 3) for q in qs
        },
    }


@dataclass(frozen=True)
class AnswerSpace:
    """Seeds, target assignments, and the geometry describing both."""

    nside: int
    seeds: pd.DataFrame
    assignments: pd.DataFrame
    seed_mesh_km: pd.DataFrame
    meta: dict

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)

    def write(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.seeds.to_csv(out_dir / SEEDS_CSV, index=False)
        self.assignments.to_csv(out_dir / ASSIGNMENTS_CSV, index=False)
        self.seed_mesh_km.to_csv(out_dir / SEED_MESH_CSV, index=True)
        (out_dir / META_JSON).write_text(json.dumps(self.meta, indent=2) + "\n")
        return out_dir


def load_answer_space(out_dir: Path) -> AnswerSpace:
    """Read a rung back. `cell_id` is coerced to int64 — a CSV round trip would
    otherwise hand back float ids, and a float cell id silently fails every
    equality test the classifier makes."""
    out_dir = Path(out_dir)
    seeds = pd.read_csv(out_dir / SEEDS_CSV)
    assignments = pd.read_csv(out_dir / ASSIGNMENTS_CSV)
    seeds["cell_id"] = seeds["cell_id"].astype("int64")
    assignments["cell_id"] = assignments["cell_id"].astype("int64")
    mesh = pd.read_csv(out_dir / SEED_MESH_CSV, index_col=0)
    meta = json.loads((out_dir / META_JSON).read_text())
    return AnswerSpace(
        nside=int(seeds["grid_resolution"].iloc[0]),
        seeds=seeds,
        assignments=assignments,
        seed_mesh_km=mesh,
        meta=meta,
    )


def build_answer_space(
    targets: pd.DataFrame,
    *,
    nside: int = H.DEFAULT_NSIDE,
    source_label: str | None = None,
    provenance: dict | None = None,
) -> AnswerSpace:
    """Quantize `targets` at one nside and derive the classes.

    `targets` needs `target_id`, `target_lat`, `target_lon`. Duplicate
    `target_id` is an error; duplicate *coordinates* are expected — distinct IPs
    at one facility — and collapse into one class.
    """
    missing = [c for c in _TARGET_COLUMNS if c not in targets.columns]
    if missing:
        raise ValueError(f"targets is missing {missing}; needs {list(_TARGET_COLUMNS)}")
    if targets.empty:
        raise ValueError("targets is empty; there is no answer space to build")
    dup = targets["target_id"].duplicated()
    if dup.any():
        raise ValueError(
            f"duplicate target_id: {sorted(targets.loc[dup, 'target_id'].unique())[:5]}"
        )

    nside = H.validate_nside(nside)
    t = targets[list(_TARGET_COLUMNS)].reset_index(drop=True).copy()
    cells = H.ang2pix(t["target_lat"], t["target_lon"], nside)

    # Classes in ascending cell id, so seed_id is a deterministic function of
    # the occupied cell set and not of the row order the targets arrived in.
    occupied = np.unique(cells)
    centres = H.pix2ang(occupied, nside)
    seed_of_cell = {int(c): i for i, c in enumerate(occupied)}

    t["cell_id"] = cells
    t["seed_id"] = [seed_of_cell[int(c)] for c in cells]
    t["cell_offset_km"] = np.round(
        elementwise_km(
            t["target_lat"], t["target_lon"],
            centres[t["seed_id"].to_numpy(), 0], centres[t["seed_id"].to_numpy(), 1],
        ),
        3,
    )

    mesh = pairwise_km(centres[:, 0], centres[:, 1])
    k = len(occupied)
    if k > 1:
        off = mesh + np.diag(np.full(k, np.inf))
        nearest_km = off.min(axis=1)
    else:
        nearest_km = np.full(k, np.nan)

    # Exact adjacency: ring-1 cell neighbours, intersected with the occupied set.
    nbrs = H.neighbours(occupied, nside)
    occupied_set = set(int(c) for c in occupied)
    degree = np.array(
        [len({int(x) for x in row if x >= 0} & occupied_set) for row in nbrs],
        dtype=int,
    )

    counts = pd.Series(t["seed_id"]).value_counts().reindex(range(k), fill_value=0)
    seeds = pd.DataFrame(
        {
            "seed_id": np.arange(k, dtype=int),
            "grid_scheme": "healpix",
            "grid_resolution": nside,
            "cell_id": occupied.astype("int64"),
            "seed_lat": centres[:, 0],
            "seed_lon": centres[:, 1],
            "n_targets": counts.to_numpy(dtype=int),
            "nearest_seed_km": np.round(nearest_km, 3),
            # Half the gap to the nearest other class: the largest error that
            # cannot, on the direct path, reach another class.
            "margin_km": np.round(nearest_km / 2.0, 3),
            "class_adjacency_degree": degree,
        }
    )

    mesh_df = pd.DataFrame(np.round(mesh, 3), index=seeds["seed_id"], columns=seeds["seed_id"])
    mesh_df.index.name = "seed_id"

    meta = {
        "source": source_label,
        **({"targets_provenance": provenance} if provenance else {}),
        "grid": H.describe(nside),
        "n_targets": int(len(t)),
        "n_unique_target_coords": int(
            len(t[["target_lat", "target_lon"]].drop_duplicates())
        ),
        "n_seeds": int(k),
        "n_singleton_seeds": int((seeds["n_targets"] == 1).sum()),
        # The ladder at and below this rung. Exactly nested, so this is a
        # coarsening of THIS rung's classes and not an independent re-binning.
        "occupied_cells_by_nside": {
            str(n): c
            for n, c in H.occupied_cells_by_nside(
                t["target_lat"], t["target_lon"], H.ladder_for(nside)
            ).items()
        },
        # How far a target sits from the centre it is represented by. This is
        # the quantization floor: no estimator, however good, scores better
        # than this against the seed.
        "cell_offset_km": _describe(t["cell_offset_km"]),
        "nearest_seed_km": _describe(nearest_km),
        "margin_km": _describe(nearest_km / 2.0),
        "class_adjacency_degree": _describe(degree.astype(float)),
        "adjacency_note": (
            "ring-1 HEALPix neighbours intersected with the occupied set — exact, "
            "unlike a path-sampling approximation"
        ),
    }

    return AnswerSpace(
        nside=nside, seeds=seeds, assignments=t, seed_mesh_km=mesh_df, meta=meta
    )


def sweep_row(space: AnswerSpace) -> dict:
    """One row of `grid_sweep.healpix.csv` — the class-count curve."""
    m = space.meta
    return {
        "grid": "healpix",
        "nside": space.nside,
        "cell_km": round(H.nominal_cell_km(space.nside), 1),
        "cell_area_km2": round(H.pixel_area_km2(space.nside), 3),
        "n_targets": m["n_targets"],
        "n_classes": m["n_seeds"],
        "n_singleton_classes": m["n_singleton_seeds"],
        "cell_offset_km_p50": m["cell_offset_km"]["percentiles"]["p50"],
        "cell_offset_km_max": m["cell_offset_km"]["max"],
        "nearest_seed_km_p50": m["nearest_seed_km"]["percentiles"]["p50"],
        "mean_adjacency_degree": m["class_adjacency_degree"]["mean"],
    }


def load_targets(run: RunPaths) -> pd.DataFrame:
    """The run's evaluated target roster, from the fold parquets.

    Read from what the benchmark actually produced rather than from
    `targets.csv`: a config listing five folds of which four ran leaves the CSV
    a superset, and scoring against targets nobody evaluated would put classes
    in the answer space that no method could ever be credited with.
    """
    import pyarrow.parquet as pq

    combos = run.combo_ids
    if not combos:
        raise ValueError(
            f"{run.setup_dir} holds no combo with a targets.parquet; "
            f"run the benchmark before building an answer space"
        )
    frames = []
    for fold in run.fold_ids:
        path = run.combo_dir(combos[0], fold) / "targets.parquet"
        if not path.exists():
            continue
        frames.append(
            pq.read_table(path, columns=list(_TARGET_COLUMNS)).to_pandas()
        )
    if not frames:
        raise ValueError(f"no targets.parquet found for {combos[0]!r} in {run.setup_dir}")
    return pd.concat(frames, ignore_index=True).drop_duplicates("target_id")


def build_for_run(
    run: RunPaths,
    *,
    nsides: tuple[int, ...] = H.NSIDE_LADDER,
    analysis_root: Path | None = None,
) -> list[AnswerSpace]:
    """Build and write every rung for one run, plus the merged sweep CSV."""
    targets = load_targets(run)
    label = f"{run.run_id}/{run.source}/{run.setup}"
    spaces: list[AnswerSpace] = []
    for nside in sorted({H.validate_nside(n) for n in nsides}, reverse=True):
        space = build_answer_space(targets, nside=nside, source_label=label)
        space.write(run.answer_space_dir(nside, root=analysis_root))
        spaces.append(space)
    if len(spaces) > 1:
        rows = pd.DataFrame([sweep_row(s) for s in spaces])
        rows.to_csv(
            run.analysis_dir("target-answer-space", root=analysis_root) / SWEEP_CSV,
            index=False,
        )
    return spaces
