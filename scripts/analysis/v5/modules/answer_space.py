"""The TG answer space: a grid partition and a cell partition, per rung.

v4's answer space had one partition, the HEALPix grid, and graded a prediction
by `ring` -- how many grids out it landed. That bounds the **distance** but is
blind to **direction**: a miss one ring out lands either inside the TG's own
serving region or in a neighbour's, and an operator cares which.

v5 adds the second partition. Both are built at one tolerance, `grid_km`:

* **grid** -- HEALPix pixels at nside, `grid_km = sqrt(area)` across.
* **cell** -- the Voronoi cell of each **seed**, bounded by the **landmass**.
  Seeds are the spherical centroids of sites grouped by complete linkage with
  diameter `grid_km`; the landmass is the US mainland buffered by `grid_km`.
  Also called the serving region: people are served by the nearest site.

A prediction then carries two labels: `ring` against the grid partition and
`cell_label` against the cell partition (see `classify`).

## Glossary

`GLOSSARY` below is binding for every name in v5 and is written into every
`meta.json` and manifest, so an artifact carries its own definitions.

## Artifacts, per rung

`grids.csv` (occupied grids), `sites.csv`, `seeds.csv`, `tgs.csv` (one row per
TG, with its site, grid and seed), `meta.json`; and `sweep.csv` one level up,
one row per rung.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules import sites as S
from scripts.analysis.v5.modules.geodesy import elementwise_km
from scripts.analysis.v5.modules.landmass import load_landmass
from scripts.analysis.v5.modules.paths import ANSWER_SPACE_KIND, RunPaths
from scripts.analysis.v5.modules.seeds import build_seeds

GLOSSARY: dict[str, str] = {
    "tg": "target: one target server behind an IP",
    "grid": "HEALPix pixel at this rung (NESTED); the grid partition",
    "grid_km": "nominal grid distance, sqrt(grid area)",
    "ring": "grid steps between the TG's grid and the prediction's grid; -1 = beyond max_ring",
    "site": "unique location of TGs, keyed (run_id, tg_lat, tg_lon)",
    "seed": "spherical centroid of sites grouped by complete linkage, diameter <= grid_km",
    "landmass": "US mainland (lower 48, Great Lakes inland) buffered by grid_km",
    "cell": "Voronoi cell of a seed bounded by the landmass; the serving region",
    "answer_space": "the TG answer space: grid partition and cell partition at one rung",
    "*_dist_to_tg_km": "distance to the raw TG coordinate",
    "*_dist_to_seed_km": "distance to the TG's seed",
}

GRIDS_CSV = "grids.csv"
SITES_CSV = "sites.csv"
SEEDS_CSV = "seeds.csv"
TGS_CSV = "tgs.csv"
META_JSON = "meta.json"
SWEEP_CSV = "sweep.csv"

#: The TG columns v5 works in. `load_tgs` is the one place the v2 benchmark's
#: `target_*` names are mapped onto them.
TG_COLUMNS = ("tg_id", "tg_lat", "tg_lon")
BENCHMARK_TG_COLUMNS = {"target_id": "tg_id", "target_lat": "tg_lat", "target_lon": "tg_lon"}

_INT_COLUMNS = {
    GRIDS_CSV: ("grid_id", "n_tgs", "n_sites"),
    SITES_CSV: ("site_id", "n_tgs", "grid_id", "seed_id"),
    SEEDS_CSV: ("seed_id", "n_sites", "n_tgs"),
    TGS_CSV: ("site_id", "tg_grid_id", "tg_seed_id"),
}


def _describe(values) -> dict:
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "min": round(float(v.min()), 3),
        "max": round(float(v.max()), 3),
        "mean": round(float(v.mean()), 3),
        "percentiles": {
            f"p{q}": round(float(np.percentile(v, q)), 3) for q in (5, 25, 50, 75, 95)
        },
    }


@dataclass(frozen=True)
class AnswerSpace:
    """Both partitions of one rung, and the TGs placed in them."""

    nside: int
    grids: pd.DataFrame
    sites: pd.DataFrame
    seeds: pd.DataFrame
    tgs: pd.DataFrame
    meta: dict

    @property
    def grid_km(self) -> float:
        return G.grid_km(self.nside)

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)

    def write(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.grids.to_csv(out_dir / GRIDS_CSV, index=False)
        self.sites.to_csv(out_dir / SITES_CSV, index=False)
        self.seeds.to_csv(out_dir / SEEDS_CSV, index=False)
        self.tgs.to_csv(out_dir / TGS_CSV, index=False)
        (out_dir / META_JSON).write_text(json.dumps(self.meta, indent=2) + "\n")
        return out_dir


def load_answer_space(out_dir: Path) -> AnswerSpace:
    """Read a rung back, with id columns coerced to int64 -- a float id from a
    CSV round trip silently fails every equality test the scorer makes."""
    out_dir = Path(out_dir)
    frames = {}
    for name, ints in _INT_COLUMNS.items():
        df = pd.read_csv(out_dir / name)
        for col in ints:
            df[col] = df[col].astype("int64")
        frames[name] = df
    meta = json.loads((out_dir / META_JSON).read_text())
    return AnswerSpace(
        nside=int(meta["grid"]["nside"]),
        grids=frames[GRIDS_CSV],
        sites=frames[SITES_CSV],
        seeds=frames[SEEDS_CSV],
        tgs=frames[TGS_CSV],
        meta=meta,
    )


def build_answer_space(
    tgs: pd.DataFrame,
    *,
    nside: int = G.DEFAULT_NSIDE,
    run_id: str,
    source_label: str | None = None,
) -> AnswerSpace:
    """Place `tgs` in both partitions at one rung.

    `tgs` needs `tg_id, tg_lat, tg_lon`. Duplicate `tg_id` is an error;
    duplicate coordinates are expected -- replicas at one site.
    """
    missing = [c for c in TG_COLUMNS if c not in tgs.columns]
    if missing:
        raise ValueError(f"tgs is missing {missing}; needs {list(TG_COLUMNS)}")
    if tgs.empty:
        raise ValueError("tgs is empty; there is no answer space to build")
    dup = tgs["tg_id"].duplicated()
    if dup.any():
        raise ValueError(f"duplicate tg_id: {sorted(tgs.loc[dup, 'tg_id'].unique())[:5]}")
    if tgs[["tg_lat", "tg_lon"]].isna().any().any():
        raise ValueError("a TG with no coordinate cannot be placed in the answer space")

    nside = G.validate_nside(nside)
    grid_km = G.grid_km(nside)
    t = tgs[list(TG_COLUMNS)].reset_index(drop=True).copy()

    # -- sites ------------------------------------------------------------
    t["site_id"] = S.site_ids(t, run_id=run_id).to_numpy()
    sites = (
        t.groupby("site_id", sort=True)
        .agg(site_lat=("tg_lat", "first"), site_lon=("tg_lon", "first"), n_tgs=("tg_id", "size"))
        .reset_index()
    )

    # -- grid partition -----------------------------------------------------
    t["tg_grid_id"] = G.ang2pix(t["tg_lat"], t["tg_lon"], nside)
    sites["grid_id"] = G.ang2pix(sites["site_lat"], sites["site_lon"], nside)
    occupied = np.unique(t["tg_grid_id"])
    centres = G.pix2ang(occupied, nside)
    grids = pd.DataFrame(
        {
            "grid_id": occupied.astype("int64"),
            "grid_lat": centres[:, 0],
            "grid_lon": centres[:, 1],
            "n_tgs": t.groupby("tg_grid_id").size().reindex(occupied).to_numpy(dtype=int),
            "n_sites": sites.groupby("grid_id").size().reindex(occupied).to_numpy(dtype=int),
        }
    )
    centre_of = dict(zip(occupied.tolist(), map(tuple, centres)))
    tg_centre = np.array([centre_of[int(g)] for g in t["tg_grid_id"]])
    t["tg_dist_to_grid_centre_km"] = np.round(
        elementwise_km(t["tg_lat"], t["tg_lon"], tg_centre[:, 0], tg_centre[:, 1]), 3
    )

    # -- cell partition -----------------------------------------------------
    landmass = load_landmass(grid_km)
    outland = ~landmass.contains(sites["site_lat"], sites["site_lon"])
    if outland.any():
        bad = sites.loc[outland, ["site_lat", "site_lon"]].round(4).values.tolist()
        raise ValueError(
            f"{int(outland.sum())} site(s) lie outside the landmass buffered by "
            f"{grid_km:.1f} km, e.g. {bad[:3]}: their own TGs would be outland. "
            f"The dataset and the landmass do not match."
        )
    seed_of_site, seeds = build_seeds(sites, grid_km)
    sites["seed_id"] = seed_of_site
    too_wide = seeds["seed_diameter_km"] > grid_km + 1e-6
    if too_wide.any():
        raise AssertionError(
            f"complete linkage produced seeds wider than {grid_km:.3f} km: "
            f"{seeds.loc[too_wide, 'seed_id'].tolist()}"
        )
    t["tg_seed_id"] = sites.set_index("site_id")["seed_id"].reindex(t["site_id"]).to_numpy()
    seed_xy = seeds.set_index("seed_id").loc[t["tg_seed_id"], ["seed_lat", "seed_lon"]].to_numpy()
    t["tg_dist_to_seed_km"] = np.round(
        elementwise_km(t["tg_lat"], t["tg_lon"], seed_xy[:, 0], seed_xy[:, 1]), 3
    )

    meta = {
        "source": source_label,
        "run_id": run_id,
        "grid": G.describe(nside),
        "grid_km": round(grid_km, 3),
        "landmass": landmass.describe(),
        "seed_rule": {
            "method": "complete linkage over site great-circle distances",
            "diameter_km": round(grid_km, 3),
            "centroid": "spherical (normalised mean unit vector)",
        },
        "n_tgs": int(len(t)),
        "n_sites": int(len(sites)),
        "n_grids": int(len(grids)),
        "n_seeds": int(len(seeds)),
        # Sites that share a seed with at least one other site.
        "n_sites_merged": int((seeds.loc[seeds["n_sites"] > 1, "n_sites"]).sum()),
        "tg_dist_to_grid_centre_km": _describe(t["tg_dist_to_grid_centre_km"]),
        "tg_dist_to_seed_km": _describe(t["tg_dist_to_seed_km"]),
        "seed_diameter_km": _describe(seeds["seed_diameter_km"]),
        "nearest_seed_km": _describe(seeds["nearest_seed_km"]),
        "glossary": GLOSSARY,
    }
    return AnswerSpace(nside=nside, grids=grids, sites=sites, seeds=seeds, tgs=t, meta=meta)


def sweep_row(space: AnswerSpace) -> dict:
    """One row of `sweep.csv`: how both partitions coarsen up the ladder."""
    m = space.meta
    return {
        "nside": space.nside,
        "grid_km": round(space.grid_km, 1),
        "n_tgs": m["n_tgs"],
        "n_sites": m["n_sites"],
        "n_grids": m["n_grids"],
        "n_seeds": m["n_seeds"],
        "n_sites_merged": m["n_sites_merged"],
        "tg_dist_to_grid_centre_km_p50": m["tg_dist_to_grid_centre_km"]["percentiles"]["p50"],
        "tg_dist_to_seed_km_p50": m["tg_dist_to_seed_km"]["percentiles"]["p50"],
        "tg_dist_to_seed_km_max": m["tg_dist_to_seed_km"]["max"],
    }


def load_tgs(run: RunPaths) -> pd.DataFrame:
    """The run's evaluated TG roster, from the fold parquets.

    From what the benchmark produced rather than `targets.csv`, which can be a
    superset when not every fold ran. The benchmark's `target_*` columns are
    renamed to `tg_*` here and nowhere else.
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
        if path.exists():
            frames.append(
                pq.read_table(path, columns=list(BENCHMARK_TG_COLUMNS)).to_pandas()
            )
    if not frames:
        raise ValueError(f"no targets.parquet found for {combos[0]!r} in {run.setup_dir}")
    return (
        pd.concat(frames, ignore_index=True)
        .rename(columns=BENCHMARK_TG_COLUMNS)
        .drop_duplicates("tg_id")
    )


def build_for_run(
    run: RunPaths,
    *,
    nsides: tuple[int, ...] = G.NSIDE_LADDER,
    analysis_root: Path | None = None,
) -> list[AnswerSpace]:
    """Build and write every rung for one run, plus `sweep.csv`."""
    tgs = load_tgs(run)
    label = f"{run.run_id}/{run.source}/{run.setup}"
    spaces: list[AnswerSpace] = []
    for nside in sorted({G.validate_nside(n) for n in nsides}, reverse=True):
        space = build_answer_space(tgs, nside=nside, run_id=run.run_id, source_label=label)
        space.write(run.answer_space_dir(nside, root=analysis_root))
        spaces.append(space)
    if len(spaces) > 1:
        pd.DataFrame([sweep_row(s) for s in spaces]).to_csv(
            run.analysis_dir(ANSWER_SPACE_KIND, root=analysis_root) / SWEEP_CSV,
            index=False,
        )
    return spaces
