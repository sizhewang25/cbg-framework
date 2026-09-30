"""The TG answer space: a grid partition and a cell partition.

v4's answer space had one partition, the HEALPix grid, and graded a prediction
by `ring` -- how many grids out it landed. That bounds the **distance** but is
blind to **direction**: a miss one ring out lands either inside the TG's own
serving region or in a neighbour's, and an operator cares which.

v5 adds the second partition. Both are built at one tolerance, `grid_km`:

* **grid** -- HEALPix pixels at nside, `grid_km = sqrt(area)` across.
* **cell** -- the Voronoi cell of each **seed**, unbounded. Seeds are the
  spherical centroids of sites grouped by complete linkage with diameter
  `grid_km`. Also called the serving region: people are served by the nearest
  site.

A prediction then carries two labels: `ring` against the grid partition and
`cell_label` against the cell partition (see `classify`).

## The cell partition is unbounded, deliberately

Every point on Earth is nearest to some seed, so `cell_label` alone will call
a prediction thousands of km away "correct" whenever it happens to fall on the
right side of a distant bisector. That is not a defect to patch here: it is
the property the evaluation measures, by reading `ring` beside `cell_label`.
Nothing in this module restricts where a cell may reach.

## A traffic-weighted run is built over its mesh

Its evaluated TGs are the ones that survived the flow filter, a subset of the
mesh. The answer space is built over the **mesh** (the config's
`mesh_csv_path`) and only the evaluated TGs are scored against it, so the
weighted arm and its mesh arm share one set of sites, seeds and cells and differ
only in who is scored. Built from the evaluated TGs instead, dropped sites would
lose their seeds and hand their cells to neighbours -- a cell-label gain every
method gets for free. `require_mesh_universe` refuses a weighted space built the
old way. Ported from v3, which had this; v4 lost it.

## Glossary

`GLOSSARY` below is binding for every name in v5 and is written into every
`meta.json` and manifest, so an artifact carries its own definitions.

## Artifacts

`grids.csv` (occupied grids), `sites.csv`, `seeds.csv`, `tgs.csv` (one row per
TG, with its site, grid and seed), and `meta.json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules import sites as S
from scripts.analysis.v5.modules.geodesy import elementwise_km
from scripts.analysis.v5.modules import projection
from scripts.analysis.v5.modules.paths import REPO_ROOT, MissingArtifactError, RunPaths
from scripts.analysis.v5.modules.seeds import build_seeds

GLOSSARY: dict[str, str] = {
    "tg": "target: one target server behind an IP",
    "grid": "HEALPix pixel at this rung (NESTED); the grid partition",
    "grid_km": "nominal grid distance, sqrt(grid area)",
    "pred_dist_to_tg_grid": "grid steps from the TG's grid to the prediction's; -1 = no prediction",
    "ring tiers": "ring0/ring1/ring2/beyond, the bands summarize groups that offset into",
    "site": "unique location of TGs, keyed (run_id, tg_lat, tg_lon)",
    "seed": "spherical centroid of sites grouped by complete linkage, diameter <= grid_km",
    "cell": "Voronoi cell of a seed, unbounded; the serving region",
    "answer_space": "the TG answer space: grid partition and cell partition at one rung",
    "*_dist_to_tg_km": "distance to the raw TG coordinate",
    "*_dist_to_seed_km": "distance to the TG's seed",
}

GRIDS_CSV = "grids.csv"
SITES_CSV = "sites.csv"
SEEDS_CSV = "seeds.csv"
TGS_CSV = "tgs.csv"
META_JSON = "meta.json"

#: The benchmark source whose evaluated TGs are a post-filter subset of a mesh.
#: Its answer space is built over that mesh instead; see `build_for_run`.
WEIGHTED_SOURCE = "traffic_weighted_csv"

#: `sites.csv` column: how many of the site's TGs the run scores. Equal to
#: `n_tgs` except on a weighted run, where a mesh site can hold none. Written by
#: `build_for_run`; absent from older artifacts, which `site_n_scored` reads as
#: "every TG scored" -- true of every run a space was built from its own TGs.
SCORED_COL = "n_tgs_scored"

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
    if SCORED_COL in frames[SITES_CSV].columns:
        frames[SITES_CSV][SCORED_COL] = frames[SITES_CSV][SCORED_COL].astype("int64")
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
    provenance: dict | None = None,
) -> AnswerSpace:
    """Place `tgs` in both partitions at one rung.

    `tgs` needs `tg_id, tg_lat, tg_lon`. Duplicate `tg_id` is an error;
    duplicate coordinates are expected -- replicas at one site.

    `provenance`, when given, is written to `meta["targets_provenance"]`. Only
    a traffic-weighted run passes one, and its presence is what
    `require_mesh_universe` checks.
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
    # No admissibility guard: an unbounded Voronoi is well defined for seeds
    # anywhere on Earth, so there is nothing a site could violate.
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
        "projection": projection.describe(),
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
    if provenance is not None:
        meta["targets_provenance"] = dict(provenance)
    return AnswerSpace(nside=nside, grids=grids, sites=sites, seeds=seeds, tgs=t, meta=meta)


def load_tgs(run: RunPaths) -> pd.DataFrame:
    """The run's evaluated TG roster, from the fold parquets.

    From what the benchmark produced rather than `targets.csv`, which can be a
    superset when not every fold ran. The benchmark's `target_*` columns are
    renamed to `tg_*` here and nowhere else -- `_mesh_tgs_for_weighted_run`
    renames too, through the same `BENCHMARK_TG_COLUMNS`.

    This is the answer space's universe for every run **except** a
    traffic-weighted one, where it is only the scored subset; see `build_for_run`.
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


def _mesh_tgs_for_weighted_run(run: RunPaths) -> tuple[pd.DataFrame, str]:
    """The **pre-filter** TG universe of a traffic-weighted run. Ported from v3.

    A weighted run evaluates only the TGs that survive the flow filter, so its
    fold parquets -- what `load_tgs` reads -- are post-filter. Building the
    answer space from them makes the weighted arm incomparable to its mesh:

    * **Fewer seeds.** A site whose TGs are all filtered out has no seed, and
      its Voronoi cell is absorbed by its neighbours. A prediction landing at
      that site, `wrong` on the mesh, can become `correct` here -- a gain every
      method gets for free.
    * **Moved seeds.** A seed is the centroid of the sites complete linkage
      groups, so losing one of a group's sites moves it.

    So the universe is the config's own `mesh_csv_path` -- the file the filter
    was applied to. Not the sibling `-mesh` run: the weighted arm reads
    `.tbweight.csv`, the mesh arm `.sanitized.csv`, and depending on another
    run having been analysed would be a needless coupling.

    Raises rather than falling back to the run's own TGs: silently scoring a
    weighted run on a reduced answer space is the bug this exists to prevent.
    """
    from scripts.analysis.v5.modules import labels

    cfg = labels.run_config_path(run)
    if cfg is None:
        raise MissingArtifactError(
            f"{run.run_id} is a {WEIGHTED_SOURCE} run, so its answer space is built "
            f"from the pre-filter mesh, but no config is reachable through "
            f"{run.target_space_json} (its `config` key). Refusing to fall back to "
            f"the run's own post-filter TGs."
        )
    node = yaml.safe_load(cfg.read_text()) or {}
    kwargs = (node.get("benchmark") or {}).get("source_kwargs") or {}
    raw = kwargs.get("mesh_csv_path")
    if not raw:
        raise MissingArtifactError(
            f"{run.run_id} is a {WEIGHTED_SOURCE} run, but {cfg} declares no "
            f"benchmark.source_kwargs.mesh_csv_path (found {sorted(kwargs)}). "
            f"Building from the run's post-filter TGs would drop seeds and make "
            f"this arm incomparable to the mesh, so it is refused."
        )
    mesh_csv = Path(raw) if Path(raw).is_absolute() else REPO_ROOT / raw
    if not mesh_csv.exists():
        raise MissingArtifactError(
            f"{run.run_id}: {cfg} points mesh_csv_path at {mesh_csv}, which does "
            f"not exist."
        )

    need = list(BENCHMARK_TG_COLUMNS)
    df = pd.read_csv(mesh_csv, usecols=lambda c: c in need)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{mesh_csv} lacks {missing}; it is not a canonical CSV")
    # One row per (VP, TG) in a canonical CSV; collapse to the TG roster. A TG
    # with two coordinates would be placed at whichever row came first.
    df = df.drop_duplicates()
    split = df["target_id"].duplicated(keep=False)
    if split.any():
        raise ValueError(
            f"{mesh_csv}: {df.loc[split, 'target_id'].nunique()} target_id(s) carry "
            f"more than one coordinate, e.g. {sorted(df.loc[split, 'target_id'])[:3]}"
        )
    # Sorted by id so the artifacts are deterministic. Nothing reads the order:
    # site and seed ids are assigned in sorted-coordinate order.
    tgs = (
        df.rename(columns=BENCHMARK_TG_COLUMNS)
        .sort_values("tg_id")
        .reset_index(drop=True)
    )
    # Recorded repo-relative, as the benchmark records its own CSV paths.
    shown = mesh_csv.relative_to(REPO_ROOT) if mesh_csv.is_relative_to(REPO_ROOT) else mesh_csv
    return tgs, str(shown)


def _check_scored_in_mesh(run: RunPaths, scored: pd.DataFrame, mesh: pd.DataFrame, src: str) -> None:
    """Every scored TG is in the mesh, at the same coordinate."""
    joined = scored.merge(mesh, on="tg_id", how="left", suffixes=("", "_mesh"))
    absent = joined["tg_lat_mesh"].isna()
    if absent.any():
        raise ValueError(
            f"{run.run_id}: {int(absent.sum())} of {len(scored)} scored TG(s) are "
            f"absent from {src} (e.g. {sorted(joined.loc[absent, 'tg_id'])[:5]}). "
            f"The weighted subset must be a subset of the mesh it was filtered "
            f"from; check benchmark.source_kwargs.mesh_csv_path."
        )
    moved = ~(
        np.isclose(joined["tg_lat"], joined["tg_lat_mesh"])
        & np.isclose(joined["tg_lon"], joined["tg_lon_mesh"])
    )
    if moved.any():
        raise ValueError(
            f"{run.run_id}: {int(moved.sum())} scored TG(s) sit at a different "
            f"coordinate in {src} (e.g. {sorted(joined.loc[moved, 'tg_id'])[:5]}); "
            f"mesh_csv_path is not the mesh this run was filtered from."
        )


def site_n_scored(space: AnswerSpace) -> pd.Series:
    """Scored TGs per site, aligned to `space.sites`; `n_tgs` where unrecorded."""
    sites = space.sites
    col = SCORED_COL if SCORED_COL in sites.columns else "n_tgs"
    return sites[col].astype("int64")


def require_mesh_universe(run: RunPaths, space: AnswerSpace) -> None:
    """Refuse a weighted run's answer space built from its own post-filter TGs.

    Such a space predates `_mesh_tgs_for_weighted_run` and carries no
    `targets_provenance`. Everything drawn or counted over it -- cell labels,
    the outcome bars and map, the contest family -- is on a reduced seed set.
    """
    if run.source == WEIGHTED_SOURCE and "targets_provenance" not in space.meta:
        raise MissingArtifactError(
            f"{run.run_id} is a {WEIGHTED_SOURCE} run but its nside={space.nside} "
            f"answer space was built from the run's post-filter TGs, not the mesh. "
            f"Rerun `build-answer-space --run-id {run.run_id}`, then `classify`."
        )


def build_for_run(
    run: RunPaths,
    *,
    nsides: tuple[int, ...] = G.NSIDE_LADDER,
    analysis_root: Path | None = None,
) -> list[AnswerSpace]:
    """Build and write every requested rung for one run.

    **A traffic-weighted run is the exception**, unconditionally: its space is
    built over the pre-filter mesh (`_mesh_tgs_for_weighted_run`), and only its
    own evaluated TGs are scored against it by `classify`. Scored TGs are then a
    subset of `tgs.csv`, and the weighted arm's site and seed ids equal the
    mesh arm's, since both are assigned in sorted-coordinate order.
    """
    scored = load_tgs(run)
    tgs, mesh_src = scored, None
    if run.source == WEIGHTED_SOURCE:
        tgs, mesh_src = _mesh_tgs_for_weighted_run(run)
        _check_scored_in_mesh(run, scored, tgs, mesh_src)
    label = f"{run.run_id}/{run.source}/{run.setup}"
    spaces: list[AnswerSpace] = []
    for nside in sorted({G.validate_nside(n) for n in nsides}, reverse=True):
        provenance = None
        if mesh_src is not None:
            provenance = _weighted_provenance(scored, tgs, mesh_src, nside=nside, run_id=run.run_id)
        space = build_answer_space(
            tgs, nside=nside, run_id=run.run_id, source_label=label, provenance=provenance
        )
        placed = space.tgs[space.tgs["tg_id"].isin(set(scored["tg_id"]))]
        space.sites[SCORED_COL] = (
            placed.groupby("site_id").size()
            .reindex(space.sites["site_id"], fill_value=0)
            .to_numpy(dtype="int64")
        )
        if provenance is not None:
            space.meta["targets_provenance"].update(
                n_sites_scored_by_run=int(placed["site_id"].nunique()),
                n_seeds_scored_by_run=int(placed["tg_seed_id"].nunique()),
                n_seeds_with_no_scored_tg=int(space.n_seeds - placed["tg_seed_id"].nunique()),
            )
        space.write(run.answer_space_dir(nside, root=analysis_root))
        spaces.append(space)
    return spaces


def _weighted_provenance(
    scored: pd.DataFrame, mesh: pd.DataFrame, src: str, *, nside: int, run_id: str
) -> dict:
    """What the artifact records about its universe, with the counterfactual."""
    would_be = build_answer_space(scored, nside=nside, run_id=run_id)
    return {
        "targets_source": src,
        "why": (
            "traffic-weighted run: the answer space is the PRE-FILTER mesh, so this "
            "arm is comparable to the mesh arm. Only the run's own evaluated TGs "
            "are scored against it."
        ),
        "n_tgs_in_space": int(len(mesh)),
        "n_tgs_scored_by_run": int(len(scored)),
        "n_seeds_if_built_from_run_tgs": int(would_be.n_seeds),
    }
