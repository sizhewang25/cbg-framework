"""Two methods, site by site: who put the prediction in the right cell.

The shared substrate under the four SPO-vs-OCT-H figures. Each of them asks a
different question of the same table, so the table is built once here and the
figures only draw it.

## A contest is per **site**, not per target

These meshes put ~20 IP replicas at one operator facility with byte-identical
coordinates. Twenty replicas of one site are one observation repeated, not
twenty, so every statistic in this family is per site and a site's *margin*
(19 of 20 against 20 of 20) is deliberately not a magnitude anywhere -- it is
a tie-break and nothing more. Effective n on the three meshes is 65 sites, not
1,269 targets.

## The site key is `(run_id, site_id)`

`sites.site_ids` assigns a dense id per run, so `site_id` 3 is a different
facility in `as01` and in `as03`. Keying a pooled frame on `site_id` alone
silently merges the three meshes; during the exploration that produced these
claims it reported 20 sites where there are 34. `site_contest` therefore takes
one run at a time and stamps `run_id` on every row, and every caller that
pools must group on both.

## Two traps, both hit before

1. **`solved_mask` is not optional.** A FALLBACK row carries the shortest-ping
   baseline's coordinate, not the method's, and `classify` labels it on that
   coordinate. Counting those rows credits Vanilla with 270 correct on `as01`
   where it earned 163 -- and all 107 of them are cell-correct, because
   Vanilla refuses precisely on the VP-adjacent targets where returning the VP
   is trivially right.
2. **No threshold decides a category.** A per-site majority rule
   (`correct_frac >= 0.5`) reads Octant-Hull's 9 of 20 at Los Angeles as a
   loss and paints the site an exclusive Spotter win -- on a site where
   Octant-Hull was the *more accurate* method. `categorize` compares the two
   counts directly and has no threshold in it at all.

## McNemar is over sites, and it reports its own floor

The test statistic is the number of sites each method wins outright; ties and
double failures are concordant and carry no information about direction. With
nine discordant sites the smallest two-sided p an exact test can return is
`2 * 0.5**9 = 0.004`, and with two it is 0.5 -- so a mesh that cannot resolve
anything still returns a number that looks like a result. `mcnemar` reports
that floor beside the p-value so an underpowered mesh says so out loud.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import cross
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules.map_answer_space import load_rung
from scripts.analysis.v5.modules.geodesy import elementwise_km, spherical_centroid
from scripts.analysis.v5.modules.methods import method_label
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths
from scripts.analysis.v5.modules.status import solved_mask

#: Which rung's answer space and `*_tgs.parquet` the contest reads. Off the
#: ladder rather than spelled, so a second rung cannot desync seeds from
#: labels.
SOURCE_NSIDE = G.NSIDE_LADDER[0]

#: The pair the paper argues about. A default for the figures, not a
#: constraint -- `load` takes any two scored methods.
DEFAULT_METHOD_A = "spotter_cbg"
DEFAULT_METHOD_B = "octant_cbg_hull"

#: `classify.CELL_LABELS`' good outcome, named once here so the conjunction
#: below reads as a definition rather than as a string comparison.
CORRECT = "correct"

#: The four ways a site can come out. `a` and `b` are positional -- the caller
#: names the methods -- so the same vocabulary serves any pair.
TIED = "tied"
A_WINS = "a_wins"
B_WINS = "b_wins"
NEITHER = "neither"

#: Fixed order: the two decisive outcomes in the middle, flanked by the two
#: that carry no direction. Legends and count tables follow it.
CATEGORIES: tuple[str, ...] = (TIED, A_WINS, B_WINS, NEITHER)

#: The key a pooled per-site frame must group on. See the module docstring.
SITE_KEY: tuple[str, str] = ("run_id", "site_id")

#: Columns `site_contest` returns, in order.
CONTEST_COLUMNS: tuple[str, ...] = (
    "run_id", "dataset", "site_id", "tg_lat", "tg_lon", "tg_seed_id",
    "method_a", "method_b", "label_a", "label_b",
    "k_a", "n_a", "k_b", "n_b", "n_tgs", "category",
)

#: Columns a scored `*_tgs.parquet` must carry for a contest. Read explicitly
#: so a parquet that has drifted fails here rather than three frames later.
TG_COLUMNS: tuple[str, ...] = (
    "tg_id", "tg_lat", "tg_lon", "site_id", "tg_seed_id", "cell_label", "status",
)


def correct_mask(frame: pd.DataFrame) -> np.ndarray:
    """Rows this method placed in the target's own cell, **solved only**.

    The one definition of "correct" this family uses. `cell_label` alone is
    not it: `classify` labels a FALLBACK row on the baseline's coordinate, so
    the mask is the conjunction. See the module docstring.
    """
    return ((frame["cell_label"] == CORRECT) & solved_mask(frame)).to_numpy()


def categorize(k_a, k_b) -> np.ndarray:
    """Per-site category from the two correct counts. No threshold, anywhere.

    `neither` when both are zero -- a site neither method reached says nothing
    about which is better, and folding it into `tied` would inflate the ties.
    Otherwise the larger count wins and an exact equality is `tied`.
    """
    a = np.asarray(k_a, dtype=np.int64)
    b = np.asarray(k_b, dtype=np.int64)
    out = np.full(a.shape, TIED, dtype=object)
    out[(a == 0) & (b == 0)] = NEITHER
    out[a > b] = A_WINS
    out[b > a] = B_WINS
    return out


def _per_site(frame: pd.DataFrame, run_id: str, method: str) -> pd.DataFrame:
    """One row per site: its coordinate, its seed, and the correct count."""
    missing = [c for c in TG_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{method} in {run_id} is missing {missing}; re-run `classify` "
            f"for this run"
        )
    dupes = frame["tg_id"].duplicated()
    if dupes.any():
        raise ValueError(
            f"{method} in {run_id} repeats {int(dupes.sum())} tg_id(s), e.g. "
            f"{sorted(frame.loc[dupes, 'tg_id'])[:3]}; each would be counted twice"
        )
    work = frame[list(TG_COLUMNS)].copy()
    work["correct"] = correct_mask(frame)
    by_site = work.groupby("site_id", sort=True)
    out = pd.DataFrame(
        {
            # Replicas of a site carry byte-identical coordinates (that is what
            # `sites.site_ids` keys on), so `first` is exact, not a sample.
            "tg_lat": by_site["tg_lat"].first(),
            "tg_lon": by_site["tg_lon"].first(),
            "tg_seed_id": by_site["tg_seed_id"].first().astype(int),
            "k": by_site["correct"].sum().astype(int),
            "n": by_site.size().astype(int),
        }
    )
    return out.rename_axis("site_id").reset_index()


def site_contest(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    run_id: str,
    method_a: str,
    method_b: str,
) -> pd.DataFrame:
    """One row per site of `run_id`: both correct counts and the category.

    The two frames are the runs' scored `*_tgs.parquet` for the two methods.
    They must cover the same TG population -- a method scored over a different
    roster would put the two counts on different denominators, and the
    per-site label this table feeds prints both beside one total.
    """
    ids_a, ids_b = set(frame_a["tg_id"]), set(frame_b["tg_id"])
    if ids_a != ids_b:
        only_a, only_b = sorted(ids_a - ids_b), sorted(ids_b - ids_a)
        raise ValueError(
            f"{method_a} and {method_b} do not score the same TGs in {run_id}: "
            f"{len(only_a)} only in {method_a} (e.g. {only_a[:3]}), "
            f"{len(only_b)} only in {method_b} (e.g. {only_b[:3]})"
        )
    a = _per_site(frame_a, run_id, method_a)
    b = _per_site(frame_b, run_id, method_b)
    out = a.merge(b, on="site_id", suffixes=("_a", "_b"), validate="one_to_one")
    for col in ("tg_lat", "tg_lon", "tg_seed_id"):
        if not out[f"{col}_a"].equals(out[f"{col}_b"]):
            raise ValueError(
                f"{method_a} and {method_b} disagree on {col} in {run_id}; "
                f"they were scored against different answer spaces"
            )
        out[col] = out[f"{col}_a"]
    out["run_id"] = run_id
    out["dataset"] = cross.short_dataset(run_id)
    out["method_a"], out["method_b"] = method_a, method_b
    out["label_a"] = method_label(method_a)
    out["label_b"] = method_label(method_b)
    # The two are equal by the roster guard above; both are kept because a
    # reader of the CSV should not have to take that on trust.
    out["n_tgs"] = out["n_a"]
    out["category"] = categorize(out["k_a"], out["k_b"])
    return out.reindex(columns=list(CONTEST_COLUMNS))


def mcnemar(b: int, c: int) -> dict:
    """Exact two-sided sign test on the discordant sites, with its own floor.

    `b` and `c` are the site counts the two methods win outright. Concordant
    sites -- ties and double failures -- carry no direction and are excluded
    by construction, which is why the discordant total is reported: it is the
    real sample size of the test.

    `p_floor` is the smallest p this many discordant sites can produce. Read
    the two together: `p = 0.5` off two discordant sites is not a null result,
    it is the whole range the test had. With no discordant sites at all there
    is no test, and `p` is 1.0 by the degenerate reading -- no evidence of a
    difference -- with a floor of 1.0 saying exactly that.
    """
    from scipy.stats import binomtest

    b, c = int(b), int(c)
    if b < 0 or c < 0:
        raise ValueError(f"win counts must be non-negative, got b={b}, c={c}")
    n = b + c
    p = 1.0 if n == 0 else float(binomtest(b, n, 0.5).pvalue)
    return {
        "mcnemar_b": b,
        "mcnemar_c": c,
        "mcnemar_n_discordant": n,
        "mcnemar_p": p,
        "mcnemar_p_floor": min(1.0, 2.0 * 0.5**n),
    }


def contest_counts(table: pd.DataFrame, **extra) -> dict:
    """Category counts and the McNemar result over a set of contest rows.

    Takes whatever slice the caller wants summarised -- one mesh's rows for a
    panel, every mesh's for the pooled line -- so pooling is the caller's
    decision and this function has no opinion on it.
    """
    seen = table["category"].value_counts()
    counts = {f"n_{k}": int(seen.get(k, 0)) for k in CATEGORIES}
    return {
        **extra,
        "n_sites": int(len(table)),
        "n_tgs": int(table["n_tgs"].sum()),
        **counts,
        "k_a": int(table["k_a"].sum()),
        "k_b": int(table["k_b"].sum()),
        **mcnemar(counts[f"n_{A_WINS}"], counts[f"n_{B_WINS}"]),
    }


def seed_cloud_centre(seed_lats, seed_lons) -> tuple[float, float]:
    """The seed cloud's centre as `(lat, lon)`, on the sphere.

    `geodesy.spherical_centroid`, which is what `seeds` already uses to place
    a seed over the sites it groups. One notion of centre in the package, so
    the centre of the seeds is found the same way each seed was.
    """
    return spherical_centroid(seed_lats, seed_lons)


def seed_cloud_centroid_km(lats, lons, seed_lats, seed_lons) -> np.ndarray:
    """Great-circle distance from each point to the **seed cloud's** centre.

    Anchored on the seeds, not on the sites: the seeds are the answer space,
    and a site set that happens to cluster would otherwise move the origin it
    is being measured against.

    On the sphere, not in `projection.PROJECTED_CRS`. The cell partition is
    defined by great-circle nearest seed and `classify` uses no projection at
    all; EPSG:5070 exists so `cells` can *draw* the partition, and measuring
    peripherality there would import a rendering concern into a number. It
    also costs the projection's 1-2% distance distortion for nothing: against
    the spherical answer the planar one is off by a median 5 km and at most 22
    over these meshes, and the two rank the 65 sites identically to Spearman
    0.9995. The claims do not turn on it; the consistency does.

    Nearly collinear with "coastal" on a CONUS answer space. It does not
    separate the two and must not be quoted as if it did.
    """
    clat, clon = seed_cloud_centre(seed_lats, seed_lons)
    lats = np.asarray(lats, dtype=float).ravel()
    return elementwise_km(
        lats, lons, np.full(lats.size, clat), np.full(lats.size, clon)
    )


@dataclass(frozen=True)
class ContestData:
    """The scored frames and seeds for one method pair over several meshes."""

    run_ids: list[str]
    method_a: str
    method_b: str
    frames: dict[tuple[str, str], pd.DataFrame]
    seeds: dict[str, pd.DataFrame]
    nside: int


def load(
    runs: list[RunPaths],
    *,
    method_a: str,
    method_b: str,
    nside: int = SOURCE_NSIDE,
    analysis_root: Path | None = None,
) -> ContestData:
    """Both methods' `*_tgs.parquet` and each run's seeds.

    Both methods must be scored in every run. A mesh carrying one of them
    would contribute a panel of half-contests, and the pooled counts the
    manifest reports would be over a population that changes per panel.
    """
    if method_a == method_b:
        raise ValueError(f"a contest needs two methods, got {method_a!r} twice")
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    seeds: dict[str, pd.DataFrame] = {}
    for run in runs:
        seeds[run.run_id] = load_rung(run, nside, analysis_root=analysis_root).seeds
        for m in (method_a, method_b):
            path = run.classify_dir(nside, root=analysis_root) / C.TGS_PARQUET.format(method=m)
            if not path.exists():
                have = scored_methods(run, nside, analysis_root=analysis_root)
                raise MissingArtifactError(
                    f"{m} is not scored in {run.run_id} at nside={nside} "
                    f"({path} missing; have {have}). Run "
                    f"`classify --run-id {run.run_id}` first."
                )
            frames[(run.run_id, m)] = pd.read_parquet(path, columns=list(TG_COLUMNS))
    return ContestData(
        run_ids=[r.run_id for r in runs],
        method_a=method_a,
        method_b=method_b,
        frames=frames,
        seeds=seeds,
        nside=int(nside),
    )


# -- the numbers ----------------------------------------------------------


def scored_methods(
    run: RunPaths, nside: int = SOURCE_NSIDE, *, analysis_root: Path | None = None
) -> list[str]:
    """Methods with a `*_tgs.parquet` at this rung. Read from disk, not a config."""
    suffix = C.TGS_PARQUET.format(method="")
    d = run.classify_dir(nside, root=analysis_root)
    return sorted(p.name[: -len(suffix)] for p in d.glob("*" + suffix))


def contest_table(data: ContestData) -> pd.DataFrame:
    """One row per `(run_id, site_id)` over every run, with `centroid_km`.

    The table all four figures start from. Runs appear in the order they were
    given, sites in `site_id` order within a run.
    """
    parts = []
    for run_id in data.run_ids:
        rows = site_contest(
            data.frames[(run_id, data.method_a)],
            data.frames[(run_id, data.method_b)],
            run_id=run_id,
            method_a=data.method_a,
            method_b=data.method_b,
        )
        seeds = data.seeds[run_id]
        # Per run, because each mesh has its own seed cloud and so its own
        # origin. Pooling the distances afterwards is the caller's decision.
        rows["centroid_km"] = seed_cloud_centroid_km(
            rows["tg_lat"], rows["tg_lon"], seeds["seed_lat"], seeds["seed_lon"]
        )
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)
