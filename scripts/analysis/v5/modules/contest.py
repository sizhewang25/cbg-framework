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
    "k_a", "n_a", "n_solved_a", "offset_sd_a",
    "k_b", "n_b", "n_solved_b", "offset_sd_b",
    "n_tgs", "category",
)

#: Columns a scored `*_tgs.parquet` must carry for a contest. Read explicitly
#: so a parquet that has drifted fails here rather than three frames later.
TG_COLUMNS: tuple[str, ...] = (
    "tg_id", "tg_lat", "tg_lon", "site_id", "tg_seed_id", "cell_label", "status",
    C.GRID_OFFSET,
)

#: The column `offset_spread` writes, in grid steps.
OFFSET_SD = "offset_sd"

#: `both_correct_targets`' columns, in order. Per target, not per site: this
#: is the one frame in the module that is not a site summary.
PAIRED_COLUMNS: tuple[str, ...] = (
    "run_id", "dataset", "tg_id", "site_id", "tg_seed_id",
    "method_a", "method_b", "label_a", "label_b",
    "offset_a", "offset_b", "diff", "closer",
)

#: Who was nearer on a target. `equal` is its own answer, not a rounding of
#: either: on 9.2% of the both-correct cohort the two land the same distance
#: out, and folding those into a winner would invent a direction.
CLOSER_A = "a"
CLOSER_B = "b"
CLOSER_EQUAL = "equal"

#: Degrees of freedom. A site's replicas are a sample of the addresses at a
#: facility, not the population of them, so the spread is estimated rather
#: than described. It needs two of them; a site with one solved row gets NaN
#: rather than 0, which would read as perfect agreement.
OFFSET_SD_DDOF = 1


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
    work["solved"] = solved_mask(frame).to_numpy()
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
            "n_solved": by_site["solved"].sum().astype(int),
        }
    )
    out[OFFSET_SD] = offset_spread(work[work["solved"]])
    return out.rename_axis("site_id").reset_index()


def offset_spread(solved: pd.DataFrame) -> pd.Series:
    """Per site: the standard deviation of `pred_dist_to_tg_grid`, in grids.

    How much a site's ~20 replicas disagree about *how far out* the answer is.
    Solved rows only -- a FALLBACK row carries the shortest-ping baseline's
    coordinate, so its offset is the baseline's. A site with fewer than two of
    them is NaN rather than 0.

    ## What it does not measure, deliberately

    This is the spread of the error **magnitude**, not of the answers. Two
    replicas five grids out in opposite directions have identical magnitudes
    and read as perfect agreement here, though they are two different answers.

    That is a known and accepted limitation, not an oversight, and it was
    measured before it was accepted: against the spread of the prediction
    cloud itself (RMS grid distance to the site's prediction centroid) the two
    rank the 65 sites at Spearman 0.90 and reach the same conclusion, and they
    disagree on 11 sites that read as perfectly consistent here while their
    predictions were genuinely up to two grids apart. The simpler statistic
    was chosen on those terms. `test_opposite_answers_read_as_agreement` pins
    the behaviour so nobody quietly "fixes" it.

    ## In grid steps, not kilometres

    The grid is the unit the rest of the evaluation is in, and a spread quoted
    in kilometres invites comparison against an error distance this figure is
    deliberately not about. One grid is `grid.grid_km(nside)`.
    """
    if not len(solved):
        return pd.Series(dtype=float, name=OFFSET_SD)
    out = solved.groupby("site_id")[C.GRID_OFFSET].std(ddof=OFFSET_SD_DDOF)
    return out.rename(OFFSET_SD)


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
    is being measured against. `contest_table` measures a *seed* against it
    for the same reason -- both ends of the measurement are then points of the
    answer space, and the number says where a serving region sits rather than
    where a target happens to sit inside one.

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


def both_correct_targets(data: "ContestData") -> pd.DataFrame:
    """One row per target **both** methods place in the correct cell.

    The cohort for the error comparison, and the only per-target frame here.
    Restricting to targets both methods got right is what makes the
    comparison fair: a method that answers fewer targets is not thereby more
    accurate on the ones it does answer, and comparing over each method's own
    correct set would score them on different populations.

    `diff = offset_a - offset_b`, in grid steps. **The sign is fixed**:
    positive means method `a` is further from the target, so on the paper's
    pair the mass sits to the *right* of zero. An earlier expectation had it
    leaning left; the data says otherwise and the convention does not move to
    suit it.
    """
    parts = []
    for run_id in data.run_ids:
        a = data.frames[(run_id, data.method_a)]
        b = data.frames[(run_id, data.method_b)]
        ids_a, ids_b = set(a["tg_id"]), set(b["tg_id"])
        if ids_a != ids_b:
            raise ValueError(
                f"{data.method_a} and {data.method_b} do not score the same TGs "
                f"in {run_id}: {len(ids_a ^ ids_b)} differ"
            )
        a = a.set_index("tg_id").sort_index()
        b = b.set_index("tg_id").sort_index()
        keep = a.index[
            correct_mask(a.reset_index()) & correct_mask(b.reset_index())
        ]
        offset_a = a.loc[keep, C.GRID_OFFSET].to_numpy()
        offset_b = b.loc[keep, C.GRID_OFFSET].to_numpy()
        diff = offset_a - offset_b
        parts.append(
            pd.DataFrame(
                {
                    "run_id": run_id,
                    "dataset": cross.short_dataset(run_id),
                    "tg_id": keep,
                    "site_id": a.loc[keep, "site_id"].to_numpy(),
                    "tg_seed_id": a.loc[keep, "tg_seed_id"].to_numpy(),
                    "method_a": data.method_a,
                    "method_b": data.method_b,
                    "label_a": method_label(data.method_a),
                    "label_b": method_label(data.method_b),
                    "offset_a": offset_a,
                    "offset_b": offset_b,
                    "diff": diff,
                    "closer": np.where(
                        diff < 0, CLOSER_A, np.where(diff > 0, CLOSER_B, CLOSER_EQUAL)
                    ),
                }
            )
        )
    out = pd.concat(parts, ignore_index=True)
    return out.reindex(columns=list(PAIRED_COLUMNS))


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

    `centroid_km` is the site's **seed** against the spherical centroid of all
    that run's seeds. Sites sharing a seed therefore share a value -- as01 has
    18 seeds over 20 sites -- which is the point: it measures the serving
    region, not the target inside it.

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
        # The site's *seed* against the centroid of all seeds, not the site
        # itself: both ends of the measurement are then points of the answer
        # space. Complete linkage caps a seed's group at one `grid_km`, so the
        # two differ by at most ~51 km, but the definition should not need
        # that bound to be coherent.
        #
        # Per run, because each mesh has its own seed cloud and so its own
        # origin. Pooling the distances afterwards is the caller's decision.
        mine = seeds.set_index("seed_id").loc[rows["tg_seed_id"].to_numpy()]
        rows["centroid_km"] = seed_cloud_centroid_km(
            mine["seed_lat"], mine["seed_lon"], seeds["seed_lat"], seeds["seed_lon"]
        )
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)
