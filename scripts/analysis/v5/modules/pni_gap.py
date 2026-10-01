"""How far a TG is from its operator's nearest PNI, against the S-P gap, clustered.

`figure_vp_distance_cdf` shows that the smallest-RTT VP is often not the
nearest VP: `gap = d_sp - d_geo` is zero for a fifth of TGs and reaches
4,000 km for others. This module asks whether the gap follows **where the
operator interconnects**: for each TG, the great-circle distance to the
nearest PNI in an operator-supplied list (`d_pni`), set against its gap. Then
it clusters the TGs on those two axes (Ward by default, k-means on request) and
writes the grouping to disk,
so the RTT box figure (`figure_pni_cluster_rtt`) reads the same clusters that
were drawn instead of recomputing them.

## The PNI list is an input, not a benchmark artifact

The benchmark has no idea where an operator peers. The list is passed in
(`--pni-csv`, columns `pni_id, pni_lat, pni_lon`) and validated here. Its
sha256 is in the manifest, and the output directory is keyed on its file stem,
so two PNI lists for one run cannot overwrite each other. On as01 the list is
marked `.approx`: facility coordinates, not the router's.

## The clustering unit is a scatter point, not a TG

~20 IP replicas share a site and share its VP geometry exactly, so they are
one observation repeated. A **point** is one distinct `(site, gap)` pair: one
per site, except where replicas at one site pick different smallest-RTT VPs
(as01 San Jose splits 15/5). Clustering runs on the points **unweighted**.
Weighting by TG count would let a 20-replica site outvote a split site's
5-replica point for no reason except the replica count. `n_tgs` travels with
each point for marker size, and every TG inherits its point's cluster.

## Clustering runs in the plotted geometry

The scatter draws both axes symlog: linear 0-100 km, log above. Both methods
are Euclidean, so they run on the **same** transform (`symlog_units`,
matplotlib's own `SymmetricalLogTransform` with the axes' parameters). Two
points that look close on the figure are close to the clustering, and a
cluster boundary can be read
straight off the picture. On raw km the 0-100 km block, which holds more than
half of as01's points, would be one dot beside a 4,000 km gap.

## Ward, not k-means, by default

`--method ward` (default) is agglomerative clustering with Ward linkage;
`--method kmeans` is k-means. Both minimise within-cluster variance, and on
each of as01, as02 and as03 alone they give identical partitions. Pooled over
the three, they differ on one point, and k-means is the one that gets it
wrong. A 7-replica AS03 point (123 km from a PNI, 79 km gap) sits 0.29 units
from the other 13 replicas of its own site, which are in the large
far-from-PNI cluster. k-means puts it in a 5-point cluster instead, because a
5-point cluster's centroid is dragged toward each of its members, this one
included. Its silhouette there is the only negative one of 80. Ward merges it
with its site first, and the pooled silhouette rises from 0.715 to 0.734.

Ward is deterministic, so it has no seed to check. The manifest instead
records the **other** method's partition at the same k and its adjusted Rand
index against the reported one, and names every point the two disagree on.

## Choosing k, and what the number is worth

`k` defaults to the silhouette argmax over `K_CANDIDATES`, computed with the
chosen method; ties go to the smaller k. On as01 that is 3 (0.77, against 0.69 at k=2 and 0.70 at k=6).
`--k` overrides. The whole silhouette curve is in the manifest, because with
this few points the runner-up is never far behind.

The number of points is what limits the result: as01 has 24 points at 20
sites, and two of its three clusters hold 3 and 7 points (2 and 5 sites).
Under k-means, refitting from `STABILITY_SEEDS` other seeds and comparing each
by adjusted Rand index checks that the partition does not depend on
initialisation. Neither method's checks say anything about sampling variance,
which with ~20 sites is large.

## Clusters are numbered by size, not by the algorithm

Raw cluster labels are arbitrary (k-means' change with the seed). Here clusters are
renumbered `1..k` by **site count descending** (ties: TG count descending,
then centroid gap and `d_pni` ascending), so `C1` is the cluster covering the
most sites and the numbering is stable across reruns. Sites rather than TGs,
because replicas are one observation repeated.

## The clusters are geometric, not mechanistic

On as01 both methods put Hillsboro (235 km from a PNI, 3,922 km gap) with the
far-from-PNI sites, not with SeaTac (19 km, 3,848 km), even though Hillsboro
and SeaTac share the same smallest-RTT VP and a >= 59 ms RTT floor. `d_pni` is
what separates them. The RTT box figure is there to show whether each cluster
also shares a latency regime. Do not read "same cluster" as "same cause".

## Privacy

Outputs carry no coordinate and no place name: TG ids, distances, RTTs,
cluster numbers. PNI ids name cities, so they stay out of the output as well.

## Pooling

`--layout pooled` merges several runs into one scatter and one clustering. Each
run is measured against **its own** PNI list first (`d_pni` against another
operator's PNIs means nothing), then the points are concatenated. Sites stay
keyed on `(run_id, lat, lon)`, so a facility two runs share is two points, and
no TG id may appear in two runs (`cross.guard_disjoint_tgs`). The manifest
records every run's CSV and PNI-list sha256, and `clusters_by_run` says which
runs each pooled cluster is made of.

Command: `plot-pni-gap`. Writes `<run>/pni-gap/<pni-stem>/` per run, and
`_cross/pni-gap/<n>-runs-<hash>/` pooled.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.scale import SymmetricalLogTransform
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score

from scripts.analysis.v5.modules import cross, edges
from scripts.analysis.v5.modules import sites as S
from scripts.analysis.v5.modules.answer_space import BENCHMARK_TG_COLUMNS
from scripts.analysis.v5.modules.figure_vp_proximity import MEASURE_COLUMNS, vp_distances
from scripts.analysis.v5.modules.geodesy import haversine_km
from scripts.analysis.v5.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    PNI_GAP_KIND,
    MissingArtifactError,
    RunPaths,
)

KIND = PNI_GAP_KIND

CLUSTERS_CSV = "pni_gap_clusters.csv"
POINTS_CSV = "pni_gap_points.csv"
MANIFEST_NAME = "pni_gap.manifest.json"

PNI_COLUMNS = ("pni_id", "pni_lat", "pni_lon")

#: Both axes: linear below `LINTHRESH_KM`, log above, capped at `AXIS_MAX_KM`.
#: `LINSCALE = 1` makes the linear block about as wide as one decade.
LINTHRESH_KM = 100.0
LINSCALE = 1.0
AXIS_MAX_KM = 4000.0

#: The k values scored by silhouette. Clipped to `n_points - 1` at run time,
#: since silhouette is undefined when every point is its own cluster.
K_CANDIDATES = tuple(range(2, 7))

#: Clustering methods. Ward is the default; see "Ward, not k-means, by default".
WARD = "ward"
KMEANS = "kmeans"
METHODS = (WARD, KMEANS)
DEFAULT_METHOD = WARD

#: k-means restarts. Cheap at ~20 points, and enough that the reported
#: partition is the objective's minimum rather than one initialisation's.
N_INIT = 50
SEED = 0
#: Refits from seeds `SEED + 1 ..`, each compared to the reported partition.
STABILITY_SEEDS = 20

#: Points at one site whose gaps differ by less than this are one point. 1 m:
#: replicas that picked the same smallest-RTT VP have bit-identical gaps, so
#: this guards float noise and merges nothing real.
GAP_DECIMALS = 3

CLUSTER_COL = "cluster"
POINT_COL = "point_id"
D_PNI = "d_pni_km"
D_GEO = "d_geo_km"
D_SP = "d_sp_km"
GAP = "gap_km"
SP_RTT = "sp_rtt_ms"


# -- inputs -------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pnis(path: Path | str) -> pd.DataFrame:
    """The operator's PNI list, validated. Returns `pni_id, pni_lat, pni_lon`.

    Strict because a bad row goes wrong without any error: a swapped lat/lon
    or a NaN coordinate changes every TG's nearest PNI and still draws a
    plausible scatter.
    """
    path = Path(path)
    if not path.exists():
        raise MissingArtifactError(f"--pni-csv {path} does not exist")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    missing = [c for c in PNI_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing PNI columns {missing}; need {list(PNI_COLUMNS)}")
    df = df[list(PNI_COLUMNS)].copy()
    if df.empty:
        raise ValueError(f"{path} lists no PNI")
    df["pni_id"] = df["pni_id"].astype(str)
    for c in ("pni_lat", "pni_lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = df[df.pni_lat.isna() | df.pni_lon.isna()]
    if len(bad):
        raise ValueError(f"{path}: {len(bad)} PNI rows have no usable coordinate, e.g. {bad.pni_id.iloc[0]!r}")
    out_of_range = df[(df.pni_lat.abs() > 90) | (df.pni_lon.abs() > 180)]
    if len(out_of_range):
        raise ValueError(
            f"{path}: {len(out_of_range)} PNI rows lie outside lat [-90, 90] / "
            f"lon [-180, 180], e.g. {out_of_range.pni_id.iloc[0]!r}. Swapped columns?"
        )
    dup = df.pni_id[df.pni_id.duplicated()]
    if len(dup):
        raise ValueError(f"{path}: duplicate pni_id {sorted(set(dup))[:5]}")
    return df.reset_index(drop=True)


def tg_coordinates(source_csv: Path) -> pd.DataFrame:
    """`tg_id, tg_lat, tg_lon`, one row per TG, read off the canonical CSV.

    Raises when a TG carries two coordinates. Replicas carry byte-identical
    coordinates, so two means the CSV is not the one `vp_distances` measured.
    """
    from scripts.libs.canonical.schema import load_canonical_csv

    df = load_canonical_csv(source_csv).rename(columns=BENCHMARK_TG_COLUMNS)
    per_tg = df.groupby("tg_id")[["tg_lat", "tg_lon"]].nunique()
    moving = per_tg[(per_tg.tg_lat > 1) | (per_tg.tg_lon > 1)]
    if len(moving):
        raise ValueError(
            f"{source_csv}: {len(moving)} TGs carry more than one coordinate, "
            f"e.g. {moving.index[0]!r}. A TG with two locations has no d_pni."
        )
    return df.groupby("tg_id", as_index=False)[["tg_lat", "tg_lon"]].first()


def nearest_pni_km(tg_lat, tg_lon, pnis: pd.DataFrame) -> np.ndarray:
    """Great-circle km from each TG to its nearest PNI (haversine, like `d_geo`)."""
    lat = np.asarray(tg_lat, dtype=float)[:, None]
    lon = np.asarray(tg_lon, dtype=float)[:, None]
    d = haversine_km(lat, lon, pnis.pni_lat.to_numpy()[None, :], pnis.pni_lon.to_numpy()[None, :])
    return d.min(axis=1)


def population(dist: pd.DataFrame, coords: pd.DataFrame, pnis: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    """One row per TG: both VP distances, the gap, `d_pni` and the S-P RTT.

    `dist` is `figure_vp_proximity.vp_distances`, the one definition of the
    two VP distances (and the one that already refuses `d_geo > d_sp`).
    Keeps `tg_lat`/`tg_lon` for the site key; `write` drops them.
    """
    pop = dist.merge(coords, on="tg_id", how="left", validate="1:1")
    if pop.tg_lat.isna().any():
        n = int(pop.tg_lat.isna().sum())
        raise ValueError(f"{n} TGs have VP distances but no coordinate; the two frames are not one CSV")
    pop = pd.DataFrame(
        {
            "tg_id": pop.tg_id,
            "tg_lat": pop.tg_lat,
            "tg_lon": pop.tg_lon,
            D_GEO: pop[MEASURE_COLUMNS["geo"]],
            D_SP: pop[MEASURE_COLUMNS["sping"]],
            SP_RTT: pop["sping_vp_rtt_ms"],
        }
    )
    pop[GAP] = (pop[D_SP] - pop[D_GEO]).clip(lower=0.0)  # vp_distances guarantees >= -1e-6
    pop[D_PNI] = nearest_pni_km(pop.tg_lat, pop.tg_lon, pnis)
    pop[S.SITE_KEY_COL] = S.site_key(pop, run_id=run_id)
    return pop.reset_index(drop=True)


def points(pop: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct `(site, gap)`, with `n_tgs`, and each TG's point id.

    Returns the points frame; also writes `POINT_COL` onto `pop` in place, so
    every TG can inherit its point's cluster. Point ids follow sorted
    `(site_key, gap)`, so they do not depend on row order.
    """
    key = pop[GAP].round(GAP_DECIMALS)
    grouped = (
        pop.assign(_gap_key=key)
        .groupby([S.SITE_KEY_COL, "_gap_key"], sort=True)
        .agg(run_id=("run_id", "first"), n_tgs=("tg_id", "size"),
             **{D_PNI: (D_PNI, "first"), GAP: (GAP, "median")})
        .reset_index()
    )
    grouped[POINT_COL] = np.arange(len(grouped))
    lookup = grouped.set_index([S.SITE_KEY_COL, "_gap_key"])[POINT_COL]
    pop[POINT_COL] = lookup.reindex(pd.MultiIndex.from_arrays([pop[S.SITE_KEY_COL], key])).to_numpy()
    return grouped.drop(columns="_gap_key")


# -- geometry and clustering --------------------------------------------------


def symlog_units(km) -> np.ndarray:
    """Distance on the drawn axis, in decades.

    matplotlib's own transform with the axes' parameters, divided by
    `LINTHRESH_KM`: 1 unit is one decade above 100 km, and 0-100 km spans
    `LINSCALE / (1 - 1/10)` units, the same proportion the figure draws.
    """
    t = SymmetricalLogTransform(base=10, linthresh=LINTHRESH_KM, linscale=LINSCALE)
    return t.transform(np.asarray(km, dtype=float)) / LINTHRESH_KM


def features(pts: pd.DataFrame) -> np.ndarray:
    return np.column_stack([symlog_units(pts[D_PNI]), symlog_units(pts[GAP])])


def _fit(X: np.ndarray, k: int, seed: int, method: str = DEFAULT_METHOD) -> np.ndarray:
    """Raw labels from `method`. `seed` matters to k-means only."""
    if method == WARD:
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    if method == KMEANS:
        with warnings.catch_warnings():
            # Raised only when k exceeds the distinct rows; `validate_k` refuses that first.
            warnings.simplefilter("error", category=UserWarning)
            return KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit_predict(X)
    raise ValueError(f"unknown clustering method {method!r}; expected one of {METHODS}")


def candidate_ks(X: np.ndarray) -> list[int]:
    n_distinct = len(np.unique(X, axis=0))
    return [k for k in K_CANDIDATES if k <= n_distinct - 1]


def validate_k(X: np.ndarray, k: int) -> None:
    n_distinct = len(np.unique(X, axis=0))
    if n_distinct < 3:
        raise ValueError(
            f"only {n_distinct} distinct scatter points; clustering needs at least 3. "
            f"A single-site run has nothing to cluster."
        )
    if not 2 <= k <= n_distinct - 1:
        raise ValueError(f"k={k} is out of range: need 2 <= k <= {n_distinct - 1} for {n_distinct} distinct points")


def silhouette_curve(X: np.ndarray, method: str = DEFAULT_METHOD) -> dict[int, float]:
    return {k: float(silhouette_score(X, _fit(X, k, SEED, method))) for k in candidate_ks(X)}


def choose_k(curve: dict[int, float]) -> int:
    """Silhouette argmax; ties go to the smaller k."""
    if not curve:
        raise ValueError("no k to choose from: fewer than 3 distinct points")
    best = max(curve.values())
    return min(k for k, s in curve.items() if np.isclose(s, best))


def ordered_labels(
    X: np.ndarray, raw: np.ndarray, site_keys: np.ndarray, n_tgs: np.ndarray
) -> np.ndarray:
    """Renumber `1..k` by site count **descending**, so `1` is the largest cluster.

    Ties go to TG count descending, then centroid gap ascending, then centroid
    `d_pni` ascending, so the numbering is total and does not depend on the
    k-means seed. A site whose replicas split across two clusters counts in
    both, as it does in `cluster_summary`.
    """
    ids = np.unique(raw)
    keys = []
    for i in ids:
        member = raw == i
        cent = X[member].mean(axis=0)
        keys.append((-len(set(site_keys[member])), -int(n_tgs[member].sum()), cent[1], cent[0]))
    order = sorted(range(len(ids)), key=lambda j: keys[j])
    rank = {ids[j]: r + 1 for r, j in enumerate(order)}
    return np.array([rank[v] for v in raw])


def stability(X: np.ndarray, k: int, reference: np.ndarray) -> dict:
    """k-means only: ARI of refits from other seeds against the reported partition."""
    aris = [
        float(adjusted_rand_score(reference, _fit(X, k, SEED + s, KMEANS)))
        for s in range(1, STABILITY_SEEDS + 1)
    ]
    return {
        "n_refits": len(aris),
        "min_ari": min(aris),
        "share_identical": float(np.mean(np.isclose(aris, 1.0))),
        "note": (
            "Refits from other k-means seeds, compared by adjusted Rand index. "
            "Checks initialisation only, not sampling: with ~20 sites the "
            "partition could change with the site set."
        ),
    }


def method_agreement(X: np.ndarray, k: int, labels: np.ndarray, method: str,
                     point_ids: np.ndarray) -> dict:
    """The other method at the same k: its ARI against `labels`, and the points it moves.

    A point "moves" when the other method puts it in a cluster whose majority
    is not its own reported cluster's majority. Names the points by id so a
    disagreement can be looked up rather than just counted.
    """
    other = KMEANS if method == WARD else WARD
    alt = _fit(X, k, SEED, other)
    majority = pd.crosstab(labels, alt).idxmax(axis=1)   # reported cluster -> alt label
    moved = [int(p) for p, lab, a in zip(point_ids, labels, alt) if majority[lab] != a]
    return {
        "other_method": other,
        "ari": round(float(adjusted_rand_score(labels, alt)), 4),
        "points_moved": moved,
    }


def cluster(
    pts: pd.DataFrame, *, k: int | None = None, method: str = DEFAULT_METHOD
) -> tuple[pd.DataFrame, dict]:
    """Assign `CLUSTER_COL` to every point. Returns the points and the k-selection record."""
    if method not in METHODS:
        raise ValueError(f"unknown clustering method {method!r}; expected one of {METHODS}")
    X = features(pts)
    validate_k(X, 2)  # refuses fewer than 3 distinct points before anything is fitted
    curve = silhouette_curve(X, method)
    chosen = choose_k(curve) if k is None else int(k)
    validate_k(X, chosen)
    labels = ordered_labels(X, _fit(X, chosen, SEED, method),
                            pts[S.SITE_KEY_COL].to_numpy(), pts["n_tgs"].to_numpy())
    out = pts.assign(**{CLUSTER_COL: labels})
    record = {
        "method": method,
        "k": chosen,
        "k_source": "silhouette argmax" if k is None else "--k",
        "silhouette": {str(kk): round(v, 4) for kk, v in curve.items()},
        "silhouette_at_k": round(float(silhouette_score(X, labels)), 4),
        "n_negative_silhouette_points": int((silhouette_samples(X, labels) < 0).sum()),
        "agreement": method_agreement(X, chosen, labels, method, pts[POINT_COL].to_numpy()),
    }
    if method == KMEANS:
        record["stability"] = stability(X, chosen, labels)
        record["n_init"] = N_INIT
        record["seed"] = SEED
    return out, record


def assign(pop: pd.DataFrame, pts: pd.DataFrame) -> pd.DataFrame:
    """Every TG inherits its point's cluster. Refuses a TG left without one."""
    out = pop.merge(pts[[POINT_COL, CLUSTER_COL]], on=POINT_COL, how="left", validate="m:1")
    if out[CLUSTER_COL].isna().any():
        raise ValueError(f"{int(out[CLUSTER_COL].isna().sum())} TGs have no cluster")
    out[CLUSTER_COL] = out[CLUSTER_COL].astype(int)
    return out


def cluster_summary(tgs: pd.DataFrame, pts: pd.DataFrame) -> pd.DataFrame:
    """Per cluster: counts and the range of each axis. No location."""
    by_pt = pts.groupby(CLUSTER_COL).agg(n_points=(POINT_COL, "size"))
    by_tg = tgs.groupby(CLUSTER_COL).agg(
        n_tgs=("tg_id", "size"),
        n_sites=(S.SITE_KEY_COL, "nunique"),
        d_pni_min_km=(D_PNI, "min"),
        d_pni_max_km=(D_PNI, "max"),
        gap_min_km=(GAP, "min"),
        gap_max_km=(GAP, "max"),
        sp_rtt_min_ms=(SP_RTT, "min"),
        sp_rtt_max_ms=(SP_RTT, "max"),
    )
    return by_pt.join(by_tg).reset_index()


# -- the whole step -----------------------------------------------------------

PER_RUN = "per-run"
POOLED = "pooled"
LAYOUTS = (PER_RUN, POOLED)


def output_dir(run_id: str, pni_csv: Path, *, analysis_root: Path | None = None) -> Path:
    """Per-run: `<root>/<run_id>/pni-gap/<pni-stem>/`, created."""
    stem = Path(pni_csv).name.removesuffix(".csv")
    out = (analysis_root or DEFAULT_ANALYSIS_ROOT) / run_id / KIND / stem
    out.mkdir(parents=True, exist_ok=True)
    return out


def pooled_output_dir(run_ids: list[str], *, analysis_root: Path | None = None) -> Path:
    """Pooled: `_cross/pni-gap/<n>-runs-<hash>/`, created.

    Keyed on the run set only. Each run's PNI list is identified by its
    sha256 in the manifest, which the RTT figure checks per run.
    """
    return cross.cross_dir(run_ids, analysis_root=analysis_root, kind=KIND)


def run_population(
    run: RunPaths, pni_csv: Path, *, source_csv: Path | None = None
) -> tuple[pd.DataFrame, dict]:
    """One run's per-TG frame against **its own** PNI list, and its provenance."""
    csv = edges.resolve_source_csv(run, source_csv)
    pnis = load_pnis(pni_csv)
    pop = population(vp_distances(run, source_csv=csv), tg_coordinates(csv), pnis, run_id=run.run_id)
    pop.insert(0, "run_id", run.run_id)
    record = {
        "run_id": run.run_id,
        "source_csv": str(csv),
        "source_csv_sha256": sha256_file(Path(csv)),
        "pni_csv": str(pni_csv),
        "pni_csv_sha256": sha256_file(Path(pni_csv)),
        "n_pnis": int(len(pnis)),
        "n_tgs": int(len(pop)),
        "n_sites": int(pop[S.SITE_KEY_COL].nunique()),
    }
    return pop, record


def compute_runs(
    runs: list[RunPaths],
    pni_csvs: dict[str, Path],
    *,
    layout: str = PER_RUN,
    k: int | None = None,
    method: str = DEFAULT_METHOD,
    source_csvs: dict[str, Path] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """`(tgs, points, meta)` over one run, or several pooled. Writes nothing.

    Pooling concatenates the runs' TGs **after** each has been measured against
    its own PNI list: `d_pni` against another operator's PNIs means nothing.
    Sites are keyed on `(run_id, lat, lon)`, so a facility two runs share is
    two points. The clustering then runs once over the pooled points.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout {layout!r}; expected one of {LAYOUTS}")
    if not runs:
        raise ValueError("pass at least one run")
    if layout == PER_RUN and len(runs) != 1:
        raise ValueError(f"layout {PER_RUN!r} takes one run; got {len(runs)}")
    missing = [r.run_id for r in runs if r.run_id not in pni_csvs]
    if missing:
        raise ValueError(f"no PNI list for {missing}: every pooled run needs its own")

    pops, records = [], []
    for run in runs:
        pop, record = run_population(run, pni_csvs[run.run_id],
                                     source_csv=(source_csvs or {}).get(run.run_id))
        pops.append(pop)
        records.append(record)
    cross.guard_disjoint_tgs(
        {r["run_id"]: set(p.tg_id) for r, p in zip(records, pops)},
        remedy="Use --layout per-run, which keeps each run on its own figure.",
    )
    pop = pd.concat(pops, ignore_index=True)
    pts = points(pop)
    pts, clustering = cluster(pts, k=k, method=method)
    tgs = assign(pop, pts)
    beyond = int(((pts[D_PNI] > AXIS_MAX_KM) | (pts[GAP] > AXIS_MAX_KM)).sum())
    meta = {
        "layout": layout,
        "run_ids": [r.run_id for r in runs],
        "runs": records,
        "n_tgs": int(len(tgs)),
        "n_sites": int(tgs[S.SITE_KEY_COL].nunique()),
        "n_points": int(len(pts)),
        "clustering": clustering,
        "axes": {
            "scale": "symlog",
            "linthresh_km": LINTHRESH_KM,
            "linscale": LINSCALE,
            "max_km": AXIS_MAX_KM,
            "n_points_beyond_max": beyond,
            "note": (
                "The clustering runs on these same symlog coordinates, so distances "
                "on the figure are distances to it. Points beyond max_km are "
                "clustered but not drawn; the panel says so when any exist."
            ),
        },
    }
    return tgs, pts, meta


def compute(
    run: RunPaths,
    pni_csv: Path,
    *,
    k: int | None = None,
    method: str = DEFAULT_METHOD,
    source_csv: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """`compute_runs` for one run, per-run layout."""
    return compute_runs(
        [run], {run.run_id: pni_csv}, layout=PER_RUN, k=k, method=method,
        source_csvs={run.run_id: source_csv} if source_csv is not None else None,
    )


def clusters_by_run(tgs: pd.DataFrame) -> list[dict]:
    """Per `(cluster, run)`: TG and site counts. Says which runs a pooled cluster is made of."""
    out = (
        tgs.groupby([CLUSTER_COL, "run_id"])
        .agg(n_tgs=("tg_id", "size"), n_sites=(S.SITE_KEY_COL, "nunique"))
        .reset_index()
    )
    return json.loads(out.to_json(orient="records"))


def write(tgs: pd.DataFrame, pts: pd.DataFrame, meta: dict, out_dir: Path) -> dict[str, Path]:
    """The clusters CSV the RTT figure consumes, the points CSV, the manifest."""
    tg_cols = ["run_id", "tg_id", POINT_COL, CLUSTER_COL, D_PNI, D_GEO, D_SP, GAP, SP_RTT]
    pt_cols = ["run_id", POINT_COL, CLUSTER_COL, "n_tgs", D_PNI, GAP]
    paths = {
        "clusters": out_dir / CLUSTERS_CSV,
        "points": out_dir / POINTS_CSV,
        "manifest": out_dir / MANIFEST_NAME,
    }
    tgs.sort_values([CLUSTER_COL, "run_id", "tg_id"])[tg_cols].to_csv(paths["clusters"], index=False)
    pts.sort_values([CLUSTER_COL, POINT_COL])[pt_cols].to_csv(paths["points"], index=False)
    summary = cluster_summary(tgs, pts)
    body = {
        **meta,
        "clusters_csv": CLUSTERS_CSV,
        "points_csv": POINTS_CSV,
        "clusters": json.loads(summary.round(3).to_json(orient="records")),
        "clusters_by_run": clusters_by_run(tgs),
        "unit": (
            "The clustering groups distinct (site, gap) points, unweighted; every TG "
            "inherits its point's cluster. Clusters are numbered by site count "
            "descending (ties: TGs descending, then centroid gap ascending). "
            "Sites are keyed on (run_id, lat, lon)."
        ),
    }
    paths["manifest"].write_text(json.dumps(body, indent=2))
    return paths


# -- what the RTT figure reads back -------------------------------------------


def read_clusters(out_dir: Path, *, run_ids: list[str]) -> tuple[pd.DataFrame, dict]:
    """The clusters CSV and its manifest, checked against the runs asking for them."""
    csv, manifest = out_dir / CLUSTERS_CSV, out_dir / MANIFEST_NAME
    if not csv.exists() or not manifest.exists():
        flags = " ".join(f"--run-id {r}" for r in run_ids)
        raise MissingArtifactError(f"{csv} missing; run `plot-pni-gap {flags}` first")
    meta = json.loads(manifest.read_text())
    if sorted(meta.get("run_ids") or []) != sorted(run_ids):
        raise ValueError(f"{manifest} was written for runs {meta.get('run_ids')!r}, not {sorted(run_ids)!r}")
    tgs = pd.read_csv(csv, dtype={"tg_id": str, "run_id": str})
    if tgs.duplicated(["run_id", "tg_id"]).any():
        raise ValueError(f"{csv} lists a TG twice")
    if len(tgs) != meta.get("n_tgs"):
        raise ValueError(f"{csv} has {len(tgs)} TGs; its manifest says {meta.get('n_tgs')}")
    return tgs, meta
