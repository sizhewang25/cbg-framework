"""Whose easy TGs are whose, and what the others did on them.

`figure_vp_proximity` answers "how close was a VP, for the TGs this method
placed best". It draws each method's cohort on its own row, which is what makes
it readable and also what hides the finding: those six cohorts are **largely
different TGs**. This module measures the sets themselves -- how big the union
is, how many TGs only one method found easy, and how each method behaved on a
reference method's cohort rather than on its own.

## The set structure is the first claim

At `p5` on the pooled as01-03 meshes the six 63-TG cohorts cover 254 distinct
TGs, 154 of which exactly one method found easy, and no TG is in more than four
of them. Six methods run on one population, each asked for its own best 5%,
and they mostly disagree about which 5% that is. `set_structure` reports the
union, the degree histogram and the full pairwise matrix, diagonal included, so
the cohort sizes and the overlaps read off one table.

## Behaviour on one cohort is the second

Overlap alone cannot separate "answered it well but not in its own top 5%" from
"refused it" from "answered it 149 km off", and at `p5` VAN and SPO both score
0% overlap for those opposite reasons. `against_reference` partitions the
reference cohort three ways -- **shared**, **answered-not-best**, **refused** --
and puts error percentiles beside them, over solved rows only.

**The reference's own row is emitted.** It is the scale the error column is
read against, not an achievement: at `p25` SOI's 13.8 km median reads as good
until the 8.8 km baseline row on the identical TGs sits above it.

`beats_reference` is the per-TG form of the same comparison and is the
defensible one. The cohort is selected on the reference's own error, so part of
any median gap is regression to the mean; a per-TG win rate is not exposed to
that. It is strictly `<`, so the reference scores 0% against itself.

## VP ranking is why the two baselines agree

`ranking_agreement` asks how often the smallest-RTT VP *is* the geographically
closest one. On these meshes that is 19.1% of the population, 67.5% of S-P's
p25 cohort and 100% of its p5 cohort: S-P is accurate exactly where latency
happens to rank the fleet correctly. Exact equality and within-1-km agreement
coincide here -- the metric is bimodal, either the same VP or a different one
a hundred kilometres away -- and both are emitted anyway, because a run that
separates them has broken the phrasing the paper uses.

## The margin is load-bearing

`baseline_margin` asks how often the shortest-ping VP's own distance beats a
method's prediction. S-P *predicts* that VP's coordinate, so at `margin_km=0`
it scores ~100% against itself on ~0.006 km of floating-point noise. The
margin is a parameter with a default of 1.0 km, never an optional refinement.

## Counts, and what they are worth

~20 IP replicas share a site and so share its VP geometry exactly. Every count
here ships `*_distinct` beside it: the number of distinct
`geo_vp_dist_to_tg_km` values among the counted TGs, which collapses a replica
cluster to one. It is a lower bound on independent observations, and a count
whose distinct twin is much smaller is one observation wearing a large number.

Target-based throughout. No coordinates, no site ids and no place names reach
any column, by construction -- the only identifiers emitted are run ids, TG ids
and method terms.

No figure. An overlap bar chart was considered and rejected: at `p5` VAN and
SPO both draw 0% meaning opposite things, and at `p25` bar height does not
track quality, with SPO's 43.8% at 152.9 km drawing level with OCT-H's 42.6% at
20.8 km. Ship the table.

Command: `report-cohort-overlap`. Writes `_cross/cohort-overlap/<datasets>[@<arm>]/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.analysis.v5.modules import cross
from scripts.analysis.v5.modules import figure_vp_proximity as V
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules.methods import method_label, method_term_table
from scripts.analysis.v5.modules.paths import RunPaths
from scripts.analysis.v5.modules.status import SHORTEST_PING

#: `_cross/<KIND>/<datasets>[@<arm>]/`.
KIND = "cohort-overlap"

#: Cohorts this report is read over. `p95`/`all` are legal -- `COHORTS` is
#: shared with the figure -- but the two the paper quotes are these.
DEFAULT_COHORTS = ("p5", "p25")

#: Default reference method for `against_reference` / `beats_reference`.
#: A parameter, not a constant: any method can be the reference. S-P is the
#: default because it is the control the paper reads the variants against.
DEFAULT_REFERENCE = SHORTEST_PING

#: Why `baseline_margin` needs one. See the module docstring.
DEFAULT_MARGIN_KM = 1.0

#: Tolerance for "the smallest-RTT VP is the closest VP", beside exact equality.
RANKING_TOLERANCE_KM = 1.0

#: Error percentiles reported on the reference cohort.
ERROR_QUANTILES = (0.25, 0.5, 0.75, 0.9, 0.95)

#: The column whose distinct values stand in for independent observations.
#: A distance, not a location: replicas of one site share it exactly, so
#: counting its distinct values collapses a replica cluster to one.
DISTINCT_COLUMN = V.MEASURE_COLUMNS[V.GEO]

#: What a prediction's accuracy is measured in, and what cohorts rank on.
ERROR_COLUMN = V.RANK_COLUMN

GEO_COL = V.MEASURE_COLUMNS[V.GEO]
SPING_COL = V.MEASURE_COLUMNS[V.SPING]

#: Every share this module emits, and the count it divides. Each of those
#: counts ships `<count>_distinct` beside it, because ~20 IP replicas share a
#: site and its VP geometry: a raw numerator overstates how many independent
#: observations a percentage rests on. The mapping is declared rather than
#: inferred so the test can walk it.
PCT_NUMERATORS = {
    "share_of_union_pct": "n_tgs",
    "shared_of_a_pct": "n_shared",
    "shared_pct": "n_shared",
    "answered_not_best_pct": "n_answered_not_best",
    "refused_pct": "n_refused",
    "beats_pct": "n_beats",
    "exact_agree_pct": "n_exact_agree",
    "within_tolerance_pct": "n_within_tolerance",
    "rtt_tied_pct": "n_rtt_tied",
    "baseline_closer_pct": "n_baseline_closer",
}

#: `{report}` is the table name; one CSV per table per cohort.
CSV_NAME = "cohort_overlap.{cohort}.{report}.csv"
MANIFEST_NAME = "cohort_overlap.{cohort}.manifest.json"

#: `set_structure` returns these three frames.
SUMMARY, DEGREE, PAIRWISE = "summary", "degree", "pairwise"

#: Table name -> the function that builds it, for the manifest.
REPORTS = (
    SUMMARY,
    DEGREE,
    PAIRWISE,
    "against_reference",
    "beats_reference",
    "ranking_agreement",
    "baseline_margin",
)


def output_dir(run_ids: list[str], *, analysis_root: Path | None = None) -> Path:
    """`_cross/cohort-overlap/<datasets>[@<arm>]/`, created."""
    return cross.cross_dir(run_ids, analysis_root=analysis_root, kind=KIND)


# -- keys and counts --------------------------------------------------------


def tg_keys(frame: pd.DataFrame) -> pd.Series:
    """`run_id|tg_id` per row.

    TGs are disjoint across runs -- `load` refuses to pool otherwise -- but the
    key carries the run anyway, so a future pooling bug shows up as an empty
    intersection rather than as a silent double count.
    """
    return frame["run_id"].astype(str) + "|" + frame["tg_id"].astype(str)


def _distinct(values: pd.Series) -> int:
    """How many distinct values a count rests on, to 2 dp.

    ~20 IP replicas share a site and so share `geo_vp_dist_to_tg_km` exactly.
    Six TGs at one distance are one observation, and this is the number that
    says so.
    """
    v = pd.Series(values).dropna()
    return int(v.round(2).nunique())


def _count(keys, lookup: pd.Series) -> tuple[int, int]:
    """`(n, n_distinct)` for a set of TG keys, against a key -> distance map."""
    keys = list(keys)
    return len(keys), _distinct(lookup.reindex(keys))


def _distinct_lookup(long: pd.DataFrame) -> pd.Series:
    """key -> `geo_vp_dist_to_tg_km`, one row per TG."""
    d = long.drop_duplicates(subset=["run_id", "tg_id"]).copy()
    return pd.Series(d[DISTINCT_COLUMN].values, index=tg_keys(d))


def cohort_members(rows: pd.DataFrame) -> dict[str, set[str]]:
    """method -> the set of TG keys in its cohort."""
    keyed = rows.assign(_key=tg_keys(rows))
    return {m: set(g._key) for m, g in keyed.groupby("method", sort=True)}


def _ordered_methods(members: dict[str, set[str]], reference: str | None = None) -> list[str]:
    """Method ids, sorted, with `reference` pulled to the front.

    Deliberately *not* sorted by any of the shares in the table. Ordering rows
    by overlap would make the table assert a ranking, and overlap is not
    quality: at `p25` SPO's 43.8% at 152.9 km outranks OCT-H's 42.6% at
    20.8 km. Alphabetical is a non-claim.
    """
    ms = sorted(members)
    if reference is not None and reference in ms:
        ms = [reference] + [m for m in ms if m != reference]
    return ms


def _labelled(method: str) -> dict:
    return {"method": method, "method_label": method_label(method)}


# -- set structure ----------------------------------------------------------


def set_structure(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """How the per-method cohorts sit against each other.

    `rows` is `figure_vp_proximity.cohort_frame` output -- never a cohort
    rebuilt locally, or the FALLBACK trap re-opens (a give-up row carries a
    real `pred_dist_to_tg_km`, the S-P VP's coordinate, so ranking without
    `status.solved_mask` fills a method's "best" cohort with refusals).

    Returns three frames:

    * `summary` -- one row: union size, singleton count, max and mean degree.
    * `degree` -- one row per degree d: how many TGs exactly d methods placed
      in their cohort.
    * `pairwise` -- one row per ordered method pair, **diagonal included**, so
      each method's own cohort size is in the same table as its overlaps.

    Degree is over the union, so the shares in `degree` sum to 100%.
    """
    members = cohort_members(rows)
    methods = _ordered_methods(members)
    lookup = _distinct_lookup(rows)
    union = sorted(set().union(*members.values())) if members else []

    degrees = pd.Series(
        {k: sum(k in s for s in members.values()) for k in union}, dtype="int64"
    )
    n_union, n_union_distinct = _count(union, lookup)
    summary = pd.DataFrame([{
        "n_methods": len(methods),
        "n_union": n_union,
        "n_union_distinct": n_union_distinct,
        "n_singletons": int((degrees == 1).sum()) if len(degrees) else 0,
        "n_all_methods": int((degrees == len(methods)).sum()) if len(degrees) else 0,
        "max_degree": int(degrees.max()) if len(degrees) else 0,
        "mean_degree": float(degrees.mean()) if len(degrees) else float("nan"),
    }])

    degree_rows = []
    for d in range(1, (int(degrees.max()) if len(degrees) else 0) + 1):
        keys = degrees.index[degrees == d]
        n, n_distinct = _count(keys, lookup)
        degree_rows.append({
            "degree": d,
            "n_tgs": n,
            "n_tgs_distinct": n_distinct,
            "share_of_union_pct": 100.0 * n / n_union if n_union else float("nan"),
        })
    degree = pd.DataFrame(degree_rows, columns=[
        "degree", "n_tgs", "n_tgs_distinct", "share_of_union_pct"])

    pair_rows = []
    for a in methods:
        for b in methods:
            shared = members[a] & members[b]
            n_shared, n_shared_distinct = _count(sorted(shared), lookup)
            n_a, n_a_distinct = _count(sorted(members[a]), lookup)
            n_b, n_b_distinct = _count(sorted(members[b]), lookup)
            union_ab = len(members[a] | members[b])
            pair_rows.append({
                "method_a": a,
                "method_label_a": method_label(a),
                "method_b": b,
                "method_label_b": method_label(b),
                "n_a": n_a,
                "n_a_distinct": n_a_distinct,
                "n_b": n_b,
                "n_b_distinct": n_b_distinct,
                "n_shared": n_shared,
                "n_shared_distinct": n_shared_distinct,
                "shared_of_a_pct": 100.0 * n_shared / n_a if n_a else float("nan"),
                "jaccard": n_shared / union_ab if union_ab else float("nan"),
            })
    pairwise = pd.DataFrame(pair_rows)
    return {SUMMARY: summary, DEGREE: degree, PAIRWISE: pairwise}


# -- behaviour on a reference cohort ---------------------------------------


def _require_reference(members: dict[str, set[str]], reference: str) -> None:
    if reference not in members:
        raise ValueError(
            f"reference {reference!r} has no cohort here; scored methods are "
            f"{sorted(members)}. Pass --reference with one of those."
        )


def _quantiles(values: pd.Series, prefix: str = "err") -> dict:
    v = pd.Series(values).dropna()
    out = {f"{prefix}_{k}": float("nan") for k in ("min_km", "max_km", "mean_km")}
    out.update({f"{prefix}_p{int(q * 100)}_km": float("nan") for q in ERROR_QUANTILES})
    if not len(v):
        return out
    out[f"{prefix}_min_km"] = float(v.min())
    out[f"{prefix}_max_km"] = float(v.max())
    out[f"{prefix}_mean_km"] = float(v.mean())
    for q in ERROR_QUANTILES:
        out[f"{prefix}_p{int(q * 100)}_km"] = float(v.quantile(q))
    return out


def against_reference(
    long: pd.DataFrame, rows: pd.DataFrame, reference: str = DEFAULT_REFERENCE
) -> pd.DataFrame:
    """What every method did on the reference method's cohort.

    One row per method, the **reference's own row included and first**. Without
    it the error column has no scale and SOI's 13.8 km at `p25` reads as good
    rather than as worse than the 8.8 km baseline on the identical TGs.

    The reference cohort is partitioned three ways, and the three shares sum to
    100%:

    * `shared` -- the TG is in this method's own cohort too;
    * `answered_not_best` -- answered, but not among its own best;
    * `refused` -- `status.solved_mask` is false (FALLBACK or worse).

    Error percentiles are over **solved rows only**, so a refusing method's
    denominator is its answered count, not the cohort size. VAN's p5 median
    rests on 27 of 63 TGs and `n_solved` says so on the row.
    """
    members = cohort_members(rows)
    _require_reference(members, reference)
    ref_keys = members[reference]
    lookup = _distinct_lookup(long)

    on_ref = long.assign(_key=tg_keys(long))
    on_ref = on_ref[on_ref._key.isin(ref_keys)]
    n_ref, n_ref_distinct = _count(sorted(ref_keys), lookup)

    out = []
    for method in _ordered_methods(members, reference):
        g = on_ref[on_ref.method == method]
        solved = g[g.solved]
        # Solved AND in its own cohort. The extra clause is a no-op for a
        # percentile cohort (those rows are solved by construction) and is what
        # keeps the three shares a partition for `all`, which keeps refusals.
        shared_keys = list(solved._key[solved._key.isin(members[method])])
        n_shared, n_shared_distinct = _count(shared_keys, lookup)
        n_solved, n_solved_distinct = _count(list(solved._key), lookup)
        n_other, n_other_distinct = _count(
            list(solved._key[~solved._key.isin(members[method])]), lookup
        )
        n_refused, n_refused_distinct = _count(list(g._key[~g.solved]), lookup)
        out.append({
            **_labelled(method),
            "is_reference": method == reference,
            "reference": reference,
            "n_reference_cohort": n_ref,
            "n_reference_cohort_distinct": n_ref_distinct,
            "n_shared": n_shared,
            "n_shared_distinct": n_shared_distinct,
            "shared_pct": 100.0 * n_shared / n_ref if n_ref else float("nan"),
            "n_answered_not_best": n_other,
            "n_answered_not_best_distinct": n_other_distinct,
            "answered_not_best_pct": 100.0 * n_other / n_ref if n_ref else float("nan"),
            "n_refused": n_refused,
            "n_refused_distinct": n_refused_distinct,
            "refused_pct": 100.0 * n_refused / n_ref if n_ref else float("nan"),
            "n_solved": n_solved,
            "n_solved_distinct": n_solved_distinct,
            **_quantiles(solved[ERROR_COLUMN]),
        })
    return pd.DataFrame(out)


def beats_reference(
    long: pd.DataFrame, rows: pd.DataFrame, reference: str = DEFAULT_REFERENCE
) -> pd.DataFrame:
    """Per-TG, how often a method landed strictly closer than the reference.

    Over the reference's own cohort, solved rows only. This is a different
    claim from the median comparison in `against_reference` and the defensible
    one: the cohort is selected on the reference's error, so part of any median
    gap is regression to the mean, while a per-TG win rate is not exposed to
    that.

    Strictly `<`. The reference scores 0% against itself, which is the point --
    a `<=` here would hand it 100% on the ties it has with itself by
    construction, and hand every method its exact ties for free.
    """
    members = cohort_members(rows)
    _require_reference(members, reference)
    ref_keys = members[reference]
    lookup = _distinct_lookup(long)

    keyed = long.assign(_key=tg_keys(long))
    on_ref = keyed[keyed._key.isin(ref_keys)]
    ref_err = on_ref[on_ref.method == reference].set_index("_key")[ERROR_COLUMN]
    n_ref = len(ref_keys)

    out = []
    for method in _ordered_methods(members, reference):
        solved = on_ref[(on_ref.method == method) & on_ref.solved].set_index("_key")
        theirs = solved[ERROR_COLUMN]
        mine = ref_err.reindex(theirs.index)
        won = theirs < mine
        n_solved, n_solved_distinct = _count(list(theirs.index), lookup)
        n_beats, n_beats_distinct = _count(list(theirs.index[won.values]), lookup)
        out.append({
            **_labelled(method),
            "is_reference": method == reference,
            "reference": reference,
            "n_reference_cohort": n_ref,
            "n_solved": n_solved,
            "n_solved_distinct": n_solved_distinct,
            "n_beats": n_beats,
            "n_beats_distinct": n_beats_distinct,
            "beats_pct": 100.0 * n_beats / n_solved if n_solved else float("nan"),
            "n_exact_ties": int((theirs == mine).sum()),
            "median_delta_km": float((theirs - mine).median()) if n_solved else float("nan"),
        })
    return pd.DataFrame(out)


# -- VP ranking agreement ---------------------------------------------------


def _agreement(frame: pd.DataFrame, lookup: pd.Series) -> dict:
    gap = frame[SPING_COL] - frame[GEO_COL]
    exact = gap.abs() <= 0.0
    within = gap.abs() <= RANKING_TOLERANCE_KM
    keys = tg_keys(frame)
    n, n_distinct = _count(list(keys), lookup)
    ties = frame["n_sping_vp_ties"]
    n_exact, n_exact_distinct = _count(list(keys[exact.values]), lookup)
    n_within, n_within_distinct = _count(list(keys[within.values]), lookup)
    n_tied, n_tied_distinct = _count(list(keys[(ties > 1).values]), lookup)
    return {
        "n": n,
        "n_distinct": n_distinct,
        "n_exact_agree": n_exact,
        "n_exact_agree_distinct": n_exact_distinct,
        "exact_agree_pct": 100.0 * float(exact.mean()) if n else float("nan"),
        "n_within_tolerance": n_within,
        "n_within_tolerance_distinct": n_within_distinct,
        "within_tolerance_pct": 100.0 * float(within.mean()) if n else float("nan"),
        "tolerance_km": RANKING_TOLERANCE_KM,
        "median_gap_km": float(gap.median()) if n else float("nan"),
        "p90_gap_km": float(gap.quantile(0.9)) if n else float("nan"),
        "max_gap_km": float(gap.max()) if n else float("nan"),
        "n_rtt_tied": n_tied,
        "n_rtt_tied_distinct": n_tied_distinct,
        "rtt_tied_pct": 100.0 * float((ties > 1).mean()) if n else float("nan"),
        "max_rtt_ties": int(ties.max()) if n else 0,
    }


def ranking_agreement(long: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """How often the smallest-RTT VP *is* the geographically closest VP.

    `scope="population"` is every evaluated TG, deduplicated to one row --
    `long` repeats each TG once per method and counting it six times would
    quote the method count as a sample size. Then one `scope="cohort"` row per
    method.

    Both `exact_agree_pct` (`sping == geo`) and `within_tolerance_pct` (within
    `RANKING_TOLERANCE_KM`) are emitted. They are equal at every scope on the
    as01-03 meshes -- the gap is either zero or ~100 km, never 0.4 km -- so the
    metric needs no threshold today. They are still reported separately: a run
    that separates them has stopped being bimodal, and the paper's "the
    smallest-RTT VP *is* the closest VP" phrasing would have to change.

    `n_rtt_tied` counts TGs with more than one VP at the minimum RTT (0.1 ms
    resolution), where "the smallest-RTT VP" is a tie-break rather than a fact.
    """
    lookup = _distinct_lookup(long)
    dedup = long.drop_duplicates(subset=["run_id", "tg_id"])
    out = [{
        "scope": "population",
        "method": "",
        "method_label": "all TGs",
        **_agreement(dedup, lookup),
    }]
    for method, g in rows.groupby("method", sort=True):
        out.append({
            "scope": "cohort",
            **_labelled(method),
            **_agreement(g, lookup),
        })
    return pd.DataFrame(out)


# -- the baseline's own distance, against a method's prediction -------------


def baseline_margin(
    rows: pd.DataFrame, margin_km: float = DEFAULT_MARGIN_KM
) -> pd.DataFrame:
    """How often the shortest-ping VP's own distance beat a method's prediction.

    Over each method's **own** cohort -- its best TGs, where it is at its
    strongest -- and over solved rows, since a refusal has no prediction to
    beat. The test is

        sping_vp_dist_to_tg_km < pred_dist_to_tg_km - margin_km

    **`margin_km` is required, not cosmetic.** S-P predicts the smallest-RTT
    VP's coordinate, so the two sides are equal by construction for it and
    differ only by ~0.006 km of floating point. At `margin_km=0` S-P therefore
    scores ~100% against itself, which is an artefact of the comparison rather
    than a finding; at the 1.0 km default it scores 0%. Any caller lowering the
    margin toward zero is measuring rounding.
    """
    if margin_km < 0:
        raise ValueError(f"margin_km must be >= 0, got {margin_km}")
    lookup = _distinct_lookup(rows)
    out = []
    for method, g in rows.groupby("method", sort=True):
        solved = g[g.solved]
        beaten = solved[SPING_COL] < solved[ERROR_COLUMN] - margin_km
        n, n_distinct = _count(list(tg_keys(solved)), lookup)
        n_beaten, n_beaten_distinct = _count(list(tg_keys(solved[beaten.values])), lookup)
        delta = solved[ERROR_COLUMN] - solved[SPING_COL]
        out.append({
            **_labelled(method),
            "margin_km": float(margin_km),
            "n": n,
            "n_distinct": n_distinct,
            "n_baseline_closer": n_beaten,
            "n_baseline_closer_distinct": n_beaten_distinct,
            "baseline_closer_pct": 100.0 * n_beaten / n if n else float("nan"),
            "median_delta_km": float(delta.median()) if n else float("nan"),
        })
    return pd.DataFrame(out)


# -- assembly ---------------------------------------------------------------


def tables(
    long: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    reference: str = DEFAULT_REFERENCE,
    margin_km: float = DEFAULT_MARGIN_KM,
) -> dict[str, pd.DataFrame]:
    """Every table this module emits, keyed by `REPORTS` name."""
    out = dict(set_structure(rows))
    out["against_reference"] = against_reference(long, rows, reference=reference)
    out["beats_reference"] = beats_reference(long, rows, reference=reference)
    out["ranking_agreement"] = ranking_agreement(long, rows)
    out["baseline_margin"] = baseline_margin(rows, margin_km=margin_km)
    return out


def _manifest(
    meta: dict,
    cohort: str,
    built: dict[str, pd.DataFrame],
    *,
    reference: str,
    margin_km: float,
) -> str:
    n_per = built[PAIRWISE].query("method_a == method_b").set_index("method_a").n_a.to_dict()
    frac = V.COHORTS[cohort]
    # From the pooled TG count, not from the rows k already selected.
    k = None if frac is None else int(round(frac * meta["n_tgs"]))
    summary = built[SUMMARY].iloc[0]
    ref_row = built["against_reference"].query("is_reference").iloc[0]
    body = {
        "report": "cohort_overlap",
        "cohort": cohort,
        "cohort_fraction": frac,
        "cohort_k_requested": k,
        "reference": reference,
        "reference_label": method_label(reference),
        "margin_km": margin_km,
        "ranking_tolerance_km": RANKING_TOLERANCE_KM,
        "rank_column": ERROR_COLUMN,
        "distinct_column": DISTINCT_COLUMN,
        "datasets": cross.dataset_slug(meta["run_ids"]),
        "run_ids": meta["run_ids"],
        "source_nside": meta["nside"],
        "n_tgs_pooled": meta["n_tgs"],
        "n_vp_per_tg_median": meta["n_vp_per_tg_median"],
        "n_cohort_per_method": {method_label(m): int(v) for m, v in n_per.items()},
        "n_cohort_short": {
            method_label(m): int(v) for m, v in n_per.items() if k is not None and int(v) < k
        },
        "method_terms": method_term_table(meta["methods"]),
        "tables": {r: CSV_NAME.format(cohort=cohort, report=r) for r in REPORTS},
        "headline": {
            "n_union": int(summary.n_union),
            "n_union_distinct": int(summary.n_union_distinct),
            "max_degree": int(summary.max_degree),
            "degree_histogram": {
                int(r.degree): int(r.n_tgs) for r in built[DEGREE].itertuples()
            },
            "reference_err_p50_km": float(ref_row.err_p50_km),
        },
        "policy": {
            "cohort_selection": (
                "figure_vp_proximity.cohort_frame, unchanged: each method's own "
                "smallest pred_dist_to_tg_km over status.solved_mask rows. "
                "Cohorts are NEVER rebuilt here -- a FALLBACK row carries a real "
                "pred_dist_to_tg_km (the S-P VP's coordinate), so ranking "
                "without the mask fills a method's best cohort with refusals."
            ),
            "reference_row": (
                "The reference method's own row is emitted in against_reference "
                "and beats_reference. It is the scale the error column is read "
                "against, not an achievement."
            ),
            "shares_partition": (
                "shared_pct + answered_not_best_pct + refused_pct = 100 over the "
                "reference cohort. Error percentiles are over solved rows only, "
                "so their denominator is n_solved, not n_reference_cohort."
            ),
            "beats_is_strict": (
                "beats_reference uses <, so the reference scores 0% against "
                "itself and no method is credited with an exact tie."
            ),
            "margin_is_required": (
                "baseline_margin at margin_km=0 scores S-P ~100% against itself "
                "on ~0.006 km of floating-point noise, because S-P predicts the "
                "smallest-RTT VP's own coordinate. The default is 1.0 km."
            ),
            "distinct_counts": (
                f"Every count backing a share ships a *_distinct twin "
                f"({sorted(PCT_NUMERATORS)}). A *_distinct column counts distinct "
                f"{DISTINCT_COLUMN} "
                f"values (2 dp) among the counted TGs. ~20 IP replicas share a "
                f"site and its VP geometry, so a raw count overstates "
                f"independence; the distinct twin is the lower bound."
            ),
            "row_order": (
                "Alphabetical by method id, reference first. Deliberately not "
                "sorted by overlap: overlap is not quality, and a sorted table "
                "would assert a ranking it cannot support."
            ),
            "no_figure": (
                "An overlap bar chart was considered and rejected: at p5 VAN and "
                "SPO both plot 0% for opposite reasons (refusal vs answered and "
                "149 km off), and at p25 bar height does not track quality."
            ),
        },
    }
    return json.dumps(body, indent=2) + "\n"


def build_for_runs(
    runs: list[RunPaths],
    *,
    cohorts: list[str] | None = None,
    methods: list[str] | None = None,
    reference: str = DEFAULT_REFERENCE,
    margin_km: float = DEFAULT_MARGIN_KM,
    nside: int = V.SOURCE_NSIDE,
    analysis_root: Path | None = None,
    source_csv: dict[str, Path] | None = None,
) -> list[Path]:
    """Seven CSVs and a manifest per cohort. Returns every path written."""
    cohorts = list(dict.fromkeys(cohorts or list(DEFAULT_COHORTS)))
    unknown = [c for c in cohorts if c not in V.COHORTS]
    if unknown:
        # Before the CSVs are read: the distances take seconds per mesh.
        raise ValueError(f"unknown cohort {unknown}; known: {sorted(V.COHORTS)}")
    if margin_km < 0:
        raise ValueError(f"margin_km must be >= 0, got {margin_km}")
    nside = G.validate_nside(nside)
    long, meta = V.load(runs, methods=methods, nside=nside, analysis_root=analysis_root,
                        source_csv=source_csv)
    if reference not in meta["methods"]:
        raise ValueError(
            f"reference {reference!r} is not scored in every run; scored: "
            f"{meta['methods']}."
        )
    out_dir = output_dir(meta["run_ids"], analysis_root=analysis_root)
    written = []
    for cohort in cohorts:
        rows = V.cohort_frame(long, cohort)
        built = tables(long, rows, reference=reference, margin_km=margin_km)
        for name in REPORTS:
            path = out_dir / CSV_NAME.format(cohort=cohort, report=name)
            built[name].to_csv(path, index=False)
            written.append(path)
        manifest = out_dir / MANIFEST_NAME.format(cohort=cohort)
        manifest.write_text(
            _manifest(meta, cohort, built, reference=reference, margin_km=margin_km)
        )
        written.append(manifest)
    return written
