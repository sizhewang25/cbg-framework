"""Which interconnect sites a measured pair *could* have crossed, at 2/3 c.

`build-pni-graph` answers "which site is most plausible" with an argmin, which
is a point estimate carrying no guarantee. This module answers the weaker and
therefore stronger question: **which sites are physically possible**, and it
answers it by exclusion, so the result is a proof rather than a fit.

## The test

For a measured pair with min-RTT `r`, site `p` is feasible iff

    THEORETICAL_SLOPE * [ d(VP, p) + d(p, TG) ]  <=  r

The feasible set is the sites lying inside an ellipse with foci VP and TG and
major axis `r / THEORETICAL_SLOPE`. That is one boolean per (pair, site), and
nothing is fitted, assumed about routing policy, or tuned.

Why this is the right instrument for §8.1. The paper needs "the target is served
through a *nearby* interconnect, not a distant one". Feasibility delivers
exactly that and needs no model of the CDN's site-selection strategy: if only
near sites are physically reachable within the observed RTT, then whatever
policy is in force selected a near site. `build-pni-graph`'s argmin, by
contrast, can only say which site would be *cheapest*.

## Why `k` is 2/3 c and not the measured speed

Exclusion is sound only when `k` upper-bounds the propagation speed. A
calibrated speed is an *average* — roughly half of all paths beat it — so using
one would wrongly exclude the true site on about half the pairs, and a
low-quantile envelope wrongly excludes at its own quantile rate on every
constraint. `THEORETICAL_SLOPE` is a physical bound, so every exclusion it makes
is valid. This is the one parameter in the module and it is not a choice.

The consequence runs the safe way. A per-target additive access floor (last-mile
serialization, which no amount of min-taking removes) inflates `r`, which
*enlarges* the ellipse, which admits *more* sites. So the test under-excludes
and every error is conservative — the opposite of the failure mode that would
make a small feasible set an artifact.

## Three reporting rules, each from a failure this layer already hit

**Never hard-AND across a target's VPs.** Intersecting every VP's ellipse is the
statistically obvious move and it is brittle for the reason `SphericalCircleMTL`
is: one bad coordinate or one anomalous RTT empties the region and the target
reports "no site is possible". So `feasible_sites.csv` reports, per (target,
site), the *share* of that target's VPs for which the site is feasible.
`feasible_under_all_vps` is emitted so the brittleness is measured, and is never
used as a filter.

**Stratify by `d(VP, TG)`, and headline the shortest-ping VP.** Selectivity is
not a constant. A near VP with a small RTT admits almost nothing; a far VP's
ellipse elongates along the VP-TG axis and admits sites near the *VP* that are
nowhere near the target, so its feasible ranks by proximity-to-target come out
non-contiguous. Any claim of the form "only the target's nearest site survives"
is a claim about near VPs, and `headline_sping_only` in `meta.json` is the block
that may be quoted.

**Split the empty set by cause.** An empty feasible set is the designated
falsifier — a small RTT with no reachable listed site means the target is served
off-list. But because `via >= direct` always, a pair whose RTT already beats
2/3 c on the *direct* geodesic has an empty set by arithmetic, and that is a
coordinate or RTT problem (`build-pni-graph` reports it as
`soi.soi_violation_share`). Only `n_empty_unexplained_by_soi` is evidence. This
is the same nesting argument `pni.py` makes for routing-vs-air violations.

## The null

With a nine-site list, `n_feasible / n_pni` has 11% granularity, so "most pairs
have a singleton feasible set" says as much about the list's sparsity as about
the network. `--decoy-trials N` re-runs the identical pass against sites drawn
uniformly from the real sites' bounding box, so the observed selectivity gets an
effect size instead of a bare number. Default 0, with a warning, because it
multiplies the work; the reportable run passes 200.

Distances are recomputed from the coordinate columns, never from
`pni_edges.csv`'s km columns: those are rounded at `pni._KM_DIGITS`, and 5e-4 km
is 5e-6 ms, which is enough to flip a boundary case.
`checks.max_abs_km_disagreement_vs_pni_edges` reports the agreement.

Command: `build-pni-feasibility`. Writes to
`outputs/analysis/v3/<run_id>/pni-feasibility/`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules import pni
from scripts.analysis.v3.modules.answer_space import elementwise_km
from scripts.analysis.v3.modules.bipartite import describe_p90, quantile_bins, truthy
from scripts.analysis.v3.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    MissingArtifactError,
    RunPaths,
    discover_runs,
    resolve_run,
)
from scripts.libs.cbg.rtt_model import THEORETICAL_SLOPE

FEASIBLE_PAIRS_CSV = "feasible_pairs.csv"
FEASIBLE_SITES_CSV = "feasible_sites.csv"
FEASIBLE_NULL_CSV = "feasible_null.csv"
META_JSON = "meta.json"

#: Slack on the km comparison. The two legs and the budget are separate arccos
#: calls, and `answer_space.pairwise_km`'s chord form loses relative precision
#: as the dot product approaches 1, so a continental distance carries ~1e-5 km
#: of float noise. 1e-4 km sits above that and three orders of magnitude below
#: `pni._KM_DIGITS`' 5e-4 rounding, so it cannot mask a real exclusion.
_FEAS_EPS_KM = 1e-4

#: Longest rank list written as a string. The machine-readable form is
#: `feasible_ranks_are_contiguous`; the string exists to be read by a human on a
#: handful of rows, and at 1,000 sites an uncapped column would be ~20 MB of one
#: CSV field. Rows over the cap set `feasible_tg_site_ranks_truncated`.
_MAX_RANKS_LISTED = 12

#: Quantile strata for `d(VP, TG)`. Quantile rather than fixed km because the
#: whole point of the stratification is that selectivity varies with VP distance
#: and the scale of that variation is a property of the dataset; a fixed km list
#: would be a tuned threshold wearing a constant's name. Edges go in `meta.json`.
_N_VP_DISTANCE_BINS = 4

#: Fixed so a decoy run is reproducible. A null whose value moves between
#: invocations cannot be quoted next to an observed number.
_DECOY_SEED = 20260910

_K_NOTE = (
    "k is THEORETICAL_SLOPE, the round-trip ms per km at 2/3 c. Exclusion is "
    "sound only when k upper-bounds propagation speed, so a calibrated or "
    "envelope-fitted speed may never be the primary here: it is an average, and "
    "about half of all paths beat it, so it would wrongly exclude the true site. "
    "Every exclusion this k makes is physically valid."
)
_CONSERVATIVE_NOTE = (
    "A per-target additive access floor inflates rtt_ms, which enlarges the "
    "ellipse, which admits more sites. The test therefore under-excludes; every "
    "error is in the conservative direction and a small feasible set cannot be "
    "an artifact of the floor."
)
_EMPTY_NOTE = (
    "The falsifier is n_empty_unexplained_by_soi alone. Since via >= direct "
    "always, a pair whose rtt_ms already beats 2/3 c on the direct geodesic has "
    "an empty feasible set by arithmetic -- that is a coordinate or RTT problem, "
    "reported by build-pni-graph as soi.soi_violation_share, not evidence that "
    "the target is served off-list."
)
_INTERSECTION_NOTE = (
    "feasible_under_all_vps is the hard intersection across a target's VPs. It "
    "is emitted to be measured, never to filter: one under-predicting "
    "constraint empties an intersection, which is why SphericalCircleMTL is "
    "brittle. vps_feasible_share is the statistic to use."
)
_HEADLINE_NOTE = (
    "Selectivity depends on d(VP, TG): a far VP's ellipse elongates along the "
    "VP-TG axis and admits sites near the VP that are far from the target, so "
    "its feasible ranks by proximity-to-target are not contiguous. Only this "
    "block -- the shortest-ping VP, one pair per target -- supports a claim of "
    "the form 'only the target's nearest site is possible'."
)
_NULL_NOTE = (
    "Decoy sites are drawn uniformly from the real sites' bounding box, same "
    "count, same pairs, same k. Without this contrast n_feasible is "
    "uninterpretable: with few sites a singleton feasible set is as much a fact "
    "about the list's sparsity as about the network."
)


@dataclass(frozen=True)
class PniFeasibility:
    """The three tables and the meta block, ready to write."""

    pairs: pd.DataFrame
    sites: pd.DataFrame
    null: pd.DataFrame
    meta: dict

    def write(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.pairs.to_csv(out_dir / FEASIBLE_PAIRS_CSV, index=False)
        self.sites.to_csv(out_dir / FEASIBLE_SITES_CSV, index=False)
        self.null.to_csv(out_dir / FEASIBLE_NULL_CSV, index=False)
        (out_dir / META_JSON).write_text(json.dumps(self.meta, indent=2) + "\n")
        return out_dir


def site_ranks_by_target(d_tg_p: np.ndarray) -> np.ndarray:
    """1-based rank of each site by its distance to each target.

    `d(p, TG)` does not vary over a target's VPs, so this is a per-target
    constant computed once as an argsort and gathered per pair, rather than
    recomputed 134 times per target.
    """
    order = np.argsort(d_tg_p, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks, order, np.arange(1, d_tg_p.shape[1] + 1)[None, :].repeat(len(order), 0), axis=1
    )
    return ranks


def feasibility_pass(
    d_vp_p: np.ndarray,
    d_tg_p: np.ndarray,
    e_vp: np.ndarray,
    e_tg: np.ndarray,
    budget_km: np.ndarray,
    *,
    tg_rank: np.ndarray | None = None,
    sel: np.ndarray | None = None,
    tg_nearest: np.ndarray | None = None,
    want_rank_strings: bool = False,
    n_targets: int | None = None,
) -> dict:
    """One vectorized ellipse test, chunked over pairs.

    Shared by the observed run and every decoy trial, so a null can never
    disagree with the observation on anything but the site coordinates. Chunked
    on `pni._ARGMIN_CHUNK_CELLS` — a cell budget rather than a row count, because
    a fixed row chunk is fine at nine sites and half a gigabyte at a thousand.
    """
    n_e = int(len(e_vp))
    n_p = int(d_vp_p.shape[1])
    if tg_rank is None:
        tg_rank = site_ranks_by_target(d_tg_p)

    n_feasible = np.zeros(n_e, dtype=np.int64)
    contains_nearest = np.zeros(n_e, dtype=bool)
    sel_feasible = np.zeros(n_e, dtype=bool)
    pni_to_tg_max = np.full(n_e, np.nan)
    pni_to_tg_min = np.full(n_e, np.nan)
    rank_best = np.zeros(n_e, dtype=np.int64)
    rank_worst = np.zeros(n_e, dtype=np.int64)
    rank_strings: list[str] = []
    # (target x site) count of the VPs for which each site is feasible. Kept as
    # a running accumulator so the chunk loop never has to hold the full boolean.
    site_counts = np.zeros((int(n_targets), n_p), dtype=np.int64) if n_targets else None

    cap = min(_MAX_RANKS_LISTED, n_p)
    chunk = max(pni._ARGMIN_CHUNK_CELLS // max(n_p, 1), 1)
    for start in range(0, n_e, chunk):
        stop = min(start + chunk, n_e)
        rows = np.arange(stop - start)
        legs = d_vp_p[e_vp[start:stop], :] + d_tg_p[e_tg[start:stop], :]
        feas = legs <= budget_km[start:stop, None] + _FEAS_EPS_KM

        n_feasible[start:stop] = feas.sum(axis=1)
        leg2 = d_tg_p[e_tg[start:stop], :]
        masked = np.where(feas, leg2, np.nan)
        with np.errstate(all="ignore"):
            pni_to_tg_max[start:stop] = np.nanmax(masked, axis=1, initial=-np.inf)
            pni_to_tg_min[start:stop] = np.nanmin(masked, axis=1, initial=np.inf)

        rk = tg_rank[e_tg[start:stop], :]
        big = n_p + 1
        rk_masked = np.where(feas, rk, big)
        rank_best[start:stop] = rk_masked.min(axis=1)
        rk_masked_lo = np.where(feas, rk, 0)
        rank_worst[start:stop] = rk_masked_lo.max(axis=1)

        if tg_nearest is not None:
            contains_nearest[start:stop] = feas[rows, tg_nearest[e_tg[start:stop]]]
        if sel is not None:
            sel_feasible[start:stop] = feas[rows, sel[start:stop]]
        if want_rank_strings:
            top = np.sort(rk_masked, axis=1)[:, :cap]
            rank_strings.extend(
                "|".join(str(int(v)) for v in row if v <= n_p) for row in top
            )
        if site_counts is not None:
            np.add.at(site_counts, e_tg[start:stop], feas)
        del legs, feas, masked, rk, rk_masked, rk_masked_lo

    empty = n_feasible == 0
    pni_to_tg_max = np.where(empty, np.nan, pni_to_tg_max)
    pni_to_tg_min = np.where(empty, np.nan, pni_to_tg_min)
    rank_best = np.where(empty, 0, rank_best)
    rank_worst = np.where(empty, 0, rank_worst)

    return {
        "n_feasible": n_feasible,
        "is_feasible_set_empty": empty,
        "is_feasible_set_singleton": n_feasible == 1,
        "feasible_contains_tg_nearest_pni": contains_nearest,
        "sel_pni_is_feasible": sel_feasible,
        "feasible_pni_to_tg_km_max": pni_to_tg_max,
        "feasible_pni_to_tg_km_min": pni_to_tg_min,
        "feasible_tg_site_rank_best": rank_best,
        "feasible_tg_site_rank_worst": rank_worst,
        # Ranks are distinct integers, so no gap means the span equals the count.
        # Contiguity does not imply the set starts at rank 1 -- both are reported.
        "feasible_ranks_are_contiguous": (~empty)
        & ((rank_worst - rank_best + 1) == n_feasible),
        "feasible_tg_site_ranks": rank_strings if want_rank_strings else None,
        "feasible_tg_site_ranks_truncated": n_feasible > cap,
        "site_feasible_counts": site_counts,
        "chunk_pairs": int(chunk),
    }


def _share(mask: np.ndarray) -> float:
    """Share of True, or NaN on an empty population rather than 0/0."""
    n = int(mask.size)
    return round(float(mask.sum()) / n, 6) if n else float("nan")


def _summary(sub: pd.DataFrame, n_pni: int) -> dict:
    """The feasibility block for one population of pairs.

    Computed identically for the whole edge set, for the shortest-ping subset and
    for each `d(VP, TG)` stratum, so the three are comparable by construction
    rather than by inspection.
    """
    if sub.empty:
        return {"n_pairs": 0}
    empty = sub["is_feasible_set_empty"].to_numpy(bool)
    singleton = sub["is_feasible_set_singleton"].to_numpy(bool)
    nearest = sub["feasible_contains_tg_nearest_pni"].to_numpy(bool)
    return {
        "n_pairs": int(len(sub)),
        "n_pni": int(n_pni),
        "n_feasible": describe_p90(sub["n_feasible"], digits=3),
        "singleton_share": _share(singleton),
        "singleton_equals_tg_nearest_share": _share(singleton & nearest),
        "contains_tg_nearest_share": _share(nearest),
        "sel_pni_is_feasible_share": _share(sub["sel_pni_is_feasible"].to_numpy(bool)),
        "best_rank_is_1_share": _share(
            (~empty) & (sub["feasible_tg_site_rank_best"].to_numpy() == 1)
        ),
        "contiguous_share": _share(sub["feasible_ranks_are_contiguous"].to_numpy(bool)),
        "feasible_pni_to_tg_km_max": describe_p90(sub["feasible_pni_to_tg_km_max"]),
        "empty_set_rate": {
            "n_empty": int(empty.sum()),
            "share_empty": _share(empty),
            "n_empty_and_soi_violating": int(sub["is_empty_and_soi_violating"].sum()),
            "n_empty_unexplained_by_soi": int(sub["is_empty_unexplained_by_soi"].sum()),
            "share_unexplained": _share(
                sub["is_empty_unexplained_by_soi"].to_numpy(bool)
            ),
            "note": _EMPTY_NOTE,
        },
    }


#: The per-pair arrays `_null_row` reads. Named explicitly because a
#: `feasibility_pass` result also carries target-indexed and scalar entries, and
#: a blanket subset by a pair mask would index those with the wrong length.
_NULL_ROW_KEYS = (
    "n_feasible",
    "is_feasible_set_empty",
    "is_feasible_set_singleton",
    "feasible_tg_site_rank_best",
    "feasible_ranks_are_contiguous",
    "feasible_pni_to_tg_km_max",
)


def _subset(res: dict, mask: np.ndarray) -> dict:
    """The per-pair arrays of a pass result, restricted to `mask`."""
    return {k: res[k][mask] for k in _NULL_ROW_KEYS}


def _null_row(res: dict, tg_nearest_hit: np.ndarray, *, trial: int, scope: str) -> dict:
    """One decoy (or observed) trial reduced to the numbers a null compares."""
    empty = res["is_feasible_set_empty"]
    singleton = res["is_feasible_set_singleton"]
    n_feas = res["n_feasible"]
    return {
        "trial": trial,
        "scope": scope,
        "n_pairs": int(n_feas.size),
        "n_feasible_p50": round(float(np.median(n_feas)), 3) if n_feas.size else np.nan,
        "n_feasible_mean": round(float(n_feas.mean()), 6) if n_feas.size else np.nan,
        "singleton_share": _share(singleton),
        "empty_share": _share(empty),
        "singleton_equals_tg_nearest_share": _share(singleton & tg_nearest_hit),
        "best_rank_is_1_share": _share(
            (~empty) & (res["feasible_tg_site_rank_best"] == 1)
        ),
        "contiguous_share": _share(res["feasible_ranks_are_contiguous"]),
        "feasible_pni_to_tg_km_max_p50": round(
            float(np.nanmedian(res["feasible_pni_to_tg_km_max"])), 3
        )
        if np.isfinite(res["feasible_pni_to_tg_km_max"]).any()
        else np.nan,
    }


def build_feasibility(
    graph: pni.PniGraph,
    *,
    decoy_trials: int = 0,
    source_label: str | None = None,
    pni_graph_dir: Path | None = None,
) -> PniFeasibility:
    """The ellipse test over one run's PNI graph, plus its optional null."""
    from scripts.analysis.v3.modules.answer_space import pairwise_km

    edges = graph.edges.reset_index(drop=True)
    # Sorted here rather than trusted from the CSV: `site_ranks_by_target`'s
    # argsort tie-break and every site-axis index below are properties of this
    # order, exactly as `build_pni_graph` re-sorts what `load_pni_sites` sorted.
    sites = graph.pni_nodes.sort_values("pni_id").reset_index(drop=True)
    n_p = int(len(sites))
    if n_p == 0:
        raise ValueError(f"{pni_graph_dir or 'the PNI graph'} has no sites in pni_nodes.csv")

    frames = pni.site_leg_frames(edges, sites)
    d_vp_p, d_tg_p = frames["d_vp_p"], frames["d_tg_p"]
    e_vp, e_tg = frames["e_vp"], frames["e_tg"]
    vp, tg = frames["vp"], frames["tg"]
    n_tg = int(len(tg))

    rtt = edges["rtt_ms"].to_numpy(float)
    budget_km = rtt / THEORETICAL_SLOPE
    direct_km = elementwise_km(
        edges["vp_lat"].to_numpy(float),
        edges["vp_lon"].to_numpy(float),
        edges["target_lat"].to_numpy(float),
        edges["target_lon"].to_numpy(float),
    )
    tg_rank = site_ranks_by_target(d_tg_p)
    tg_nearest = d_tg_p.argmin(axis=1).astype(np.intp)
    sel = pd.Index(sites["pni_id"].astype(str)).get_indexer(
        edges["sel_pni_id"].astype(str)
    ).astype(np.intp)
    if sel.min(initial=0) < 0:
        raise ValueError("pni_edges.csv names a sel_pni_id absent from pni_nodes.csv")

    res = feasibility_pass(
        d_vp_p, d_tg_p, e_vp, e_tg, budget_km,
        tg_rank=tg_rank, sel=sel, tg_nearest=tg_nearest,
        want_rank_strings=True, n_targets=n_tg,
    )

    # An air-side SoI violation empties the set by arithmetic, since via >=
    # direct. Split so the falsifier is not fired by data quality.
    ideal_direct = THEORETICAL_SLOPE * direct_km
    soi_violating = (ideal_direct > pni._IDEAL_MS_EPS) & (rtt < ideal_direct)
    empty = res["is_feasible_set_empty"]

    strat_idx, strat_edges = quantile_bins(pd.Series(direct_km), n_bins=_N_VP_DISTANCE_BINS)
    is_sping = (
        truthy(edges["is_sping_vp"]).to_numpy(bool)
        if "is_sping_vp" in edges.columns
        else np.zeros(len(edges), dtype=bool)
    )

    pairs = pd.DataFrame(
        {
            "vp_id": edges["vp_id"].to_numpy(),
            "target_id": edges["target_id"].to_numpy(),
            "rtt_ms": rtt,
            "is_sping_vp": is_sping,
            "vp_to_tg_km": np.round(direct_km, pni._KM_DIGITS),
            "vp_to_tg_via_pni_km": edges["vp_to_tg_via_pni_km"].to_numpy(float),
            "vp_to_tg_km_stratum": strat_idx.to_numpy(int),
            "sel_pni_id": edges["sel_pni_id"].to_numpy(),
            "sel_pni_is_feasible": res["sel_pni_is_feasible"],
            "tg_nearest_pni_id": sites["pni_id"].to_numpy()[tg_nearest[e_tg]],
            "tg_to_nearest_pni_km": np.round(
                d_tg_p[e_tg, tg_nearest[e_tg]], pni._KM_DIGITS
            ),
            "k_ms_per_km": THEORETICAL_SLOPE,
            "feasible_budget_km": np.round(budget_km, pni._KM_DIGITS),
            "n_pni": n_p,
            "n_feasible": res["n_feasible"],
            "feasible_share": np.round(res["n_feasible"] / n_p, pni._RATIO_DIGITS),
            "is_feasible_set_empty": empty,
            "is_feasible_set_singleton": res["is_feasible_set_singleton"],
            "is_empty_and_soi_violating": empty & soi_violating,
            "is_empty_unexplained_by_soi": empty & ~soi_violating,
            "feasible_contains_tg_nearest_pni": res["feasible_contains_tg_nearest_pni"],
            "feasible_equals_tg_nearest_pni": res["is_feasible_set_singleton"]
            & res["feasible_contains_tg_nearest_pni"],
            "feasible_pni_to_tg_km_max": np.round(
                res["feasible_pni_to_tg_km_max"], pni._KM_DIGITS
            ),
            "feasible_pni_to_tg_km_min": np.round(
                res["feasible_pni_to_tg_km_min"], pni._KM_DIGITS
            ),
            "feasible_tg_site_rank_best": res["feasible_tg_site_rank_best"],
            "feasible_tg_site_rank_worst": res["feasible_tg_site_rank_worst"],
            "feasible_tg_site_ranks": res["feasible_tg_site_ranks"],
            "feasible_tg_site_ranks_truncated": res["feasible_tg_site_ranks_truncated"],
            "feasible_ranks_are_contiguous": res["feasible_ranks_are_contiguous"],
        }
    )
    if "weight" in edges.columns:
        pairs.insert(4, "weight", edges["weight"].to_numpy(float))

    sites_tbl, site_meta = _site_table(
        sites, tg, d_tg_p, d_vp_p, e_vp, e_tg, budget_km, tg_rank, tg_nearest, sel,
        res["site_feasible_counts"], is_sping,
    )

    null = _null_table(
        vp, tg, e_vp, e_tg, budget_km, sites, res, tg_nearest, is_sping,
        trials=decoy_trials, pairwise_km=pairwise_km,
    )

    meta = _meta(
        source_label=source_label,
        pni_graph_dir=pni_graph_dir,
        graph=graph,
        pairs=pairs,
        sites_tbl=sites_tbl,
        site_meta=site_meta,
        null=null,
        n_p=n_p,
        n_vps=int(len(vp)),
        n_targets=n_tg,
        strat_edges=strat_edges,
        direct_km=direct_km,
        edges=edges,
        chunk_pairs=res["chunk_pairs"],
        decoy_trials=decoy_trials,
    )
    return PniFeasibility(pairs=pairs, sites=sites_tbl, null=null, meta=meta)


def _site_table(
    sites: pd.DataFrame,
    tg: pd.DataFrame,
    d_tg_p: np.ndarray,
    d_vp_p: np.ndarray,
    e_vp: np.ndarray,
    e_tg: np.ndarray,
    budget_km: np.ndarray,
    tg_rank: np.ndarray,
    tg_nearest: np.ndarray,
    sel: np.ndarray,
    site_counts: np.ndarray,
    is_sping: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    """One row per (target, site): how many of that target's VPs admit it.

    This is the artifact that replaces a hard intersection. `vps_feasible_share`
    is the statistic; `feasible_under_all_vps` is the intersection, emitted so
    its brittleness is a measured number rather than a claim.
    """
    n_tg, n_p = d_tg_p.shape
    n_vps = np.bincount(e_tg, minlength=n_tg)

    selecting = np.zeros((n_tg, n_p), dtype=np.int64)
    np.add.at(selecting, (e_tg, sel), 1)

    # The shortest-ping VP's own ellipse, one small pass. It is the pair the
    # headline rests on, so its per-site verdict is carried here rather than
    # being re-derived by a consumer joining on two ids.
    sping_feas = np.zeros((n_tg, n_p), dtype=bool)
    sp = np.flatnonzero(is_sping)
    if sp.size:
        legs = d_vp_p[e_vp[sp], :] + d_tg_p[e_tg[sp], :]
        sping_feas[e_tg[sp]] = legs <= budget_km[sp, None] + _FEAS_EPS_KM

    rep_tg = np.repeat(np.arange(n_tg), n_p)
    rep_p = np.tile(np.arange(n_p), n_tg)
    counts = site_counts[rep_tg, rep_p]
    per_target = n_vps[rep_tg]
    is_nearest = rep_p == tg_nearest[rep_tg]

    frame = pd.DataFrame(
        {
            "target_id": tg["target_id"].to_numpy()[rep_tg],
            "pni_id": sites["pni_id"].to_numpy()[rep_p],
            "pni_lat": sites["pni_lat"].to_numpy()[rep_p],
            "pni_lon": sites["pni_lon"].to_numpy()[rep_p],
            "pni_to_tg_km": np.round(d_tg_p[rep_tg, rep_p], pni._KM_DIGITS),
            "tg_site_rank": tg_rank[rep_tg, rep_p],
            "is_tg_nearest_pni": is_nearest,
            "n_vps": per_target,
            "n_vps_feasible": counts,
            "vps_feasible_share": np.round(
                np.where(per_target > 0, counts / np.maximum(per_target, 1), np.nan),
                pni._RATIO_DIGITS,
            ),
            "sping_vp_is_feasible": sping_feas[rep_tg, rep_p],
            "n_vps_selecting_this_pni": selecting[rep_tg, rep_p],
            "feasible_under_all_vps": (per_target > 0) & (counts == per_target),
            "feasible_under_no_vps": counts == 0,
        }
    )

    # Suppressed only when a site is admitted by no VP of that target *and* is
    # not that target's nearest. The nearest-site row always survives so a
    # downstream join on (target_id, tg_nearest_pni_id) can never miss.
    drop = frame["feasible_under_no_vps"] & ~frame["is_tg_nearest_pni"]
    kept = frame[~drop].reset_index(drop=True)

    inter = frame.groupby("target_id")["feasible_under_all_vps"].sum()
    meta = {
        "n_site_rows_before_suppression": int(len(frame)),
        "n_site_rows_suppressed_all_infeasible": int(drop.sum()),
        "n_site_rows": int(len(kept)),
        "vps_feasible_share": describe_p90(
            kept["vps_feasible_share"], digits=pni._RATIO_DIGITS
        ),
        "tg_nearest_pni_vps_feasible_share": describe_p90(
            kept.loc[kept["is_tg_nearest_pni"], "vps_feasible_share"],
            digits=pni._RATIO_DIGITS,
        ),
        "intersection": {
            "n_sites_feasible_under_all_vps": describe_p90(inter, digits=3),
            "n_targets_with_empty_intersection": int((inter == 0).sum()),
            "share_targets_with_empty_intersection": _share((inter == 0).to_numpy()),
            "note": _INTERSECTION_NOTE,
        },
        "note": (
            "One row per (target, site) that at least one of that target's VPs "
            "admits, plus the target's nearest site unconditionally."
        ),
    }
    return kept, meta


def _null_table(
    vp: pd.DataFrame,
    tg: pd.DataFrame,
    e_vp: np.ndarray,
    e_tg: np.ndarray,
    budget_km: np.ndarray,
    sites: pd.DataFrame,
    observed: dict,
    tg_nearest: np.ndarray,
    is_sping: np.ndarray,
    *,
    trials: int,
    pairwise_km,
) -> pd.DataFrame:
    """The observed row (trial -1) and one row per decoy trial, per scope."""
    rows: list[dict] = []
    obs_hit = observed["feasible_contains_tg_nearest_pni"]
    rows.append(_null_row(observed, obs_hit, trial=-1, scope="all"))
    if is_sping.any():
        rows.append(
            _null_row(_subset(observed, is_sping), obs_hit[is_sping], trial=-1, scope="sping")
        )

    if trials <= 0:
        return pd.DataFrame(rows)

    lat = sites["pni_lat"].to_numpy(float)
    lon = sites["pni_lon"].to_numpy(float)
    rng = np.random.default_rng(_DECOY_SEED)
    vlat, vlon = vp["vp_lat"].to_numpy(float), vp["vp_lon"].to_numpy(float)
    tlat, tlon = tg["target_lat"].to_numpy(float), tg["target_lon"].to_numpy(float)
    for trial in range(trials):
        dlat = rng.uniform(lat.min(), lat.max(), lat.size)
        dlon = rng.uniform(lon.min(), lon.max(), lon.size)
        dv = pairwise_km(vlat, vlon, dlat, dlon)
        dt = pairwise_km(tlat, tlon, dlat, dlon)
        near = dt.argmin(axis=1).astype(np.intp)
        res = feasibility_pass(dv, dt, e_vp, e_tg, budget_km, tg_nearest=near)
        hit = res["feasible_contains_tg_nearest_pni"]
        rows.append(_null_row(res, hit, trial=trial, scope="all"))
        if is_sping.any():
            rows.append(
                _null_row(_subset(res, is_sping), hit[is_sping], trial=trial, scope="sping")
            )
    return pd.DataFrame(rows)


def _null_block(null: pd.DataFrame, *, trials: int) -> dict:
    """Observed minus decoy, per scope, as the effect size for selectivity."""
    if trials <= 0:
        return {
            "decoy_trials": 0,
            "note": _NULL_NOTE,
            "warning": (
                "no null was computed, so n_feasible and singleton_share have no "
                "effect size and must not be quoted as evidence of selectivity"
            ),
        }
    out: dict = {"decoy_trials": int(trials), "decoy_seed": _DECOY_SEED, "note": _NULL_NOTE}
    for scope in null["scope"].unique():
        obs = null[(null["scope"] == scope) & (null["trial"] < 0)]
        dec = null[(null["scope"] == scope) & (null["trial"] >= 0)]
        if obs.empty or dec.empty:
            continue
        block = {}
        for col in ("n_feasible_mean", "singleton_share", "empty_share",
                    "singleton_equals_tg_nearest_share", "best_rank_is_1_share",
                    "feasible_pni_to_tg_km_max_p50"):
            o = float(obs[col].iloc[0])
            d = dec[col].to_numpy(float)
            d = d[np.isfinite(d)]
            block[col] = {
                "observed": round(o, 6),
                "decoy_p50": round(float(np.median(d)), 6) if d.size else float("nan"),
                "decoy_p5": round(float(np.percentile(d, 5)), 6) if d.size else float("nan"),
                "decoy_p95": round(float(np.percentile(d, 95)), 6) if d.size else float("nan"),
                # Where the observation sits in the decoy distribution.
                # Deliberately direction-neutral: "selective" means a *high*
                # singleton_share but a *low* empty_share, so a single
                # better-or-worse framing would be wrong for half these columns.
                "decoy_share_at_or_below_observed": _share(d <= o)
                if d.size
                else float("nan"),
            }
        out[scope] = block
    return out


def _meta(
    *,
    source_label: str | None,
    pni_graph_dir: Path | None,
    graph: pni.PniGraph,
    pairs: pd.DataFrame,
    sites_tbl: pd.DataFrame,
    site_meta: dict,
    null: pd.DataFrame,
    n_p: int,
    n_vps: int,
    n_targets: int,
    strat_edges: list[float],
    direct_km: np.ndarray,
    edges: pd.DataFrame,
    chunk_pairs: int,
    decoy_trials: int,
) -> dict:
    g_inputs = graph.meta.get("inputs", {})
    pni_csv = g_inputs.get("pni_csv")
    synthetic = bool(pni_csv) and "synthetic" in Path(str(pni_csv)).name.lower()

    strata = []
    for b in sorted(pairs["vp_to_tg_km_stratum"].unique()):
        sub = pairs[pairs["vp_to_tg_km_stratum"] == b]
        strata.append(
            {
                "stratum": int(b),
                "vp_to_tg_km_lo": round(float(strat_edges[int(b)]), 3),
                "vp_to_tg_km_hi": round(float(strat_edges[int(b) + 1]), 3),
                **_summary(sub, n_p),
            }
        )

    carried = edges["vp_to_tg_km"].to_numpy(float)
    disagreement = float(np.nanmax(np.abs(carried - direct_km))) if len(edges) else 0.0

    return {
        "source": source_label,
        "scope": {
            "test": "THEORETICAL_SLOPE * [d(VP,p) + d(p,TG)] <= rtt_ms",
            "geometry": "sites inside the ellipse with foci VP and TG, major axis rtt_ms / k",
            "grid": "none: no seed, cell or answer space enters this artifact",
            "policy": "no routing policy is assumed; feasibility is an exclusion, not a fit",
        },
        "inputs": {
            "pni_graph_dir": str(pni_graph_dir) if pni_graph_dir else None,
            "pni_csv": pni_csv,
            "pni_csv_sha256": _short_sha(pni_csv),
            "pni_site_list_is_synthetic": synthetic,
            "peer_asn": g_inputs.get("peer_asn"),
            "n_pni": n_p,
            "n_pairs": int(len(pairs)),
            "n_vps": n_vps,
            "n_targets": n_targets,
            "n_sping_pairs": int(pairs["is_sping_vp"].sum()),
            "weight_column_present": "weight" in pairs.columns,
        },
        "k": {
            "k_ms_per_km": THEORETICAL_SLOPE,
            "implied_km_per_ms": round(2.0 / THEORETICAL_SLOPE, 3),
            "provenance": "rtt_model.THEORETICAL_SLOPE -- 2/3 c, round trip",
            "policy": _K_NOTE,
        },
        "feasibility": _summary(pairs, n_p),
        "headline_sping_only": {
            **_summary(pairs[pairs["is_sping_vp"]], n_p),
            "note": _HEADLINE_NOTE,
        },
        "by_vp_distance_stratum": {
            "n_bins_requested": _N_VP_DISTANCE_BINS,
            "edges_km": [round(float(e), 3) for e in strat_edges],
            "binning": "bipartite.quantile_bins over d(VP,TG); edges are data-derived",
            "strata": strata,
        },
        "per_target_site_share": site_meta,
        "null_model": _null_block(null, trials=decoy_trials),
        "checks": {
            # A pair's feasible set is empty exactly when its *argmin* two-leg
            # path exceeds the RTT budget, which is `routing_inflation < 1`. So
            # this equals build-pni-graph's routing_soi_violation_share by
            # construction, computed through a completely different code path.
            # A disagreement means the two modules' geometry has drifted.
            "empty_share_vs_pni_graph_routing_soi": {
                "empty_share": _share(pairs["is_feasible_set_empty"].to_numpy(bool)),
                "pni_graph_routing_soi_violation_share": graph.meta.get("soi", {}).get(
                    "routing_soi_violation_share"
                ),
                "pni_graph_soi_violation_share": graph.meta.get("soi", {}).get(
                    "soi_violation_share"
                ),
            },
            "max_abs_km_disagreement_vs_pni_edges": round(disagreement, 9),
            "feas_eps_km": _FEAS_EPS_KM,
            "km_digits_in_pni_edges": pni._KM_DIGITS,
            "chunk_pairs": chunk_pairs,
            "chunk_cells_budget": pni._ARGMIN_CHUNK_CELLS,
            "note": (
                "Legs are recomputed from the coordinate columns, never from "
                "pni_edges.csv's km columns: those are rounded at km_digits, and "
                "5e-4 km is 5e-6 ms, enough to flip a boundary case. The "
                "disagreement above is the rounding, and is expected at or below "
                "half of 10**-km_digits."
            ),
        },
        "degenerate": {
            "n_pairs_colocated_vp_tg": int(
                (THEORETICAL_SLOPE * direct_km <= pni._IDEAL_MS_EPS).sum()
            ),
            "note": (
                "A colocated VP and target cannot violate the direct-geodesic "
                "floor, so it is excluded from is_empty_and_soi_violating and "
                "would land in is_empty_unexplained_by_soi if its set were empty."
            ),
        },
        "notes": {
            "conservative": _CONSERVATIVE_NOTE,
            "selectivity_depends_on_site_count": (
                "n_feasible is bounded by n_pni, so its granularity is 1/n_pni. "
                "Read it against null_model, never alone."
            ),
        },
    }


def _short_sha(path: str | None) -> str | None:
    """First 12 hex of the site list's sha256, or None when it is unreadable.

    Provenance rather than integrity: every number in this artifact is
    conditional on one small file, and a run whose site list changed silently
    would otherwise be indistinguishable from one that did not.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def build_for_run(
    run: RunPaths,
    *,
    analysis_root: Path | None = None,
    pni_graph: Path | None = None,
    decoy_trials: int = 0,
) -> tuple[PniFeasibility, Path]:
    """Read this run's PNI graph, run the test, and say where it belongs."""
    graph_dir = Path(pni_graph) if pni_graph else run.pni_graph_dir(root=analysis_root)
    graph = pni.load_pni_graph(graph_dir)
    result = build_feasibility(
        graph,
        decoy_trials=decoy_trials,
        source_label=f"{run.run_id}/{run.source}/{run.setup}",
        pni_graph_dir=graph_dir,
    )
    return result, run.pni_feasibility_dir(root=analysis_root)


def load_feasibility(path: Path) -> PniFeasibility:
    """Read back a written artifact, or say which command writes it."""
    path = Path(path)
    missing = [f for f in (FEASIBLE_PAIRS_CSV, FEASIBLE_SITES_CSV) if not (path / f).exists()]
    if missing:
        raise MissingArtifactError(
            f"{path} is missing {missing}; run `cli build-pni-feasibility --run-id <run>` first"
        )
    null_path = path / FEASIBLE_NULL_CSV
    meta_path = path / META_JSON
    return PniFeasibility(
        pairs=pd.read_csv(path / FEASIBLE_PAIRS_CSV),
        sites=pd.read_csv(path / FEASIBLE_SITES_CSV),
        null=pd.read_csv(null_path) if null_path.exists() else pd.DataFrame(),
        meta=json.loads(meta_path.read_text()) if meta_path.exists() else {},
    )


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("build-pni-feasibility")
    def build_pni_feasibility_cmd(
        run_id: str = typer.Option(
            None, help="Run to test. Omit with --all-runs to do every run that has a PNI graph."
        ),
        all_runs: bool = typer.Option(
            False,
            "--all-runs",
            help="Every run under --outputs-root. Safe here, unlike "
            "build-pni-graph: this reads the already-per-run pni-graph/ rather "
            "than a single --pni-csv that could only describe one peering.",
        ),
        pni_graph: Path = typer.Option(
            None,
            "--pni-graph",
            help="A pni-graph/ directory to read instead of this run's own.",
        ),
        decoy_trials: int = typer.Option(
            0,
            "--decoy-trials",
            help="Permutation null: re-run the test against N site lists of the "
            "same size drawn from the real sites' bounding box. Without it "
            "n_feasible has no effect size. Use 200 for a reportable number.",
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """Which PNI sites a pair could physically have crossed, at 2/3 c (§8.1).

        Writes feasible_pairs.csv, feasible_sites.csv, feasible_null.csv and
        meta.json into pni-feasibility/. Grid-free: no `<grid>-<resolution>`
        leaf, because no seed or answer space enters.
        """
        if all_runs == (run_id is not None):
            raise typer.BadParameter("pass exactly one of --run-id or --all-runs")
        if all_runs and pni_graph is not None:
            raise typer.BadParameter("--pni-graph cannot be combined with --all-runs")

        runs = discover_runs(outputs_root) if all_runs else [resolve_run(run_id, outputs_root)]
        for run in runs:
            result, out_dir = build_for_run(
                run,
                analysis_root=analysis_root,
                pni_graph=pni_graph,
                decoy_trials=decoy_trials,
            )
            result.write(out_dir)

            if result.meta["inputs"]["pni_site_list_is_synthetic"]:
                typer.echo(
                    "warning: the site list basename says SYNTHETIC; every "
                    "candidate-set statistic here is conditional on it.",
                    err=True,
                )
            if decoy_trials <= 0:
                typer.echo(
                    "warning: no null computed (--decoy-trials 0), so "
                    "singleton_share has no effect size.",
                    err=True,
                )
            head = result.meta["headline_sping_only"]
            typer.echo(
                f"{run.run_id}: {result.meta['inputs']['n_pairs']} pairs over "
                f"{result.meta['inputs']['n_pni']} sites; sping VPs "
                f"{head.get('singleton_equals_tg_nearest_share', float('nan')):.1%} "
                f"singleton-and-nearest -> {out_dir}"
            )
