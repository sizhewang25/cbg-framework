"""The tripartite (VP, PNI, target) graph: which interconnect a measured pair
most plausibly crossed, and what that does to its RTT.

`build-bipartite-graph` describes the dataset's geometry with RTT excluded, and
`build-proximity` lets RTT back in only to identify the shortest-ping VP. This
module adds the third node type. The operator knows where it interconnects with
a peer, and paper §7.3 argues that knowledge changes what an RTT means: a packet
from a user plane to a hypergiant server does not fly the great circle, it
crosses a private network interconnect on the way. So the distance an RTT is
evidence about is not `d(VP, TG)` but `d(VP, PNI) + d(PNI, TG)`.

## One PNI set per run

`eval_source/*_eval_stats.json` records `n_unique_target_asns == 1` on the
operator runs, so one run is one peer ASN and therefore one interconnect list.
That is why `--pni-csv` is a required per-run input and why `--all-runs` is
refused: a single file cannot describe three different peerings. `pni_asn` is a
required column, checked against the run's `target_asn` where the source CSV
still carries it, so feeding as02's sites to as01 is an error rather than a
plausible-looking artifact.

## The assignment is an argmin, not a measurement

Each measured pair is assigned the site minimizing the two-leg path,

    sel_pni = argmin_p [ d(VP, p) + d(p, TG) ]

with no traceroute and no BGP behind it. Two properties follow, and both are the
reason every emitted column says `sel_pni_` rather than `pni_`:

* It is **parameter-free**, so nothing is fitted and nothing can be tuned to
  flatter a result.
* It is a **lower bound** on the routed path, hence an upper bound on how much
  of an RTT's inflation geometry can account for.

Critically it does *not* assume the target is served through its own nearest
site. That is §7.3's "good peering" hypothesis, and defining the assignment by
it would make the hypothesis unfalsifiable. Instead the two rules are compared:
`sel_pni_is_tg_nearest` per pair, and `agreement.sel_pni_is_tg_nearest_share`
in `meta.json` as the headline number.

## Three quantities and one identity

    vp_to_tg_detour_ratio      = via / direct                    (>= 1)
    vp_to_tg_air_inflation     = rtt / (SLOPE * direct)
    vp_to_tg_routing_inflation = rtt / (SLOPE * via)

where `direct = d(VP, TG)`, `via = d(VP, PNI) + d(PNI, TG)` and `SLOPE` is
`THEORETICAL_SLOPE`, the round-trip ms per km at 2/3 c. Both inflations divide
by the same constant, so

    vp_to_tg_air_inflation == vp_to_tg_detour_ratio * vp_to_tg_routing_inflation

holds exactly, not approximately. That is the point of the module: the air
inflation v2 already computes is a *product* of a geometric term the
interconnect topology explains and a residual it does not (queueing, fixed cost
at the interconnect, non-geodesic fibre), and until now only the product was
observable. `checks.inflation_identity` reports the residual rather than
asserting it, following `proximity`'s stance that a violation is a statement
about the data and a stack trace would hide it.

**The identity is per-pair only.** The per-target minima of the three
quantities are attained by different VPs in general, so `min(air)` is not
`min(detour) * min(routing)`. No `min_..._detour_ratio` column is emitted, so
that arithmetic is not available to be done by accident.

## Air inflation and `gc_km` are carried, never recomputed

Both come from `eval_source.build_pairs`, which owns the definition and the
speed constant. Recomputing either here would put two constants behind one
name, which is the coupling `proximity._carry` exists to avoid.

## Routing SOI violations check the assignment, not the network

Routing inflation below 1 says the pair cannot have crossed the site it was
assigned — the bent path is already longer than the RTT can pay for. It is
therefore a validity check on the argmin, and a large
`soi.routing_soi_violation_share` means the metrics here should not be believed.
The air-side share is reported beside it because `detour >= 1` implies
`routing <= air`, so the routing violation set strictly contains the air one;
printing the routing number alone invites reading it as bad data.

## No grid

Nothing here reads a seed, a cell or an answer space, so the output carries no
`<grid>-<resolution>` leaf and the command declares no `--grid`/`--resolution`/
`--sweep` options. See `RunPaths.pni_graph_dir`.

## Handoff

`pni_edges.csv` is written as CSV, not parquet, so the existing `plot-pni-delay`
consumes it with no adapter:

    plot-pni-delay --csv outputs/analysis/v3/<run_id>/pni-graph/pni_edges.csv \\
        --pni-prefix sel_pni --tg-prefix target --rtt-col rtt_ms

That module reports the same quantity additively, as
`residual_ms == SLOPE * via * (routing_inflation - 1)`. Both forms are kept: the
ratio is scale-free and is what the inflation decomposition needs, the residual
is what reads correctly against the y=x floor on a scatter.

Command: `build-pni-graph`. Writes to
`outputs/analysis/v3/<run_id>/pni-graph/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules import io
from scripts.analysis.v3.modules.answer_space import pairwise_km
from scripts.analysis.v3.modules.bipartite import describe_p90, resolve_source_csv
from scripts.analysis.v3.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    MissingArtifactError,
    RunPaths,
    resolve_run,
)
from scripts.libs.cbg.rtt_model import THEORETICAL_SLOPE

PNI_EDGES_CSV = "pni_edges.csv"
TARGET_NODES_CSV = "target_nodes.csv"
PNI_NODES_CSV = "pni_nodes.csv"
META_JSON = "meta.json"

#: Cell budget for the chunked argmin, ~64 MB of float64. Sized as cells rather
#: than rows (`bipartite._CROSS_CHUNK_TARGETS` counts rows) because the second
#: axis here is the site count and is not bounded by the run: a fixed 65k-row
#: chunk would be 0.5 GB at 1,000 sites. On as01's 53,262 pairs this is a single
#: chunk at any realistic site count, so it exists for the same reason
#: bipartite's does — the sparse deployment case, not the runs in the repo.
_ARGMIN_CHUNK_CELLS = 8_000_000

#: km slack for "these two sites are equally plausible for this pair". Matches
#: `proximity._TIE_EPS`.
_TIE_EPS = 1e-9

#: Reused verbatim from `eval_source.build_pairs`'s `ideal_ms > 1e-9` guard, and
#: applied in the same ms space to *both* inflation denominators and to the
#: detour denominator. One threshold in one space is what makes the three
#: columns' NaN sets nest: `via >= direct`, so `via == 0` implies `direct == 0`
#: and all three go NaN together, leaving the identity vacuous rather than
#: violated on those rows.
_IDEAL_MS_EPS = 1e-9

#: Relative slack for snapping `vp_to_tg_detour_ratio` onto its floor of 1.
#: `via >= direct` is the triangle inequality, so any deficit is numerical: the
#: two legs and the direct distance come from three separate `arccos` calls. On
#: as01 the worst measured deficit is below 5e-7, i.e. under the 6 dp the column
#: is written at, so without this the floor would hold only by accident of
#: rounding. Snap-then-count is `bipartite.measurement_efficiency`'s stance.
_FLOOR_TOL = 1e-6

#: A ratio living just above 1 is destroyed by 3 dp. `bipartite._RATIO_DIGITS`.
_RATIO_DIGITS = 6
_KM_DIGITS = 3

_PNI_REQUIRED = ("pni_id", "pni_lat", "pni_lon", "pni_asn")
_PNI_OPTIONAL = ("pni_country", "pni_region", "pni_city", "pni_capacity")

#: The PNI file was specified as "shaped like the canonical target CSV", so a
#: `target_`-prefixed header is accepted and renamed here — one dict, one place,
#: the `io._SPING_RENAME` discipline.
_PNI_RENAME = {
    "target_id": "pni_id",
    "target_lat": "pni_lat",
    "target_lon": "pni_lon",
    "target_asn": "pni_asn",
    "target_country": "pni_country",
    "target_region": "pni_region",
    "target_city": "pni_city",
    "capacity": "pni_capacity",
}

#: The site-selection policies `build-pni-graph` can assign under. `argmin` is
#: the default because it assumes no policy; the other two are the pure rules
#: `detect-pni-strategy` scores, and are used only when that command declares
#: one. A run's meta.json records which was in force, since every number below
#: it inherits the choice.
STRATEGIES = ("argmin", "tg_nearest", "vp_nearest")

_ASSIGNMENT_RULES = {
    "argmin": "argmin_p [d(VP,p) + d(p,TG)]",
    "tg_nearest": "argmin_p d(p,TG)  -- the target's own nearest site, fixed per target",
    "vp_nearest": "argmin_p d(VP,p)  -- the VP's own nearest site, fixed per VP",
}
_ASSIGNMENT_RULE = _ASSIGNMENT_RULES["argmin"]

_ASSIGNMENT_NOTE = (
    "Geometric and parameter-free: no traceroute or BGP evidence enters. It is a "
    "lower bound on the routed path, so it upper-bounds the share of RTT "
    "inflation that topology can explain. It does not assume the target is "
    "served through its own nearest site — that is measured, as "
    "agreement.sel_pni_is_tg_nearest_share."
)

_IDENTITY_NOTE = (
    "vp_to_tg_air_inflation == vp_to_tg_detour_ratio * vp_to_tg_routing_inflation, "
    "exactly, because both inflations divide by THEORETICAL_SLOPE. Per-pair only: "
    "the per-target minima are attained by different VPs, so the identity does "
    "not survive the minimum and no min detour-ratio column is emitted."
)

_SOI_NOTE = (
    "Air-side violations are RTT faster than 2/3 c on the direct geodesic — a "
    "property of the data. Routing-side violations are RTT faster than 2/3 c on "
    "the assigned two-leg path, which is a check on the assignment: those pairs "
    "cannot have crossed the site they were assigned. Since detour >= 1 implies "
    "routing <= air, the routing violation set contains the air one and the "
    "routing share is always the larger of the two."
)


# ---- inputs -----------------------------------------------------------------


def load_pni_sites(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    """The interconnect list, plus what had to be done to it to be usable.

    Raises for anything that makes the whole artifact meaningless (no usable
    row, a missing required column, one id at two coordinates) and counts
    anything that only costs a row. The distinction follows `build_pairs`
    (NaN for a value that is undefined) against `build_bipartite` (raise when
    the inputs do not describe one graph).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise MissingArtifactError(f"{csv_path}: no such file; pass --pni-csv")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    renamed = False
    if "pni_id" not in df.columns and "target_id" in df.columns:
        df = df.rename(columns={k: v for k, v in _PNI_RENAME.items() if k in df.columns})
        renamed = True
    elif "pni_capacity" not in df.columns and "capacity" in df.columns:
        df = df.rename(columns={"capacity": "pni_capacity"})

    missing = [c for c in _PNI_REQUIRED if c not in df.columns]
    if missing:
        raise typer.BadParameter(
            f"{csv_path} is missing {missing}. --pni-csv needs "
            f"{list(_PNI_REQUIRED)} (a `target_`-prefixed header is accepted and "
            f"renamed); optional: {list(_PNI_OPTIONAL)}. Found: {list(df.columns)}"
        )

    n_read = int(len(df))
    keep = [*_PNI_REQUIRED, *(c for c in _PNI_OPTIONAL if c in df.columns)]
    df = df[keep].copy()
    df["pni_id"] = df["pni_id"].astype(str)
    for col in ("pni_lat", "pni_lon"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "pni_capacity" in df.columns:
        df["pni_capacity"] = pd.to_numeric(df["pni_capacity"], errors="coerce")

    # Dropped rather than carried: np.argmin over a row containing NaN returns
    # the NaN column's index, so one unplaceable site would silently capture
    # every pair in the dataset.
    finite = np.isfinite(df["pni_lat"].to_numpy(float)) & np.isfinite(
        df["pni_lon"].to_numpy(float)
    )
    n_dropped = int((~finite).sum())
    df = df[finite]

    n_before = len(df)
    df = df.drop_duplicates(subset=[*_PNI_REQUIRED, *(c for c in _PNI_OPTIONAL if c in df.columns)])
    n_collapsed = int(n_before - len(df))

    coords = df.groupby("pni_id")[["pni_lat", "pni_lon"]].nunique().max(axis=1)
    bad = coords[coords > 1]
    if not bad.empty:
        raise ValueError(
            f"{csv_path}: {len(bad)} pni_id(s) carry more than one coordinate "
            f"(e.g. {bad.index[:3].tolist()}); every distance here would be ambiguous"
        )

    if df.empty:
        raise ValueError(
            f"{csv_path}: no usable site rows ({n_read} read, {n_dropped} dropped "
            f"for a non-finite coordinate). The assignment is the output, so an "
            f"empty site list has no meaningful result; pass --pni-csv"
        )

    # Sorted so the argmin's first-minimum tie-break is the lexicographically
    # smallest pni_id — deterministic across runs, machines and pandas versions.
    df = df.sort_values("pni_id").reset_index(drop=True)

    n_coords = len(df.drop_duplicates(["pni_lat", "pni_lon"]))
    diag = {
        "n_pni_rows_read": n_read,
        "n_pni_rows_dropped_nan_coords": n_dropped,
        "n_pni_duplicate_rows_collapsed": n_collapsed,
        "n_pni": int(len(df)),
        "n_distinct_pni_coordinates": int(n_coords),
        "pni_columns_present": [c for c in _PNI_OPTIONAL if c in df.columns],
        "pni_renamed_from_target_prefix": renamed,
    }
    return df, diag


def resolve_peer_asn(
    pairs: pd.DataFrame, pni: pd.DataFrame, declared: int | None = None
) -> tuple[int, dict]:
    """The peer ASN this artifact is about, and how firmly it is established.

    One run is one peer, so a multi-ASN site list is refused. Where the source
    CSV still carries `target_asn` the two are cross-checked and a mismatch
    raises. On the operator runs that column was dropped by the parquet
    reconstruction (`datasets/final/*.reconstruction.json` lists it under
    `columns_unrecoverable`), so there is nothing to check against; `--peer-asn`
    asserts it explicitly and otherwise the result is recorded as unverified
    rather than silently blessed.
    """
    pni_asns = sorted({int(a) for a in pd.to_numeric(pni["pni_asn"], errors="coerce").dropna()})
    if len(pni_asns) != 1:
        raise typer.BadParameter(
            f"--pni-csv declares {len(pni_asns)} peer ASNs {pni_asns[:5]}; one run "
            f"is one peer ASN (n_unique_target_asns == 1 on the operator runs), so "
            f"split the file per ASN and run once per run_id"
        )
    pni_asn = pni_asns[0]

    run_asns: list[int] = []
    if "target_asn" in pairs.columns:
        run_asns = sorted(
            {int(a) for a in pd.to_numeric(pairs["target_asn"], errors="coerce").dropna()}
        )

    if run_asns and pni_asn not in run_asns:
        raise typer.BadParameter(
            f"--pni-csv is for AS{pni_asn}, but this run's targets are in "
            f"{[f'AS{a}' for a in run_asns[:5]]}. Refusing: the sites would be "
            f"assigned to a peering they do not belong to"
        )
    if declared is not None and declared != pni_asn:
        raise typer.BadParameter(
            f"--peer-asn {declared} contradicts --pni-csv, which declares AS{pni_asn}"
        )

    if run_asns:
        verified = "source_csv target_asn"
    elif declared is not None:
        verified = "--peer-asn (source CSV has no target_asn)"
    else:
        verified = "unverified: source CSV has no target_asn and --peer-asn was not given"

    return pni_asn, {
        "peer_asn": pni_asn,
        "peer_asn_verified_against": verified,
        "run_target_asns": run_asns,
    }


# ---- the assignment ---------------------------------------------------------


def site_leg_frames(pairs: pd.DataFrame, pni: pd.DataFrame) -> dict:
    """The two-leg distance matrices, plus the pair -> node-row index arrays.

    Factored out of `assign_pni` because three consumers need exactly these
    frames: the argmin assignment below, `pni_feasibility`'s ellipse test, and
    `pni_linearity`'s fixed-site axis. Rebuilding them per consumer would be
    cheap in time and expensive in agreement — `assign_pni`'s tie rule is a
    property of the site frame's *order*, so a second construction that sorted
    differently would silently pick different winners while every artifact still
    looked plausible.

    Both node frames are sorted by id, so the returned index arrays are stable
    under any permutation of `pairs`. `d_vp_p` is (unique VPs x sites), `d_tg_p`
    is (unique targets x sites), `d_pp` is (sites x sites); the site axis is in
    `pni`'s row order, which `load_pni_sites` and `build_pni_graph` both sort by
    `pni_id`.
    """
    vp = (
        pairs.drop_duplicates("vp_id")[["vp_id", "vp_lat", "vp_lon"]]
        .sort_values("vp_id")
        .reset_index(drop=True)
    )
    tg = (
        pairs.drop_duplicates("target_id")[["target_id", "target_lat", "target_lon"]]
        .sort_values("target_id")
        .reset_index(drop=True)
    )

    e_vp = pd.Index(vp["vp_id"]).get_indexer(pairs["vp_id"]).astype(np.intp)
    e_tg = pd.Index(tg["target_id"]).get_indexer(pairs["target_id"]).astype(np.intp)
    if e_vp.min(initial=0) < 0 or e_tg.min(initial=0) < 0:
        raise ValueError("a pair references a vp_id/target_id absent from its own frame")

    p_lat = pni["pni_lat"].to_numpy(float)
    p_lon = pni["pni_lon"].to_numpy(float)
    return {
        "vp": vp,
        "tg": tg,
        "e_vp": e_vp,
        "e_tg": e_tg,
        "d_vp_p": pairwise_km(
            vp["vp_lat"].to_numpy(float), vp["vp_lon"].to_numpy(float), p_lat, p_lon
        ),
        "d_tg_p": pairwise_km(
            tg["target_lat"].to_numpy(float), tg["target_lon"].to_numpy(float), p_lat, p_lon
        ),
        "d_pp": pairwise_km(p_lat, p_lon),
    }


def assign_pni(
    pairs: pd.DataFrame,
    pni: pd.DataFrame,
    *,
    frames: dict | None = None,
    strategy: str = "argmin",
) -> dict:
    """Assign every measured pair its argmin site; return the raw arrays.

    `frames` accepts an already-built `site_leg_frames` result, so a caller that
    needs the matrices for its own reduction as well -- `pni_strategy` scores
    three selection rules over them -- does not build them a second time. At a
    thousand sites that second build is hundreds of megabytes, not a rounding
    error.

    Chunked over pairs rather than over VP rows: the pair-indexed cost is
    `n_pairs * n_pni` cells against `n_vps * n_targets * n_pni` for the tensor
    form, which is never worse (`n_pairs <= n_vps * n_targets`), needs no second
    gather to get back to pairs, and degrades gracefully on the sparse edge sets
    a deployment has.

    `strategy` selects the rule. `argmin` is the default and the only one that
    assumes no policy; `tg_nearest` and `vp_nearest` are the pure rules, used
    when `detect-pni-strategy` has declared one. Under those two the choice is a
    per-target or per-VP constant, so no chunked scan and no tie-break arises
    and `tied` is all-False -- there is nothing to tie.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {list(STRATEGIES)}, got {strategy!r}")
    frames = frames if frames is not None else site_leg_frames(pairs, pni)
    vp, tg = frames["vp"], frames["tg"]
    e_vp, e_tg = frames["e_vp"], frames["e_tg"]
    d_vp_p, d_tg_p, d_pp = frames["d_vp_p"], frames["d_tg_p"], frames["d_pp"]

    n_e = int(len(pairs))
    n_p = int(len(pni))
    sel = np.empty(n_e, dtype=np.intp)
    best = np.empty(n_e, dtype=float)
    tied = np.zeros(n_e, dtype=bool)
    chunk = max(_ARGMIN_CHUNK_CELLS // max(n_p, 1), 1)

    tg_nearest = d_tg_p.argmin(axis=1).astype(np.intp)
    if strategy != "argmin":
        fixed = (
            tg_nearest[e_tg]
            if strategy == "tg_nearest"
            else d_vp_p.argmin(axis=1).astype(np.intp)[e_vp]
        )
        sel = fixed
        best = d_vp_p[e_vp, sel] + d_tg_p[e_tg, sel]
        return _assignment(
            vp, tg, e_vp, e_tg, sel, tied, best, d_vp_p, d_tg_p, d_pp,
            tg_nearest, chunk, strategy,
        )

    for start in range(0, n_e, chunk):
        stop = min(start + chunk, n_e)
        cost = d_vp_p[e_vp[start:stop], :] + d_tg_p[e_tg[start:stop], :]
        # First site *within `_TIE_EPS` of* the row minimum, not `argmin`. The
        # two legs are gathered from two separately built matrices, so a genuine
        # tie -- a VP and a target that both sit at sites, where both routes are
        # the direct distance -- can differ in the last ULP and let float noise
        # pick the winner. `argmin` would then contradict the documented rule
        # and reorder under an input permutation.
        within = cost <= cost.min(axis=1, keepdims=True) + _TIE_EPS
        j = within.argmax(axis=1)
        rows = np.arange(stop - start)
        sel[start:stop] = j
        best[start:stop] = cost[rows, j]
        tied[start:stop] = within.sum(axis=1) > 1
        del cost, within

    return _assignment(
        vp, tg, e_vp, e_tg, sel, tied, best, d_vp_p, d_tg_p, d_pp,
        tg_nearest, chunk, strategy,
    )


def _assignment(
    vp, tg, e_vp, e_tg, sel, tied, best, d_vp_p, d_tg_p, d_pp, tg_nearest, chunk, strategy
) -> dict:
    """The return contract, shared so every strategy emits the same keys."""
    return {
        "vp": vp,
        "tg": tg,
        "e_vp": e_vp,
        "e_tg": e_tg,
        "sel": sel,
        "tied": tied,
        "via_km": best,
        "vp_to_sel_pni_km": d_vp_p[e_vp, sel],
        "sel_pni_to_tg_km": d_tg_p[e_tg, sel],
        "tg_nearest": tg_nearest,
        "tg_nearest_km": d_tg_p[np.arange(len(tg)), tg_nearest],
        "sel_pni_to_tg_nearest_pni_km": d_pp[sel, tg_nearest[e_tg]],
        "chunk_pairs": int(chunk),
        "strategy": strategy,
    }


def compute_ratios(rtt_ms: np.ndarray, direct_km: np.ndarray, via_km: np.ndarray) -> dict:
    """The detour ratio and the two inflations, sharing one guard.

    `air` is *not* computed here — it is carried from `build_pairs`. It is
    recomputed only inside the identity check, against the carried column, which
    is how a drift in either constant becomes visible.
    """
    ideal_direct = THEORETICAL_SLOPE * direct_km
    ideal_via = THEORETICAL_SLOPE * via_km
    ok_direct = ideal_direct > _IDEAL_MS_EPS
    ok_via = ideal_via > _IDEAL_MS_EPS
    detour = np.where(ok_direct, via_km / np.where(ok_direct, direct_km, 1.0), np.nan)
    deficit = 1.0 - detour
    snap = np.isfinite(detour) & (deficit > 0) & (deficit <= _FLOOR_TOL)
    detour = np.where(snap, 1.0, detour)

    return {
        "vp_to_tg_detour_ratio": detour,
        "n_detour_snapped_to_floor": int(snap.sum()),
        "vp_to_tg_routing_inflation": np.where(ok_via, rtt_ms / np.where(ok_via, ideal_via, 1.0), np.nan),
        "n_pairs_colocated_vp_tg": int((~ok_direct).sum()),
        "n_pairs_zero_via_pni": int((~ok_via).sum()),
        "n_air_defined": int(ok_direct.sum()),
        "n_routing_defined": int(ok_via.sum()),
    }


# ---- node tables ------------------------------------------------------------


def _target_nodes(edges: pd.DataFrame, assignment: dict, pni: pd.DataFrame) -> pd.DataFrame:
    """One row per target. Every target has degree >= 1 by construction.

    Targets are derived from the pair frame rather than from an answer space, so
    an unmeasured target cannot appear — unlike `proximity`, which keeps those
    rows deliberately. That is why no coverage caveat is needed on these
    denominators.
    """
    tg = assignment["tg"].copy()
    tg["tg_nearest_pni_id"] = pni["pni_id"].to_numpy()[assignment["tg_nearest"]]
    tg["tg_to_nearest_pni_km"] = assignment["tg_nearest_km"]

    g = edges.groupby("target_id", sort=True)
    agg = pd.DataFrame(
        {
            "degree_to_vp": g["vp_id"].size(),
            "n_distinct_sel_pni": g["sel_pni_id"].nunique(),
            "tg_sel_pni_is_nearest_share": g["sel_pni_is_tg_nearest"].mean(),
            "min_vp_to_tg_km": g["vp_to_tg_km"].min(),
            "min_vp_to_tg_via_pni_km": g["vp_to_tg_via_pni_km"].min(),
            "min_vp_to_tg_air_inflation": g["vp_to_tg_air_inflation"].min(),
            "min_vp_to_tg_routing_inflation": g["vp_to_tg_routing_inflation"].min(),
        }
    )

    # Modal site, broken on (-count, sel_pni_id) so it is lexicographic rather
    # than pandas-version-dependent as `value_counts().idxmax()` would be.
    counts = (
        edges.groupby(["target_id", "sel_pni_id"], sort=True)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["target_id", "n", "sel_pni_id"], ascending=[True, False, True])
    )
    modal = counts.drop_duplicates("target_id").set_index("target_id")
    agg["tg_modal_sel_pni_id"] = modal["sel_pni_id"]
    agg["tg_modal_sel_pni_n_pairs"] = modal["n"]
    agg["tg_modal_sel_pni_share"] = modal["n"] / agg["degree_to_vp"]

    # The shortest-ping VP's own edge, lifted to the target row. Carried here
    # rather than left to a join because the RTT-vs-two-leg-delay scatter is a
    # per-target figure: one point per target, drawn on the VP the baseline
    # actually picked, not on all of a target's VPs.
    if "is_sping_vp" in edges.columns:
        sping = edges[edges["is_sping_vp"]].drop_duplicates("target_id").set_index("target_id")
        rename = {
            "vp_id": "sping_vp_id",
            "rtt_ms": "sping_vp_rtt_ms",
            "sel_pni_id": "sping_vp_sel_pni_id",
            "vp_to_tg_km": "sping_vp_to_tg_km",
            "vp_to_tg_via_pni_km": "sping_vp_to_tg_via_pni_km",
            "vp_to_tg_detour_ratio": "sping_vp_to_tg_detour_ratio",
            "vp_to_tg_air_inflation": "sping_vp_to_tg_air_inflation",
            "vp_to_tg_routing_inflation": "sping_vp_to_tg_routing_inflation",
            "sel_pni_is_tg_nearest": "sping_vp_sel_pni_is_tg_nearest",
        }
        for src, dst in rename.items():
            agg[dst] = sping[src]

    out = tg.merge(agg, left_on="target_id", right_index=True, how="left")
    return out.reset_index(drop=True)


def _pni_nodes(edges: pd.DataFrame, assignment: dict, pni: pd.DataFrame) -> pd.DataFrame:
    """One row per site, including sites nothing selected.

    Per-site distributions are four scalars on the CSV rather than a JSON block
    per site: one block per site is unreadable at any realistic site count, and
    this CSV *is* the per-site artifact — the same split `bipartite` makes
    between `vp_nodes.csv` and `meta.json`. A never-selected site keeps its row
    with NaN quantiles, because it is part of the denominator that
    `n_selected_at_least_once` is taken over.
    """
    out = pni.copy()
    g = edges.groupby("sel_pni_id", sort=True)
    stats = pd.DataFrame(
        {
            "n_pairs_selecting": g.size(),
            "n_targets_selecting": g["target_id"].nunique(),
            "n_vps_selecting": g["vp_id"].nunique(),
        }
    )
    for col, label in (
        ("vp_to_sel_pni_km", "vp_to_sel_pni_km"),
        ("sel_pni_to_tg_km", "sel_pni_to_tg_km"),
    ):
        q = g[col].quantile([0.0, 0.5, 0.9, 1.0]).unstack()
        q.columns = [f"{label}_{n}" for n in ("min", "p50", "p90", "max")]
        stats = stats.join(q)

    nearest_counts = (
        pd.Series(pni["pni_id"].to_numpy()[assignment["tg_nearest"]])
        .value_counts()
        .rename("n_targets_nearest")
    )

    out = out.merge(stats, left_on="pni_id", right_index=True, how="left")
    out = out.merge(nearest_counts, left_on="pni_id", right_index=True, how="left")
    for col in ("n_pairs_selecting", "n_targets_selecting", "n_vps_selecting", "n_targets_nearest"):
        out[col] = out[col].fillna(0).astype(int)
    return out.reset_index(drop=True)


# ---- the artifact -----------------------------------------------------------


@dataclass(frozen=True)
class PniGraph:
    edges: pd.DataFrame
    target_nodes: pd.DataFrame
    pni_nodes: pd.DataFrame
    meta: dict

    def write(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.edges.to_csv(out_dir / PNI_EDGES_CSV, index=False)
        self.target_nodes.to_csv(out_dir / TARGET_NODES_CSV, index=False)
        self.pni_nodes.to_csv(out_dir / PNI_NODES_CSV, index=False)
        (out_dir / META_JSON).write_text(json.dumps(self.meta, indent=2) + "\n")
        return out_dir


def build_pni_graph(
    pairs: pd.DataFrame,
    pni: pd.DataFrame,
    *,
    strategy: str = "argmin",
    pair_split: pd.DataFrame | None = None,
    source_label: str | None = None,
    source_csv: Path | None = None,
    pni_csv: Path | None = None,
    pni_diag: dict | None = None,
    peer_asn: int | None = None,
    sping: pd.DataFrame | None = None,
) -> PniGraph:
    """The pure core: an already-loaded `build_pairs` frame and site list in.

    `pairs` must carry `gc_km` and `inflation`, i.e. it must have come through
    `eval_source.build_pairs`, because those two columns are carried rather than
    recomputed.
    """
    for col in ("gc_km", "inflation", "rtt_ms"):
        if col not in pairs.columns:
            raise ValueError(
                f"`pairs` is missing {col!r}; pass the frame from "
                f"eval_source.build_pairs so gc_km/inflation keep one definition"
            )

    _, asn_diag = resolve_peer_asn(pairs, pni, peer_asn)
    # Sorted here and not only in `load_pni_sites`, because the tie rule is
    # "lowest pni_id among tied sites" and that is a property of the frame's
    # order. Trusting the caller to have sorted would make the assignment depend
    # on how the site list happened to arrive.
    pni = pni.sort_values("pni_id").reset_index(drop=True)
    a = assign_pni(pairs, pni, strategy=strategy)

    rtt = pairs["rtt_ms"].to_numpy(float)
    direct = pairs["gc_km"].to_numpy(float)
    air = pairs["inflation"].to_numpy(float)
    r = compute_ratios(rtt, direct, a["via_km"])

    sel_ids = pni["pni_id"].to_numpy()[a["sel"]]
    edges = pd.DataFrame(
        {
            "vp_id": pairs["vp_id"].to_numpy(),
            "vp_lat": pairs["vp_lat"].to_numpy(float),
            "vp_lon": pairs["vp_lon"].to_numpy(float),
            "target_id": pairs["target_id"].to_numpy(),
            "target_lat": pairs["target_lat"].to_numpy(float),
            "target_lon": pairs["target_lon"].to_numpy(float),
            "rtt_ms": rtt,
            "sel_pni_id": sel_ids,
            "sel_pni_lat": pni["pni_lat"].to_numpy()[a["sel"]],
            "sel_pni_lon": pni["pni_lon"].to_numpy()[a["sel"]],
        }
    )
    if "weight" in pairs.columns:
        edges["weight"] = pairs["weight"].to_numpy(float)
    for col in _PNI_OPTIONAL:
        if col in pni.columns:
            edges[f"sel_{col}"] = pni[col].to_numpy()[a["sel"]]

    edges["vp_to_sel_pni_km"] = a["vp_to_sel_pni_km"]
    edges["sel_pni_to_tg_km"] = a["sel_pni_to_tg_km"]
    edges["vp_to_tg_via_pni_km"] = a["via_km"]
    edges["vp_to_tg_km"] = direct
    edges["vp_to_tg_detour_ratio"] = r["vp_to_tg_detour_ratio"]
    edges["vp_to_tg_air_inflation"] = air
    edges["vp_to_tg_routing_inflation"] = r["vp_to_tg_routing_inflation"]
    edges["tg_nearest_pni_id"] = pni["pni_id"].to_numpy()[a["tg_nearest"]][a["e_tg"]]
    edges["sel_pni_to_tg_nearest_pni_km"] = a["sel_pni_to_tg_nearest_pni_km"]
    # Identity of the site, not proximity of it. A distance of 0 with differing
    # ids means two records for one facility, which `sel_pni_to_tg_nearest_pni_km`
    # shows as harmless; folding that into the flag would report agreement the
    # argmin did not actually reach.
    edges["sel_pni_is_tg_nearest"] = edges["sel_pni_id"].to_numpy() == edges[
        "tg_nearest_pni_id"
    ].to_numpy()
    edges["sel_pni_is_tied"] = a["tied"]

    # The identity of the baseline's VP comes from `io.load_sping_vp`, the same
    # reader `classify` and `proximity` use, rather than from re-minimizing RTT
    # here -- a second derivation would break ties differently and the flag
    # would stop naming the VP the baseline is actually scored on.
    if pair_split is not None:
        # Carried rather than recomputed so the split has exactly one
        # definition. Every study downstream that must score on held-out pairs
        # reads this column, and `plot-pni-delay --where is_holdout` works on it
        # unchanged.
        want = {
            (str(v), str(t)): bool(h)
            for v, t, h in zip(
                pair_split["vp_id"], pair_split["target_id"], pair_split["is_holdout"]
            )
        }
        missing = 0
        flags = []
        for v, t in zip(edges["vp_id"], edges["target_id"]):
            key = (str(v), str(t))
            if key not in want:
                missing += 1
            flags.append(want.get(key, False))
        if missing:
            raise ValueError(
                f"--split-csv covers {len(want)} pairs but {missing} of this run's "
                f"{len(edges)} edges are absent from it; a partial split would "
                f"silently re-denominate every study scored on it"
            )
        edges["is_holdout"] = flags

    if sping is not None:
        want = set(
            zip(sping["target_id"].astype(str), sping["sping_vp_id"].astype(str))
        )
        edges["is_sping_vp"] = [
            (str(t), str(v)) in want
            for t, v in zip(edges["target_id"], edges["vp_id"])
        ]

    # The identity, on unrounded values, over rows where all three are defined.
    detour = r["vp_to_tg_detour_ratio"]
    routing = r["vp_to_tg_routing_inflation"]
    both = np.isfinite(air) & np.isfinite(detour) & np.isfinite(routing) & (air != 0)
    residual = (
        float(np.max(np.abs((detour[both] * routing[both] - air[both]) / air[both])))
        if both.any()
        else float("nan")
    )

    target_nodes = _target_nodes(edges, a, pni)
    pni_nodes = _pni_nodes(edges, a, pni)

    n_pairs = int(len(edges))
    agree = edges["sel_pni_is_tg_nearest"].to_numpy()
    meta = {
        "source": source_label,
        "scope": {
            "assignment": _ASSIGNMENT_RULE,
            "rtt": "one min-RTT per (vp, target) pair, from eval_source.build_pairs",
            "grid": "none: no seed, cell or answer space enters this artifact",
            "capacity": "carried as a column; no derived statistic",
        },
        "inputs": {
            "source_csv": str(source_csv) if source_csv else None,
            "pni_csv": str(pni_csv) if pni_csv else None,
            "n_pairs": n_pairs,
            "n_vps": int(len(a["vp"])),
            "n_targets": int(len(a["tg"])),
            "weight_column_present": "weight" in pairs.columns,
            "n_sping_edges_flagged": int(edges["is_sping_vp"].sum())
            if "is_sping_vp" in edges.columns
            else None,
            **(pni_diag or {}),
            **asn_diag,
        },
        "assignment": {
            "strategy": strategy,
            "rule": _ASSIGNMENT_RULES[strategy],
            "strategy_source": (
                "argmin is the default and assumes no policy; a pure rule here "
                "means detect-pni-strategy declared one, and every number "
                "downstream of this artifact inherits the choice"
            ),
            "n_ties": int(edges["sel_pni_is_tied"].sum()),
            "tie_share": round(float(edges["sel_pni_is_tied"].mean()), _RATIO_DIGITS),
            "tie_eps_km": _TIE_EPS,
            "tie_rule": "lowest pni_id among tied sites; the site frame is sorted by pni_id",
            "n_pni_never_selected": int((pni_nodes["n_pairs_selecting"] == 0).sum()),
            "chunk_pairs": a["chunk_pairs"],
            "chunk_cells_budget": _ARGMIN_CHUNK_CELLS,
            "note": _ASSIGNMENT_NOTE,
        },
        "agreement": {
            "sel_pni_is_tg_nearest_share": round(float(agree.mean()), _RATIO_DIGITS),
            "n_pairs_agreeing": int(agree.sum()),
            "n_pairs": n_pairs,
            "per_target_share": describe_p90(
                target_nodes["tg_sel_pni_is_nearest_share"], digits=_RATIO_DIGITS
            ),
            "note": (
                "Share of measured pairs whose argmin site is also the target's own "
                "nearest site. This is the testable form of §7.3's good-peering "
                "claim, and it is measured rather than assumed by the assignment."
            ),
        },
        "distances": {
            name: describe_p90(edges[name], digits=_KM_DIGITS)
            for name in (
                "vp_to_sel_pni_km",
                "sel_pni_to_tg_km",
                "vp_to_tg_via_pni_km",
                "vp_to_tg_km",
                "sel_pni_to_tg_nearest_pni_km",
            )
        }
        | {"tg_to_nearest_pni_km": describe_p90(target_nodes["tg_to_nearest_pni_km"], digits=_KM_DIGITS)},
        "ratios": {
            name: describe_p90(edges[name], digits=_RATIO_DIGITS)
            for name in (
                "vp_to_tg_detour_ratio",
                "vp_to_tg_air_inflation",
                "vp_to_tg_routing_inflation",
            )
        }
        | {"note": _IDENTITY_NOTE},
        "soi": {
            "soi_violation_share": round(
                float(np.nanmean(air[np.isfinite(air)] < 1.0)) if r["n_air_defined"] else float("nan"),
                _RATIO_DIGITS,
            ),
            "routing_soi_violation_share": round(
                float(np.nanmean(routing[np.isfinite(routing)] < 1.0))
                if r["n_routing_defined"]
                else float("nan"),
                _RATIO_DIGITS,
            ),
            "n_air_defined": r["n_air_defined"],
            "n_routing_defined": r["n_routing_defined"],
            "note": _SOI_NOTE,
        },
        "degenerate": {
            "n_pairs_colocated_vp_tg": r["n_pairs_colocated_vp_tg"],
            "n_pairs_zero_via_pni": r["n_pairs_zero_via_pni"],
            "note": (
                "Undefined ratios are NaN, not inf: JSON cannot encode inf, and "
                "describe_p90 filters non-finite values, so an inf would vanish "
                "from a block without changing its n. The n of any affected block "
                "is below n_pairs by exactly these counts."
            ),
        },
        "checks": {
            "inflation_identity": {
                "max_abs_rel_residual": residual,
                "n_checked": int(both.sum()),
                "note": _IDENTITY_NOTE,
            },
            "detour_ratio_min": round(float(np.nanmin(detour)), _RATIO_DIGITS)
            if np.isfinite(detour).any()
            else float("nan"),
            "n_detour_snapped_to_floor": r["n_detour_snapped_to_floor"],
            "floor_tol": _FLOOR_TOL,
            "n_detour_below_one": int(np.sum(detour[np.isfinite(detour)] < 1.0)),
            "note": (
                "detour_ratio >= 1 is the triangle inequality, so a deficit is "
                "numerical: the two legs and the direct distance are three "
                "separate arccos calls. Deficits within floor_tol are snapped to "
                "exactly 1.0 and counted; n_detour_below_one counts what survived "
                "that, and is expected to be 0. Reported, not raised: a stack "
                "trace would hide it."
            ),
        },
        "handoff": {
            "plot_pni_delay_cmd": (
                "python -m scripts.analysis.v3.cli plot-pni-delay --csv "
                f"<out>/{PNI_EDGES_CSV} --pni-prefix sel_pni --tg-prefix target "
                "--rtt-col rtt_ms"
            ),
            "note": (
                "plot-pni-delay reports the same quantity additively, as "
                "residual_ms == THEORETICAL_SLOPE * vp_to_tg_via_pni_km * "
                "(vp_to_tg_routing_inflation - 1), and its below-floor share is "
                "routing_soi_violation_share by construction."
            ),
        },
    }

    for col in ("vp_to_sel_pni_km", "sel_pni_to_tg_km", "vp_to_tg_via_pni_km", "vp_to_tg_km",
                "sel_pni_to_tg_nearest_pni_km"):
        edges[col] = edges[col].round(_KM_DIGITS)
    for col in ("vp_to_tg_detour_ratio", "vp_to_tg_air_inflation", "vp_to_tg_routing_inflation"):
        edges[col] = edges[col].round(_RATIO_DIGITS)

    return PniGraph(edges=edges, target_nodes=target_nodes, pni_nodes=pni_nodes, meta=meta)


def build_for_run(
    run: RunPaths,
    *,
    pni_csv: Path,
    analysis_root: Path | None = None,
    source_csv: Path | None = None,
    peer_asn: int | None = None,
    strategy: str = "argmin",
    split_csv: Path | None = None,
) -> tuple[PniGraph, Path]:
    """Resolve this run's inputs, build, and say where the result belongs."""
    from scripts.benchmark.v2.eval_source import build_pairs, load_canonical_csv

    csv_path = resolve_source_csv(run, source_csv)
    pairs = build_pairs(load_canonical_csv(csv_path))
    pni, diag = load_pni_sites(pni_csv)

    # Optional: a run whose eval_source predates the shortest_ping_vp_* columns
    # still produces every other column, so this degrades rather than refuses.
    try:
        sping = io.load_sping_vp(run)
    except (MissingArtifactError, ValueError):
        sping = None

    graph = build_pni_graph(
        pairs,
        pni,
        strategy=strategy,
        pair_split=pd.read_csv(split_csv) if split_csv else None,
        source_label=f"{run.run_id}/{run.source}/{run.setup}",
        source_csv=csv_path,
        pni_csv=Path(pni_csv),
        pni_diag=diag,
        peer_asn=peer_asn,
        sping=sping,
    )
    return graph, run.pni_graph_dir(root=analysis_root)


def load_pni_graph(path: Path) -> PniGraph:
    """Read back a written artifact, or say which command writes it."""
    path = Path(path)
    missing = [f for f in (PNI_EDGES_CSV, TARGET_NODES_CSV, PNI_NODES_CSV) if not (path / f).exists()]
    if missing:
        raise MissingArtifactError(
            f"{path} is missing {missing}; run `cli build-pni-graph --run-id <run> "
            f"--pni-csv <sites.csv>` first"
        )
    meta_path = path / META_JSON
    return PniGraph(
        edges=pd.read_csv(path / PNI_EDGES_CSV),
        target_nodes=pd.read_csv(path / TARGET_NODES_CSV),
        pni_nodes=pd.read_csv(path / PNI_NODES_CSV),
        meta=json.loads(meta_path.read_text()) if meta_path.exists() else {},
    )


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("build-pni-graph")
    def build_pni_graph_cmd(
        run_id: str = typer.Option(
            ..., help="Run to describe. One run is one peer ASN, so there is no --all-runs."
        ),
        pni_csv: Path = typer.Option(
            ...,
            "--pni-csv",
            exists=True,
            dir_okay=False,
            help="Interconnect sites for this run's peer ASN: pni_id, pni_lat, "
            "pni_lon, pni_asn required; pni_country/region/city/capacity optional. "
            "A `target_`-prefixed header is accepted and renamed.",
        ),
        peer_asn: int = typer.Option(
            None,
            "--peer-asn",
            help="Assert the peer ASN when the source CSV has no target_asn to "
            "check --pni-csv against (the operator runs lost that column). A "
            "mismatch is refused either way.",
        ),
        strategy: str = typer.Option(
            "argmin",
            "--strategy",
            help="Site-selection rule: `argmin` (default, assumes no policy), "
            "`tg_nearest` or `vp_nearest`. Pass a pure rule only when "
            "`detect-pni-strategy` has declared one; it records the verdict as "
            "asn_verdict.strategy.",
        ),
        split_csv: Path = typer.Option(
            None,
            "--split-csv",
            exists=True,
            dir_okay=False,
            help="pair_split.csv from `detect-pni-strategy`. Adds an "
            "`is_holdout` column so a study downstream can be scored on pairs "
            "the strategy was not chosen from.",
        ),
        source_csv: Path = typer.Option(
            None,
            help="Canonical (vp_id, target_id, rtt_ms) CSV. Defaults to the path "
            "the run's eval_stats.json records.",
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """Assign each measured pair the PNI it most plausibly crossed (§7.3).

        Writes pni_edges.csv, target_nodes.csv, pni_nodes.csv and meta.json into
        pni-graph/. Grid-free: no `<grid>-<resolution>` leaf, because no seed or
        answer space enters.
        """
        run = resolve_run(run_id, outputs_root)
        if strategy not in STRATEGIES:
            raise typer.BadParameter(f"--strategy must be one of {list(STRATEGIES)}")
        graph, out_dir = build_for_run(
            run,
            pni_csv=pni_csv,
            analysis_root=analysis_root,
            source_csv=source_csv,
            peer_asn=peer_asn,
            strategy=strategy,
            split_csv=split_csv,
        )
        graph.write(out_dir)

        verified = graph.meta["inputs"]["peer_asn_verified_against"]
        if verified.startswith("unverified"):
            typer.echo(
                f"warning: peer ASN {graph.meta['inputs']['peer_asn']} is "
                f"{verified}. Pass --peer-asn to assert it.",
                err=True,
            )
        agreement = graph.meta["agreement"]["sel_pni_is_tg_nearest_share"]
        typer.echo(
            f"{run.run_id}: {graph.meta['inputs']['n_pairs']} pairs over "
            f"{graph.meta['inputs']['n_pni']} sites under --strategy {strategy}, "
            f"{agreement:.1%} assigned the target's nearest site -> {out_dir}"
        )
