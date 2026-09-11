"""Which site-selection rule does this peer actually use? Detected, not assumed.

`build-pni-graph` has to pick a rule for assigning each measured pair an
interconnect, and until now that rule was hardcoded to the argmin. Every number
downstream then inherited the choice silently. This module makes it a
**measurement**: it scores three candidate policies against the observed
min-RTTs, emits a verdict per target and per VP, and only names an ASN-level
strategy when the per-unit verdicts actually agree.

## The three policies, and why a rank test is the right instrument

    tg_nearest    the target is served through its own nearest site
    vp_nearest    the VP egresses at its own nearest site
    argmin        the pair crosses whichever site minimizes the two-leg path

Each implies a different predictor of min-RTT, and the implication is cleanest
in *ranks*. Under `tg_nearest`, `d(PNI,TG)` is a constant across a target's VPs,
so within that target min-RTT should order by `d(VP, tg_nearest_pni)`. Under
`vp_nearest`, `d(VP,PNI)` is a constant across a VP's targets, so within that VP
min-RTT should order by `d(vp_nearest_pni, TG)`.

Ranks rather than levels for a specific reason: **an additive per-unit constant
cannot reorder that unit's observations.** Both the constant leg and the unit's
access floor drop out of a within-unit rank correlation for free, with no floor
model and nothing to fit. The price is that the same invariance makes a rank
test blind to the constant leg's *magnitude* — it cannot distinguish
"`d(PNI,TG)` is constant" from "`d(PNI,TG)` is zero" — so the constant is a
question for an intercept, not for this module.

`air` (`d(VP,TG)`) is scored alongside as the null: the path did not detour
through any listed site. It gets a verdict name rather than being folded into
"other", because it is a real hypothesis and on a well-peered dataset it may
well be the right one.

## Abstention is by construction, not by threshold

Where all three rules pick the same site for every one of a unit's
observations, the three predictors are the *same column* and no correlation can
separate them. Such a unit is `undetermined` — not because a margin failed to
clear a cutoff, but because there was nothing to distinguish. Measured on as02:
the argmin and the target's-nearest rule agree on 45% of pairs, and only 138 of
412 targets show any separation at all, so abstention is the common case and it
is reported beside every verdict rather than under it.

`min_pairwise_predictor_rho` carries the same fact continuously: when the least
similar pair of predictors still correlates at 0.99, a verdict is arithmetic
rather than evidence.

## The held-out split, which is what makes stage 3 mean anything

A rule chosen to maximize agreement with min-RTT would make any later
"min-RTT tracks routing distance" result circular — a fit evaluated on itself.
So each target's VPs are split in half by a seeded draw: the verdict is computed
on the *detect* half, and `pair_split.csv` marks the other half `is_holdout` for
`build-pni-graph` to carry and the correlation study to score on. The verdict is
also recomputed on the holdout half and the two are compared, so the split
doubles as a stability check on the detection itself.

## An ASN-level strategy is declared only when one exists

The per-unit verdicts are emitted as a distribution. A single strategy is named
only when one verdict holds an outright **majority** of the determinate units,
because a plurality on a 40/35/25 split is not a strategy — it is evidence that
serving is heterogeneous, which is itself a finding. Otherwise the peer is
labelled `mixed`, and `build-pni-graph` falls back to the argmin: the
parameter-free rule, which is a defensible default precisely because it assumes
no policy rather than because mixedness implies it.

Command: `detect-pni-strategy`. Writes to
`outputs/analysis/v3/<run_id>/pni-strategy/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules import io, pni
from scripts.analysis.v3.modules.answer_space import elementwise_km
from scripts.analysis.v3.modules.bipartite import (
    describe_p90,
    group_corr,
    resolve_source_csv,
)
from scripts.analysis.v3.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    MissingArtifactError,
    RunPaths,
    resolve_run,
)

TARGET_VERDICTS_CSV = "target_verdicts.csv"
VP_VERDICTS_CSV = "vp_verdicts.csv"
PAIR_SPLIT_CSV = "pair_split.csv"
META_JSON = "meta.json"

#: The four predictors, in report order. `air` first because it is the null
#: every policy is scored against.
AXES = ("air", "routing_tg_nearest", "routing_vp_nearest", "routing_argmin")

#: Which policy each axis stands for when it wins. `air` names a real
#: hypothesis -- no listed site is on the path -- rather than being an "other".
VERDICT_OF_AXIS = {
    "air": "direct_no_pni",
    "routing_tg_nearest": "tg_nearest",
    "routing_vp_nearest": "vp_nearest",
    "routing_argmin": "argmin_or_mixed",
}
VERDICTS = tuple(VERDICT_OF_AXIS[a] for a in AXES)

#: A unit that cannot separate the policies: too few observations, rules that
#: never disagree, or an exact tie between the top two predictors. All three are
#: by construction rather than by threshold.
UNDETERMINED = "undetermined"
#: What an ASN is called when no verdict holds a majority.
MIXED = "mixed"

#: Minimum observations for a within-unit correlation. Matches v2's
#: `DEFAULT_SPEARMAN_MIN_PAIRS`, so a unit is underpowered by the same rule in
#: both layers. Underpowered units keep their row with NaN correlations.
_MIN_OBS = 8

#: Fixed so the detect/holdout split -- and therefore every downstream number
#: scored on it -- is reproducible across invocations and machines.
_SPLIT_SEED = 20260910

#: An ASN-level strategy needs an outright majority of determinate units, not a
#: plurality: on a 40/35/25 split no strategy is in force, and naming the 40 one
#: would turn heterogeneous serving into a false homogeneity.
_MAJORITY = 0.5

_RANK_NOTE = (
    "Correlations are within-unit and on ranks. An additive per-unit constant "
    "-- the fixed leg of the policy, and the unit's access floor -- cannot "
    "reorder that unit's observations, so both drop out with no floor model and "
    "nothing fitted. The same invariance makes the test blind to the constant "
    "leg's magnitude, which is a question for an intercept rather than for a "
    "correlation."
)
_ABSTAIN_NOTE = (
    "A unit whose three rules pick the same site for every observation has three "
    "identical predictors, so its verdict is undetermined by construction rather "
    "than by a margin failing a cutoff. Read every verdict share against "
    "n_undetermined; abstention is the common case."
)
_SPLIT_NOTE = (
    "The verdict is computed on the detect half only. Scoring a later "
    "min-RTT-versus-routing-distance study on the same pairs the rule was chosen "
    "from would be a fit evaluated on itself, so pair_split.csv marks the "
    "holdout half for build-pni-graph to carry downstream."
)
_MAJORITY_NOTE = (
    "An ASN-level strategy is named only on an outright majority of determinate "
    "units. A plurality is reported as mixed, which is a finding about "
    "heterogeneous serving rather than a failure to decide; build-pni-graph then "
    "falls back to the argmin because it assumes no policy, not because mixed "
    "implies it."
)


@dataclass(frozen=True)
class PniStrategy:
    """Per-unit verdicts, the split, and the ASN-level call."""

    target_verdicts: pd.DataFrame
    vp_verdicts: pd.DataFrame
    pair_split: pd.DataFrame
    meta: dict

    @property
    def strategy(self) -> str:
        """The ASN-level verdict, or `mixed` when no policy was established."""
        return self.meta["answer"]["strategy"]

    @property
    def use_strategy(self) -> str:
        """What to pass to `build-pni-graph --strategy`: never `mixed`."""
        return self.meta["answer"]["use_strategy"]

    def write(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.target_verdicts.to_csv(out_dir / TARGET_VERDICTS_CSV, index=False)
        self.vp_verdicts.to_csv(out_dir / VP_VERDICTS_CSV, index=False)
        self.pair_split.to_csv(out_dir / PAIR_SPLIT_CSV, index=False)
        (out_dir / META_JSON).write_text(json.dumps(self.meta, indent=2) + "\n")
        return out_dir


def policy_columns(pairs: pd.DataFrame, sites: pd.DataFrame) -> dict:
    """The four predictor columns in km, plus the per-pair site choice of each rule.

    One `site_leg_frames` build shared with the argmin, which takes the frames
    rather than rebuilding them -- at a thousand sites the second build is
    hundreds of megabytes.
    """
    frames = pni.site_leg_frames(pairs, sites)
    d_vp_p, d_tg_p = frames["d_vp_p"], frames["d_tg_p"]
    e_vp, e_tg = frames["e_vp"], frames["e_tg"]

    tg_nearest = d_tg_p.argmin(axis=1).astype(np.intp)
    vp_nearest = d_vp_p.argmin(axis=1).astype(np.intp)
    sel = pni.assign_pni(pairs, sites, frames=frames)["sel"]

    p_tg, p_vp = tg_nearest[e_tg], vp_nearest[e_vp]
    axes = {
        "air": elementwise_km(
            pairs["vp_lat"].to_numpy(float), pairs["vp_lon"].to_numpy(float),
            pairs["target_lat"].to_numpy(float), pairs["target_lon"].to_numpy(float),
        ),
        "routing_tg_nearest": d_vp_p[e_vp, p_tg] + d_tg_p[e_tg, p_tg],
        "routing_vp_nearest": d_vp_p[e_vp, p_vp] + d_tg_p[e_tg, p_vp],
        "routing_argmin": d_vp_p[e_vp, sel] + d_tg_p[e_tg, sel],
    }
    return {
        "axes": axes,
        "frames": frames,
        "site_of_rule": {"tg_nearest": p_tg, "vp_nearest": p_vp, "argmin": sel},
        # A pair whose three rules coincide cannot separate them: its three
        # routing predictors are literally the same number.
        "rules_differ": ~((p_tg == p_vp) & (p_vp == sel)),
        "tg_nearest": tg_nearest,
        "vp_nearest": vp_nearest,
    }


def half_split(e_tg: np.ndarray, *, seed: int = _SPLIT_SEED) -> np.ndarray:
    """Mark half of each target's pairs as holdout, deterministically.

    Split *within* target rather than globally, so every target contributes to
    both halves and a target with few VPs cannot land wholly on one side. The
    rank-on-a-seeded-uniform form gives a stable assignment that does not depend
    on the row order the pairs arrive in.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"t": e_tg, "u": rng.random(len(e_tg))})
    rank = frame.groupby("t")["u"].rank(method="first")
    size = frame.groupby("t")["u"].transform("size")
    return (rank > np.ceil(size / 2.0)).to_numpy()


def unit_verdicts(
    frame: pd.DataFrame, key: str, *, min_obs: int = _MIN_OBS
) -> pd.DataFrame:
    """Per-unit rank correlations for every axis, and the verdict they imply.

    `frame` carries `key`, `rtt_ms`, one column per axis, and `rules_differ`.
    Used for both groupings: keyed on `target_id` it tests the target-side
    policies, keyed on `vp_id` the VP-side one, with identical arithmetic so the
    two halves of the verdict cannot disagree for a reason other than the data.
    """
    ranks = frame.groupby(key, sort=True)[["rtt_ms", *AXES]].rank(method="average")
    ranked = pd.concat([frame[[key]], ranks.add_suffix("__rank")], axis=1)

    out = pd.DataFrame(index=sorted(frame[key].unique()))
    out.index.name = key
    out["n_obs"] = frame.groupby(key, sort=True).size()
    out["n_obs_rules_differ"] = frame.groupby(key, sort=True)["rules_differ"].sum()
    out["rules_differ_share"] = (out["n_obs_rules_differ"] / out["n_obs"]).round(
        pni._RATIO_DIGITS
    )

    for axis in AXES:
        out[f"rho_{axis}"] = group_corr(ranked, key, "rtt_ms__rank", f"{axis}__rank")
        out[f"r_{axis}"] = group_corr(frame, key, "rtt_ms", axis)

    # How distinguishable the predictors are *from each other*, independent of
    # the outcome: when even the least similar pair correlates at 0.99, a
    # winner is arithmetic rather than evidence.
    pair_rhos = [
        group_corr(ranked, key, f"{a}__rank", f"{b}__rank")
        for i, a in enumerate(AXES)
        for b in AXES[i + 1:]
    ]
    out["min_pairwise_predictor_rho"] = (
        pd.concat(pair_rhos, axis=1).abs().min(axis=1).round(pni._RATIO_DIGITS)
    )

    rho = out[[f"rho_{a}" for a in AXES]]

    # Ranked on a filled array rather than `idxmax`, which is deprecated on
    # all-NA rows and would raise.
    vals = rho.to_numpy(dtype=float)
    filled = np.where(np.isnan(vals), -np.inf, vals)
    order = np.argsort(-filled, axis=1, kind="stable")
    ranked_vals = np.take_along_axis(filled, order, axis=1)
    axis_names = np.asarray(AXES)
    with np.errstate(invalid="ignore"):
        margin = ranked_vals[:, 0] - ranked_vals[:, 1]

    out["is_underpowered"] = out["n_obs"] < min_obs
    # Three ways to abstain, all by construction and none a tuned cutoff:
    # too few observations; rules that never disagree, so the three routing
    # predictors are one column; and an exact tie between the top two, where
    # naming a winner would be reporting the sort order rather than the data.
    # The tie case is the one that fires when a target sits on its own nearest
    # site -- `air` and `routing_tg_nearest` are then literally equal, which is
    # the same collapse that makes the whole mechanism unobservable there.
    out["is_undetermined"] = (
        out["is_underpowered"]
        | (out["n_obs_rules_differ"] == 0)
        | rho.isna().all(axis=1)
        | (margin == 0.0)
    )
    undetermined = out["is_undetermined"].to_numpy(bool)

    out["winner_axis"] = pd.Series(axis_names[order[:, 0]], index=out.index).where(
        ~undetermined
    )
    out["runner_up_axis"] = pd.Series(axis_names[order[:, 1]], index=out.index).where(
        ~undetermined
    )
    out["margin_rho"] = np.where(
        undetermined | ~np.isfinite(margin), np.nan, np.round(margin, pni._RATIO_DIGITS)
    )
    out["verdict"] = np.where(
        undetermined, UNDETERMINED, [VERDICT_OF_AXIS[a] for a in axis_names[order[:, 0]]]
    )

    ratio = [c for c in out.columns if c.startswith(("rho_", "r_"))]
    out[ratio] = out[ratio].round(pni._RATIO_DIGITS)
    return out.reset_index()


def _verdict_distribution(units: pd.DataFrame) -> dict:
    """Counts and shares over determinate units, with abstention beside them."""
    n = int(len(units))
    determinate = units[units["verdict"] != UNDETERMINED]
    counts = {v: int((determinate["verdict"] == v).sum()) for v in VERDICTS}
    n_det = int(len(determinate))
    return {
        "n_units": n,
        "n_determinate": n_det,
        "n_undetermined": n - n_det,
        "undetermined_share": round((n - n_det) / n, 6) if n else float("nan"),
        "n_underpowered": int(units["is_underpowered"].sum()),
        "counts": counts,
        "shares": {v: (round(c / n_det, 6) if n_det else float("nan")) for v, c in counts.items()},
        "margin_rho": describe_p90(determinate["margin_rho"], digits=pni._RATIO_DIGITS),
        "rules_differ_share": describe_p90(
            units["rules_differ_share"], digits=pni._RATIO_DIGITS
        ),
        "min_pairwise_predictor_rho": describe_p90(
            units["min_pairwise_predictor_rho"], digits=pni._RATIO_DIGITS
        ),
        "note": _ABSTAIN_NOTE,
    }


def _declare(target_dist: dict, vp_dist: dict, stability: dict) -> dict:
    """Name a strategy only if it is both concentrated and reproducible.

    Two gates, neither with a tuned constant. **Concentration** is an outright
    majority of determinate units, because a plurality on a 40/35/25 split is
    heterogeneous serving rather than a strategy. **Reproducibility** is that the
    verdict recomputed on the holdout half beats chance agreement, where chance
    is `sum(share**2)` over the observed class distribution rather than 1/4 --
    an uneven distribution agrees with itself often by accident, and comparing
    against a flat prior would flatter the detector.

    The second gate is the one that matters in practice. A share can be a clean
    majority while every per-unit verdict rests on a margin of 0.005 in Spearman,
    which is exactly the regime the synthetic site list produces; only the split
    exposes it, because a margin has no scale of its own to be judged against.
    """
    shares = target_dist["shares"]
    top = max(shares, key=lambda v: (shares[v] if np.isfinite(shares[v]) else -1))
    top_share = shares[top]
    concentrated = bool(np.isfinite(top_share) and top_share > _MAJORITY)
    chance = float(sum(s * s for s in shares.values() if np.isfinite(s)))
    observed = stability.get("agreement_share", float("nan"))
    if chance >= 1.0:
        # Every determinate unit shares one verdict, so chance agreement is 1
        # and kappa is undefined. Perfect agreement is then the only evidence
        # available, and it is sufficient: a single unit flipping on the holdout
        # half drops `observed` below 1 while `chance` stays there.
        reproducible = bool(np.isfinite(observed) and observed >= 1.0)
    else:
        reproducible = bool(np.isfinite(observed) and observed > chance)
    declared = concentrated and reproducible
    return {
        "strategy": top if declared else MIXED,
        "declared_from": "target-side verdicts over determinate units",
        "top_verdict": top,
        "top_share": top_share,
        "majority_required": _MAJORITY,
        "is_concentrated": concentrated,
        "reproducibility": {
            "holdout_agreement_share": observed,
            "chance_agreement_share": round(chance, 6),
            "cohens_kappa": round((observed - chance) / (1 - chance), 6)
            if np.isfinite(observed) and chance < 1
            else float("nan"),
            "is_reproducible": reproducible,
        },
        "is_declared": declared,
        "fallback": None if declared else "argmin",
        "vp_side_top_verdict": max(
            vp_dist["shares"],
            key=lambda v: (vp_dist["shares"][v] if np.isfinite(vp_dist["shares"][v]) else -1),
        ),
        "vp_side_top_share": max(
            (s for s in vp_dist["shares"].values() if np.isfinite(s)), default=float("nan")
        ),
        "note": _MAJORITY_NOTE,
    }


def _split_stability(detect: pd.DataFrame, holdout: pd.DataFrame, key: str) -> dict:
    """Does the verdict survive being recomputed on the other half?"""
    a = detect.set_index(key)["verdict"]
    b = holdout.set_index(key)["verdict"].reindex(a.index)
    both = (a != UNDETERMINED) & (b != UNDETERMINED) & b.notna()
    n = int(both.sum())
    return {
        "n_determinate_in_both_halves": n,
        "n_agreeing": int((a[both] == b[both]).sum()),
        "agreement_share": round(float((a[both] == b[both]).mean()), 6) if n else float("nan"),
        "note": (
            "The verdict recomputed on the holdout half, compared to the detect "
            "half. Low agreement means the verdict is noise even where it is "
            "determinate, which no margin on a single half would reveal."
        ),
    }


def _votes(dist: dict) -> dict:
    """The tally a reader actually wants: one count per policy, plus cant_tell."""
    return {**dist["counts"], "cant_tell": dist["n_undetermined"]}


def _answer_block(v: dict) -> dict:
    """The verdict and the two checks behind it, in plain field names.

    `use_strategy` is the value to pass to `build-pni-graph --strategy`: the
    declared policy when there is one, `argmin` otherwise. Spelling it out means
    a caller never has to know that `mixed` is not a valid strategy name.
    """
    rep = v["reproducibility"]
    reasons = []
    if not v["is_concentrated"]:
        reasons.append(
            f"no policy won more than half the targets "
            f"(best was {v['top_verdict']} at {v['top_share']:.1%})"
        )
    if not rep["is_reproducible"]:
        reasons.append(
            f"the answer did not survive being recomputed on the held-out half "
            f"({rep['holdout_agreement_share']:.1%} agreement against "
            f"{rep['chance_agreement_share']:.1%} by luck)"
        )
    return {
        "strategy": v["strategy"],
        "use_strategy": v["strategy"] if v["is_declared"] else "argmin",
        "why": (
            f"{v['top_verdict']} won {v['top_share']:.1%} of the targets that "
            f"gave an answer, and the split agreed "
            f"{rep['holdout_agreement_share']:.1%} of the time against "
            f"{rep['chance_agreement_share']:.1%} by luck"
            if v["is_declared"]
            else "; ".join(reasons)
        ),
        "checks": {
            "best_guess": v["top_verdict"],
            "best_guess_share": v["top_share"],
            "needs_share_above": v["majority_required"],
            "passed_majority": v["is_concentrated"],
            "repeat_agreement": rep["holdout_agreement_share"],
            "repeat_by_luck": rep["chance_agreement_share"],
            "passed_repeat": rep["is_reproducible"],
        },
    }


def build_strategy(
    pairs: pd.DataFrame,
    sites: pd.DataFrame,
    *,
    source_label: str | None = None,
    source_csv: Path | None = None,
    pni_csv: Path | None = None,
    pni_diag: dict | None = None,
    peer_asn: int | None = None,
) -> PniStrategy:
    """Score the three policies per target and per VP, and declare or abstain."""
    pairs = pairs.reset_index(drop=True)
    sites = sites.sort_values("pni_id").reset_index(drop=True)
    if sites.empty:
        raise ValueError("--pni-csv has no sites")

    cols = policy_columns(pairs, sites)
    is_holdout = half_split(cols["frames"]["e_tg"])

    frame = pd.DataFrame(
        {
            "target_id": pairs["target_id"].to_numpy(),
            "vp_id": pairs["vp_id"].to_numpy(),
            "rtt_ms": pairs["rtt_ms"].to_numpy(float),
            "rules_differ": cols["rules_differ"],
        }
    )
    for axis in AXES:
        frame[axis] = cols["axes"][axis]

    detect = frame[~is_holdout]
    held = frame[is_holdout]
    targets = unit_verdicts(detect, "target_id")
    vps = unit_verdicts(detect, "vp_id")
    target_dist = _verdict_distribution(targets)
    vp_dist = _verdict_distribution(vps)

    pair_split = pd.DataFrame(
        {
            "vp_id": frame["vp_id"],
            "target_id": frame["target_id"],
            "is_holdout": is_holdout,
            "rules_differ": cols["rules_differ"],
        }
    )

    stability = _split_stability(targets, unit_verdicts(held, "target_id"), "target_id")
    asn_verdict = _declare(target_dist, vp_dist, stability)
    meta = {
        # Four top-level keys on purpose. `answer` is what a reader needs and
        # `votes` is the tally behind it; everything that justifies, diagnoses or
        # reproduces those two lives under `details`, so the file opens on the
        # result rather than on provenance.
        "answer": _answer_block(asn_verdict),
        "votes": {
            "by_target": _votes(target_dist),
            "by_vp": _votes(vp_dist),
            "note": (
                "One vote per target (and per VP) for the policy that best "
                "orders its min-RTTs. cant_tell means the guesses were not "
                "distinguishable for that unit, not that they tied on merit."
            ),
        },
        "inputs": {
            "source_csv": str(source_csv) if source_csv else None,
            "pni_csv": str(pni_csv) if pni_csv else None,
            "peer_asn": peer_asn,
            "n_pairs": int(len(pairs)),
            "n_vps": int(pairs["vp_id"].nunique()),
            "n_targets": int(pairs["target_id"].nunique()),
            "n_pni": int(len(sites)),
        },
        "details": {
            "scope": {
                "question": "which site-selection policy do this peer's min-RTTs order by",
                "policies": list(VERDICT_OF_AXIS.values()),
                "scoring": (
                    "Per unit: rank its observations by min-RTT, rank them by "
                    "each policy's distance, and score the agreement between the "
                    "two rankings. Highest score wins."
                ),
                "grid": "none: no seed, cell or answer space enters this artifact",
            },
            "by_target": target_dist,
            "by_vp": vp_dist,
            "asn_verdict": asn_verdict,
            "pni_file": pni_diag or {},
            "discrimination": {
                "n_pairs_rules_differ": int(cols["rules_differ"].sum()),
                "pairs_rules_differ_share": round(float(cols["rules_differ"].mean()), 6),
                "note": (
                    "The ceiling on what any comparison can resolve. Where the "
                    "rules pick the same site, their distances are one column."
                ),
            },
            "split": {
                "seed": _SPLIT_SEED,
                "scheme": "half of each target's pairs, ranked on a seeded uniform",
                "n_detect": int((~is_holdout).sum()),
                "n_holdout": int(is_holdout.sum()),
                "stability_by_target": stability,
                "note": _SPLIT_NOTE,
            },
            "notes": {"ranks": _RANK_NOTE, "abstention": _ABSTAIN_NOTE},
        },
        "source": source_label,
    }
    return PniStrategy(
        target_verdicts=targets, vp_verdicts=vps, pair_split=pair_split, meta=meta
    )


def build_for_run(
    run: RunPaths,
    *,
    pni_csv: Path,
    analysis_root: Path | None = None,
    source_csv: Path | None = None,
    peer_asn: int | None = None,
) -> tuple[PniStrategy, Path]:
    """Resolve this run's inputs, detect, and say where the result belongs."""
    from scripts.benchmark.v2.eval_source import build_pairs, load_canonical_csv

    csv_path = resolve_source_csv(run, source_csv)
    pairs = build_pairs(load_canonical_csv(csv_path))
    sites, diag = pni.load_pni_sites(pni_csv)
    asn, asn_diag = pni.resolve_peer_asn(pairs, sites, peer_asn)

    result = build_strategy(
        pairs,
        sites,
        source_label=f"{run.run_id}/{run.source}/{run.setup}",
        source_csv=csv_path,
        pni_csv=Path(pni_csv),
        pni_diag={**diag, **asn_diag},
        peer_asn=asn,
    )
    return result, run.pni_strategy_dir(root=analysis_root)


def load_strategy(path: Path) -> PniStrategy:
    """Read back a written artifact, or say which command writes it."""
    path = Path(path)
    missing = [
        f for f in (TARGET_VERDICTS_CSV, VP_VERDICTS_CSV, PAIR_SPLIT_CSV)
        if not (path / f).exists()
    ]
    if missing:
        raise MissingArtifactError(
            f"{path} is missing {missing}; run `cli detect-pni-strategy --run-id "
            f"<run> --pni-csv <sites.csv>` first"
        )
    meta_path = path / META_JSON
    return PniStrategy(
        target_verdicts=pd.read_csv(path / TARGET_VERDICTS_CSV),
        vp_verdicts=pd.read_csv(path / VP_VERDICTS_CSV),
        pair_split=pd.read_csv(path / PAIR_SPLIT_CSV),
        meta=json.loads(meta_path.read_text()) if meta_path.exists() else {},
    )


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("detect-pni-strategy")
    def detect_pni_strategy_cmd(
        run_id: str = typer.Option(
            ..., help="Run to detect for. One run is one peer ASN, so there is no --all-runs."
        ),
        pni_csv: Path = typer.Option(
            ...,
            "--pni-csv",
            exists=True,
            dir_okay=False,
            help="Interconnect sites for this run's peer ASN, as build-pni-graph takes them.",
        ),
        peer_asn: int = typer.Option(
            None, "--peer-asn", help="Assert the peer ASN when the source CSV has no target_asn."
        ),
        source_csv: Path = typer.Option(
            None, help="Canonical CSV. Defaults to the path the run's eval_stats.json records."
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """Detect which site-selection policy this peer's min-RTTs order by (§7.3).

        Writes target_verdicts.csv, vp_verdicts.csv, pair_split.csv and
        meta.json into pni-strategy/. Feed `asn_verdict.strategy` to
        `build-pni-graph --strategy` and `pair_split.csv` to its `--split-csv`,
        so the correlation study downstream is scored on held-out pairs.
        """
        run = resolve_run(run_id, outputs_root)
        result, out_dir = build_for_run(
            run,
            pni_csv=pni_csv,
            analysis_root=analysis_root,
            source_csv=source_csv,
            peer_asn=peer_asn,
        )
        result.write(out_dir)

        answer = result.meta["answer"]
        votes = result.meta["votes"]["by_target"]
        if result.strategy == MIXED:
            typer.echo(f"note: {answer['why']}.", err=True)
        typer.echo(
            f"{run.run_id}: strategy={result.strategy} "
            f"(pass --strategy {result.use_strategy} to build-pni-graph); "
            f"{sum(v for k, v in votes.items() if k != 'cant_tell')} targets "
            f"answered, {votes['cant_tell']} could not -> {out_dir}"
        )
