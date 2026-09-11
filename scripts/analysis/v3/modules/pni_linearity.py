"""Does min-RTT track *routing* distance better than air distance, as a number?

§7.3 claims min-RTT is a good proxy for `d(VP,PNI) + d(PNI,TG)` rather than for
`d(VP,TG)`, and §8.1 leans on that claim to explain why the RTT ranking picks a
VP near the target. Until now the evidence was two figures read side by side.
This module reduces the comparison to a small set of statistics.

## Three axes, because two would be self-fulfilling

    air                  x = d(VP, TG)                                  control
    routing_selected       x = d(VP, sel_pni) + d(sel_pni, TG)            argmin
    routing_tg_nearest   x = d(VP, tg_nearest_pni) + d(tg_nearest, TG)  guard

The third is not decoration. `sel_pni` is chosen to *minimize* the two-leg sum
per pair, so `routing_selected` is a lower envelope over the site list: it
compresses x toward `d(VP,TG)` on exactly the pairs where the sites are on the
corridor, which partly manufactures the linearity being tested. Fixing the site
per target — the target's own nearest, a constant across its VPs — makes x a
clean function of VP position, so a gain on that axis is a real one. Where the
two routing axes disagree is where the argmin's freedom was doing the work.

## Pooled r-squared is the weaker half

With ~134 VPs per target, most of the spread in the pooled cloud is targets
sitting at different access-latency levels, so a model can win on pooled
r-squared by ordering *targets* better while explaining nothing within a target.
`target_linearity.csv` therefore computes each correlation **within** a target
and reports the paired difference across targets — a paired design over 412
observations, which is much harder to argue with than two pooled numbers.

Pooled is still reported, in two forms. `delta_r2` is the raw difference, and
`residual_variance_removed = 1 - (1 - r2_num) / (1 - r2_den)` is the same fact
in the form that reads: 0.834 to 0.887 is 31.9% of the residual variance
removed, which is a great deal more legible than 0.053.

## The comparison has to be stratified

Where the target sits at a site, `d(PNI,TG) = 0` and the routing axes are
*identical* to the air axis, so the difference is zero by construction — not by
measurement. Under §7.3's own premise that is the majority of targets, so a
pooled paired difference is diluted toward nothing by the very population the
paper says dominates. The prediction is an **interaction**: no difference at
co-located targets, growing with `tg_to_nearest_pni_km`. `interaction` in
`meta.json` is that prediction as one rank correlation, computed over the
per-target values and never from the bins, which exist only for presentation.

## The OLS slope is not a speed

Reported as `implied_km_per_ms` and labelled a diagnostic, because OLS trades
slope against intercept: on as02 the routing fit's slope came out *steeper*
(0.0164 against 0.0153 ms/km) than the air fit's despite a uniformly larger x,
purely because its intercept fell 10.31 to 6.51 ms. A speed estimate needs a
low-quantile envelope with a per-target intercept, which is a separate command.
`bipartite.ols` is shared with `plot-pni-delay`, so the r-squared tabulated here
is the one that figure draws.

Command: `compare-pni-linearity`. Writes to
`outputs/analysis/v3/<run_id>/pni-linearity/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules import pni
from scripts.analysis.v3.modules.answer_space import elementwise_km
from scripts.analysis.v3.modules.bipartite import (
    describe_p90,
    group_corr,
    ols,
    quantile_bins,
    spearman,
    truthy,
)
from scripts.analysis.v3.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    MissingArtifactError,
    RunPaths,
    discover_runs,
    resolve_run,
)
from scripts.libs.cbg.rtt_model import THEORETICAL_SLOPE

FITS_CSV = "linearity_fits.csv"
COMPARE_CSV = "linearity_compare.csv"
TARGET_CSV = "target_linearity.csv"
BY_PNI_DISTANCE_CSV = "linearity_by_pni_distance.csv"
META_JSON = "meta.json"

#: The three x axes, in report order. `air` is first because it is the control
#: every other axis is scored against, and `_compare` reads that order.
AXES = ("air", "routing_selected", "routing_tg_nearest")

#: Minimum VPs for a per-target correlation. Matches v2's
#: `DEFAULT_SPEARMAN_MIN_PAIRS`, so an underpowered target is underpowered by
#: the same rule in both layers. Below it the row is emitted with NaN
#: correlations and `is_underpowered`, never dropped -- a dropped target is a
#: silent change of population, and the sparse targets are exactly the ones a
#: reader would want to know were excluded.
_MIN_VPS_FOR_TARGET_FIT = 8

#: Quantile strata for `tg_to_nearest_pni_km`. Quantile rather than fixed km for
#: `pni_feasibility._N_VP_DISTANCE_BINS`' reason, and doubly so here: the
#: stratifying variable is the one §7.3 predicts the effect scales with, so a
#: hand-picked cut point would be choosing where to find the result.
_N_PNI_DISTANCE_BINS = 4

_AXIS_NOTE = (
    "routing_selected uses sel_pni, chosen to minimize the two-leg sum per pair, "
    "so it is a lower envelope over the site list and its linearity is partly "
    "self-fulfilling. routing_tg_nearest fixes the site per target, which makes "
    "x a clean function of VP position; it is the honest routing axis and the "
    "gap between the two measures how much the argmin's freedom contributed."
)
_PAIRED_NOTE = (
    "Correlations are computed within a target and then differenced, so the "
    "between-target level effect -- targets sitting at different access-latency "
    "floors -- cannot produce a gain. A pooled r-squared can improve by ordering "
    "targets better while explaining nothing within one."
)
_INTERACTION_NOTE = (
    "Where the target sits at a site the routing and air axes are identical, so "
    "the per-target difference is zero by construction rather than by "
    "measurement. Under 7.3's premise that is most targets, so the pooled "
    "paired difference is diluted by the population the claim is about. The "
    "prediction is this rank correlation being positive, not the pooled median "
    "being large."
)
_SLOPE_NOTE = (
    "implied_km_per_ms is a diagnostic, not a calibration. OLS trades slope "
    "against intercept, so a longer x can yield a steeper slope; a speed "
    "estimate needs a low-quantile envelope with a per-target intercept."
)


@dataclass(frozen=True)
class PniLinearity:
    """The four tables and the meta block, ready to write."""

    fits: pd.DataFrame
    compare: pd.DataFrame
    targets: pd.DataFrame
    by_pni_distance: pd.DataFrame
    meta: dict

    def write(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.fits.to_csv(out_dir / FITS_CSV, index=False)
        self.compare.to_csv(out_dir / COMPARE_CSV, index=False)
        self.targets.to_csv(out_dir / TARGET_CSV, index=False)
        self.by_pni_distance.to_csv(out_dir / BY_PNI_DISTANCE_CSV, index=False)
        (out_dir / META_JSON).write_text(json.dumps(self.meta, indent=2) + "\n")
        return out_dir


def axis_columns(edges: pd.DataFrame, graph: pni.PniGraph) -> tuple[dict, np.ndarray, dict]:
    """The three x columns in km, the recomputed direct distance, and diagnostics.

    `air` is recomputed from the coordinates rather than read from
    `pni_edges.csv`, whose km columns are rounded at `pni._KM_DIGITS`; the
    disagreement is reported so the carried and recomputed forms are visibly the
    same quantity. `routing_tg_nearest` needs `d(VP, tg_nearest_pni)`, which is
    not on any artifact, so it comes from the shared site-leg matrices.
    """
    sites = graph.pni_nodes.sort_values("pni_id").reset_index(drop=True)
    frames = pni.site_leg_frames(edges, sites)
    d_vp_p, d_tg_p = frames["d_vp_p"], frames["d_tg_p"]
    e_vp, e_tg = frames["e_vp"], frames["e_tg"]

    direct_km = elementwise_km(
        edges["vp_lat"].to_numpy(float),
        edges["vp_lon"].to_numpy(float),
        edges["target_lat"].to_numpy(float),
        edges["target_lon"].to_numpy(float),
    )
    tg_nearest = d_tg_p.argmin(axis=1).astype(np.intp)
    near = tg_nearest[e_tg]
    axes = {
        "air": direct_km,
        "routing_selected": edges["vp_to_tg_via_pni_km"].to_numpy(float),
        "routing_tg_nearest": d_vp_p[e_vp, near] + d_tg_p[e_tg, near],
    }
    diag = {
        "n_sites": int(len(sites)),
        "tg_to_nearest_pni_km": d_tg_p[e_tg, near],
        "max_abs_km_disagreement_vs_pni_edges": float(
            np.nanmax(np.abs(edges["vp_to_tg_km"].to_numpy(float) - direct_km))
        )
        if len(edges)
        else 0.0,
    }
    return axes, direct_km, diag


def _fit_row(axis: str, x: np.ndarray, y: np.ndarray, n_targets: int) -> dict:
    slope, intercept, r, r2 = ols(x, y)
    ok = np.isfinite(x) & np.isfinite(y)
    resid = y[ok] - (slope * x[ok] + intercept) if np.isfinite(slope) else np.array([])
    below = y[ok] < THEORETICAL_SLOPE * x[ok]
    return {
        "axis": axis,
        "n_pairs": int(ok.sum()),
        "n_targets": int(n_targets),
        "ols_slope_ms_per_km": round(slope, 9),
        "ols_intercept_ms": round(intercept, 6),
        "implied_km_per_ms": round(2.0 / slope, 3) if np.isfinite(slope) and slope > 0 else np.nan,
        "pearson_r": round(r, 6),
        "r2": round(r2, 6),
        "residual_rmse_ms": round(float(np.sqrt((resid**2).mean())), 6) if resid.size else np.nan,
        "n_below_soi_floor": int(below.sum()),
        "share_below_soi_floor": round(float(below.mean()), 6) if below.size else np.nan,
    }


def _compare_rows(fits: pd.DataFrame) -> pd.DataFrame:
    """Each routing axis against the air control, and the two routing axes."""
    by = fits.set_index("axis")
    pairs = [
        ("routing_selected", "air"),
        ("routing_tg_nearest", "air"),
        ("routing_tg_nearest", "routing_selected"),
    ]
    rows = []
    for num, den in pairs:
        if num not in by.index or den not in by.index:
            continue
        r2n, r2d = float(by.loc[num, "r2"]), float(by.loc[den, "r2"])
        # The proportional reduction in error: what share of the residual
        # variance the denominator model leaves that the numerator removes.
        removed = 1.0 - (1.0 - r2n) / (1.0 - r2d) if np.isfinite(r2d) and r2d < 1 else np.nan
        rows.append(
            {
                "axis_num": num,
                "axis_den": den,
                "r2_num": round(r2n, 6),
                "r2_den": round(r2d, 6),
                "delta_r2": round(r2n - r2d, 6),
                "residual_variance_removed": round(removed, 6),
                "delta_rmse_ms": round(
                    float(by.loc[num, "residual_rmse_ms"] - by.loc[den, "residual_rmse_ms"]), 6
                ),
                "implied_km_per_ms_num": by.loc[num, "implied_km_per_ms"],
                "implied_km_per_ms_den": by.loc[den, "implied_km_per_ms"],
            }
        )
    return pd.DataFrame(rows)


def _target_table(
    edges: pd.DataFrame, axes: dict, tg_to_nearest: np.ndarray
) -> tuple[pd.DataFrame, list[float]]:
    """One row per target: the within-target correlations and their differences."""
    frame = pd.DataFrame({"target_id": edges["target_id"].to_numpy(), "rtt_ms": edges["rtt_ms"].to_numpy(float)})
    for axis in AXES:
        frame[axis] = axes[axis]
    # One rank column per axis plus rtt, so Pearson-on-ranks reuses group_corr
    # and therefore the same tie convention as the raw correlation.
    ranks = frame.groupby("target_id", sort=True)[["rtt_ms", *AXES]].rank(method="average")
    ranked = pd.concat([frame[["target_id"]], ranks.add_suffix("__rank")], axis=1)

    out = pd.DataFrame(index=sorted(frame["target_id"].unique()))
    out.index.name = "target_id"
    out["n_vps"] = frame.groupby("target_id", sort=True).size()
    for axis in AXES:
        out[f"r_{axis}"] = group_corr(frame, "target_id", "rtt_ms", axis)
        out[f"rho_{axis}"] = group_corr(ranked, "target_id", "rtt_ms__rank", f"{axis}__rank")

    per_target_pni_km = (
        pd.Series(tg_to_nearest, index=frame["target_id"]).groupby(level=0, sort=True).first()
    )
    out["tg_to_nearest_pni_km"] = per_target_pni_km.round(pni._KM_DIGITS)

    under = out["n_vps"] < _MIN_VPS_FOR_TARGET_FIT
    for axis in AXES:
        out.loc[under, [f"r_{axis}", f"rho_{axis}"]] = np.nan
    for short, axis in (("selected", "routing_selected"), ("tg_nearest", "routing_tg_nearest")):
        out[f"delta_r_{short}"] = out[f"r_{axis}"] - out["r_air"]
        out[f"delta_rho_{short}"] = out[f"rho_{axis}"] - out["rho_air"]
        # A "win" is a stronger *negative* correlation is not what we want: rtt
        # rises with distance, so a better axis has a *more positive* r. The
        # comparison is therefore on r itself, not on |r|.
        out[f"routing_wins_{short}"] = out[f"r_{axis}"] > out["r_air"]
    out["is_underpowered"] = under

    strat, edges_km = quantile_bins(out["tg_to_nearest_pni_km"], n_bins=_N_PNI_DISTANCE_BINS)
    out["tg_to_nearest_pni_km_stratum"] = strat.to_numpy(int)

    cols = [
        "n_vps",
        "tg_to_nearest_pni_km",
        "tg_to_nearest_pni_km_stratum",
        *[f"r_{a}" for a in AXES],
        *[f"rho_{a}" for a in AXES],
        "delta_r_selected",
        "delta_r_tg_nearest",
        "delta_rho_selected",
        "delta_rho_tg_nearest",
        "routing_wins_selected",
        "routing_wins_tg_nearest",
        "is_underpowered",
    ]
    ratio = [c for c in cols if c.startswith(("r_", "rho_", "delta_"))]
    out[ratio] = out[ratio].round(pni._RATIO_DIGITS)
    return out.reset_index()[["target_id", *cols]], edges_km


def _stratum_rows(targets: pd.DataFrame, edges_km: list[float]) -> pd.DataFrame:
    """One row per `tg_to_nearest_pni_km` stratum, plus a pooled row."""

    def block(sub: pd.DataFrame, *, stratum, kind, lo, hi) -> dict:
        row = {
            "stratum": stratum,
            "stratum_kind": kind,
            "tg_to_nearest_pni_km_lo": lo,
            "tg_to_nearest_pni_km_hi": hi,
            "n_targets": int(len(sub)),
            "tg_to_nearest_pni_km_p50": round(
                float(np.nanmedian(sub["tg_to_nearest_pni_km"])), 3
            )
            if len(sub)
            else np.nan,
            "share_underpowered": round(float(sub["is_underpowered"].mean()), 6)
            if len(sub)
            else np.nan,
        }
        for short in ("selected", "tg_nearest"):
            v = sub[f"delta_r_{short}"].to_numpy(float)
            v = v[np.isfinite(v)]
            for q in (25, 50, 75):
                row[f"delta_r_{short}_p{q}"] = (
                    round(float(np.percentile(v, q)), 6) if v.size else np.nan
                )
            w = sub.loc[~sub["is_underpowered"], f"routing_wins_{short}"]
            row[f"routing_wins_share_{short}"] = (
                round(float(w.mean()), 6) if len(w) else np.nan
            )
        return row

    rows = [
        block(targets, stratum="pooled", kind="pooled", lo=np.nan, hi=np.nan)
    ]
    for b in sorted(targets["tg_to_nearest_pni_km_stratum"].unique()):
        sub = targets[targets["tg_to_nearest_pni_km_stratum"] == b]
        rows.append(
            block(
                sub,
                stratum=int(b),
                kind="pni_distance_bin",
                lo=round(float(edges_km[int(b)]), 3),
                hi=round(float(edges_km[int(b) + 1]), 3),
            )
        )
    return pd.DataFrame(rows)


def build_linearity(
    graph: pni.PniGraph,
    *,
    source_label: str | None = None,
    pni_graph_dir: Path | None = None,
    holdout_only: bool = False,
) -> PniLinearity:
    """Fit the three axes, pool, pair by target, and stratify."""
    edges = graph.edges.reset_index(drop=True)
    if edges.empty:
        raise ValueError(f"{pni_graph_dir or 'the PNI graph'} has no edges")

    # Scoring on the pairs the site-selection rule was chosen from would make
    # "min-RTT tracks routing distance" a fit evaluated on itself, so when
    # `detect-pni-strategy` supplied a split this restricts to its holdout half.
    n_all = int(len(edges))
    if holdout_only:
        if "is_holdout" not in edges.columns:
            raise ValueError(
                "--holdout-only needs an is_holdout column; rerun build-pni-graph "
                "with --split-csv <pni-strategy/pair_split.csv>"
            )
        edges = edges[truthy(edges["is_holdout"])].reset_index(drop=True)
        if edges.empty:
            raise ValueError("--holdout-only selected no edges")

    axes, direct_km, diag = axis_columns(edges, graph)
    rtt = edges["rtt_ms"].to_numpy(float)
    n_targets = int(edges["target_id"].nunique())

    fits = pd.DataFrame([_fit_row(a, axes[a], rtt, n_targets) for a in AXES])
    compare = _compare_rows(fits)
    targets, edges_km = _target_table(edges, axes, diag["tg_to_nearest_pni_km"])
    strata = _stratum_rows(targets, edges_km)

    is_sping = (
        truthy(edges["is_sping_vp"]).to_numpy(bool)
        if "is_sping_vp" in edges.columns
        else np.zeros(len(edges), dtype=bool)
    )
    sping_fits = (
        pd.DataFrame(
            [_fit_row(a, axes[a][is_sping], rtt[is_sping], n_targets) for a in AXES]
        )
        if is_sping.any()
        else pd.DataFrame()
    )

    meta = {
        "source": source_label,
        "scope": {
            "question": "is min-RTT more linear in routing distance than in air distance",
            "estimator": "bipartite.ols, shared with plot-pni-delay so the tabulated r2 is the drawn one",
            "grid": "none: no seed, cell or answer space enters this artifact",
        },
        "inputs": {
            "pni_graph_dir": str(pni_graph_dir) if pni_graph_dir else None,
            "strategy": graph.meta.get("assignment", {}).get("strategy"),
            "holdout_only": bool(holdout_only),
            "n_pairs_before_split": n_all,
            "n_pairs": int(len(edges)),
            "n_targets": n_targets,
            "n_vps": int(edges["vp_id"].nunique()),
            "n_pni": diag["n_sites"],
            "n_sping_pairs": int(is_sping.sum()),
        },
        "axes": {
            "definitions": {
                "air": "d(VP,TG)",
                "routing_selected": "d(VP,sel_pni) + d(sel_pni,TG), under the graph's --strategy",
                "routing_tg_nearest": "d(VP,tg_nearest_pni) + d(tg_nearest_pni,TG)",
            },
            "note": _AXIS_NOTE,
        },
        "pooled": {
            "fits": {r["axis"]: {k: v for k, v in r.items() if k != "axis"}
                     for r in fits.to_dict("records")},
            "compare": {f"{r['axis_num']}_over_{r['axis_den']}":
                        {k: v for k, v in r.items() if not k.startswith("axis_")}
                        for r in compare.to_dict("records")},
            "sping_only_fits": {r["axis"]: {k: v for k, v in r.items() if k != "axis"}
                                for r in sping_fits.to_dict("records")},
            "note": _SLOPE_NOTE,
        },
        "paired": _paired_block(targets),
        "interaction": _interaction_block(targets),
        "strata": {
            "n_bins_requested": _N_PNI_DISTANCE_BINS,
            "edges_km": [round(float(e), 3) for e in edges_km],
            "binning": "bipartite.quantile_bins over tg_to_nearest_pni_km; presentation only",
            "note": _INTERACTION_NOTE,
        },
        "checks": {
            "max_abs_km_disagreement_vs_pni_edges": round(
                diag["max_abs_km_disagreement_vs_pni_edges"], 9
            ),
            "km_digits_in_pni_edges": pni._KM_DIGITS,
            "min_vps_for_target_fit": _MIN_VPS_FOR_TARGET_FIT,
            "n_targets_underpowered": int(targets["is_underpowered"].sum()),
            "note": (
                "The air axis is recomputed from coordinates; the disagreement "
                "above is pni_edges.csv's rounding and is expected at or below "
                "half of 10**-km_digits. Underpowered targets keep their row "
                "with NaN correlations rather than being dropped."
            ),
        },
        "notes": {
            "paired_design": _PAIRED_NOTE,
            "argmin_self_fulfilling": _AXIS_NOTE,
        },
    }
    return PniLinearity(
        fits=fits, compare=compare, targets=targets, by_pni_distance=strata, meta=meta
    )


def _paired_block(targets: pd.DataFrame) -> dict:
    """The paired within-target difference, which is the headline statistic."""
    out: dict = {"n_targets": int(len(targets)), "note": _PAIRED_NOTE}
    usable = targets[~targets["is_underpowered"]]
    out["n_targets_usable"] = int(len(usable))
    for short in ("selected", "tg_nearest"):
        d = usable[f"delta_r_{short}"].to_numpy(float)
        d = d[np.isfinite(d)]
        out[f"delta_r_{short}"] = describe_p90(d, digits=pni._RATIO_DIGITS)
        out[f"routing_wins_{short}"] = {
            "n_wins": int((d > 0).sum()),
            "n_losses": int((d < 0).sum()),
            "n_ties": int((d == 0).sum()),
            "share_wins": round(float((d > 0).mean()), 6) if d.size else float("nan"),
        }
    return out


def _interaction_block(targets: pd.DataFrame) -> dict:
    """The prediction as one rank correlation, computed without the bins."""
    usable = targets[~targets["is_underpowered"]]
    out: dict = {"n_targets": int(len(usable)), "note": _INTERACTION_NOTE}
    for short in ("selected", "tg_nearest"):
        out[f"spearman_delta_r_{short}_vs_tg_to_nearest_pni_km"] = round(
            spearman(usable[f"delta_r_{short}"], usable["tg_to_nearest_pni_km"]),
            pni._RATIO_DIGITS,
        )
    return out


def build_for_run(
    run: RunPaths,
    *,
    analysis_root: Path | None = None,
    pni_graph: Path | None = None,
    holdout_only: bool = False,
) -> tuple[PniLinearity, Path]:
    """Read this run's PNI graph, compare the axes, and say where it belongs."""
    graph_dir = Path(pni_graph) if pni_graph else run.pni_graph_dir(root=analysis_root)
    graph = pni.load_pni_graph(graph_dir)
    result = build_linearity(
        graph,
        source_label=f"{run.run_id}/{run.source}/{run.setup}",
        pni_graph_dir=graph_dir,
        holdout_only=holdout_only,
    )
    return result, run.pni_linearity_dir(root=analysis_root)


def load_linearity(path: Path) -> PniLinearity:
    """Read back a written artifact, or say which command writes it."""
    path = Path(path)
    missing = [f for f in (FITS_CSV, COMPARE_CSV, TARGET_CSV) if not (path / f).exists()]
    if missing:
        raise MissingArtifactError(
            f"{path} is missing {missing}; run `cli compare-pni-linearity --run-id <run>` first"
        )
    by = path / BY_PNI_DISTANCE_CSV
    meta_path = path / META_JSON
    return PniLinearity(
        fits=pd.read_csv(path / FITS_CSV),
        compare=pd.read_csv(path / COMPARE_CSV),
        targets=pd.read_csv(path / TARGET_CSV),
        by_pni_distance=pd.read_csv(by) if by.exists() else pd.DataFrame(),
        meta=json.loads(meta_path.read_text()) if meta_path.exists() else {},
    )


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("compare-pni-linearity")
    def compare_pni_linearity_cmd(
        run_id: str = typer.Option(
            None, help="Run to compare. Omit with --all-runs to do every run with a PNI graph."
        ),
        all_runs: bool = typer.Option(
            False, "--all-runs", help="Every run under --outputs-root."
        ),
        pni_graph: Path = typer.Option(
            None, "--pni-graph", help="A pni-graph/ directory to read instead of this run's own."
        ),
        holdout_only: bool = typer.Option(
            False,
            "--holdout-only",
            help="Score only the pairs `detect-pni-strategy` held out. Use this "
            "whenever the graph was built under a detected --strategy: scoring "
            "on the pairs the rule was chosen from is a fit evaluated on itself.",
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """Is min-RTT more linear in routing distance than in air distance (§7.3)?

        Writes linearity_fits.csv, linearity_compare.csv, target_linearity.csv,
        linearity_by_pni_distance.csv and meta.json into pni-linearity/.
        Grid-free: no `<grid>-<resolution>` leaf, because no seed enters.
        """
        if all_runs == (run_id is not None):
            raise typer.BadParameter("pass exactly one of --run-id or --all-runs")
        if all_runs and pni_graph is not None:
            raise typer.BadParameter("--pni-graph cannot be combined with --all-runs")

        runs = discover_runs(outputs_root) if all_runs else [resolve_run(run_id, outputs_root)]
        for run in runs:
            result, out_dir = build_for_run(
                run, analysis_root=analysis_root, pni_graph=pni_graph,
                holdout_only=holdout_only,
            )
            result.write(out_dir)
            cmp_block = result.meta["pooled"]["compare"].get("routing_selected_over_air", {})
            paired = result.meta["paired"]["routing_wins_tg_nearest"]
            typer.echo(
                f"{run.run_id}: pooled residual variance removed "
                f"{cmp_block.get('residual_variance_removed', float('nan')):.1%} "
                f"(selected over air); per-target routing wins "
                f"{paired['share_wins']:.1%} of {result.meta['paired']['n_targets_usable']} "
                f"targets -> {out_dir}"
            )
