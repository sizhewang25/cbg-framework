"""Spotter's Figure 3 normality diagnostics, on an operator mesh CSV.

Reproduces the three panels of Laki et al., *Spotter: A Model Based Active
Geolocation Service* (2011), Sec. IV and Fig. 3, and tests its two claims about
the delay-distance law `f_d(s)` -- the distribution of great-circle distance at
a fixed RTT:

1. `f_d(s) = N(mu(d), sigma(d)^2)` -- normal once standardized.
2. `f_d` is **landmark-independent**, so one pooled `(mu, sigma)` pair
   describes every landmark and no per-landmark calibration is needed.

The paper reported `mu = -0.078, sigma = 1.035` on ~40,000 PlanetLab node-pair
measurements and verified claim 2 with a Q-Q plot of five landmarks against the
pooled distribution.

## Why this command exists beside the ClickHouse one

`scripts/libs/cbg_feasibility/spotter_normality_check.py` runs the same three
panels on RIPE Atlas `ping_10k_to_anchors` and is written up in
`notes/2026-05-17-spotter-normality-check.md`. An audit of that script against
the paper found four misalignments, all of which this command fixes:

* **The Q-Q plot grouped on the wrong endpoint.** The paper's claim is about
  the landmark *from which the measurement was performed* (Sec. III, and Fig.
  3c's caption "Q-Q plot for five selected landmarks"). That script's
  `plot_panel_c` groups by `dst` -- on a probes->anchors table that is the
  target, so it tested target-independence. Here `--group-by` defaults to
  `vp_id` and the operator meshes make the VP/target roles deterministic.
* **The RTT range.** 0-200 ms there, versus the paper's 0-80 ms. The operator
  meshes top out at ~92 ms, so the regimes match without a cap.
* **The sigma estimator.** The paper's `1.035` is a least-squares Gaussian fit
  to the empirical *density* (Fig. 3b's red curve peaks at 0.386 =
  1/(1.035*sqrt(2*pi)), with its mode at z = -0.085), not a sample moment.
  That script reports moments truncated to |z| < 4 and overlays N(0,1). Here
  all three estimators are reported and panel (b) overlays both curves, as the
  paper's own figure does.
* **Silent drops.** Its `standardize()` returns a *shorter array than its
  input* with no count. On these datasets that hides 0.8-3.9% of rows -- see
  below, it is the whole finding.

## The sigma(d) <= 0 pathology

`mu(d)` and `sigma(d)` are unconstrained polynomials fitted to per-bin moments
with **equal weight per bin** regardless of bin count. On the operator meshes
the bins run 31 to 2,769 points, and the sparse leftmost bin pulls the deg-2
`sigma` polynomial negative *inside* the calibration range: roots at 5.51 ms
(as01), 2.22 (as02), 2.40 (as03). as01's `mu(d)` is negative there too
(`mu(1 ms) = -153 km`, a negative mean distance).

Rows in that interval have no usable `z`, and rows just outside it divide by a
near-zero sigma, so the untruncated `sigma_z` comes out at 6-20. That is a
property of the fit, not of the data: under a `sigma > 50 km` reference guard
all three estimators collapse to ~1.0, i.e. the pooled normal describes these
datasets about as well as it described PlanetLab.

So this command reports the affected counts and how the dropped rows differ
from the kept ones, rather than truncating them away. Note the sigma<=0 regime
this used to chart is gone: `sigma(d)` is now fitted in log space and is
positive by construction, so the `sigma_domain` diagnostic and the panel-(a)
shading that went with it were removed.

## Landmark-independence is reported as a number

At n ~ 5*10^4 every classical normality test rejects at p < 10^-100, and a
per-VP `1/sqrt(n)` sampling spread is invalid here because a VP's ~400
observations come from <= 23 distinct target coordinates and are correlated
through shared geometry. So the verdict rests on `sd_of_group_mean_z` against
an **RTT-stratified permutation null** (relabel the landmark within RTT bins,
preserving per-bin counts), with KS distances reported beside their critical
values rather than as p-values.

Command: `plot-spotter-normality`. Writes three PNGs, a manifest JSON, a
per-landmark CSV covering *every* group, and the bin fit.

Note the deliberate divergence from `plot-distance-rtt`, which is read beside
this one: there `--x-max` clips the view and every statistic is over all pairs.
Here `--rtt-max` **filters the fit population**, because the paper's
calibration set is its RTT range and a fit over a wider range is a different
model. `--view-rtt-max` clips panel (a)'s axis without touching the fit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules import config as config_mod
from scripts.analysis.v3.modules.answer_space import elementwise_km
from scripts.analysis.v3.modules.diagram.common.draw import plt
from scripts.analysis.v3.modules.diagram.common.palette import (
    _C_AXIS,
    _C_GRID,
    _C_INK,
    _C_INK_2,
    _C_MUTED,
    _VARIANT_HUES,
)
from scripts.libs.spotter.spotter_model import SpotterRTTModel, fit_mu_sigma, sigma_km

#: Round-trip ms per km at 2/3 c, the constant every CBG model here is anchored
#: on. Drawn in panel (a) as the physical floor.
THEORETICAL_SLOPE = 0.01

#: What the paper published, and the evidence for reading its (mu, sigma) as a
#: density fit rather than as sample moments. Carried into the manifest so a
#: reader never has to take the comparison on trust.
PAPER = {
    "ref": "Laki et al., Spotter: A Model Based Active Geolocation Service (2011), Fig. 3",
    "mu_z": -0.078,
    "sigma_z": 1.035,
    "n_points": 40000,
    "rtt_range_ms": [0, 80],
    "distance_max_km": 5000,
    "sigma_z_estimator": "least-squares Gaussian fit to the empirical density",
    "estimator_evidence": (
        "Fig. 3b's red curve peaks at 0.386 = 1/(1.035*sqrt(2*pi)) rather than at "
        "N(0,1)'s 0.399, with its mode at z = -0.085; digitized from "
        "papers/references/Laki et al. - 2011 - Spotter A model based active "
        "geolocation service.pdf, p. 6"
    ),
    "mu_d_shape": (
        "convex over 0-80 ms: the fitted mean rises from ~28 km/ms below 30 ms to "
        "~62 km/ms at 75 ms, reaching 3,400 km at 80 ms"
    ),
}

#: Grey for the measurement cloud, red for the fitted curve -- panel (a) and
#: (b) both follow `figure_distance_rtt`'s highlight-against-context pairing,
#: where grey reading as background is the intent.
_C_CLOUD = "#8a8a8a"
_C_FIT = "#d32f2f"

PNG_A_SUFFIX = "_spotter_fig3a_scatter.png"
PNG_B_SUFFIX = "_spotter_fig3b_standardized.png"
PNG_C_SUFFIX = "_spotter_fig3c_qq.png"
MANIFEST_SUFFIX = "_spotter_normality.json"
LANDMARK_CSV_SUFFIX = "_landmark_independence.csv"
SUMMARY_JSON = "spotter_normality_summary.json"

#: The four groupings, and which endpoint each one tests.
GROUP_BY_CHOICES = ("vp_id", "vp_coord", "target_id", "target_coord")


# ---- loading ----------------------------------------------------------------


def load_mesh(
    csv_path: Path,
    *,
    rtt_col: str = "rtt_ms",
    vp_prefix: str = "vp",
    tg_prefix: str = "target",
) -> tuple[pd.DataFrame, int]:
    """One row per measured pair, with identities kept, plus the dropped count.

    The filter matches `figure_distance_rtt.load_pairs` -- missing coordinate or
    non-positive RTT -- so the population is the same one the benchmark ran on.
    It differs from that loader in keeping `vp_id` and `target_id`, which panel
    (c) groups on and which `load_pairs` discards.

    Distance comes from `answer_space.elementwise_km` (R = 6371 km, chord
    form), which agrees with the deployed `cbg.rtt_model.haversine_distance` to
    ~1e-9 km -- so these are the distances `normal_dist` itself fits on. The
    ClickHouse arm used a 6367 km haversine.
    """
    need_num = [
        f"{vp_prefix}_lat",
        f"{vp_prefix}_lon",
        f"{tg_prefix}_lat",
        f"{tg_prefix}_lon",
        rtt_col,
    ]
    need_id = [f"{vp_prefix}_id", f"{tg_prefix}_id"]
    df = pd.read_csv(csv_path)
    missing = [c for c in need_num + need_id if c not in df.columns]
    if missing:
        raise typer.BadParameter(
            f"{csv_path} lacks {missing}; have {sorted(df.columns)[:14]}... "
            f"adjust --rtt-col / --vp-prefix / --tg-prefix"
        )
    for c in need_num:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n_before = len(df)
    df = df[df[need_num].notna().all(axis=1) & (df[rtt_col] > 0)]
    out = pd.DataFrame(
        {
            "vp_id": df[f"{vp_prefix}_id"].astype(str).to_numpy(),
            "target_id": df[f"{tg_prefix}_id"].astype(str).to_numpy(),
            "vp_lat": df[f"{vp_prefix}_lat"].to_numpy(dtype=float),
            "vp_lon": df[f"{vp_prefix}_lon"].to_numpy(dtype=float),
            "tg_lat": df[f"{tg_prefix}_lat"].to_numpy(dtype=float),
            "tg_lon": df[f"{tg_prefix}_lon"].to_numpy(dtype=float),
            "rtt_ms": df[rtt_col].to_numpy(dtype=float),
        }
    )
    out["distance_km"] = elementwise_km(
        out["vp_lat"].to_numpy(), out["vp_lon"].to_numpy(),
        out["tg_lat"].to_numpy(), out["tg_lon"].to_numpy(),
    )
    return out.reset_index(drop=True), n_before - len(out)


def csv_from_config(run_id: str) -> tuple[Path, dict]:
    """The dataset CSV declared by `configs/<run_id>.yaml`.

    Same single-sourcing argument as `figure_distance_rtt.csvs_from_config`:
    the path lives in `benchmark.source_kwargs` because that is what the
    benchmark was run on, and copying it into an `analysis:` block would create
    a second declaration free to drift.
    """
    kwargs, cfg_path = config_mod.source_kwargs_for_run(
        run_id, needed_for="the CSV to diagnose"
    )
    key = "csv_path" if "csv_path" in kwargs else "mesh_csv_path"
    if key not in kwargs:
        raise typer.BadParameter(
            f"--run-id {run_id}: {cfg_path} declares no benchmark.source_kwargs "
            f"csv path (found keys {sorted(kwargs)}). The published as0X configs "
            f"carry `benchmark: {{}}` because their CSVs were reconstructed from "
            f"run outputs -- pass --csv explicitly for those."
        )
    path = config_mod.resolve_repo_path(kwargs[key])
    if not path.exists():
        raise typer.BadParameter(f"--run-id {run_id}: {cfg_path} points at {path}, which does not exist")
    return path, {"config": str(cfg_path), "key": f"benchmark.source_kwargs.{key}"}


def discreteness(df: pd.DataFrame) -> dict:
    """How atomic the distance marginal is -- the honest limit of this test.

    `f_d(s)` is a *density* in the paper. Here distance can only take one value
    per (VP coordinate, target coordinate) pair, so conditioning on an RTT bin
    selects a subset of those atoms and the conditional law is a discrete
    mixture. The counters below quantify that: the operator meshes carry ~87 VP
    coordinates against only ~20 target coordinates, so ~1,700-2,000 distinct
    distances stand behind 53-61k observations, where the paper's ~40k
    PlanetLab pairs spanned ~10^4 distinct geometries.

    Reported rather than caveated in prose because it bounds what any normality
    verdict can mean, and because it is why the KS p-values are approximate
    (ties) and why the permutation null in `permutation_null` is required
    rather than optional (per-group rows are not iid).
    """
    vp_coord = list(zip(df["vp_lat"], df["vp_lon"]))
    tg_coord = list(zip(df["tg_lat"], df["tg_lon"]))
    n_vp_coords = len(set(vp_coord))
    n_tg_coords = len(set(tg_coord))
    pairs = pd.Series(list(zip(vp_coord, tg_coord)))
    per_pair = pairs.value_counts()
    return {
        "n_rows": int(len(df)),
        "n_vp_ids": int(df["vp_id"].nunique()),
        "n_vp_coords": n_vp_coords,
        "n_target_ids": int(df["target_id"].nunique()),
        "n_target_coords": n_tg_coords,
        "n_geometry_pairs": int(per_pair.size),
        "n_geometry_pairs_possible": int(n_vp_coords * n_tg_coords),
        "geometry_pair_coverage": round(float(per_pair.size / (n_vp_coords * n_tg_coords)), 6),
        "n_distinct_distance_km": int(df["distance_km"].round(6).nunique()),
        "obs_per_geometry_pair_p50": int(per_pair.median()),
        "obs_per_geometry_pair_max": int(per_pair.max()),
    }


# ---- the fit ----------------------------------------------------------------


@dataclass
class Fit:
    """A fitted `SpotterRTTModel` plus the rows it was fitted on.

    No bin moments. `fit_mu_sigma` fits mu to the raw pairs and sigma to its
    residuals, so there are no per-bin means for panel (a) to draw and no
    second call to recover them. The bin dots, the `*_bin_fit.csv` artifact and
    the `n_bins_used` / `bin_counts` manifest keys went with the binning.

    `fit_rtt` / `fit_dist` are the *masked* arrays the polynomials saw, kept so
    the diagnostics below score the model on its own training rows.
    """

    model: SpotterRTTModel
    kwargs: dict
    n_unphysical: int
    fit_rtt: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    fit_dist: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))


def fit_pooled(rtt: np.ndarray, dist: np.ndarray, **kwargs) -> Fit:
    """Fit the deployed model.

    Calls `scripts.libs.spotter.spotter_model.SpotterRTTModel.fit` rather than
    re-implementing it, so what this command diagnoses is the model
    `normal_dist` actually deploys. `target_coverage` is pinned to None -- not
    inherited as a default -- because that is what holds `k = 1.0`, and `k = 1`
    is what makes panel (a)'s band the paper's `mu(d) +/- sigma(d)` rather
    than a coverage-calibrated annulus. The mesh configs leave it unset too.

    The physical-validity mask is recomputed here rather than left to `fit()`,
    because `fit()` applies it internally and reports nothing, and a non-zero
    count is a data error (a bad coordinate or a sub-2/3-c RTT) that should not
    be invisible.
    """
    rtt = np.asarray(rtt, dtype=float)
    dist = np.asarray(dist, dtype=float)
    unphysical = rtt < THEORETICAL_SLOPE * dist

    model = SpotterRTTModel()
    model.fit(rtt, dist, target_coverage=None, **kwargs)
    if not model.fitted:
        raise typer.BadParameter(f"the Spotter fit failed: {model.fit_message}")

    keep = (
        np.isfinite(rtt) & np.isfinite(dist) & (rtt > 0) & (dist > 0) & ~unphysical
    )
    return Fit(
        model=model,
        kwargs=dict(kwargs),
        n_unphysical=int(unphysical.sum()),
        fit_rtt=rtt[keep],
        fit_dist=dist[keep],
    )


# ---- standardization --------------------------------------------------------

#: Why a row has no usable `z`. Kept as strings in the returned array so a call
#: site can tabulate the reasons without re-deriving them.
REASON_OK = "ok"
REASON_SIGMA_NONPOS = "sigma_nonpos"
REASON_SIGMA_BELOW_REF = "sigma_below_ref"
REASON_NONFINITE = "nonfinite"


@dataclass
class Standardized:
    """`z` aligned 1:1 with its input, NaN where invalid, with the reason why.

    The contract that matters is `len(z) == len(rtt)`. The ClickHouse arm's
    `standardize()` returns a *filtered, shorter* array, which is how a 3.9%
    drop of as01's shortest-distance rows stayed invisible through a whole
    note: every caller silently compared populations of different sizes.
    Here every consumer has to write `z[s.valid]`, and the count is one
    attribute away at each of those call sites.
    """

    z: np.ndarray
    valid: np.ndarray
    reason: np.ndarray
    diagnostics: dict


def standardize(
    rtt: np.ndarray,
    dist: np.ndarray,
    p_mu: np.ndarray,
    p_log_sigma: np.ndarray,
    *,
    sigma_ref_km: float = 0.0,
) -> Standardized:
    """`z = (s - mu(d)) / sigma(d)`, the paper's Eq. 3 rearranged.

    Uses the raw polynomial, deliberately: the flat hold above
    `SpotterRTTModel.cutoff_rtt` and the line-through-origin below `rtt_min`
    are *prediction-time* safety devices, and panel (b) is a diagnostic of
    `f_d(s)` as fitted. Rows in those regimes are counted in the manifest
    instead (`fit.n_rows_above_cutoff`).

    `sigma_ref_km` excludes rows whose fitted sigma is positive but tiny, under
    their own reason code. It is what makes the per-landmark statistics
    interpretable: a 1,700 km residual divided by a 3 km sigma produces |z| in
    the hundreds, and a handful of those rows dominate every moment computed
    downstream. It does **not** refit anything.

    `p_log_sigma` is log-space, so sigma is recovered by exponentiating.
    REASON_SIGMA_NONPOS is retained as a reason code but is now unreachable
    from the fit -- `exp` has no zero. It still fires for a hand-built model,
    which is the only way to get a non-positive sigma now.
    """
    rtt = np.asarray(rtt, dtype=float)
    dist = np.asarray(dist, dtype=float)
    mu = np.polyval(p_mu, rtt)
    with np.errstate(over="ignore"):
        sig = sigma_km(p_log_sigma, rtt)

    reason = np.full(len(rtt), REASON_OK, dtype=object)
    reason[~(sig > 0)] = REASON_SIGMA_NONPOS
    if sigma_ref_km > 0:
        reason[(sig > 0) & (sig <= sigma_ref_km)] = REASON_SIGMA_BELOW_REF

    with np.errstate(divide="ignore", invalid="ignore"):
        z = (dist - mu) / np.where(sig == 0, np.nan, sig)
    reason[(reason == REASON_OK) & ~np.isfinite(z)] = REASON_NONFINITE

    valid = reason == REASON_OK
    z = np.where(valid, z, np.nan)

    n = len(rtt)
    dropped = ~valid
    diagnostics = {
        "n_rows": n,
        "n_valid": int(valid.sum()),
        "n_sigma_nonpos": int((reason == REASON_SIGMA_NONPOS).sum()),
        "n_sigma_below_ref": int((reason == REASON_SIGMA_BELOW_REF).sum()),
        "n_nonfinite": int((reason == REASON_NONFINITE).sum()),
        "share_dropped": round(float(dropped.sum() / n), 6) if n else 0.0,
    }
    # The bias of the drop, not just its size. These rows are the short-RTT,
    # short-distance ones -- exactly the near-target pairs a geolocation system
    # is judged on -- so dropping them silently flatters every downstream
    # number.
    if dropped.any():
        diagnostics["dropped"] = {
            "rtt_ms_max": round(float(rtt[dropped].max()), 4),
            "distance_km_p50": round(float(np.median(dist[dropped])), 2),
            "distance_km_p95": round(float(np.percentile(dist[dropped], 95)), 2),
        }
        diagnostics["kept"] = {
            "distance_km_p50": round(float(np.median(dist[valid])), 2),
            "distance_km_p95": round(float(np.percentile(dist[valid], 95)), 2),
        }
    return Standardized(z=z, valid=valid, reason=reason, diagnostics=diagnostics)


# ---- estimators -------------------------------------------------------------


def density_ls_fit(z: np.ndarray, *, bins: int = 80, z_range: float = 4.0) -> tuple[float, float]:
    """`(mu, sigma)` of a normal least-squares-fitted to the binned density.

    The paper's estimator. Its Fig. 3b draws empirical density points with a
    red normal through them whose peak is 0.386, i.e. `1/(1.035*sqrt(2*pi))` --
    a fitted curve, not the standard normal, and not a moment of the sample.

    Worth knowing before quoting it: adopting it makes `sigma` agree with the
    paper more closely and `mu` agree *less* closely on all three operator
    meshes. It is the faithful estimator, not the flattering one.
    """
    from scipy.optimize import curve_fit
    from scipy.stats import norm

    heights, edges = np.histogram(z, bins=bins, range=(-z_range, z_range), density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    (mu, sigma), _ = curve_fit(
        lambda x, m, s: norm.pdf(x, m, s), centers, heights, p0=[0.0, 1.0]
    )
    return float(mu), float(abs(sigma))


def estimators(z_valid: np.ndarray, *, hist_bins: int = 80, z_range: float = 4.0) -> dict:
    """All three (mu, sigma) pairs, plus the shape moments, side by side.

    Three estimators because they disagree by more than the effect anyone reads
    off them, and the disagreement is systematic rather than noise:

    * `_moment` -- the sample moments. Tail-sensitive, so on these datasets it
      is dominated by the near-zero-sigma rows and reads 6-20.
    * `_moment_trunc4` -- moments over |z| < 4, what the ClickHouse arm
      reports. Benign-looking precisely because the truncation removes the rows
      the sigma pathology produced.
    * `_density_ls` -- the paper's estimator.

    `skew` and `excess_kurtosis` are over the truncated set too, and they are
    the reason the May note's leptokurtosis story must not be carried over:
    as02/as03 come out near-mesokurtic.
    """
    from scipy.stats import kurtosis, skew

    z = np.asarray(z_valid, dtype=float)
    z = z[np.isfinite(z)]
    zt = z[np.abs(z) < z_range]
    mu_ls, sigma_ls = density_ls_fit(z, bins=hist_bins, z_range=z_range)
    return {
        "n": int(z.size),
        "n_within_z_range": int(zt.size),
        "z_range": z_range,
        "mu_z_moment": round(float(z.mean()), 4),
        "sigma_z_moment": round(float(z.std()), 4),
        "mu_z_moment_trunc4": round(float(zt.mean()), 4),
        "sigma_z_moment_trunc4": round(float(zt.std()), 4),
        "mu_z_density_ls": round(mu_ls, 4),
        "sigma_z_density_ls": round(sigma_ls, 4),
        "skew_trunc4": round(float(skew(zt)), 4),
        "excess_kurtosis_trunc4": round(float(kurtosis(zt)), 4),
        "paper_mu_z": PAPER["mu_z"],
        "paper_sigma_z": PAPER["sigma_z"],
        "paper_estimator": PAPER["sigma_z_estimator"],
    }


# ---- per-landmark statistics ------------------------------------------------


def group_keys(df: pd.DataFrame, group_by: str) -> pd.Series:
    """The grouping column for panel (c), as a string series.

    `vp_id` is the default because the paper's claim is about the landmark that
    *performed* the measurement. `vp_coord` collapses co-located VPs (the
    operator meshes carry 134 `vp_id`s over 87 coordinates), which is arguably
    more faithful still -- the paper's landmark is a physical origin, so two
    ids at one coordinate that disagree are an instance effect, not a landmark
    effect. `target_*` reproduces the ClickHouse arm's wrong-endpoint grouping
    on purpose: with only ~20 target coordinates it is visibly the weaker test,
    and showing that beside the VP grouping is the clearest way to make the
    point.
    """
    if group_by == "vp_id":
        return df["vp_id"].astype(str)
    if group_by == "target_id":
        return df["target_id"].astype(str)
    if group_by == "vp_coord":
        return df["vp_lat"].round(6).astype(str) + "," + df["vp_lon"].round(6).astype(str)
    if group_by == "target_coord":
        return df["tg_lat"].round(6).astype(str) + "," + df["tg_lon"].round(6).astype(str)
    raise typer.BadParameter(f"--group-by must be one of {GROUP_BY_CHOICES}, got {group_by!r}")


def quantile_grid(n_quantiles: int) -> np.ndarray:
    """Quantile probabilities for the Q-Q plot, tails trimmed to 2%/98%.

    49 points by default, against the ClickHouse arm's 199. A group here holds
    ~400 observations over <= 23 distinct distances, so 199 probabilities
    resolve past the data into visible staircases, and a 0.5% quantile on
    n = 400 is a single order statistic.
    """
    return np.linspace(0.02, 0.98, n_quantiles)


def landmark_stats(
    df: pd.DataFrame,
    std: Standardized,
    keys: pd.Series,
    *,
    qs: np.ndarray,
    min_per_group: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    """One row per group -- every group, never a top-N.

    `qq_slope` is `polyfit(pooled_quantiles, group_quantiles, 1)` over the
    5-95% range, so it is `sigma_group / sigma_pooled` in the orientation panel
    (c) draws: **slope < 1 means the group is narrower than pooled**. Worth
    stating because the ClickHouse arm plots the axes the other way round
    (observed on x), where the same slope means the opposite, and
    `notes/2026-05-17` reads it in the inverted sense.

    Returns `(stats, pooled_z, per_group_quantiles)`.
    """
    from scipy.stats import ks_2samp, wasserstein_distance

    z = std.z
    valid = std.valid
    pooled = z[valid]
    pooled_q = np.quantile(pooled, qs)
    # The slope is fitted over the central 5-95% only: the extreme quantiles of
    # a ~400-row group are single order statistics and would lever the fit.
    core = (qs >= 0.05) & (qs <= 0.95)

    # Positional indexing below relies on the frame being 0..n-1, which
    # `load_mesh` and the RTT filter both guarantee with `reset_index`.
    assert df.index.equals(pd.RangeIndex(len(df))), "landmark_stats needs a 0..n-1 index"

    rows = []
    curves: dict[str, np.ndarray] = {}
    for key, idx in keys.groupby(keys).groups.items():
        pos = np.asarray(idx, dtype=int)
        gv = valid[pos]
        gz = z[pos][gv]
        n_valid = int(gz.size)
        row = {
            "group": str(key),
            "n": int(len(pos)),
            "n_valid": n_valid,
            "share_dropped": round(float(1 - n_valid / len(pos)), 6),
            "n_distinct_distance": int(pd.Series(df["distance_km"].to_numpy()[pos]).round(6).nunique()),
            "mean_rtt_ms": round(float(df["rtt_ms"].to_numpy()[pos].mean()), 4),
            "median_distance_km": round(float(np.median(df["distance_km"].to_numpy()[pos])), 2),
        }
        if n_valid < min_per_group:
            row.update({"used": False, "n_distinct_z": n_valid})
            rows.append(row)
            continue

        gq = np.quantile(gz, qs)
        slope, intercept = np.polyfit(pooled_q[core], gq[core], 1)
        resid = gq[core] - (slope * pooled_q[core] + intercept)
        ss_tot = float(((gq[core] - gq[core].mean()) ** 2).sum())
        ks = ks_2samp(gz, pooled)
        row.update(
            {
                "used": True,
                "n_distinct_z": int(pd.Series(gz).round(9).nunique()),
                "mean_z": round(float(gz.mean()), 4),
                "sd_z": round(float(gz.std()), 4),
                "median_z": round(float(np.median(gz)), 4),
                "p05_z": round(float(np.quantile(gz, 0.05)), 4),
                "p95_z": round(float(np.quantile(gz, 0.95)), 4),
                "qq_slope": round(float(slope), 4),
                "qq_intercept": round(float(intercept), 4),
                "qq_r2": round(float(1 - (resid**2).sum() / ss_tot), 6) if ss_tot > 0 else None,
                "ks_d_vs_pooled": round(float(ks.statistic), 4),
                "ks_p_vs_pooled": float(ks.pvalue),
                "wasserstein_vs_pooled": round(float(wasserstein_distance(gz, pooled)), 4),
            }
        )
        rows.append(row)
        curves[str(key)] = gq

    stats = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    if "ks_p_vs_pooled" in stats.columns:
        stats["ks_p_bh"] = _benjamini_hochberg(stats["ks_p_vs_pooled"].to_numpy(dtype=float))
    return stats, pooled_q, curves


def _benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values, NaN preserved.

    Present because 134 simultaneous KS tests without a correction is not a
    claim -- though at this n the correction changes nothing, which is itself
    worth reporting.
    """
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    vals = p[ok]
    if vals.size == 0:
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    m = vals.size
    adj = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def eta_squared(z: np.ndarray, keys: np.ndarray) -> float:
    """Share of `Var(z)` explained by group identity, one-way.

    The plainest phrasing of the claim under test: if `f_d` really is
    landmark-independent then knowing which landmark took the measurement
    explains ~none of the variance in the standardized residual.
    """
    z = np.asarray(z, dtype=float)
    grand = z.mean()
    ss_tot = float(((z - grand) ** 2).sum())
    if ss_tot == 0:
        return 0.0
    ss_between = 0.0
    for _, idx in pd.Series(keys).groupby(keys).groups.items():
        g = z[np.asarray(idx, dtype=int)]
        ss_between += g.size * (g.mean() - grand) ** 2
    return float(ss_between / ss_tot)


def permutation_null(
    z: np.ndarray,
    keys: np.ndarray,
    rtt: np.ndarray,
    *,
    n_permutations: int,
    bin_size_ms: float,
    strata: str,
    seed: int,
) -> dict:
    """`sd_of_group_mean_z` against a null that relabels the landmark.

    Needed because the obvious reference is wrong. Under iid sampling a group
    of ~400 would show a mean-z spread of `1/sqrt(400) ~ 0.05`; but a group's
    observations come from <= 23 distinct target coordinates and are strongly
    correlated through that shared geometry, so the iid figure understates the
    null spread badly and would manufacture a rejection.

    `strata="rtt-bin"` shuffles labels **within** RTT bins, preserving each
    group's per-bin row counts. That matters because `mu(d)` is a polynomial
    with structure in its residuals, so a landmark that only ever measures
    short RTTs would differ in mean z from one that measures long RTTs even
    under a perfectly landmark-independent law -- an RTT-coverage effect, not a
    landmark effect. Stratifying holds the RTT mix fixed and isolates identity.

    `strata="none"` is the paper's claim read literally (`f_d` independent of
    the landmark, full stop). The gap between the two is informative on its own.
    """
    rng = np.random.default_rng(seed)
    z = np.asarray(z, dtype=float)
    keys = np.asarray(keys)

    def spread(labels: np.ndarray) -> float:
        means = pd.Series(z).groupby(labels).mean()
        return float(means.std())

    observed = spread(keys)

    if strata == "rtt-bin":
        bin_of = np.floor(np.asarray(rtt, dtype=float) / bin_size_ms)
        blocks = [np.nonzero(bin_of == k)[0] for k in np.unique(bin_of)]
    elif strata == "none":
        blocks = [np.arange(z.size)]
    else:
        raise typer.BadParameter(f"--permute-strata must be rtt-bin or none, got {strata!r}")

    draws = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = keys.copy()
        for block in blocks:
            shuffled[block] = rng.permutation(keys[block])
        draws[i] = spread(shuffled)

    return {
        "strata": strata,
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "bin_size_ms": bin_size_ms if strata == "rtt-bin" else None,
        "statistic": "sd across groups of mean z",
        "observed": round(observed, 6),
        "null_mean": round(float(draws.mean()), 6),
        "null_sd": round(float(draws.std()), 6),
        "null_p50": round(float(np.median(draws)), 6),
        "null_p95": round(float(np.quantile(draws, 0.95)), 6),
        "z_score": (
            round(float((observed - draws.mean()) / draws.std()), 3) if draws.std() > 0 else None
        ),
        # The one number that is comparable ACROSS groupings. The raw spread
        # and eta^2 both grow with the number of groups, so `vp_id` (134) and
        # `target_id` (399) cannot be ranked by either -- but each null is
        # drawn at that grouping's own group count, so the ratio can be.
        "observed_over_null_p50": (
            round(float(observed / np.median(draws)), 3) if np.median(draws) > 0 else None
        ),
        "p_value": round(float((1 + (draws >= observed).sum()) / (1 + n_permutations)), 6),
    }


def spread_summary(stats: pd.DataFrame, *, z: np.ndarray, keys: np.ndarray) -> dict:
    """The headline landmark-independence numbers, over used groups only."""
    used = stats[stats["used"]] if "used" in stats.columns else stats.iloc[:0]
    if used.empty:
        raise typer.BadParameter(
            "no group cleared --min-per-group, so there is no landmark-independence "
            "statistic to report; lower --min-per-group or widen the population"
        )
    q = used["qq_slope"].astype(float)
    return {
        "n_groups": int(len(stats)),
        "n_groups_used": int(len(used)),
        "sd_of_group_mean_z": round(float(used["mean_z"].astype(float).std()), 4),
        "group_mean_z": {
            "p05": round(float(used["mean_z"].astype(float).quantile(0.05)), 4),
            "p50": round(float(used["mean_z"].astype(float).median()), 4),
            "p95": round(float(used["mean_z"].astype(float).quantile(0.95)), 4),
        },
        "group_sd_z": {
            "p05": round(float(used["sd_z"].astype(float).quantile(0.05)), 4),
            "p50": round(float(used["sd_z"].astype(float).median()), 4),
            "p95": round(float(used["sd_z"].astype(float).quantile(0.95)), 4),
        },
        "qq_slope": {
            "min": round(float(q.min()), 4),
            "p25": round(float(q.quantile(0.25)), 4),
            "p50": round(float(q.median()), 4),
            "p75": round(float(q.quantile(0.75)), 4),
            "max": round(float(q.max()), 4),
            "iqr": round(float(q.quantile(0.75) - q.quantile(0.25)), 4),
        },
        "qq_intercept_p50": round(float(used["qq_intercept"].astype(float).median()), 4),
        "eta_squared": round(eta_squared(z, keys), 4),
        "ks_d_vs_pooled": {
            "p50": round(float(used["ks_d_vs_pooled"].astype(float).median()), 4),
            "max": round(float(used["ks_d_vs_pooled"].astype(float).max()), 4),
        },
        "n_groups_ks_rejected_05": int((used["ks_p_vs_pooled"].astype(float) < 0.05).sum()),
        "n_groups_ks_rejected_05_bh": int((used["ks_p_bh"].astype(float) < 0.05).sum()),
    }


def pooled_gof(z: np.ndarray, mu_ls: float, sigma_ls: float) -> dict:
    """KS distances against both normals, each beside its critical value.

    Reported as a distance and a ratio rather than as a p-value on purpose. At
    n ~ 5*10^4 the 5% critical value is ~0.006, so any real structure rejects
    at p < 10^-100 and the p-value carries no information about effect size.
    "D is 9x the critical value" is the honest form of the same statement.

    Also approximate, not exact: `z` has ~100-200 distinct values per group and
    the distance marginal is supported on ~2,000 atoms, so the ties violate
    the KS null's continuity assumption.
    """
    from scipy.stats import kstest

    n = int(z.size)
    crit = 1.36 / np.sqrt(n)
    d_std = float(kstest(z, "norm", args=(0.0, 1.0)).statistic)
    d_fit = float(kstest(z, "norm", args=(mu_ls, sigma_ls)).statistic)
    # No Anderson-Darling here, deliberately. It would only be legible on a
    # subsample -- on the full n its verdict is a foregone rejection -- so its
    # answer would be a function of the subsample size we happened to pick,
    # which is the opposite of what a reported statistic should be. The KS
    # distance beside its critical value says the same thing without inviting
    # that reading. (`scipy.stats.anderson` is also mid-deprecation on the
    # p-value side as of 1.17.)
    return {
        "n": n,
        "ks_d_vs_n01": round(d_std, 4),
        "ks_d_vs_fitted": round(d_fit, 4),
        "ks_critical_5pct": round(float(crit), 6),
        "ks_d_over_critical_vs_n01": round(float(d_std / crit), 2),
        "ks_d_over_critical_vs_fitted": round(float(d_fit / crit), 2),
        "note": (
            "read D/critical as the effect size; at this n any p-value is a "
            "foregone rejection, and ties make it approximate anyway"
        ),
    }


def pick_groups(stats: pd.DataFrame, *, n: int, how: str, seed: int) -> list[str]:
    """Which groups panel (c) draws by name.

    The paper drew "five selected landmarks ... considered to represent the
    whole landmark set well" -- editorially unfalsifiable. The ClickHouse arm
    took the top five by sample count, reproducible but not adversarial, and on
    an operator mesh every VP has ~400 observations so that rule is arbitrary.

    `spread` (the default) ranks every qualifying group by `qq_slope` and draws
    the min, p25, median, p75 and max. Deterministic, and it brackets the worst
    case instead of hiding it. The whole population is drawn behind them as an
    envelope regardless, so the choice only decides which five get named.
    """
    used = stats[stats["used"]].copy()
    if used.empty:
        return []
    if how == "count":
        return used.nlargest(n, "n_valid")["group"].astype(str).tolist()
    if how == "random":
        return (
            used.sample(n=min(n, len(used)), random_state=seed)["group"].astype(str).tolist()
        )
    if how != "spread":
        raise typer.BadParameter(f"--group-pick must be spread, count or random, got {how!r}")
    ordered = used.sort_values("qq_slope").reset_index(drop=True)
    if len(ordered) <= n:
        return ordered["group"].astype(str).tolist()
    pos = np.unique(np.linspace(0, len(ordered) - 1, n).round().astype(int))
    return ordered.loc[pos, "group"].astype(str).tolist()


# ---- panels -----------------------------------------------------------------


def build_panel_a(
    df: pd.DataFrame,
    fit: Fit,
    *,
    view_rtt_max: float | None,
    title: str | None,
    out_png: Path,
    note: str,
) -> None:
    """Fig. 3(a): the delay-distance cloud with `mu(d)` and `mu(d) +/- sigma(d)`.

    Axis labels follow the paper's ("round trip time [ms]", "great circle
    distance [1000 km]") so the two figures can be read against each other.

    The negative-sigma interval is shaded rather than left to the JSON: a band
    that inverts is the kind of thing a reader should see happen, and the dashed
    `mu - sigma` curve crossing above `mu + sigma` is only legible if the
    interval is marked.
    """
    model = fit.model
    hi = float(view_rtt_max) if view_rtt_max else float(model.rtt_max)
    fig, ax = plt.subplots(figsize=(6.0, 5.4), dpi=200)

    ax.scatter(
        df["rtt_ms"], df["distance_km"] / 1000.0,
        s=2.2, alpha=0.06, color=_C_CLOUD, edgecolors="none", zorder=2,
        label=f"measured pairs  (n={len(df):,})",
    )

    grid = np.linspace(model.rtt_min, hi, 400)
    mu = np.polyval(model.p_mu, grid)
    sig = sigma_km(model.p_log_sigma, grid)
    ax.plot(grid, mu / 1000.0, "-", lw=1.8, color=_C_FIT, zorder=5, label=r"$\mu(d)$")
    ax.plot(grid, (mu + sig) / 1000.0, "--", lw=1.0, color=_C_FIT, zorder=5,
            label=r"$\mu(d) \pm \sigma(d)$")
    ax.plot(grid, (mu - sig) / 1000.0, "--", lw=1.0, color=_C_FIT, zorder=5)

    # No per-bin dots. They existed to make the left-edge misbehaviour
    # attributable -- well-behaved dots, misbehaving curve through them -- and
    # both the misbehaviour and the binning are gone: mu is fitted to the raw
    # pairs under a monotonicity constraint. The scatter cloud already shows
    # the population the curve is fitted to.

    span = np.array([0.0, hi])
    ax.plot(span, span / THEORETICAL_SLOPE / 1000.0, ls=":", lw=1.1, color=_C_INK_2, zorder=4,
            label="2/3 c round-trip bound")

    # No sigma<=0 shading: sigma is fitted in log space, so it is positive by
    # construction and there is nothing left to mark. The shading, and the
    # sigma_domain diagnostic that fed it, were removed with that fix.

    ax.set_xlim(0, hi)
    ax.set_ylim(0, max(5.0, float(df["distance_km"].max()) / 1000.0 * 1.05))
    ax.set_xlabel("round trip time [ms]")
    ax.set_ylabel("great circle distance [1000 km]")
    ax.set_title(title or "(a) Delay-distance plot", color=_C_INK)
    _frame(ax, note)
    _save(fig, out_png)


def build_panel_b(
    z_valid: np.ndarray,
    est: dict,
    *,
    hist_bins: int,
    z_range: float,
    title: str | None,
    out_png: Path,
    note: str,
) -> None:
    """Fig. 3(b): standardized residuals against **both** normals.

    Two curves, which is what the paper's own figure shows and what the
    ClickHouse arm omits: the dashed `N(0,1)` is what the model claims, the
    solid `N(mu_hat, sigma_hat)` is the paper's own least-squares density fit.
    Drawing only the first makes any width error look like a modelling failure;
    drawing only the second hides it.

    No KDE, deliberately -- `z` sits on a discrete distance support and a
    smoother would erase the structure the manifest caveats.
    """
    from scipy.stats import norm

    fig, ax = plt.subplots(figsize=(6.0, 5.4), dpi=200)
    ax.hist(
        z_valid, bins=hist_bins, range=(-z_range, z_range), density=True,
        color=_C_CLOUD, alpha=0.45, zorder=2,
        label=f"data  (n={est['n']:,})",
    )
    xs = np.linspace(-z_range, z_range, 400)
    ax.plot(xs, norm.pdf(xs), "--", lw=1.4, color=_C_INK_2, zorder=4,
            label=r"$\mathcal{N}(0,1)$  (the model's claim)")
    mu_ls, sigma_ls = est["mu_z_density_ls"], est["sigma_z_density_ls"]
    ax.plot(xs, norm.pdf(xs, mu_ls, sigma_ls), "-", lw=1.8, color=_C_FIT, zorder=5,
            label=fr"fitted $\mathcal{{N}}$({mu_ls:+.3f}, {sigma_ls:.3f}$^2$)  (paper's estimator)")

    ax.set_xlim(-z_range, z_range)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$\Phi(z)$ standardized probability density")
    ax.set_title(title or "(b) Standardized delay-distance distribution", color=_C_INK)
    _frame(ax, note)
    _save(fig, out_png)


def build_panel_c(
    stats: pd.DataFrame,
    pooled_q: np.ndarray,
    curves: dict[str, np.ndarray],
    drawn: list[str],
    *,
    group_by: str,
    title: str | None,
    out_png: Path,
    note: str,
) -> None:
    """Fig. 3(c): per-landmark Q-Q against the pooled distribution.

    **Pooled on x, per-landmark on y.** The paper draws it the other way round
    (observed on x), which inverts the meaning of the slope; in this
    orientation `slope = sigma_group / sigma_pooled`, so a group narrower than
    pooled reads below the diagonal, which is what a reader expects.

    Every qualifying group is drawn as a faint 5-95% envelope and five are
    named on top. The envelope is what answers the cherry-picking objection the
    paper's "five selected landmarks" invites -- the reader sees the whole
    population's spread, not a curated subset of it.

    Markers, not lines: a group's quantiles come from ~400 rows over <= 23
    distinct distances, and a joining line would read interpolation as data.
    """
    fig, ax = plt.subplots(figsize=(6.0, 5.4), dpi=200)

    used = [g for g in stats[stats["used"]]["group"].astype(str) if g in curves]
    if used:
        band = np.vstack([curves[g] for g in used])
        ax.fill_between(
            pooled_q, np.quantile(band, 0.05, axis=0), np.quantile(band, 0.95, axis=0),
            color=_C_MUTED, alpha=0.18, lw=0, zorder=2,
            label=f"5-95% envelope, all {len(used)} {group_by}s",
        )

    slope_by = dict(zip(stats["group"].astype(str), stats["qq_slope"]))
    for i, g in enumerate(drawn):
        if g not in curves:
            continue
        n_valid = int(stats.loc[stats["group"].astype(str) == g, "n_valid"].iloc[0])
        ax.scatter(
            pooled_q, curves[g], s=9, alpha=0.9, zorder=4,
            color=_VARIANT_HUES[i % len(_VARIANT_HUES)], edgecolors="none",
            label=f"{g}  (n={n_valid:,}, slope {slope_by[g]:.2f})",
        )

    lim = 3.0
    ax.plot([-lim, lim], [-lim, lim], "-", lw=1.0, color=_C_INK_2, zorder=3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("pooled standardized value  (expected)")
    ax.set_ylabel(f"per-{group_by} standardized value  (observed)")
    ax.set_title(title or f"(c) Q-Q plot, {group_by} vs pooled", color=_C_INK)
    _frame(ax, note)
    _save(fig, out_png)


def _frame(ax, note: str) -> None:
    """The shared axis dressing and the bottom-right provenance note."""
    ax.grid(True, color=_C_GRID, lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_C_AXIS)
    leg = ax.legend(loc="upper left", fontsize=7.5, frameon=False, scatterpoints=1)
    for handle in leg.legend_handles:
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([28.0])
        if hasattr(handle, "set_alpha"):
            handle.set_alpha(1.0)
    ax.annotate(
        note, xy=(0.98, 0.02), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=6.5, color=_C_MUTED,
    )


def _save(fig, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# ---- verdict ----------------------------------------------------------------


def verdict(name: str, spread: dict, perm: dict, est: dict) -> str:
    """One line, templated so it cannot be rounded into a stronger claim.

    Both halves of the paper's proposition get their own clause, because they
    can and do come apart: the pooled normal can describe the data well while
    the landmark still explains a chunk of the residual.
    """
    p = perm["p_value"]
    held = p > 0.05
    return (
        f"{name}: landmark-independence {'HOLDS' if held else 'REJECTED'} "
        f"(sd of per-{spread.get('group_by', 'group')} mean z = {spread['sd_of_group_mean_z']:.3f} "
        f"vs {perm['strata']} permutation null p50 {perm['null_p50']:.3f} / "
        f"p95 {perm['null_p95']:.3f}, p = {p:.4f}); "
        f"Q-Q slope p25-p75 {spread['qq_slope']['p25']:.2f}-{spread['qq_slope']['p75']:.2f} "
        f"over {spread['n_groups_used']} groups; group identity explains "
        f"{spread['eta_squared']:.1%} of Var(z). "
        f"Pooled sigma_z = {est['sigma_z_density_ls']:.3f} (paper's estimator) / "
        f"{est['sigma_z_moment_trunc4']:.3f} (truncated moment) vs the paper's "
        f"{PAPER['sigma_z']}."
    )


# ---- command ----------------------------------------------------------------


def analyse_one(
    csv_path: Path,
    *,
    provenance: dict,
    out_dir: Path,
    rtt_min: float,
    rtt_max: float,
    view_rtt_max: float,
    fit_kwargs: dict,
    sigma_ref_km: float,
    group_by: str,
    n_groups: int,
    group_pick: str,
    group_seed: int,
    n_quantiles: int,
    min_per_group: int,
    hist_bins: int,
    z_range: float,
    n_permutations: int,
    permute_strata: str,
    rtt_col: str,
    vp_prefix: str,
    tg_prefix: str,
    title: str | None,
) -> dict:
    """The whole pipeline for one CSV. Returns the manifest."""
    from scipy.stats import levene

    df, n_dropped_load = load_mesh(
        csv_path, rtt_col=rtt_col, vp_prefix=vp_prefix, tg_prefix=tg_prefix
    )
    n_read = len(df) + n_dropped_load

    # --rtt-max filters the POPULATION, unlike plot-distance-rtt's view cuts.
    # The paper's calibration set is its RTT range, and a fit over a wider
    # range is a different model -- see the module docstring.
    n_before_filter = len(df)
    if rtt_min > 0:
        df = df[df["rtt_ms"] >= rtt_min]
    if rtt_max > 0:
        df = df[df["rtt_ms"] <= rtt_max]
    df = df.reset_index(drop=True)
    n_rtt_filtered = n_before_filter - len(df)

    rtt = df["rtt_ms"].to_numpy(dtype=float)
    dist = df["distance_km"].to_numpy(dtype=float)

    fit = fit_pooled(rtt, dist, **fit_kwargs)
    model = fit.model

    # Primary: the paper's literal recipe, pathology and all. The guarded pass
    # is computed beside it because the per-landmark statistics are
    # uninterpretable without it -- a handful of near-zero-sigma rows otherwise
    # dominate every group moment.
    std = standardize(rtt, dist, model.p_mu, model.p_log_sigma, sigma_ref_km=0.0)
    std_guard = standardize(rtt, dist, model.p_mu, model.p_log_sigma, sigma_ref_km=sigma_ref_km)

    est = estimators(std.z[std.valid], hist_bins=hist_bins, z_range=z_range)
    est_guard = estimators(std_guard.z[std_guard.valid], hist_bins=hist_bins, z_range=z_range)

    keys = group_keys(df, group_by)
    qs = quantile_grid(n_quantiles)
    stats, pooled_q, curves = landmark_stats(
        df, std, keys, qs=qs, min_per_group=min_per_group
    )
    stats_guard, _, _ = landmark_stats(
        df, std_guard, keys, qs=qs, min_per_group=min_per_group
    )

    spread = spread_summary(stats, z=std.z[std.valid], keys=keys.to_numpy()[std.valid])
    spread["group_by"] = group_by
    spread_guard = spread_summary(
        stats_guard, z=std_guard.z[std_guard.valid], keys=keys.to_numpy()[std_guard.valid]
    )
    spread_guard["group_by"] = group_by

    perm = permutation_null(
        std_guard.z[std_guard.valid], keys.to_numpy()[std_guard.valid],
        rtt[std_guard.valid],
        n_permutations=n_permutations, bin_size_ms=fit_kwargs["bin_size_ms"],
        strata=permute_strata, seed=group_seed,
    )
    gof = pooled_gof(
        std_guard.z[std_guard.valid], est_guard["mu_z_density_ls"], est_guard["sigma_z_density_ls"]
    )
    used_groups = [g for g in stats_guard[stats_guard["used"]]["group"].astype(str)]
    samples = [
        std_guard.z[std_guard.valid][keys.to_numpy()[std_guard.valid] == g] for g in used_groups
    ]
    gof["levene_p_median_centered"] = (
        float(levene(*samples, center="median").pvalue) if len(samples) > 1 else None
    )

    drawn = pick_groups(stats, n=n_groups, how=group_pick, seed=group_seed)

    stem = csv_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    note_common = (
        f"{csv_path.name}\n"
        f"n={len(df):,}  dropped at load={n_dropped_load:,}  "
        f"rtt-filtered={n_rtt_filtered:,}  unphysical={fit.n_unphysical:,}"
    )
    build_panel_a(
        df, fit,
        view_rtt_max=view_rtt_max or (rtt_max if rtt_max > 0 else None),
        title=title, out_png=out_dir / f"{stem}{PNG_A_SUFFIX}",
        note=(
            f"{note_common}\n"
            f"k=1.0 (paper's band)  cutoff_rtt={model.cutoff_rtt:.2f} ms  "
            f"monotone cubic on {len(fit.fit_rtt):,} raw pairs"
        ),
    )
    build_panel_b(
        std.z[std.valid], est, hist_bins=hist_bins, z_range=z_range,
        title=title, out_png=out_dir / f"{stem}{PNG_B_SUFFIX}",
        note=(
            f"{note_common}\n"
            f"sigma_z: {est['sigma_z_moment']:.3f} moment / "
            f"{est['sigma_z_moment_trunc4']:.3f} |z|<{z_range:g} / "
            f"{est['sigma_z_density_ls']:.3f} density-LS (the paper's, = {PAPER['sigma_z']})\n"
            f"the paper's estimator buys a closer sigma and a worse mu\n"
            f"{std.diagnostics['n_sigma_nonpos']:,} rows "
            f"({std.diagnostics['share_dropped']:.1%}) have sigma(d) <= 0 and no z\n"
            f"primary pass: NO sigma guard (the paper's literal recipe)"
        ),
    )
    build_panel_c(
        stats, pooled_q, curves, drawn,
        group_by=group_by,
        title=title, out_png=out_dir / f"{stem}{PNG_C_SUFFIX}",
        note=(
            f"{note_common}\n"
            f"{len(qs)} quantiles, 2-98%; slope fitted over 5-95%\n"
            f"five named by --group-pick {group_pick}; every group is in the CSV\n"
            f"primary pass: NO sigma guard, so extreme slopes may be the sigma(d)\n"
            f"pathology rather than landmark dependence -- see the guarded block"
        ),
    )

    stats.to_csv(out_dir / f"{stem}{LANDMARK_CSV_SUFFIX}", index=False)
    manifest = {
        "command": "plot-spotter-normality",
        "paper": PAPER,
        "inputs": {
            "csv": str(csv_path),
            "resolved_from": provenance or {"source": "--csv"},
            "rtt_col": rtt_col,
            "vp_prefix": vp_prefix,
            "tg_prefix": tg_prefix,
            "distance_helper": "answer_space.elementwise_km (R=6371 km, chord form)",
            "distance_helper_note": (
                "agrees with the deployed cbg.rtt_model.haversine_distance to ~1e-9 km; "
                "the ClickHouse arm used R=6367 km"
            ),
        },
        "population": {
            "n_rows_read": int(n_read),
            "n_dropped_load": int(n_dropped_load),
            "n_rtt_filtered": int(n_rtt_filtered),
            "n_unphysical": int(fit.n_unphysical),
            "rtt_filter_ms": [rtt_min or None, rtt_max or None],
            "rtt": {
                "min": round(float(rtt.min()), 4),
                "p50": round(float(np.median(rtt)), 4),
                "max": round(float(rtt.max()), 4),
            },
            "distance_km": {
                "min": round(float(dist.min()), 2),
                "p50": round(float(np.median(dist)), 2),
                "max": round(float(dist.max()), 2),
            },
            "discreteness": discreteness(df),
        },
        "fit": {
            "source": "scripts.libs.spotter.spotter_model.SpotterRTTModel.fit (the deployed model)",
            "kwargs": dict(fit_kwargs),
            "target_coverage": None,
            "p_mu": [float(c) for c in model.p_mu],
            "p_log_sigma": [float(c) for c in model.p_log_sigma],
            "rtt_min": round(float(model.rtt_min), 4),
            "rtt_max": round(float(model.rtt_max), 4),
            "cutoff_rtt": round(float(model.cutoff_rtt), 4),
            "k": float(model.k),
            "metadata": model.metadata,
            "n_rows_above_cutoff": int((rtt > model.cutoff_rtt).sum()),
            "fitted": bool(model.fitted),
            "fit_message": model.fit_message,
        },
        # The drop counts are nested under the pass that produced them, not
        # merged in: the primary pass applies no guard, so its
        # `n_sigma_below_ref` is 0 by construction, and a reader scanning a
        # flattened block would read that as "no row has a small sigma".
        "standardize": std.diagnostics,
        "standardized": est,
        "standardized_guarded": {
            "sigma_ref_km": sigma_ref_km,
            "diagnostics": std_guard.diagnostics,
            **est_guard,
        },
        "landmark_independence": {
            **spread,
            "min_per_group": min_per_group,
            "n_quantiles": int(len(qs)),
            "group_pick": group_pick,
            "drawn_groups": drawn,
            "guarded": spread_guard,
            "permutation": perm,
            "goodness_of_fit": gof,
            "csv": f"{stem}{LANDMARK_CSV_SUFFIX}",
        },
        "verdict": verdict(csv_path.stem, spread_guard, perm, est_guard),
        "policy": (
            "--rtt-max FILTERS the fit population, because the paper's calibration set "
            "is its RTT range and a fit over a wider range is a different model. "
            "--view-rtt-max clips panel (a)'s axis only. Contrast plot-distance-rtt, "
            "where every cut is view-only. The PRIMARY standardization applies no "
            "sigma guard (the paper's literal recipe); `standardized_guarded` and "
            "`landmark_independence.guarded` repeat it excluding rows whose fitted "
            "sigma is below --sigma-ref-km, which is what makes the per-group moments "
            "interpretable. Nothing is refitted between the two."
        ),
        "caveats": [
            "KS p-values are approximate: z sits on a discrete distance support "
            "(see population.discreteness), so the null's continuity assumption is violated.",
            "Per-group z are not iid -- a group's rows share a handful of target "
            "coordinates -- so 1/sqrt(n) is not the null spread and the permutation "
            "reference is required rather than optional.",
            "At this n every classical normality test rejects; read the KS D/critical "
            "ratio as the effect size and treat the p-value as a footnote.",
            "The primary sigma_z is dominated by rows near the sigma(d) root; compare "
            "standardized_guarded before drawing any conclusion from it.",
            "sd_of_group_mean_z and eta_squared both grow with the number of groups, so "
            "they cannot rank one --group-by against another. Use "
            "permutation.observed_over_null_p50, whose null is drawn at each grouping's "
            "own group count.",
        ],
    }
    (out_dir / f"{stem}{MANIFEST_SUFFIX}").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def register(app: typer.Typer) -> None:
    @app.command("plot-spotter-normality")
    def plot_spotter_normality_cmd(
        csv: list[Path] = typer.Option(
            None, "--csv", exists=True, dir_okay=False,
            help="Canonical mesh CSV: vp_id/vp_lat/vp_lon, target_id/target_lat/"
                 "target_lon, rtt_ms. Repeatable. Overrides --run-id.",
        ),
        run_id: list[str] = typer.Option(
            None, "--run-id",
            help="Take the CSV from configs/<run-id>.yaml's "
                 "benchmark.source_kwargs.csv_path. Repeatable.",
        ),
        rtt_min: float = typer.Option(
            0.0, "--rtt-min", help="Lower RTT cut, ms. 0 = off. FILTERS the fit population.",
        ),
        rtt_max: float = typer.Option(
            0.0, "--rtt-max",
            help="Upper RTT cut, ms. 0 = off (default), which keeps every row: the "
                 "operator meshes top out near 92 ms, already the paper's regime. "
                 "Pass 80 for the paper's literal plotted range. FILTERS the fit "
                 "population, unlike plot-distance-rtt's view-only cuts.",
        ),
        view_rtt_max: float = typer.Option(
            0.0, "--view-rtt-max", help="Clip panel (a)'s x axis only. 0 = follow --rtt-max.",
        ),
        # Defaults are the mesh configs' spotter_cbg ltd_kwargs, verbatim, so
        # this command diagnoses the model those runs actually fitted. A test
        # cross-checks them against the YAML.
        deg_mu: int = typer.Option(3, "--deg-mu", help="Polynomial degree for mu(d)."),
        deg_sigma: int = typer.Option(2, "--deg-sigma", help="Polynomial degree for sigma(d)."),
        bin_size_ms: float = typer.Option(
            5.0, "--bin-size-ms", help="Cutoff-scan bin width, and the permutation strata width.",
        ),
        cutoff_min_points: int = typer.Option(
            5, "--cutoff-min-points", help="Points per bin needed to extend cutoff_rtt.",
        ),
        sigma_ref_km: float = typer.Option(
            50.0, "--sigma-ref-km",
            help="Reference guard for the SECONDARY reported pass: rows whose fitted "
                 "sigma is below this are excluded from it. Does not refit, and does "
                 "not touch the primary pass.",
        ),
        group_by: str = typer.Option(
            "vp_id", "--group-by",
            help=f"Panel (c) grouping, one of {GROUP_BY_CHOICES}. vp_id is the paper's "
                 f"landmark -- the endpoint that performed the measurement.",
        ),
        n_groups: int = typer.Option(5, "--n-groups", help="Groups NAMED in panel (c)."),
        group_pick: str = typer.Option(
            "spread", "--group-pick",
            help="spread (min/p25/median/p75/max by Q-Q slope) | count | random.",
        ),
        group_seed: int = typer.Option(42, "--group-seed", help="Seed for --group-pick random and the permutation null."),
        n_quantiles: int = typer.Option(49, "--n-quantiles", help="Q-Q probabilities, 2-98%."),
        min_per_group: int = typer.Option(50, "--min-per-group", help="Groups below this are measured but not used."),
        hist_bins: int = typer.Option(80, "--hist-bins", help="Panel (b) histogram bins."),
        z_range: float = typer.Option(4.0, "--z-range", help="Panel (b) x range, and the truncation for the moment estimator."),
        n_permutations: int = typer.Option(200, "--n-permutations", help="Permutation draws for the landmark null."),
        permute_strata: str = typer.Option(
            "rtt-bin", "--permute-strata",
            help="rtt-bin (hold each group's RTT mix fixed) | none (the paper's claim read literally).",
        ),
        rtt_col: str = typer.Option("rtt_ms", "--rtt-col"),
        vp_prefix: str = typer.Option("vp", "--vp-prefix"),
        tg_prefix: str = typer.Option("target", "--tg-prefix"),
        title: str = typer.Option(None, "--title", help="Override the panel titles."),
        out_dir: Path = typer.Option(
            None, "--out-dir",
            help="Where to write. Default: alongside the input CSV (--csv mode) or "
                 "outputs/analysis/v3/<run-id>/spotter-normality/ (--run-id mode).",
        ),
        outputs_root: Path = typer.Option(
            None, "--outputs-root", help="Override the benchmark outputs root (--run-id mode).",
        ),
        analysis_root: Path = typer.Option(None, "--analysis_root", help="Override the analysis output root."),
    ) -> None:
        """Spotter Fig. 3 on a mesh CSV: normal fit + per-landmark Q-Q.

        Writes <stem>_spotter_fig3{a,b,c}*.png, <stem>_spotter_normality.json,
        <stem>_landmark_independence.csv (every group) and <stem>_bin_fit.csv.

        Panel (c) groups by the LANDMARK that took the measurement, which is
        what the paper's landmark-independence claim is about; the verdict is a
        permutation effect size, not a p-value.
        """
        # An explicit --csv wins over --run-id, as the help states -- not a
        # strict XOR, because `--config` injects the config's top-level run_id
        # into every command that declares one.
        inputs: list[tuple[Path, dict, Path]] = []
        if csv:
            for path in csv:
                inputs.append((path, {}, Path(out_dir) if out_dir else path.parent))
        elif run_id:
            for rid in run_id:
                path, prov = csv_from_config(rid)
                dest = (
                    Path(out_dir)
                    if out_dir
                    else _run_out_dir(rid, analysis_root, outputs_root)
                )
                inputs.append((path, prov, dest))
        else:
            raise typer.BadParameter(
                "pass --csv, or --run-id to read the CSV from "
                "configs/<run-id>.yaml's benchmark.source_kwargs"
            )

        if group_by not in GROUP_BY_CHOICES:
            raise typer.BadParameter(f"--group-by must be one of {GROUP_BY_CHOICES}")

        fit_kwargs = {
            "deg_mu": deg_mu,
            "deg_sigma": deg_sigma,
            "bin_size_ms": bin_size_ms,
            "cutoff_min_points": cutoff_min_points,
        }

        manifests = []
        for path, prov, dest in inputs:
            manifest = analyse_one(
                path, provenance=prov, out_dir=dest,
                rtt_min=rtt_min, rtt_max=rtt_max, view_rtt_max=view_rtt_max,
                fit_kwargs=fit_kwargs, sigma_ref_km=sigma_ref_km,
                group_by=group_by, n_groups=n_groups, group_pick=group_pick,
                group_seed=group_seed, n_quantiles=n_quantiles,
                min_per_group=min_per_group, hist_bins=hist_bins, z_range=z_range,
                n_permutations=n_permutations, permute_strata=permute_strata,
                rtt_col=rtt_col, vp_prefix=vp_prefix, tg_prefix=tg_prefix, title=title,
            )
            manifests.append(manifest)
            typer.echo(f"wrote {dest}/{path.stem}{MANIFEST_SUFFIX}")
            typer.echo(f"  {manifest['verdict']}")

        if len(manifests) > 1:
            # One table across the datasets, so three manifests do not have to
            # be diffed by hand to compare them.
            root = Path(out_dir) if out_dir else inputs[0][2]
            summary = {
                "command": "plot-spotter-normality",
                "paper": PAPER,
                "datasets": [
                    {
                        "csv": m["inputs"]["csv"],
                        "n": m["population"]["discreteness"]["n_rows"],
                        "sigma_z_moment": m["standardized"]["sigma_z_moment"],
                        "sigma_z_moment_trunc4": m["standardized"]["sigma_z_moment_trunc4"],
                        "sigma_z_density_ls": m["standardized"]["sigma_z_density_ls"],
                        "sigma_z_guarded_moment": m["standardized_guarded"]["sigma_z_moment"],
                        "n_sigma_nonpos": m["standardize"]["n_sigma_nonpos"],
                        "sd_of_group_mean_z": m["landmark_independence"]["sd_of_group_mean_z"],
                        "sd_of_group_mean_z_guarded": (
                            m["landmark_independence"]["guarded"]["sd_of_group_mean_z"]
                        ),
                        "permutation_p": m["landmark_independence"]["permutation"]["p_value"],
                        "eta_squared_guarded": (
                            m["landmark_independence"]["guarded"]["eta_squared"]
                        ),
                        "verdict": m["verdict"],
                    }
                    for m in manifests
                ],
            }
            (root / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n")
            typer.echo(f"wrote {root / SUMMARY_JSON}")


def _run_out_dir(rid: str, analysis_root: Path | None, outputs_root: Path | None) -> Path:
    """`outputs/analysis/v3/<run_id>/spotter-normality/`, grid-free.

    Resolved through `paths.resolve_run` rather than by string-joining, as
    every other v3 command does -- which does mean `--run-id` mode needs the
    run's benchmark outputs to exist, since that is where (source, setup) are
    discovered. `--csv` mode has no such requirement, and is the way in for a
    dataset that has not been scored yet.
    """
    from scripts.analysis.v3.modules.paths import DEFAULT_OUTPUTS_ROOT, resolve_run

    run = resolve_run(rid, outputs_root or DEFAULT_OUTPUTS_ROOT)
    return run.spotter_normality_dir(root=analysis_root)
