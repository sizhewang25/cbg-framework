"""Why a pooled normal is the wrong delay-distance model on operator meshes.

Backs notes/2026-09-19-normal-dist-wrong-for-operator-hypergiant.md. Four
analyses, each isolating one assumption in Laki et al. (2011) Sec. IV and
measuring what it costs on `datasets/final/as0{1,2,3}-*.mainland.sanitized.csv`
(AT&T VPs -> Akamai / Google / Netflix serving IPs):

1. `shape`     -- one-sided noise against a symmetric family. All delay noise
                  ADDS RTT, so at fixed RTT the distance distribution is
                  left-skewed. A Gaussian cannot represent that.
2. `envelope`  -- bound versus quantile. CBG's per-VP bestline is fitted to sit
                  above the data; `mu + sigma` sits at the ~84th percentile by
                  construction. Compares containment and radius at identical
                  rows.
3. `sides`     -- the annulus's inner edge, which has no physical basis: there
                  is no lower bound on distance given RTT.
4. `landmark`  -- the landmark-independence claim: a two-way (VP, target-site)
                  variance decomposition, and how much one additive RTT offset
                  per VP removes.

The fit is always the deployed model (`SpotterRTTModel` with the mesh configs'
`spotter_cbg` ltd_kwargs), and the panel figures come from
`scripts.analysis.v3.modules.figure_spotter_normality` -- this script reuses its
loader, fit and `standardize` rather than re-deriving any of them, so the two
cannot disagree.

Layering note: this imports from `scripts.analysis.v3.modules`, which is
backwards from the usual libs <- analysis direction. Deliberate. The
alternative is a fourth copy of the bin-and-polyfit logic in this repo, and a
copy is how `spotter_normality_check.py` ended up silently describing a
different model than the one deployed. A one-off analysis script depending
upward is the cheaper mistake.

Run:
    .venv/bin/python -m scripts.libs.cbg_feasibility.spotter_assumption_breakdown
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.v3.modules.figure_spotter_normality import (
    estimators,
    eta_squared,
    fit_pooled,
    group_keys,
    load_mesh,
    standardize,
)
from scripts.libs.cbg.rtt_model import RTTDistanceModel
from scripts.libs.spotter.spotter_model import sigma_km

#: The mesh configs' `spotter_cbg.ltd_kwargs`, verbatim.
FIT_KWARGS = dict(
    deg_mu=3, deg_sigma=2, bin_size_ms=5.0, cutoff_min_points=5,
)

#: Rows whose fitted sigma is below this are excluded from the per-VP
#: statistics. Not a refit -- near the sigma(d) root a 1,700 km residual over a
#: 3 km sigma puts |z| in the hundreds, and a handful of those rows otherwise
#: set every moment. See the note's "sigma(d) <= 0" section.
SIGMA_REF_KM = 50.0

#: What the operator datasets are, for the manifest.
PEERS = {"as01": "AT&T x Akamai", "as02": "AT&T x Google", "as03": "AT&T x Netflix"}

DEFAULT_OUT = Path("scripts/libs/cbg_feasibility/outputs/spotter_assumption_breakdown")


def prepare(csv_path: Path):
    """`(rows, model)` with `z` attached, guarded, plus the VP and site keys."""
    df, _ = load_mesh(csv_path)
    fit = fit_pooled(df.rtt_ms.values, df.distance_km.values, **FIT_KWARGS)
    model = fit.model
    std = standardize(
        df.rtt_ms.values, df.distance_km.values,
        model.p_mu, model.p_log_sigma, sigma_ref_km=SIGMA_REF_KM,
    )
    d = df[std.valid].copy()
    d["z"] = std.z[std.valid]
    d["vp"] = group_keys(df, "vp_id").to_numpy()[std.valid]
    # The target side is hypergiant serving infrastructure; ~20 coordinates
    # stand behind ~400 target ids, so the SITE is the unit that carries
    # topology, not the id.
    d["site"] = group_keys(df, "target_coord").to_numpy()[std.valid]
    return d.reset_index(drop=True), model, std


def shape(d: pd.DataFrame) -> dict:
    """Assumption 1: the conditional distance distribution is symmetric.

    Delay noise is one-sided -- queueing, backhaul and detours only ever add
    RTT. At a fixed RTT that pushes the true distance BELOW the typical value,
    i.e. toward negative z. A left-skewed z is therefore the signature the
    physics predicts, and no symmetric family can carry it.
    """
    est = estimators(d.z.values)
    return {
        "skew": est["skew_trunc4"],
        "excess_kurtosis": est["excess_kurtosis_trunc4"],
        "sigma_z_density_ls": est["sigma_z_density_ls"],
        "sigma_z_moment": est["sigma_z_moment"],
        "interpretation": (
            "negative skew = heavy left tail = RTT inflated relative to distance, "
            "the direction additive delay noise must produce"
        ),
    }


def sides(d: pd.DataFrame, k: float = 1.0) -> dict:
    """Assumption 2: a two-sided constraint is meaningful.

    `k = 1.0` is what the mesh configs deploy (`target_coverage` unset). The
    inner edge `mu - k*sigma` has no physical justification -- there is no lower
    bound on distance given an RTT, since a nearby target can be reached by an
    arbitrarily bad route. Splitting the miss rate by side shows how much the
    invented edge costs.
    """
    lo = float((d.z < -k).mean())
    hi = float((d.z > k).mean())
    ideal = float(2 * (1 - 0.8413)) if k == 1.0 else None
    per_vp = (d.z.abs() > k).groupby(d.vp).mean()
    return {
        "k": k,
        "miss_inside_inner_hole": round(lo, 4),
        "miss_beyond_outer_edge": round(hi, 4),
        "miss_total": round(lo + hi, 4),
        "miss_for_a_perfect_normal": round(ideal, 4) if ideal else None,
        "per_vp_miss": {
            "p05": round(float(per_vp.quantile(0.05)), 4),
            "p50": round(float(per_vp.median()), 4),
            "p95": round(float(per_vp.quantile(0.95)), 4),
        },
        "interpretation": (
            "the aggregate miss rate matches a perfect normal, but per-VP it "
            "swings by ~5x -- and multilateration is a worst-case operation over "
            "VPs, not an average one"
        ),
    }


def envelope(d: pd.DataFrame, model) -> dict:
    """Assumption 3: a conditional mean can stand in for a bound.

    CBG's per-VP bestline (`scripts.libs.cbg.rtt_model.RTTDistanceModel`, the
    `low_envelope` LTD) is fitted to lie ABOVE the delay-distance points, so its
    radius tracks the worst case at each RTT. `mu + sigma` is the ~84th
    percentile whatever the tail does. The two can have the same median radius
    and wildly different containment -- which is the point.

    Note the second difference this exposes: the bestline is fitted PER VP, so
    it absorbs each VP's fixed delay offset into its own intercept. CBG never
    makes the landmark-independence assumption that `landmark()` measures.
    """
    radii = []
    for vp, g in d.groupby("vp"):
        mdl = RTTDistanceModel(
            anchor_ip=str(vp),
            anchor_lat=float(g.vp_lat.iloc[0]),
            anchor_lon=float(g.vp_lon.iloc[0]),
        )
        if not mdl.fit(
            np.asarray(g.distance_km.values, dtype=float),
            np.asarray(g.rtt_ms.values, dtype=float),
        ):
            continue
        radii.append(
            pd.Series([mdl.predict_distance(float(x)) for x in g.rtt_ms.values],
                      index=g.index)
        )
    if not radii:
        return {"error": "no VP admitted a bestline fit"}

    d = d.assign(cbg_r=pd.concat(radii).reindex(d.index)).dropna(subset=["cbg_r"])
    rtt = np.clip(d.rtt_ms.values, model.rtt_min, model.rtt_max)
    mu, sg = np.polyval(model.p_mu, rtt), sigma_km(model.p_log_sigma, rtt)
    outer, inner = mu + sg, mu - sg
    truth, cbg_r = d.distance_km.values, d.cbg_r.values
    ratio = outer / cbg_r
    return {
        "n": int(len(d)),
        "containment_cbg_disk": round(float((truth <= cbg_r).mean()), 4),
        "containment_spotter_annulus": round(
            float(((truth >= inner) & (truth <= outer)).mean()), 4),
        "containment_spotter_outer_only": round(float((truth <= outer).mean()), 4),
        "outer_over_cbg_radius": {
            "p10": round(float(np.percentile(ratio, 10)), 3),
            "p50": round(float(np.percentile(ratio, 50)), 3),
            "p90": round(float(np.percentile(ratio, 90)), 3),
        },
        "median_radius_km": {
            "cbg": round(float(np.median(cbg_r)), 1),
            "spotter_outer": round(float(np.median(outer)), 1),
            "spotter_inner": round(float(np.median(inner)), 1),
            "truth": round(float(np.median(truth)), 1),
        },
        "interpretation": (
            "comparable median radius, ~15pp less containment: what is lost is "
            "not the envelope's height but its envelope-ness"
        ),
    }


def landmark(d: pd.DataFrame, model, offset_grid=np.linspace(-25, 25, 501)) -> dict:
    """Assumption 4: `f_d(s)` is landmark-independent.

    Two readings. The two-way decomposition asks how much of Var(z) is
    structure at all -- if (VP, site) identity determines the residual, there is
    no common distribution for the residual to be drawn from, and exchangeability
    across landmarks cannot hold. The offset model then asks what SHAPE the
    per-VP part has, by giving each VP one free additive RTT offset and
    re-standardizing.

    `sqrt(1 - eta^2)` is the Q-Q slope the landmark effect predicts on its own:
    Var(pooled) = within + between, and slope = sigma_VP / sigma_pooled, so
    between-VP variance alone forces most slopes below 1.
    """
    z, vp, site = d.z.values, d.vp.values, d.site.values
    grand = z.mean()
    ss_tot = float(((z - grand) ** 2).sum())
    vp_mean = d.groupby("vp").z.transform("mean")
    st_mean = d.groupby("site").z.transform("mean")
    cell_mean = d.groupby(["vp", "site"]).z.transform("mean")
    ss_add = float((((vp_mean + st_mean - grand)) - grand).pow(2).sum())
    ss_cell = float(((cell_mean - grand) ** 2).sum())

    by_vp = pd.Series(z).groupby(vp)
    tau = float(by_vp.mean().std(ddof=0))
    omega = float(np.sqrt((by_vp.var(ddof=0) * by_vp.size()).sum() / len(z)))
    eta = eta_squared(z, vp)

    # One additive RTT offset per VP, least squares on distance.
    delta = {}
    for v, g in d.groupby("vp"):
        r, s = g.rtt_ms.values, g.distance_km.values
        sse = [
            float(((s - np.polyval(model.p_mu, np.clip(r - x, model.rtt_min, model.rtt_max))) ** 2).sum())
            for x in offset_grid
        ]
        delta[v] = float(offset_grid[int(np.argmin(sse))])
    delta_s = pd.Series(delta)
    adj = np.clip(d.rtt_ms.values - d.vp.map(delta).values, model.rtt_min, model.rtt_max)
    mu2, sg2 = np.polyval(model.p_mu, adj), sigma_km(model.p_log_sigma, adj)
    ok = sg2 > SIGMA_REF_KM
    z2 = (d.distance_km.values[ok] - mu2[ok]) / sg2[ok]
    km_bias = (np.polyval(model.p_mu, np.clip(d.rtt_ms.values, model.rtt_min, model.rtt_max))
               - np.polyval(model.p_mu, adj))
    per_vp_km = pd.Series(km_bias, index=d.index).groupby(d.vp).mean().abs()

    return {
        "variance_decomposition": {
            "pooled_sd": round(float(z.std()), 4),
            "within_vp_sd": round(omega, 4),
            "between_vp_sd": round(tau, 4),
            "eta_squared_vp": round(eta, 4),
            "share_vp_main": round(float(((vp_mean - grand) ** 2).sum() / ss_tot), 4),
            "share_site_main": round(float(((st_mean - grand) ** 2).sum() / ss_tot), 4),
            "share_interaction": round(max(0.0, (ss_cell - ss_add) / ss_tot), 4),
            "share_within_cell": round(1 - ss_cell / ss_tot, 4),
        },
        "predicted_qq_slope_sqrt_1_minus_eta2": round(float(np.sqrt(1 - eta)), 4),
        "offset_model": {
            "delta_ms": {
                "p05": round(float(delta_s.quantile(0.05)), 3),
                "p50": round(float(delta_s.median()), 3),
                "p95": round(float(delta_s.quantile(0.95)), 3),
            },
            "censored_at_grid_edge": int(
                ((delta_s <= offset_grid[0] + 1e-9) | (delta_s >= offset_grid[-1] - 1e-9)).sum()
            ),
            "corr_mean_z_vs_delta": round(
                float(np.corrcoef(
                    pd.Series(z).groupby(vp).mean().values,
                    delta_s.reindex(pd.Series(z).groupby(vp).mean().index).values)[0, 1]), 4),
            "eta_squared_after": round(float(eta_squared(z2, d.vp.values[ok])), 4),
            "between_vp_sd_after": round(
                float(pd.Series(z2).groupby(d.vp.values[ok]).mean().std(ddof=0)), 4),
            "km_bias_per_vp": {
                "p50": round(float(per_vp_km.median()), 1),
                "p90": round(float(per_vp_km.quantile(0.9)), 1),
                "max": round(float(per_vp_km.max()), 1),
            },
        },
        "interpretation": (
            "the residual is near-deterministic given (VP, site); the per-VP part "
            "is a fixed additive RTT offset, and one scalar per VP removes ~90% of it"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", action="append",
                    help="Dataset stem, e.g. as01. Repeatable. Default: as01 as02 as03.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    out: dict = {"fit_kwargs": FIT_KWARGS, "sigma_ref_km": SIGMA_REF_KM, "datasets": {}}
    for name in args.dataset or ["as01", "as02", "as03"]:
        hits = glob.glob(f"datasets/final/{name}-*.mainland.sanitized.csv")
        if not hits:
            raise SystemExit(f"no CSV for {name} under datasets/final/")
        csv_path = Path(sorted(hits)[0])
        d, model, _ = prepare(csv_path)
        res = {
            "csv": str(csv_path),
            "peer": PEERS.get(name),
            "n_rows_guarded": int(len(d)),
            "n_vps": int(d.vp.nunique()),
            "n_target_sites": int(d.site.nunique()),
            "shape": shape(d),
            "sides_k1": sides(d, k=1.0),
            "sides_k2": sides(d, k=2.0),
            "envelope": envelope(d, model),
            "landmark": landmark(d, model),
        }
        out["datasets"][name] = res

        s, si, e, la = res["shape"], res["sides_k1"], res["envelope"], res["landmark"]
        v = la["variance_decomposition"]
        print(f"\n=== {name}  ({res['peer']}, n={res['n_rows_guarded']:,}, "
              f"{res['n_vps']} VPs, {res['n_target_sites']} target sites)")
        print(f"  shape     skew {s['skew']:+.2f}  kurtosis {s['excess_kurtosis']:+.2f}  "
              f"sigma_z {s['sigma_z_density_ls']:.3f} (density LS)")
        print(f"  sides     k=1 miss {si['miss_total']:.1%} "
              f"({si['miss_inside_inner_hole']:.1%} inner hole / "
              f"{si['miss_beyond_outer_edge']:.1%} outer edge); "
              f"perfect normal would miss {si['miss_for_a_perfect_normal']:.1%}")
        print(f"            per-VP miss p05/p50/p95 "
              f"{si['per_vp_miss']['p05']:.0%}/{si['per_vp_miss']['p50']:.0%}/"
              f"{si['per_vp_miss']['p95']:.0%}")
        print(f"  envelope  containment CBG {e['containment_cbg_disk']:.1%} vs "
              f"annulus {e['containment_spotter_annulus']:.1%} vs "
              f"outer-only {e['containment_spotter_outer_only']:.1%};  "
              f"radius ratio p50 {e['outer_over_cbg_radius']['p50']:.2f}")
        print(f"  landmark  Var(z): within {v['within_vp_sd']:.3f} + between "
              f"{v['between_vp_sd']:.3f};  VP {v['share_vp_main']:.1%} / site "
              f"{v['share_site_main']:.1%} / interaction {v['share_interaction']:.1%} / "
              f"within-cell {v['share_within_cell']:.1%}")
        print(f"            eta^2 {v['eta_squared_vp']:.1%} -> "
              f"{la['offset_model']['eta_squared_after']:.1%} with one delta per VP "
              f"(corr {la['offset_model']['corr_mean_z_vs_delta']:+.2f}, median bias "
              f"{la['offset_model']['km_bias_per_vp']['p50']:.0f} km)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "breakdown.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
