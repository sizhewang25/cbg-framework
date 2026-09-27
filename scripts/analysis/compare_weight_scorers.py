"""Compare weight-scorer arms from a *-wsweep run against their tau=50 parent.

The tau=50 baseline is NOT an arm in the sweep config: the sweep reuses its
parent mesh run's source_kwargs, so the two runs materialize byte-identical
folds and the parent's `octant_cbg_hull` / `octant_cbg_spl` combos already ARE
tau=50. This joins them in on `target_id`, which is what makes every comparison
here paired.

Percentiles are reported with a SITE-clustered bootstrap. The as0* target sets
are ~20 IP replicas per coordinate, so ~400 targets carry only ~20 independent
sites; resampling targets would understate every interval by roughly sqrt(20).

Usage:
    python -m scripts.analysis.compare_weight_scorers --run-id as01-260728-260802-wsweep
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUTS = Path("outputs/benchmark/v2")
SETUP = "generic_csv/anchors_to_probes"

#: Sweep suffix -> (label, dynamic range w(3ms)/w(55ms)). The range is what
#: orders the arms on the steepness axis; it is the quantity the count-imbalance
#: argument is about, not the parameter value.
ARMS = [
    ("unw",   "unweighted",  1.0),
    ("ip1",   "1/rtt",       55.0 / 3.0),
    ("ip2",   "1/rtt^2",     (55.0 / 3.0) ** 2),
    ("ip3",   "1/rtt^3",     (55.0 / 3.0) ** 3),
    ("tau10", "exp(-rtt/10)", math.exp(52 / 10)),
    ("tau5",  "exp(-rtt/5)",  math.exp(52 / 5)),
    ("tau1",  "exp(-rtt/1)",  math.exp(52 / 1)),
]
VARIANTS = [("octant_cbg_hull", "OCT-H"), ("octant_cbg_spl", "OCT-S")]


def load_combo(run_id: str, combo: str) -> pd.DataFrame | None:
    frames = []
    for fold in sorted((OUTPUTS / run_id / SETUP).glob("fold_*")):
        tp = fold / combo / "targets.parquet"
        if not (fold / combo / "run.json").exists() or not tp.exists():
            continue
        d = pd.read_parquet(tp, columns=["target_id", "target_lat", "target_lon",
                                         "error_km", "status"])
        d["fold"] = fold.name
        frames.append(d)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def boot_ci(df: pd.DataFrame, col_a: str, col_b: str, q: float,
            n: int = 3000, seed: int = 0) -> tuple[float, float, float]:
    """Paired site-clustered bootstrap on quantile(a) - quantile(b).

    Returns (lo, hi, p_wrong_sign) where `p_wrong_sign` is the share of
    resamples landing on the opposite side of zero from the point estimate.
    That share is reported because a bare CI hides how marginal a verdict is,
    and here it is often very marginal: with 20 sites and a coarse error tail,
    many resamples produce an identical p90, so the bootstrap distribution
    piles mass exactly on 0.0 and the interval endpoint sits on the boundary.
    """
    rng = np.random.default_rng(seed)
    sites = df["site"].unique()
    groups = {s: g for s, g in df.groupby("site")}
    out = np.empty(n)
    for i in range(n):
        sub = pd.concat([groups[s] for s in rng.choice(sites, len(sites), replace=True)])
        out[i] = sub[col_a].quantile(q) - sub[col_b].quantile(q)
    lo, hi = np.percentile(out, [2.5, 97.5])
    med = np.median(out)
    p_wrong = float(np.mean(out >= 0) if med < 0 else np.mean(out <= 0))
    return float(lo), float(hi), p_wrong


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--baseline-run-id", default=None,
                    help="defaults to the sweep run id with -wsweep -> -mesh")
    ap.add_argument("--bootstrap", type=int, default=3000)
    args = ap.parse_args()

    base_run = args.baseline_run_id or args.run_id.replace("-wsweep", "-mesh")

    for combo_base, term in VARIANTS:
        print(f"\n{'=' * 78}\n{term}  ({combo_base})   run={args.run_id}\n{'=' * 78}")

        base = load_combo(base_run, combo_base)
        if base is None:
            print(f"  no tau=50 baseline at {base_run}/{combo_base}; skipping")
            continue
        base = base.rename(columns={"error_km": "tau50"})
        base["site"] = (base.target_lat.round(4).astype(str) + ","
                        + base.target_lon.round(4).astype(str))
        merged = base[["target_id", "site", "fold", "tau50"]].copy()

        present = []
        for tag, label, dyn in ARMS:
            arm = load_combo(args.run_id, f"{combo_base}_{tag}")
            if arm is None:
                continue
            n_folds = arm.fold.nunique()
            merged = merged.merge(
                arm[["target_id", "error_km"]].rename(columns={"error_km": tag}),
                on="target_id", how="inner")
            present.append((tag, label, dyn, n_folds))

        if not present:
            print("  no completed sweep arms yet")
            continue

        n_sites = merged.site.nunique()
        print(f"n = {len(merged)} targets, {n_sites} sites, "
              f"folds joined on target_id")

        # The join is an INNER join across arms, so a half-finished run silently
        # narrows every comparison to the targets the slowest arm has reached.
        # That is the right thing for a paired test and the wrong thing to read
        # as a result, so say so loudly rather than printing a quiet small n.
        fold_counts = {label: nf for _, label, _, nf in present}
        if len(set(fold_counts.values())) > 1:
            print("\n  !! PARTIAL RUN — arms have unequal fold coverage:")
            for label, nf in sorted(fold_counts.items(), key=lambda x: x[1]):
                print(f"       {label:<15} {nf}/5 folds")
            print("     The table below is the INTERSECTION of all arms, so it "
                  "reports\n     whatever the least-complete arm has reached. "
                  "Directional only.")
        print()
        print(f"{'arm':<15}{'dyn.range':>11}{'p50':>8}{'p75':>8}{'p90':>8}"
              f"{'<40km':>8}{'folds':>7}")
        print("-" * 65)
        # tau=50 parent first, as the reference row.
        b = merged["tau50"]
        print(f"{'exp(-rtt/50)*':<15}{2.8:>11.1f}{b.median():>8.1f}"
              f"{b.quantile(.75):>8.1f}{b.quantile(.90):>8.1f}"
              f"{100 * (b < 40).mean():>7.1f}%{'(parent)':>7}")
        for tag, label, dyn, nf in sorted(present, key=lambda x: x[2]):
            v = merged[tag]
            print(f"{label:<15}{dyn:>11.4g}{v.median():>8.1f}"
                  f"{v.quantile(.75):>8.1f}{v.quantile(.90):>8.1f}"
                  f"{100 * (v < 40).mean():>7.1f}%{nf:>7}")
        print("\n  * tau=50 is the parent mesh run's combo, joined in on "
              "target_id (identical folds by construction).")

        print(f"\nPaired vs tau=50, site-clustered bootstrap "
              f"({args.bootstrap} resamples over {n_sites} sites):")
        print(f"{'arm':<15}{'stat':>6}{'delta':>9}{'95% CI':>24}{'p_wrong':>9}   verdict")
        print("-" * 82)
        for tag, label, dyn, _ in sorted(present, key=lambda x: x[2]):
            for q in (0.5, 0.9):
                lo, hi, p_wrong = boot_ci(merged, tag, "tau50", q, n=args.bootstrap)
                d = merged[tag].quantile(q) - b.quantile(q)
                # Strictly exclude zero. `(lo < 0) == (hi < 0)` is NOT the same
                # test: it calls [0.0, 5.9] separable, but that interval
                # contains zero.
                if lo > 0 or hi < 0:
                    # "marginal" when the interval clears zero by less than a
                    # kilometre — true at 95% but not worth a claim.
                    sep = "separable (MARGINAL)" if min(abs(lo), abs(hi)) < 1.0 \
                        else "SEPARABLE"
                else:
                    sep = "not separable"
                print(f"{label:<15}{'p' + str(int(q * 100)):>6}{d:>9.1f}"
                      f"   [{lo:>8.2f},{hi:>8.2f}]{p_wrong:>9.3f}   {sep}")
            better = int((merged[tag] < b - 1).sum())
            worse = int((merged[tag] > b + 1).sum())
            print(f"{'':<15}{'pairs':>6}   better on {better}, worse on {worse}, "
                  f"of {len(merged)}\n")


if __name__ == "__main__":
    main()
