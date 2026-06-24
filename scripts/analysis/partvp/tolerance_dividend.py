"""Tolerance dividend per (run, combo).

The bounded answer space forgives bounded coordinate error: a prediction can be
>R km off yet still snap to the correct centroid. The **tolerance dividend** is
that gap, per variant:

    same_centroid_acc  = P(prediction snaps to the truth's centroid)      (Tier-1 ∪ Tier-2)
    within_r           = P(prediction within R km of the truth centroid)  (Tier-1)
    dividend_abs       = same_centroid_acc - within_r                     (= Tier-2 share, pp)
    dividend_rel       = dividend_abs / same_centroid_acc                 (share of *correct*
                                                                           answers won only by tolerance)

Reads the per-target tier tables from extract_features (which carry `match` and
`within_r` against the R=50 km cluster answer space). Prints a table and writes
tolerance_dividend.csv.

CLI:
    python -m scripts.analysis.partvp.tolerance_dividend \\
        --features "scripts/analysis/outputs/partvp/data/*.parquet" \\
                   "scripts/analysis/outputs/partvp/data_eu/*.parquet" \\
        --out scripts/analysis/outputs/partvp/analysis/tolerance_dividend.csv
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

TEXTBOOK = ["vanilla_cbg", "million_scale_cbg", "octant_cbg", "spotter_cbg"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    paths = sorted({p for g in args.features for p in glob.glob(g)})
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    # Guard against overlapping inputs (e.g. a combined region_confidence table
    # alongside the per-run files): one row per (run, combo, target).
    df = df.drop_duplicates(["run_id", "combo_id", "target_id"], keep="first")

    rows = []
    for (run, combo), v in df.groupby(["run_id", "combo_id"]):
        if combo not in TEXTBOOK:
            continue
        n = len(v)
        acc = float(v["match"].mean())          # same-centroid (Tier-1 ∪ Tier-2)
        within = float(v["within_r"].mean())     # within R (Tier-1)
        div = acc - within
        rows.append({
            "run_id": run, "combo_id": combo, "n": n,
            "same_centroid_acc": round(acc, 4),
            "within_r": round(within, 4),
            "dividend_abs": round(div, 4),
            "dividend_rel": round(div / acc, 4) if acc > 0 else float("nan"),
        })
    out = pd.DataFrame(rows).sort_values(["run_id", "combo_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)
    for run, g in out.groupby("run_id"):
        print(f"\n=== {run} ===")
        print(g[["combo_id", "n", "same_centroid_acc", "within_r", "dividend_abs", "dividend_rel"]]
              .to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
