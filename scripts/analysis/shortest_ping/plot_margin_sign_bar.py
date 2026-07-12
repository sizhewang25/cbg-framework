"""Plot margin-sign percentage bars for overall/success/failure.

Given the per-target features CSV from
`scripts.analysis.shortest_ping.per_target_features`, this script draws a
stacked percentage bar chart with:
  x-axis: population bucket (overall, success, failure)
  y-axis: percentage of targets
  stacks:
    - positive margin (> 0)
    - non-positive margin (<= 0), i.e., negative including zero

The goal is to visualize how often targets are distinguishable by VP margin
sign across all targets and by classification result.

CLI example:
  python -m scripts.analysis.shortest_ping.plot_margin_sign_bar \
      --features-csv scripts/analysis/outputs/config-test02/shortest_ping/config-test02_shortest_ping_features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REQUIRED_COLS = ["cls_result"]


def _validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if "target_distinguishable_vp_margin_km" not in df.columns and "sign_of_margin" not in df.columns:
        missing.append("target_distinguishable_vp_margin_km|sign_of_margin")
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")


def _to_success_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def _margin_is_positive(df: pd.DataFrame) -> pd.Series:
    if "target_distinguishable_vp_margin_km" in df.columns:
        vals = pd.to_numeric(df["target_distinguishable_vp_margin_km"], errors="coerce")
        return vals > 0

    # Fallback when only sign label exists.
    sign = df["sign_of_margin"].astype(str).str.strip().str.lower()
    return sign == "positive"


def _percent_positive(mask_positive: pd.Series) -> float:
    n = int(mask_positive.notna().sum())
    if n == 0:
        return 0.0
    return float(mask_positive.mean() * 100.0)


def plot_margin_sign_bar(df: pd.DataFrame, out_png: Path, out_pdf: Path | None) -> None:
    data = df.copy()
    success = _to_success_mask(data["cls_result"])
    positive = _margin_is_positive(data)

    buckets = {
        "overall": positive,
        "success": positive[success == True],
        "failure": positive[success == False],
    }

    labels = ["overall", "success", "failure"]
    pct_pos = [_percent_positive(buckets[k]) for k in labels]
    pct_non_pos = [100.0 - v for v in pct_pos]

    x = range(len(labels))
    bar_width = 0.55
    c_pos = "#2ca02c"
    c_non_pos = "#d62728"

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.bar(x, pct_pos, width=bar_width, color=c_pos, label="Positive (> 0)")
    ax.bar(
        x,
        pct_non_pos,
        width=bar_width,
        bottom=pct_pos,
        color=c_non_pos,
        label="Non-positive (<= 0)",
    )

    for i, (p1, p2) in enumerate(zip(pct_pos, pct_non_pos)):
        if p1 > 0:
            ax.text(i, p1 / 2.0, f"{p1:.1f}%", ha="center", va="center", fontsize=9, color="white")
        if p2 > 0:
            ax.text(i, p1 + p2 / 2.0, f"{p2:.1f}%", ha="center", va="center", fontsize=9, color="white")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Targets (%)")
    ax.set_title("Target Distribution by Distinguishable VP Margin Sign")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="center right")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-csv",
        type=Path,
        required=True,
        help="Per-target features CSV produced by per_target_features.py",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="Output PNG path (default: alongside input CSV)",
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=None,
        help="Optional output PDF path (default: alongside input CSV)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv)
    _validate_columns(df, args.features_csv)

    if args.out_png is None:
        args.out_png = args.features_csv.with_name(
            args.features_csv.stem + "_margin_sign_bar.png"
        )
    if args.out_pdf is None:
        args.out_pdf = args.features_csv.with_name(
            args.features_csv.stem + "_margin_sign_bar.pdf"
        )

    plot_margin_sign_bar(df, out_png=args.out_png, out_pdf=args.out_pdf)
    print(f"Saved {args.out_png}")
    print(f"Saved {args.out_pdf}")


if __name__ == "__main__":
    main()
