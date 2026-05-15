"""
Probe-level RTT-vs-error scatter for Vanilla and Million-Scale CBG.

This script reuses the evaluation helpers from evaluate_million_scale.py,
reruns the two CBG pipelines, and tests whether probes with smaller minimum
RTTs tend to have lower geolocation error.

Outputs:
  - scripts/libs/million_scale/outputs/rtt_error_scatter/rtt_error_scatter.png
  - scripts/libs/million_scale/outputs/rtt_error_scatter/rtt_error_data.csv
  - scripts/libs/million_scale/outputs/rtt_error_scatter/rtt_error_summary.json
"""

import json
import math
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# evaluate_million_scale imports Octant helpers at module load time. This
# script does not use Octant, so provide a narrow fallback stub when the
# optional geom_median dependency is unavailable in the local environment.
try:
    from geom_median.numpy import compute_geometric_median as _compute_geometric_median  # noqa: F401
except ModuleNotFoundError:
    geom_median_module = types.ModuleType("geom_median")
    geom_median_numpy_module = types.ModuleType("geom_median.numpy")

    def _missing_compute_geometric_median(*args, **kwargs):
        raise ModuleNotFoundError(
            "geom_median is required for Octant geolocation but not for RTT-error scatter."
        )

    geom_median_numpy_module.compute_geometric_median = _missing_compute_geometric_median
    geom_median_module.numpy = geom_median_numpy_module
    sys.modules.setdefault("geom_median", geom_median_module)
    sys.modules.setdefault("geom_median.numpy", geom_median_numpy_module)

from scripts.libs.million_scale.evaluate_million_scale import (  # noqa: E402
    ASN,
    fit_lp_models,
    load_data,
    run_million_scale_cbg,
    run_vanilla_cbg,
)


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (16, 7)
plt.rcParams["font.size"] = 11


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "rtt_error_scatter"
BIN_WIDTH_MS = 10
MIN_BIN_COUNT = 5


def build_probe_level_dataframe(df_asn, vanilla_results, million_scale_results):
    """Merge per-probe min RTT with per-probe errors from both methods."""
    probe_min_rtt = (
        df_asn.groupby("src_ip", as_index=False)["min_rtt"]
        .min()
        .rename(columns={"src_ip": "probe_ip", "min_rtt": "min_rtt_ms"})
    )

    vanilla_df = (
        pd.DataFrame(vanilla_results)[["probe_ip", "error_km", "intersection", "n_anchors"]]
        .rename(
            columns={
                "error_km": "vanilla_error_km",
                "intersection": "vanilla_intersection",
                "n_anchors": "vanilla_n_anchors",
            }
        )
    )

    million_scale_df = (
        pd.DataFrame(million_scale_results)[["probe_ip", "error_km", "intersection", "n_anchors"]]
        .rename(
            columns={
                "error_km": "million_scale_error_km",
                "intersection": "million_scale_intersection",
                "n_anchors": "million_scale_n_anchors",
            }
        )
    )

    merged = (
        probe_min_rtt.merge(vanilla_df, on="probe_ip", how="inner")
        .merge(million_scale_df, on="probe_ip", how="inner")
        .dropna(subset=["vanilla_error_km", "million_scale_error_km"])
        .sort_values("min_rtt_ms")
        .reset_index(drop=True)
    )

    return merged


def compute_binned_trend(df_plot, error_col, bin_width_ms=BIN_WIDTH_MS, min_bin_count=MIN_BIN_COUNT):
    """Compute fixed-width RTT bins and median error per bin."""
    if df_plot.empty:
        return pd.DataFrame(
            columns=[
                "bin_start_ms",
                "bin_end_ms",
                "bin_mid_ms",
                "count",
                "median_error_km",
                "used_for_overlay",
            ]
        )

    max_rtt = float(df_plot["min_rtt_ms"].max())
    max_bin_edge = max(bin_width_ms, int(math.ceil(max_rtt / bin_width_ms) * bin_width_ms))
    bins = np.arange(0, max_bin_edge + bin_width_ms + 1e-9, bin_width_ms)

    binned = df_plot.assign(
        bin_index=pd.cut(
            df_plot["min_rtt_ms"],
            bins=bins,
            right=False,
            include_lowest=True,
            labels=False,
        )
    )

    trend = (
        binned.dropna(subset=["bin_index"])
        .assign(bin_index=lambda x: x["bin_index"].astype(int))
        .groupby("bin_index", as_index=False)
        .agg(
            count=("probe_ip", "size"),
            median_error_km=(error_col, "median"),
        )
    )

    trend["bin_start_ms"] = trend["bin_index"] * bin_width_ms
    trend["bin_end_ms"] = trend["bin_start_ms"] + bin_width_ms
    trend["bin_mid_ms"] = trend["bin_start_ms"] + (bin_width_ms / 2.0)
    trend["used_for_overlay"] = trend["count"] >= min_bin_count

    return trend[
        [
            "bin_start_ms",
            "bin_end_ms",
            "bin_mid_ms",
            "count",
            "median_error_km",
            "used_for_overlay",
        ]
    ]


def safe_corr(series_a, series_b, method="pearson"):
    """Return a JSON-safe correlation value."""
    corr = series_a.corr(series_b, method=method)
    if pd.isna(corr):
        return None
    return float(corr)


def format_corr(corr):
    """Format correlation values for plot titles."""
    if corr is None:
        return "nan"
    return f"{corr:.2f}"


def summarize_method(df_plot, error_col, trend_df):
    """Build JSON summary for one error series."""
    return {
        "plotted_probes": int(len(df_plot)),
        "median_error_km": float(df_plot[error_col].median()),
        "mean_error_km": float(df_plot[error_col].mean()),
        "p75_error_km": float(df_plot[error_col].quantile(0.75)),
        "p90_error_km": float(df_plot[error_col].quantile(0.90)),
        "pearson_corr_min_rtt_vs_error": safe_corr(df_plot["min_rtt_ms"], df_plot[error_col], method="pearson"),
        "spearman_corr_min_rtt_vs_error": safe_corr(df_plot["min_rtt_ms"], df_plot[error_col], method="spearman"),
        "trend_bins": trend_df.to_dict(orient="records"),
    }


def plot_rtt_error_scatter(df_plot, trend_by_method, output_path):
    """Plot side-by-side RTT-vs-error scatter panels for both methods."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

    x_max = max(float(df_plot["min_rtt_ms"].max()), BIN_WIDTH_MS)
    y_max = max(
        float(df_plot["vanilla_error_km"].max()),
        float(df_plot["million_scale_error_km"].max()),
    )

    x_limit = math.ceil(x_max / BIN_WIDTH_MS) * BIN_WIDTH_MS
    y_limit = max(100.0, y_max * 1.05)

    panels = [
        ("vanilla_error_km", "Vanilla CBG", "black"),
        ("million_scale_error_km", "Million-Scale CBG", "blue"),
    ]

    for ax, (error_col, title, color) in zip(axes, panels):
        pearson_corr = safe_corr(df_plot["min_rtt_ms"], df_plot[error_col], method="pearson")
        spearman_corr = safe_corr(df_plot["min_rtt_ms"], df_plot[error_col], method="spearman")

        ax.scatter(
            df_plot["min_rtt_ms"],
            df_plot[error_col],
            s=35,
            alpha=0.45,
            c=color,
            edgecolors="none",
            label=f"Probes (n={len(df_plot)})",
        )

        trend_df = trend_by_method[error_col]
        overlay_df = trend_df[trend_df["used_for_overlay"]]
        if not overlay_df.empty:
            ax.plot(
                overlay_df["bin_mid_ms"],
                overlay_df["median_error_km"],
                color=color,
                linewidth=2.5,
                marker="o",
                markersize=5,
                markeredgecolor="white",
                label=f"{BIN_WIDTH_MS} ms-bin median (n>={MIN_BIN_COUNT})",
                zorder=4,
            )

        ax.set_title(
            f"{title}\n"
            f"Pearson r={format_corr(pearson_corr)}, "
            f"Spearman r={format_corr(spearman_corr)}",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Probe Minimum RTT (ms)", fontsize=12)
        ax.set_xlim(0, x_limit)
        ax.set_ylim(0, y_limit)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

    axes[0].set_ylabel("Geolocation Error (km)", fontsize=12)
    fig.suptitle(
        f"Probe Minimum RTT vs Geolocation Error - AS{ASN}",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    _, df_asn = load_data()

    print("\n" + "=" * 60)
    print("FITTING LP MODELS")
    print("=" * 60)
    lp_models = fit_lp_models(df_asn)

    print("\n" + "=" * 60)
    print("RUNNING VANILLA CBG")
    print("=" * 60)
    vanilla_results, _, _ = run_vanilla_cbg(df_asn, lp_models)

    print("\n" + "=" * 60)
    print("RUNNING MILLION-SCALE CBG")
    print("=" * 60)
    million_scale_results, _, _ = run_million_scale_cbg(df_asn)

    print("\n" + "=" * 60)
    print("BUILDING PROBE-LEVEL RTT-ERROR TABLE")
    print("=" * 60)
    df_plot = build_probe_level_dataframe(df_asn, vanilla_results, million_scale_results)
    if df_plot.empty:
        raise RuntimeError("No probes have valid errors for both Vanilla and Million-Scale CBG.")
    print(f"  Plotted probes: {len(df_plot)}")
    print(f"  Probe minimum RTT range: {df_plot['min_rtt_ms'].min():.2f} to {df_plot['min_rtt_ms'].max():.2f} ms")
    print(f"  Vanilla median error: {df_plot['vanilla_error_km'].median():.1f} km")
    print(f"  Million-Scale median error: {df_plot['million_scale_error_km'].median():.1f} km")

    trend_by_method = {
        "vanilla_error_km": compute_binned_trend(df_plot, "vanilla_error_km"),
        "million_scale_error_km": compute_binned_trend(df_plot, "million_scale_error_km"),
    }

    csv_path = OUTPUT_DIR / "rtt_error_data.csv"
    df_plot.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    png_path = OUTPUT_DIR / "rtt_error_scatter.png"
    fig = plot_rtt_error_scatter(df_plot, trend_by_method, png_path)
    plt.close(fig)

    summary = {
        "asn": ASN,
        "rtt_metric": "probe_min_rtt_ms",
        "plotted_probes": int(len(df_plot)),
        "bin_width_ms": BIN_WIDTH_MS,
        "min_bin_count": MIN_BIN_COUNT,
        "rtt_range_ms": {
            "min": float(df_plot["min_rtt_ms"].min()),
            "max": float(df_plot["min_rtt_ms"].max()),
        },
        "methods": {
            "vanilla_cbg": summarize_method(df_plot, "vanilla_error_km", trend_by_method["vanilla_error_km"]),
            "million_scale_cbg": summarize_method(
                df_plot,
                "million_scale_error_km",
                trend_by_method["million_scale_error_km"],
            ),
        },
    }

    json_path = OUTPUT_DIR / "rtt_error_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
