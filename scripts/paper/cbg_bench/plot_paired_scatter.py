"""Within-pair paired error scatter plots for the four fundamental CBG variants.

For each of the two within-pair contrasts defined in _variant_style.VARIANT_PAIRS,
produce one scatter figure where each point is one (setup, target) combination:
  x = error_km of the first variant in the pair
  y = error_km of the second variant in the pair
Points are colored by closest_vp_km (colorbar labeled "closest-VP distance (km)").

Axes are log-log. A y = x diagonal is drawn for reference.
Targets are included only if BOTH variants have status == 'SUCCESS'.

Outputs (under cfg.out_dir):
  paired_<vx>__<vy>.png  – figure
  paired_<vx>__<vy>.json – companion data (error_x, error_y, closest_vp_km,
                            setup, n_points, n_dropped)

Usage::

    python -m scripts.paper.cbg_bench.plot_paired_scatter
    python -m scripts.paper.cbg_bench.plot_paired_scatter --config path/to/config.yaml
    python -m scripts.paper.cbg_bench.plot_paired_scatter --per-setup
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for headless runs)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from scripts.paper.cbg_bench._io import (
    Config,
    Setup,
    dump_json,
    ensure_out_dir,
    load_config,
    load_setup_long,
)
from scripts.paper.cbg_bench import _variant_style as st

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "four_variants.yaml"

# Minimum positive floor to avoid log(0) issues
_LOG_FLOOR = 0.1  # km


def _build_pair_frame(
    cfg: Config,
    variant_x: str,
    variant_y: str,
    setups: Optional[list[Setup]] = None,
) -> tuple[pd.DataFrame, int]:
    """Return a tidy frame for one pair, pooled across all setups.

    Returns (df, n_dropped).
    df columns: error_x, error_y, closest_vp_km, setup (slug).
    Only rows where BOTH variants are 'SUCCESS' are kept.
    """
    if setups is None:
        setups = cfg.setups

    kept_frames: list[pd.DataFrame] = []
    total_candidates = 0

    for setup in setups:
        long = load_setup_long(cfg, setup, variants=[variant_x, variant_y])

        vx_df = long[long["combo_id"] == variant_x].copy()
        vy_df = long[long["combo_id"] == variant_y].copy()

        # Keep only SUCCESS rows
        vx_ok = vx_df[vx_df["status"] == "SUCCESS"][
            ["target_id", "error_km", "closest_vp_km"]
        ].rename(columns={"error_km": "error_x", "closest_vp_km": "closest_vp_km_x"})
        vy_ok = vy_df[vy_df["status"] == "SUCCESS"][
            ["target_id", "error_km"]
        ].rename(columns={"error_km": "error_y"})

        # Count candidates: targets that appear in at least one variant
        total_this_setup = len(
            pd.concat(
                [vx_df["target_id"], vy_df["target_id"]]
            ).drop_duplicates()
        )
        total_candidates += total_this_setup

        # Inner join: only targets where both succeeded
        merged = vx_ok.merge(vy_ok, on="target_id", how="inner")
        if merged.empty:
            continue

        merged["closest_vp_km"] = merged["closest_vp_km_x"]
        merged.drop(columns=["closest_vp_km_x"], inplace=True)
        merged["setup"] = setup.slug
        kept_frames.append(merged)

    if not kept_frames:
        pooled = pd.DataFrame(
            columns=["target_id", "error_x", "error_y", "closest_vp_km", "setup"]
        )
        return pooled, total_candidates

    pooled = pd.concat(kept_frames, ignore_index=True)
    n_dropped = total_candidates - len(pooled)
    return pooled, n_dropped


def _make_scatter(
    df: pd.DataFrame,
    variant_x: str,
    variant_y: str,
    out_path: Path,
    title: str,
) -> None:
    """Render and save one log-log scatter figure."""
    ex = np.maximum(df["error_x"].values, _LOG_FLOOR)
    ey = np.maximum(df["error_y"].values, _LOG_FLOOR)
    cvp = df["closest_vp_km"].values

    fig, ax = plt.subplots(figsize=(6, 5))

    scatter = ax.scatter(
        ex, ey,
        c=cvp,
        cmap="viridis",
        alpha=0.5,
        s=12,
        linewidths=0,
        rasterized=True,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("closest-VP distance (km)", fontsize=9)

    # y = x reference line spanning data range
    lo = min(ex.min(), ey.min())
    hi = max(ex.max(), ey.max())
    ref = np.array([lo, hi])
    ax.plot(ref, ref, "k--", linewidth=0.8, alpha=0.7, label="y = x")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(st.label(variant_x) + " error (km)", fontsize=10)
    ax.set_ylabel(st.label(variant_y) + " error (km)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _stem(variant_x: str, variant_y: str) -> str:
    return f"paired_{variant_x}__{variant_y}"


def _run_pair(
    cfg: Config,
    variant_x: str,
    variant_y: str,
    setups: Optional[list[Setup]],
    out_dir: Path,
    label_suffix: str = "",
) -> None:
    """Build data, save figure and JSON for one (pair, [setup]) combination."""
    typer.echo(f"  pair ({variant_x}, {variant_y}){label_suffix} ...")

    df, n_dropped = _build_pair_frame(cfg, variant_x, variant_y, setups)
    n_points = len(df)

    typer.echo(
        f"    included: {n_points}  dropped (not both SUCCESS): {n_dropped}"
    )

    stem = _stem(variant_x, variant_y) + label_suffix.replace(" ", "_")
    png_path = out_dir / f"{stem}.png"
    json_path = out_dir / f"{stem}.json"

    title = f"{st.label(variant_x)} vs {st.label(variant_y)}"
    if label_suffix:
        title += f"  [{label_suffix.strip()}]"

    if n_points == 0:
        typer.echo("    WARNING: no points to plot — skipping figure.")
        dump_json(
            {
                "pair": [variant_x, variant_y],
                "n_points": 0,
                "n_dropped": n_dropped,
                "error_x": [],
                "error_y": [],
                "closest_vp_km": [],
                "setup": [],
            },
            json_path,
        )
        return

    _make_scatter(df, variant_x, variant_y, png_path, title)

    dump_json(
        {
            "pair": [variant_x, variant_y],
            "n_points": n_points,
            "n_dropped": n_dropped,
            "error_x": df["error_x"].tolist(),
            "error_y": df["error_y"].tolist(),
            "closest_vp_km": df["closest_vp_km"].tolist(),
            "setup": df["setup"].tolist(),
        },
        json_path,
    )
    typer.echo(f"    wrote {png_path.name}  +  {json_path.name}")


def main(
    config: Path = typer.Option(
        _DEFAULT_CONFIG,
        "--config",
        help="Path to the four_variants YAML config.",
        show_default=True,
    ),
    per_setup: bool = typer.Option(
        False,
        "--per-setup",
        help=(
            "Emit one figure per (pair, setup) rather than pooling across all setups. "
            "The pooled figures are always produced; this adds per-setup siblings."
        ),
    ),
) -> None:
    """Produce within-pair paired error scatter plots for the four CBG variants."""
    cfg = load_config(config)
    out_dir = ensure_out_dir(cfg)
    typer.echo(f"Config : {config}")
    typer.echo(f"Out dir: {out_dir}")
    typer.echo(f"Setups : {[s.slug for s in cfg.setups]}")
    typer.echo(f"Pairs  : {st.VARIANT_PAIRS}")
    typer.echo("")

    for variant_x, variant_y in st.VARIANT_PAIRS:
        typer.echo(f"=== Pair: {variant_x} / {variant_y} ===")

        # Pooled across all setups (required deliverable)
        _run_pair(cfg, variant_x, variant_y, setups=None, out_dir=out_dir)

        # Optional per-setup breakdown
        if per_setup:
            for setup in cfg.setups:
                _run_pair(
                    cfg,
                    variant_x,
                    variant_y,
                    setups=[setup],
                    out_dir=out_dir,
                    label_suffix=f" {setup.slug}",
                )

        typer.echo("")

    typer.echo("Done.")


if __name__ == "__main__":
    typer.run(main)
