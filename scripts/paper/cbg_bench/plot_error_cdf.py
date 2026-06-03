"""Deliverable 1 — 6 per-setup SUCCESS-only error CDFs.

One PNG + one JSON per VP setup (slug). Each figure plots the empirical CDF of
`error_km` over SUCCESS-only rows, one curve per variant, with fallback rates
in the legend and reference threshold lines on the x-axis.

Usage::

    # all 6 setups (default config)
    python -m scripts.paper.cbg_bench.plot_error_cdf

    # specific setups only
    python -m scripts.paper.cbg_bench.plot_error_cdf --slug as7018 --slug as7922

    # custom config
    python -m scripts.paper.cbg_bench.plot_error_cdf --config path/to/config.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.ticker import ScalarFormatter

from scripts.paper.cbg_bench._io import (
    Config,
    Setup,
    dump_json,
    ensure_out_dir,
    fallback_rate,
    load_config,
    load_setup_long,
    load_shortest_ping_error,
)
from scripts.paper.cbg_bench import _variant_style as st

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parent / "config" / "four_variants.yaml"
)


def _plot_setup_cdf(
    cfg: Config,
    setup: Setup,
    variants: list[str],
) -> None:
    """Plot SUCCESS-only error CDF for one setup; write PNG + JSON."""
    df = load_setup_long(cfg, setup, variants=variants)

    fig, ax = plt.subplots(figsize=(9, 6))

    json_variants: dict[str, dict] = {}

    for combo_id in variants:
        df_combo = df[df["combo_id"] == combo_id]
        if df_combo.empty:
            logger.warning("No rows for combo=%s setup=%s — skipping", combo_id, setup.slug)
            continue

        fb = fallback_rate(df_combo)

        df_success = df_combo[df_combo["status"] == "SUCCESS"]
        errors = df_success["error_km"].dropna().to_numpy(dtype=float)

        n_success = len(errors)
        n_total = len(df_combo)

        # Build legend label with fallback rate.
        leg_label = f"{st.label(combo_id)} (fallback {fb * 100:.1f}%)"

        if n_success > 0:
            sorted_e = np.sort(errors)
            cdf_y = np.arange(1, n_success + 1) / n_success
            ax.plot(
                sorted_e,
                cdf_y,
                color=st.color(combo_id),
                linewidth=2,
                alpha=0.85,
                label=leg_label,
            )
        else:
            # Still add a legend entry so the reader knows this variant existed.
            ax.plot([], [], color=st.color(combo_id), linewidth=2, alpha=0.85, label=leg_label)
            sorted_e = np.array([], dtype=float)
            cdf_y = np.array([], dtype=float)

        json_variants[combo_id] = {
            "error_km_sorted": sorted_e,
            "cdf_y": cdf_y,
            "n_success": n_success,
            "n_total": n_total,
            "fallback_rate": fb,
        }

    # Shortest-ping baseline (per target: error of the smallest-latency VP).
    # Drawn as a single black dashed reference line, same for every variant.
    sp_errors = (
        load_shortest_ping_error(cfg, setup)["shortest_ping_km"]
        .dropna()
        .to_numpy(dtype=float)
    )
    if sp_errors.size > 0:
        sp_sorted = np.sort(sp_errors)
        sp_cdf_y = np.arange(1, sp_sorted.size + 1) / sp_sorted.size
        ax.plot(
            sp_sorted,
            sp_cdf_y,
            color="black",
            linestyle="--",
            linewidth=1.6,
            alpha=0.9,
            label=f"Shortest ping (p50 {np.median(sp_sorted):.0f} km)",
            zorder=5,
        )
        json_variants["shortest_ping"] = {
            "error_km_sorted": sp_sorted,
            "cdf_y": sp_cdf_y,
            "n_targets": int(sp_sorted.size),
        }

    # Vertical threshold reference lines.
    for thresh in cfg.thresholds_km:
        ax.axvline(
            x=thresh,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.4,
        )

    ax.set_xscale("log")
    x_fmt = ScalarFormatter()
    x_fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(x_fmt)
    ax.set_xlim(left=1, right=20000)
    ax.set_ylim(0, 1)

    ax.set_xlabel("Error distance (km)", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title(f"{setup.slug}  [{setup.region}]  — SUCCESS-only error CDF", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)

    plt.tight_layout()

    png_path = cfg.out_dir / f"cdf_{setup.slug}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PNG: %s", png_path)

    json_path = cfg.out_dir / f"cdf_{setup.slug}.json"
    dump_json(json_variants, json_path)
    logger.info("Saved JSON: %s", json_path)


def main(
    config: Path = typer.Option(
        _DEFAULT_CONFIG,
        "--config",
        help="Path to the four_variants YAML config.",
    ),
    slug: Optional[List[str]] = typer.Option(
        None,
        "--slug",
        help=(
            "Setup slug(s) to plot (repeatable). "
            "Defaults to all setups defined in the config."
        ),
    ),
) -> None:
    """Plot SUCCESS-only error CDFs for each VP setup (one figure per setup)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(config)
    ensure_out_dir(cfg)

    # Resolve which setups to process.
    if slug:
        setups = [cfg.setup_by_slug(s) for s in slug]
    else:
        setups = list(cfg.setups)

    # Resolve variants: VARIANT_ORDER intersected with cfg.variants (preserving order).
    variants = [v for v in st.VARIANT_ORDER if v in cfg.variants]
    if not variants:
        typer.echo("No matching variants found (VARIANT_ORDER ∩ cfg.variants is empty).", err=True)
        raise typer.Exit(code=1)

    for setup in setups:
        typer.echo(f"Processing setup: {setup.slug} [{setup.region}]")
        _plot_setup_cdf(cfg, setup, variants)

    typer.echo(f"\nDone. Output dir: {cfg.out_dir}")


if __name__ == "__main__":
    typer.run(main)
