"""Error-km vs closest-VP-distance scatter, faceted by variant.

One figure per VP setup (6 total).  Each figure contains a 2×2 grid of
sub-plots — one per variant — where:

  X axis : closest-VP great-circle distance (km), ascending
  Y axis : error_km (log scale by default)

Only SUCCESS rows are plotted; the fallback count for each variant is
annotated in the subplot title.  Horizontal dashed reference lines mark
the threshold distances from the config.

Outputs under `cfg.out_dir` (scripts/paper/cbg_bench/<run_id>/):
  error_vs_vpdist_<slug>.png
  error_vs_vpdist_<slug>.json   — per-variant arrays + n_success / n_total

Usage::

    python -m scripts.paper.cbg_bench.plot_error_vs_vp_dist
    python -m scripts.paper.cbg_bench.plot_error_vs_vp_dist --slug as7018 --slug as3209
    python -m scripts.paper.cbg_bench.plot_error_vs_vp_dist --no-log-y
    python -m scripts.paper.cbg_bench.plot_error_vs_vp_dist --config path/to/other.yaml
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import typer

from scripts.paper.cbg_bench._io import (
    dump_json,
    ensure_out_dir,
    load_config,
    load_setup_long,
)
from scripts.paper.cbg_bench import _variant_style as st

logger = logging.getLogger(__name__)

# ---- threshold reference lines (matching plot_per_target_sorted.py) ----------

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parent / "config" / "four_variants.yaml"
)

_THRESHOLD_STYLE = dict(linewidth=0.7, linestyle="--", alpha=0.45, zorder=1)
_THRESHOLD_COLORS = {
    100: "green",
    500: "goldenrod",
    1000: "orange",
    2500: "tomato",
    5000: "red",
}


# ---- per-setup figure --------------------------------------------------------

def _plot_setup(
    slug: str,
    df_long,
    variant_order: list[str],
    thresholds_km: list[float],
    out_png: Path,
    *,
    log_y: bool = True,
    setup_region: str = "",
) -> dict:
    """Draw 2×2 facet figure; return the JSON data payload."""
    n_variants = len(variant_order)
    ncols = 2
    nrows = (n_variants + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 5 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    json_data: dict = {}

    for idx, variant in enumerate(variant_order):
        ax = axes_flat[idx]
        df_v = df_long[df_long["combo_id"] == variant].copy()
        n_total = len(df_v)
        df_ok = df_v[df_v["status"] == "SUCCESS"].copy()
        n_success = len(df_ok)
        n_fallback = n_total - n_success

        # sort ascending by closest_vp_km
        df_ok = df_ok.sort_values("closest_vp_km").reset_index(drop=True)

        x = df_ok["closest_vp_km"].to_numpy()
        y = df_ok["error_km"].to_numpy()

        c = st.color(variant)

        # scatter (primary) + light connecting line
        ax.plot(x, y, color=c, linewidth=0.4, alpha=0.25, zorder=2)
        ax.scatter(x, y, color=c, s=4, alpha=0.55, linewidths=0, zorder=3)

        # threshold reference lines
        for km in thresholds_km:
            tc = _THRESHOLD_COLORS.get(int(km), "gray")
            ax.axhline(km, color=tc, label=f"{km} km", **_THRESHOLD_STYLE)

        # axes formatting
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            yticks = [t for t in [10, 50, 100, 500, 1000, 2500, 5000, 10000, 20000]
                      if y.size == 0 or t <= max(y.max(), 1) * 1.5]
            if yticks:
                ax.set_yticks(yticks)

        ax.set_xlabel("Closest-VP distance (km)", fontsize=10)
        ax.set_ylabel("Error (km)", fontsize=10)
        ax.grid(True, which="major", linewidth=0.4, alpha=0.5)
        ax.grid(True, which="minor", linewidth=0.2, alpha=0.25)

        # title: variant label + fallback annotation
        fallback_note = (
            f"fallback={n_fallback}/{n_total}" if n_fallback > 0 else f"n={n_success}"
        )
        ax.set_title(f"{st.label(variant)}  [{fallback_note}]", fontsize=11)

        # threshold legend inside first subplot only
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, fontsize=7, loc="upper left", framealpha=0.8)

        # JSON payload for this variant
        json_data[variant] = {
            "closest_vp_km": x.tolist(),
            "error_km": y.tolist(),
            "n_success": int(n_success),
            "n_total": int(n_total),
        }

    # hide any unused subplots (if variants < nrows*ncols)
    for idx in range(n_variants, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    region_tag = f" [{setup_region}]" if setup_region else ""
    fig.suptitle(
        f"Error vs Closest-VP Distance — {slug}{region_tag}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    logger.info("Saved %s", out_png)
    plt.close(fig)
    return json_data


# ---- CLI ---------------------------------------------------------------------

def main(
    config: Path = typer.Option(
        _DEFAULT_CONFIG,
        "--config",
        help="Path to the four-variants YAML config.",
        show_default=True,
    ),
    slug: Optional[List[str]] = typer.Option(
        None,
        "--slug",
        help="Setup slug(s) to plot (repeatable). Default: all setups.",
    ),
    log_y: bool = typer.Option(
        True,
        "--log-y/--no-log-y",
        help="Use log scale on the Y axis (default: on).",
    ),
) -> None:
    """Plot error_km vs closest-VP-distance, faceted by variant.

    Produces one PNG + one JSON per setup under cfg.out_dir.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(config)
    ensure_out_dir(cfg)

    # resolve which setups to run
    requested = set(slug) if slug else None
    setups_to_run = [
        s for s in cfg.setups
        if requested is None or s.slug in requested
    ]
    if not setups_to_run:
        typer.echo(
            f"No matching setups. Available: {[s.slug for s in cfg.setups]}",
            err=True,
        )
        raise typer.Exit(code=1)

    # determine variant order: intersection of VARIANT_ORDER and cfg.variants,
    # preserving canonical order
    variant_order = [v for v in st.VARIANT_ORDER if v in cfg.variants]
    if not variant_order:
        # fallback: whatever is in cfg.variants
        variant_order = list(cfg.variants)

    for setup in setups_to_run:
        typer.echo(f"[{setup.slug}] loading data...")
        try:
            df_long = load_setup_long(cfg, setup, variants=variant_order)
        except FileNotFoundError as exc:
            typer.echo(f"[{setup.slug}] SKIP — {exc}", err=True)
            continue

        out_png = cfg.out_dir / f"error_vs_vpdist_{setup.slug}.png"
        out_json = cfg.out_dir / f"error_vs_vpdist_{setup.slug}.json"

        typer.echo(f"[{setup.slug}] plotting {len(variant_order)} variants → {out_png.name}")
        json_data = _plot_setup(
            slug=setup.slug,
            df_long=df_long,
            variant_order=variant_order,
            thresholds_km=cfg.thresholds_km,
            out_png=out_png,
            log_y=log_y,
            setup_region=setup.region,
        )

        dump_json(json_data, out_json)
        typer.echo(f"[{setup.slug}] JSON → {out_json.name}")

    typer.echo("Done.")


if __name__ == "__main__":
    typer.run(main)
