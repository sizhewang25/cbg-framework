"""Study 1 — error_km vs closest-VP distance, SOI as the geometry baseline.

Re-states deliverable 3 on **log–log, equal-scale axes** so a ``y=x`` line reads
directly ("did CBG localize tighter than the distance to its nearest VP?"), with
the threshold distances drawn as reference lines on **both** axes. Figures are
clean scatter plots — no binned overlay; binned p50/p90 are computed into the
JSON only (to quantify the ramp-vs-cliff shape for the report).

Per setup we emit two figures + one JSON under ``cfg.out_dir``:
  study1_soi_<slug>.png    — SOI hero panel (the main explanator)
  study1_facet_<slug>.png  — 2×2 cross-method facet (shared axes)
  study1_<slug>.json       — per-variant arrays + binned summary + axis settings

Usage::

    python -m scripts.paper.cbg_bench.plot_study1_distance
    python -m scripts.paper.cbg_bench.plot_study1_distance --slug as7018 --slug as16509
    python -m scripts.paper.cbg_bench.plot_study1_distance --config path/to/other.yaml
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import typer

from scripts.paper.cbg_bench._io import (
    Config,
    Setup,
    diagonal_split,
    dump_json,
    ensure_out_dir,
    load_config,
    load_setup_long,
)
from scripts.paper.cbg_bench import _variant_style as st

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "four_variants.yaml"

# threshold reference lines — shared palette with plot_error_vs_vp_dist.py
_THRESHOLD_STYLE = dict(linewidth=0.7, linestyle="--", alpha=0.45, zorder=1)
_THRESHOLD_COLORS = {
    100: "green",
    500: "goldenrod",
    1000: "orange",
    2500: "tomato",
    5000: "red",
}
_LOG_TICKS = [1, 10, 100, 500, 1000, 2500, 5000, 10000, 25000]


# ---- shared axis styling -----------------------------------------------------

def _style_axes(ax, lo: float, hi: float, thresholds_km: list[float]) -> None:
    """Apply log–log equal-scale styling + threshold gridlines on both axes."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_box_aspect(1)  # square box + identical limits ⇒ y=x at 45°

    ticks = [t for t in _LOG_TICKS if lo <= t <= hi]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(mticker.ScalarFormatter())
        axis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # y=x reference (diagonal)
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.9, linestyle="-",
            alpha=0.6, zorder=4)

    # threshold lines on BOTH axes
    for km in thresholds_km:
        tc = _THRESHOLD_COLORS.get(int(km), "gray")
        ax.axvline(km, color=tc, **_THRESHOLD_STYLE)
        ax.axhline(km, color=tc, label=f"{km} km", **_THRESHOLD_STYLE)

    ax.set_xlabel("Closest-VP distance (km)", fontsize=10)
    ax.set_ylabel("Error (km)", fontsize=10)
    ax.grid(True, which="major", linewidth=0.4, alpha=0.4)


def _success_xy(df_v):
    """SUCCESS-only (closest_vp_km, error_km) arrays for one variant frame."""
    df_ok = df_v[df_v["status"] == "SUCCESS"]
    x = df_ok["closest_vp_km"].to_numpy(dtype=float)
    y = df_ok["error_km"].to_numpy(dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


# ---- per-setup rendering -----------------------------------------------------

def _build_json(
    df_long,
    variant_order: list[str],
    lo: float,
    hi: float,
    thresholds_km: list[float],
) -> dict:
    """Assemble the per-setup JSON payload.

    Each variant carries the SUCCESS-only scatter arrays plus a ``diagonal_split``
    summarising position relative to the ``error = closest-VP distance`` line.
    """
    payload: dict = {
        "axis_lo_km": lo,
        "axis_hi_km": hi,
        "threshold_lines_km": list(thresholds_km),
        "variants": {},
    }
    for variant in variant_order:
        df_v = df_long[df_long["combo_id"] == variant]
        n_total = int(len(df_v))
        x, y = _success_xy(df_v)
        payload["variants"][variant] = {
            "closest_vp_km": x.tolist(),
            "error_km": y.tolist(),
            "n_success": int(x.size),
            "n_total": n_total,
            "diagonal_split": diagonal_split(x, y, tol=0.02),
        }
    return payload


def _scatter_panel(ax, variant: str, df_v, lo: float, hi: float,
                   thresholds_km: list[float], with_legend: bool = False) -> None:
    """Draw one variant's SUCCESS-only scatter on a styled log–log panel."""
    x, y = _success_xy(df_v)
    n_total = int(len(df_v))
    n_fallback = n_total - int(x.size)

    _style_axes(ax, lo, hi, thresholds_km)
    ax.scatter(x, y, color=st.color(variant), s=5, alpha=0.55, linewidths=0,
               zorder=3)

    note = f"fallback={n_fallback}/{n_total}" if n_fallback > 0 else f"n={int(x.size)}"
    ax.set_title(f"{st.label(variant)}  [{note}]", fontsize=11)
    if with_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=7, loc="lower right", framealpha=0.85)


def _render_setup(cfg: Config, setup: Setup, df_long, variant_order: list[str],
                  lo: float, hi: float) -> None:
    thresholds = cfg.thresholds_km
    region_tag = f" [{setup.region}]" if setup.region else ""

    # --- SOI hero panel ---
    soi = "million_scale_cbg"
    if soi in variant_order:
        fig, ax = plt.subplots(figsize=(7, 7))
        _scatter_panel(ax, soi, df_long[df_long["combo_id"] == soi],
                       lo, hi, thresholds, with_legend=True)
        ax.set_title(f"{st.label(soi)} — geometry baseline — {setup.slug}{region_tag}",
                     fontsize=12)
        fig.tight_layout()
        hero_png = cfg.out_dir / f"study1_soi_{setup.slug}.png"
        fig.savefig(hero_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", hero_png)

    # --- cross-method facet (2×2, shared axes) ---
    ncols = 2
    nrows = (len(variant_order) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()
    for idx, variant in enumerate(variant_order):
        _scatter_panel(axes_flat[idx], variant,
                       df_long[df_long["combo_id"] == variant],
                       lo, hi, thresholds, with_legend=(idx == 0))
    for idx in range(len(variant_order), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"Error vs Closest-VP Distance (log–log) — {setup.slug}{region_tag}",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    facet_png = cfg.out_dir / f"study1_facet_{setup.slug}.png"
    fig.savefig(facet_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", facet_png)

    # --- JSON ---
    payload = _build_json(df_long, variant_order, lo, hi, thresholds)
    out_json = cfg.out_dir / f"study1_{setup.slug}.json"
    dump_json(payload, out_json)
    logger.info("Saved %s", out_json)


# ---- axis range resolution ---------------------------------------------------

def _resolve_axis_hi(cfg: Config, loaded: Dict[str, "object"],
                     variant_order: list[str]) -> float:
    """Shared upper limit: config override, else ceil(max SUCCESS error/5000)*5000
    across ALL config setups (so panels and setups share identical axes)."""
    if cfg.axis_hi_km is not None:
        return float(cfg.axis_hi_km)

    max_err = 0.0
    for s in cfg.setups:
        df = loaded.get(s.slug)
        if df is None:
            try:
                df = load_setup_long(cfg, s, variants=variant_order)
            except FileNotFoundError:
                continue
        ok = df[df["status"] == "SUCCESS"]["error_km"].to_numpy(dtype=float)
        ok = ok[np.isfinite(ok)]
        if ok.size:
            max_err = max(max_err, float(ok.max()))

    if max_err <= 0:
        return 25000.0
    return math.ceil(max_err / 5000.0) * 5000.0


# ---- CLI ---------------------------------------------------------------------

def main(
    config: Path = typer.Option(
        _DEFAULT_CONFIG, "--config", help="Path to the four-variants YAML config.",
        show_default=True,
    ),
    slug: Optional[List[str]] = typer.Option(
        None, "--slug", help="Setup slug(s) to plot (repeatable). Default: all.",
    ),
) -> None:
    """Plot Study 1 (error vs closest-VP distance, log–log) per setup."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(config)
    ensure_out_dir(cfg)

    requested = set(slug) if slug else None
    setups_to_run = [s for s in cfg.setups if requested is None or s.slug in requested]
    if not setups_to_run:
        typer.echo(f"No matching setups. Available: {[s.slug for s in cfg.setups]}",
                   err=True)
        raise typer.Exit(code=1)

    variant_order = [v for v in st.VARIANT_ORDER if v in cfg.variants] or list(cfg.variants)

    # load requested setups (cache for axis-hi computation)
    loaded: Dict[str, object] = {}
    for s in setups_to_run:
        try:
            loaded[s.slug] = load_setup_long(cfg, s, variants=variant_order)
        except FileNotFoundError as exc:
            typer.echo(f"[{s.slug}] SKIP — {exc}", err=True)

    if not loaded:
        typer.echo("No setup data found.", err=True)
        raise typer.Exit(code=1)

    lo = float(cfg.axis_lo_km)
    hi = _resolve_axis_hi(cfg, loaded, variant_order)
    typer.echo(f"Shared log axes: [{lo:g}, {hi:g}] km")

    for s in setups_to_run:
        df_long = loaded.get(s.slug)
        if df_long is None:
            continue
        typer.echo(f"[{s.slug}] rendering {len(variant_order)} variants...")
        _render_setup(cfg, s, df_long, variant_order, lo, hi)

    typer.echo(f"Done. Output dir: {cfg.out_dir}")


if __name__ == "__main__":
    typer.run(main)
