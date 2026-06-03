"""Deliverable 2: per-setup percentile + fallback summary table.

For each VP setup, builds a table (one row per variant) with:
  - p<N> columns (SUCCESS-only error_km percentiles, derived from cfg.percentiles)
  - fallback_pct (fraction of ALL rows that are FALLBACK × 100)
  - n_success, n_total counts

Outputs (under cfg.out_dir):
  - table_<slug>.png   — matplotlib table figure
  - table_<slug>.json  — ranked rows (same order as the figure)
  - summary_table.md   — combined human-readable markdown (all setups)

Usage::

    python -m scripts.paper.cbg_bench.summary_table
    python -m scripts.paper.cbg_bench.summary_table --slug as7018 --slug as7922
    python -m scripts.paper.cbg_bench.summary_table --rank-by p75 --descending
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from scripts.paper.cbg_bench._io import (
    Config,
    Setup,
    dump_json,
    ensure_out_dir,
    fallback_rate,
    load_config,
    load_setup_long,
)
from scripts.paper.cbg_bench import _variant_style as st

# ---------------------------------------------------------------------------
# Default config path (relative to repo root, resolved at runtime)
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = Path(__file__).resolve().parents[0] / "config" / "four_variants.yaml"


# ---------------------------------------------------------------------------
# Core table logic
# ---------------------------------------------------------------------------

def _build_table_rows(
    cfg: Config,
    setup: Setup,
    rank_by: str,
    descending: bool,
) -> list[dict]:
    """Load data for one setup and return sorted list of row dicts."""
    df_long = load_setup_long(cfg, setup)

    pct_cols = [f"p{int(p)}" for p in cfg.percentiles]
    rows: list[dict] = []

    for combo_id in cfg.variants:
        df_combo = df_long[df_long["combo_id"] == combo_id].copy()
        n_total = len(df_combo)
        fb_rate = fallback_rate(df_combo)
        df_success = df_combo[df_combo["status"] == "SUCCESS"]
        n_success = len(df_success)

        row: dict = {
            "combo_id": combo_id,
            "label": st.label(combo_id),
        }
        for p, col in zip(cfg.percentiles, pct_cols):
            if n_success > 0:
                row[col] = float(np.percentile(df_success["error_km"].dropna(), p))
            else:
                row[col] = float("nan")

        row["fallback_pct"] = fb_rate * 100.0
        row["n_success"] = n_success
        row["n_total"] = n_total
        rows.append(row)

    # Sort
    def _sort_key(r: dict):
        v = r.get(rank_by, float("nan"))
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return float("inf") if not descending else float("-inf")
        return v

    rows.sort(key=_sort_key, reverse=descending)
    return rows


# ---------------------------------------------------------------------------
# PNG renderer
# ---------------------------------------------------------------------------

def _render_png(
    rows: list[dict],
    cfg: Config,
    setup: Setup,
    out_path: Path,
) -> None:
    """Render the ranked table as a readable matplotlib table figure."""
    pct_cols = [f"p{int(p)}" for p in cfg.percentiles]
    col_headers = ["Variant"] + pct_cols + ["fallback %", "n_success", "n_total"]

    cell_data = []
    for row in rows:
        cells = [row["label"]]
        for col in pct_cols:
            v = row.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("—")
            else:
                cells.append(f"{v:.1f}")
        cells.append(f"{row['fallback_pct']:.1f}")
        cells.append(str(row["n_success"]))
        cells.append(str(row["n_total"]))
        cell_data.append(cells)

    n_rows = len(cell_data)
    n_cols = len(col_headers)
    fig_width = max(10, n_cols * 1.6)
    fig_height = max(2, (n_rows + 1) * 0.55 + 1.2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    title = f"{setup.slug}  [{setup.region}]  —  error_km percentiles (SUCCESS only) + fallback %"
    fig.suptitle(title, fontsize=11, y=0.97, ha="center")

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(n_cols)))

    # Style header row
    for col_idx in range(n_cols):
        cell = tbl[0, col_idx]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row shading; tint variant name by color
    combo_ids = [row["combo_id"] for row in rows]
    for row_idx, (combo_id, _) in enumerate(zip(combo_ids, cell_data), start=1):
        base_color = "#f2f2f2" if row_idx % 2 == 0 else "#ffffff"
        for col_idx in range(n_cols):
            cell = tbl[row_idx, col_idx]
            cell.set_facecolor(base_color)
        # Tint variant name cell lightly with the variant color
        name_cell = tbl[row_idx, 0]
        vc = st.color(combo_id)
        # Convert hex to rgba with low alpha for background
        r = int(vc[1:3], 16) / 255
        g = int(vc[3:5], 16) / 255
        b = int(vc[5:7], 16) / 255
        name_cell.set_facecolor((r, g, b, 0.18))
        name_cell.set_text_props(color=vc, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def _rows_to_markdown(rows: list[dict], cfg: Config, setup: Setup) -> str:
    """Render one setup's ranked table as a markdown section."""
    pct_cols = [f"p{int(p)}" for p in cfg.percentiles]
    headers = ["Variant"] + pct_cols + ["fallback %", "n_success", "n_total"]
    sep = ["-" * max(len(h), 8) for h in headers]

    lines: list[str] = [
        f"## {setup.slug}  [{setup.region}]",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        cells = [row["label"]]
        for col in pct_cols:
            v = row.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("—")
            else:
                cells.append(f"{v:.1f}")
        cells.append(f"{row['fallback_pct']:.1f}")
        cells.append(str(row["n_success"]))
        cells.append(str(row["n_total"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(
    config: Path = typer.Option(
        _DEFAULT_CONFIG,
        "--config",
        help="Path to the four-variant config YAML.",
    ),
    slug: Optional[List[str]] = typer.Option(
        None,
        "--slug",
        help="Setup slug to process (repeatable). Defaults to all setups.",
    ),
    rank_by: str = typer.Option(
        "p50",
        "--rank-by",
        help="Column name to rank rows by (e.g. p50, p75, fallback_pct).",
    ),
    descending: bool = typer.Option(
        False,
        "--descending/--ascending",
        help="Sort order: ascending by default.",
    ),
) -> None:
    """Build per-setup percentile + fallback summary tables (PNG, JSON, Markdown)."""
    cfg = load_config(config)
    ensure_out_dir(cfg)

    # Resolve which setups to process
    if slug:
        setups = [cfg.setup_by_slug(s) for s in slug]
    else:
        setups = list(cfg.setups)

    all_md_sections: list[str] = [
        "# CBG Benchmark Summary Table\n",
        f"Rank by: **{rank_by}** ({'descending' if descending else 'ascending'})\n",
    ]

    for setup in setups:
        typer.echo(f"Processing setup: {setup.slug} [{setup.region}]")
        try:
            rows = _build_table_rows(cfg, setup, rank_by=rank_by, descending=descending)
        except FileNotFoundError as exc:
            typer.echo(f"  WARNING: skipping {setup.slug} — {exc}", err=True)
            continue

        # PNG
        png_path = cfg.out_dir / f"table_{setup.slug}.png"
        _render_png(rows, cfg, setup, png_path)
        typer.echo(f"  -> {png_path}")

        # JSON
        json_path = cfg.out_dir / f"table_{setup.slug}.json"
        dump_json(
            {
                "setup": setup.slug,
                "region": setup.region,
                "rank_by": rank_by,
                "descending": descending,
                "rows": rows,
            },
            json_path,
        )
        typer.echo(f"  -> {json_path}")

        # Accumulate markdown
        all_md_sections.append(_rows_to_markdown(rows, cfg, setup))

    # Combined markdown
    md_path = cfg.out_dir / "summary_table.md"
    md_path.write_text("\n".join(all_md_sections))
    typer.echo(f"  -> {md_path}")

    typer.echo("Done.")


if __name__ == "__main__":
    typer.run(main)
