"""Shared loaders + palette helper for v2-benchmark-driven analysis plots.

Reads the parquets/JSON written by `scripts/benchmark/v2/` against the schemas
in `scripts/benchmark/v2/schema.py`. Layout-agnostic: combo discovery globs
`**/targets.parquet` under the run root, so it works for both
`<run_id>/<source>/<slice>/<combo_id>/` (older runs) and
`<run_id>/<source>/<setup_id>/<slice_id>/<combo_id>/` (newer runs).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq


def discover_combos(
    run_dir: Path,
    source: Optional[str] = None,
    slice_: Optional[str] = None,
) -> list[Path]:
    """Return combo directories under `run_dir`, sorted by combo_id.

    A "combo directory" is any directory containing `targets.parquet`. When
    `source` / `slice_` are given, only paths whose components include them
    are kept.
    """
    combo_dirs = [p.parent for p in run_dir.glob("**/targets.parquet")]
    if source is not None:
        combo_dirs = [d for d in combo_dirs if source in d.parts]
    if slice_ is not None:
        combo_dirs = [d for d in combo_dirs if slice_ in d.parts]
    return sorted(combo_dirs, key=lambda d: d.name)


def load_targets(combo_dir: Path) -> pa.Table:
    """Read `<combo_dir>/targets.parquet` (TARGETS_SCHEMA)."""
    return pq.read_table(combo_dir / "targets.parquet")


def load_summary(run_dir: Path) -> pa.Table:
    """Read `<run_dir>/summary.parquet` (SUMMARY_SCHEMA)."""
    return pq.read_table(run_dir / "summary.parquet")


def load_run_json(combo_dir: Path) -> dict:
    """Read `<combo_dir>/run.json` (combo metadata + fit stats + peak RSS)."""
    return json.loads((combo_dir / "run.json").read_text())


def palette(combo_ids: list[str]) -> dict[str, str]:
    """Deterministic combo→hex mapping from matplotlib `tab20`.

    Sorted by combo_id so the same combo gets the same color across re-runs
    and across the three plots.
    """
    cmap = plt.get_cmap("tab20")
    sorted_ids = sorted(combo_ids)
    return {
        cid: _to_hex(cmap(i % cmap.N))
        for i, cid in enumerate(sorted_ids)
    }


def _to_hex(rgba: tuple[float, float, float, float]) -> str:
    r, g, b, _ = rgba
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))
