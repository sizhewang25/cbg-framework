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
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# outputs root written by the v2 runner: scripts/benchmark/v2/outputs/<run_id>/.
# `_v2_io.py` lives at scripts/analysis/, so parents[1] == scripts/.
DEFAULT_OUTPUTS_ROOT = Path(__file__).resolve().parents[1] / "benchmark" / "v2" / "outputs"

# Where analysis scripts write their figures/tables.
ANALYSIS_OUTPUTS_ROOT = Path(__file__).resolve().parent / "outputs"


# ---- geo filter (shared across every analysis script) ------------------------
#
# `geo-eval` annotates each targets.parquet with `target_continent` /
# `target_country` (see scripts/benchmark/v2/geo_eval.py). Setting a process-wide
# geo filter here makes `load_targets` transparently restrict every loaded target
# set to one geography, so *all* analysis scripts inherit geo slicing by simply
# registering the two CLI args and calling `set_geo_filter_from_args` in their
# main() — no change to their data-handling code. Output paths are routed under a
# `geo/<level>/<value>/` segment so filtered artifacts never mix with global ones.

_GEO_LEVEL_COLUMN = {"continent": "target_continent", "country": "target_country"}
_ACTIVE_GEO: Optional[tuple[str, str]] = None  # (level, value)


def add_geo_filter_args(parser) -> None:
    """Register `--geo-level` / `--geo-value` on an argparse parser.

    Call once per analysis-script `main()`, then `set_geo_filter_from_args(args)`.
    """
    g = parser.add_argument_group("geo filter (optional)")
    g.add_argument(
        "--geo-level", choices=tuple(_GEO_LEVEL_COLUMN), default=None,
        help="Restrict every loaded target set to one geography. Requires the "
             "geo-eval columns (run `cli geo-eval --run-id <id>` first). Pair "
             "with --geo-value.",
    )
    g.add_argument(
        "--geo-value", default=None,
        help="Value for --geo-level: a continent name ('Europe', "
             "'North America') or an ISO alpha-2 country code ('US', 'FR').",
    )


def set_geo_filter_from_args(args) -> Optional[tuple[str, str]]:
    """Activate the process-wide geo filter from parsed args (idempotent).

    Returns the active `(level, value)` or None. Raises if exactly one of
    `--geo-level` / `--geo-value` is given.
    """
    level = getattr(args, "geo_level", None)
    value = getattr(args, "geo_value", None)
    if (level is None) != (value is None):
        raise SystemExit("--geo-level and --geo-value must be passed together")
    global _ACTIVE_GEO
    _ACTIVE_GEO = (level, value) if level is not None else None
    return _ACTIVE_GEO


def active_geo_filter() -> Optional[tuple[str, str]]:
    """Current `(level, value)` geo filter, or None."""
    return _ACTIVE_GEO


def geo_segment() -> Optional[Path]:
    """`geo/<level>/<value>` path segment for the active filter, or None."""
    if _ACTIVE_GEO is None:
        return None
    level, value = _ACTIVE_GEO
    safe = str(value).replace(" ", "_").replace("/", "_")
    return Path("geo") / level / safe


def analysis_out_dir(run_dir: Path, *subdirs: str) -> Path:
    """Default analysis output dir for a run, geo segment inserted after run_id.

    `outputs/<run_id>/[geo/<level>/<value>/]<subdirs...>` — so a continent=Europe
    filter routes a script's outputs under `outputs/<run_id>/geo/continent/Europe/`.
    """
    parts: list[str] = [run_dir.name]
    seg = geo_segment()
    if seg is not None:
        parts.append(str(seg))
    return ANALYSIS_OUTPUTS_ROOT.joinpath(*parts, *subdirs)


def route_geo_path(path: Path) -> Path:
    """Insert the active geo segment as a parent of `path`'s final component.

    For explicit `--out` targets the geo filter can't be placed after a run_id
    (there isn't one in the path), so the segment is inserted just above the
    file/dir name: `.../foo.png` → `.../geo/<level>/<value>/foo.png`. No-op when
    no filter is active.
    """
    seg = geo_segment()
    if seg is None:
        return Path(path)
    p = Path(path)
    return p.parent / seg / p.name


def resolve_run_dir(
    config: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    outputs_root: Optional[Path] = None,
) -> Path:
    """Resolve the run directory (`outputs/<run_id>/`) from a benchmark config.

    `run_dir` wins if given (explicit override). Otherwise the `run_id` field is
    read from the config YAML and joined onto `outputs_root` (defaults to the
    v2 runner's outputs tree). This lets the airport-analysis scripts take the
    same config file that drove the benchmark as their single input.
    """
    if run_dir is not None:
        return Path(run_dir)
    if config is None:
        raise ValueError("pass either --config or --run-dir")
    import yaml

    cfg = yaml.safe_load(Path(config).read_text())
    run_id = cfg["run_id"]
    root = Path(outputs_root) if outputs_root is not None else DEFAULT_OUTPUTS_ROOT
    return root / run_id


def discover_combos(
    run_dir: Path,
    source: Optional[str] = None,
    slice_: Optional[str] = None,
    combos: Optional[Iterable[str]] = None,
) -> list[Path]:
    """Return combo directories under `run_dir`, sorted by combo_id.

    A "combo directory" is any directory containing `targets.parquet`. When
    `source` / `slice_` are given, only paths whose components include them
    are kept. When `combos` is given (non-empty iterable of combo_ids), only
    combo directories whose name is in that set are kept — None or empty
    means "include every combo found on disk".

    With `slice_=None` on a K-fold layout (`<source>/<setup>/fold_*/<combo_id>/`)
    the result contains one entry per (fold, combo_id); the same combo_id
    appears K times. Use `group_combos_by_id` to fold those into one entry
    per combo_id for merged-fold analyses.
    """
    combo_dirs = [p.parent for p in run_dir.glob("**/targets.parquet")]
    if source is not None:
        combo_dirs = [d for d in combo_dirs if source in d.parts]
    if slice_ is not None:
        combo_dirs = [d for d in combo_dirs if slice_ in d.parts]
    if combos:
        allowed = set(combos)
        combo_dirs = [d for d in combo_dirs if d.name in allowed]
    return sorted(combo_dirs, key=lambda d: d.name)


def group_combos_by_id(combo_dirs: list[Path]) -> dict[str, list[Path]]:
    """Group `discover_combos` output by combo_id (= directory name).

    With per-fold output layout each combo_id appears once per fold. The
    returned dict maps `combo_id -> [fold_0_dir, fold_1_dir, ...]` with the
    per-combo list sorted by parent dir name (= fold id).
    """
    grouped: dict[str, list[Path]] = {}
    for d in combo_dirs:
        grouped.setdefault(d.name, []).append(d)
    for cid in grouped:
        grouped[cid].sort(key=lambda d: d.parent.name)
    return grouped


def load_targets(combo_dir: Path) -> pa.Table:
    """Read `<combo_dir>/targets.parquet` (TARGETS_SCHEMA).

    When a process-wide geo filter is active (see `set_geo_filter_from_args`),
    rows are restricted to the matching `target_continent` / `target_country`
    in place, so every caller transparently sees only the selected geography.
    """
    tbl = pq.read_table(combo_dir / "targets.parquet")
    if _ACTIVE_GEO is not None:
        level, value = _ACTIVE_GEO
        col = _GEO_LEVEL_COLUMN[level]
        if col not in tbl.schema.names:
            raise KeyError(
                f"{combo_dir / 'targets.parquet'} has no '{col}' column — run "
                "`python -m scripts.benchmark.v2.cli geo-eval --run-id <id>` "
                "first to annotate the geo columns."
            )
        tbl = tbl.filter(pc.equal(tbl.column(col), value))
    return tbl


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
