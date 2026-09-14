"""Cut a public and a private dataset down to the places they both occupy.

Operationalizes the paper's **best-effort VP-topology matching** (§7.3): the public RIPE
slice and the proprietary operator mesh are reduced to a common footprint, so a
cross-dataset accuracy difference is less likely to be an artifact of VP placement
alone.

**Reciprocal means both sides are filtered.** Neither dataset is a mask for the other:
each keeps the nodes whose place the *other* also occupies, and both filtered datasets
are written out. That symmetry is the point -- a public node with no private
counterpart and a private node with no public counterpart are the same kind of
mismatch, and leaving either in would leave the two datasets describing different
geography.

Matching is **per side and independent**: VPs are matched against VPs, targets against
targets. A VP being near the other dataset's target says nothing about topology overlap,
so collapsing the two roles would answer a different question and inflate retention.

The merge scale is a grid cell, not a clustering radius
-------------------------------------------------------
Two nodes count as the same place when they land in the same cell of a fixed
tessellation -- H3 resolution 4 (~45 km centre-to-centre) by default, which is the same
quantizer §7.4 uses to build the answer space. Reusing it means "same place" means one
thing across the codebase rather than two.

This replaces an earlier complete-linkage agglomerative approach. A grid is a
*partition*, and that is the property that matters here: cell membership depends on a
node's own coordinate and nothing else, so adding, removing or reordering nodes cannot
reshape anyone else's region, there is no seed and no RNG, and the result is
reproducible from the inputs alone. Agglomerative merging has none of those: it is
data-dependent by construction, needs a full pairwise distance matrix, and needs a
repair pass to bound a region's radius. Its cost is that grid lines fall where the grid
falls, so two nodes 5 km apart across a boundary are different places -- the same
straddling cost §7.4.2 states for the answer space, and the reason `common_cells` is
reported rather than assumed away.

Retention is presence-only: every node whose cell the other dataset also occupies
survives. There is deliberately no per-cell count balancing -- that would need an RNG
and make the output seed-dependent, and the per-cell density mismatch is reported in the
summary instead of being silently corrected.

An edge survives only if **both** of its endpoints do, so pruning nodes prunes flows. As
with `filter_weighted_flows`, that can push a surviving target below the 3 observing VPs
multilateration needs; it is reported as `targets_out_by_vp_count` and deliberately not
enforced here.

CLI::

    .venv/bin/python -m scripts.processing.source.reciprocal_grid_filter \
        --public  datasets/ripe_as7018/as7018-us-test01.csv \
        --private datasets/final/as01-20260728-20260802.mainland.sanitized.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from scripts.analysis.v3.modules.grid import (
    DEFAULT_GRID,
    GRID_HELP,
    GRID_NAMES,
    Grid,
    get_grid,
)

logger = logging.getLogger(__name__)

_OUTPUT_SUFFIX = ".reciprocal.csv"
_SUMMARY_SUFFIX = ".summary.json"

#: The two datasets, filtered symmetrically. Order fixes the summary's key order and the
#: order of the log lines; it carries no precedence.
_DATASETS = ("public", "private")

#: Each dataset is matched against the other's cells -- this is the reciprocity, so it
#: is named rather than spelled out as an index dance at the one place it is used.
_OTHER = {"public": "private", "private": "public"}

#: The two node roles, each matched only against its own kind.
_SIDES = ("vp", "target")

#: Per side, what both datasets must carry. `rtt_ms` is deliberately absent: matching is
#: geometry only, so requiring RTTs would reject a perfectly good coordinate-only set.
_GEO_COLUMNS = {side: (f"{side}_id", f"{side}_lat", f"{side}_lon") for side in _SIDES}


def _default_output(input_path: Path) -> Path:
    # NOT `with_suffix("").with_suffix(...)`: on a multi-dot stem like
    # `x.mainland.sanitized.csv` that idiom eats `.sanitized`.
    return input_path.with_name(input_path.stem + _OUTPUT_SUFFIX)


def _default_summary(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + _SUMMARY_SUFFIX)


def read_canonical_csv(path: Path) -> pd.DataFrame:
    """Read a canonical CSV as **text**, every column, every value exactly as written.

    This filter only ever selects whole rows and writes them back, so nothing here needs
    a parsed frame: the numeric work happens on a private copy inside `node_frame`.
    Reading as `str` therefore makes the pass-through exact, and rules out three ways a
    parse-then-reserialize round trip silently edits data it was only meant to subset:

    * **Float precision.** `to_csv` formats float64 to 16 significant digits, which is
      one digit short of a guaranteed round trip. On the AS7018 slice that rewrote 7 of
      4129 RTTs to a neighbouring double -- physically meaningless at 1e-16 ms, but a
      filter that edits values is not a filter.
    * **The NA sentinel.** A literal `"NA"` -- North America, or Namibia -- parses to
      NaN and writes back as an empty cell. `keep_default_na=False` is a stronger
      version of the `_raw_str` converter `generic_csv` applies for the same reason.
    * **Schema.** Deliberately not `eval_source.load_canonical_csv`, which projects down
      to the required columns plus three optionals; a filtered dataset must keep every
      column its input had.

    The one thing not preserved is the line terminator: `to_csv` writes LF where the
    RIPE canonical CSVs carry CRLF, matching the sibling scripts in this package.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip().str.lower()
    return df


def node_frame(df: pd.DataFrame, side: str, *, label: str) -> pd.DataFrame:
    """One row per distinct node id, with its coordinate.

    Raises when an id carries more than one coordinate -- the same integrity check
    `bipartite.load_edges` makes. Without it a node quantizes into two different cells
    and its survival depends on which duplicate row pandas happened to keep.
    """
    id_col, lat_col, lon_col = _GEO_COLUMNS[side]
    missing = [c for c in _GEO_COLUMNS[side] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing {side} columns {missing}; present: {list(df.columns)}"
        )

    # A private numeric copy: the caller's frame stays text so it can be written back
    # byte-for-byte.
    nodes = df.loc[:, [id_col, lat_col, lon_col]].copy()
    nodes[id_col] = nodes[id_col].astype(str)
    for c in (lat_col, lon_col):
        nodes[c] = pd.to_numeric(nodes[c], errors="coerce")
    if nodes[[lat_col, lon_col]].isna().any(axis=None):
        bad = nodes.loc[nodes[[lat_col, lon_col]].isna().any(axis=1), id_col]
        raise ValueError(
            f"{label} has {len(bad)} {side} rows with a missing/non-numeric coordinate "
            f"(e.g. {bad.head(3).tolist()})"
        )

    nodes = nodes.drop_duplicates().sort_values(id_col).reset_index(drop=True)
    conflicting = nodes[id_col][nodes[id_col].duplicated()].unique()
    if len(conflicting):
        raise ValueError(
            f"{label}: {len(conflicting)} {side} ids carry more than one coordinate "
            f"(e.g. {conflicting[:3].tolist()}). One node would land in two cells, so "
            f"its survival would depend on row order -- fix the export."
        )
    return nodes


def match_side(
    nodes: dict[str, pd.DataFrame],
    side: str,
    *,
    grid: Grid,
    resolution: int,
) -> tuple[dict[str, pd.Series], dict]:
    """Reciprocal node match for one role. Returns (surviving ids per dataset, report).

    Both datasets are quantized on the *same* grid at the *same* resolution; that shared
    binning is the entire matching criterion. Each dataset is then masked by the
    **other's** occupied cells, which is where the symmetry actually lives.
    """
    _, lat_col, lon_col = _GEO_COLUMNS[side]
    parts = {
        name: grid.partition(nodes[name][lat_col], nodes[name][lon_col], resolution)
        for name in _DATASETS
    }

    keep, report = {}, {}
    for name in _DATASETS:
        mask = parts[name].member_mask(parts[_OTHER[name]].cells)
        n_in, n_out = len(nodes[name]), int(mask.sum())
        keep[name] = nodes[name].loc[mask, f"{side}_id"]
        report[f"{name}_nodes_in"] = n_in
        report[f"{name}_nodes_out"] = n_out
        report[f"{name}_nodes_dropped"] = n_in - n_out
        report[f"{name}_retention_pct"] = round(100 * n_out / n_in, 3) if n_in else 0.0
        report[f"{name}_cells"] = parts[name].n_occupied

    common = set(map(str, parts["public"].cells)) & set(map(str, parts["private"].cells))
    report["common_cells"] = len(common)
    for name in _DATASETS:
        report[f"{name}_cells_unmatched"] = parts[name].n_occupied - len(common)
    return keep, report


def _dataset_report(kept: pd.DataFrame, n_rows_in: int) -> dict:
    """Edge-level outcome for one filtered dataset.

    `targets_out_by_vp_count` is the min-VP caveat: multilateration needs 3 constraints
    and node pruning can push a surviving target under that. Reported, not enforced --
    the same deliberate policy as `filter_weighted_flows`.
    """
    vp_per_tg = kept.groupby("target_id")["vp_id"].nunique()
    buckets = {"1": 0, "2": 0, "3+": 0}
    for n in vp_per_tg:
        buckets["1" if n == 1 else "2" if n == 2 else "3+"] += 1
    return {
        "rows_in": int(n_rows_in),
        "rows_out": int(len(kept)),
        "rows_retention_pct": (
            round(100 * len(kept) / n_rows_in, 3) if n_rows_in else 0.0
        ),
        "vps_out": int(kept["vp_id"].nunique()),
        "targets_out": int(kept["target_id"].nunique()),
        "targets_out_by_vp_count": buckets,
        "targets_out_with_ge_3_vps": int(buckets["3+"]),
    }


def reciprocal_filter(
    public: pd.DataFrame,
    private: pd.DataFrame,
    *,
    grid: Grid,
    resolution: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Filter both datasets to their common footprint.

    Returns `(kept_public, kept_private, summary)`. Columns and row order are preserved
    in both.
    """
    resolution = grid.validate_resolution(resolution)
    frames = {"public": public, "private": private}

    survivors: dict[str, dict[str, pd.Series]] = {}
    cells_report: dict[str, dict] = {}
    for side in _SIDES:
        nodes = {
            name: node_frame(frames[name], side, label=name) for name in _DATASETS
        }
        survivors[side], cells_report[side] = match_side(
            nodes, side, grid=grid, resolution=resolution
        )

    kept = {}
    for name in _DATASETS:
        # An edge needs both endpoints: keeping a row whose VP survived but whose target
        # did not would reintroduce exactly the unmatched geography this removes.
        mask = pd.Series(True, index=frames[name].index)
        for side in _SIDES:
            mask &= (
                frames[name][f"{side}_id"]
                .astype(str)
                .isin(set(survivors[side][name]))
            )
        kept[name] = frames[name][mask].copy()

    summary = {
        "grid": grid.describe(resolution),
        "cells": cells_report,
        **{
            name: _dataset_report(kept[name], len(frames[name]))
            for name in _DATASETS
        },
    }
    return kept["public"], kept["private"], summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--public", type=Path, required=True,
                        help="Public canonical CSV (e.g. the RIPE slice).")
    parser.add_argument("--private", type=Path, required=True,
                        help="Private/proprietary canonical CSV (e.g. the operator mesh).")
    parser.add_argument("--grid", default=DEFAULT_GRID, choices=list(GRID_NAMES),
                        help=GRID_HELP)
    parser.add_argument("--resolution", type=int, default=None,
                        help="Grid resolution. Defaults to the chosen grid's own default "
                             "(h3 res 4, ~45 km).")
    parser.add_argument("--out-public", type=Path, default=None,
                        help=f"Filtered public CSV. Defaults to <public-stem>{_OUTPUT_SUFFIX}.")
    parser.add_argument("--out-private", type=Path, default=None,
                        help=f"Filtered private CSV. Defaults to <private-stem>{_OUTPUT_SUFFIX}.")
    parser.add_argument("--summary", type=Path, default=None,
                        help="Summary JSON covering both sides. One file, because the "
                             "match is joint. Defaults beside the public output.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    inputs = {"public": args.public, "private": args.private}
    for name, path in inputs.items():
        if not path.exists():
            raise SystemExit(f"{name} input not found: {path}")

    grid = get_grid(args.grid)
    resolution = grid.DEFAULT_RESOLUTION if args.resolution is None else args.resolution
    try:
        resolution = grid.validate_resolution(resolution)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    kept_public, kept_private, summary = reciprocal_filter(
        read_canonical_csv(args.public),
        read_canonical_csv(args.private),
        grid=grid,
        resolution=resolution,
    )
    kept = {"public": kept_public, "private": kept_private}

    out_paths = {
        "public": args.out_public or _default_output(args.public),
        "private": args.out_private or _default_output(args.private),
    }
    if out_paths["public"] == out_paths["private"]:
        raise SystemExit(
            f"both outputs resolve to {out_paths['public']}; pass --out-public / "
            f"--out-private so one does not overwrite the other"
        )
    summary_path = args.summary or _default_summary(out_paths["public"])
    summary = {
        "inputs": {name: str(p) for name, p in inputs.items()},
        "outputs": {name: str(p) for name, p in out_paths.items()},
        **summary,
    }

    for name in _DATASETS:
        out_paths[name].parent.mkdir(parents=True, exist_ok=True)
        kept[name].to_csv(out_paths[name], index=False)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    logger.info(
        "reciprocal match on %s %s=%d (%.1f km cells)",
        grid.name, grid.resolution_arg, resolution,
        summary["grid"]["nominal_cell_km"],
    )
    for side in _SIDES:
        c = summary["cells"][side]
        logger.info(
            "  %-8s: %d common cells (public %d of %d, private %d of %d)",
            side + "s", c["common_cells"],
            c["common_cells"], c["public_cells"],
            c["common_cells"], c["private_cells"],
        )
        for name in _DATASETS:
            logger.info(
                "      %-7s %d / %d nodes kept (%.2f%%)",
                name, c[f"{name}_nodes_out"], c[f"{name}_nodes_in"],
                c[f"{name}_retention_pct"],
            )
    for name in _DATASETS:
        r = summary[name]
        logger.info(
            "  %-7s: %d / %d rows (%.2f%%) | %d VPs, %d targets | by VP count %s",
            name, r["rows_out"], r["rows_in"], r["rows_retention_pct"],
            r["vps_out"], r["targets_out"], r["targets_out_by_vp_count"],
        )
        logger.info("      wrote %s", out_paths[name])
    logger.info("  wrote %s", summary_path)


if __name__ == "__main__":
    main()
