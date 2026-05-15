"""Plot error-distance CDFs from benchmark CSV files.

The input is one or more checkpoint probe-result CSVs, benchmark raw CSVs,
or directories containing CSVs. If no input is provided, the script reads
``data/*.csv`` next to this file. Inputs are merged and plotted as one CDF.
Use ``--intersected-only`` to include only probe rows where CBG produced an
intersection region instead of falling back.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
DEFAULT_OUTPUT_NAME = "error_cdf_all_variants.png"
DEFAULT_INTERSECTED_OUTPUT_NAME = "error_cdf_intersected_only.png"
FAMILY_ORDER = {
    "V": 0,
    "M": 1,
    "O": 2,
    "L": 3,
    "S": 4,
    "B": 5,
}
FAMILY_COLORS = {
    "V": "#000000",
    "M": "#0072B2",
    "O": "#D55E00",
    "L": "#000000",
    "S": "#0072B2",
    "B": "#D55E00",
}
DISPLAY_LABELS = {
    "original cbg": "Vanilla CBG",
    "o2": "V2",
    "o3": "V3",
    "b1": "Octant",
    "b2": "O2",
    "b3": "O3",
    "b4": "O4",
    "million scale": "Million-scale CBG",
    "million-scale cbg": "Million-scale CBG",
    "octant": "Octant",
}
ID_LINESTYLES = {
    "1": "-",
    "2": "--",
    "3": "-.",
    "4": ":",
}


@dataclass(frozen=True)
class PlotGroup:
    """One data folder worth of CSVs and its output subfolder name."""

    name: str
    csvs: list[Path]


def load_errors_by_combo(
    input_csvs: Sequence[Path],
    *,
    intersected_only: bool = False,
) -> dict[str, list[float]]:
    """Load and merge error distances from benchmark CSV files."""
    errors_by_combo: dict[str, list[float]] = defaultdict(list)
    missing_error_files = []
    missing_intersection_files = []

    for input_csv in expand_csvs(input_csvs):
        with input_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            if "error_km" not in fieldnames:
                missing_error_files.append(input_csv)
                continue
            if intersected_only and "did_intersect" not in fieldnames:
                missing_intersection_files.append(input_csv)
                continue

            for row in reader:
                # benchmark_phase_raw.csv has many phase rows; checkpoint
                # probe-results CSVs do not have a phase column.
                phase = row.get("phase")
                if phase and phase != "total_geolocate":
                    continue

                if intersected_only and not _parse_bool(row.get("did_intersect")):
                    continue

                error = _parse_float(row.get("error_km"))
                if error is None:
                    continue

                combo_id = row.get("combo_id") or _combo_id_from_path(input_csv)
                errors_by_combo[combo_id].append(error)

    if missing_error_files:
        paths = "\n".join(f"  - {path}" for path in missing_error_files)
        raise ValueError(
            "The following raw CSV files do not contain an error_km column:\n"
            f"{paths}\n"
            "Pass checkpoint probe-results CSVs, or rerun the benchmark with "
            "a raw schema that contains error_km."
        )
    if missing_intersection_files:
        paths = "\n".join(f"  - {path}" for path in missing_intersection_files)
        raise ValueError(
            "The following CSV files do not contain a did_intersect column:\n"
            f"{paths}\n"
            "Pass checkpoint probe-results CSVs, or rerun the benchmark with "
            "a raw schema that contains did_intersect."
        )

    return dict(errors_by_combo)


def plot_error_cdf_from_csvs(
    input_csvs: Sequence[Path],
    output_path: Path,
    *,
    max_x_km: float = 3000.0,
    thresholds: Iterable[float] = (100.0, 500.0, 1000.0),
    title: str = "Error Distance CDF",
    intersected_only: bool = False,
) -> plt.Figure:
    """Plot merged error CDFs from one or more CSV inputs."""
    errors_by_combo = load_errors_by_combo(
        input_csvs,
        intersected_only=intersected_only,
    )
    if not errors_by_combo:
        suffix = " for intersected rows" if intersected_only else ""
        raise ValueError(f"No plottable error_km values found{suffix}.")

    errors_by_label = _display_errors_by_label(errors_by_combo)
    fig, ax = plt.subplots(figsize=(9, 6))

    for label in sorted(errors_by_label, key=_label_sort_key):
        errors = np.array(errors_by_label[label], dtype=float)
        if len(errors) == 0:
            continue
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(
            sorted_errors,
            cdf,
            linewidth=2,
            alpha=0.85,
            color=_line_color(label),
            linestyle=_line_style(label),
            label=label,
        )

    threshold_colors = {100.0: "green", 500.0: "orange", 1000.0: "red"}
    for threshold in thresholds:
        ax.axvline(
            x=float(threshold),
            color=threshold_colors.get(float(threshold), "gray"),
            linestyle=":",
            alpha=0.45,
            linewidth=1.5,
        )

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.35, linewidth=1)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Error Distance (km)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_xlim(0, max_x_km)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    _add_stats_box(ax, errors_by_label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", output_path)
    return fig


def _display_errors_by_label(
    errors_by_combo: dict[str, list[float]],
) -> dict[str, list[float]]:
    errors_by_label: dict[str, list[float]] = defaultdict(list)
    for combo_id, errors in errors_by_combo.items():
        errors_by_label[_display_label(combo_id)].extend(errors)
    return dict(errors_by_label)


def _add_stats_box(ax, errors_by_combo: dict[str, list[float]]) -> None:
    lines = ["combo                n     p5    p25    p50    p75    p90"]
    for combo_id in sorted(errors_by_combo, key=_label_sort_key):
        errors = np.array(errors_by_combo[combo_id], dtype=float)
        if len(errors) == 0:
            continue
        label = combo_id[:18]
        lines.append(
            f"{label:<18} {len(errors):4d} "
            f"{np.percentile(errors, 5):6.0f} "
            f"{np.percentile(errors, 25):6.0f} "
            f"{np.median(errors):6.0f} "
            f"{np.percentile(errors, 75):6.0f} "
            f"{np.percentile(errors, 90):6.0f}"
        )

    ax.text(
        0.98,
        0.02,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        family="monospace",
    )


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def expand_csvs(inputs: Sequence[Path]) -> list[Path]:
    """Expand files and directories into sorted CSV file paths."""
    expanded: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            expanded.extend(sorted(input_path.glob("*.csv")))
        else:
            expanded.append(input_path)

    missing = [path for path in expanded if not path.exists()]
    if missing:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Input CSV path(s) do not exist:\n{paths}")

    csvs = [path for path in expanded if path.suffix.lower() == ".csv"]
    if not csvs:
        raise ValueError("No CSV files found in the provided inputs.")
    return csvs


def discover_plot_groups(inputs: Sequence[Path]) -> list[PlotGroup]:
    """Return plot groups from files, directories, or the default data root.

    Directory inputs with CSVs directly inside are one group. Directory inputs
    with CSV-containing subdirectories produce one group per immediate child,
    allowing ``data/literature`` to map to ``outputs/literature``.
    """
    groups: list[PlotGroup] = []
    for input_path in inputs:
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        if input_path.is_file():
            groups.append(PlotGroup(_group_name_for_file(input_path), [input_path]))
            continue

        direct_csvs = sorted(input_path.glob("*.csv"))
        if direct_csvs:
            groups.append(PlotGroup(input_path.name, direct_csvs))

        for child in sorted(path for path in input_path.iterdir() if path.is_dir()):
            child_csvs = sorted(child.glob("*.csv"))
            if child_csvs:
                groups.append(PlotGroup(child.name, child_csvs))

    if not groups:
        raise ValueError("No CSV files found in the provided data folder(s).")
    return groups


def _group_name_for_file(path: Path) -> str:
    parent = path.parent
    if parent != DEFAULT_DATA_DIR and parent.parent == DEFAULT_DATA_DIR:
        return parent.name
    return path.stem


def _combo_id_from_path(path: Path) -> str:
    stem = path.stem
    suffix = "_probe_results"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    label = stem.replace("_", " ")
    all_us_suffix = " all us"
    if label.endswith(all_us_suffix):
        label = label[: -len(all_us_suffix)]
    return label


def _display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label.casefold(), label)


def _label_sort_key(label: str) -> tuple[int, str]:
    family = _label_family(label)
    family_order = FAMILY_ORDER.get(family, len(FAMILY_ORDER))
    return family_order, _variant_order(label), label.casefold()


def _label_family(label: str) -> str:
    normalized = label.casefold()
    if normalized == "vanilla cbg" or label[:1].upper() == "V":
        return "V"
    if normalized in {"million-scale cbg", "million scale"} or label[:1].upper() == "M":
        return "M"
    if normalized == "octant" or label[:1].upper() == "O":
        return "O"
    return label[:1].upper()


def _variant_order(label: str) -> int:
    normalized = label.casefold()
    if normalized in {"vanilla cbg", "million-scale cbg", "million scale", "octant"}:
        return 1

    suffix = label[1:]
    if suffix.isdigit():
        return int(suffix)
    return 100


def _line_color(label: str) -> str | None:
    family = _label_family(label)
    return FAMILY_COLORS.get(family)


def _line_style(label: str) -> str:
    suffix = label[1:]
    return ID_LINESTYLES.get(suffix, "-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot error-distance CDFs from benchmark checkpoint CSV files.",
    )
    parser.add_argument(
        "input_csvs",
        nargs="*",
        type=Path,
        default=[DEFAULT_DATA_DIR],
        help=(
            "Checkpoint probe-results CSVs, benchmark raw CSVs, or directories "
            f"containing CSVs. Defaults to {DEFAULT_DATA_DIR}."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Exact output PNG path. Only valid for a single data group. "
            "By default, writes to --output-root/<data-folder>/--output-name."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root directory. Defaults to {DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output filename inside each output subfolder. Defaults to {DEFAULT_OUTPUT_NAME}.",
    )
    parser.add_argument(
        "--max-x-km",
        type=float,
        default=3000.0,
        help="Maximum x-axis value in km.",
    )
    parser.add_argument(
        "--threshold",
        dest="thresholds",
        type=float,
        action="append",
        default=None,
        help="Vertical threshold line in km. May be repeated.",
    )
    parser.add_argument(
        "--title",
        default="Error Distance CDF",
        help="Plot title.",
    )
    parser.add_argument(
        "--intersected-only",
        action="store_true",
        help="Only plot rows where did_intersect is true.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    groups = discover_plot_groups(args.input_csvs)
    if args.output is not None and len(groups) != 1:
        raise ValueError("--output can only be used when there is exactly one data group.")

    for group in groups:
        output_name = args.output_name
        if args.intersected_only and args.output_name == DEFAULT_OUTPUT_NAME:
            output_name = DEFAULT_INTERSECTED_OUTPUT_NAME
        output_path = args.output or args.output_root / group.name / output_name
        title = args.title
        if title == "Error Distance CDF":
            group_title = group.name.replace("_", " ").title()
            if args.intersected_only:
                title = f"{group_title} Intersected Error Distance CDF"
            else:
                title = f"{group_title} Error Distance CDF"
        fig = plot_error_cdf_from_csvs(
            group.csvs,
            output_path,
            max_x_km=args.max_x_km,
            thresholds=args.thresholds or (100.0, 500.0, 1000.0),
            title=title,
            intersected_only=args.intersected_only,
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
