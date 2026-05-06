"""Plot error-distance CDFs from benchmark CSV files.

The input is one or more checkpoint probe-result CSVs, benchmark raw CSVs,
or directories containing CSVs. If no input is provided, the script reads
``data/*.csv`` next to this file. Inputs are merged and plotted as one CDF.
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
LABEL_ORDER = {
    "l1": 0,
    "l2": 1,
    "s1": 2,
    "s2": 3,
    "b1": 4,
    "b2": 5,
    "b3": 6,
    "b4": 7,
    "original cbg": 0,
    "million scale": 1,
    "octant": 2,
}
FAMILY_COLORS = {
    "L": "#000000",
    "S": "#0072B2",
    "B": "#D55E00",
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


def load_errors_by_combo(input_csvs: Sequence[Path]) -> dict[str, list[float]]:
    """Load and merge error distances from benchmark CSV files."""
    errors_by_combo: dict[str, list[float]] = defaultdict(list)
    missing_error_files = []

    for input_csv in expand_csvs(input_csvs):
        with input_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            if "error_km" not in fieldnames:
                missing_error_files.append(input_csv)
                continue

            for row in reader:
                # benchmark_phase_raw.csv has many phase rows; checkpoint
                # probe-results CSVs do not have a phase column.
                phase = row.get("phase")
                if phase and phase != "total_geolocate":
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

    return dict(errors_by_combo)


def plot_error_cdf_from_csvs(
    input_csvs: Sequence[Path],
    output_path: Path,
    *,
    max_x_km: float = 3000.0,
    thresholds: Iterable[float] = (100.0, 500.0, 1000.0),
    title: str = "Error Distance CDF",
) -> plt.Figure:
    """Plot merged error CDFs from one or more CSV inputs."""
    errors_by_combo = load_errors_by_combo(input_csvs)
    if not errors_by_combo:
        raise ValueError("No plottable error_km values found in input CSV files.")

    fig, ax = plt.subplots(figsize=(9, 6))

    for combo_id in sorted(errors_by_combo, key=_label_sort_key):
        errors = np.array(errors_by_combo[combo_id], dtype=float)
        if len(errors) == 0:
            continue
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(
            sorted_errors,
            cdf,
            linewidth=2,
            alpha=0.85,
            color=_line_color(combo_id),
            linestyle=_line_style(combo_id),
            label=combo_id,
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

    _add_stats_box(ax, errors_by_combo)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", output_path)
    return fig


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


def _label_sort_key(label: str) -> tuple[int, str]:
    return LABEL_ORDER.get(label.lower(), len(LABEL_ORDER)), label.lower()


def _line_color(label: str) -> str | None:
    family = label[:1].upper()
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
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    groups = discover_plot_groups(args.input_csvs)
    if args.output is not None and len(groups) != 1:
        raise ValueError("--output can only be used when there is exactly one data group.")

    for group in groups:
        output_path = args.output or args.output_root / group.name / args.output_name
        title = args.title
        if title == "Error Distance CDF":
            title = f"{group.name.replace('_', ' ').title()} Error Distance CDF"
        fig = plot_error_cdf_from_csvs(
            group.csvs,
            output_path,
            max_x_km=args.max_x_km,
            thresholds=args.thresholds or (100.0, 500.0, 1000.0),
            title=title,
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
