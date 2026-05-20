"""Standalone phase-local memory benchmark plot."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.libs.plotting.plot_benchmark_summary import (
    load_json,
    plot_phase_memory,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot stacked phase-local memory and process max RSS from a "
            "CBG benchmark summary JSON file."
        ),
    )
    parser.add_argument(
        "--benchmark-summary",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "outputs"
            / "benchmark_phase_summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "benchmark_phase_memory.png",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fig = plot_phase_memory(load_json(args.benchmark_summary), args.output)
    plt.close(fig)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
