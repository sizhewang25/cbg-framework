"""CLI entry point: run all configured pipeline combinations and generate plots.

Usage:
    python -m scripts.benchmark.run_evaluation
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.libs.core.combinations import (
    COMBINATIONS,
    DIFF_PAIRS,
    SPECS_BY_ID,
)
from scripts.libs.core.evaluate import (
    evaluate_all,
    print_statistics,
)
from scripts.libs.core.benchmarking import BenchmarkRecorder
from scripts.libs.core.summary import save_json_summary
from scripts.libs.plotting.plot_error_cdf import plot_error_cdf
from scripts.libs.plotting.plot_error_diff_cdf import plot_error_diff_cdf
from scripts.libs.plotting.plot_benchmark_summary import plot_benchmark_summary
from scripts.libs.plotting.plot_rtt_error_scatter import plot_rtt_error_scatter

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "single_shot"
LOG_DIR = OUTPUT_DIR / "logs"


def _setup_logging(output_dir: Path) -> None:
    """Configure root logger with file + console handlers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "evaluation.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # File handler — full log
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)

    logger.info("Logging to %s", log_path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _setup_logging(LOG_DIR)
    total_start = time.perf_counter()

    # 1. Evaluate all configured combinations end-to-end per setting
    logger.info("=" * 60)
    logger.info("EVALUATING ALL COMBINATIONS")
    logger.info("=" * 60)
    benchmark_recorder = BenchmarkRecorder()
    evaluation_run = evaluate_all(
        COMBINATIONS,
        benchmark_recorder=benchmark_recorder,
    )
    all_results = evaluation_run.all_results

    benchmark_raw_path = OUTPUT_DIR / "benchmark_phase_raw.csv"
    benchmark_summary_path = OUTPUT_DIR / "benchmark_phase_summary.json"
    benchmark_recorder.write_raw_csv(benchmark_raw_path)
    benchmark_recorder.write_summary_json(benchmark_summary_path)
    logger.info("Saved: %s", benchmark_raw_path)
    logger.info("Saved: %s", benchmark_summary_path)

    # 2. Statistics table
    print_statistics(all_results, COMBINATIONS)

    # 3. Error CDF
    logger.info("=" * 60)
    logger.info("GENERATING ERROR CDF")
    logger.info("=" * 60)
    fig = plot_error_cdf(all_results, COMBINATIONS, OUTPUT_DIR / "error_cdf_all.png")
    plt.close(fig)

    # 4. Error-Diff CDF
    logger.info("=" * 60)
    logger.info("GENERATING ERROR-DIFF CDF")
    logger.info("=" * 60)
    fig = plot_error_diff_cdf(
        all_results, SPECS_BY_ID, DIFF_PAIRS,
        OUTPUT_DIR / "error_diff_cdf.png",
    )
    plt.close(fig)

    # 5. RTT-Error Scatter
    logger.info("=" * 60)
    logger.info("GENERATING RTT-ERROR SCATTER")
    logger.info("=" * 60)
    fig = plot_rtt_error_scatter(
        all_results, COMBINATIONS, OUTPUT_DIR / "rtt_error_scatter.png",
    )
    plt.close(fig)

    # 6. Percentile Maps
    logger.info("=" * 60)
    logger.info("GENERATING PERCENTILE MAPS")
    logger.info("=" * 60)
    try:
        from scripts.libs.plotting.plot_percentile_maps import (
            plot_percentile_maps,
        )
        plot_percentile_maps(
            all_results,
            SPECS_BY_ID,
            evaluation_run.artifacts_by_combo,
            OUTPUT_DIR / "maps",
        )
    except ImportError as e:
        logger.warning("Skipping percentile maps (missing dependency): %s", e)
    except Exception as e:
        logger.error("Percentile maps failed: %s", e)

    # 7. JSON summary
    save_json_summary(
        all_results,
        OUTPUT_DIR / "evaluation_summary.json",
        evaluation_run.artifacts_by_combo,
        benchmark_raw_path=benchmark_raw_path,
        benchmark_summary_path=benchmark_summary_path,
    )

    # 8. Benchmark summary charts from the generated JSON summaries
    logger.info("=" * 60)
    logger.info("GENERATING BENCHMARK SUMMARY CHARTS")
    logger.info("=" * 60)
    plot_benchmark_summary(
        OUTPUT_DIR / "evaluation_summary.json",
        benchmark_summary_path,
        benchmark_raw_path,
        OUTPUT_DIR,
    )

    elapsed = time.perf_counter() - total_start
    logger.info("Total runtime: %.1fs", elapsed)
    logger.info("Done.")


if __name__ == "__main__":
    main()
