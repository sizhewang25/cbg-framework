"""Runner for scaled benchmark evaluations."""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.dataset import (  # noqa: E402
    CSVDataLoader,
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_ROOT,
)
from scripts.libs.core.benchmarking import BenchmarkRecorder  # noqa: E402
from scripts.libs.core.combinations import (  # noqa: E402
    DIFF_PAIRS,
    SPECS_BY_ID,
    PipelineSpec,
)
from scripts.libs.core.evaluate import (  # noqa: E402
    DistanceModelCache,
    ProbeResult,
    SettingEvaluation,
    print_statistics,
    run_setting,
)
from scripts.libs.core.summary import save_json_summary  # noqa: E402
from scripts.libs.core.reporting import count_result_outcomes  # noqa: E402
from scripts.libs.plotting.plot_benchmark_summary import (  # noqa: E402
    plot_benchmark_summary,
)
from scripts.libs.plotting.plot_error_cdf import plot_error_cdf  # noqa: E402
from scripts.libs.plotting.plot_error_diff_cdf import (  # noqa: E402
    plot_error_diff_cdf,
)
from scripts.libs.plotting.plot_rtt_error_scatter import (  # noqa: E402
    plot_rtt_error_scatter,
)

logger = logging.getLogger(__name__)

DEFAULT_COMBO_IDS = (
    "Vanilla CBG", "V2", "V3",
    "Million-scale CBG", "M2", "M3",
    "Octant", "O2", "O3", "O4",
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_combo_ids(combo_ids: str | Sequence[str]) -> List[str]:
    """Parse comma-separated or sequence combo ids."""
    if isinstance(combo_ids, str):
        parsed = [part.strip() for part in combo_ids.split(",") if part.strip()]
    else:
        parsed = [str(part).strip() for part in combo_ids if str(part).strip()]
    if not parsed:
        raise ValueError("At least one combo id is required")
    unknown = [combo_id for combo_id in parsed if combo_id not in SPECS_BY_ID]
    if unknown:
        raise ValueError(f"Unknown combo ids: {', '.join(unknown)}")
    return parsed


def select_combinations(combo_ids: str | Sequence[str]) -> List[PipelineSpec]:
    return [SPECS_BY_ID[combo_id] for combo_id in parse_combo_ids(combo_ids)]


def make_run_id(now: datetime | None = None) -> str:
    """Return a UTC timestamp run id with microsecond precision."""
    timestamp = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.strftime("%Y%m%dT%H%M%S%fZ")


def default_run_output_dir(
    dataset_id: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
) -> Path:
    """Return the default timestamped output directory for one dataset run."""
    resolved_run_id = _validate_run_id(run_id or make_run_id())
    return output_root / "runs" / resolved_run_id / dataset_id


def run_benchmark_evaluation(
    dataset_id: str,
    input_csv: Path = DEFAULT_INPUT_CSV,
    combo_ids: str | Sequence[str] = DEFAULT_COMBO_IDS,
    output_dir: Path | None = None,
    preselected: bool = False,
    generate_maps: bool = False,
    run_id: str | None = None,
) -> Path:
    """Run selected CBG combinations for one scaled dataset."""
    resolved_run_id = _validate_run_id(run_id or make_run_id())
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else default_run_output_dir(
            dataset_id,
            run_id=resolved_run_id,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = select_combinations(combo_ids)
    active_pairs = _active_diff_pairs(specs)
    data_loader = CSVDataLoader(
        input_csv=input_csv,
        dataset_id=dataset_id,
        preselected=preselected,
    )

    start = time.perf_counter()
    benchmark_recorder = BenchmarkRecorder()
    model_cache = DistanceModelCache()
    all_results: Dict[str, List[ProbeResult]] = {}
    artifacts_by_combo: Dict[str, SettingEvaluation] = {}
    completed_specs: List[PipelineSpec] = []
    benchmark_raw_path = output_dir / "benchmark_phase_raw.csv"
    benchmark_summary_path = output_dir / "benchmark_phase_summary.json"
    summary_path = output_dir / "evaluation_summary.json"
    run_started_at_utc = _utc_now()

    for spec in specs:
        print(f"[{dataset_id}] running {spec.combo_id}: {spec.label}", flush=True)
        try:
            artifact = run_setting(
                spec,
                model_cache=model_cache,
                benchmark_recorder=benchmark_recorder,
                data_loader=data_loader,
            )
        except Exception as exc:
            _write_checkpoint_progress(
                output_dir,
                dataset_id=dataset_id,
                requested_specs=specs,
                completed_specs=completed_specs,
                failed_combo_id=spec.combo_id,
                error=str(exc),
                run_id=resolved_run_id,
                run_started_at_utc=run_started_at_utc,
            )
            benchmark_recorder.write_raw_csv(benchmark_raw_path)
            benchmark_recorder.write_summary_json(benchmark_summary_path)
            raise

        all_results[spec.combo_id] = artifact.results
        artifacts_by_combo[spec.combo_id] = artifact
        completed_specs.append(spec)

        _write_combo_checkpoint(
            output_dir,
            artifact,
            data_loader.manifest(),
            run_id=resolved_run_id,
            run_started_at_utc=run_started_at_utc,
        )
        _write_incremental_outputs(
            output_dir=output_dir,
            input_csv=input_csv,
            data_loader=data_loader,
            benchmark_recorder=benchmark_recorder,
            all_results=all_results,
            artifacts_by_combo=artifacts_by_combo,
            completed_specs=completed_specs,
            requested_specs=specs,
            is_complete=len(completed_specs) == len(specs),
            run_id=resolved_run_id,
            run_started_at_utc=run_started_at_utc,
        )

        elapsed_s = artifact.benchmark_ms.get("setting_total_ms", 0.0) / 1000.0
        print(
            f"[{dataset_id}] checkpointed {spec.combo_id}: "
            f"{len(artifact.results)} probes, {elapsed_s:.1f}s",
            flush=True,
        )

    print_statistics(all_results, specs)

    fig = plot_error_cdf(all_results, specs, output_dir / "error_cdf_all.png")
    plt.close(fig)

    if active_pairs:
        fig = plot_error_diff_cdf(
            all_results,
            {spec.combo_id: spec for spec in specs},
            active_pairs,
            output_dir / "error_diff_cdf.png",
        )
        plt.close(fig)

    fig = plot_rtt_error_scatter(
        all_results,
        specs,
        output_dir / "rtt_error_scatter.png",
    )
    plt.close(fig)

    if generate_maps:
        from scripts.libs.plotting.plot_percentile_maps import (
            plot_percentile_maps,
        )

        plot_percentile_maps(
            all_results,
            {spec.combo_id: spec for spec in specs},
            artifacts_by_combo,
            output_dir / "maps",
        )

    plot_benchmark_summary(
        summary_path,
        benchmark_summary_path,
        benchmark_raw_path,
        output_dir,
    )

    elapsed = time.perf_counter() - start
    logger.info("Finished %s in %.1fs", dataset_id, elapsed)
    return summary_path


def _write_incremental_outputs(
    output_dir: Path,
    input_csv: Path,
    data_loader: CSVDataLoader,
    benchmark_recorder: BenchmarkRecorder,
    all_results: Dict[str, List[ProbeResult]],
    artifacts_by_combo: Dict[str, SettingEvaluation],
    completed_specs: Sequence[PipelineSpec],
    requested_specs: Sequence[PipelineSpec],
    is_complete: bool,
    run_id: str | None = None,
    run_started_at_utc: str | None = None,
) -> None:
    """Persist accumulated outputs after each completed combo."""
    benchmark_raw_path = output_dir / "benchmark_phase_raw.csv"
    benchmark_summary_path = output_dir / "benchmark_phase_summary.json"
    summary_path = output_dir / "evaluation_summary.json"

    benchmark_recorder.write_raw_csv(benchmark_raw_path)
    benchmark_recorder.write_summary_json(benchmark_summary_path)
    save_json_summary(
        all_results,
        summary_path,
        artifacts_by_combo,
        benchmark_raw_path=benchmark_raw_path,
        benchmark_summary_path=benchmark_summary_path,
        combinations=list(completed_specs),
        diff_pairs=_active_diff_pairs(completed_specs),
        dataset=input_csv.name,
        asn=None,
        dataset_metadata=_checkpoint_metadata(
            data_loader,
            output_dir=output_dir,
            requested_specs=requested_specs,
            completed_specs=completed_specs,
            is_complete=is_complete,
            run_id=run_id,
            run_started_at_utc=run_started_at_utc,
        ),
    )
    _write_checkpoint_progress(
        output_dir,
        dataset_id=data_loader.dataset_id,
        requested_specs=requested_specs,
        completed_specs=completed_specs,
        failed_combo_id=None,
        error=None,
        run_id=run_id,
        run_started_at_utc=run_started_at_utc,
    )


def _write_combo_checkpoint(
    output_dir: Path,
    artifact: SettingEvaluation,
    dataset_metadata: Dict[str, Any],
    run_id: str | None = None,
    run_started_at_utc: str | None = None,
) -> None:
    """Write per-combo probe results and metadata checkpoint files."""
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    combo_id = artifact.spec.combo_id

    results_path = checkpoint_dir / f"{combo_id}_probe_results.csv"
    _write_probe_results_csv(results_path, artifact.results)

    counts = count_result_outcomes(artifact.results)
    metadata = {
        "combo_id": combo_id,
        "label": artifact.spec.label,
        "run_id": run_id,
        "run_output_dir": _display_path(output_dir),
        "run_started_at_utc": run_started_at_utc,
        "dataset_metadata": dataset_metadata,
        "data_fingerprint": artifact.data_fingerprint,
        "benchmark_ms": {
            key: round(float(value), 3)
            for key, value in artifact.benchmark_ms.items()
        },
        "n_probes": counts.total_probes,
        "estimated_count": counts.estimated_count,
        "intersection_count": counts.intersection_count,
        "fallback_count": counts.fallback_count,
        "no_estimate_count": counts.no_estimate_count,
        "probe_results_csv": _display_path(results_path),
        "written_at_utc": _utc_now(),
    }
    _atomic_write_json(checkpoint_dir / f"{combo_id}_checkpoint.json", metadata)


def _write_probe_results_csv(output_path: Path, results: Sequence[ProbeResult]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    fieldnames = list(ProbeResult.__dataclass_fields__.keys())
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    tmp_path.replace(output_path)


def _write_checkpoint_progress(
    output_dir: Path,
    dataset_id: str,
    requested_specs: Sequence[PipelineSpec],
    completed_specs: Sequence[PipelineSpec],
    failed_combo_id: str | None,
    error: str | None,
    run_id: str | None = None,
    run_started_at_utc: str | None = None,
) -> None:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "run_output_dir": _display_path(output_dir),
        "run_started_at_utc": run_started_at_utc,
        "requested_combo_ids": [spec.combo_id for spec in requested_specs],
        "completed_combo_ids": [spec.combo_id for spec in completed_specs],
        "failed_combo_id": failed_combo_id,
        "error": error,
        "is_complete": len(completed_specs) == len(requested_specs)
        and failed_combo_id is None,
        "evaluation_summary_json": _display_path(output_dir / "evaluation_summary.json"),
        "benchmark_raw_csv": _display_path(output_dir / "benchmark_phase_raw.csv"),
        "benchmark_summary_json": _display_path(output_dir / "benchmark_phase_summary.json"),
        "updated_at_utc": _utc_now(),
    }
    _atomic_write_json(checkpoint_dir / "progress.json", progress)


def _checkpoint_metadata(
    data_loader: CSVDataLoader,
    output_dir: Path,
    requested_specs: Sequence[PipelineSpec],
    completed_specs: Sequence[PipelineSpec],
    is_complete: bool,
    run_id: str | None = None,
    run_started_at_utc: str | None = None,
) -> Dict[str, Any]:
    metadata = data_loader.manifest()
    metadata["run_id"] = run_id
    metadata["run_output_dir"] = _display_path(output_dir)
    metadata["run_started_at_utc"] = run_started_at_utc
    metadata["checkpoint"] = {
        "requested_combo_ids": [spec.combo_id for spec in requested_specs],
        "completed_combo_ids": [spec.combo_id for spec in completed_specs],
        "is_complete": is_complete,
        "updated_at_utc": _utc_now(),
    }
    return metadata


def _atomic_write_json(output_path: Path, data: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2) + "\n")
    tmp_path.replace(output_path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_summaries(results_root: Path) -> List[Dict[str, Any]]:
    """Collect per-dataset evaluation summaries into compact rows."""
    rows = []
    for summary_path in results_root.glob("**/evaluation_summary.json"):
        with open(summary_path) as f:
            summary = json.load(f)
        metadata = summary.get("dataset_metadata", {})
        combinations = summary.get("combinations", {})
        rows.append(
            {
                "dataset_id": metadata.get("dataset_id", summary_path.parent.name),
                "run_id": metadata.get("run_id"),
                "run_started_at_utc": metadata.get("run_started_at_utc"),
                "run_output_dir": metadata.get(
                    "run_output_dir",
                    _display_path(summary_path.parent),
                ),
                "n_rows": metadata.get("n_rows"),
                "n_probes": metadata.get("n_probes"),
                "n_anchors": metadata.get("n_anchors"),
                "n_combinations": summary.get("n_combinations"),
                "combo_ids": ",".join(combinations.keys()),
                "summary_path": _display_path(summary_path),
            }
        )
    rows.sort(
        key=lambda row: (
            _dataset_sort_key(str(row["dataset_id"])),
            str(row.get("run_id") or ""),
        )
    )
    return rows


def write_summary_index(results_root: Path, output_json: Path, output_csv: Path) -> None:
    """Write JSON and CSV indexes over all per-dataset summaries."""
    rows = collect_summaries(results_root)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2) + "\n")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_id",
        "run_id",
        "run_started_at_utc",
        "run_output_dir",
        "n_rows",
        "n_probes",
        "n_anchors",
        "n_combinations",
        "combo_ids",
        "summary_path",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _active_diff_pairs(specs: Sequence[PipelineSpec]) -> List[tuple[str, str]]:
    active_ids = {spec.combo_id for spec in specs}
    return [
        (id_a, id_b)
        for id_a, id_b in DIFF_PAIRS
        if id_a in active_ids and id_b in active_ids
    ]


def _dataset_sort_key(dataset_id: str) -> tuple[int, int | str]:
    if dataset_id.startswith("top"):
        try:
            return (0, int(dataset_id.removeprefix("top")))
        except ValueError:
            return (2, dataset_id)
    if dataset_id == "all_us":
        return (1, 0)
    return (2, dataset_id)


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.match(run_id):
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, numbers, dots, underscores, or hyphens"
        )
    return run_id


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
