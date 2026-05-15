"""Dataset selection helpers for scaled Vultr-7 CBG benchmarks."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.libs.core.benchmarking import BenchmarkRecorder  # noqa: E402
from scripts.libs.core.combinations import PipelineSpec  # noqa: E402
from scripts.libs.core.evaluate import (  # noqa: E402
    PreparedEvaluationData,
    _measure_setting_phase,
    fingerprint_dataframe,
    prepare_evaluation_inputs,
)
from scripts.libs.cbg_feasibility.rtt_model import haversine_distance  # noqa: E402

DEFAULT_INPUT_CSV = PROJECT_ROOT / "datasets" / "cbg_test" / "vultr_pings_us_only.csv"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "vultr7"
DEFAULT_DATASET_IDS = tuple([f"top{k}" for k in range(1, 11)] + ["all_us"])

REQUIRED_COLUMNS = (
    "src_ip",
    "dst_ip",
    "min_rtt",
    "probe_asn",
    "probe_latitude",
    "probe_longitude",
    "anchor_latitude",
    "anchor_longitude",
    "anchor_city",
)


@dataclass(frozen=True)
class DatasetSpec:
    """One materialized probe-scale dataset."""

    dataset_id: str
    label: str
    selected_asns: Optional[Tuple[int, ...]]
    n_rows: int
    n_probes: int
    n_anchors: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "label": self.label,
            "selected_asns": list(self.selected_asns)
            if self.selected_asns is not None
            else None,
            "n_rows": self.n_rows,
            "n_probes": self.n_probes,
            "n_anchors": self.n_anchors,
        }


def load_vultr7_us_csv(input_csv: Path = DEFAULT_INPUT_CSV) -> pd.DataFrame:
    """Load the Vultr US-only ping CSV and validate required columns."""
    df = pd.read_csv(input_csv)
    validate_columns(df)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if the input dataframe cannot drive CBG evaluation."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def rank_top_asns(df: pd.DataFrame, top_n: int = 10) -> List[int]:
    """Rank probe ASNs by unique probe count, then row count."""
    validate_columns(df)
    ranked = (
        df.dropna(subset=["probe_asn"])
        .groupby("probe_asn")
        .agg(probes=("src_ip", "nunique"), rows=("src_ip", "size"))
        .reset_index()
        .sort_values(["probes", "rows", "probe_asn"], ascending=[False, False, True])
        .head(top_n)
    )
    return [int(asn) for asn in ranked["probe_asn"]]


def dataset_ids(max_top_k: int = 10) -> Tuple[str, ...]:
    return tuple([f"top{k}" for k in range(1, max_top_k + 1)] + ["all_us"])


def selected_asns_for_dataset(
    df: pd.DataFrame,
    dataset_id: str,
    max_top_k: int = 10,
) -> Optional[Tuple[int, ...]]:
    """Return selected ASNs for a dataset id; None means all ASNs."""
    if dataset_id == "all_us":
        return None

    if not dataset_id.startswith("top"):
        raise ValueError(f"Unknown dataset_id: {dataset_id}")

    try:
        top_k = int(dataset_id.removeprefix("top"))
    except ValueError as exc:
        raise ValueError(f"Invalid top-k dataset_id: {dataset_id}") from exc

    if top_k < 1 or top_k > max_top_k:
        raise ValueError(f"dataset_id must be top1..top{max_top_k} or all_us")

    ranked = rank_top_asns(df, top_n=max_top_k)
    if top_k > len(ranked):
        raise ValueError(f"Requested top{top_k}, but only {len(ranked)} ASNs exist")
    return tuple(ranked[:top_k])


def select_dataset(
    df: pd.DataFrame,
    dataset_id: str,
    max_top_k: int = 10,
) -> Tuple[pd.DataFrame, DatasetSpec]:
    """Select a cumulative top-k or all-US dataset from the Vultr-7 CSV."""
    selected, selected_asns, label = _select_raw_dataset(
        df,
        dataset_id,
        max_top_k=max_top_k,
    )
    spec = _make_dataset_spec(selected, dataset_id, label, selected_asns)
    selected = prepare_cbg_dataframe(selected)
    return selected, spec


def prepare_cbg_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add CBG model-training columns expected by legacy fitters."""
    validate_columns(df)
    prepared = df.copy()
    prepared["distance_km"] = prepared.apply(
        lambda row: haversine_distance(
            row["probe_latitude"],
            row["probe_longitude"],
            row["anchor_latitude"],
            row["anchor_longitude"],
        ),
        axis=1,
    )
    return prepared


def build_dataset_specs(
    input_csv: Path = DEFAULT_INPUT_CSV,
    max_top_k: int = 10,
) -> List[DatasetSpec]:
    """Return all default scale dataset specs without writing outputs."""
    df = load_vultr7_us_csv(input_csv)
    specs = []
    for dataset_id in dataset_ids(max_top_k=max_top_k):
        selected, selected_asns, label = _select_raw_dataset(
            df,
            dataset_id,
            max_top_k=max_top_k,
        )
        spec = _make_dataset_spec(selected, dataset_id, label, selected_asns)
        specs.append(spec)
    return specs


def materialize_dataset(
    dataset_id: str,
    input_csv: Path,
    output_csv: Path,
    manifest_output: Optional[Path] = None,
    max_top_k: int = 10,
) -> DatasetSpec:
    """Write a selected benchmark dataset and JSON manifest."""
    df = load_vultr7_us_csv(input_csv)
    selected, spec = select_dataset(df, dataset_id, max_top_k=max_top_k)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_csv, index=False)

    if manifest_output is not None:
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest = spec.to_dict()
        manifest["input_csv"] = _display_path(input_csv)
        manifest["output_csv"] = _display_path(output_csv)
        manifest_output.write_text(json.dumps(manifest, indent=2) + "\n")

    return spec


class CSVDataLoader:
    """Per-setting data loader for benchmark CSV inputs."""

    def __init__(
        self,
        input_csv: Path,
        dataset_id: str,
        preselected: bool = False,
        max_top_k: int = 10,
    ) -> None:
        self.input_csv = input_csv
        self.dataset_id = dataset_id
        self.preselected = preselected
        self.max_top_k = max_top_k
        self.latest_spec: Optional[DatasetSpec] = None

    def __call__(
        self,
        spec: PipelineSpec,
        benchmark_recorder: Optional[BenchmarkRecorder],
        benchmark_ms: Optional[Dict[str, float]],
    ) -> PreparedEvaluationData:
        with _measure_setting_phase(
            spec,
            "load_data",
            benchmark_ms,
            benchmark_recorder,
            metadata=lambda: self._metadata(),
        ):
            df = load_vultr7_us_csv(self.input_csv)
            if self.preselected:
                df_asn = prepare_cbg_dataframe(df)
                self.latest_spec = describe_preselected_dataset(
                    df_asn,
                    self.dataset_id,
                )
            else:
                df_asn, self.latest_spec = select_dataset(
                    df,
                    self.dataset_id,
                    max_top_k=self.max_top_k,
                )

        with _measure_setting_phase(
            spec,
            "prepare_data",
            benchmark_ms,
            benchmark_recorder,
        ):
            anchor_coords, probe_targets = prepare_evaluation_inputs(df_asn)

        with _measure_setting_phase(
            spec,
            "data_fingerprint",
            benchmark_ms,
            benchmark_recorder,
            track_tracemalloc=False,
        ):
            data_fingerprint = fingerprint_dataframe(df_asn)

        return PreparedEvaluationData(
            df_asn=df_asn,
            anchor_coords=anchor_coords,
            probe_targets=probe_targets,
            data_fingerprint=data_fingerprint,
        )

    def _metadata(self) -> Dict[str, Any]:
        metadata = {
            "dataset_id": self.dataset_id,
            "input_csv": _display_path(self.input_csv),
            "preselected": self.preselected,
        }
        if self.latest_spec is not None:
            spec_data = self.latest_spec.to_dict()
            spec_data.pop("label", None)
            if spec_data["selected_asns"] is not None:
                spec_data["selected_asns"] = ",".join(
                    str(asn) for asn in spec_data["selected_asns"]
                )
            metadata.update(spec_data)
        return metadata

    def manifest(self) -> Dict[str, Any]:
        if self.latest_spec is None:
            return {
                "dataset_id": self.dataset_id,
                "input_csv": _display_path(self.input_csv),
                "preselected": self.preselected,
            }
        manifest = self.latest_spec.to_dict()
        manifest["input_csv"] = _display_path(self.input_csv)
        manifest["preselected"] = self.preselected
        return manifest


def describe_preselected_dataset(df: pd.DataFrame, dataset_id: str) -> DatasetSpec:
    """Describe a CSV that has already been filtered/materialized."""
    asns = (
        None
        if dataset_id == "all_us"
        else tuple(int(asn) for asn in sorted(df["probe_asn"].dropna().unique()))
    )
    return DatasetSpec(
        dataset_id=dataset_id,
        label=dataset_id,
        selected_asns=asns,
        n_rows=int(len(df)),
        n_probes=int(df["src_ip"].nunique()),
        n_anchors=int(df["dst_ip"].nunique()),
    )


def _select_raw_dataset(
    df: pd.DataFrame,
    dataset_id: str,
    max_top_k: int,
) -> Tuple[pd.DataFrame, Optional[Tuple[int, ...]], str]:
    selected_asns = selected_asns_for_dataset(df, dataset_id, max_top_k=max_top_k)
    if selected_asns is None:
        return df.copy(), None, "all_us"

    selected = df[df["probe_asn"].isin([float(asn) for asn in selected_asns])].copy()
    label = "+".join(f"AS{asn}" for asn in selected_asns)
    return selected, selected_asns, label


def _make_dataset_spec(
    selected: pd.DataFrame,
    dataset_id: str,
    label: str,
    selected_asns: Optional[Tuple[int, ...]],
) -> DatasetSpec:
    return DatasetSpec(
        dataset_id=dataset_id,
        label=label,
        selected_asns=selected_asns,
        n_rows=int(len(selected)),
        n_probes=int(selected["src_ip"].nunique()),
        n_anchors=int(selected["dst_ip"].nunique()),
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
