"""Per-combo runner — one (LTD, MTL, CTR) triple over one (source, slice).

Inputs:
  - materialized parquets at inputs/<source>/<slice>/
  - combo identifier + the three stage names (+ optional kwargs)

Outputs (written to <run_dir>):
  - run.json           : combo metadata + fit stats + run-level memory
  - targets.parquet    : one row per eval target with full forensics
  - fit_checkpoint.pkl : pickled LTD (or .stateless marker for stateless LTDs)

The runner is what `cli.py run-combo` invokes. Snakemake fans out (source,
slice, combo) triples across this entry point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry.base import BaseGeometry
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon

from scripts.benchmark.v2 import schema as bench_schema
from scripts.benchmark.v2.checkpoint import save_ltd_checkpoint
from scripts.benchmark.v2.inputs import (
    load_eval_targets_parquet,
    load_fit_samples_parquet,
)
from scripts.benchmark.v2.instrument import (
    TimingMemoryInstrument,
    measure_block,
    peak_rss_bytes,
)
from scripts.framework.v2 import CBGModel
from scripts.framework.v2.mtl.base import MTLResult
from scripts.libs.cbg.rtt_model import haversine_distance


@dataclass(frozen=True)
class ComboSpec:
    """A single (LTD, MTL, CTR) triple to evaluate.

    `base_seed`, when non-None, makes stochastic stages deterministic:
    a per-target seed is derived from (base_seed, target_index) via
    numpy.random.SeedSequence and applied to any stage exposing an `rng`
    attribute (today, only MonteCarloMedoidCTR). Recorded in the row's
    `seed` column so a single (combo, target) can be replayed exactly.
    """
    combo_id: str
    ltd: str
    mtl: str
    ctr: str
    ltd_kwargs: dict[str, Any]
    mtl_kwargs: dict[str, Any]
    ctr_kwargs: dict[str, Any]
    base_seed: Optional[int] = None


def run_one_combo(
    spec: ComboSpec,
    *,
    inputs_dir: Path,
    out_dir: Path,
    run_id: str,
    source_name: str,
    slice_name: str,
    setup_name: str = "probes_to_anchors",
    enable_fallback: bool = True,
) -> Path:
    """Fit + geolocate every eval target for one combo. Write run outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rss_start = peak_rss_bytes()

    # --- 1. Load inputs ------------------------------------------------------
    fit_samples = load_fit_samples_parquet(inputs_dir / "fit_samples.parquet")
    eval_targets = load_eval_targets_parquet(inputs_dir / "eval_observations.parquet")

    # --- 2. Construct + fit model -------------------------------------------
    model = CBGModel.from_config(
        ltd=spec.ltd, mtl=spec.mtl, ctr=spec.ctr,
        ltd_kwargs=spec.ltd_kwargs or None,
        mtl_kwargs=spec.mtl_kwargs or None,
        ctr_kwargs=spec.ctr_kwargs or None,
        enable_fallback=enable_fallback,
    )
    with measure_block("fit") as fit_meas:
        fit_result = model.fit(fit_samples)
    save_ltd_checkpoint(model.ltd, fit_result, combo_dir=out_dir)

    # --- 3. Loop targets, geolocate with instrumentation --------------------
    # Streaming writer: one row group per target. Each writer.write_table call
    # flushes a row group to disk, so the data behind every completed target
    # is durable. Parquet's footer is still only written on close(), so a hard
    # crash leaves a footer-less file — recoverable but not directly readable.
    status_counts = {"SUCCESS": 0, "FALLBACK": 0, "ERROR": 0}
    targets_path = out_dir / "targets.parquet"
    has_stochastic_ctr = hasattr(model.ctr, "rng")
    with pq.ParquetWriter(str(targets_path), bench_schema.TARGETS_SCHEMA) as writer:
        for target_index, target in enumerate(eval_targets):
            target_seed = _derive_target_seed(spec.base_seed, target_index)
            if has_stochastic_ctr and target_seed is not None:
                model.ctr.rng = np.random.default_rng(target_seed)

            instr = TimingMemoryInstrument()
            result = model.geolocate(target.obs, instrument=instr)
            status_counts[result.status.name] += 1

            row = _build_target_row(target, result, instr, seed=target_seed)
            writer.write_table(_row_to_table(row))

    # --- 5. Write run.json ---------------------------------------------------
    # `rss_start` was sampled before fit/per-target work — that's the
    # baseline (Python + libs + inputs). `rss_end` is the lifetime peak.
    # Both come from `getrusage().ru_maxrss`, which is monotonic, so
    # `rss_end >= rss_start` by construction (no max() needed).
    rss_end = peak_rss_bytes()
    run_meta = {
        "run_id": run_id,
        "source": source_name,
        "setup": setup_name,
        "slice": slice_name,
        "combo_id": spec.combo_id,
        "ltd": spec.ltd,
        "mtl": spec.mtl,
        "ctr": spec.ctr,
        "ltd_kwargs": spec.ltd_kwargs,
        "mtl_kwargs": spec.mtl_kwargs,
        "ctr_kwargs": spec.ctr_kwargs,
        "base_seed": spec.base_seed,
        "enable_fallback": enable_fallback,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_fit_samples": len(fit_samples),
        "n_targets": len(eval_targets),
        "status_counts": status_counts,
        "fit_success": fit_result.success,
        "fit_error": fit_result.error.name if fit_result.error else None,
        "fit_ms": fit_meas["duration_ns"] / 1e6,
        "fit_alloc_peak_bytes": fit_meas["alloc_peak_bytes"],
        "fit_rss_peak_bytes": fit_meas["rss_peak_bytes"],
        "run_baseline_rss_bytes": rss_start,
        "run_peak_rss_bytes": rss_end,
    }
    (out_dir / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n")
    return out_dir


# ---- per-target row construction --------------------------------------------

def _derive_target_seed(base_seed: Optional[int], target_index: int) -> Optional[int]:
    """Per-target seed via SeedSequence. None passes through unchanged.

    SeedSequence is the documented way to spawn deterministic child seeds
    from a base — avoids hash-collision risk and decouples targets from
    each other (target N's RNG is independent of target N-1's state)."""
    if base_seed is None:
        return None
    state = np.random.SeedSequence([int(base_seed), int(target_index)]).generate_state(1, dtype=np.uint32)
    return int(state[0])


def _build_target_row(
    target,
    geo_result,
    instr: TimingMemoryInstrument,
    *,
    seed: Optional[int] = None,
) -> dict:
    """Flatten a (target, GeoResult, instrument) triple into a TARGETS_SCHEMA row."""
    ltd_rec = instr.get("ltd")
    mtl_rec = instr.get("mtl")
    ctr_rec = instr.get("ctr")

    # Per-VP LTD prediction forensics — nested list-of-struct column.
    ltd_predictions: list[dict] = []
    for r in geo_result.ltd_results:
        ltd_predictions.append({
            "vp_id": str(r.vp_id) if r.vp_id is not None else "",
            "success": r.success,
            "error": r.error.name if r.error else None,
            "upper_km": (r.tg_distance.upper_km if r.tg_distance else None),
            "lower_km": (r.tg_distance.lower_km if r.tg_distance else None),
        })

    pred_lat, pred_lon = (None, None)
    error_km = None
    if geo_result.coord is not None:
        pred_lat = geo_result.coord.lat
        pred_lon = geo_result.coord.lon
        error_km = haversine_distance(
            target.true_coord.lat, target.true_coord.lon,
            pred_lat, pred_lon,
        )

    mtl_result: Optional[MTLResult] = geo_result.mtl_result
    ctr_result = geo_result.ctr_result

    return {
        "target_id": target.target_id,
        "target_lat": target.true_coord.lat,
        "target_lon": target.true_coord.lon,
        "n_obs": len(target.obs),
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "status": geo_result.status.name,
        "error": geo_result.error.name if geo_result.error else None,
        "error_km": error_km,
        "ltd_ms": (ltd_rec.duration_ns / 1e6) if ltd_rec else 0.0,
        "ltd_alloc_peak_bytes": ltd_rec.alloc_peak_bytes if ltd_rec else 0,
        "ltd_rss_peak_bytes": ltd_rec.rss_peak_bytes if ltd_rec else 0,
        "mtl_ms": (mtl_rec.duration_ns / 1e6) if mtl_rec else None,
        "mtl_alloc_peak_bytes": mtl_rec.alloc_peak_bytes if mtl_rec else None,
        "mtl_rss_peak_bytes": mtl_rec.rss_peak_bytes if mtl_rec else None,
        "ctr_ms": (ctr_rec.duration_ns / 1e6) if ctr_rec else None,
        "ctr_alloc_peak_bytes": ctr_rec.alloc_peak_bytes if ctr_rec else None,
        "ctr_rss_peak_bytes": ctr_rec.rss_peak_bytes if ctr_rec else None,
        "n_ltd_success": sum(1 for r in geo_result.ltd_results if r.success),
        "ltd_predictions": ltd_predictions,
        "mtl_success": mtl_result.success if mtl_result else None,
        "mtl_error": (mtl_result.error.name if mtl_result and mtl_result.error else None),
        "mtl_intersection_kind": _intersection_kind(mtl_result),
        "ctr_success": ctr_result.success if ctr_result else None,
        "ctr_error": (ctr_result.error.name if ctr_result and ctr_result.error else None),
        "seed": seed,
    }


def _intersection_kind(mtl: Optional[MTLResult]) -> Optional[str]:
    if mtl is None or mtl.intersection is None:
        return None if mtl is None else "none"
    inter = mtl.intersection
    if isinstance(inter, MultiPolygon):
        return "multipolygon"
    if isinstance(inter, Polygon):
        return "polygon"
    if isinstance(inter, BaseGeometry):
        return type(inter).__name__.lower()
    # list[Coord] from spherical methods
    if isinstance(inter, list):
        return "vertex_list"
    return "unknown"


def _row_to_table(row: dict) -> pa.Table:
    """Wrap a single TARGETS_SCHEMA-shaped row dict as a 1-row pa.Table.

    Streaming writes go one row at a time so each target survives a crash
    as a complete row group on disk.
    """
    columns = {field.name: [row.get(field.name)] for field in bench_schema.TARGETS_SCHEMA}
    return pa.table(columns, schema=bench_schema.TARGETS_SCHEMA)
