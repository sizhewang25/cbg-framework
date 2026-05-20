"""PyArrow schemas — single source of truth for every parquet the v2 benchmark writes.

Keeping schemas in one module ensures the materialize → run → summarize pipeline
agrees on column names, types, and nullability without notebook-level drift.

Three input parquets (materialize-inputs):
  - VP_CONFIGS_SCHEMA       : one row per VP
  - FIT_SAMPLES_SCHEMA      : one row per (vp, target) training observation
  - EVAL_OBSERVATIONS_SCHEMA: one row per (target, vp) eval observation

Two output parquets (run-combo + summarize):
  - TARGETS_SCHEMA          : one row per eval target, with nested per-VP LTD
                              predictions and per-stage timing/memory
  - SUMMARY_SCHEMA          : one row per combo, aggregated across targets
"""

from __future__ import annotations

import pyarrow as pa

# ---- Inputs ------------------------------------------------------------------

VP_CONFIGS_SCHEMA = pa.schema([
    pa.field("vp_id", pa.string(), nullable=False),
    pa.field("lat", pa.float64(), nullable=False),
    pa.field("lon", pa.float64(), nullable=False),
    pa.field("asn", pa.int64(), nullable=True),
    pa.field("country", pa.string(), nullable=True),
])

FIT_SAMPLES_SCHEMA = pa.schema([
    pa.field("vp_id", pa.string(), nullable=False),
    pa.field("vp_lat", pa.float64(), nullable=False),
    pa.field("vp_lon", pa.float64(), nullable=False),
    pa.field("probe_id", pa.string(), nullable=False),
    pa.field("probe_lat", pa.float64(), nullable=False),
    pa.field("probe_lon", pa.float64(), nullable=False),
    pa.field("latency_ms", pa.float64(), nullable=False),
])

EVAL_OBSERVATIONS_SCHEMA = pa.schema([
    pa.field("target_id", pa.string(), nullable=False),
    pa.field("target_lat", pa.float64(), nullable=False),
    pa.field("target_lon", pa.float64(), nullable=False),
    pa.field("vp_id", pa.string(), nullable=False),
    pa.field("vp_lat", pa.float64(), nullable=False),
    pa.field("vp_lon", pa.float64(), nullable=False),
    pa.field("latency_ms", pa.float64(), nullable=False),
])

# ---- Outputs -----------------------------------------------------------------

# Per-VP LTD forensics nested into each target row. Keeps targets.parquet to
# one row per target while still capturing every LTDResult.
_LTD_PREDICTION_FIELD = pa.field(
    "ltd_predictions",
    pa.list_(pa.struct([
        pa.field("vp_id", pa.string(), nullable=False),
        pa.field("success", pa.bool_(), nullable=False),
        pa.field("error", pa.string(), nullable=True),       # Error.name or None
        pa.field("upper_km", pa.float64(), nullable=True),
        pa.field("lower_km", pa.float64(), nullable=True),
    ])),
    nullable=False,
)

TARGETS_SCHEMA = pa.schema([
    # Identification + ground truth
    pa.field("target_id", pa.string(), nullable=False),
    pa.field("target_lat", pa.float64(), nullable=False),
    pa.field("target_lon", pa.float64(), nullable=False),
    pa.field("n_obs", pa.int32(), nullable=False),

    # Final prediction
    pa.field("pred_lat", pa.float64(), nullable=True),
    pa.field("pred_lon", pa.float64(), nullable=True),
    pa.field("status", pa.string(), nullable=False),         # SUCCESS|FALLBACK|ERROR
    pa.field("error", pa.string(), nullable=True),           # Error.name or None
    pa.field("error_km", pa.float64(), nullable=True),       # haversine(true, pred)

    # Per-stage timing + memory (nanoseconds, bytes). Nullable because MTL/CTR
    # are skipped on early failures.
    pa.field("ltd_ms", pa.float64(), nullable=False),
    pa.field("ltd_peak_bytes", pa.int64(), nullable=False),
    pa.field("mtl_ms", pa.float64(), nullable=True),
    pa.field("mtl_peak_bytes", pa.int64(), nullable=True),
    pa.field("ctr_ms", pa.float64(), nullable=True),
    pa.field("ctr_peak_bytes", pa.int64(), nullable=True),

    # Stage outcome summaries (in addition to nested per-VP LTD)
    pa.field("n_ltd_success", pa.int32(), nullable=False),
    _LTD_PREDICTION_FIELD,
    pa.field("mtl_success", pa.bool_(), nullable=True),
    pa.field("mtl_error", pa.string(), nullable=True),
    pa.field("mtl_intersection_kind", pa.string(), nullable=True),  # polygon|multipolygon|vertex_list|none
    pa.field("ctr_success", pa.bool_(), nullable=True),
    pa.field("ctr_error", pa.string(), nullable=True),
])

# Uniform per-metric stat block: p5, p25, p50, p75, p95, mean, std. All stats
# are float64 — quantiles/mean/std of integer columns (peak_bytes) are still
# floats. Building this list once and reusing it for every metric guarantees
# the column naming stays consistent across the seven metrics.
SUMMARY_STATS = ("p5", "p25", "p50", "p75", "p95", "mean", "std")
SUMMARY_METRICS = (
    "error_km",
    "ltd_ms",
    "ltd_peak_bytes",
    "mtl_ms",
    "mtl_peak_bytes",
    "ctr_ms",
    "ctr_peak_bytes",
)


def _stat_fields(metric: str) -> list[pa.Field]:
    return [
        pa.field(f"{metric}_{stat}", pa.float64(), nullable=True)
        for stat in SUMMARY_STATS
    ]


# One row per combo. Built by `summarize` from each combo's targets.parquet.
SUMMARY_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("slice", pa.string(), nullable=False),
        pa.field("combo_id", pa.string(), nullable=False),
        pa.field("ltd", pa.string(), nullable=False),
        pa.field("mtl", pa.string(), nullable=False),
        pa.field("ctr", pa.string(), nullable=False),

        pa.field("n_targets", pa.int32(), nullable=False),
        pa.field("n_success", pa.int32(), nullable=False),
        pa.field("n_fallback", pa.int32(), nullable=False),
        pa.field("n_error", pa.int32(), nullable=False),
    ]
    # 7 metrics × 7 stats = 49 fields, in (metric, stat) order.
    + [field for metric in SUMMARY_METRICS for field in _stat_fields(metric)]
    + [
        # Run-level singletons from run.json.
        pa.field("fit_ms", pa.float64(), nullable=True),
        pa.field("fit_peak_bytes", pa.int64(), nullable=True),
        pa.field("run_peak_rss_bytes", pa.int64(), nullable=True),
    ]
)
