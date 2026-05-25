"""Materialize a DataSource into the four input parquets the runner consumes.

The runner is fan-out across (source × slice × combo). Querying ClickHouse or
re-reading a CSV per-combo would be wasteful, so this step caches the source
into a deterministic on-disk shape that downstream combos all read from.

Layout written by `materialize_inputs(source, root=..., run_id=...)`:
    <root>/<source_name>/<run_id>/<setup_id>/<slice_id>/
        vp_configs.parquet
        tg_configs.parquet
        fit_samples.parquet
        eval_observations.parquet
        manifest.json

`run_id` is a mandatory path component so different runs that materialize the
same source under different source_kwargs (e.g. different per-ASN VP corpora
under `ripe_atlas_asn_corpora`) get parallel directory trees instead of
clobbering each other.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.benchmark.v2 import schema as bench_schema
from scripts.benchmark.v2.sources.base import DataSource, EvalTarget, TgConfig, VpConfig
from scripts.framework.v2 import FitSample


DEFAULT_INPUTS_ROOT = Path(__file__).resolve().parent / "inputs"


def inputs_dir_for(
    source: DataSource,
    root: Path = DEFAULT_INPUTS_ROOT,
    *,
    run_id: str,
) -> Path:
    """Canonical inputs-dir path for one source configuration.

    Layout: `<root>/<source.name>/<run_id>/<setup_id>/<slice_id>/`. `run_id`
    isolates materialized inputs across runs that share a source but differ
    in source_kwargs (e.g. per-ASN VP corpora). `setup_id` separates the two
    role assignments (probes_to_anchors vs anchors_to_probes).
    """
    return root / source.name / run_id / source.setup_id() / source.slice_id()


def outputs_combo_dir(
    outputs_root: Path,
    run_id: str,
    source: DataSource,
    combo_id: str,
) -> Path:
    """Canonical outputs-dir path for one (run, combo) cell.

    Layout: `<outputs_root>/<run_id>/<source.name>/<setup_id>/<slice_id>/<combo_id>/`.
    Mirrors `inputs_dir_for` so swapping `setup` keeps inputs and outputs
    in parallel trees — no chance of two configurations writing run.json
    into the same directory.
    """
    return (
        outputs_root
        / run_id
        / source.name
        / source.setup_id()
        / source.slice_id()
        / combo_id
    )


def materialize_inputs(
    source: DataSource,
    *,
    run_id: str,
    root: Path = DEFAULT_INPUTS_ROOT,
) -> Path:
    """Write source's VPs, fit samples, and eval observations to parquet.

    Returns the output directory path."""
    out_dir = inputs_dir_for(source, root, run_id=run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_vps = _write_vp_configs(source.iter_vp_configs(), out_dir / "vp_configs.parquet")
    n_tgs = _write_tg_configs(source.iter_tg_configs(), out_dir / "tg_configs.parquet")
    n_fit = _write_fit_samples(source.iter_fit_samples(), out_dir / "fit_samples.parquet")
    n_obs, n_targets = _write_eval_observations(
        source.iter_eval_targets(), out_dir / "eval_observations.parquet",
    )

    manifest = {
        "source": source.name,
        "run_id": run_id,
        "setup": source.setup_id(),
        "slice": source.slice_id(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_vps": n_vps,
        "n_tg_configs": n_tgs,
        "n_fit_samples": n_fit,
        "n_eval_observations": n_obs,
        "n_eval_targets": n_targets,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_dir


# ---- writers (per-parquet) ---------------------------------------------------

def _write_vp_configs(rows: Iterable[VpConfig], path: Path) -> int:
    items = list(rows)
    table = pa.table(
        {
            "vp_id": [r.vp_id for r in items],
            "lat": [r.lat for r in items],
            "lon": [r.lon for r in items],
            "asn": [r.asn for r in items],
            "country": [r.country for r in items],
            "city": [r.city for r in items],
        },
        schema=bench_schema.VP_CONFIGS_SCHEMA,
    )
    pq.write_table(table, path)
    return len(items)


def _write_tg_configs(rows: Iterable[TgConfig], path: Path) -> int:
    items = list(rows)
    table = pa.table(
        {
            "tg_id": [r.tg_id for r in items],
            "lat": [r.lat for r in items],
            "lon": [r.lon for r in items],
            "asn": [r.asn for r in items],
            "country": [r.country for r in items],
            "city": [r.city for r in items],
        },
        schema=bench_schema.TG_CONFIGS_SCHEMA,
    )
    pq.write_table(table, path)
    return len(items)


def _write_fit_samples(rows: Iterable[FitSample], path: Path) -> int:
    items = list(rows)
    # FitSample doesn't carry a probe identifier — it's just (vp_id, vp_coord,
    # probe_coord, latency). For the materialized parquet we synthesize a
    # probe_id from the probe_coord rounded to 4dp; for RIPE Atlas this gives
    # the anchor IP indirectly through the eval pass.
    table = pa.table(
        {
            "vp_id": [str(s.vp_id) for s in items],
            "vp_lat": [s.vp_coord.lat for s in items],
            "vp_lon": [s.vp_coord.lon for s in items],
            "probe_id": [f"{s.probe_coord.lat:.4f}_{s.probe_coord.lon:.4f}" for s in items],
            "probe_lat": [s.probe_coord.lat for s in items],
            "probe_lon": [s.probe_coord.lon for s in items],
            "latency_ms": [float(s.latency) for s in items],
        },
        schema=bench_schema.FIT_SAMPLES_SCHEMA,
    )
    pq.write_table(table, path)
    return len(items)


def _write_eval_observations(
    targets: Iterable[EvalTarget], path: Path,
) -> tuple[int, int]:
    # Flatten to one row per (target, vp). target_lat/lon repeated per row —
    # parquet's compression handles the redundancy and the consumer doesn't
    # have to grouping-join to find ground truth.
    target_ids: list[str] = []
    target_lats: list[float] = []
    target_lons: list[float] = []
    vp_ids: list[str] = []
    vp_lats: list[float] = []
    vp_lons: list[float] = []
    latencies: list[float] = []

    n_targets = 0
    for t in targets:
        n_targets += 1
        for vp_id, vp_coord, latency in t.obs:
            target_ids.append(t.target_id)
            target_lats.append(t.true_coord.lat)
            target_lons.append(t.true_coord.lon)
            vp_ids.append(str(vp_id))
            vp_lats.append(vp_coord.lat)
            vp_lons.append(vp_coord.lon)
            latencies.append(float(latency))

    table = pa.table(
        {
            "target_id": target_ids,
            "target_lat": target_lats,
            "target_lon": target_lons,
            "vp_id": vp_ids,
            "vp_lat": vp_lats,
            "vp_lon": vp_lons,
            "latency_ms": latencies,
        },
        schema=bench_schema.EVAL_OBSERVATIONS_SCHEMA,
    )
    pq.write_table(table, path)
    return len(target_ids), n_targets


# ---- readers (consumed by runner) -------------------------------------------

def load_fit_samples_parquet(path: Path) -> list[FitSample]:
    """Read fit_samples.parquet back into FitSample objects."""
    from scripts.framework.v2.types import Coord, Latency, VpId

    table = pq.read_table(path)
    rows = table.to_pylist()
    return [
        FitSample(
            vp_id=VpId(r["vp_id"]),
            vp_coord=Coord(lat=r["vp_lat"], lon=r["vp_lon"]),
            probe_coord=Coord(lat=r["probe_lat"], lon=r["probe_lon"]),
            latency=Latency(r["latency_ms"]),
        )
        for r in rows
    ]


def load_eval_targets_parquet(path: Path) -> "list[EvalTarget]":
    """Read eval_observations.parquet back into per-target groups.

    Imported lazily because the EvalTarget dataclass lives next to DataSource.
    """
    from scripts.framework.v2.types import Coord, Latency, VpId

    table = pq.read_table(path)
    rows = table.to_pylist()
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r["target_id"], []).append(r)

    out: list[EvalTarget] = []
    for target_id in sorted(grouped):
        first = grouped[target_id][0]
        true_coord = Coord(lat=first["target_lat"], lon=first["target_lon"])
        obs = [
            (
                VpId(r["vp_id"]),
                Coord(lat=r["vp_lat"], lon=r["vp_lon"]),
                Latency(r["latency_ms"]),
            )
            for r in grouped[target_id]
        ]
        out.append(EvalTarget(target_id=target_id, true_coord=true_coord, obs=obs))
    return out
