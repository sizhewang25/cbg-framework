"""Config loader + data assembly for the four-variant paper figures.

One config (YAML) declares the run_id, the six VP setups (run stems), the four
variants, and the input/output roots. Every figure script loads the same config
and pulls a tidy per-target frame via `load_setup_long`, so the data layer lives
here once rather than being re-derived per figure.

Path conventions (v2 benchmark):
  targets  : <outputs_root>/<run_stem>/<source>/<setup>/<fold>/<combo>/targets.parquet
  eval obs : <inputs_root>/<source>/<run_stem>/<setup>/<fold>/eval_observations.parquet

closest-VP distance is geometric: per target, the minimum great-circle distance
to any of its VPs (from eval_observations). It is the cross-continent /
home-continent signal and is independent of RTT.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM

# repo root = .../cbg-framework  (this file: scripts/paper/cbg_bench/_io.py)
REPO_ROOT = Path(__file__).resolve().parents[3]
CBG_BENCH_DIR = Path(__file__).resolve().parent

_TARGET_COLS = ["target_id", "target_lat", "target_lon", "status", "error_km"]


@dataclass(frozen=True)
class Setup:
    """One VP setup = one benchmark run (config stem)."""
    slug: str
    run_stem: str
    region: str = ""


@dataclass(frozen=True)
class Config:
    run_id: str
    source: str
    setup: str
    folds: list[str]
    variants: list[str]
    percentiles: list[float]
    thresholds_km: list[float]
    inputs_root: Path
    outputs_root: Path
    setups: list[Setup]

    @property
    def out_dir(self) -> Path:
        """Figure/JSON output directory: scripts/paper/cbg_bench/<run_id>/."""
        return CBG_BENCH_DIR / self.run_id

    def setup_by_slug(self, slug: str) -> Setup:
        for s in self.setups:
            if s.slug == slug:
                return s
        raise KeyError(f"No setup with slug {slug!r}. Have: {[s.slug for s in self.setups]}")


def load_config(path: str | Path) -> Config:
    """Parse a four-variant figure config YAML into a Config."""
    raw = yaml.safe_load(Path(path).read_text())

    def _root(key: str, default: str) -> Path:
        val = Path(raw.get(key, default))
        return val if val.is_absolute() else REPO_ROOT / val

    return Config(
        run_id=raw["run_id"],
        source=raw["source"],
        setup=raw["setup"],
        folds=list(raw["folds"]),
        variants=list(raw["variants"]),
        percentiles=list(raw.get("percentiles", [5, 25, 50, 75, 95])),
        thresholds_km=list(raw.get("thresholds_km", [100, 500, 1000, 2500, 5000])),
        inputs_root=_root("inputs_root", "scripts/benchmark/v2/inputs"),
        outputs_root=_root("outputs_root", "scripts/benchmark/v2/outputs"),
        setups=[Setup(**s) for s in raw["setups"]],
    )


# ---- path resolution --------------------------------------------------------

def targets_path(cfg: Config, setup: Setup, fold: str, combo: str) -> Path:
    return (cfg.outputs_root / setup.run_stem / cfg.source / cfg.setup
            / fold / combo / "targets.parquet")


def eval_obs_path(cfg: Config, setup: Setup, fold: str) -> Path:
    return (cfg.inputs_root / cfg.source / setup.run_stem / cfg.setup
            / fold / "eval_observations.parquet")


# ---- geometry ---------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance (km); same radius as rtt_model."""
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    dlat = lat2 - lat1
    dlon = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(a))


# ---- data assembly ----------------------------------------------------------

def load_combo_targets(cfg: Config, setup: Setup, combo: str) -> pd.DataFrame:
    """Concatenate <combo>/targets.parquet across all folds for one setup.

    Folds partition the target set, so the union is the full per-setup target
    list. Returns columns: target_id, target_lat, target_lon, status,
    error_km, fold.
    """
    frames: list[pd.DataFrame] = []
    for fold in cfg.folds:
        p = targets_path(cfg, setup, fold, combo)
        if not p.exists():
            continue
        df = pq.read_table(p, columns=_TARGET_COLS).to_pandas()
        df["fold"] = fold
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No targets.parquet for combo={combo!r} setup={setup.slug!r} "
            f"under {cfg.outputs_root / setup.run_stem}"
        )
    return pd.concat(frames, ignore_index=True)


def load_closest_vp(cfg: Config, setup: Setup) -> pd.DataFrame:
    """Per-target closest-VP great-circle distance (km) for one setup.

    Returns columns: target_id, closest_vp_km.
    """
    frames: list[pd.DataFrame] = []
    for fold in cfg.folds:
        p = eval_obs_path(cfg, setup, fold)
        if not p.exists():
            continue
        df = pq.read_table(
            p, columns=["target_id", "target_lat", "target_lon", "vp_lat", "vp_lon"]
        ).to_pandas()
        df["vp_km"] = haversine_km(df["target_lat"], df["target_lon"],
                                   df["vp_lat"], df["vp_lon"])
        frames.append(df[["target_id", "vp_km"]])
    if not frames:
        raise FileNotFoundError(
            f"No eval_observations.parquet for setup={setup.slug!r} "
            f"under {cfg.inputs_root / cfg.source / setup.run_stem}"
        )
    allobs = pd.concat(frames, ignore_index=True)
    closest = (allobs.groupby("target_id", as_index=False)["vp_km"]
               .min().rename(columns={"vp_km": "closest_vp_km"}))
    return closest


def load_shortest_ping_error(cfg: Config, setup: Setup) -> pd.DataFrame:
    """Per-target shortest-ping baseline error (km) for one setup.

    For each (fold, target), predict the target location as the coordinates of
    the VP with the smallest observed latency; error = haversine(true target,
    that VP). Folds partition the targets, so this is one row per target.
    Returns columns: target_id, fold, shortest_ping_km.
    """
    frames: list[pd.DataFrame] = []
    for fold in cfg.folds:
        p = eval_obs_path(cfg, setup, fold)
        if not p.exists():
            continue
        df = pq.read_table(
            p,
            columns=["target_id", "target_lat", "target_lon",
                     "vp_lat", "vp_lon", "latency_ms"],
        ).to_pandas()
        idx = df.groupby("target_id")["latency_ms"].idxmin()
        nearest = df.loc[idx].copy()
        nearest["shortest_ping_km"] = haversine_km(
            nearest["target_lat"], nearest["target_lon"],
            nearest["vp_lat"], nearest["vp_lon"],
        )
        nearest["fold"] = fold
        frames.append(nearest[["target_id", "fold", "shortest_ping_km"]])
    if not frames:
        raise FileNotFoundError(
            f"No eval_observations.parquet for setup={setup.slug!r} "
            f"under {cfg.inputs_root / cfg.source / setup.run_stem}"
        )
    return pd.concat(frames, ignore_index=True)


def load_setup_long(cfg: Config, setup: Setup,
                    variants: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Tidy long frame for one setup: one row per (variant, target).

    Columns: combo_id, target_id, target_lat, target_lon, status, error_km,
    fold, closest_vp_km. error_km is NaN/None for non-located rows; status
    distinguishes SUCCESS vs FALLBACK.
    """
    variants = list(variants) if variants is not None else cfg.variants
    closest = load_closest_vp(cfg, setup)
    out: list[pd.DataFrame] = []
    for combo in variants:
        df = load_combo_targets(cfg, setup, combo)
        df = df.merge(closest, on="target_id", how="left")
        df.insert(0, "combo_id", combo)
        out.append(df)
    return pd.concat(out, ignore_index=True)


def fallback_rate(df_combo: pd.DataFrame) -> float:
    """Fraction of rows with status == FALLBACK (over all rows for the combo)."""
    n = len(df_combo)
    if n == 0:
        return float("nan")
    return float((df_combo["status"] == "FALLBACK").sum()) / n


# ---- output helpers ---------------------------------------------------------

def ensure_out_dir(cfg: Config) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    return cfg.out_dir


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON-serializable: {type(o)}")


def dump_json(obj: Any, path: str | Path) -> Path:
    """Write `obj` as pretty JSON next to a figure (same basename, .json)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n")
    return path
