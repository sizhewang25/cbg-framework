"""Run the four textbook CBG combos of a cfg_textbook YAML over its folds.

Drives `runner.run_one_combo` directly (the one-off shell drivers were removed),
reusing the already-materialized inputs. Used to (re)generate a participant-
emitting run so `extract_features.py` can build its feature table.

    python -m scripts.analysis.partvp.run_textbook_config \\
        --config scripts/analysis/partvp/cfg_textbook/north_america_as7018_final_na.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from scripts.benchmark.v2.inputs import (
    DEFAULT_INPUTS_ROOT,
    inputs_dir_for,
    outputs_combo_dir,
)
from scripts.benchmark.v2.runner import ComboSpec, run_one_combo
from scripts.benchmark.v2 import cli as v2cli

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = yaml.safe_load(args.config.read_text())
    run_id = cfg["run_id"]
    source = cfg["source"]
    setup = cfg.get("setup", "probes_to_anchors")
    slices = cfg["slices"]
    src_kwargs = cfg["source_kwargs"]
    seed = cfg.get("seed")
    enable_fallback = cfg.get("enable_fallback", True)
    outputs_root = Path(cfg.get("outputs_root", "scripts/benchmark/v2/outputs"))

    for slice_ in slices:
        src = v2cli.SOURCES[source](slice=slice_, setup=setup, **src_kwargs)
        inputs_dir = inputs_dir_for(src, DEFAULT_INPUTS_ROOT, run_id=run_id)
        if not (inputs_dir / "eval_observations.parquet").exists():
            raise SystemExit(f"inputs not materialized: {inputs_dir}")
        for c in cfg["combos"]:
            spec = ComboSpec(
                combo_id=c["combo_id"],
                ltd=c["ltd"], mtl=c["mtl"], ctr=c["ctr"],
                ltd_kwargs=c.get("ltd_kwargs", {}) or {},
                mtl_kwargs=c.get("mtl_kwargs", {}) or {},
                ctr_kwargs=c.get("ctr_kwargs", {}) or {},
                base_seed=seed,
            )
            out_dir = outputs_combo_dir(outputs_root, run_id, src, c["combo_id"])
            logger.info("run %s / %s -> %s", slice_, c["combo_id"], out_dir)
            run_one_combo(
                spec, inputs_dir=inputs_dir, out_dir=out_dir, run_id=run_id,
                source_name=source, slice_name=slice_, setup_name=setup,
                enable_fallback=enable_fallback,
            )
    logger.info("done: %s", run_id)


if __name__ == "__main__":
    main()
