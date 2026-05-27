# Snakemake workflow to render MTL world-map HTMLs for every per-ASN
# benchmark v2 run, covering the `vanilla_cbg` and `million_scale_cbg`
# combos. ASN configs are auto-discovered by glob (`*_as*.yaml`).
#
# Run from the repo root:
#   snakemake -s scripts/visualization/benchmark/v2/mtl_world_map.smk -j 4
#
# Prereq: the bench (scripts/benchmark/v2/Snakefile) must already have
# produced `targets.parquet` for each (run, source, setup, fold, combo)
# cell — this workflow only renders, it does not re-run the pipeline.

import yaml
from pathlib import Path

CONFIG_DIR = Path("scripts/benchmark/v2/config")
BENCH_OUT  = Path("scripts/benchmark/v2/outputs")
VIZ_OUT    = Path("scripts/visualization/benchmark/v2/outputs")

# Combos we render maps for. Anything not present in a config is skipped.
TARGET_COMBOS = ["vanilla_cbg", "million_scale_cbg", "octant_cbg", "spotter_cbg"]

# Discover ASN configs and remember per-run metadata for the input function.
CONFIG_META = {}
for path in sorted(CONFIG_DIR.glob("*_as*.yaml")):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    available = {c.get("combo_id") for c in cfg.get("combos", [])}
    CONFIG_META[cfg["run_id"]] = {
        "path": str(path),
        "source": cfg["source"],
        "setup": cfg.get("setup", "probes_to_anchors"),
        "slices": cfg["slices"],
        "combos": [c for c in TARGET_COMBOS if c in available],
    }

PAIRS = [
    (run_id, combo)
    for run_id, meta in CONFIG_META.items()
    for combo in meta["combos"]
]

CLI = "python -m scripts.visualization.benchmark.v2.mtl_world_map"

# ---- Targets ----------------------------------------------------------------
rule all:
    input:
        [str(VIZ_OUT / run_id / f"{combo}_map.html") for run_id, combo in PAIRS]

# ---- One HTML per (run_id, combo) ------------------------------------------
def _per_fold_targets(wc):
    meta = CONFIG_META[wc.run_id]
    return [
        str(BENCH_OUT / wc.run_id / meta["source"] / meta["setup"]
            / slice_ / wc.combo / "targets.parquet")
        for slice_ in meta["slices"]
    ]

rule render_map:
    input:
        targets = _per_fold_targets,
        config  = lambda wc: CONFIG_META[wc.run_id]["path"],
        script  = "scripts/visualization/benchmark/v2/mtl_world_map.py",
    output:
        html = VIZ_OUT / "{run_id}" / "{combo}_map.html",
    params:
        out_dir = str(VIZ_OUT),
    shell:
        CLI + " --config {input.config} --combo {wildcards.combo}"
        " --out-dir {params.out_dir}"
