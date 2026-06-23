# Snakemake workflow to render the centroid-aware "cluster" world-map HTMLs for
# the four textbook CBG variants across the five VP-target configs, the visual
# companion to the failure characterization (scripts/analysis/partvp/
# characterize_failures.py).
#
# Per (run_id, combo) this produces:
#   <VIZ_OUT>/<run_id>/<combo>_cluster_map.html
#   <VIZ_OUT>/<run_id>/static/<combo>/<fold>__<target_id>.json   (feasible region)
# The HTML lazy-fetches the per-target polygon JSONs at view time, so serve the
# tree via a local web server (`python -m http.server`) — file:// blocks fetch().
#
# Run from the repo root:
#   snakemake -s scripts/visualization/benchmark/v2/cluster_world_map.smk -j 4
#
# Prereqs (this workflow only renders + builds the attribution table):
#   * the benchmark runs exist (targets.parquet with mtl_participants) under
#     scripts/benchmark/v2/outputs/<global run> and .../outputs_partvp/<regional>;
#   * the per-(combo,target) feature tables exist under
#     scripts/analysis/partvp/outputs/data{,_eu}/  (extract_features.py).
# The `attribution` rule then derives per_target_failures.parquet from those.

import yaml
from pathlib import Path

VIZ_OUT = Path("scripts/visualization/benchmark/v2/outputs_cluster")
ATTRIBUTION = Path("scripts/analysis/partvp/outputs/analysis_fail/per_target_failures.parquet")
TEXTBOOK = ["vanilla_cbg", "million_scale_cbg", "octant_cbg", "spotter_cbg"]
PYTHON = ".venv/bin/python"

# config label -> config YAML that defines the four textbook combos for that run.
CONFIG_PATHS = {
    "global-global":  "scripts/benchmark/v2/config/global_as16509_final.yaml",
    "europe-europe":  "scripts/analysis/partvp/cfg_textbook/europe_as3215_eu.yaml",
    "europe-country": "scripts/analysis/partvp/cfg_textbook/europe_as3215_final_fr.yaml",
    "na-na":          "scripts/analysis/partvp/cfg_textbook/north_america_as7018_final_na.yaml",
    "na-us":          "scripts/analysis/partvp/cfg_textbook/north_america_as7018_final_us.yaml",
}

# Resolve each config to (run_id, source, setup, slices, combos, path).
CONFIG_META = {}
for label, path in CONFIG_PATHS.items():
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    available = {c.get("combo_id") for c in cfg.get("combos", [])}
    CONFIG_META[cfg["run_id"]] = {
        "path": path,
        "source": cfg["source"],
        "setup": cfg.get("setup", "probes_to_anchors"),
        "slices": cfg["slices"],
        "combos": [c for c in TEXTBOOK if c in available],
    }

PAIRS = [
    (run_id, combo)
    for run_id, meta in CONFIG_META.items()
    for combo in meta["combos"]
]


def _outputs_base(run_id, source, setup, combo):
    """The outputs root (outputs_partvp/ for regional, outputs/ for global) that
    actually holds this run's combo — mirrors cluster_world_map._resolve_outputs_root."""
    for name in ("outputs_partvp", "outputs"):
        base = Path("scripts/benchmark/v2") / name / run_id / source / setup
        if base.is_dir() and any(
            (base / f / combo / "targets.parquet").exists()
            for f in (s for s in CONFIG_META[run_id]["slices"])
        ):
            return base
    # Fall back to outputs/ so the rule still has a (missing) input to trigger on.
    return Path("scripts/benchmark/v2/outputs") / run_id / source / setup


def _per_fold_targets(wc):
    meta = CONFIG_META[wc.run_id]
    base = _outputs_base(wc.run_id, meta["source"], meta["setup"], wc.combo)
    return [str(base / slice_ / wc.combo / "targets.parquet") for slice_ in meta["slices"]]


CLI = f"{PYTHON} -m scripts.visualization.benchmark.v2.cluster_world_map"


rule all:
    input:
        [str(VIZ_OUT / run_id / f"{combo}_cluster_map.html") for run_id, combo in PAIRS]


# Build the attribution table once (all configs x variants) from the feature
# tables. Declared as a checkpoint-free rule; its output feeds every render.
rule attribution:
    input:
        script = "scripts/analysis/partvp/characterize_failures.py",
    output:
        parquet = ATTRIBUTION,
    shell:
        f"{PYTHON} -m scripts.analysis.partvp.characterize_failures"


rule render_cluster_map:
    input:
        targets = _per_fold_targets,
        config  = lambda wc: CONFIG_META[wc.run_id]["path"],
        attribution = ATTRIBUTION,
        script  = "scripts/visualization/benchmark/v2/cluster_world_map.py",
        html_tmpl = "scripts/visualization/benchmark/v2/templates/cluster_world_map.html",
        js_tmpl   = "scripts/visualization/benchmark/v2/templates/cluster_world_map.js",
    output:
        html   = VIZ_OUT / "{run_id}" / "{combo}_cluster_map.html",
        static = directory(VIZ_OUT / "{run_id}" / "static" / "{combo}"),
    shell:
        CLI + " --config {input.config} --combo {wildcards.combo}"
        " --attribution {input.attribution} --out-dir " + str(VIZ_OUT)
