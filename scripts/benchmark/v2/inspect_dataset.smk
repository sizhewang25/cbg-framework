# Pre-benchmark dataset geometry: canonical CSV -> target space -> the §7.3 /
# §8.1 artifacts, driven by the same unified config yaml that drives
# ./Snakefile (only `run_id`, `source`, `setup`, `slices`, `source_kwargs` and
# the optional `precheck:` / `analysis.common:` blocks are read; the combo grid
# is ignored).
#
# Run with:
#   snakemake -s scripts/benchmark/v2/inspect_dataset.smk \
#       --configfile configs/as01-260728-260802-mesh.yaml -j 1
#
# Every rule here describes the *dataset*, never a method's output, so none of
# them needs a benchmark run: `materialize-target-space --configfile` builds the
# node sets by constructing the config's DataSource per slice, which is the same
# union the folds would later record. Run this before spending a run — if the
# VP geometry or the answer space is not what you expected, the run will not
# fix it.
#
# The rules run the repo's own `.venv/bin/python`, found relative to this file,
# so neither PATH nor the environment `snakemake` was launched from matters.
# Override with `--config python=/path/to/python` if the dependencies live
# elsewhere. A missing dependency is reported once, up front, naming the
# interpreter — rather than as a traceback from whichever rule hit it first.
#
# Outputs (two trees, both keyed by run_id rather than by CSV stem, because
# that is where scripts/analysis/v3 looks for them):
#
#   <outputs_root>/<run_id>/<source>/<setup>/
#       targets.csv  vps.csv          # the evaluated node sets
#       clusters/{clusters,assignments}.csv + meta.json
#       target_space.json             # provenance + the canonical edge CSV
#   <outputs_root>/<run_id>/eval_source/
#       <stem>_eval_per_target.csv, _eval_clusters.csv, _eval_stats.json
#       <stem>_clusters/, _vp_mesh_km.csv, _cluster_mesh_km.csv, _cluster_map.png
#   <analysis_root>/<run_id>/
#       target-answer-space/<grid>-<res>/{seeds,assignments,seed_mesh_km}.csv
#                                        answer_space_map.png
#       bipartite-graph/<grid>-<res>/{vp_nodes,target_nodes}.csv + CDFs
#                                    bipartite_nodes_map.png
#                                    bipartite_flows_map.png
#                                    distance_cdf.png
#       target-proximity/<grid>-<res>/target_labels.csv
#
# The maps read the CSVs the build-* rules wrote and nothing else, so their
# styling is per-run rather than structural: the two plot rules pass the config
# through to the v3 CLI's own `--config`, which applies an
# `analysis.plot-answer-space:` / `analysis.plot-bipartite-graph:` block (e.g.
# `us_only: true`) as the command's defaults.
#
# Config keys (all optional, all with the CLIs' own defaults):
#   analysis.common.grid        (default h3)
#   analysis.common.resolution  (default [4]; a list fans out)
#   precheck.cluster_radius_km  (default 50)   .top_n_neighbors    (default 5)
#   precheck.anycast_delta_ms   (default 10)   .spearman_min_pairs (default 8)
#   precheck.allow_mesh_edge_set (default false) — see EDGE_SET_IS_SUPERSET

import subprocess
import sys
from pathlib import Path

# ---- Interpreter ------------------------------------------------------------
# Every rule shells out to `python -m ...`, so which interpreter that resolves
# to decides whether the run finds pandas or h3 at all. Pinned here rather than
# left to PATH: the dependencies live in the repo's own .venv, while `snakemake`
# is routinely launched from somewhere else entirely -- and the failure that
# causes surfaces deep in a rule, as a bare ModuleNotFoundError for whichever
# import happened to come first.
_REPO_ROOT = Path(workflow.basedir).parents[2]
_VENV_PYTHON = _REPO_ROOT / ".venv" / "bin" / "python"
PYTHON = str(
    config.get("python")
    or (_VENV_PYTHON if _VENV_PYTHON.exists() else sys.executable)
)


def _missing_modules(python, modules):
    """Which of `modules` the interpreter cannot import, via one subprocess."""
    code = (
        "import importlib.util, sys; "
        "print(' '.join(m for m in sys.argv[1:] "
        "if importlib.util.find_spec(m) is None))"
    )
    try:
        out = subprocess.run(
            [python, "-c", code, *modules],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"cannot run the configured interpreter {python!r}: {exc}")
    if out.returncode != 0:
        raise ValueError(
            f"cannot run the configured interpreter {python!r}: "
            f"{out.stderr.strip() or out.returncode}"
        )
    return out.stdout.split()


# ---- Resolve config ---------------------------------------------------------
# Same nested-then-flat resolution as ./Snakefile's `bcfg`, so a key under a
# unified config's `benchmark:` block is visible here too. Without this a
# nested `eval_kept_traffic_fraction` is silently ignored and the precheck
# scores a wider set than the benchmark evaluates.
_BENCH = config.get("benchmark") or {}


def bcfg(key, default=None):
    return _BENCH[key] if key in _BENCH else config.get(key, default)


# The path of the config itself, which `materialize-target-space --configfile`
# re-reads. Snakemake records it, so it is never reconstructed from `run_id`.
_CONFIGFILES = list(getattr(workflow, "configfiles", []) or [])
if not _CONFIGFILES:
    raise ValueError(
        "inspect_dataset.smk must be invoked with --configfile <config.yaml>: "
        "materialize-target-space re-reads that file to build the source"
    )
CONFIGFILE = Path(_CONFIGFILES[0])

RUN_ID = bcfg("run_id")
SOURCE = bcfg("source")
if not RUN_ID or not SOURCE:
    # `benchmark: {}` is the documented shape for an analysis-only config over
    # an already-published run, so say that rather than just naming the missing
    # keys — those runs' node sets already exist and are not rebuilt from here.
    empty_block = isinstance(config.get("benchmark"), dict) and not _BENCH
    raise ValueError(
        f"{CONFIGFILE}: `run_id` and `source` are both required"
        + (
            " — this config has an empty `benchmark: {}` block, meaning it is "
            "analysis-only over a run whose outputs already exist. Run the "
            "scripts/analysis/v3 commands against that run directly; there is "
            "no canonical CSV here to build a target space from."
            if empty_block else ""
        )
    )
SETUP = bcfg("setup", "probes_to_anchors")
SRC_KWARGS = bcfg("source_kwargs", {}) or {}

OUTPUTS_ROOT = Path(bcfg("outputs_root", "outputs/benchmark/v2"))
ANALYSIS_ROOT = Path(bcfg("analysis_root", "outputs/analysis/v3"))

# ---- Which CSV is this run's edge set? --------------------------------------
# `weighted_csv_path` (traffic_weighted_csv, precomputed mode) IS the
# traffic-weighted dataset, so score it directly and apply no further eval
# filter. Every other source scores its mesh and mirrors the filters below.
# Mirrors `cli.py`'s `_edge_csv_from_kwargs`, which writes the same decision
# into target_space.json.
_PRECOMPUTED_WEIGHTED = "weighted_csv_path" in SRC_KWARGS
if _PRECOMPUTED_WEIGHTED:
    CSV_PATH = Path(SRC_KWARGS["weighted_csv_path"])
elif "mesh_csv_path" in SRC_KWARGS:
    CSV_PATH = Path(SRC_KWARGS["mesh_csv_path"])
elif "csv_path" in SRC_KWARGS:
    CSV_PATH = Path(SRC_KWARGS["csv_path"])
elif "test_path" in SRC_KWARGS:
    CSV_PATH = Path(SRC_KWARGS["test_path"])
else:
    raise ValueError(
        "inspect_dataset.smk needs source_kwargs.csv_path (generic_csv), "
        "source_kwargs.mesh_csv_path / weighted_csv_path "
        "(traffic_weighted_csv), or source_kwargs.test_path "
        "(generic_presplit) in the config yaml"
    )

# ---- Eval-side filters ------------------------------------------------------
# Read from the same keys the real benchmark run uses (not from `precheck:`) so
# the precheck can never drift from what materialize-inputs would evaluate.
MIN_OBS = SRC_KWARGS.get("min_obs")
# In precomputed mode the CSV above is already the filtered subset, so
# re-applying a threshold would cut it twice.
EVAL_PAIR_WEIGHT_MIN = (
    None if _PRECOMPUTED_WEIGHTED
    else (bcfg("eval_pair_weight_min") or SRC_KWARGS.get("eval_pair_weight_min"))
)
EVAL_KEPT_TRAFFIC_FRACTION = (
    None if _PRECOMPUTED_WEIGHTED
    else (bcfg("eval_kept_traffic_fraction")
          or SRC_KWARGS.get("eval_kept_traffic_fraction"))
)
if EVAL_PAIR_WEIGHT_MIN is not None and EVAL_KEPT_TRAFFIC_FRACTION is not None:
    raise ValueError(
        "config has both eval_pair_weight_min and eval_kept_traffic_fraction "
        "— the benchmark run itself only accepts one, so the precheck can't "
        "pick a side"
    )

# An on-the-fly traffic-weighted arm prunes flows in memory, so no file holds
# its edge set and CSV_PATH above is a strict *superset* of it. The node-set
# rules are unaffected (they go through the DataSource, which applies the
# filter), but the bipartite graph is edge-level and would report mesh density
# as this arm's — so it is left out of `rule all` rather than published wrong.
#
# `precheck.allow_mesh_edge_set: true` opts back in, and is what makes that
# choice reviewable: it describes the mesh graph, which is a real object and a
# fair upper bound on the arm's connectivity, as long as nobody reads it as the
# weighted arm's own. build-bipartite-graph refuses the substitution unless it
# is handed the CSV explicitly, so the flag has to reach it as --source-csv.
EDGE_SET_IS_SUPERSET = (
    EVAL_PAIR_WEIGHT_MIN is not None or EVAL_KEPT_TRAFFIC_FRACTION is not None
)

# ---- Precheck knobs ---------------------------------------------------------
PRECHECK = bcfg("precheck", {}) or {}
ALLOW_MESH_EDGE_SET = bool(PRECHECK.get("allow_mesh_edge_set", False))
CLUSTER_RADIUS_KM = float(PRECHECK.get("cluster_radius_km", 50.0))
TOP_N_NEIGHBORS = int(PRECHECK.get("top_n_neighbors", 5))
ANYCAST_DELTA_MS = float(PRECHECK.get("anycast_delta_ms", 10.0))
SPEARMAN_MIN_PAIRS = int(PRECHECK.get("spearman_min_pairs", 8))

# ---- Answer-space quantization ----------------------------------------------
_ANALYSIS = (config.get("analysis") or {}).get("common") or {}
GRID = _ANALYSIS.get("grid", "h3")
_RES = _ANALYSIS.get("resolution", [4])
RESOLUTIONS = [int(r) for r in (_RES if isinstance(_RES, (list, tuple)) else [_RES])]

# ---- Derived paths ----------------------------------------------------------
STEM = CSV_PATH.stem
SETUP_DIR = OUTPUTS_ROOT / RUN_ID / SOURCE / SETUP
EVAL_DIR = OUTPUTS_ROOT / RUN_ID / "eval_source"

TARGET_SPACE_JSON = SETUP_DIR / "target_space.json"
EVAL_STATS = EVAL_DIR / f"{STEM}_eval_stats.json"
EVAL_PER_TARGET = EVAL_DIR / f"{STEM}_eval_per_target.csv"

BENCH_CLI = f"{PYTHON} -m scripts.benchmark.v2.cli"
V3_CLI = f"{PYTHON} -m scripts.analysis.v3.cli"

# Preflight. The grid backend is checked by name because it is imported lazily
# (h3grid.py / healpix.py import inside the function), so an interpreter missing
# it gets all the way to build-answer-space before saying so.
_GRID_BACKEND = {"h3": "h3", "healpix": "astropy_healpix"}.get(GRID)
_missing = _missing_modules(
    PYTHON,
    ["pandas", "numpy", "pyarrow", "typer", "yaml"]
    + ([_GRID_BACKEND] if _GRID_BACKEND else []),
)
if _missing:
    raise ValueError(
        f"interpreter {PYTHON} is missing: {', '.join(_missing)}. "
        f"This workflow defaults to {_VENV_PYTHON} "
        f"({'present' if _VENV_PYTHON.exists() else 'ABSENT'}); pass "
        f"`--config python=/path/to/python` to point it elsewhere."
        # Checked rather than asserted: this venv currently has no activate
        # script, and "activate it first" is the natural wrong guess.
        + (
            f" Note {_VENV_PYTHON.parent} has no `activate` script, so"
            f" `source .venv/bin/activate` is not how to reach it — invoke the"
            f" interpreter by path."
            if _VENV_PYTHON.exists() and not (_VENV_PYTHON.parent / "activate").exists()
            else ""
        )
    )


def _analysis_out(kind, filename):
    return [
        str(ANALYSIS_ROOT / RUN_ID / kind / f"{GRID}-{r}" / filename)
        for r in RESOLUTIONS
    ]


ANSWER_SPACE = _analysis_out("target-answer-space", "seeds.csv")
BIPARTITE = _analysis_out("bipartite-graph", "meta.json")
PROXIMITY = _analysis_out("target-proximity", "target_labels.csv")
ANSWER_SPACE_MAP = _analysis_out("target-answer-space", "answer_space_map.png")
BIPARTITE_MAPS = _analysis_out("bipartite-graph", "bipartite_flows_map.png")

SKIP_BIPARTITE = EDGE_SET_IS_SUPERSET and not ALLOW_MESH_EDGE_SET
if SKIP_BIPARTITE:
    print(
        f"[inspect_dataset] NOTE: {RUN_ID} filters traffic on the fly, so no file "
        f"holds its edge set and the bipartite graph is excluded from `all` — "
        f"{CSV_PATH} has more edges than this arm evaluates. Either derive a "
        f"weighted CSV "
        f"(scripts/processing/source/derive_traffic_weighted_cbg_data.smk) and "
        f"point `weighted_csv_path` at it, which restores the rule with an exact "
        f"edge set, or set `precheck.allow_mesh_edge_set: true` to describe the "
        f"mesh graph and read it as such.",
        file=sys.stderr,
    )


# ---- Targets ----------------------------------------------------------------
rule all:
    input:
        str(TARGET_SPACE_JSON),
        str(EVAL_STATS),
        ANSWER_SPACE,
        ANSWER_SPACE_MAP,
        PROXIMITY,
        *([] if SKIP_BIPARTITE else [BIPARTITE, BIPARTITE_MAPS]),


# ---- [1] target space: config + canonical CSV -> node sets + clusters -------
# No benchmark output is read. The node sets come from the config's DataSource
# (one construction per slice, unioned), so `min_obs` drops and the
# traffic-weighted survivor set are already applied — this is the roster the
# run would evaluate, not the CSV's raw unique ids.
rule target_space:
    input:
        csv = str(CSV_PATH),
        config = str(CONFIGFILE),
    output:
        manifest = str(TARGET_SPACE_JSON),
        targets = str(SETUP_DIR / "targets.csv"),
        vps = str(SETUP_DIR / "vps.csv"),
        clusters = str(SETUP_DIR / "clusters" / "clusters.csv"),
    params:
        outputs_root = str(OUTPUTS_ROOT),
        radius = CLUSTER_RADIUS_KM,
    shell:
        BENCH_CLI + " materialize-target-space"
        " --configfile {input.config}"
        " --outputs-root {params.outputs_root}"
        " --radius-km {params.radius}"


# ---- [2] eval_source: canonical CSV -> per-target / per-cluster metrics -----
# Deliberately a separate rule rather than `materialize-target-space
# --with-eval-source`: that flag is the convenience path and only forwards the
# radius, while `precheck:` carries four more knobs. --out-dir is the run's own
# eval_source/, which is where build-proximity reads the shortest-ping VP from
# (eval-source's own default is the CSV's directory, where nothing looks).
rule eval_source:
    input:
        csv = str(CSV_PATH),
    output:
        stats = str(EVAL_STATS),
        per_target = str(EVAL_PER_TARGET),
        clusters_dir = directory(str(EVAL_DIR / f"{STEM}_clusters")),
    params:
        out_dir = str(EVAL_DIR),
        radius = CLUSTER_RADIUS_KM,
        top_n = TOP_N_NEIGHBORS,
        anycast_delta = ANYCAST_DELTA_MS,
        spearman_min = SPEARMAN_MIN_PAIRS,
        min_obs_flag = f"--min-obs {MIN_OBS}" if MIN_OBS is not None else "",
        eval_pair_weight_min_flag = (
            f"--eval-pair-weight-min {EVAL_PAIR_WEIGHT_MIN}"
            if EVAL_PAIR_WEIGHT_MIN is not None else ""
        ),
        eval_kept_traffic_fraction_flag = (
            f"--eval-kept-traffic-fraction {EVAL_KEPT_TRAFFIC_FRACTION}"
            if EVAL_KEPT_TRAFFIC_FRACTION is not None else ""
        ),
    shell:
        BENCH_CLI + " eval-source"
        " --csv {input.csv}"
        " --out-dir {params.out_dir}"
        " --cluster-radius-km {params.radius}"
        " --top-n-neighbors {params.top_n}"
        " --anycast-delta-ms {params.anycast_delta}"
        " --spearman-min-pairs {params.spearman_min}"
        " {params.min_obs_flag}"
        " {params.eval_pair_weight_min_flag}"
        " {params.eval_kept_traffic_fraction_flag}"


# ---- [3] answer space: targets -> grid cells -> seeds -----------------------
rule answer_space:
    input:
        targets = str(SETUP_DIR / "targets.csv"),
    output:
        seeds = str(ANALYSIS_ROOT / RUN_ID / "target-answer-space" / f"{GRID}-{{res}}" / "seeds.csv"),
    params:
        run_id = RUN_ID,
        outputs_root = str(OUTPUTS_ROOT),
        analysis_root = str(ANALYSIS_ROOT),
        grid = GRID,
    shell:
        V3_CLI + " build-answer-space"
        " --run-id {params.run_id}"
        " --outputs-root {params.outputs_root}"
        " --analysis-root {params.analysis_root}"
        " --grid {params.grid}"
        " -r {wildcards.res}"


# ---- [4] bipartite graph: VP nodes + target nodes + measured edges ----------
# The only rule here that reads the edge set rather than the node sets, hence
# the `target_space.json` dependency: build-bipartite-graph resolves the
# canonical CSV from it when no benchmark run has written eval_stats yet.
rule bipartite_graph:
    input:
        manifest = str(TARGET_SPACE_JSON),
        vps = str(SETUP_DIR / "vps.csv"),
        seeds = str(ANALYSIS_ROOT / RUN_ID / "target-answer-space" / f"{GRID}-{{res}}" / "seeds.csv"),
        csv = str(CSV_PATH),
    output:
        meta = str(ANALYSIS_ROOT / RUN_ID / "bipartite-graph" / f"{GRID}-{{res}}" / "meta.json"),
    params:
        run_id = RUN_ID,
        outputs_root = str(OUTPUTS_ROOT),
        analysis_root = str(ANALYSIS_ROOT),
        grid = GRID,
        source_csv_flag = (
            f"--source-csv {CSV_PATH}" if EDGE_SET_IS_SUPERSET else ""
        ),
    shell:
        V3_CLI + " build-bipartite-graph"
        " --run-id {params.run_id}"
        " --outputs-root {params.outputs_root}"
        " --analysis-root {params.analysis_root}"
        " --grid {params.grid}"
        " -r {wildcards.res}"
        " {params.source_csv_flag}"


# ---- [5] proximity: the four-flag VP diamond per target ---------------------
# Needs eval_source: the shortest-ping VP is resolved there once, and
# build-proximity reads that row rather than re-minimizing over RTT, so the
# baseline's VP cannot differ between the two.
rule proximity:
    input:
        seeds = str(ANALYSIS_ROOT / RUN_ID / "target-answer-space" / f"{GRID}-{{res}}" / "seeds.csv"),
        per_target = str(EVAL_PER_TARGET),
        vps = str(SETUP_DIR / "vps.csv"),
    output:
        labels = str(ANALYSIS_ROOT / RUN_ID / "target-proximity" / f"{GRID}-{{res}}" / "target_labels.csv"),
    params:
        run_id = RUN_ID,
        outputs_root = str(OUTPUTS_ROOT),
        analysis_root = str(ANALYSIS_ROOT),
        grid = GRID,
    shell:
        V3_CLI + " build-proximity"
        " --run-id {params.run_id}"
        " --outputs-root {params.outputs_root}"
        " --analysis-root {params.analysis_root}"
        " --grid {params.grid}"
        " -r {wildcards.res}"


# ---- [6] answer-space map ---------------------------------------------------
# `--config` goes before the command name: it belongs to the Typer *group*, and
# supplies this command's `analysis.plot-answer-space:` block as click defaults.
# The explicit flags below still win, so only styling comes from the config.
rule plot_answer_space:
    input:
        seeds = str(ANALYSIS_ROOT / RUN_ID / "target-answer-space" / f"{GRID}-{{res}}" / "seeds.csv"),
        config = str(CONFIGFILE),
    output:
        png = str(ANALYSIS_ROOT / RUN_ID / "target-answer-space" / f"{GRID}-{{res}}" / "answer_space_map.png"),
    params:
        run_id = RUN_ID,
        outputs_root = str(OUTPUTS_ROOT),
        analysis_root = str(ANALYSIS_ROOT),
        grid = GRID,
    shell:
        V3_CLI + " --config {input.config} plot-answer-space"
        " --run-id {params.run_id}"
        " --outputs-root {params.outputs_root}"
        " --analysis-root {params.analysis_root}"
        " --grid {params.grid}"
        " -r {wildcards.res}"


# ---- [7] bipartite topology + flow maps -------------------------------------
# Three PNGs from one invocation; the flow map is named as the output because
# it is the one that has to be regenerated when the edge set changes.
rule plot_bipartite_graph:
    input:
        meta = str(ANALYSIS_ROOT / RUN_ID / "bipartite-graph" / f"{GRID}-{{res}}" / "meta.json"),
        seeds = str(ANALYSIS_ROOT / RUN_ID / "target-answer-space" / f"{GRID}-{{res}}" / "seeds.csv"),
        config = str(CONFIGFILE),
    output:
        flows = str(ANALYSIS_ROOT / RUN_ID / "bipartite-graph" / f"{GRID}-{{res}}" / "bipartite_flows_map.png"),
        nodes = str(ANALYSIS_ROOT / RUN_ID / "bipartite-graph" / f"{GRID}-{{res}}" / "bipartite_nodes_map.png"),
        cdf = str(ANALYSIS_ROOT / RUN_ID / "bipartite-graph" / f"{GRID}-{{res}}" / "distance_cdf.png"),
    params:
        run_id = RUN_ID,
        outputs_root = str(OUTPUTS_ROOT),
        analysis_root = str(ANALYSIS_ROOT),
        grid = GRID,
    shell:
        V3_CLI + " --config {input.config} plot-bipartite-graph"
        " --run-id {params.run_id}"
        " --outputs-root {params.outputs_root}"
        " --analysis-root {params.analysis_root}"
        " --grid {params.grid}"
        " -r {wildcards.res}"
