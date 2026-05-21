# Snakemake primer — how our Snakefiles actually work

Snakemake is a build system pretending to be a workflow engine. The mental model is the same as `make`: **declare which files you want; Snakemake figures out the steps to make them.**

## The core idea

```
You declare:  "I want this output file."
              ↓
Snakemake:   "OK, which rule produces it?"
              ↓
              "That rule needs these input files."
              ↓
              "Which rule produces those?"  (recurse until you hit files that exist on disk)
              ↓
              Builds a DAG, runs the leaves first, fans out with -j N.
```

It's all about **paths**. Rules don't have names you call; they're matched by the shape of file paths they can produce.

## Anatomy of one rule

From [scripts/analysis/Snakefile](../scripts/analysis/Snakefile):

```python
rule phase_runtime:                                      # ← name (for logs only)
    input:
        summary = RUN_DIR / "summary.parquet",           # ← files this rule needs
    output:
        png = SLICE_OUT / "plot_phase_runtime.png",      # ← files this rule produces
    params:
        run_dir = str(RUN_DIR),                          # ← values for the shell template
        source = SOURCE,
        stat = RUNTIME_STAT,
    shell:
        CLI + ".plot_phase_runtime"                      # ← the command, with {params}/{wildcards} substituted
        " --run-dir {params.run_dir}"
        " --source {params.source}"
        " --slice {wildcards.slice}"
        " --stat {params.stat}"
        " --out {output.png}"
```

Snakemake runs this rule **whenever the requested output path matches the `output:` pattern** *and* the input file exists (or another rule can produce it).

## Wildcards — the bit that's confusing

The output is `SLICE_OUT / "plot_phase_runtime.png"`, where `SLICE_OUT = ANALYSIS_ROOT / RUN_ID / SOURCE / SETUP / "{slice}"`. That `{slice}` is a **wildcard** — a placeholder Snakemake matches against actual paths.

If you ask Snakemake for:
```
scripts/analysis/outputs/smoke-003/vultr_csv/anchors_to_probes/top1/plot_phase_runtime.png
```

Snakemake pattern-matches against `phase_runtime`'s output template and concludes `{slice} = top1`. From that point on, `{wildcards.slice}` is literally `"top1"` everywhere in the rule body, including the shell command.

This is what lets one rule produce N output files. The wildcard is the rule's "parameter."

## `rule all` — the entry point

```python
rule all:
    input:
        expand(str(SLICE_OUT / "{plot}.png"), slice=SLICES, plot=_PLOTS)
```

`rule all` is the conventional name for "the target rule." When you run `snakemake` with no rule arg, it runs `rule all` and asks for everything in its `input:`. Note it's `input:`, not `output:` — `rule all` doesn't *make* anything; it requests files.

`expand()` is a string formatter. Given `SLICES=["top1"]` and `_PLOTS=["plot_error_cdf", "plot_phase_memory", ...]`, it produces a cross product:

```
scripts/analysis/outputs/smoke-003/.../top1/plot_error_cdf.png
scripts/analysis/outputs/smoke-003/.../top1/plot_phase_memory.png
scripts/analysis/outputs/smoke-003/.../top1/plot_phase_runtime.png
…
```

Each of those paths is resolved by Snakemake walking back through the rules.

## The DAG

For smoke-003, `rule all` requests 5 PNGs. Snakemake works backward:

```
plot_error_cdf.png       needs summary.parquet  +  eval_observations.parquet
plot_error_cdf_for_*     needs summary.parquet  +  eval_observations.parquet
plot_phase_memory.png    needs summary.parquet
plot_phase_runtime.png   needs summary.parquet
plot_error_diff_cdf.png  needs summary.parquet
```

All five depend on the same `summary.parquet` (which already exists), so the rules are independent of each other. With `-j 4`, Snakemake runs four at once.

The benchmark Snakefile chains further:

```
run_combo[*]   needs materialize         → materialize needs inputs/ manifest
summarize      needs run_combo[*]
```

When you launch `snakemake -j 4`, Snakemake runs `materialize` first (single dep), fans out N `run_combo` jobs four at a time, and runs `summarize` once all combos finish. **That parallelism is free** — you didn't write it; the dependency graph implies it.

## Idempotency = "if the file exists and is newer than inputs, the rule is satisfied"

Snakemake checks output mtimes against input mtimes. If the output is newer than every input, the rule is **skipped**. That's why re-running `snakemake` is fast — only stale or missing outputs get rebuilt. `--forceall` or `--forcerun <rule>` overrides this.

That's also why our full-us-002 run resumed at "10 new combos" — the 4 pre-existing combo dirs already had `targets.parquet`, so Snakemake skipped them.

## Vocabulary cheat sheet

| Thing | What it means |
| --- | --- |
| `rule foo:` | A recipe: "I can produce these outputs from these inputs." |
| `input:` | Files this rule needs. Snakemake recursively builds them if missing. |
| `output:` | Files this rule produces. Pattern-matched by the requestor. |
| `params:` | Extra values for the shell command — not files, just strings. |
| `shell:` | The actual command. `{input.x}`, `{output.x}`, `{params.x}`, `{wildcards.x}` are templated in. |
| `{slice}` | A wildcard — captured from the requested output path. |
| `expand(pattern, x=...)` | Cartesian-product helper for building lists of paths. |
| `rule all:` | The conventional entry rule — `input:` lists everything you want built. |
| `-j N` | Run up to N jobs in parallel. Snakemake never schedules a job whose inputs aren't ready. |
| `-n` | Dry-run — print the DAG without executing. Great for sanity checks. |

## How our two Snakefiles compose

```
benchmark/v2/Snakefile           analysis/Snakefile
├── materialize  (1 job)  ┐
├── run_combo  (N jobs)   │ ─► summary.parquet ─► error_cdf, phase_memory, …
└── summarize  (1 job)    ┘                       (5 plots per slice)
```

The handoff is `outputs/<run_id>/summary.parquet`. The benchmark workflow writes it; the analysis workflow consumes it. That's why our analysis configs always reference `run_id` — that's the path coordinate tying the two workflows together.

## Quick read-it-yourself exercise

Open [scripts/benchmark/v2/Snakefile](../scripts/benchmark/v2/Snakefile) and try to trace:

1. What does `rule all` ask for?
2. Where does `{combo_id}` come from? (Hint: it's a wildcard from `output:` matched back to the config's `combos:` list via `expand()`.)
3. If you change `--configfile` to point at a config with 8 combos instead of 4, how many `run_combo` jobs run, and what triggers that count?

Answer to (3): "8, because `expand(combo_id=COMBO_IDS)` builds 8 output paths in `rule all`'s input, so the DAG has 8 leaves of that rule." That's the whole story — config drives `expand()`, `expand()` builds the request list, Snakemake fills in the wildcards and runs the rules backward.
