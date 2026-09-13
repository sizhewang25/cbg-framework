# 02 — Accuracy plotting

Reproducing §8.1's accuracy table and outcome bars for the three operator
datasets, on a device that has a working environment from
[01 — Installation](01-installation.md).

End state: `outputs/analysis/v3/_cross/accuracy-table/as01+as02+as03/` holding
`accuracy_table.h3-4.{csv,md}` and `outcome_bars.h3-4.top1.png`.

---

## 0. The two inputs a clone does not carry

Both are gitignored, so the repository alone cannot produce the figure. Copy
them from a device that has them, or regenerate them with
`scripts.benchmark.v2.cli`.

| Input | Path | Size | Needed by |
|---|---|---|---|
| v2 benchmark run outputs | `outputs/benchmark/v2/<run_id>/` | 6.5–7.2 MB per run | every command below |
| Canonical measurement CSV | `datasets/final/<basename>.csv` | ~4 MB per run | `build-proximity` only |

```bash
rsync -a <source-host>:.../cbg-framework/outputs/benchmark/v2/as0{1,2,3}-260728-260802 \
         outputs/benchmark/v2/

rsync -a <source-host>:.../cbg-framework/datasets/final/as0{1,2,3}-*.mainland.sanitized.csv \
         datasets/final/
```

Within a run directory the parts actually read are `<source>/<setup>/{vps.csv,
targets.csv}`, `<source>/<setup>/fold_*/<combo>/{targets.parquet,run.json}` and
`eval_source/*_{eval_per_target.csv,eval_stats.json}`.

**The CSV path is not derived from `run_id`.** It is read from the `csv` key in
`outputs/benchmark/v2/<run_id>/eval_source/*_eval_stats.json` and resolved
relative to the repository root (`bipartite.resolve_source_csv`), with a fallback
that globs `datasets/**/<eval_basename>.csv`. Stage the CSV anywhere else and
you must pass `--source-csv` explicitly.

---

## 1. Per-run pipeline

```bash
export PATH="$PWD/.venv/bin:$PATH"

for r in as01-260728-260802 as02-260728-260802 as03-260728-260802; do
  python -m scripts.analysis.v3.cli build-answer-space --run-id $r --grid h3 -r 4
  python -m scripts.analysis.v3.cli classify           --run-id $r --grid h3 -r 4
  python -m scripts.analysis.v3.cli build-proximity    --run-id $r --grid h3 -r 4
done
```

| Step | Reads | Writes under `outputs/analysis/v3/<run_id>/` |
|---|---|---|
| `build-answer-space` | `targets.csv` | `target-answer-space/h3-4/` |
| `classify` | `fold_*/<combo>/targets.parquet`, `eval_source/` | `target-cls-accuracy/h3-4/topn_accuracy.csv` |
| `build-proximity` | `vps.csv`, `eval_source/`, canonical CSV | `target-proximity/h3-4/` |

Expected console output, which doubles as the check that the inputs arrived
intact:

```
as01: 399 targets -> K=18 seeds   ...   best top1=0.729 (octant_cbg_hull)
as02: 412 targets -> K=22 seeds   ...   best top1=0.650 (octant_cbg_hull)
as03: 458 targets -> K=22 seeds   ...   best top1=0.502 (octant_cbg_hull)
```

**Keep `--grid h3 -r 4` identical on every command,** including those in §2.
The grid and resolution form the `h3-4` slug that names both the output
directories and the final filenames; mixing rungs produces a
`MissingArtifactError` naming the command to re-run.

**If you only want the PNG, skip `build-proximity`.** The bars need `classify`
alone. It is `table-accuracy` that needs the proximity labels, for its
per-dataset context table.

---

## 2. The table and the figure

```bash
python -m scripts.analysis.v3.cli table-accuracy --grid h3 -r 4 \
  --run-id as01-260728-260802 --run-id as02-260728-260802 --run-id as03-260728-260802

python -m scripts.analysis.v3.cli plot-outcome-bars --grid h3 -r 4 --top-n 1 \
  --layout pooled --layout compare \
  --run-id as01-260728-260802 --run-id as02-260728-260802 --run-id as03-260728-260802
```

Or let the tracked cross-run config supply the parameters. `--config` belongs to
the CLI *group*, so it goes **before** the command name:

```bash
python -m scripts.analysis.v3.cli --config configs/cross-as01-as03.yaml table-accuracy
python -m scripts.analysis.v3.cli --config configs/cross-as01-as03.yaml plot-outcome-bars
```

`configs/cross-as01-as03.yaml` already pins the three runs, `grid: h3`,
`resolution: [4]`, `top_n: 1` and `layout: [pooled, compare]`. It is the only
unified config naming several runs; the per-run commands in §1 reject it by name
and want `configs/<run_id>.yaml` or `--all-runs`.

---

## 3. What you get

All in `outputs/analysis/v3/_cross/accuracy-table/as01+as02+as03/`:

| File | From |
|---|---|
| `accuracy_table.h3-4.csv` / `.md` | `table-accuracy` — one row per (run, method) |
| `dataset_context.h3-4.csv` | `table-accuracy` — per-run proximity shares |
| `outcome_bars.h3-4.top1.png` / `.csv` | `plot-outcome-bars --layout pooled` |
| `outcome_bars_by_dataset.h3-4.top1.png` / `.csv` | `plot-outcome-bars --layout compare` |
| `*.manifest.json` | provenance for each of the above |

Every figure has a CSV twin carrying the numbers it draws, and each `.md` is the
paste-ready form for the paper. The whole chain takes a few minutes on this
dataset; `classify` dominates.

---

## 4. Four things that surprise people

- **Neither cross-run command has `--all-runs`, by design.** §7.3 declines the
  operator/public head-to-head, so `as7018_us_test01` must be a separate
  invocation and lands in its own `_cross/accuracy-table/as7018_us_test01/`.
  Which runs share one table is the caller's decision, deliberately.
- **No pooled row in `table-accuracy`.** as01/02/03 are one VP fleet under
  different peering and routing conditions, and that difference *is* the
  comparison §8.1 rests on, so the command offers no averaging switch.
  `table-headline` is the one that pools, as a target-count micro-average.
- **`classify --topn` must include both 1 and 3.** The table prints both
  columns and errors out naming the missing Ns otherwise. The default `(1, 3)`
  already covers it — do not narrow it.
- **The hatched "traffic-weighted" bars are not your data.** No run carries
  traffic weights, and `PROVISIONAL_WEIGHTED` in `modules/headline_table.py` is
  a hard-coded placeholder from an earlier run — which is why those bars render
  filled rather than as the empty outlines the module docstring describes. Pass
  `--weighted-run-id <dataset>=<run_id>` once a weighted run exists.

---

## Reference

`scripts/analysis/v3/README.md` documents the full pipeline, every other command
(`plot-venn`, `plot-error-cdf`, `plot-pareto`, `plot-mtl-map`, …) and the
reasoning behind the answer space, the proximity diamond and the table layouts.
`scripts/analysis/v3/SCHEMA.md` documents the artifact columns.
