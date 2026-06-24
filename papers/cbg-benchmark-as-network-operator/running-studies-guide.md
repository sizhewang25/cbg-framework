# Running the Paper Studies on a New Dataset

**Audience:** Someone who wants to reproduce the §6.1–§6.5 results from the paper
on a different (e.g. proprietary operator) dataset. Assumes Python 3.11/3.12 +
`poetry install` is done.

---

## Mental model

Every study in the paper runs in two stages:

```
Stage A — Benchmark  (scripts/benchmark/v2/)
  Your CSV → [materialize-inputs] → inputs/ parquets
           → [Snakemake run]       → outputs/<run_id>/ (targets.parquet per fold × combo)

Stage B — Analysis   (scripts/analysis/partvp/)
  outputs/<run_id>/  → [run_textbook_config] → outputs_partvp/<run_id>/ (with mtl_participants)
                     → [extract_features]    → data/<run_id>.parquet
                     → [tolerance_dividend]  → analysis/tolerance_dividend.csv    (§6.2)
                     → [fleet_geometry]      → analysis_fleet/                    (§6.1/§6.2)
                     → [assess_vp_proximity] → analysis_fleet/                    (§6.1)
                     → [characterize_failures]→ analysis_fail/                    (§6.3)
                     → [region_confidence]   → analysis/region_confidence.csv     (§6.3 flag)
                     → [rq3_practicality]    → analysis_rq3/                      (§6.5)
```

Stage A produces raw per-target predictions. Stage B slices and interprets them.
The `outputs_partvp/` re-run in Stage B is needed because it stores
`mtl_participants` (the VPs that decided the intersection polygon), which
`extract_features.py` and `region_confidence.py` depend on.

---

## Step 0 — Prepare your data as a CSV

`generic_csv` is the universal entry point for any dataset. Produce a CSV with
these columns (one row per RTT observation):

| Column | Type | Meaning |
|---|---|---|
| `vp_id` | str | Unique ID for the vantage point |
| `vp_lat`, `vp_lon` | float | VP's ground-truth coordinates |
| `target_id` | str | Unique ID for the target IP/host |
| `target_lat`, `target_lon` | float | Target's ground-truth coordinates |
| `rtt_ms` | float (> 0) | Round-trip time in milliseconds |

Optional columns (`vp_asn`, `vp_country`, `target_asn`, `target_country`, …)
are useful for geo-slicing but not required.

**Role assignment**: `vp_*` columns are always the vantage points; `target_*`
are always the things being geolocated. For a proprietary operator dataset,
VPs = operator's infrastructure sites, targets = the peer ASN's IPs you want
to locate.

Generate a synthetic smoke CSV to test the pipeline without real data:
```bash
poetry run python -m scripts.benchmark.v2.sources._make_smoke_csv /tmp/smoke.csv
# wrote /tmp/smoke.csv (750 rows, 25 VPs × 30 targets)
```

---

## Step 1 — Create the benchmark config

Copy an existing final config as a starting point:
```bash
cp scripts/benchmark/v2/config/north_america_as7018_final_us.yaml \
   scripts/benchmark/v2/config/my_operator_final.yaml
```

Edit the top of the file:
```yaml
run_id: my_operator_final        # used as the output directory name — pick something descriptive
source: generic_csv
setup: probes_to_anchors         # VPs → targets direction
slices: [fold_0, fold_1, fold_2, fold_3, fold_4]  # 5-fold CV

source_kwargs:
  csv_path: datasets/my_operator/pings.csv   # your CSV path
  k: 5                                       # fold count; match slices above
  seed: 42

enable_fallback: true
seed: 42
```

Keep the `combos:` section as-is — it defines the four textbook variants plus
ablations. For a minimal run (just the four textbook combos), strip all but
`vanilla_cbg`, `million_scale_cbg`, `octant_cbg`, `spotter_cbg`.

**Important:** The `run_id` is used for all downstream analysis. Choose a stable
name; renaming later requires updating every analysis path.

---

## Step 2 — Materialize inputs

Inputs are the three parquets (`vp_configs`, `fit_samples`, `eval_observations`)
that all folds share for a given source. Materialize once per source/setup/slice:

```bash
for fold in fold_0 fold_1 fold_2 fold_3 fold_4; do
  poetry run python -m scripts.benchmark.v2.cli materialize-inputs \
      --source generic_csv \
      --setup probes_to_anchors \
      --slice $fold \
      --csv-path datasets/my_operator/pings.csv \
      --k 5 --seed 42
done
```

Outputs land in `scripts/benchmark/v2/inputs/generic_csv/my_operator/probes_to_anchors/fold_N/`.

Check that each fold's `eval_observations.parquet` is non-empty:
```bash
python -c "
import pyarrow.parquet as pq, pathlib
for p in pathlib.Path('scripts/benchmark/v2/inputs/generic_csv').rglob('eval_observations.parquet'):
    t = pq.read_table(p)
    print(p.parts[-3], len(t), 'targets')
"
```

---

## Step 3 — Run the benchmark (Stage A)

**Recommended: Snakemake** (parallelizes fold × combo grid):
```bash
poetry run snakemake -s scripts/benchmark/v2/Snakefile \
    --configfile scripts/benchmark/v2/config/my_operator_final.yaml \
    -j 8    # parallel jobs; use 4 for 16 GB RAM, 8–16 for 64 GB
```

This writes to `scripts/benchmark/v2/outputs/my_operator_final/`.

Verify completion — each fold × combo directory should have `targets.parquet`
and `run.json`:
```bash
find scripts/benchmark/v2/outputs/my_operator_final -name run.json \
  | sort | while read f; do
    echo "$f: $(python -c "import json; d=json.load(open('$f')); print(d.get('status','?'), d.get('n_targets','?'), 'targets')")"
done
```

---

## Step 4 — Create the analysis config

The analysis scripts use a YAML that points back at the benchmark run.
Copy from `scripts/analysis/config/`:
```bash
cp scripts/analysis/config/north_america_as7018_final_us.yaml \
   scripts/analysis/config/my_operator_final_us.yaml
```

Edit:
```yaml
run_id: my_operator_final
source: generic_csv
setup: probes_to_anchors
slices: [fold_0, fold_1, fold_2, fold_3, fold_4]

benchmark_config: scripts/benchmark/v2/config/my_operator_final.yaml
config_label: my-op-us     # short label for plots/CSVs
v2_outputs_root: outputs_partvp  # where run_textbook_config.py writes (Step 5)

textbook_combos:
  - vanilla_cbg
  - million_scale_cbg
  - octant_cbg
  - spotter_cbg

feature_parquet: scripts/analysis/outputs/partvp/data/my_operator_final.parquet

# Plotting params (for failure attribution figure)
cluster_radius_km: 50
n_clusters: 32    # update after running build_answer_space (Step 5)
```

---

## Step 5 — Re-run textbook combos with participant logging (outputs_partvp)

The Stage A Snakemake run does not store `mtl_participants` (the VPs that
contributed to the intersection polygon) because it's expensive. The analysis
scripts need it. Re-run just the four textbook combos through
`run_textbook_config.py`:

```bash
python -m scripts.analysis.partvp.run_textbook_config \
    --config scripts/analysis/config/my_operator_final_us.yaml
```

This writes to `scripts/benchmark/v2/outputs_partvp/my_operator_final/` with
`mtl_participants` populated in each `targets.parquet`.

Verify:
```bash
python -c "
import pyarrow.parquet as pq, pathlib
for p in pathlib.Path('scripts/benchmark/v2/outputs_partvp/my_operator_final').rglob('targets.parquet'):
    t = pq.read_table(p)
    has_part = 'mtl_participants' in t.schema.names
    print(p.parts[-3], p.parts[-2], len(t), 'rows, mtl_participants:', has_part)
" | sort
```

Expected: every fold × combo row shows `mtl_participants: True`.

---

## Step 6 — Extract per-target features

`extract_features.py` joins the benchmark outputs to the observation geometry
and builds the tidy feature table used by all downstream analysis scripts.

```bash
python -m scripts.analysis.partvp.extract_features \
    --run-dir scripts/benchmark/v2/outputs_partvp/my_operator_final \
    --out scripts/analysis/outputs/partvp/data/my_operator_final.parquet
```

The output is a single parquet with one row per `(combo_id, target_id)` carrying
features: `avail_min_vp_km`, `n_part`, `part_circ_var`, `part_min_infl`,
`nearest_other_centroid_km`, `within_r`, `match` (classification correct?), etc.

Check it:
```bash
python -c "
import pandas as pd
df = pd.read_parquet('scripts/analysis/outputs/partvp/data/my_operator_final.parquet')
print(df.groupby('combo_id')['match'].agg(['sum','count','mean']).round(3))
"
```

This is your first quick accuracy check — compare to the paper's §6.2 table.

---

## Step 7 — Tolerance dividend (§6.2 table)

```bash
python -m scripts.analysis.partvp.tolerance_dividend \
    --features scripts/analysis/outputs/partvp/data/my_operator_final.parquet \
    --out scripts/analysis/outputs/partvp/analysis/tolerance_dividend.csv
```

Reads: `within_r` (within 50 km of truth centroid) and `match` (same-centroid).
Outputs: `tolerance_dividend.csv` with `same_centroid_acc`, `within_r`,
`dividend_abs` (pp), `dividend_rel` (share of correct answers won by tolerance).

Interpretation: a large `dividend_rel` (>30%) means the bounded answer space is
doing real work — many correct answers would be invisible to a coordinate-error
metric.

---

## Step 8 — Fleet-geometry analysis (§6.1 accuracy table + AUC)

Two scripts work together. First, compute the per-target fleet geometry:

```bash
python -m scripts.analysis.partvp.fleet_geometry_explainability \
    --attribution scripts/analysis/outputs/partvp/analysis_fail/per_target_failures.parquet \
    --out-dir scripts/analysis/outputs/partvp/analysis_fleet
```

If `per_target_failures.parquet` doesn't exist yet, run `characterize_failures.py`
(Step 9) first, then come back. Alternatively, pass `--run-dir` pointing at
`outputs_partvp/my_operator_final` directly; the script will build the fleet
frame from scratch.

Then compute the proximity-failure coverage:

```bash
python -m scripts.analysis.partvp.assess_vp_proximity_failures \
    --fleet-frame scripts/analysis/outputs/partvp/analysis_fleet/fleet_geometry_per_target.parquet \
    --out-dir scripts/analysis/outputs/partvp/analysis_fleet
```

Key outputs:
- `fleet_geometry_by_config.csv`: `fleet_abs_km` distribution (median, 77th %-ile, etc.)
- `fleet_geometry_auc.csv`: AUC of `fleet_abs_km` predicting failure per variant
- `vp_proximity_failure_assessment_by_setup.csv`: % failures explained by missing VP
- `VP_PROXIMITY_FAILURE_ASSESSMENT.md`: narrative summary

For §6.1: the AUC and "% missing VP" numbers go in the proximity-limited regime
discussion. AUC 0.84–0.96 for Million-scale is the reference; if your fleet is
similarly sparse you should see comparable numbers.

---

## Step 9 — Failure characterization (§6.3 taxonomy + attribution)

```bash
python -m scripts.analysis.partvp.characterize_failures \
    --configs scripts/analysis/config/my_operator_final_us.yaml \
    --out-dir scripts/analysis/outputs/partvp/analysis_fail
```

The `--configs` flag (added in the current codebase) accepts one or more
analysis YAML files and overrides the hardcoded `CONFIGS` dict. Without it, the
script uses the hardcoded RIPE configs — always pass `--configs` for a new setup.

Key outputs:
- `failure_taxonomy.csv`: per-(config, variant) — EMPTY%, EXCLUSIVE%, INCL-miss%
- `failure_separation.csv`: per-feature AUC (match vs. fail separation power)
- `failure_attribution.png`: stacked failure attribution bars
- `WHEN_CBG_FAILS.md`: narrative with proxy-rule AUC and mechanism attribution

For §6.3: the `failure_taxonomy.csv` gives you the EMPTY/EXCLUSIVE/INCLUSIVE
partition (note: this is the proxy-rule version, not the Shapely polygon-disk
version; see §6.3 note on the two methods). `failure_separation.csv` gives the
feature AUCs: expect `fleet_abs_km` AUC ~0.84–0.96 and `nearest_other_centroid_km`
AUC ~0.64–0.68 for a typical setup.

---

## Step 10 — Region confidence flag (§6.3 L1 precision)

```bash
python -m scripts.analysis.partvp.region_confidence \
    --run-dir scripts/benchmark/v2/outputs_partvp/my_operator_final \
    --out-csv scripts/analysis/outputs/partvp/analysis/region_confidence.csv \
    --out-parquet scripts/analysis/outputs/partvp/data/region_confidence_my_operator_final.parquet
```

Key number: `l1_precision` (region overlaps exactly one answer cell → correct?)
Reference: 0.66–0.91 for RIPE setups, capturing 30–56% of correct answers.
A higher L1 precision means the setup has well-separated answer cells (the
bounded answer space has room to be unambiguous).

---

## Step 11 — Production cost (§6.5 throughput table)

```bash
python -m scripts.analysis.partvp.rq3_practicality \
    --run-dir scripts/benchmark/v2/outputs/my_operator_final \
    --features scripts/analysis/outputs/partvp/data/my_operator_final.parquet \
    --out-dir scripts/analysis/outputs/partvp/analysis_rq3
```

Note: pass `outputs/` (Stage A, not `outputs_partvp/`) here — you want the
production timing, not the participant-emitting rerun which is slower.

Key output: `rq3_runtime_my_operator_final.csv` with `ctr_ms_p50`, throughput
(targets/s), and classification accuracy per combo. Reference: geometric centroid
~0.25 ms (~65 targets/s), Monte Carlo ~200–390 ms (~3 targets/s), ~20× gap.

---

## Summary: full command sequence

```bash
RUN_ID=my_operator_final
CFG=scripts/benchmark/v2/config/${RUN_ID}.yaml
ACFG=scripts/analysis/config/${RUN_ID}_us.yaml

# Stage A: benchmark
for fold in fold_0 fold_1 fold_2 fold_3 fold_4; do
  poetry run python -m scripts.benchmark.v2.cli materialize-inputs \
      --source generic_csv --setup probes_to_anchors --slice $fold \
      --csv-path datasets/my_operator/pings.csv
done
poetry run snakemake -s scripts/benchmark/v2/Snakefile --configfile $CFG -j 8

# Stage B: analysis
python -m scripts.analysis.partvp.run_textbook_config --config $ACFG
python -m scripts.analysis.partvp.extract_features \
    --run-dir scripts/benchmark/v2/outputs_partvp/${RUN_ID} \
    --out scripts/analysis/outputs/partvp/data/${RUN_ID}.parquet
python -m scripts.analysis.partvp.tolerance_dividend \
    --features scripts/analysis/outputs/partvp/data/${RUN_ID}.parquet \
    --out scripts/analysis/outputs/partvp/analysis/tolerance_dividend.csv
python -m scripts.analysis.partvp.characterize_failures \
    --configs $ACFG \
    --out-dir scripts/analysis/outputs/partvp/analysis_fail
python -m scripts.analysis.partvp.fleet_geometry_explainability \
    --attribution scripts/analysis/outputs/partvp/analysis_fail/per_target_failures.parquet \
    --out-dir scripts/analysis/outputs/partvp/analysis_fleet
python -m scripts.analysis.partvp.assess_vp_proximity_failures \
    --fleet-frame scripts/analysis/outputs/partvp/analysis_fleet/fleet_geometry_per_target.parquet \
    --out-dir scripts/analysis/outputs/partvp/analysis_fleet
python -m scripts.analysis.partvp.region_confidence \
    --run-dir scripts/benchmark/v2/outputs_partvp/${RUN_ID} \
    --out-csv scripts/analysis/outputs/partvp/analysis/region_confidence.csv \
    --out-parquet scripts/analysis/outputs/partvp/data/region_confidence_${RUN_ID}.parquet
python -m scripts.analysis.partvp.rq3_practicality \
    --run-dir scripts/benchmark/v2/outputs/${RUN_ID} \
    --features scripts/analysis/outputs/partvp/data/${RUN_ID}.parquet \
    --out-dir scripts/analysis/outputs/partvp/analysis_rq3
```

---

## What maps to what in the paper

| Paper section | Key output file | Key numbers |
|---|---|---|
| §6.1 accuracy table | `outputs/my_operator_final/summary.parquet` | `same_centroid_acc` per combo × fold |
| §6.1 fleet geometry | `analysis_fleet/fleet_geometry_by_config.csv` | `fleet_abs_km` median, % missing VP |
| §6.1 VP proximity AUC | `analysis_fleet/fleet_geometry_auc.csv` | AUC of `fleet_abs_km` predicting failure |
| §6.1 % failures explained | `analysis_fleet/vp_proximity_failure_assessment_by_setup.csv` | `failure_share_explained_by_missing_vp` |
| §6.2 accuracy table | `data/my_operator_final.parquet` | `match.mean()` by combo |
| §6.2 tolerance dividend | `analysis/tolerance_dividend.csv` | `dividend_abs`, `dividend_rel` |
| §6.3 failure taxonomy | `analysis_fail/failure_taxonomy.csv` | EMPTY%, EXCLUSIVE%, INCL-miss% |
| §6.3 feature AUCs | `analysis_fail/failure_separation.csv` | AUC per feature × combo |
| §6.3 narrative | `analysis_fail/WHEN_CBG_FAILS.md` | mechanism attribution percentages |
| §6.3 confidence flag | `analysis/region_confidence.csv` | `l1_precision`, `l1_coverage` |
| §6.5 cost | `analysis_rq3/rq3_runtime_my_operator_final.csv` | `ctr_ms_p50`, throughput, accuracy |

---

## Common pitfalls

**Wrong `outputs_root` in run_textbook_config:** The analysis YAML's
`v2_outputs_root: outputs_partvp` tells `run_textbook_config.py` to write under
`scripts/benchmark/v2/outputs_partvp/`. If you see `extract_features.py` failing
to find `mtl_participants`, the run ended up in `outputs/` instead. Check the
YAML key.

**Missing target-distinguishing VP margin is negative for most targets:** This
is expected for a global fleet vs. global targets (§6.1 extremely-limited).
It means the fleet physically cannot distinguish most cells. Don't diagnose it
as a code bug — it's the §6.1 finding.

**Spotter accuracy collapses to ~0–7%:** Also expected, even with perfect VP
proximity. Spotter's `normal_dist` LTD produces large inner bounds that collapse
the annulus intersection. This is the §6.2 structural-collapse fingerprint, not
a data problem.

**Vanilla collapses in DE-scale proximity:** Low-envelope LTD under-predicts
at very short VP–target distances (< 10 km), producing a band that excludes the
truth. This is the §6.2 metro-scale breakage. If your operator VPs are co-located
with targets (< 5 km), expect Vanilla EMPTY rates of 30–40%.

**`characterize_failures.py` uses hardcoded CONFIGS by default:** Always pass
`--configs your_analysis.yaml`. The hardcoded dict references the RIPE setups
and will mismatch your `outputs_partvp/` paths.
