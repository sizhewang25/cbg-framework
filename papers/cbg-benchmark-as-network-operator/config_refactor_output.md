# Config Refactor Output

Date: 2026-06-24

## 1. Analysis Config Files Created

Eight new files added to `scripts/analysis/config/`:

| File | run_id | config_label | feature_parquet | v2_outputs_root |
|------|--------|--------------|-----------------|-----------------|
| `europe_as3209_final_de.yaml` | `europe_as3209_final_de` | `eu-de` | `data/europe_as3209_final_de.parquet` | `outputs_partvp` |
| `europe_as3209_final_eu.yaml` | `europe_as3209_final_eu` | `eu-eu` | `data/europe_as3209_final_eu.parquet` | `outputs_partvp` |
| `europe_as3215_final_eu.yaml` | `europe_as3215_final_eu` | `eu-eu-as3215` | `data/europe_as3215_final_eu.parquet` | `outputs_partvp` |
| `europe_as3215_final_fr.yaml` | `europe_as3215_final_fr` | `europe-country` | `data/europe_as3215_final_fr.parquet` | `outputs_partvp` |
| `north_america_as7018_final_na.yaml` | `north_america_as7018_final_na` | `na-na` | `data/north_america_as7018_final_na.parquet` | `outputs_partvp` |
| `north_america_as7018_final_us.yaml` | `north_america_as7018_final_us` | `na-us` | `data/north_america_as7018_final_us.parquet` | `outputs_partvp` |
| `north_america_as7922_final_na.yaml` | `north_america_as7922_final_na` | `na-na-as7922` | `data/north_america_as7922_final_na.parquet` | `outputs` (not partvp) |
| `north_america_as7922_final_us.yaml` | `north_america_as7922_final_us` | `na-us-as7922` | `data/north_america_as7922_final_us.parquet` | `outputs_partvp` |

Each file contains:
- `run_id`, `source`, `setup`, `slices` copied from the corresponding benchmark config
- `benchmark_config: scripts/benchmark/v2/config/<name>.yaml` pointer
- `config_label`, `textbook_combos`, `feature_parquet`, `v2_outputs_root`
- Plotting params: `merge_folds: true`, `group_by: ltd`, `phase_stat: p95`, `include_fit: true`, `runtime_stat: p50`

Note: `north_america_as7922_final_na` outputs live in `scripts/benchmark/v2/outputs/` (not `outputs_partvp/`) because this run was not part of the partvp pipeline; its `v2_outputs_root` is set accordingly. No feature parquet exists yet for the three configs whose parquets are missing (`europe_as3209_final_eu`, `europe_as3215_final_eu`, `north_america_as7922_final_na`).

## 2. Change Made to `characterize_failures.py`

### Added: `import yaml` and `from typing import Any`

At the top of the import block (after `from __future__ import annotations`).

### Added: `_load_configs_from_yamls()` helper (lines 329–365)

```python
def _load_configs_from_yamls(yaml_paths: list[Path]) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Load CONFIGS dict and TEXTBOOK list from a list of analysis config YAML paths."""
    configs: dict[str, tuple[str, str]] = {}
    textbook_sets: list[set[str]] = []
    for p in yaml_paths:
        with open(p) as f:
            cfg: dict[str, Any] = yaml.safe_load(f)
        label = cfg["config_label"]
        feat = cfg["feature_parquet"]
        run_id = cfg["run_id"]
        root = cfg.get("v2_outputs_root", "scripts/benchmark/v2/outputs")
        run_dir = str(Path(root) / run_id)
        configs[label] = (feat, run_dir)
        tc = cfg.get("textbook_combos")
        if tc:
            textbook_sets.append(set(tc))
    # intersection of textbook_combos across all configs, order from first YAML
    if textbook_sets:
        common = textbook_sets[0]
        for s in textbook_sets[1:]:
            common = common & s
        first_tc = list(yaml.safe_load(open(yaml_paths[0]))["textbook_combos"])
        textbook = [c for c in first_tc if c in common]
    else:
        textbook = list(TEXTBOOK)
    return configs, textbook
```

### Added: `--configs` argument to `main()` argparse

```python
ap.add_argument(
    "--configs", nargs="+", type=Path, default=None, metavar="YAML",
    help=(
        "List of analysis config YAML paths (scripts/analysis/config/<name>.yaml). "
        "Each YAML must contain config_label, feature_parquet, run_id, and optionally "
        "v2_outputs_root (default: scripts/benchmark/v2/outputs) and textbook_combos. "
        "When given, replaces the hardcoded CONFIGS dict and TEXTBOOK list. "
        "Without --configs the built-in CONFIGS dict is used (backward-compatible)."
    ),
)
```

### Added: config override logic in `main()`

```python
global CONFIGS, TEXTBOOK
if args.configs:
    CONFIGS, TEXTBOOK = _load_configs_from_yamls(args.configs)
    logger.info("Loaded %d configs from --configs: %s", len(CONFIGS), list(CONFIGS))
```

**Usage example:**
```bash
python -m scripts.analysis.partvp.characterize_failures \
    --configs scripts/analysis/config/north_america_as7018_final_us.yaml \
             scripts/analysis/config/europe_as3209_final_de.yaml \
    --out-dir scripts/analysis/partvp/outputs/analysis_fail_new
```

The hardcoded `CONFIGS` dict is preserved unchanged for backward compatibility when `--configs` is not given.

## 3. Status of `outputs_partvp` Re-Run

Both re-runs were executed via `run_textbook_config.py` with the `cfg_textbook/` YAMLs (which specify `planar_circle + geometric_centroid` for vanilla/million_scale and output to `outputs_partvp/`).

**`north_america_as7018_final_us`**: Completed successfully. All 5 folds × 4 combos re-run.

**`europe_as3209_final_de`**: Two invocations were launched (one at 00:46, one at 00:57 UTC). The EU VP corpus is significantly larger than NA-US — each fold took ~15 min vs ~2 min for NA-US. Both processes ran in parallel across folds; all 5 folds × 4 combos completed successfully.

## 4. Verification: `run.json` shows `planar_circle` for `vanilla_cbg`?

### `north_america_as7018_final_us` — **Y (all 5 folds confirmed)**

```
fold_0: mtl=planar_circle  ctr=geometric_centroid
fold_1: mtl=planar_circle  ctr=geometric_centroid
fold_2: mtl=planar_circle  ctr=geometric_centroid
fold_3: mtl=planar_circle  ctr=geometric_centroid
fold_4: mtl=planar_circle  ctr=geometric_centroid
```

Same result for `million_scale_cbg`: all 5 folds show `mtl=planar_circle`.

### `europe_as3209_final_de` — **Y (all 5 folds confirmed)**

```
fold_0: mtl=planar_circle  ctr=geometric_centroid
fold_1: mtl=planar_circle  ctr=geometric_centroid
fold_2: mtl=planar_circle  ctr=geometric_centroid
fold_3: mtl=planar_circle  ctr=geometric_centroid
fold_4: mtl=planar_circle  ctr=geometric_centroid
```

Same result for `million_scale_cbg`: all 5 folds show `mtl=planar_circle`.

The stale `spherical_circle + boundary_vertex_mean` config has been fully replaced in both run directories.
