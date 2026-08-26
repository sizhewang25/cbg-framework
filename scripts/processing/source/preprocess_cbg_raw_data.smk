"""Preprocess raw CBG measurement CSVs: mainland filter → SOI sanitization.

Two-stage pipeline:
  1. filter_mainland_and_min_pair_observations  – drop non-mainland-US rows,
     require each target is seen by >= MIN_VP_OBS distinct VPs.
  2. sanitize_target_ground_truth_soi           – iterative speed-of-Internet
     pruning; removes targets / VP pairs whose RTT implies an impossible distance.

Usage (from repo root):
  snakemake -s scripts/processing/source/preprocess_cbg_raw_data.smk \\
      -j 1

Override defaults via --config:
  raw          : path to the raw input CSV
  out_dir      : output directory          (default: datasets/final)
  min_vp_obs   : min distinct VPs/target   (default: 3)
  obs_mode     : target-vps | pair-rows    (default: target-vps)
  vp_col       : VP id column              (default: VP_ID)
  target_col   : target id column          (default: TARGET_ID)
  eps_km       : SOI epsilon (km)          (default: 1e-9)
  soi_threshold: max violation fraction before full-target removal (default: 0.05)
"""

from pathlib import Path

# ── configurable inputs ───────────────────────────────────────────────────────
RAW        = Path(config["raw"])  # required: raw input CSV
_stem      = RAW.stem             # dataset name; drives every derived path
OUT_DIR    = Path(config.get("out_dir",      "datasets/final"))
MIN_VP_OBS = int(config.get("min_vp_obs",    3))
OBS_MODE   = config.get("obs_mode",          "target-vps")
VP_COL     = config.get("vp_col",            "VP_ID")
TARGET_COL = config.get("target_col",        "TARGET_ID")
EPS_KM     = float(config.get("eps_km",      1e-9))
SOI_THRESH = float(config.get("soi_threshold", 0.05))

# ── derive intermediate / final paths ────────────────────────────────────────
OUTPUTS_DIR = Path("scripts/processing/source/outputs") / _stem  # per-dataset intermediate dir
MAINLAND    = OUTPUTS_DIR / f"{_stem}.mainland.csv"
SANITIZED   = OUT_DIR     / f"{_stem}.mainland.sanitized.csv"    # final output only
# SOI side-outputs: script writes them next to --output, then the rule moves them
PAIRS_CSV    = OUTPUTS_DIR / f"{_stem}.mainland.sanitized.pairs.csv"
OUTLIERS_CSV = OUTPUTS_DIR / f"{_stem}.mainland.sanitized.outliers.csv"
SUMMARY_JSON = OUTPUTS_DIR / f"{_stem}.mainland.sanitized.summary.json"

# ── rules ─────────────────────────────────────────────────────────────────────

rule all:
    input: SANITIZED, PAIRS_CSV, OUTLIERS_CSV, SUMMARY_JSON


rule filter_mainland:
    """Drop non-mainland-US rows and targets with too few observing VPs."""
    input:
        csv = str(RAW),
    output:
        csv = str(MAINLAND),
    params:
        min_obs    = MIN_VP_OBS,
        obs_mode   = OBS_MODE,
        vp_col     = VP_COL,
        target_col = TARGET_COL,
    shell:
        """
        .venv/bin/python -m scripts.processing.source.filter_mainland_and_min_pair_observations \
            --input {input.csv} \
            --output {output.csv} \
            --min-observations {params.min_obs} \
            --observation-mode {params.obs_mode} \
            --vp-col {params.vp_col} \
            --target-col {params.target_col}
        """


rule soi_sanitize:
    """Remove targets / VP pairs that violate the speed-of-Internet constraint."""
    input:
        csv = str(MAINLAND),
    output:
        csv     = str(SANITIZED),
        pairs   = str(PAIRS_CSV),
        outliers = str(OUTLIERS_CSV),
        summary = str(SUMMARY_JSON),
    params:
        eps_km      = EPS_KM,
        threshold   = SOI_THRESH,
        outputs_dir = str(OUTPUTS_DIR),
        # side-output paths written by the script next to --output
        _pairs   = str(SANITIZED.parent / (SANITIZED.stem + ".pairs.csv")),
        _outliers = str(SANITIZED.parent / (SANITIZED.stem + ".outliers.csv")),
        _summary = str(SANITIZED.parent / (SANITIZED.stem + ".summary.json")),
    shell:
        """
        mkdir -p {params.outputs_dir}
        .venv/bin/python -m scripts.processing.source.sanitize_target_ground_truth_soi \
            --input {input.csv} \
            --output {output.csv} \
            --epsilon-km {params.eps_km} \
            --target-violation-fraction-for-removal {params.threshold}
        mv {params._pairs}   {output.pairs}
        mv {params._outliers} {output.outliers}
        mv {params._summary} {output.summary}
        """
