"""Characterize datasets listed in a cross-datasets analysis config.

Given one cross-datasets YAML (for example
`scripts/analysis/config/cross_datasets/cross_dataset_comparison02.yaml`), this
script resolves each dataset config, loads:

- benchmark eval stats JSON (`outputs/<run_id>/eval_source/*_eval_stats.json`)
- source CSV (`source_kwargs.csv_path`)

and writes one aggregated CSV with per-dataset counts:

- number of VPs
- number of targets
- number of pairs
- number of target ASNs
- number of target clusters

Run from repo root:

    .venv/bin/python -m scripts.analysis.dataset_props.topology.characterize_datasets \
        --cross-config scripts/analysis/config/cross_datasets/cross_dataset_comparison02.yaml \
        --output tasks/dataset_characterization/aggregated.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from scripts.analysis.dataset_props.topology.plot_min_inflation_boxplots import (
    plot_min_inflation_boxplots,
)


DEFAULT_OUTPUTS_ROOT = Path(__file__).resolve().parents[3] / "benchmark" / "v2" / "outputs"


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _resolve_dataset_configs(cross_cfg_path: Path, cross_cfg: dict[str, Any]) -> list[Path]:
    datasets = cross_cfg.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"'datasets' must be a non-empty list in {cross_cfg_path}")
    resolved: list[Path] = []
    for i, item in enumerate(datasets):
        if not isinstance(item, str):
            raise ValueError(f"datasets[{i}] must be a string path")
        p = Path(item)
        if not p.is_absolute():
            p = (cross_cfg_path.parent / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Dataset config not found: {p}")
        resolved.append(p)
    return resolved


def _resolve_outputs_root(cross_cfg: dict[str, Any], cwd: Path) -> Path:
    raw = cross_cfg.get("v2_outputs_root")
    if raw is None:
        return DEFAULT_OUTPUTS_ROOT
    p = Path(str(raw))
    if not p.is_absolute():
        p = (cwd / p).resolve()
    return p


def _resolve_benchmark_config(
    dataset_cfg_path: Path,
    dataset_cfg: dict[str, Any],
    benchmark_config_override: Path | None,
) -> Path:
    if benchmark_config_override is not None:
        return benchmark_config_override

    bench_raw = dataset_cfg.get("benchmark_config")
    if isinstance(bench_raw, str) and bench_raw:
        p = Path(bench_raw)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"benchmark_config not found: {p}")

    run_id = dataset_cfg.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"Missing/invalid run_id in {dataset_cfg_path}")

    guessed = (
        Path(__file__).resolve().parents[3]
        / "benchmark"
        / "v2"
        / "config"
        / f"{run_id}.yaml"
    )
    if guessed.exists():
        return guessed

    raise FileNotFoundError(
        f"Could not resolve benchmark config for {dataset_cfg_path}. "
        "Add 'benchmark_config:' in the dataset config or pass --benchmark-config-root."
    )


def _find_eval_stats(eval_source_dir: Path, expected_csv_path: str | None) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(eval_source_dir.glob("*_eval_stats.json"))
    if not candidates:
        raise FileNotFoundError(f"No *_eval_stats.json under {eval_source_dir}")

    parsed: list[tuple[Path, dict[str, Any]]] = []
    for p in candidates:
        try:
            parsed.append((p, json.loads(p.read_text())))
        except Exception as exc:
            raise ValueError(f"Failed to parse JSON: {p}: {exc}") from exc

    if expected_csv_path:
        expected_name = Path(expected_csv_path).name
        matched = [
            item for item in parsed
            if Path(str(item[1].get("csv", ""))).name == expected_name
        ]
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            # If multiple files claim the same source CSV, use the newest one.
            matched.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
            return matched[0]

    if len(parsed) == 1:
        return parsed[0]

    parsed.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
    return parsed[0]


def _count_unique_target_asns(csv_path: Path) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {csv_path}")

    header = pd.read_csv(csv_path, nrows=0)
    cols = list(header.columns)
    asn_col = None
    for candidate in ("target_asn", "target_as", "dst_asn", "asn"):
        if candidate in cols:
            asn_col = candidate
            break
    if asn_col is None:
        return 0

    series = pd.read_csv(csv_path, usecols=[asn_col])[asn_col]
    return int(series.dropna().nunique())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric_summary(stats: dict[str, Any], metric_name: str) -> dict[str, float]:
    metrics = stats.get("metrics", {})
    metric = metrics.get(metric_name, {}) if isinstance(metrics, dict) else {}
    percentiles = metric.get("percentiles", {}) if isinstance(metric, dict) else {}
    if not isinstance(percentiles, dict):
        percentiles = {}
    return {
        "n": _safe_float(metric.get("n")),
        "min": _safe_float(metric.get("min")),
        "mean": _safe_float(metric.get("mean")),
        "p5": _safe_float(percentiles.get("p5")),
        "p25": _safe_float(percentiles.get("p25")),
        "p50": _safe_float(percentiles.get("p50")),
        "p75": _safe_float(percentiles.get("p75")),
        "p95": _safe_float(percentiles.get("p95")),
        "max": _safe_float(metric.get("max")),
    }


def _latex_escape(text: str) -> str:
    # Keep escaping minimal for dataset labels used in this project.
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def _base_dataset_name(name: str) -> str:
    suffix = " (Weighted)"
    if name.endswith(suffix):
        return name[:-len(suffix)]
    return name


def _format_with_reduction(weighted_value: int, original_value: int) -> str:
    if original_value <= 0:
        return str(weighted_value)
    reduction_pct = round((original_value - weighted_value) * 100.0 / original_value)
    return f"{weighted_value} ({int(reduction_pct)}\\%)"


def build_latex_table_from_csv(csv_path: Path, latex_output: Path) -> Path:
    """Build a grouped LaTeX table from the aggregated dataset-characterization CSV.

    Weighted rows are placed after their original dataset row and show reduction
    percentages (integer, no decimals) versus the original in parentheses.
    """
    df = pd.read_csv(csv_path)
    required_cols = {
        "dataset",
        "n_vps",
        "n_target_asns",
        "n_targets",
        "n_target_clusters",
        "n_pairs",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {', '.join(missing)}")

    # Preserve dataset order from the CSV while grouping weighted runs with base runs.
    seen_bases: set[str] = set()
    ordered_bases: list[str] = []
    for name in df["dataset"].astype(str):
        base = _base_dataset_name(name)
        if base not in seen_bases:
            seen_bases.add(base)
            ordered_bases.append(base)

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Dataset characterization. Weighted rows show reduction vs. original in parentheses.}")
    lines.append(r"\label{tab:dataset-characterization}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & \# VP & \# TG ASN & \# TG & \# TG Cluster & \# Pairs \\")
    lines.append(r"\midrule")

    for i, base in enumerate(ordered_bases):
        base_rows = df[df["dataset"].astype(str) == base]
        weighted_rows = df[df["dataset"].astype(str) == f"{base} (Weighted)"]

        if base_rows.empty:
            # If only a weighted row exists, render it as-is.
            if not weighted_rows.empty:
                wr = weighted_rows.iloc[0]
                wname = _latex_escape(str(wr["dataset"]))
                lines.append(
                    f"{wname} & {int(wr['n_vps'])} & {int(wr['n_target_asns'])} & "
                    f"{int(wr['n_targets'])} & {int(wr['n_target_clusters'])} & {int(wr['n_pairs'])} \\\\")
        else:
            br = base_rows.iloc[0]
            bname = _latex_escape(str(br["dataset"]))
            lines.append(
                f"{bname} & {int(br['n_vps'])} & {int(br['n_target_asns'])} & "
                f"{int(br['n_targets'])} & {int(br['n_target_clusters'])} & {int(br['n_pairs'])} \\\\")

            if not weighted_rows.empty:
                wr = weighted_rows.iloc[0]
                wname = _latex_escape(str(wr["dataset"]))
                lines.append(
                    f"{wname} & "
                    f"{_format_with_reduction(int(wr['n_vps']), int(br['n_vps']))} & "
                    f"{_format_with_reduction(int(wr['n_target_asns']), int(br['n_target_asns']))} & "
                    f"{_format_with_reduction(int(wr['n_targets']), int(br['n_targets']))} & "
                    f"{_format_with_reduction(int(wr['n_target_clusters']), int(br['n_target_clusters']))} & "
                    f"{_format_with_reduction(int(wr['n_pairs']), int(br['n_pairs']))} \\\\")

        if i != len(ordered_bases) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_output.parent.mkdir(parents=True, exist_ok=True)
    latex_output.write_text("\n".join(lines) + "\n")
    return latex_output


def characterize(
    cross_config: Path,
    output: Path,
    benchmark_config_root: Path | None = None,
) -> pd.DataFrame:
    cross_config = cross_config.resolve()
    cross_cfg = _load_yaml(cross_config)
    dataset_cfg_paths = _resolve_dataset_configs(cross_config, cross_cfg)

    labels = cross_cfg.get("labels")
    if labels is not None and (not isinstance(labels, list) or len(labels) != len(dataset_cfg_paths)):
        raise ValueError("'labels' must be a list with the same length as 'datasets'")

    outputs_root = _resolve_outputs_root(cross_cfg, Path.cwd())
    rows: list[dict[str, Any]] = []

    for idx, cfg_path in enumerate(dataset_cfg_paths):
        cfg = _load_yaml(cfg_path)

        run_id = cfg.get("run_id")
        source = cfg.get("source")
        setup = cfg.get("setup")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"Missing/invalid run_id in {cfg_path}")

        benchmark_cfg_override = None
        if benchmark_config_root is not None:
            benchmark_cfg_override = (benchmark_config_root / f"{run_id}.yaml").resolve()
            if not benchmark_cfg_override.exists():
                raise FileNotFoundError(f"Benchmark config not found: {benchmark_cfg_override}")

        benchmark_cfg_path = _resolve_benchmark_config(cfg_path, cfg, benchmark_cfg_override)
        benchmark_cfg = _load_yaml(benchmark_cfg_path)

        source_kwargs = benchmark_cfg.get("source_kwargs", {})
        if not isinstance(source_kwargs, dict):
            raise ValueError(f"source_kwargs must be a mapping in {benchmark_cfg_path}")
        csv_raw = source_kwargs.get("csv_path")
        if not isinstance(csv_raw, str) or not csv_raw:
            raise ValueError(f"Missing source_kwargs.csv_path in {benchmark_cfg_path}")

        csv_path = Path(csv_raw)
        if not csv_path.is_absolute():
            csv_path = (Path.cwd() / csv_path).resolve()

        eval_source_dir = outputs_root / run_id / "eval_source"
        eval_stats_path, stats = _find_eval_stats(eval_source_dir, expected_csv_path=csv_raw)

        clusters = stats.get("target_clustering", {})
        n_target_clusters = int(clusters.get("n_clusters", 0)) if isinstance(clusters, dict) else 0
        proximity = stats.get("proximity", {})
        rtt_quality = stats.get("rtt_quality", {})
        min_inflation = _metric_summary(stats, "min_inflation")
        cell_gap_km = _metric_summary(stats, "cell_gap_km")

        rows.append(
            {
                "dataset": labels[idx] if isinstance(labels, list) else cfg_path.stem,
                "dataset_config": str(cfg_path),
                "benchmark_config": str(benchmark_cfg_path),
                "run_id": run_id,
                "source": source,
                "setup": setup,
                "source_csv": str(csv_path),
                "eval_stats_json": str(eval_stats_path),
                "n_vps": int(stats.get("n_vps", 0)),
                "n_targets": int(stats.get("n_targets", 0)),
                "n_pairs": int(stats.get("n_pairs", 0)),
                "n_target_asns": _count_unique_target_asns(csv_path),
                "n_target_clusters": n_target_clusters,
                "has_used_proximity_share": _safe_float(
                    proximity.get("has_used_proximity_share") if isinstance(proximity, dict) else None
                ),
                "has_not_used_proximity_share": _safe_float(
                    proximity.get("has_not_used_proximity_share") if isinstance(proximity, dict) else None
                ),
                "no_proximity_share": _safe_float(
                    proximity.get("no_proximity_share") if isinstance(proximity, dict) else None
                ),
                "closest_is_shortest_ping_share": _safe_float(
                    rtt_quality.get("closest_is_shortest_ping_share") if isinstance(rtt_quality, dict) else None
                ),
                "cbg_opportunity_share": _safe_float(
                    proximity.get("cbg_opportunity_share") if isinstance(proximity, dict) else None
                ),
                "min_inflation_n": int(min_inflation["n"]),
                "min_inflation_min": min_inflation["min"],
                "min_inflation_mean": min_inflation["mean"],
                "min_inflation_p5": min_inflation["p5"],
                "min_inflation_p25": min_inflation["p25"],
                "min_inflation_p50": min_inflation["p50"],
                "min_inflation_p75": min_inflation["p75"],
                "min_inflation_p95": min_inflation["p95"],
                "min_inflation_max": min_inflation["max"],
                "cell_gap_km_mean": cell_gap_km["mean"],
            }
        )

    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cross-config",
        type=Path,
        required=True,
        help="Cross-datasets config YAML with a 'datasets' list.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/analysis/outputs/cross_dataset_comparison02/dataset_props/dataset_characterization_aggregated.csv"),
        help="Output aggregated CSV path.",
    )
    parser.add_argument(
        "--benchmark-config-root",
        type=Path,
        default=None,
        help=(
            "Optional folder containing <run_id>.yaml benchmark configs. "
            "Useful when dataset configs are analysis configs without source_kwargs."
        ),
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=Path("scripts/analysis/outputs/cross_dataset_comparison02/dataset_props/dataset_characterization_table.tex"),
        help="Output LaTeX table path built from the aggregated CSV.",
    )
    parser.add_argument(
        "--plots-out-dir",
        type=Path,
        default=Path("scripts/analysis/outputs/cross_dataset_comparison02/dataset_props/plots"),
        help="Output directory for generated plots, including the capped min-inflation boxplot.",
    )
    args = parser.parse_args()

    bcr = args.benchmark_config_root.resolve() if args.benchmark_config_root is not None else None
    df = characterize(args.cross_config, args.output, benchmark_config_root=bcr)
    print(f"Wrote {len(df)} rows to {args.output}")
    tex_path = build_latex_table_from_csv(args.output, args.latex_output)
    print(f"Wrote LaTeX table to {tex_path}")
    png_path, pdf_path, csv_path = plot_min_inflation_boxplots(args.output, args.plots_out_dir)
    print(f"Wrote min_inflation boxplot to {png_path}")
    print(f"Wrote min_inflation boxplot to {pdf_path}")
    print(f"Wrote min_inflation boxplot values to {csv_path}")


if __name__ == "__main__":
    main()
