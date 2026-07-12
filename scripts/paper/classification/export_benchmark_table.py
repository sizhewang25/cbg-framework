"""Export top-k classification benchmark CSV to a LaTeX table.

Input CSV columns (required):
	run_id, model, top1_cls_acc, top3_cls_acc

Output LaTeX layout:
- Rows: datasets (run_id)
- Columns: model groups, each with subcolumns "Top 1" and "Top 3"

Example:
	/home/sw6456/geomodel/cbg-framework/.venv/bin/python \
	  -m scripts.paper.classification.export_benchmark_table \
	  --in-csv scripts/paper/classification/outputs/topk_classification_benchmark.csv \
	  --out-tex scripts/paper/classification/outputs/topk_classification_benchmark.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"run_id", "model", "top1_cls_acc", "top3_cls_acc"}


def _validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
	missing = REQUIRED_COLUMNS.difference(df.columns)
	if missing:
		want = ", ".join(sorted(REQUIRED_COLUMNS))
		miss = ", ".join(sorted(missing))
		raise ValueError(f"{csv_path} is missing required columns: {miss}. Expected: {want}")


def _alias_run_id(run_id: str) -> str:
	name = str(run_id)
	weighted = False

	if name.endswith("-weighted-final"):
		name = name[: -len("-weighted-final")]
		weighted = True
	elif name.endswith("-final"):
		name = name[: -len("-final")]

	if "-" in name:
		name = name.split("-", 1)[1]

	return f"{name}-wgt" if weighted else name


def _latex_escape(text: str) -> str:
	# Minimal escaping for labels used in table headers/index.
	return (
		str(text)
		.replace("\\", "\\textbackslash{}")
		.replace("_", "\\_")
		.replace("%", "\\%")
		.replace("&", "\\&")
	)


def _build_wide_table(df: pd.DataFrame) -> pd.DataFrame:
	run_order = list(dict.fromkeys(df["run_id"].astype(str).tolist()))
	model_order = list(dict.fromkeys(df["model"].astype(str).tolist()))
	if "Shortest Ping" in model_order:
		model_order = ["Shortest Ping"] + [m for m in model_order if m != "Shortest Ping"]

	# Keep first occurrence if duplicates appear for the same (run_id, model).
	dedup = (
		df[["run_id", "model", "top1_cls_acc", "top3_cls_acc"]]
		.drop_duplicates(subset=["run_id", "model"], keep="first")
		.copy()
	)
	dedup["run_id"] = dedup["run_id"].astype(str)
	dedup["model"] = dedup["model"].astype(str)

	top1 = dedup.pivot(index="run_id", columns="model", values="top1_cls_acc")
	top3 = dedup.pivot(index="run_id", columns="model", values="top3_cls_acc")

	top1 = top1.reindex(index=run_order, columns=model_order)
	top3 = top3.reindex(index=run_order, columns=model_order)

	data: dict[tuple[str, str], pd.Series] = {}
	for model in model_order:
		data[(model, "Top 1")] = top1[model]
		data[(model, "Top 3")] = top3[model]

	out = pd.DataFrame(data, index=run_order)
	out.index = [_alias_run_id(run_id) for run_id in out.index]
	out.index.name = "Dataset"
	out.columns = pd.MultiIndex.from_tuples(out.columns)
	return out


def export_latex(
	in_csv: Path,
	out_tex: Path,
	decimals: int = 0,
) -> None:
	df = pd.read_csv(in_csv)
	_validate_columns(df, in_csv)

	wide = _build_wide_table(df)

	n_models = len(wide.columns.get_level_values(0).unique())
	# Add vertical separators between model groups.
	colfmt = "c|" + "|".join(["cc"] * n_models)
	models = list(wide.columns.get_level_values(0).unique())

	first_header_cells = [""]
	for i, model in enumerate(models):
		spec = "c|" if i < len(models) - 1 else "c"
		first_header_cells.append(f"\\multicolumn{{2}}{{{spec}}}{{{_latex_escape(model)}}}")
	first_header = " & ".join(first_header_cells) + r" \\"

	cmidrules = " ".join(
		f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(models))
	)

	second_header = "Dataset" + "".join(" & Top 1 & Top 3" for _ in models) + r" \\"

	body_lines: list[str] = []
	for dataset, row in wide.iterrows():
		top1_vals = row.xs("Top 1", level=1)
		top3_vals = row.xs("Top 3", level=1)
		top1_max = top1_vals.max(skipna=True)
		top3_max = top3_vals.max(skipna=True)

		top1_winners: set[str] = set()
		top3_winners: set[str] = set()
		if pd.notna(top1_max):
			top1_winners = set(top1_vals[top1_vals == top1_max].index.tolist())
		if pd.notna(top3_max):
			top3_winners = set(top3_vals[top3_vals == top3_max].index.tolist())

		cells = [_latex_escape(dataset)]
		for col in wide.columns:
			model, metric = col
			val = row[col]
			if pd.isna(val):
				cells.append("")
			else:
				formatted = f"{val * 100:.{decimals}f}"
				is_top1_winner = metric == "Top 1" and model in top1_winners
				is_top3_winner = metric == "Top 3" and model in top3_winners
				if is_top1_winner or is_top3_winner:
					formatted = f"\\textbf{{{formatted}}}"
				cells.append(formatted)
		body_lines.append(" & ".join(cells) + r" \\")

	latex = "\n".join(
		[
			f"\\begin{{tabular}}{{{colfmt}}}",
			r"\toprule",
			first_header,
			cmidrules,
			second_header,
			r"\midrule",
			*body_lines,
			r"\bottomrule",
			r"\end{tabular}",
		]
	)
	compact_latex = "\n".join(
		[
			"% Requires \\usepackage{graphicx} in your LaTeX preamble.",
			"% Optional: \\usepackage{booktabs} for \\toprule/\\midrule/\\bottomrule/\\cmidrule.",
			"\\begingroup",
			"\\setlength{\\tabcolsep}{2.5pt}",
			"\\renewcommand{\\arraystretch}{0.95}",
			"\\scriptsize",
			"\\resizebox{\\columnwidth}{!}{%",
			latex.rstrip(),
			"}",
			"\\endgroup",
		]
	) + "\n"

	out_tex.parent.mkdir(parents=True, exist_ok=True)
	out_tex.write_text(compact_latex)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--in-csv",
		type=Path,
		required=True,
		help="Input benchmark CSV path.",
	)
	parser.add_argument(
		"--out-tex",
		type=Path,
		required=True,
		help="Output LaTeX table path.",
	)
	parser.add_argument(
		"--decimals",
		type=int,
		default=0,
		help="Number of decimal places for percentage values.",
	)
	args = parser.parse_args()

	export_latex(
		args.in_csv,
		args.out_tex,
		decimals=args.decimals,
	)
	print(f"Saved {args.out_tex}")


if __name__ == "__main__":
	main()
