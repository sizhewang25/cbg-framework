"""Export dataset-properties CSV to a LaTeX table.

Input CSV columns (required):
    dataset, # VP, # ASN, # Target, # Target Cluster, % Colocated VPs

Output LaTeX layout:
- Rows: datasets
- Columns: Dataset, # VP, # ASN, # Target, # Target Cluster, % Colocated VPs

Example:
    /home/sw6456/geomodel/cbg-framework/.venv/bin/python \
      -m scripts.paper.datasets.export_dataset_properties_table \
      --in-csv scripts/paper/datasets/outputs/dataset_properties.csv \
      --out-tex scripts/paper/datasets/outputs/dataset_properties.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "dataset",
    "# VP",
    "# ASN",
    "# Target",
    "# Target Cluster",
    "% Colocated VPs",
}


def _validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        want = ", ".join(sorted(REQUIRED_COLUMNS))
        miss = ", ".join(sorted(missing))
        raise ValueError(f"{csv_path} is missing required columns: {miss}. Expected: {want}")


def _latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def _format_num(value: object, decimals: int = 1) -> str:
    if pd.isna(value):
        return ""
    try:
        num = float(value)
    except Exception:
        return _latex_escape(str(value))

    if num.is_integer():
        return str(int(num))
    return f"{num:.{decimals}f}"


def export_latex(
    in_csv: Path,
    out_tex: Path,
    percent_decimals: int = 1,
) -> None:
    df = pd.read_csv(in_csv)
    _validate_columns(df, in_csv)

    cols = [
        "dataset",
        "# VP",
        "# ASN",
        "# Target",
        "# Target Cluster",
        "% Colocated VPs",
    ]
    table = df[cols].copy()

    body_lines: list[str] = []
    for _, row in table.iterrows():
        cells = [
            _latex_escape(row["dataset"]),
            _format_num(row["# VP"], decimals=0),
            _format_num(row["# ASN"], decimals=0),
            _format_num(row["# Target"], decimals=0),
            _format_num(row["# Target Cluster"], decimals=0),
            _format_num(row["% Colocated VPs"], decimals=percent_decimals),
        ]
        body_lines.append(" & ".join(cells) + r" \\")

    header = "Dataset & \\# VP & \\# ASN & \\# Target & \\# Target Cluster & \\% Colocated VPs \\\\"
    latex = "\n".join(
        [
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            header,
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    compact_latex = "\n".join(
        [
            "% Requires \\usepackage{graphicx} in your LaTeX preamble.",
            "% Optional: \\usepackage{booktabs} for \\toprule/\\midrule/\\bottomrule.",
            "\\begingroup",
            "\\setlength{\\tabcolsep}{3pt}",
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
        help="Input dataset properties CSV path.",
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        required=True,
        help="Output LaTeX table path.",
    )
    parser.add_argument(
        "--percent-decimals",
        type=int,
        default=1,
        help="Number of decimal places for '% Colocated VPs'.",
    )
    args = parser.parse_args()

    export_latex(
        in_csv=args.in_csv,
        out_tex=args.out_tex,
        percent_decimals=args.percent_decimals,
    )
    print(f"Saved {args.out_tex}")


if __name__ == "__main__":
    main()
