"""Randomly sample N rows from an input CSV.

CLI::

    python -m scripts.processing.source.sample_rows \
        --input datasets/raw/20260417-20260504.csv \
        --n 1000 \
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_INPUT = Path("datasets/raw/20260417-20260504.csv")


def _default_output(input_path: Path, n: int) -> Path:
    return input_path.with_suffix("").with_suffix(f".sample{n}.csv")


def sample_rows(df: pd.DataFrame, n: int, seed: int | None) -> pd.DataFrame:
    if n < 0:
        raise ValueError("n must be >= 0")
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT,
                        help="Input CSV to sample from.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV. Defaults to <input>.sample<n>.csv.")
    parser.add_argument("--n", type=int, required=True,
                        help="Number of rows to sample. Clamped to the input row count.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible sampling. Default: unseeded.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    raw_df = pd.read_csv(args.input)
    sampled_df = sample_rows(raw_df, n=args.n, seed=args.seed)

    out_path = args.output or _default_output(args.input, args.n)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(out_path, index=False)

    n_raw = len(raw_df)
    n_sampled = len(sampled_df)
    logger.info("row sampling")
    logger.info("  rows   : %d / %d (%.2f%%)",
                n_sampled, n_raw, 100 * n_sampled / n_raw if n_raw else 0.0)
    logger.info("  seed   : %s", args.seed)
    logger.info("  wrote %s", out_path)


if __name__ == "__main__":
    main()
