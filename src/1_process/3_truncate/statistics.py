"""Compute coverage/frequency statistics for aggregated word-frequency files."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

AGGREGATED_DIR = Path("data/1_aggregated")
OUTPUT_CSV = Path("results/1_process/3_truncate/statistics.csv")

COVERAGE_LEVELS: list[float] = [
    50, 55, 60, 65, 70,                                      # step 5
    72, 74, 76, 78, 80,                                      # step 2
    81, 82, 83, 84, 85, 86, 87, 88, 89, 90,                  # step 1
    90.5, 91, 91.5, 92, 92.5, 93, 93.5, 94, 94.5, 95,       # step 0.5
    95.5, 96, 96.5, 97, 97.5, 98, 98.5, 99, 99.5, 100,      # step 0.5
]


def _cov_column(pct: float) -> str:
    """Column name for a coverage percentage, e.g. 90.5 → 'cov_90_5'."""
    if pct == int(pct):
        return f"cov_{int(pct)}"
    return f"cov_{str(pct).replace('.', '_')}"


def compute_file_stats(path: Path, coverage_levels: list[float] = COVERAGE_LEVELS) -> dict:
    """Return a flat dict of statistics for one aggregated CSV."""
    df = pd.read_csv(path, dtype={"word": str, "frequency": "int64"})
    freqs = df["frequency"].values
    total = int(freqs.sum())
    distinct = len(freqs)
    cumsum = np.cumsum(freqs)

    row: dict = {
        "file": path.stem,
        "distinct_words": distinct,
        "total_frequency": total,
    }

    # Coverage levels
    for pct in coverage_levels:
        threshold = total * pct / 100.0
        idx = min(int(np.searchsorted(cumsum, threshold, side="left")), distinct - 1)
        prefix = _cov_column(pct)
        row[f"{prefix}_types"] = idx + 1
        row[f"{prefix}_tokens"] = int(cumsum[idx])

    # Frequency cutoffs (freq >= N)
    for min_freq in (100, 10):
        count = int(np.searchsorted(-freqs, -min_freq, side="right"))
        tokens = int(cumsum[count - 1]) if count > 0 else 0
        row[f"freq{min_freq}_types"] = count
        row[f"freq{min_freq}_tokens"] = tokens

    # Rank cutoffs (top N% of vocabulary)
    for share, label in ((0.05, "5pct"), (0.30, "30pct")):
        n = max(1, int(share * distinct))
        tokens = int(cumsum[n - 1]) if n <= len(cumsum) else int(cumsum[-1])
        row[f"rank{label}_types"] = n
        row[f"rank{label}_tokens"] = tokens

    return row


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute statistics for aggregated frequency files.")
    parser.add_argument("--agg-dir", type=Path, default=AGGREGATED_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    files = sorted(args.agg_dir.glob("*.csv"))
    if not files:
        print(f"No CSV files found in {args.agg_dir}")
        sys.exit(1)

    rows = [compute_file_stats(p) for p in tqdm(files, desc="Statistics")]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} languages to {args.output}")


if __name__ == "__main__":
    main()
