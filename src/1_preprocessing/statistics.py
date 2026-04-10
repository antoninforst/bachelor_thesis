"""
Print summary statistics for every aggregated frequency file.

For each CSV in data/1_aggregated/ reports:
  - distinct word count
  - total word count (sum of frequencies)
  - word count at the 95 % and 99 % cumulative frequency cutoffs
    together with the frequency of the last included word

Usage:
    python src/0_data_processing/statistics.py                  # all files
    python src/0_data_processing/statistics.py --agg-dir path   # custom dir
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

AGGREGATED_DIR = Path("data/1_aggregated")

# Default coverage levels (in percent)
DEFAULT_COVERAGE_LEVELS: list[float] = [
    50, 55, 60, 65, 70,                                      # step 5
    72, 74, 76, 78, 80,                                      # step 2
    81, 82, 83, 84, 85, 86, 87, 88, 89, 90,                  # step 1
    90.5, 91, 91.5, 92, 92.5, 93, 93.5, 94, 94.5, 95,       # step 0.5
    95.5, 96, 96.5, 97, 97.5, 98, 98.5, 99, 99.5, 100,      # step 0.5
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CoverageInfo:
    """Types and tokens at a given cumulative-frequency coverage level."""

    percentage: float
    types: int
    tokens: int


@dataclass
class FreqCutoffInfo:
    """Types and tokens for a minimum-frequency cutoff."""

    min_frequency: int
    types: int
    tokens: int


@dataclass
class RankCutoffInfo:
    """Types and tokens for a rank-share cutoff (top N% of vocabulary)."""

    rank_share: float
    types: int
    tokens: int


@dataclass
class FileStats:
    """Aggregated statistics for one frequency file."""

    name: str
    distinct_words: int
    total_frequency: int
    coverages: list[CoverageInfo]
    freq_100: FreqCutoffInfo
    freq_10: FreqCutoffInfo
    rank_5pct: RankCutoffInfo
    rank_30pct: RankCutoffInfo

    def report(self) -> str:
        lines = [
            f"  File               : {self.name}",
            f"  Distinct words     : {self.distinct_words:,}",
            f"  Total frequency    : {self.total_frequency:,}",
        ]
        for cov in self.coverages:
            if cov.percentage in (50, 80, 90, 95, 99):
                pct_types = cov.types / self.distinct_words * 100
                lines.append(
                    f"  {cov.percentage:5.1f}% coverage   : {cov.types:,} types "
                    f"({pct_types:.2f}% of vocab), {cov.tokens:,} tokens"
                )
        lines.extend([
            f"  freq >= 100        : {self.freq_100.types:,} types, {self.freq_100.tokens:,} tokens",
            f"  freq >= 10         : {self.freq_10.types:,} types, {self.freq_10.tokens:,} tokens",
            f"  rank 5%            : {self.rank_5pct.types:,} types, {self.rank_5pct.tokens:,} tokens",
            f"  rank 30%           : {self.rank_30pct.types:,} types, {self.rank_30pct.tokens:,} tokens",
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

class StatisticsCalculator:
    """Computes frequency statistics for a single aggregated CSV."""

    def __init__(self, path: Path, coverage_levels: Optional[list[float]] = None):
        self._path = path
        self._coverage_levels = coverage_levels or DEFAULT_COVERAGE_LEVELS

    def compute(self) -> FileStats:
        df = pd.read_csv(self._path, dtype={"word": str, "frequency": "int64"})
        # File is already sorted by frequency descending from aggregation step
        frequencies = df["frequency"].values
        total = int(frequencies.sum())
        distinct = len(frequencies)

        # Precompute cumulative sum once — reused by all cutoff methods
        cumsum = np.cumsum(frequencies)

        coverages = self._find_coverages(cumsum, total, distinct)
        freq_100 = self._find_freq_cutoff(frequencies, cumsum, 100)
        freq_10 = self._find_freq_cutoff(frequencies, cumsum, 10)
        rank_5pct = self._find_rank_cutoff(cumsum, distinct, 0.05)
        rank_30pct = self._find_rank_cutoff(cumsum, distinct, 0.30)

        return FileStats(
            name=self._path.name,
            distinct_words=distinct,
            total_frequency=total,
            coverages=coverages,
            freq_100=freq_100,
            freq_10=freq_10,
            rank_5pct=rank_5pct,
            rank_30pct=rank_30pct,
        )

    def _find_coverages(self, cumsum, total: int, distinct: int) -> list[CoverageInfo]:
        """Find types & tokens for every configured coverage level via binary search."""
        results: list[CoverageInfo] = []
        for pct in self._coverage_levels:
            threshold = total * pct / 100.0
            idx = int(np.searchsorted(cumsum, threshold, side="left"))
            if idx >= distinct:
                idx = distinct - 1
            results.append(CoverageInfo(
                percentage=pct,
                types=idx + 1,
                tokens=int(cumsum[idx]),
            ))
        return results

    @staticmethod
    def _find_freq_cutoff(frequencies, cumsum, min_freq: int) -> FreqCutoffInfo:
        """Types & tokens for words with frequency >= *min_freq*."""
        # frequencies sorted descending → searchsorted on negated values
        count = int(np.searchsorted(-frequencies, -min_freq, side="right"))
        tokens = int(cumsum[count - 1]) if count > 0 else 0
        return FreqCutoffInfo(
            min_frequency=min_freq,
            types=count,
            tokens=tokens,
        )

    @staticmethod
    def _find_rank_cutoff(cumsum, distinct: int, share: float) -> RankCutoffInfo:
        """Types & tokens for keeping the top *share* fraction of words by rank."""
        n = max(1, int(share * distinct))
        tokens = int(cumsum[n - 1]) if n <= len(cumsum) else int(cumsum[-1])
        return RankCutoffInfo(
            rank_share=share * 100,
            types=n,
            tokens=tokens,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cov_prefix(pct: float) -> str:
    """Column-name prefix for a coverage percentage, e.g. 90.5 → 'cov_90_5'."""
    if pct == int(pct):
        return f"cov_{int(pct)}"
    return f"cov_{str(pct).replace('.', '_')}"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print statistics for aggregated frequency files."
    )
    parser.add_argument(
        "--agg-dir",
        type=Path,
        default=AGGREGATED_DIR,
        help=f"Directory with aggregated CSVs (default: {AGGREGATED_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save results as CSV.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    files = sorted(args.agg_dir.glob("*.csv"))

    if not files:
        print(f"No CSV files found in {args.agg_dir}")
        sys.exit(1)

    all_stats: list[FileStats] = []

    if args.output:
        print(f"Processing {len(files)} file(s) from {args.agg_dir}")
        for i, path in enumerate(files, 1):
            pct = i / len(files)
            bar_len = 40
            filled = int(bar_len * pct)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{bar}] {i}/{len(files)} {path.stem}", end="", flush=True)
            calc = StatisticsCalculator(path)
            all_stats.append(calc.compute())
        print()
    else:
        print(f"Statistics for {len(files)} file(s) in {args.agg_dir}")
        print("=" * 55)
        for path in files:
            calc = StatisticsCalculator(path)
            stats = calc.compute()
            all_stats.append(stats)
            print()
            print(stats.report())
            print("-" * 55)

    if args.output:
        rows = []
        for s in all_stats:
            row = {
                "file": Path(s.name).stem,
                "distinct_words": s.distinct_words,
                "total_frequency": s.total_frequency,
            }
            for cov in s.coverages:
                prefix = _cov_prefix(cov.percentage)
                row[f"{prefix}_types"] = cov.types
                row[f"{prefix}_tokens"] = cov.tokens
            row.update({
                "freq100_types": s.freq_100.types,
                "freq100_tokens": s.freq_100.tokens,
                "freq10_types": s.freq_10.types,
                "freq10_tokens": s.freq_10.tokens,
                "rank5pct_types": s.rank_5pct.types,
                "rank5pct_tokens": s.rank_5pct.tokens,
                "rank30pct_types": s.rank_30pct.types,
                "rank30pct_tokens": s.rank_30pct.tokens,
            })
            rows.append(row)
        df_out = pd.DataFrame(rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(args.output, index=False)
        print(f"\n→ Saved to {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
