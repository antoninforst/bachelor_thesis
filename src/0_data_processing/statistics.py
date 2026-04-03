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

import pandas as pd

AGGREGATED_DIR = Path("data/1_aggregated")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CutoffInfo:
    """Statistics for a single cumulative-frequency cutoff."""

    percentage: float
    words_needed: int
    last_word_frequency: int


@dataclass
class FileStats:
    """Aggregated statistics for one frequency file."""

    name: str
    distinct_words: int
    total_frequency: int
    cutoff_95: CutoffInfo
    cutoff_99: CutoffInfo

    def report(self) -> str:
        pct_95 = self.cutoff_95.words_needed / self.distinct_words * 100
        pct_99 = self.cutoff_99.words_needed / self.distinct_words * 100
        lines = [
            f"  File               : {self.name}",
            f"  Distinct words     : {self.distinct_words:,}",
            f"  Total frequency    : {self.total_frequency:,}",
            f"  95 % cutoff        : {self.cutoff_95.words_needed:,} words "
            f"({pct_95:.2f} % of vocabulary, last freq = {self.cutoff_95.last_word_frequency:,})",
            f"  99 % cutoff        : {self.cutoff_99.words_needed:,} words "
            f"({pct_99:.2f} % of vocabulary, last freq = {self.cutoff_99.last_word_frequency:,})",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

class StatisticsCalculator:
    """Computes frequency statistics for a single aggregated CSV."""

    def __init__(self, path: Path):
        self._path = path

    def compute(self) -> FileStats:
        df = pd.read_csv(self._path, dtype={"word": str, "frequency": "int64"})
        # File is already sorted by frequency descending from aggregation step
        frequencies = df["frequency"].values
        total = int(frequencies.sum())
        distinct = len(frequencies)

        cutoff_95 = self._find_cutoff(frequencies, total, 0.95)
        cutoff_99 = self._find_cutoff(frequencies, total, 0.99)

        return FileStats(
            name=self._path.name,
            distinct_words=distinct,
            total_frequency=total,
            cutoff_95=cutoff_95,
            cutoff_99=cutoff_99,
        )

    @staticmethod
    def _find_cutoff(frequencies, total: int, ratio: float) -> CutoffInfo:
        """Return how many top words are needed to cover *ratio* of total frequency."""
        threshold = total * ratio
        cumsum = 0
        for i, freq in enumerate(frequencies):
            cumsum += freq
            if cumsum >= threshold:
                return CutoffInfo(
                    percentage=ratio * 100,
                    words_needed=i + 1,
                    last_word_frequency=int(freq),
                )
        # All words needed (shouldn't normally happen)
        return CutoffInfo(
            percentage=ratio * 100,
            words_needed=len(frequencies),
            last_word_frequency=int(frequencies[-1]) if len(frequencies) else 0,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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

    print(f"Statistics for {len(files)} file(s) in {args.agg_dir}")
    print("=" * 55)

    all_stats: list[FileStats] = []
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
            pct_95 = s.cutoff_95.words_needed / s.distinct_words * 100
            pct_99 = s.cutoff_99.words_needed / s.distinct_words * 100
            rows.append({
                "file": Path(s.name).stem,
                "distinct_words": s.distinct_words,
                "total_frequency": s.total_frequency,
                "cutoff_95_words": s.cutoff_95.words_needed,
                "cutoff_95_pct_vocab": round(pct_95, 2),
                "cutoff_95_last_freq": s.cutoff_95.last_word_frequency,
                "cutoff_99_words": s.cutoff_99.words_needed,
                "cutoff_99_pct_vocab": round(pct_99, 2),
                "cutoff_99_last_freq": s.cutoff_99.last_word_frequency,
            })
        df_out = pd.DataFrame(rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(args.output, index=False)
        print(f"\n→ Saved to {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
