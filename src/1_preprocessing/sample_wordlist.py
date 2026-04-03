"""
Generate a shorter word list by sampling words weighted by log-rank.

Each word at rank r is sampled with probability proportional to 1/r, giving
uniform expected density in log-space — all frequency bands are equally
represented.  Optionally generates a log-log verification plot.

Usage:
    python src/1_preprocessing/sample_wordlist.py -n 5000                # all files
    python src/1_preprocessing/sample_wordlist.py -n 5000 ces eng        # selected
    python src/1_preprocessing/sample_wordlist.py -n 5000 --plot         # with plot
    python src/1_preprocessing/sample_wordlist.py -n 5000 --seed 42      # reproducible
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ANNOTATED_DIR = Path("data/2_annotated")
SAMPLES_DIR = Path("data/4_samples")
DEFAULT_SIZE = 5000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SampleStats:
    """Statistics for a single sampling run."""

    lang: str
    original_count: int
    sampled_count: int
    original_total_freq: int
    sampled_total_freq: int
    min_freq_original: int
    min_freq_sampled: int
    max_freq_sampled: int

    def report(self) -> str:
        lines = [
            f"  Language           : {self.lang}",
            f"  Original words     : {self.original_count:,}",
            f"  Sampled words      : {self.sampled_count:,}",
            f"  Original total freq: {self.original_total_freq:,}",
            f"  Sampled total freq : {self.sampled_total_freq:,}",
            f"  Freq range (sample): {self.min_freq_sampled:,} – {self.max_freq_sampled:,}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

class WordlistSampler:
    """Samples words from an annotated CSV using log-rank weighting."""

    def __init__(self, path: Path, size: int, seed: Optional[int] = None):
        self._path = path
        self._size = size
        self._seed = seed

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, SampleStats]:
        """Return (original_df, sampled_df, sampled_original_indices, stats)."""
        df = pd.read_csv(self._path, dtype={"word": str, "frequency": "int64", "lemma": str},
                         keep_default_na=False)

        available = len(df)
        actual_size = min(self._size, available)
        if actual_size < self._size:
            print(f"  Warning: requested {self._size} but only {available} available.",
                  file=sys.stderr)

        indices = self._sample_indices(available, actual_size)
        sorted_indices = np.sort(indices)
        sampled = df.iloc[sorted_indices].copy().reset_index(drop=True)

        stats = SampleStats(
            lang=self._path.stem,
            original_count=available,
            sampled_count=len(sampled),
            original_total_freq=int(df["frequency"].sum()),
            sampled_total_freq=int(sampled["frequency"].sum()),
            min_freq_original=int(df["frequency"].min()),
            min_freq_sampled=int(sampled["frequency"].min()),
            max_freq_sampled=int(sampled["frequency"].max()),
        )
        return df, sampled, sorted_indices, stats

    def _sample_indices(self, n: int, size: int) -> np.ndarray:
        """Sample *size* indices from 0..n-1 with P(i) ∝ 1/(i+1)."""
        rng = np.random.default_rng(self._seed)
        ranks = np.arange(1, n + 1, dtype=np.float64)
        probs = 1.0 / ranks
        probs /= probs.sum()
        return rng.choice(n, size=size, replace=False, p=probs)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(original: pd.DataFrame, sampled_indices: np.ndarray,
                    sampled: pd.DataFrame, lang: str,
                    output: Optional[Path] = None) -> None:
    """Log-log rank vs frequency plot comparing original and sample.

    Sampled words are plotted at their *original* rank so they sit on
    the original curve.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Original — as a continuous line
    orig_rank = np.arange(1, len(original) + 1)
    ax.plot(orig_rank, original["frequency"].values,
            linewidth=0.8, alpha=0.7, label="Original", color="steelblue")

    # Sampled — plotted at their original rank positions
    sample_orig_ranks = sampled_indices + 1  # 0-based index → 1-based rank
    ax.scatter(sample_orig_ranks, sampled["frequency"].values,
               s=6, alpha=0.7, label="Sampled", color="tomato", zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Log-Log Frequency Distribution — {lang}")
    ax.legend()
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f"  Plot saved to {output}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a shorter word list weighted by log-rank."
    )
    parser.add_argument(
        "langs", nargs="*", metavar="LANG",
        help="Three-letter language codes (default: all).",
    )
    parser.add_argument(
        "-n", "--size", type=int, default=DEFAULT_SIZE,
        help=f"Number of words to sample (default: {DEFAULT_SIZE}).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate a log-log verification plot for each language.",
    )
    parser.add_argument(
        "--ann-dir", type=Path, default=ANNOTATED_DIR,
        help=f"Directory with annotated CSVs (default: {ANNOTATED_DIR}).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=SAMPLES_DIR,
        help=f"Output directory (default: {SAMPLES_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    out_dir = args.out_dir

    all_files = sorted(args.ann_dir.glob("*.csv"))
    if not all_files:
        print(f"No CSV files in {args.ann_dir}")
        sys.exit(1)

    if args.langs:
        selected = {l.lower() for l in args.langs}
        files = [f for f in all_files if f.stem in selected]
        unknown = selected - {f.stem for f in files}
        if unknown:
            print(f"Warning: no file for: {', '.join(sorted(unknown))}")
    else:
        files = all_files

    if not files:
        print("Nothing to process.")
        sys.exit(1)

    print(f"Sampling {args.size} words from {len(files)} file(s)")
    print("=" * 55)

    for path in files:
        print(f"\n[{path.stem}]")
        sampler = WordlistSampler(path, args.size, args.seed)
        original, sampled, sampled_indices, stats = sampler.run()

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}_sampled.csv"
        sampled.to_csv(out_path, index=False)
        print(f"  → Wrote {out_path} ({len(sampled):,} words)")
        print()
        print(stats.report())

        if args.plot:
            plot_path = out_dir / f"{path.stem}_sample_plot.png"
            plot_comparison(original, sampled_indices, sampled, path.stem, plot_path)

        print("-" * 55)

    print("\nDone.")


if __name__ == "__main__":
    main()
