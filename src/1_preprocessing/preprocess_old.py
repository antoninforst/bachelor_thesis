#!/usr/bin/env python3
"""Frequency list analysis tool.

Analyzes word frequency CSVs (Item, Frequency) with optional metadata rows.
Supports coverage analysis, plotting, truncation, and weighted word sampling.

Usage examples:
    python analyze.py coverage data/ces.csv
    python analyze.py coverage-report data/ -p "^[a-z]{3}\\.csv$" -o report.csv
    python analyze.py plot data/ces.csv -o freq_plot.png
    python analyze.py truncate data/ces.csv -c 95 -o ces_top95.csv
    python analyze.py wordlist data/ces.csv -n 5000 -o words.txt --seed 42
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------

@dataclass
class CoverageStats:
    """Stats for a single cumulative coverage threshold."""

    threshold: float
    words_needed: int
    total_words: int
    percentage: float
    min_frequency: int


class FrequencyList:
    """Word-frequency list with coverage analysis capabilities.

    Parameters
    ----------
    df : DataFrame with at least columns ``Item`` (str) and ``Frequency`` (int).
    name : Optional label (e.g. source file stem).
    """

    def __init__(self, df: pd.DataFrame, name: str = "") -> None:
        self.df = df.copy()
        self.name = name
        self._total: int = int(self.df["Frequency"].sum())
        self._compute_coverage()

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_csv(cls, path: Path, skip_metadata: int = 2) -> FrequencyList:
        """Load from a CSV file, skipping *skip_metadata* rows before the header."""
        df = pd.read_csv(
            path,
            sep=",",
            encoding="utf-8",
            header=skip_metadata,
            dtype={"Item": str, "Frequency": "Int64"},
        )
        df = df.dropna(subset=["Frequency"])
        return cls(df, name=path.stem)

    # -- internals ----------------------------------------------------------

    def _compute_coverage(self) -> None:
        self.df["Coverage"] = self.df["Frequency"] / self._total
        self.df["CumulativeCoverage"] = self.df["Frequency"].cumsum() / self._total

    # -- public API ---------------------------------------------------------

    @property
    def total(self) -> int:
        return self._total

    def coverage_at(self, threshold: float) -> CoverageStats:
        """Return stats for a cumulative coverage *threshold* (0–1)."""
        mask = self.df["CumulativeCoverage"] <= threshold
        count = int(mask.sum())
        # include the first word that crosses the threshold
        if count < len(self.df):
            count += 1
        return CoverageStats(
            threshold=threshold,
            words_needed=count,
            total_words=len(self.df),
            percentage=count / len(self.df) * 100,
            min_frequency=int(self.df.iloc[count - 1]["Frequency"]),
        )

    def truncate(self, threshold: float) -> pd.DataFrame:
        """Return rows up to and including the *threshold* coverage boundary."""
        stats = self.coverage_at(threshold)
        return self.df.iloc[: stats.words_needed][["Item", "Frequency"]].copy()

    def sample_wordlist(
        self,
        size: int,
        coverage: float = 1.0,
        start_rank: int = 1,
        seed: int | None = None,
    ) -> list[str]:
        """Sample *size* unique words with uniform expected density in log-rank.

        Each word at rank *r* is sampled with probability proportional to 1/r,
        so all frequency bands are equally represented in expectation.

        Parameters
        ----------
        coverage : float
            Cumulative coverage threshold (0–1). Only words up to this
            coverage are considered; the long tail is discarded.
        start_rank : int
            First rank to include (1-based). Words before this rank
            (i.e. the most frequent) are skipped.
        seed : int | None
            Random seed for reproducibility.
        """
        if coverage < 1.0:
            stats = self.coverage_at(coverage)
            pool = self.df.iloc[: stats.words_needed]
        else:
            pool = self.df

        # Trim the top by start_rank
        if start_rank > 1:
            pool = pool.iloc[start_rank - 1 :]

        available = len(pool)
        if size > available:
            print(
                f"Warning: requested {size} words but only {available} available. "
                f"Returning all.",
                file=sys.stderr,
            )
            size = available

        rng = np.random.default_rng(seed)
        ranks = np.arange(1, available + 1, dtype=float)
        probs = 1.0 / ranks
        probs /= probs.sum()
        indices = rng.choice(available, size=size, replace=False, p=probs)
        return pool.iloc[sorted(indices)]["Item"].tolist()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_csv_files(folder: Path, pattern: str) -> list[Path]:
    """Return sorted CSV paths in *folder* whose name matches *pattern*."""
    regex = re.compile(pattern)
    return sorted(p for p in folder.glob("*.csv") if regex.search(p.name))


def plot_frequency(fl: FrequencyList, output: str | None = None) -> None:
    """Produce a log-log plot of frequency vs rank."""
    rank = np.arange(1, len(fl.df) + 1)
    log_rank = np.log10(rank)
    log_freq = np.log10(fl.df["Frequency"].to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(log_rank, log_freq, linewidth=0.8)
    ax.set_xlabel("Log Rank")
    ax.set_ylabel("Log Frequency")
    ax.set_title(f"Log-Log Frequency — {fl.name}")

    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=150)
        print(f"Plot saved to {output}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------

def cmd_coverage(args: argparse.Namespace) -> None:
    fl = FrequencyList.from_csv(Path(args.file), skip_metadata=args.skip_metadata)
    for t in args.thresholds:
        s = fl.coverage_at(t / 100)
        print(
            f"Coverage {s.threshold:.0%}: "
            f"{s.words_needed:,} words ({s.percentage:.2f}% of {s.total_words:,}), "
            f"min frequency = {s.min_frequency:,}"
        )


def cmd_coverage_report(args: argparse.Namespace) -> None:
    folder = Path(args.folder)
    files = find_csv_files(folder, args.pattern)
    if not files:
        print(f"No files matching '{args.pattern}' in {folder}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    for path in files:
        fl = FrequencyList.from_csv(path, skip_metadata=args.skip_metadata)
        for t in args.thresholds:
            s = fl.coverage_at(t / 100)
            rows.append(
                {
                    "file": path.name,
                    "threshold": f"{s.threshold:.0%}",
                    "words_needed": s.words_needed,
                    "total_words": s.total_words,
                    "percentage": round(s.percentage, 2),
                    "min_frequency": s.min_frequency,
                }
            )

    report = pd.DataFrame(rows)
    out = Path(args.output)
    report.to_csv(out, index=False)
    print(f"Report saved to {out} ({len(files)} file(s), {len(rows)} row(s))")


def cmd_plot(args: argparse.Namespace) -> None:
    fl = FrequencyList.from_csv(Path(args.file), skip_metadata=args.skip_metadata)
    plot_frequency(fl, output=args.output)


def cmd_truncate(args: argparse.Namespace) -> None:
    fl = FrequencyList.from_csv(Path(args.file), skip_metadata=args.skip_metadata)
    truncated = fl.truncate(args.coverage / 100)
    out = Path(args.output)
    truncated.to_csv(out, index=False)
    print(f"Truncated to {len(truncated):,} words at {args.coverage}% coverage → {out}")


def cmd_wordlist(args: argparse.Namespace) -> None:
    fl = FrequencyList.from_csv(Path(args.file), skip_metadata=args.skip_metadata)
    words = fl.sample_wordlist(
        args.size, coverage=args.coverage / 100, start_rank=args.start_rank, seed=args.seed,
    )
    out = Path(args.output)
    out.write_text("\n".join(words), encoding="utf-8")
    print(f"Word list of {len(words):,} words → {out}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frequency list analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skip-metadata",
        type=int,
        default=0,
        help="Number of metadata rows before the CSV header (default: 2)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # -- coverage -----------------------------------------------------------
    p = sub.add_parser("coverage", help="Show cumulative coverage stats for a file")
    p.add_argument("file", help="Path to frequency CSV")
    p.add_argument(
        "-t",
        "--thresholds",
        type=float,
        nargs="+",
        default=[95, 99],
        help="Coverage thresholds in %% (default: 95 99)",
    )

    # -- coverage-report ----------------------------------------------------
    p = sub.add_parser(
        "coverage-report",
        help="Batch coverage report for all matching CSVs in a folder",
    )
    p.add_argument("folder", help="Folder containing frequency CSVs")
    p.add_argument(
        "-p",
        "--pattern",
        default=r".*\.csv$",
        help="Regex to match filenames (default: .*\\.csv$)",
    )
    p.add_argument(
        "-t",
        "--thresholds",
        type=float,
        nargs="+",
        default=[95, 99],
        help="Coverage thresholds in %% (default: 95 99)",
    )
    p.add_argument(
        "-o", "--output", default="coverage_report.csv", help="Output CSV path"
    )

    # -- plot ---------------------------------------------------------------
    p = sub.add_parser("plot", help="Plot frequency vs rank (linear + log)")
    p.add_argument("file", help="Path to frequency CSV")
    p.add_argument(
        "-o", "--output", help="Save plot to file (shows interactive window if omitted)"
    )

    # -- truncate -----------------------------------------------------------
    p = sub.add_parser(
        "truncate", help="Export frequency list truncated at a coverage threshold"
    )
    p.add_argument("file", help="Path to frequency CSV")
    p.add_argument(
        "-c",
        "--coverage",
        type=float,
        default=95,
        help="Coverage threshold in %% (default: 95)",
    )
    p.add_argument("-o", "--output", required=True, help="Output CSV path")

    # -- wordlist -----------------------------------------------------------
    p = sub.add_parser(
        "wordlist", help="Sample a word list weighted by log-frequency"
    )
    p.add_argument("file", help="Path to frequency CSV")
    p.add_argument(
        "-n", "--size", type=int, required=True, help="Number of words to sample"
    )
    p.add_argument(
        "-c",
        "--coverage",
        type=float,
        default=100,
        help="Cumulative coverage threshold in %% (default: 100, i.e. all words)",
    )
    p.add_argument(
        "-s",
        "--start-rank",
        type=int,
        default=1,
        help="First rank to include, skipping the most frequent words (default: 1)",
    )
    p.add_argument("-o", "--output", required=True, help="Output file path")
    p.add_argument("--seed", type=int, help="Random seed for reproducibility")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "coverage": cmd_coverage,
        "coverage-report": cmd_coverage_report,
        "plot": cmd_plot,
        "truncate": cmd_truncate,
        "wordlist": cmd_wordlist,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
