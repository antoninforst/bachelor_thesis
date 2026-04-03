"""
Preprocess aggregated frequency files for annotation.

Reads each CSV from data/1_aggregated/, cuts it at a cumulative frequency
cutoff (default 95 %), adds an empty ``lemma`` column, and writes the
result to data/2_annotated/<lang>.csv.

Usage:
    python src/0_data_processing/preprocess.py                # all files, 95 %
    python src/0_data_processing/preprocess.py --cutoff 0.99  # all files, 99 %
    python src/0_data_processing/preprocess.py ces eng        # selected codes
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

AGGREGATED_DIR = Path("data/1_aggregated")
ANNOTATED_DIR = Path("data/2_annotated")
DEFAULT_CUTOFF = 0.95


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PreprocessStats:
    """Statistics produced by a single file preprocessing run."""

    name: str
    original_words: int
    kept_words: int
    cutoff: float
    total_frequency: int
    kept_frequency: int
    last_frequency: int

    def report(self) -> str:
        removed = self.original_words - self.kept_words
        lines = [
            f"  File               : {self.name}",
            f"  Cutoff             : {self.cutoff * 100:.0f} %",
            f"  Original words     : {self.original_words:,}",
            f"  Kept words         : {self.kept_words:,}  (removed {removed:,})",
            f"  Total frequency    : {self.total_frequency:,}",
            f"  Kept frequency     : {self.kept_frequency:,}",
            f"  Last word frequency: {self.last_frequency:,}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

class Preprocessor:
    """Cuts an aggregated frequency file at a cumulative-frequency cutoff."""

    def __init__(self, path: Path, output_dir: Path, cutoff: float):
        self._path = path
        self._output_dir = output_dir
        self._cutoff = cutoff

    def run(self) -> PreprocessStats:
        df = pd.read_csv(self._path, dtype={"word": str, "frequency": "int64"})
        total_freq = int(df["frequency"].sum())
        threshold = total_freq * self._cutoff

        cumsum = np.cumsum(df["frequency"].values)
        # Find first index where cumsum reaches the threshold
        cut_idx = int(np.searchsorted(cumsum, threshold, side="left"))
        # Include that row (so cumsum >= threshold)
        kept = df.iloc[: cut_idx + 1].copy()

        kept["lemma"] = ""

        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / self._path.name
        kept.to_csv(out_path, index=False)
        print(f"  → Wrote {out_path} ({len(kept):,} words)")

        return PreprocessStats(
            name=self._path.name,
            original_words=len(df),
            kept_words=len(kept),
            cutoff=self._cutoff,
            total_frequency=total_freq,
            kept_frequency=int(kept["frequency"].sum()),
            last_frequency=int(kept["frequency"].iloc[-1]),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess aggregated CSVs: cut at cumulative-frequency cutoff and add lemma column."
    )
    parser.add_argument(
        "langs",
        nargs="*",
        metavar="LANG",
        help="Three-letter language codes to process (default: all).",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=DEFAULT_CUTOFF,
        help=f"Cumulative frequency cutoff ratio (default: {DEFAULT_CUTOFF}).",
    )
    parser.add_argument(
        "--agg-dir",
        type=Path,
        default=AGGREGATED_DIR,
        help=f"Directory with aggregated CSVs (default: {AGGREGATED_DIR}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ANNOTATED_DIR,
        help=f"Output directory (default: {ANNOTATED_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    all_files = sorted(args.agg_dir.glob("*.csv"))

    if not all_files:
        print(f"No CSV files found in {args.agg_dir}")
        sys.exit(1)

    if args.langs:
        selected = {lang.lower() for lang in args.langs}
        files = [f for f in all_files if f.stem in selected]
        unknown = selected - {f.stem for f in files}
        if unknown:
            print(f"Warning: no aggregated file for: {', '.join(sorted(unknown))}")
    else:
        files = all_files

    if not files:
        print("Nothing to process.")
        sys.exit(1)

    print(f"Preprocessing {len(files)} file(s) with cutoff {args.cutoff * 100:.0f} %")
    print("=" * 55)

    for path in files:
        print(f"\n[{path.stem}]")
        proc = Preprocessor(path, args.out_dir, args.cutoff)
        stats = proc.run()
        print()
        print(stats.report())
        print("-" * 55)

    print("\nDone.")


if __name__ == "__main__":
    main()
