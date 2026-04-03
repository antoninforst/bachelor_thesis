"""
Aggregate raw frequency CSV files by language code.

Groups files in data/0_raw/ by their first three characters (language code),
merges duplicate words, sums frequencies, filters non-words, and writes
sorted results to data/1_aggregated/<lang>.csv.

Usage:
    python src/0_data_processing/aggregate.py          # process all language codes
    python src/0_data_processing/aggregate.py ces eng   # process only ces and eng
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import regex

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/0_raw")
AGGREGATED_DIR = Path("data/1_aggregated")
CHUNK_SIZE = 500_000  # rows per chunk – keeps memory bounded for 50M+ line files
LETTER_PATTERN = regex.compile(r"\p{L}")  # matches any Unicode letter


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class AggregationStats:
    """Collects statistics about the aggregation process."""

    lang: str
    files_processed: int = 0
    total_rows_read: int = 0
    duplicates_merged: int = 0
    words_filtered: int = 0
    words_remaining: int = 0
    skipped_header_rows: int = 0

    def report(self) -> str:
        lines = [
            f"  Language code     : {self.lang}",
            f"  Files processed   : {self.files_processed}",
            f"  Total rows read   : {self.total_rows_read:,}",
            f"  Header rows skip. : {self.skipped_header_rows:,}",
            f"  Duplicates merged : {self.duplicates_merged:,}",
            f"  Non-words filtered: {self.words_filtered:,}",
            f"  Words remaining   : {self.words_remaining:,}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contains_letter(word: str) -> bool:
    """Return True if *word* contains at least one Unicode letter."""
    return bool(LETTER_PATTERN.search(str(word)))


def _group_files_by_lang(raw_dir: Path) -> dict[str, list[Path]]:
    """Return {lang_code: [path, …]} for every CSV in *raw_dir*."""
    groups: dict[str, list[Path]] = {}
    for path in sorted(raw_dir.glob("*.csv")):
        lang = path.name[:3]
        groups.setdefault(lang, []).append(path)
    return groups


def _detect_header_rows(path: Path) -> int:
    """
    Detect how many metadata rows precede the actual 'Item,Frequency' header.

    Some Leipzig files have two extra lines like:
        "corpus","ces_mixed_2012"
        "subcorpus","-"
    before the real header.
    """
    with open(path, encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            stripped = line.strip().strip('"').lower()
            if stripped.startswith("item"):
                return idx
            if idx > 10:  # safety – give up after 10 lines
                break
    return 0


def _read_file_chunked(path: Path) -> tuple[pd.DataFrame, int, int]:
    """
    Read a single CSV in chunks and return (df, total_rows, skipped_header_rows).

    The returned DataFrame has columns ['word', 'frequency'] with frequency
    already summed per word (handles intra-file duplicates).
    """
    skip = _detect_header_rows(path)

    chunks: list[pd.DataFrame] = []
    total_rows = 0

    reader = pd.read_csv(
        path,
        skiprows=skip,
        names=["word", "frequency"],
        header=0,
        dtype={"word": str, "frequency": str},
        na_filter=False,
        chunksize=CHUNK_SIZE,
        quoting=0,  # QUOTE_MINIMAL – handles quoted and unquoted values
        on_bad_lines="skip",
    )

    for chunk in reader:
        total_rows += len(chunk)
        # Coerce non-numeric frequencies (e.g. "-") to NaN, then drop them
        chunk["frequency"] = pd.to_numeric(chunk["frequency"], errors="coerce")
        chunk = chunk.dropna(subset=["frequency"])
        chunk["frequency"] = chunk["frequency"].astype("int64")
        # Aggregate within chunk to reduce memory
        agg = chunk.groupby("word", sort=False)["frequency"].sum().reset_index()
        chunks.append(agg)

    if not chunks:
        return pd.DataFrame(columns=["word", "frequency"]), 0, skip

    combined = pd.concat(chunks, ignore_index=True)
    combined = combined.groupby("word", sort=False)["frequency"].sum().reset_index()
    return combined, total_rows, skip


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

class Aggregator:
    """Aggregates raw frequency files for a single language code."""

    def __init__(self, lang: str, files: list[Path], output_dir: Path):
        self._lang = lang
        self._files = files
        self._output_dir = output_dir
        self._stats = AggregationStats(lang=lang)

    def run(self) -> AggregationStats:
        """Execute the full aggregation pipeline and write the result."""
        merged = self._load_and_merge()
        filtered = self._filter_non_words(merged)
        self._write(filtered)
        return self._stats

    # ------------------------------------------------------------------
    def _load_and_merge(self) -> pd.DataFrame:
        """Read every file and merge into a single DataFrame."""
        parts: list[pd.DataFrame] = []

        for path in self._files:
            print(f"  Reading {path.name} …")
            df, rows, skipped = _read_file_chunked(path)
            self._stats.files_processed += 1
            self._stats.total_rows_read += rows
            self._stats.skipped_header_rows += skipped
            parts.append(df)

        if not parts:
            return pd.DataFrame(columns=["word", "frequency"])

        combined = pd.concat(parts, ignore_index=True)
        unique_before = len(combined)
        combined = combined.groupby("word", sort=False)["frequency"].sum().reset_index()
        self._stats.duplicates_merged = unique_before - len(combined)
        return combined

    # ------------------------------------------------------------------
    def _filter_non_words(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows whose word column contains no Unicode letter."""
        mask = df["word"].map(_contains_letter)
        removed = (~mask).sum()
        self._stats.words_filtered = int(removed)
        result = df.loc[mask].copy()
        self._stats.words_remaining = len(result)
        return result

    # ------------------------------------------------------------------
    def _write(self, df: pd.DataFrame) -> None:
        """Sort by frequency descending and write to the aggregated folder."""
        df = df.sort_values("frequency", ascending=False, ignore_index=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / f"{self._lang}.csv"
        df.to_csv(out_path, index=False)
        print(f"  → Wrote {out_path} ({len(df):,} words)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate raw frequency CSVs by language code."
    )
    parser.add_argument(
        "langs",
        nargs="*",
        metavar="LANG",
        help="Three-letter language codes to process (default: all).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Directory with raw CSV files (default: {RAW_DIR}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=AGGREGATED_DIR,
        help=f"Output directory (default: {AGGREGATED_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    all_groups = _group_files_by_lang(args.raw_dir)

    if not all_groups:
        print(f"No CSV files found in {args.raw_dir}")
        sys.exit(1)

    # Determine which language codes to process
    if args.langs:
        selected = {l.lower() for l in args.langs}
        unknown = selected - set(all_groups)
        if unknown:
            print(f"Warning: no files found for language code(s): {', '.join(sorted(unknown))}")
        groups = {k: v for k, v in all_groups.items() if k in selected}
    else:
        groups = all_groups

    if not groups:
        print("Nothing to process.")
        sys.exit(1)

    print(f"Language codes to process: {', '.join(sorted(groups))}")
    print(f"{'=' * 50}")

    for lang in sorted(groups):
        files = groups[lang]
        print(f"\n[{lang}] Processing {len(files)} file(s):")
        aggregator = Aggregator(lang, files, args.out_dir)
        stats = aggregator.run()
        print(f"\n{stats.report()}")
        print(f"{'-' * 50}")

    print("\nDone.")


if __name__ == "__main__":
    main()
