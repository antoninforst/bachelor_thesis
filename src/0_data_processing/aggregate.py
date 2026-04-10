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
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import re

csv.field_size_limit(10 * 1024 * 1024)  # 10 MB – some Glot500 rows are very large

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/0_raw")
AGGREGATED_DIR = Path("data/1_aggregated")
_HAS_LETTER = re.compile(r"[^\W\d_]").search  # matches any Unicode letter
_STRIP_PUNCT = re.compile(r'^[.,?!"]+|[.,?!"]+$')  # non-functional punctuation on sides


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
    words_after_punct_clean: int = 0
    skipped_header_rows: int = 0

    def report(self) -> str:
        lines = [
            f"  Language code     : {self.lang}",
            f"  Files processed   : {self.files_processed}",
            f"  Total rows read   : {self.total_rows_read:,}",
            f"  Header rows skip. : {self.skipped_header_rows:,}",
            f"  Duplicates merged : {self.duplicates_merged:,}",
            f"  After punct clean : {self.words_after_punct_clean:,}",
            f"  Non-words filtered: {self.words_filtered:,}",
            f"  Words remaining   : {self.words_remaining:,}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contains_letter(word: str) -> bool:
    """Return True if *word* contains at least one Unicode letter."""
    return bool(_HAS_LETTER(word))


def _group_files_by_lang(raw_dir: Path) -> dict[str, list[Path]]:
    """Return {lang_code: [path, …]} for every CSV in *raw_dir*."""
    groups: dict[str, list[Path]] = {}
    for path in sorted(raw_dir.glob("*.csv")):
        lang = path.name[:3]
        groups.setdefault(lang, []).append(path)
    return groups


def _read_file_into_dict(
    path: Path,
    freq_dict: dict[str, int],
) -> tuple[int, int]:
    """
    Read a single CSV and merge word frequencies into *freq_dict* in-place.

    Returns (rows_read, skipped_header_rows).
    """
    skipped_header = 0
    rows_read = 0
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        # Skip metadata rows until we find the header (Item/word)
        for line in reader:
            if line and line[0].strip().lower() in ("item", "word"):
                skipped_header = reader.line_num - 1
                break
            skipped_header += 1
        for row in reader:
            if len(row) < 2:
                continue
            word = row[0]
            try:
                freq = int(row[1])
            except (ValueError, IndexError):
                continue
            rows_read += 1
            if word in freq_dict:
                freq_dict[word] += freq
            else:
                freq_dict[word] = freq
    return rows_read, skipped_header


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
        freq_dict = self._load_and_merge()
        cleaned = self._clean_punctuation(freq_dict)
        filtered = self._filter_non_words(cleaned)
        self._write(filtered)
        return self._stats

    # ------------------------------------------------------------------
    def _load_and_merge(self) -> dict[str, int]:
        """Read every file and merge into a single dict {word: total_freq}."""
        freq_dict: dict[str, int] = {}
        total_rows_across_files = 0

        for path in self._files:
            print(f"  Reading {path.name} …")
            rows, skipped = _read_file_into_dict(path, freq_dict)
            self._stats.files_processed += 1
            self._stats.total_rows_read += rows
            self._stats.skipped_header_rows += skipped
            total_rows_across_files += rows

        self._stats.duplicates_merged = total_rows_across_files - len(freq_dict)
        return freq_dict

    # ------------------------------------------------------------------
    def _clean_punctuation(self, freq_dict: dict[str, int]) -> dict[str, int]:
        """Strip non-functional punctuation (.,?!") from sides and re-merge."""
        cleaned: dict[str, int] = {}
        for word, freq in freq_dict.items():
            stripped = _STRIP_PUNCT.sub("", word)  # strip from both ends
            if stripped:
                if stripped in cleaned:
                    cleaned[stripped] += freq
                else:
                    cleaned[stripped] = freq
        self._stats.words_after_punct_clean = len(cleaned)
        return cleaned

    # ------------------------------------------------------------------
    def _filter_non_words(self, freq_dict: dict[str, int]) -> dict[str, int]:
        """Remove entries whose word contains no Unicode letter."""
        filtered = {w: f for w, f in freq_dict.items() if _contains_letter(w)}
        self._stats.words_filtered = len(freq_dict) - len(filtered)
        self._stats.words_remaining = len(filtered)
        return filtered

    # ------------------------------------------------------------------
    def _write(self, freq_dict: dict[str, int]) -> None:
        """Sort by frequency descending and write to the aggregated folder."""
        sorted_items = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / f"{self._lang}.csv"
        with open(out_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["word", "frequency"])
            writer.writerows(sorted_items)
        print(f"  → Wrote {out_path} ({len(sorted_items):,} words)")


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
    parser.add_argument(
        "--ignore",
        type=Path,
        default=None,
        help="Path to ignored_files.csv to exclude files from aggregation.",
    )
    return parser.parse_args(argv)


def _load_ignore_set(path: Path) -> set[str]:
    """Load filenames to ignore from a CSV with a 'filename' column."""
    ignored: set[str] = set()
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ignored.add(row["filename"])
    return ignored


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    # Load ignore list if provided
    ignored_files: set[str] = set()
    if args.ignore:
        if not args.ignore.exists():
            print(f"Ignore file not found: {args.ignore}")
            sys.exit(1)
        ignored_files = _load_ignore_set(args.ignore)
        print(f"Ignoring {len(ignored_files)} file(s) from {args.ignore}")

    all_groups = _group_files_by_lang(args.raw_dir)

    # Filter out ignored files
    if ignored_files:
        all_groups = {
            lang: [f for f in files if f.name not in ignored_files]
            for lang, files in all_groups.items()
        }
        all_groups = {k: v for k, v in all_groups.items() if v}

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
