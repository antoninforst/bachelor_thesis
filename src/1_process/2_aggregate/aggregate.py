"""
Aggregate raw frequency CSV files by language code.

Groups files in data/0_raw/ by their first three characters (language code),
merges duplicate words, sums frequencies, cleans and normalises words,
and writes sorted results to data/1_aggregated/<lang>.csv.

Usage:
    python src/1_process/2_aggregate/aggregate.py                        # process all
    python src/1_process/2_aggregate/aggregate.py ces eng                # specific langs
    python src/1_process/2_aggregate/aggregate.py --repair               # re-clean aggregated
    python src/1_process/2_aggregate/aggregate.py --repair ces eng       # re-clean specific
"""

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import re

csv.field_size_limit(10 * 1024 * 1024)  # 10 MB

RAW_DIR = Path("data/0_raw")
AGGREGATED_DIR = Path("data/1_aggregated")
REPORT_DIR = Path("results/1_process")
SCRIPTS_CSV = Path("src/1_process/1_filter/scripts.csv")
LANGUAGE_OVERVIEW_CSV = Path("results/1_process/2_aggregate/language_overview.csv")
FALLBACK_SCRIPT = "xxxx"

# Any unicode letter
_HAS_LETTER = re.compile(r"[^\W\d_]").search

STRIP_CHARS = '.,?!":()…“ ” „ ‟ ‘ ’ « »'
_STRIP_PUNCT = re.compile(
    r"^[{0}]+|[{0}]+$".format(re.escape(STRIP_CHARS))
)

@dataclass
class FileReport:
    lng_shortcut: str
    file_name: str
    total_tokens: int
    total_tokens_after_ignoring: int
    total_types: int
    ignored_word_count: int
    avg_deleted_punc_per_token: float
    new_types_from_previous_file: int

def _contains_letter(word: str) -> bool:
    """Return True if *word* contains at least one Unicode letter."""
    return bool(_HAS_LETTER(word))

def _strip_punctuation(freq_dict: dict[str, int]) -> dict[str, int]:
    """Strip non-functional punctuation from word boundaries and re-merge."""
    cleaned: dict[str, int] = {}
    for word, freq in freq_dict.items():
        stripped = _STRIP_PUNCT.sub("", word)
        if stripped:
            cleaned[stripped] = cleaned.get(stripped, 0) + freq
    return cleaned


def _lowercase(freq_dict: dict[str, int]) -> dict[str, int]:
    """Lowercase all words and merge duplicates."""
    lowered: dict[str, int] = {}
    for word, freq in freq_dict.items():
        key = word.lower()
        lowered[key] = lowered.get(key, 0) + freq
    return lowered


def _filter_non_words(freq_dict: dict[str, int]) -> dict[str, int]:
    """Remove entries whose word contains no Unicode letter."""
    return {w: f for w, f in freq_dict.items() if _contains_letter(w)}


def clean_freq_dict(freq_dict: dict[str, int]) -> dict[str, int]:
    """Run the full cleaning pipeline on a frequency dict."""
    freq_dict = _strip_punctuation(freq_dict)
    freq_dict = _lowercase(freq_dict)
    freq_dict = _filter_non_words(freq_dict)
    return freq_dict


def _group_files_by_lang(raw_dir: Path) -> dict[str, list[Path]]:
    """Return {lang_code: [path, …]} for every CSV in *raw_dir*."""
    groups: dict[str, list[Path]] = {}
    for path in sorted(raw_dir.glob("*.csv")):
        lang = path.name[:3]
        groups.setdefault(lang, []).append(path)
    return groups


def _load_script_codes(scripts_csv: Path) -> dict[str, str]:
    """Read {script_name: iso_code} from scripts.csv (e.g. Latin -> Latn)."""
    mapping: dict[str, str] = {}
    with open(scripts_csv, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter=";")
        for row in reader:
            if len(row) >= 2 and row[0].strip():
                mapping[row[0].strip()] = row[1].strip()
    return mapping


def _load_lang_scripts(
    overview_csv: Path,
    script_codes: dict[str, str],
) -> dict[str, str]:
    """Read language_overview.csv and return {lang: iso_script_code}.

    Maps primary_script name (e.g. "Latin") through script_codes to get
    the ISO 15924 code (e.g. "Latn").
    """
    mapping: dict[str, str] = {}
    with open(overview_csv, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            lang = row["used_shortcut"].strip()
            script_name = row["primary_script"].strip()
            mapping[lang] = script_codes.get(script_name, FALLBACK_SCRIPT)
    return mapping


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

    def __init__(self, lang: str, files: list[Path], output_dir: Path,
                 script_code: str = FALLBACK_SCRIPT):
        self._lang = lang
        self._files = files
        self._output_dir = output_dir
        self._script_code = script_code

    def run(self, on_file_done=None) -> list[FileReport]:
        """Execute the full aggregation pipeline, write result, return per-file reports."""
        cumulative_types: set[str] = set()
        cumulative_freq: dict[str, int] = {}
        reports: list[FileReport] = []

        for path in self._files:
            raw_dict: dict[str, int] = {}
            _read_file_into_dict(path, raw_dict)

            total_tokens_raw = sum(raw_dict.values())
            raw_type_count = len(raw_dict)

            # Avg punctuation characters stripped per token
            total_punc_chars = 0
            for word, freq in raw_dict.items():
                stripped = _STRIP_PUNCT.sub("", word)
                total_punc_chars += freq * (len(word) - len(stripped))
            avg_punc = total_punc_chars / total_tokens_raw if total_tokens_raw else 0.0

            cleaned = clean_freq_dict(raw_dict)

            total_tokens_clean = sum(cleaned.values())
            clean_type_count = len(cleaned)

            # New types not seen in previous files
            file_types = set(cleaned.keys())
            new_types = file_types - cumulative_types

            cumulative_types.update(file_types)
            for w, f in cleaned.items():
                cumulative_freq[w] = cumulative_freq.get(w, 0) + f

            reports.append(FileReport(
                lng_shortcut=self._lang,
                file_name=path.name,
                total_tokens=total_tokens_raw,
                total_tokens_after_ignoring=total_tokens_clean,
                total_types=clean_type_count,
                ignored_word_count=raw_type_count - clean_type_count,
                avg_deleted_punc_per_token=round(avg_punc, 4),
                new_types_from_previous_file=len(new_types),
            ))

            if on_file_done:
                on_file_done(self._lang)

        self._write(cumulative_freq)
        return reports

    # ------------------------------------------------------------------
    def _write(self, freq_dict: dict[str, int]) -> None:
        """Sort by frequency descending and write to the aggregated folder."""
        sorted_items = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / f"{self._lang}_{self._script_code}.csv"
        with open(out_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["word", "frequency"])
            writer.writerows(sorted_items)


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
    parser.add_argument(
        "--scripts-csv",
        type=Path,
        default=SCRIPTS_CSV,
        help=f"Path to scripts.csv (default: {SCRIPTS_CSV}).",
    )
    parser.add_argument(
        "--lang-overview",
        type=Path,
        default=LANGUAGE_OVERVIEW_CSV,
        help=f"Path to language_overview.csv (default: {LANGUAGE_OVERVIEW_CSV}).",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Re-clean already aggregated CSVs instead of aggregating from raw.",
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


def _repair_aggregated(agg_dir: Path, langs: list[str]) -> None:
    """Re-apply the cleaning pipeline to already-aggregated CSVs."""
    files = sorted(agg_dir.glob("*.csv"))
    if langs:
        selected = {l.lower() for l in langs}
        files = [f for f in files if f.stem in selected]

    if not files:
        print("No aggregated files found.")
        sys.exit(1)

    print(f"Repairing {len(files)} aggregated file(s) in {agg_dir}")
    print(f"{'=' * 50}")
    total_merged = 0
    for path in files:
        freq: dict[str, int] = {}
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            next(reader)  # skip header
            for row in reader:
                if len(row) < 2:
                    continue
                freq[row[0]] = freq.get(row[0], 0) + int(row[1])
        before = len(freq)
        freq = clean_freq_dict(freq)
        merged = before - len(freq)
        total_merged += merged
        sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["word", "frequency"])
            writer.writerows(sorted_items)
        print(f"  {path.stem}: {before:,} → {len(sorted_items):,} ({merged:,} cleaned)")

    print(f"\nDone. {total_merged:,} words cleaned total.")


def _print_progress(current: int, total: int, lang: str) -> None:
    """Print a single-line progress bar to stderr."""
    bar_len = 40
    frac = current / total if total else 1.0
    filled = int(bar_len * frac)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
    pct = frac * 100
    print(
        f"\r[{bar}] {current}/{total} ({pct:5.1f}%)  {lang:<5}",
        end="", file=sys.stderr, flush=True,
    )


def _write_report(reports: list[FileReport], report_dir: Path) -> None:
    """Write the per-file aggregation report CSV."""
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "aggregation_report.csv"
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "lng_shortcut", "file_name", "total_tokens",
            "total_tokens_after_ignoring", "total_types(rows)",
            "ignored_word_count", "avg_deleted_punc_per_token",
            "new_types_from_previous_file",
        ])
        for r in reports:
            writer.writerow([
                r.lng_shortcut, r.file_name, r.total_tokens,
                r.total_tokens_after_ignoring, r.total_types,
                r.ignored_word_count, r.avg_deleted_punc_per_token,
                r.new_types_from_previous_file,
            ])
    print(f"\nReport written to {out}", file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    if args.repair:
        _repair_aggregated(args.out_dir, args.langs)
        return

    # Load ignore list if provided
    ignored_files: set[str] = set()
    if args.ignore:
        if not args.ignore.exists():
            print(f"Ignore file not found: {args.ignore}", file=sys.stderr)
            sys.exit(1)
        ignored_files = _load_ignore_set(args.ignore)

    all_groups = _group_files_by_lang(args.raw_dir)

    # Filter out ignored files
    if ignored_files:
        all_groups = {
            lang: [f for f in files if f.name not in ignored_files]
            for lang, files in all_groups.items()
        }
        all_groups = {k: v for k, v in all_groups.items() if v}

    if not all_groups:
        print(f"No CSV files found in {args.raw_dir}", file=sys.stderr)
        sys.exit(1)

    # Load script mapping for output filenames
    lang_scripts: dict[str, str] = {}
    if args.scripts_csv.exists() and args.lang_overview.exists():
        script_codes = _load_script_codes(args.scripts_csv)
        lang_scripts = _load_lang_scripts(args.lang_overview, script_codes)
    elif args.lang_overview.exists():
        print("Warning: scripts.csv not found, using fallback script codes",
              file=sys.stderr)
    else:
        print("Warning: language_overview.csv not found, using fallback script codes",
              file=sys.stderr)

    # Determine which language codes to process
    if args.langs:
        selected = {l.lower() for l in args.langs}
        unknown = selected - set(all_groups)
        if unknown:
            print(
                f"Warning: no files for: {', '.join(sorted(unknown))}",
                file=sys.stderr,
            )
        groups = {k: v for k, v in all_groups.items() if k in selected}
    else:
        groups = all_groups

    if not groups:
        print("Nothing to process.", file=sys.stderr)
        sys.exit(1)

    total_files = sum(len(f) for f in groups.values())
    counter = [0]

    def on_file(lang: str) -> None:
        counter[0] += 1
        _print_progress(counter[0], total_files, lang)

    all_reports: list[FileReport] = []
    for lang in sorted(groups):
        files = groups[lang]
        script_code = lang_scripts.get(lang, FALLBACK_SCRIPT)
        aggregator = Aggregator(lang, files, args.out_dir, script_code)
        reports = aggregator.run(on_file_done=on_file)
        all_reports.extend(reports)

    _write_report(all_reports, REPORT_DIR)


if __name__ == "__main__":
    main()
