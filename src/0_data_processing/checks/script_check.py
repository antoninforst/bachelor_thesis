"""
Check the dominant Unicode writing script of each raw frequency file.

For every file in data/0_raw/, reads all word *types* (ignoring frequency),
counts letters by Unicode script, and reports the dominant script.
A file is "good" if >= 95 % of its letters belong to a single script.

Produces a report highlighting:
  - files where no script reaches 95 % ("strange" files)
  - languages whose files disagree on the dominant script

Usage:
    python src/0_data_processing/checks/script_check.py
    python src/0_data_processing/checks/script_check.py afr ces
"""

import argparse
import bisect
import csv
import sys
import unicodedata

csv.field_size_limit(10 * 1024 * 1024)  # 10 MB – some Glot500 rows are very large
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/0_raw")
OUTPUT_DIR = Path("results/0_data_processing")
GOOD_THRESHOLD = 95.0  # percent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_files_by_lang(raw_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in sorted(raw_dir.glob("*.csv")):
        lang = path.name[:3]
        groups.setdefault(lang, []).append(path)
    return groups


def _read_words(path: Path) -> list[str]:
    """Read word types from a CSV (one entry per word, ignoring frequency)."""
    words: list[str] = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for line in reader:
            if line and line[0].strip().lower() in ("item", "word"):
                break
        for row in reader:
            if not row:
                continue
            words.append(row[0])
    return words


# Ranges for common scripts (start, end, script_name).
# fmt: off
_SCRIPT_RANGES_RAW = [
    (0x0000, 0x007F, "Latin"),
    (0x0080, 0x00FF, "Latin"),
    (0x0100, 0x024F, "Latin"),
    (0x0250, 0x02AF, "Latin"),
    (0x1E00, 0x1EFF, "Latin"),
    (0x2C60, 0x2C7F, "Latin"),
    (0xA720, 0xA7FF, "Latin"),
    (0xAB30, 0xAB6F, "Latin"),
    (0x0370, 0x03FF, "Greek"),
    (0x1F00, 0x1FFF, "Greek"),
    (0x0400, 0x04FF, "Cyrillic"),
    (0x0500, 0x052F, "Cyrillic"),
    (0x2DE0, 0x2DFF, "Cyrillic"),
    (0xA640, 0xA69F, "Cyrillic"),
    (0x0530, 0x058F, "Armenian"),
    (0xFB00, 0xFB06, "Armenian"),
    (0x0590, 0x05FF, "Hebrew"),
    (0xFB1D, 0xFB4F, "Hebrew"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0750, 0x077F, "Arabic"),
    (0x08A0, 0x08FF, "Arabic"),
    (0xFB50, 0xFDFF, "Arabic"),
    (0xFE70, 0xFEFF, "Arabic"),
    (0x0700, 0x074F, "Syriac"),
    (0x0780, 0x07BF, "Thaana"),
    (0x0900, 0x097F, "Devanagari"),
    (0xA8E0, 0xA8FF, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x0E80, 0x0EFF, "Lao"),
    (0x0F00, 0x0FFF, "Tibetan"),
    (0x1000, 0x109F, "Myanmar"),
    (0x10A0, 0x10FF, "Georgian"),
    (0x2D00, 0x2D2F, "Georgian"),
    (0x1100, 0x11FF, "Hangul"),
    (0x3130, 0x318F, "Hangul"),
    (0xAC00, 0xD7AF, "Hangul"),
    (0xD7B0, 0xD7FF, "Hangul"),
    (0x1200, 0x137F, "Ethiopic"),
    (0x1380, 0x139F, "Ethiopic"),
    (0x2D80, 0x2DDF, "Ethiopic"),
    (0xAB00, 0xAB2F, "Ethiopic"),
    (0x13A0, 0x13FF, "Cherokee"),
    (0x1400, 0x167F, "Canadian_Aboriginal"),
    (0x1680, 0x169F, "Ogham"),
    (0x16A0, 0x16FF, "Runic"),
    (0x1780, 0x17FF, "Khmer"),
    (0x1800, 0x18AF, "Mongolian"),
    (0x3040, 0x309F, "Hiragana"),
    (0x30A0, 0x30FF, "Katakana"),
    (0x31F0, 0x31FF, "Katakana"),
    (0x4E00, 0x9FFF, "Han"),
    (0x3400, 0x4DBF, "Han"),
    (0x20000, 0x2A6DF, "Han"),
    (0x2A700, 0x2B73F, "Han"),
    (0xF900, 0xFAFF, "Han"),
    (0x2F00, 0x2FDF, "Han"),
    (0x1A00, 0x1A1F, "Buginese"),
    (0x1B00, 0x1B7F, "Balinese"),
    (0xA000, 0xA4CF, "Yi"),
    (0x1950, 0x197F, "Tai_Le"),
    (0x1980, 0x19DF, "New_Tai_Lue"),
    (0x2D30, 0x2D7F, "Tifinagh"),
    (0xA500, 0xA63F, "Vai"),
    (0x10300, 0x1032F, "Old_Italic"),
    (0x10330, 0x1034F, "Gothic"),
    (0x10400, 0x1044F, "Deseret"),
    (0x10800, 0x1083F, "Cypriot"),
]
# fmt: on

# Build sorted, non-overlapping lookup lists for bisect.
_SCRIPT_RANGES_RAW.sort(key=lambda r: r[0])
_RANGE_STARTS = [s for s, _, _ in _SCRIPT_RANGES_RAW]
_RANGE_ENDS = [e for _, e, _ in _SCRIPT_RANGES_RAW]
_RANGE_SCRIPTS = [sc for _, _, sc in _SCRIPT_RANGES_RAW]


@lru_cache(maxsize=8192)
def _script_name(cp: int) -> str:
    """Return the script name for codepoint *cp* using binary search."""
    idx = bisect.bisect_right(_RANGE_STARTS, cp) - 1
    if idx >= 0 and _RANGE_STARTS[idx] <= cp <= _RANGE_ENDS[idx]:
        return _RANGE_SCRIPTS[idx]
    return "Unknown"


def _letter_script_counts(words: list[str]) -> Counter:
    """Count letters by script (sampling every 5th word, first 3 letters)."""
    counts: Counter = Counter()
    for word in words[::5]:
        for ch in word[:3]:
            if ch.isalpha():
                counts[_script_name(ord(ch))] += 1
    return counts


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class FileReport:
    filename: str
    lang: str
    total_letters: int
    dominant_script: str
    dominant_pct: float
    is_good: bool
    top_scripts: list[tuple[str, float]]  # [(script, pct), …]
    script_pcts: dict[str, float] = None  # {script: pct} for all scripts

    def __post_init__(self):
        if self.script_pcts is None:
            self.script_pcts = {s: p for s, p in self.top_scripts}


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _analyze_file(args: tuple[Path, str]) -> FileReport:
    path, lang = args
    words = _read_words(path)
    counts = _letter_script_counts(words)
    total = sum(counts.values())
    if total == 0:
        return FileReport(
            filename=path.name, lang=lang, total_letters=0,
            dominant_script="(empty)", dominant_pct=0.0, is_good=False,
            top_scripts=[],
        )
    top = counts.most_common()
    top_with_pct = [(s, c / total * 100) for s, c in top]
    dominant_script, dominant_pct = top_with_pct[0]
    return FileReport(
        filename=path.name,
        lang=lang,
        total_letters=total,
        dominant_script=dominant_script,
        dominant_pct=round(dominant_pct, 2),
        is_good=dominant_pct >= GOOD_THRESHOLD,
        top_scripts=[(s, round(p, 2)) for s, p in top_with_pct],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check dominant writing script of raw frequency files."
    )
    parser.add_argument(
        "langs", nargs="*", metavar="LANG",
        help="Three-letter language codes to process (default: all).",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW_DIR,
        help=f"Directory with raw CSV files (default: {RAW_DIR}).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--threshold", type=float, default=GOOD_THRESHOLD,
        help=f"Percent threshold for 'good' (default: {GOOD_THRESHOLD}).",
    )
    parser.add_argument(
        "--ignore", type=Path, default=None,
        help="Path to ignored_files.csv to exclude files from analysis.",
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
    threshold = args.threshold

    # Check output directory is writable before doing any work
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "script_check.csv"
    try:
        with open(out_path, "w", encoding="utf-8", newline="") as fh:
            pass  # just test we can open for writing
    except PermissionError:
        print(f"ERROR: Cannot write to {out_path} – is the file open?")
        sys.exit(1)

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

    if args.langs:
        selected = {l.lower() for l in args.langs}
        unknown = selected - set(all_groups)
        if unknown:
            print(f"Warning: no files found for language code(s): "
                  f"{', '.join(sorted(unknown))}")
        groups = {k: v for k, v in all_groups.items() if k in selected}
    else:
        groups = all_groups

    if not groups:
        print("Nothing to process.")
        sys.exit(1)

    print(f"Languages to process: {', '.join(sorted(groups))}")
    print(f"Threshold: {threshold}%")
    print(f"{'=' * 70}")

    # Open CSV for continuous progress saving (simple format)
    progress_path = args.out_dir / "script_check_progress.csv"
    progress_fh = open(progress_path, "w", encoding="utf-8", newline="")
    progress_writer = csv.writer(progress_fh)
    progress_writer.writerow(["file", "language", "dominant_script", "dominant_pct", "top_scripts"])

    all_reports: list[FileReport] = []
    strange_files: list[FileReport] = []
    mixed_langs: list[tuple[str, dict[str, list[str]]]] = []

    for lang in sorted(groups):
        files = groups[lang]
        print(f"\n[{lang}] {len(files)} file(s):")
        lang_reports: list[FileReport] = []

        for path in files:
            r = _analyze_file((path, lang))
            r.is_good = r.dominant_pct >= threshold
            lang_reports.append(r)
            all_reports.append(r)

            mark = "OK" if r.is_good else "!!"
            print(f"  [{mark}] {r.filename}: {r.dominant_script} "
                  f"({r.dominant_pct:.1f}%)")
            if not r.is_good:
                for script, pct in r.top_scripts[:4]:
                    print(f"        {script}: {pct:.1f}%")
                strange_files.append(r)

            # Save progress immediately
            scripts_str = "; ".join(f"{s} {p:.1f}%" for s, p in r.top_scripts[:4])
            progress_writer.writerow([r.filename, r.lang, r.dominant_script, r.dominant_pct, scripts_str])
            progress_fh.flush()

        # Check if all files in this language agree on dominant script
        scripts_in_lang: dict[str, list[str]] = {}
        for r in lang_reports:
            scripts_in_lang.setdefault(r.dominant_script, []).append(r.filename)
        if len(scripts_in_lang) > 1:
            mixed_langs.append((lang, scripts_in_lang))
            print(f"  ⚠ Mixed scripts in {lang}:")
            for script, fnames in sorted(scripts_in_lang.items()):
                print(f"      {script}: {len(fnames)} file(s)")
        else:
            script = next(iter(scripts_in_lang))
            print(f"  All files: {script}")

    progress_fh.close()

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total files analyzed: {len(all_reports)}")
    print(f"Strange files (< {threshold}% single script): {len(strange_files)}")
    print(f"Languages with mixed scripts: {len(mixed_langs)}")

    if strange_files:
        print(f"\n--- Strange files ---")
        for r in strange_files:
            scripts_str = ", ".join(f"{s} {p:.1f}%" for s, p in r.top_scripts[:4])
            print(f"  {r.filename}  →  {scripts_str}")

    if mixed_langs:
        print(f"\n--- Languages with mixed dominant scripts ---")
        for lang, scripts_in_lang in mixed_langs:
            parts = []
            for script, fnames in sorted(scripts_in_lang.items()):
                parts.append(f"{script}({len(fnames)})")
            print(f"  {lang}: {', '.join(parts)}")

    # ------------------------------------------------------------------
    # Determine dominant script per language (by total letter count)
    # ------------------------------------------------------------------
    lang_script_totals: dict[str, Counter] = {}
    for r in all_reports:
        if r.lang not in lang_script_totals:
            lang_script_totals[r.lang] = Counter()
        for script, pct in r.top_scripts:
            lang_script_totals[r.lang][script] += r.total_letters * pct / 100

    lang_dominant: dict[str, tuple[str, str]] = {}  # lang -> (1st, 2nd)
    for lang, totals in lang_script_totals.items():
        top2 = totals.most_common(2)
        first = top2[0][0] if len(top2) >= 1 else ""
        second = top2[1][0] if len(top2) >= 2 else ""
        lang_dominant[lang] = (first, second)

    # Collect all script names across all reports
    all_scripts: set[str] = set()
    for r in all_reports:
        all_scripts.update(r.script_pcts.keys())
    all_scripts.discard("Unknown")
    sorted_scripts = sorted(all_scripts)

    # ------------------------------------------------------------------
    # Write final CSV with full columns
    # ------------------------------------------------------------------
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        header = [
            "file", "language",
            "dominant_script_share", "second_script_share",
        ] + sorted_scripts
        writer.writerow(header)
        for r in all_reports:
            first_script, second_script = lang_dominant[r.lang]
            dominant_share = r.script_pcts.get(first_script, 0.0)
            second_share = r.script_pcts.get(second_script, 0.0)
            row = [
                r.filename, r.lang,
                round(dominant_share, 2), round(second_share, 2),
            ]
            for script in sorted_scripts:
                row.append(round(r.script_pcts.get(script, 0.0), 2))
            writer.writerow(row)

    # Clean up progress file
    progress_path.unlink(missing_ok=True)

    print(f"\n→ Wrote {out_path} ({len(all_reports)} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
