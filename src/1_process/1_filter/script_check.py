"""
Check the main writing script of each raw frequency file.

Usage:
    python src/1_process/1_filter/script_check.py
    python src/1_process/1_filter/script_check.py ces eng
    python src/1_process/1_filter/script_check.py --ignore-csv results/1_process/1_filter/ignored_files.csv
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Optional

import regex
from tqdm import tqdm

csv.field_size_limit(10 * 1024 * 1024)

RAW_DIR = Path("data/0_raw")
OUTPUT_DIR = Path("results/1_process/1_filter")
SCRIPTS_CSV = Path("src/1_process/1_filter/scripts.csv")
DEFAULT_THRESHOLD = 95.0
OTHER_THRESHOLD = 20.0
WORD_STEP = 5
LETTER_STEP = 3
MAX_SAMPLED_WORDS = 10_000


def _load_scripts(path: Path) -> list[str]:
    """Read script names from scripts.csv (first column)."""
    scripts: list[str] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) == 1 and "," in row[0]:
                row = row[0].split(",")
            if row and row[0].strip():
                scripts.append(row[0].strip())
    return scripts


SCRIPTS: list[str] = []  # populated in main()
SCRIPT_PATTERNS: dict[str, regex.Pattern] = {}  # populated in main()


def _init_scripts(scripts_csv: Path) -> None:
    """Load scripts from CSV and build regex patterns."""
    global SCRIPTS, SCRIPT_PATTERNS
    SCRIPTS = _load_scripts(scripts_csv)
    SCRIPT_PATTERNS = {
        script: regex.compile(rf"\A\p{{Script={script}}}\Z")
        for script in SCRIPTS
    }


def _group_files_by_lang(raw_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in sorted(raw_dir.glob("*.csv")):
        lang = path.name[:3]
        groups.setdefault(lang, []).append(path)
    return groups


def _iter_sampled_words(path: Path):
    """Read every configured word type from a CSV, ignoring frequency."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for line in reader:
            if line and line[0].strip().lower() in ("item", "word"):
                break
        sampled_words = 0
        for index, row in enumerate(reader):
            if not row:
                continue
            if index % WORD_STEP == 0:
                yield row[0]
                sampled_words += 1
                if sampled_words >= MAX_SAMPLED_WORDS:
                    break


@cache
def _script_name(ch: str) -> str:
    for script, pattern in SCRIPT_PATTERNS.items():
        if pattern.fullmatch(ch):
            return script
    return "Other"


def _letter_script_counts(path: Path) -> Counter:
    """Count letters by script using the configured word and letter sampling."""
    counts: Counter = Counter()
    for word in _iter_sampled_words(path):
        for ch in word[::LETTER_STEP]:
            if ch.isalpha():
                counts[_script_name(ch)] += 1
    return counts


@dataclass
class FileReport:
    filename: str
    lang: str
    total_letters: int
    dominant_script: str
    dominant_pct: float
    is_good: bool
    top_scripts: list[tuple[str, float]]
    script_pcts: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.script_pcts:
            self.script_pcts = {s: p for s, p in self.top_scripts}

    @property
    def other_pct(self) -> float:
        return self.script_pcts.get("Other", 0.0)


def _percentages(counts: Counter) -> tuple[int, list[tuple[str, float]]]:
    total = sum(counts.values())
    if total == 0:
        return 0, []
    return total, [(script, round(count / total * 100, 2))
                   for script, count in counts.most_common()]


def _analyze_file(path: Path, lang: str, threshold: float) -> FileReport:
    total, top_scripts = _percentages(_letter_script_counts(path))
    if total == 0:
        return FileReport(
            filename=path.name, lang=lang, total_letters=0,
            dominant_script="(empty)", dominant_pct=0.0, is_good=False,
            top_scripts=[],
        )
    dominant_script, dominant_pct = top_scripts[0]
    return FileReport(
        filename=path.name,
        lang=lang,
        total_letters=total,
        dominant_script=dominant_script,
        dominant_pct=dominant_pct,
        is_good=dominant_pct >= threshold,
        top_scripts=top_scripts,
    )


def _select_groups(
    raw_dir: Path,
    langs: list[str],
    ignored_files: set[str],
) -> dict[str, list[Path]]:
    groups = _group_files_by_lang(raw_dir)
    if ignored_files:
        groups = {
            lang: [path for path in files if path.name not in ignored_files]
            for lang, files in groups.items()
        }
        groups = {lang: files for lang, files in groups.items() if files}

    if not langs:
        return groups

    selected = {lang.lower() for lang in langs}
    unknown = selected - set(groups)
    if unknown:
        print(f"Warning: no files found for language code(s): "
              f"{', '.join(sorted(unknown))}")
    return {lang: files for lang, files in groups.items() if lang in selected}


def _language_scripts(reports: list[FileReport]) -> dict[str, tuple[str, str]]:
    totals: dict[str, defaultdict[str, float]] = {}
    for report in reports:
        totals.setdefault(report.lang, defaultdict(float))
        for script, pct in report.top_scripts:
            totals[report.lang][script] += report.total_letters * pct / 100

    result: dict[str, tuple[str, str]] = {}
    for lang, script_totals in totals.items():
        top = sorted(script_totals.items(), key=lambda item: item[1], reverse=True)
        first = top[0][0] if top else ""
        second = top[1][0] if len(top) > 1 else ""
        result[lang] = (first, second)
    return result


def _script_columns(reports: list[FileReport]) -> list[str]:
    scripts: set[str] = set()
    for report in reports:
        scripts.update(report.script_pcts)
    return sorted(scripts)


def _write_report(path: Path, reports: list[FileReport]) -> None:
    language_scripts = _language_scripts(reports)
    script_columns = _script_columns(reports)
    header = [
        "file", "language",
        "language_script", "file_script",
        "language_script_share", "file_script_share",
        "second_language_script_share",
    ] + script_columns

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for report in reports:
            language_script, second_script = language_scripts[report.lang]
            row = [
                report.filename, report.lang,
                language_script, report.dominant_script,
                round(report.script_pcts.get(language_script, 0.0), 2),
                round(report.script_pcts.get(report.dominant_script, 0.0), 2),
                round(report.script_pcts.get(second_script, 0.0), 2),
            ]
            for script in script_columns:
                row.append(round(report.script_pcts.get(script, 0.0), 2))
            writer.writerow(row)


def _iter_files(groups: dict[str, list[Path]]) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for lang in sorted(groups):
        for path in groups[lang]:
            files.append((lang, path))
    return files


def _print_summary(reports: list[FileReport], strange_files: list[FileReport]) -> None:
    mixed_langs = 0
    by_lang: dict[str, set[str]] = defaultdict(set)
    for report in reports:
        by_lang[report.lang].add(report.dominant_script)
    for scripts in by_lang.values():
        if len(scripts) > 1:
            mixed_langs += 1

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total files analyzed: {len(reports)}")
    print(f"Strange files (< {DEFAULT_THRESHOLD}% single script): {len(strange_files)}")
    print(f"Languages with mixed scripts: {mixed_langs}")

    other_files = [report for report in reports if report.other_pct >= OTHER_THRESHOLD]
    print(f"Files with >= {OTHER_THRESHOLD}% Other script letters: {len(other_files)}")

    if strange_files:
        print("\n--- Strange files ---")
        for report in strange_files:
            scripts = ", ".join(f"{s} {p:.1f}%" for s, p in report.top_scripts[:4])
            print(f"  {report.filename}  ->  {scripts}")

    if other_files:
        print("\n--- Files with many Other script letters ---")
        for report in other_files:
            scripts = ", ".join(f"{s} {p:.1f}%" for s, p in report.top_scripts[:4])
            print(f"  {report.filename}  ->  {scripts}")


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
        "--ignore-csv", "--ignore", type=Path, default=None,
        help="Path to ignored_files.csv to exclude files from analysis.",
    )
    parser.add_argument(
        "--scripts-csv", type=Path, default=SCRIPTS_CSV,
        help=f"Path to scripts.csv (default: {SCRIPTS_CSV}).",
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
    threshold = DEFAULT_THRESHOLD

    if not args.scripts_csv.exists():
        print(f"Scripts CSV not found: {args.scripts_csv}")
        sys.exit(1)
    _init_scripts(args.scripts_csv)
    print(f"Loaded {len(SCRIPTS)} scripts from {args.scripts_csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "script_check.csv"
    try:
        with open(out_path, "w", encoding="utf-8", newline=""):
            pass
    except PermissionError:
        print(f"ERROR: Cannot write to {out_path}; is the file open?")
        sys.exit(1)

    ignored_files: set[str] = set()
    if args.ignore_csv:
        if not args.ignore_csv.exists():
            print(f"Ignore file not found: {args.ignore_csv}")
            sys.exit(1)
        ignored_files = _load_ignore_set(args.ignore_csv)
        print(f"Ignoring {len(ignored_files)} file(s) from {args.ignore_csv}")

    groups = _select_groups(args.raw_dir, args.langs, ignored_files)
    if not groups:
        print(f"No CSV files found in {args.raw_dir}")
        sys.exit(1)

    print(f"Threshold: {threshold}%")
    print(f"Other-script warning: {OTHER_THRESHOLD}%")
    print(f"Sampling: every {WORD_STEP}th word, every {LETTER_STEP}rd letter")
    print(f"Maximum sampled words per file: {MAX_SAMPLED_WORDS}")

    all_reports: list[FileReport] = []
    strange_files: list[FileReport] = []

    for lang, path in tqdm(_iter_files(groups), desc="Checking scripts", unit="file"):
        report = _analyze_file(path, lang, threshold)
        all_reports.append(report)
        if not report.is_good:
            strange_files.append(report)

    _print_summary(all_reports, strange_files)
    _write_report(out_path, all_reports)

    print(f"\nWrote {out_path} ({len(all_reports)} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
