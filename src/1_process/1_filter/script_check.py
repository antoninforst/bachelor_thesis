"""
Check the dominant writing script of each raw frequency file.

For each file, samples words and classifies letters by Unicode script property.
Outputs a CSV reporting script composition per file.

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
from pathlib import Path
from typing import Optional

import regex
from tqdm import tqdm

csv.field_size_limit(10 * 1024 * 1024)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/0_raw")
OUTPUT_DIR = Path("results/1_process/1_filter")
SCRIPTS_CSV = Path("src/1_process/1_filter/scripts.csv")
WORD_STEP = 5
LETTER_STEP = 3
MAX_SAMPLED_WORDS = 10_000


# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------


class ScriptDetector:
    """Classifies characters by Unicode script property."""

    def __init__(self, scripts: list[str]):
        self.scripts = scripts
        self._patterns = {
            script: regex.compile(rf"\A\p{{Script={script}}}\Z")
            for script in scripts
        }
        self._cache: dict[str, str] = {}

    @classmethod
    def from_csv(cls, path: Path) -> "ScriptDetector":
        scripts: list[str] = []
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if len(row) == 1 and "," in row[0]:
                    row = row[0].split(",")
                if row and row[0].strip():
                    scripts.append(row[0].strip())
        return cls(scripts)

    def classify(self, ch: str) -> str:
        if ch in self._cache:
            return self._cache[ch]
        for script, pattern in self._patterns.items():
            if pattern.fullmatch(ch):
                self._cache[ch] = script
                return script
        self._cache[ch] = "Other"
        return "Other"


# ---------------------------------------------------------------------------
# File reading & analysis
# ---------------------------------------------------------------------------


def _group_files_by_lang(raw_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in sorted(raw_dir.glob("*.csv")):
        lang = path.name[:3]
        groups.setdefault(lang, []).append(path)
    return groups


def _iter_sampled_words(path: Path):
    """Yield every WORD_STEP-th word from a raw CSV."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for line in reader:
            if line and line[0].strip().lower() in ("item", "word"):
                break
        sampled = 0
        for index, row in enumerate(reader):
            if not row:
                continue
            if index % WORD_STEP == 0:
                yield row[0]
                sampled += 1
                if sampled >= MAX_SAMPLED_WORDS:
                    break


def _letter_script_counts(path: Path, detector: ScriptDetector) -> Counter:
    """Count letters by script using word and letter sampling."""
    counts: Counter = Counter()
    for word in _iter_sampled_words(path):
        for ch in word[::LETTER_STEP]:
            if ch.isalpha():
                counts[detector.classify(ch)] += 1
    return counts


@dataclass
class FileReport:
    filename: str
    lang: str
    total_letters: int
    dominant_script: str
    dominant_pct: float
    top_scripts: list[tuple[str, float]]
    script_pcts: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.script_pcts:
            self.script_pcts = {s: p for s, p in self.top_scripts}

    @property
    def other_pct(self) -> float:
        return self.script_pcts.get("Other", 0.0)


def _analyze_file(path: Path, lang: str, detector: ScriptDetector) -> FileReport:
    counts = _letter_script_counts(path, detector)
    total = sum(counts.values())

    if total == 0:
        return FileReport(
            filename=path.name, lang=lang, total_letters=0,
            dominant_script="(empty)", dominant_pct=0.0, top_scripts=[],
        )

    top_scripts = [
        (script, round(count / total * 100, 2))
        for script, count in counts.most_common()
    ]
    dominant_script, dominant_pct = top_scripts[0]

    return FileReport(
        filename=path.name,
        lang=lang,
        total_letters=total,
        dominant_script=dominant_script,
        dominant_pct=dominant_pct,
        top_scripts=top_scripts,
    )


# ---------------------------------------------------------------------------
# Language-level scripts & output
# ---------------------------------------------------------------------------


def _language_scripts(reports: list[FileReport]) -> dict[str, tuple[str, str]]:
    """Determine the top-2 scripts for each language across all its files."""
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


def _write_report(path: Path, reports: list[FileReport]) -> None:
    lang_scripts = _language_scripts(reports)
    script_columns = sorted({s for r in reports for s in r.script_pcts})

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
            lang_script, second_script = lang_scripts[report.lang]
            row = [
                report.filename, report.lang,
                lang_script, report.dominant_script,
                round(report.script_pcts.get(lang_script, 0.0), 2),
                round(report.script_pcts.get(report.dominant_script, 0.0), 2),
                round(report.script_pcts.get(second_script, 0.0), 2),
            ]
            for script in script_columns:
                row.append(round(report.script_pcts.get(script, 0.0), 2))
            writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_ignore_set(path: Path) -> set[str]:
    """Load filenames from a CSV with a 'filename' column."""
    ignored: set[str] = set()
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            ignored.add(row["filename"])
    return ignored


def _select_groups(
    raw_dir: Path, langs: list[str], ignored_files: set[str]
) -> dict[str, list[Path]]:
    groups = _group_files_by_lang(raw_dir)

    if ignored_files:
        groups = {
            lang: [p for p in files if p.name not in ignored_files]
            for lang, files in groups.items()
        }
        groups = {lang: files for lang, files in groups.items() if files}

    if not langs:
        return groups

    selected = {lang.lower() for lang in langs}
    unknown = selected - set(groups)
    if unknown:
        print(f"Warning: no files for: {', '.join(sorted(unknown))}")
    return {lang: files for lang, files in groups.items() if lang in selected}


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


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    if not args.scripts_csv.exists():
        print(f"Scripts CSV not found: {args.scripts_csv}")
        sys.exit(1)

    detector = ScriptDetector.from_csv(args.scripts_csv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "script_check.csv"

    ignored_files: set[str] = set()
    if args.ignore_csv:
        if not args.ignore_csv.exists():
            print(f"Ignore file not found: {args.ignore_csv}")
            sys.exit(1)
        ignored_files = _load_ignore_set(args.ignore_csv)

    groups = _select_groups(args.raw_dir, args.langs, ignored_files)
    if not groups:
        print(f"No CSV files found in {args.raw_dir}")
        sys.exit(1)

    all_files = [
        (lang, path)
        for lang in sorted(groups)
        for path in groups[lang]
    ]

    reports = [
        _analyze_file(path, lang, detector)
        for lang, path in tqdm(all_files, desc="Checking scripts", unit="file")
    ]

    _write_report(out_path, reports)
    print(f"Wrote {out_path} ({len(reports)} rows)")


if __name__ == "__main__":
    main()
