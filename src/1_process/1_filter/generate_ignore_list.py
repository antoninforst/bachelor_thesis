"""
Generate a list of raw files to ignore based on hapax overlap and script check.

Reads hapax_overlap.csv and (optionally) script_check.csv to decide which files
should be excluded from aggregation. For overlapping pairs, the smaller file
(higher overlap share) is ignored. For script mismatches, files below the
dominant-script threshold are ignored.

Output: results/1_process/1_filter/ignored_files.csv

Usage:
    python src/1_process/1_filter/generate_ignore_list.py
    python src/1_process/1_filter/generate_ignore_list.py --script-csv results/1_process/1_filter/script_check.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OVERLAP_CSV = Path("results/1_process/1_filter/hapax_overlap.csv")
SCRIPT_CSV = Path("results/1_process/1_filter/script_check.csv")
MANUAL_IGNORE = Path("src/1_process/1_filter/manual_ignore.txt")
OUTPUT_DIR = Path("results/1_process/1_filter")
DEFAULT_THRESHOLD = 33.3
DEFAULT_SCRIPT_THRESHOLD = 75.0

# ---------------------------------------------------------------------------
# Ignore-set builders
# ---------------------------------------------------------------------------


def _build_overlap_ignore(overlap_path: Path, threshold: float) -> dict[str, str]:
    """
    For each pair where max(share_a, share_b) > threshold, ignore the file
    with the higher share (the smaller one).
    """
    ignore: dict[str, str] = {}

    with open(overlap_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            share_a = float(row["share_a_pct"])
            share_b = float(row["share_b_pct"])
            file_a = row["file_a"]
            file_b = row["file_b"]

            if max(share_a, share_b) <= threshold:
                continue

            # Ignore the file with the bigger share (= smaller file)
            if share_a >= share_b:
                victim, other, pct = file_a, file_b, share_a
            else:
                victim, other, pct = file_b, file_a, share_b

            if victim in ignore or other in ignore:
                continue

            ignore[victim] = f"{pct:.1f}% hapax overlap with {other}"

    return ignore


def _build_script_ignore(script_path: Path, threshold: float) -> dict[str, str]:
    """
    Ignore files whose language_script_share falls below threshold.
    """
    ignore: dict[str, str] = {}

    with open(script_path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            share = float(row["language_script_share"])
            if share >= threshold:
                continue

            filename = row["file"]
            file_script = row["file_script"]
            lang_script = row["language_script"]

            if file_script == lang_script:
                ignore[filename] = (
                    f"lang script {lang_script} only {share:.1f}% in file"
                    f" (need >= {threshold:.1f}%)"
                )
            else:
                file_script_pct = float(row["file_script_share"])
                ignore[filename] = (
                    f"file is {file_script_pct:.1f}% {file_script},"
                    f" but lang script is {lang_script}"
                    f" (only {share:.1f}% in file, need >= {threshold:.1f}%)"
                )

    return ignore


def _read_manual_ignore(path: Path) -> dict[str, str]:
    """Read manual_ignore.txt: one filename per line, # comments allowed."""
    ignore: dict[str, str] = {}
    if not path.exists():
        return ignore
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ignore[line] = "manual ignore"
    return ignore


# ---------------------------------------------------------------------------
# Merge & output
# ---------------------------------------------------------------------------


def _merge_ignore(base: dict[str, str], new: dict[str, str]) -> int:
    """Merge new entries into base without overwriting. Returns count of added."""
    added = 0
    for fname, reason in new.items():
        if fname not in base:
            base[fname] = reason
            added += 1
    return added


def _write_ignore_csv(out_path: Path, ignore: dict[str, str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "lang", "reason"])
        for filename in sorted(ignore):
            lang = filename[:3]
            writer.writerow([filename, lang, ignore[filename]])


def _print_summary(ignore: dict[str, str]) -> None:
    print(f"{'=' * 60}")
    for filename in sorted(ignore):
        lang = filename[:3]
        print(f"  [{lang}] {filename}  -- {ignore[filename]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ignore list from hapax overlap and script check data."
    )
    parser.add_argument(
        "--overlap-csv", type=Path, default=OVERLAP_CSV,
        help=f"Path to hapax_overlap.csv (default: {OVERLAP_CSV}).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Hapax overlap %% above which a file is ignored (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--script-csv", type=Path, default=None,
        help=f"Path to script_check.csv (default: {SCRIPT_CSV}).",
    )
    parser.add_argument(
        "--script-threshold", type=float, default=None,
        help=(
            f"Dominant-script share below which a file is ignored "
            f"(default: {DEFAULT_SCRIPT_THRESHOLD}). Pass 0 to disable."
        ),
    )
    parser.add_argument(
        "--manual-ignore", type=Path, default=MANUAL_IGNORE,
        help=f"Path to manual_ignore.txt (default: {MANUAL_IGNORE}).",
    )
    return parser.parse_args(argv)


def _collect_manual(args: argparse.Namespace) -> dict[str, str]:
    """Load manual ignores and report status."""
    ignore = _read_manual_ignore(args.manual_ignore)
    if ignore:
        print(f"Files ignored (manual) : {len(ignore)}")
    elif args.manual_ignore.exists():
        print(f"Manual ignore file empty: {args.manual_ignore}")
    else:
        print(f"Manual ignore file not found: {args.manual_ignore}")
    return ignore


def _collect_overlap(args: argparse.Namespace) -> dict[str, str]:
    """Build overlap-based ignore set."""
    print(f"Hapax-overlap threshold: {args.threshold}%")
    return _build_overlap_ignore(args.overlap_csv, args.threshold)


def _collect_script(args: argparse.Namespace) -> dict[str, str]:
    """Build script-based ignore set if --script-csv was given and exists."""
    script_threshold = (
        args.script_threshold
        if args.script_threshold is not None
        else DEFAULT_SCRIPT_THRESHOLD
    )
    if script_threshold <= 0 or not args.script_csv:
        return {}

    if not args.script_csv.exists():
        print(f"Script CSV not found: {args.script_csv} -- skipping script check")
        return {}

    print(f"Script-share threshold : {script_threshold}%")
    return _build_script_ignore(args.script_csv, script_threshold)


_COLLECTORS = [
    ("manual", _collect_manual),
    ("overlap", _collect_overlap),
    ("script", _collect_script),
]


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    if not args.overlap_csv.exists():
        print(f"Overlap CSV not found: {args.overlap_csv}")
        sys.exit(1)

    ignore: dict[str, str] = {}
    for label, collector in _COLLECTORS:
        n_added = _merge_ignore(ignore, collector(args))
        print(f"Files ignored ({label:>7}): {n_added}")

    print(f"Total files to ignore  : {len(ignore)}")
    _print_summary(ignore)

    out_path = args.out_dir / "ignored_files.csv"
    _write_ignore_csv(out_path, ignore)
    print(f"\n-> Wrote {out_path} ({len(ignore)} files)")
    print("Done.")


if __name__ == "__main__":
    main()
