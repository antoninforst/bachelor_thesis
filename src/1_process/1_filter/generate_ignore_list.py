"""
Generate a list of raw files to ignore based on hapax overlap.

Reads results/1_process/1_filter/hapax_overlap.csv and for each pair where the
overlap share exceeds a threshold (default 25 %), marks the file with
the *bigger* share percentage (i.e. the smaller file) for ignoring.

Output: results/1_process/1_filter/ignored_files.csv

Usage:
    python src/1_process/1_filter/generate_ignore_list.py
    python src/1_process/1_filter/generate_ignore_list.py --threshold 30
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
DEFAULT_THRESHOLD = 33.3  # percent (hapax overlap)
DEFAULT_SCRIPT_THRESHOLD = 75.0  # percent (dominant script share)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _build_ignore_set(overlap_path: Path, threshold: float) -> dict[str, str]:
    """
    Return {filename: reason} for files that should be ignored.

    For each pair where max(share_a, share_b) > threshold, the file
    with the higher share (the smaller one) is added to the ignore set.
    """
    ignore: dict[str, str] = {}  # filename -> reason string

    with open(overlap_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            share_a = float(row["share_a_pct"])
            share_b = float(row["share_b_pct"])
            file_a = row["file_a"]
            file_b = row["file_b"]
            lang = row["lang"]

            max_share = max(share_a, share_b)
            if max_share <= threshold:
                continue

            # Ignore the file with the bigger share (= smaller file)
            if share_a >= share_b:
                victim = file_a
                other = file_b
                pct = share_a
            else:
                victim = file_b
                other = file_a
                pct = share_b

            # If the victim is already ignored, skip
            if victim in ignore:
                continue

            # If the other file is already ignored, this pair is resolved
            if other in ignore:
                continue

            reason = f"{pct:.1f}% hapax overlap with {other}"
            ignore[victim] = reason

    return ignore


def _read_manual_ignore(path: Path) -> dict[str, str]:
    """Read manual_ignore.txt and return {filename: reason}."""
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


_KNOWN_COLUMNS = {
    "file", "language",
    "language_script", "file_script",
    "language_script_share", "file_script_share",
    "second_language_script_share",
}


def _build_script_ignore_set(script_path: Path, threshold: float) -> dict[str, str]:
    """
    Return {filename: reason} for files whose language script share
    falls below *threshold*.
    """
    ignore: dict[str, str] = {}

    with open(script_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    for row in rows:
        share = float(row["language_script_share"])
        if share < threshold:
            filename = row["file"]
            file_script = row["file_script"]
            file_script_pct = float(row["file_script_share"])
            lang_script = row["language_script"]
            if file_script == lang_script:
                ignore[filename] = (
                    f"lang script {lang_script} only {share:.1f}% in file"
                    f" (need >= {threshold:.1f}%)"
                )
            else:
                ignore[filename] = (
                    f"file is {file_script_pct:.1f}% {file_script},"
                    f" but lang script is {lang_script}"
                    f" (only {share:.1f}% in file, need >= {threshold:.1f}%)"
                )

    return ignore


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ignore list from hapax overlap data."
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
        help=f"Share %% above which a file is ignored (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--script-csv", type=Path, default=SCRIPT_CSV,
        help=f"Path to script_check.csv (default: {SCRIPT_CSV}).",
    )
    parser.add_argument(
        "--script-threshold", type=float, default=None,
        help=(
            f"Dominant-script share below which a file is ignored "
            f"(default: {DEFAULT_SCRIPT_THRESHOLD}). "
            f"Pass 0 to disable script checking."
        ),
    )
    parser.add_argument(
        "--manual-ignore", type=Path, default=MANUAL_IGNORE,
        help=f"Path to manual_ignore.txt (default: {MANUAL_IGNORE}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    if not args.overlap_csv.exists():
        print(f"Overlap CSV not found: {args.overlap_csv}")
        sys.exit(1)

    # Manual ignore (always first)
    ignore = _read_manual_ignore(args.manual_ignore)
    if ignore:
        print(f"Files ignored (manual) : {len(ignore)}")
    elif args.manual_ignore.exists():
        print(f"Manual ignore file empty: {args.manual_ignore}")
    else:
        print(f"Manual ignore file not found: {args.manual_ignore}")

    # Hapax overlap
    overlap_ignore = _build_ignore_set(args.overlap_csv, args.threshold)
    new_overlap = 0
    for fname, reason in overlap_ignore.items():
        if fname not in ignore:
            ignore[fname] = reason
            new_overlap += 1

    print(f"Hapax-overlap threshold: {args.threshold}%")
    print(f"Files ignored (overlap): {new_overlap}")

    # Script-check mode
    script_threshold = (
        args.script_threshold
        if args.script_threshold is not None
        else DEFAULT_SCRIPT_THRESHOLD
    )
    if script_threshold > 0 and args.script_csv.exists():
        script_ignore = _build_script_ignore_set(args.script_csv, script_threshold)
        # Merge without overwriting existing reasons
        new_count = 0
        for fname, reason in script_ignore.items():
            if fname not in ignore:
                ignore[fname] = reason
                new_count += 1
        print(f"Script-share threshold : {script_threshold}%")
        print(f"Files ignored (script) : {new_count}")
    elif script_threshold > 0:
        print(f"Script CSV not found: {args.script_csv} -- skipping script check")

    print(f"Total files to ignore  : {len(ignore)}")
    print(f"{'=' * 60}")

    for filename in sorted(ignore):
        lang = filename[:3]
        print(f"  [{lang}] {filename}  -- {ignore[filename]}")

    # Write output
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "ignored_files.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "lang", "reason"])
        for filename in sorted(ignore):
            lang = filename[:3]
            writer.writerow([filename, lang, ignore[filename]])

    print(f"\n-> Wrote {out_path} ({len(ignore)} files)")
    print("Done.")


if __name__ == "__main__":
    main()
