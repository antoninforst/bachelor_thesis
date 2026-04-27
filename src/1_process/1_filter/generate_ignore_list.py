"""
Create ignored_files.csv from hapax overlap, with optional script-check additions.

Usage:
    python src/1_process/1_filter/generate_ignore_list.py
    python src/1_process/1_filter/generate_ignore_list.py --script-csv results/1_process/1_filter/script_check.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

OVERLAP_CSV = Path("results/1_process/1_filter/hapax_overlap.csv")
DEFAULT_SCRIPT_CSV = Path("results/1_process/1_filter/script_check.csv")
OUTPUT_DIR = Path("results/1_process/1_filter")
DEFAULT_THRESHOLD = 33.3
DEFAULT_SCRIPT_THRESHOLD = 75.0


def _build_hapax_ignores(overlap_csv: Path, threshold: float) -> dict[str, str]:
    """Return files ignored because one file mostly repeats another file's hapaxes."""
    ignored: dict[str, str] = {}

    with open(overlap_csv, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            share_a = float(row["share_a_pct"])
            share_b = float(row["share_b_pct"])
            if max(share_a, share_b) <= threshold:
                continue

            if share_a >= share_b:
                filename = row["file_a"]
                other = row["file_b"]
                pct = share_a
            else:
                filename = row["file_b"]
                other = row["file_a"]
                pct = share_b

            if filename in ignored or other in ignored:
                continue

            ignored[filename] = f"{pct:.1f}% hapax overlap with {other}"

    return ignored


_SCRIPT_META_COLUMNS = {
    "file",
    "language",
    "language_script",
    "file_script",
    "language_script_share",
    "file_script_share",
    "second_language_script_share",
    "dominant_script_share",
    "second_script_share",
}


def _build_script_ignores(script_csv: Path, threshold: float) -> dict[str, str]:
    """Return files ignored because they poorly match the language's script."""
    ignored: dict[str, str] = {}

    with open(script_csv, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        script_columns = [
            column for column in reader.fieldnames or []
            if column not in _SCRIPT_META_COLUMNS
        ]
        rows = list(reader)

    for row in rows:
        language_share = float(
            row.get("language_script_share") or row.get("dominant_script_share") or 0
        )
        if language_share >= threshold:
            continue

        filename = row["file"]
        file_script = row.get("file_script") or max(
            script_columns, key=lambda column: float(row.get(column) or 0)
        )
        file_share = float(row.get("file_script_share") or row.get(file_script) or 0)
        language_script = row.get("language_script") or min(
            script_columns,
            key=lambda column: abs(float(row.get(column) or 0) - language_share),
        )

        if file_script == language_script:
            ignored[filename] = (
                f"{language_script} only {language_share:.1f}% "
                f"(need >= {threshold:.1f}%)"
            )
        else:
            ignored[filename] = (
                f"file is {file_share:.1f}% {file_script}, "
                f"language script {language_script} is {language_share:.1f}% "
                f"(need >= {threshold:.1f}%)"
            )

    return ignored


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ignored_files.csv from hapax overlap and optional script checks."
    )
    parser.add_argument(
        "--overlap-csv",
        type=Path,
        default=OVERLAP_CSV,
        help=f"Path to hapax_overlap.csv (default: {OVERLAP_CSV}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Hapax-overlap percent above which a file is ignored (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--script-csv",
        type=Path,
        default=None,
        help=f"Add script-based ignores from this CSV, usually {DEFAULT_SCRIPT_CSV}.",
    )
    parser.add_argument(
        "--script-threshold",
        type=float,
        default=DEFAULT_SCRIPT_THRESHOLD,
        help=(
            "Language-script percent below which a file is ignored "
            f"(default: {DEFAULT_SCRIPT_THRESHOLD})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    if not args.overlap_csv.exists():
        print(f"Overlap CSV not found: {args.overlap_csv}")
        sys.exit(1)

    ignored = _build_hapax_ignores(args.overlap_csv, args.threshold)

    print(f"Creating ignored_files.csv from {args.overlap_csv}")
    print(f"Hapax threshold: {args.threshold}%")
    print(f"Added from hapax overlap: {len(ignored)}")

    if args.script_csv and not args.script_csv.exists():
        print(f"Script CSV not found: {args.script_csv}; skipping script check")
    elif args.script_csv:
        script_ignored = _build_script_ignores(args.script_csv, args.script_threshold)
        added = 0
        skipped = 0
        for filename, reason in script_ignored.items():
            if filename in ignored:
                skipped += 1
            else:
                ignored[filename] = reason
                added += 1
        print(f"Adding script ignores from {args.script_csv}")
        print(f"Script threshold: {args.script_threshold}%")
        print(f"Added from script check: {added}")
        print(f"Already ignored: {skipped}")
    else:
        print("No script CSV passed; using hapax overlap only.")

    print(f"Total files to ignore: {len(ignored)}")
    print(f"{'=' * 60}")

    for filename in sorted(ignored):
        lang = filename[:3]
        print(f"  [{lang}] {filename} - {ignored[filename]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "ignored_files.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "lang", "reason"])
        for filename in sorted(ignored):
            lang = filename[:3]
            writer.writerow([filename, lang, ignored[filename]])

    print(f"\nWrote {out_path} ({len(ignored)} files)")
    print("Done.")


if __name__ == "__main__":
    main()
