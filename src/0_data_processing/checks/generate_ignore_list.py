"""
Generate a list of raw files to ignore based on hapax overlap.

Reads results/0_data_processing/hapax_overlap.csv and for each pair where the
overlap share exceeds a threshold (default 25 %), marks the file with
the *bigger* share percentage (i.e. the smaller file) for ignoring.

Output: results/0_data_processing/ignored_files.csv

Usage:
    python src/0_data_processing/checks/generate_ignore_list.py
    python src/0_data_processing/checks/generate_ignore_list.py --threshold 30
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OVERLAP_CSV = Path("results/0_data_processing/hapax_overlap.csv")
OUTPUT_DIR = Path("results/0_data_processing")
DEFAULT_THRESHOLD = 33.3  # percent


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
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    if not args.overlap_csv.exists():
        print(f"Overlap CSV not found: {args.overlap_csv}")
        sys.exit(1)

    ignore = _build_ignore_set(args.overlap_csv, args.threshold)

    print(f"Threshold: {args.threshold}%")
    print(f"Files to ignore: {len(ignore)}")
    print(f"{'=' * 60}")

    for filename in sorted(ignore):
        lang = filename[:3]
        print(f"  [{lang}] {filename}  — {ignore[filename]}")

    # Write output
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "ignored_files.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "lang", "reason"])
        for filename in sorted(ignore):
            lang = filename[:3]
            writer.writerow([filename, lang, ignore[filename]])

    print(f"\n→ Wrote {out_path} ({len(ignore)} files)")
    print("Done.")


if __name__ == "__main__":
    main()
