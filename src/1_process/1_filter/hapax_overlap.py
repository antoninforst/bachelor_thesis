"""
Compute hapax-legomenon overlap between pairs of raw frequency files.

For each language (grouped by the first three characters of the filename),
extracts hapax legomena (frequency == 1) from each file, then computes
the pairwise overlap percentage in both directions.

Output CSV columns:
    lang, file_a, file_b, hapaxes_a, hapaxes_b, overlap,
    share_a_pct, share_b_pct

If a language has many file-pairs, only the 16 pairs with the highest
max(share_a, share_b) are kept.

Usage:
    python src/1_process/1_filter/hapax_overlap.py            # all languages
    python src/1_process/1_filter/hapax_overlap.py ces eng    # selected languages
"""

import argparse
import csv
import itertools
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

csv.field_size_limit(10 * 1024 * 1024)

RAW_DIR = Path("data/0_raw")
OUTPUT_DIR = Path("results/1_process/1_filter")
MAX_PAIRS_PER_LANG = 16

_HAS_LETTER = re.compile(r"[^\W\d_]").search


def _contains_letter(word: str) -> bool:
    return bool(_HAS_LETTER(word))


def _group_files_by_lang(raw_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in sorted(raw_dir.glob("*.csv")):
        lang = path.name[:3]
        groups.setdefault(lang, []).append(path)
    return groups


def _read_hapaxes(path: Path) -> set[str]:
    """Return the set of hapax legomena (words with frequency == 1) from a CSV."""
    hapaxes: set[str] = set()
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for line in reader:
            if line and line[0].strip().lower() in ("item", "word"):
                break
        for row in reader:
            if len(row) < 2:
                continue
            word = row[0]
            try:
                freq = int(row[1])
            except (ValueError, IndexError):
                continue
            if freq == 1 and _contains_letter(word):
                hapaxes.add(word)
    return hapaxes


@dataclass
class PairResult:
    lang: str
    file_a: str
    file_b: str
    hapaxes_a: int
    hapaxes_b: int
    overlap: int
    share_a_pct: float  # overlap / hapaxes_a * 100
    share_b_pct: float  # overlap / hapaxes_b * 100


def _compute_pairs(lang: str, files: list[Path]) -> list[PairResult]:
    """Compute hapax overlap for every unordered pair of files."""
    hapax_map: dict[str, set[str]] = {}
    for path in files:
        hapax_map[path.name] = _read_hapaxes(path)

    results: list[PairResult] = []
    for (name_a, set_a), (name_b, set_b) in itertools.combinations(
        hapax_map.items(), 2
    ):
        overlap = set_a & set_b
        n_overlap = len(overlap)
        n_a = len(set_a)
        n_b = len(set_b)
        share_a = (n_overlap / n_a * 100) if n_a else 0.0
        share_b = (n_overlap / n_b * 100) if n_b else 0.0
        results.append(
            PairResult(
                lang=lang,
                file_a=name_a,
                file_b=name_b,
                hapaxes_a=n_a,
                hapaxes_b=n_b,
                overlap=n_overlap,
                share_a_pct=round(share_a, 2),
                share_b_pct=round(share_b, 2),
            )
        )

    if len(results) > MAX_PAIRS_PER_LANG:
        results.sort(key=lambda r: max(r.share_a_pct, r.share_b_pct), reverse=True)
        results = results[:MAX_PAIRS_PER_LANG]

    return results


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute hapax-legomenon overlap between raw frequency files."
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
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=MAX_PAIRS_PER_LANG,
        help=f"Max pairs per language to keep (default: {MAX_PAIRS_PER_LANG}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    global MAX_PAIRS_PER_LANG
    MAX_PAIRS_PER_LANG = args.max_pairs

    all_groups = _group_files_by_lang(args.raw_dir)

    if not all_groups:
        print(f"No CSV files found in {args.raw_dir}")
        sys.exit(1)

    if args.langs:
        selected = {lang_code.lower() for lang_code in args.langs}
        unknown = selected - set(all_groups)
        if unknown:
            print(f"Warning: no files found for language code(s): "
                  f"{', '.join(sorted(unknown))}")
        groups = {k: v for k, v in all_groups.items() if k in selected}
    else:
        groups = all_groups

    groups = {k: v for k, v in groups.items() if len(v) >= 2}

    if not groups:
        print("No languages with at least 2 files to compare.")
        sys.exit(1)

    all_results: list[PairResult] = []

    for lang in tqdm(sorted(groups), desc="Checking hapax overlap", unit="lang"):
        files = groups[lang]
        results = _compute_pairs(lang, files)
        all_results.extend(results)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "hapax_overlap.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "lang", "file_a", "file_b",
            "hapaxes_a", "hapaxes_b", "overlap",
            "share_a_pct", "share_b_pct",
        ])
        for r in all_results:
            writer.writerow([
                r.lang, r.file_a, r.file_b,
                r.hapaxes_a, r.hapaxes_b, r.overlap,
                r.share_a_pct, r.share_b_pct,
            ])
    print(f"\nWrote {out_path} ({len(all_results):,} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
