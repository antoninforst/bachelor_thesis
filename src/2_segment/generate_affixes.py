#!/usr/bin/env python3
"""Generate prefix and suffix log-frequency files from aggregated word frequency CSVs.

For each word in the frequency file, every possible prefix (1..len-1) and suffix
(1..len-1) accumulates log1p(word_freq).  This favours affixes that appear in
many distinct words rather than affixes dominated by a single high-frequency word.
Output: affix,score  sorted descending.

Usage:
    python generate_affixes.py -freq ../../data/1_aggregated/deu.csv \
                               -out_dir ../../data/1_aggregated

    Produces:  deu.prep  (prefix → log-freq)
               deu.post  (suffix → log-freq)

    python generate_affixes.py -freq_dir ../../data/1_aggregated   # all languages
"""

import argparse
import csv
import glob
import math
import os
import sys
from collections import defaultdict


def build_affix_freqs(
    freq_path: str,
    max_len: int = 12,
    min_freq: int = 10,
) -> tuple:
    """Read a frequency CSV and accumulate log1p(freq) per affix.

    For every word with frequency f, each of its prefixes/suffixes gets
    log1p(f) added to its score.  This means an affix occurring in many
    different words scores higher than one boosted by a single frequent word.

    Only affixes whose accumulated score >= *min_freq* are kept.
    Returns (prefix_scores, suffix_scores) as dicts {string: float}.
    """
    prefix_scores: dict[str, float] = defaultdict(float)
    suffix_scores: dict[str, float] = defaultdict(float)

    with open(freq_path, "r", encoding="utf-8") as f:
        next(f, None)  # skip header
        for line in f:
            sep = line.rfind(",")
            if sep < 0:
                continue
            word = line[:sep].strip().lower()
            try:
                freq = int(line[sep + 1:])
            except ValueError:
                continue
            if len(word) < 2 or freq <= 0 or not word.isalpha():
                continue

            log_freq = math.log1p(freq)
            limit = min(len(word), max_len)
            for k in range(1, limit):
                prefix_scores[word[:k]] += log_freq
                suffix_scores[word[-k:]] += log_freq

    # Filter out rare affixes
    prefix_scores = {k: v for k, v in prefix_scores.items() if v >= min_freq}
    suffix_scores = {k: v for k, v in suffix_scores.items() if v >= min_freq}

    return dict(prefix_scores), dict(suffix_scores)


def write_affix_file(path: str, affix_scores: dict[str, float]) -> None:
    """Write affix,score CSV sorted by score descending.

    Final score = int(log(accumulated_score) * 100).
    """
    rows = sorted(affix_scores.items(), key=lambda x: x[1], reverse=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["affix", "score"])
        for affix, raw_score in rows:
            final = int(math.log(raw_score) * 100) if raw_score > 0 else 0
            writer.writerow([affix, final])


def process_one(freq_path: str, out_dir: str, max_len: int = 12, min_freq: int = 10) -> None:
    lang = os.path.splitext(os.path.basename(freq_path))[0]
    print(f"  {lang}: reading {freq_path} ...", end="", flush=True)

    prefix_scores, suffix_scores = build_affix_freqs(freq_path, max_len, min_freq)

    prep_path = os.path.join(out_dir, f"{lang}.prep")
    post_path = os.path.join(out_dir, f"{lang}.post")

    write_affix_file(prep_path, prefix_scores)
    write_affix_file(post_path, suffix_scores)

    print(f"  {len(prefix_scores)} prefixes, {len(suffix_scores)} suffixes")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate prefix/suffix log-frequency files.")
    ap.add_argument("-freq", default=None, help="Single frequency CSV to process.")
    ap.add_argument("-freq_dir", default=None, help="Directory with frequency CSVs (process all).")
    ap.add_argument("-out_dir", default=None,
                    help="Output directory (default: same as input).")
    ap.add_argument("-max_len", type=int, default=12,
                    help="Max prefix/suffix length in characters (default: 12).")
    ap.add_argument("-min_freq", type=int, default=10,
                    help="Minimum accumulated raw frequency to keep an affix (default: 10).")
    args = ap.parse_args()

    if not args.freq and not args.freq_dir:
        print("Provide -freq <file> or -freq_dir <dir>")
        sys.exit(1)

    files = []
    if args.freq:
        files.append(args.freq)
        out = args.out_dir or os.path.dirname(args.freq)
    else:
        files = sorted(glob.glob(os.path.join(args.freq_dir, "*.csv")))
        out = args.out_dir or args.freq_dir

    if not files:
        print("No CSV files found.")
        sys.exit(1)

    os.makedirs(out, exist_ok=True)
    print(f"Processing {len(files)} file(s), max affix length = {args.max_len}")

    for f in files:
        process_one(f, out, args.max_len, args.min_freq)

    print("Done.")


if __name__ == "__main__":
    main()
