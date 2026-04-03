#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch morphological analysis — evaluate/train all languages in parallel.

Discovers .morf files in the training-data directory, runs generate.run_pipeline()
for each language using a process pool, and writes aggregate result CSVs.

Usage:
    python generate_multiple.py -train_dir ../../data/2_1_segmentation/training_data \
                                -freq_dir ../../data/1_aggregated \
                                -result_dir ../../data/3_results

    python generate_multiple.py                          # uses default paths
    python generate_multiple.py -workers 4 -folds 10     # custom parallelism & folds
"""

import argparse
import csv
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

from generate import run_pipeline


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _run_one(
    lang: str,
    source: str,
    freq: Optional[str],
    folds: int,
    seed: int,
    save_model: Optional[str],
    save_root_model: Optional[str],
    eval_only: bool,
) -> dict:
    """Wrapper called in a child process."""
    try:
        result = run_pipeline(
            source=source,
            freq=freq,
            folds=folds,
            seed=seed,
            save_model=save_model,
            save_root_model=save_root_model,
            eval_only=eval_only,
            quiet=True,
        )
        return result
    except Exception as e:
        return {"lang": lang, "error": str(e)}


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

SEG_COLUMNS = [
    "lang",
    "n_words",
    "has_freq",
    "avg_levenshtein",
    "levenshtein_score",
    "avg_loss",
    "word_accuracy",
    "n_correct",
]

ROOT_COLUMNS = [
    "lang",
    "n_words",
    "has_freq",
    "avg_levenshtein",
    "levenshtein_score",
    "avg_loss",
    "word_accuracy",
    "n_correct",
]


def _write_csv(path: str, columns: List[str], rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Batch morphological analysis — evaluate/train all languages.")
    ap.add_argument("-train_dir", default="../../data/2_1_segmentation/training_data",
                    help="Directory with .morf training files.")
    ap.add_argument("-freq_dir", default="../../data/1_aggregated",
                    help="Directory with frequency CSV files (one per language).")
    ap.add_argument("-model_dir", default="../../data/2_1_segmentation/models",
                    help="Directory to save models into.")
    ap.add_argument("-result_dir", default="../../data/3_results",
                    help="Directory to write result CSVs into.")
    ap.add_argument("-folds", type=int, default=5, help="K-fold cross-validation folds (default: 5).")
    ap.add_argument("-seed", type=int, default=42, help="Random seed (default: 42).")
    ap.add_argument("-workers", type=int, default=None,
                    help="Max parallel workers (default: number of CPUs).")
    ap.add_argument("-eval_only", action="store_true",
                    help="Only evaluate, do not train/save final models.")
    args = ap.parse_args()

    # Discover languages
    pattern = os.path.join(args.train_dir, "*.morf")
    morf_files = sorted(glob.glob(pattern))
    if not morf_files:
        print(f"No .morf files found in {args.train_dir}")
        sys.exit(1)

    langs = [os.path.splitext(os.path.basename(f))[0] for f in morf_files]
    print(f"Found {len(langs)} languages: {', '.join(langs)}")

    # Build per-language arguments
    tasks: List[dict] = []
    for lang, source in zip(langs, morf_files):
        freq_path = os.path.join(args.freq_dir, f"{lang}.csv")
        freq = freq_path if os.path.isfile(freq_path) else None

        save_model = None
        save_root = None
        if not args.eval_only:
            save_model = os.path.join(args.model_dir, f"{lang}_seg.pkl")
            save_root = os.path.join(args.model_dir, f"{lang}_root.pkl")

        tasks.append(dict(
            lang=lang,
            source=source,
            freq=freq,
            folds=args.folds,
            seed=args.seed,
            save_model=save_model,
            save_root_model=save_root,
            eval_only=args.eval_only,
        ))

    # Run in parallel
    results: Dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, **t): t["lang"] for t in tasks}
        for future in as_completed(futures):
            lang = futures[future]
            result = future.result()
            results[lang] = result
            if "error" in result:
                print(f"  [FAIL] {lang}: {result['error']}")
            else:
                seg = result.get("seg_metrics")
                root = result.get("root_metrics")
                seg_f1 = f"{seg['levenshtein_score']:.4f}" if seg else "—"
                root_f1 = f"{root['levenshtein_score']:.4f}" if root else "—"
                freq_flag = "freq" if result.get("has_freq") else "no-freq"
                print(f"  [OK]   {lang:>5s}  seg_lev={seg_f1}  root_lev={root_f1}  ({result['n_words']} words, {freq_flag})")

    # Build result CSVs
    seg_rows: List[dict] = []
    root_rows: List[dict] = []
    for lang in sorted(results):
        r = results[lang]
        if "error" in r:
            continue
        seg = r.get("seg_metrics")
        root = r.get("root_metrics")
        if seg:
            seg_rows.append({
                "lang": lang,
                "n_words": seg["n_words"],
                "has_freq": r.get("has_freq", False),
                "avg_levenshtein": round(seg["avg_levenshtein"], 4),
                "levenshtein_score": round(seg["levenshtein_score"], 4),
                "avg_loss": round(seg["avg_loss"], 4),
                "word_accuracy": round(seg["word_accuracy"], 4),
                "n_correct": seg["n_correct"],
            })
        if root:
            root_rows.append({
                "lang": lang,
                "n_words": root["n_words"],
                "has_freq": r.get("has_freq", False),
                "avg_levenshtein": round(root["avg_levenshtein"], 4),
                "levenshtein_score": round(root["levenshtein_score"], 4),
                "avg_loss": round(root["avg_loss"], 4),
                "word_accuracy": round(root["word_accuracy"], 4),
                "n_correct": root["n_correct"],
            })

    seg_path = os.path.join(args.result_dir, "result_segment_true.csv")
    root_path = os.path.join(args.result_dir, "result_root_true.csv")

    if seg_rows:
        _write_csv(seg_path, SEG_COLUMNS, seg_rows)
        print(f"\nSegmentation results → {seg_path}")
    if root_rows:
        _write_csv(root_path, ROOT_COLUMNS, root_rows)
        print(f"Root results         → {root_path}")

    # Summary
    n_ok = sum(1 for r in results.values() if "error" not in r)
    n_fail = len(results) - n_ok
    print(f"\nDone: {n_ok} succeeded, {n_fail} failed out of {len(results)} languages.")


if __name__ == "__main__":
    main()
