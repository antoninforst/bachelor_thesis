#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Universal morphological segmentation + root identification pipeline.

Trains a single model on all available annotated languages (.morf files)
and applies it to any language that has a frequency word list.

Three modes:
  train-eval  — Leave-language-out cross-validation across training languages
  train-save  — Train on all training data, save universal models
  segment     — Apply saved universal models to data/2_annotated CSVs

Usage examples:
  python generate_universal.py train-eval
  python generate_universal.py train-save
  python generate_universal.py segment
  python generate_universal.py segment --improve
  python generate_universal.py segment --langs ces eng deu
"""

import argparse
import csv
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from segmentation import (
    MorphSegmenter,
    load_frequency_map,
    load_segmented_words,
    load_model as load_seg_model,
    save_model as save_seg_model,
    build_examples as build_seg_examples,
    segmented_to_raw_and_boundaries,
    build_boundary_labels,
    extract_gap_features,
    _boundaries_to_segmented,
    _levenshtein as seg_levenshtein,
)
from root_identification import (
    RootIdentifier,
    load_words_with_roots,
    load_model as load_root_model,
    save_model as save_root_model,
    build_examples as build_root_examples,
    extract_morph_features,
    _annotate_morphs,
    _levenshtein as root_levenshtein,
)


# ---------------------------------------------------------------------------
# Defaults (relative to repo root)
# ---------------------------------------------------------------------------

DEFAULT_TRAIN_DIR = "data/2_1_segmentation/training_data"
DEFAULT_FREQ_DIR = "data/1_aggregated"
DEFAULT_MODEL_DIR = "data/2_1_segmentation/models"
DEFAULT_RESULT_DIR = "data/3_results"
DEFAULT_ANNOTATED_DIR = "data/2_annotated"


# ---------------------------------------------------------------------------
# Data discovery helpers
# ---------------------------------------------------------------------------


def discover_training_langs(train_dir: str) -> List[str]:
    """Return sorted list of language codes with .morf files."""
    pattern = os.path.join(train_dir, "*.morf")
    return sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(pattern)
    )


def discover_annotated_langs(annotated_dir: str) -> List[str]:
    """Return sorted list of language codes with .csv files in 2_annotated."""
    pattern = os.path.join(annotated_dir, "*.csv")
    return sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(pattern)
    )


def load_freq_map_if_exists(
    lang: str, freq_dir: str
) -> Optional[Dict[str, int]]:
    """Load frequency map for a language, or return None if file missing."""
    path = os.path.join(freq_dir, f"{lang}.csv")
    if os.path.isfile(path):
        return load_frequency_map(path)
    return None


def load_all_training_data(
    train_dir: str,
    freq_dir: str,
) -> Dict[str, dict]:
    """Discover all .morf files and load their data + optional freq maps.

    Returns {lang: {"seg_words": [...], "root_data": [...], "freq_map": ... or None}}
    """
    langs = discover_training_langs(train_dir)
    result: Dict[str, dict] = {}
    for lang in langs:
        morf_path = os.path.join(train_dir, f"{lang}.morf")
        seg_words = load_segmented_words(morf_path)
        root_data = load_words_with_roots(morf_path)
        freq_map = load_freq_map_if_exists(lang, freq_dir)
        result[lang] = {
            "seg_words": seg_words,
            "root_data": root_data,
            "freq_map": freq_map,
        }
        freq_info = f"{len(freq_map)} entries" if freq_map else "no freq file"
        print(f"  {lang}: {len(seg_words)} words ({freq_info})")
    return result


# ---------------------------------------------------------------------------
# Pooling: build features from multiple languages (each with own freq_map)
# ---------------------------------------------------------------------------


def pool_seg_examples(
    all_data: Dict[str, dict],
    langs: List[str],
) -> Tuple[List[Dict[str, object]], np.ndarray]:
    """Pool segmentation features from multiple languages."""
    X_all: List[Dict[str, object]] = []
    y_all: List[int] = []
    for lang in langs:
        d = all_data[lang]
        X, y = build_seg_examples(d["seg_words"], d["freq_map"])
        X_all.extend(X)
        y_all.extend(y.tolist())
    return X_all, np.array(y_all, dtype=int)


def pool_root_examples(
    all_data: Dict[str, dict],
    langs: List[str],
) -> Tuple[List[Dict[str, object]], np.ndarray]:
    """Pool root identification features from multiple languages."""
    X_all: List[Dict[str, object]] = []
    y_all: List[int] = []
    for lang in langs:
        d = all_data[lang]
        X, y = build_root_examples(d["root_data"], d["freq_map"])
        X_all.extend(X)
        y_all.extend(y.tolist())
    return X_all, np.array(y_all, dtype=int)


def train_seg_from_pool(
    X_dicts: List[Dict[str, object]],
    y: np.ndarray,
) -> MorphSegmenter:
    """Train a MorphSegmenter from pre-built feature dicts."""
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dicts)
    clf = LogisticRegression(
        solver="lbfgs", class_weight="balanced", C=1.0, max_iter=5000,
    )
    clf.fit(X, y)
    return MorphSegmenter(vec, clf, freq_map=None)


def train_root_from_pool(
    X_dicts: List[Dict[str, object]],
    y: np.ndarray,
) -> RootIdentifier:
    """Train a RootIdentifier from pre-built feature dicts."""
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dicts)
    clf = LogisticRegression(
        solver="lbfgs", class_weight="balanced", C=1.0, max_iter=5000,
    )
    clf.fit(X, y)
    return RootIdentifier(vec, clf, freq_map=None)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_seg_on_lang(
    seg_model: MorphSegmenter,
    seg_words: List[str],
    freq_map: Optional[Dict[str, int]],
) -> dict:
    """Evaluate a trained segmentation model on one language's words."""
    seg_model.freq_map = freq_map
    word_results = []
    total_dist = 0
    total_score = 0.0

    for seg in seg_words:
        raw, boundaries = segmented_to_raw_and_boundaries(seg)
        if len(raw) < 2:
            continue
        true_labels = build_boundary_labels(raw, boundaries)
        feats = [extract_gap_features(raw, i, freq_map) for i in range(len(raw) - 1)]
        Xw = seg_model.vec.transform(feats)
        preds = seg_model.clf.predict(Xw).tolist()

        gold_bounds = [i for i, v in enumerate(true_labels) if v == 1]
        pred_bounds = [i for i, v in enumerate(preds) if v == 1]
        dist = seg_levenshtein(gold_bounds, pred_bounds)
        max_len = max(len(gold_bounds), len(pred_bounds))
        score = 1.0 - dist / max_len if max_len > 0 else 1.0

        total_dist += dist
        total_score += score

        pred_seg = _boundaries_to_segmented(raw, preds)
        word_results.append((0.0, dist, seg, pred_seg))

    word_results.sort(key=lambda x: x[1], reverse=True)
    n_words = len(word_results)
    n_correct = sum(1 for _, d, _, _ in word_results if d == 0)

    return {
        "avg_levenshtein": total_dist / n_words if n_words else 0.0,
        "levenshtein_score": total_score / n_words if n_words else 0.0,
        "avg_loss": 0.0,
        "word_accuracy": n_correct / n_words if n_words else 0.0,
        "n_words": n_words,
        "n_correct": n_correct,
        "word_results": word_results,
    }


def evaluate_root_on_lang(
    root_model: RootIdentifier,
    root_data: List[Tuple[List[str], List[bool]]],
    freq_map: Optional[Dict[str, int]],
) -> dict:
    """Evaluate a trained root model on one language's words."""
    root_model.freq_map = freq_map
    word_results = []
    total_dist = 0
    total_score = 0.0

    for morphs, is_root in root_data:
        true_labels = [1 if r else 0 for r in is_root]
        feats = [extract_morph_features(morphs, i, freq_map) for i in range(len(morphs))]
        Xw = root_model.vec.transform(feats)
        pred_labels = root_model.clf.predict(Xw).tolist()

        # Guarantee at least one root
        if not any(p == 1 for p in pred_labels):
            proba = root_model.clf.predict_proba(Xw)
            cls1 = list(root_model.clf.classes_).index(1)
            best = int(np.argmax(proba[:, cls1]))
            pred_labels[best] = 1

        gold_positions = [i for i, v in enumerate(true_labels) if v == 1]
        pred_positions = [i for i, v in enumerate(pred_labels) if v == 1]
        dist = root_levenshtein(gold_positions, pred_positions)
        max_len = max(len(gold_positions), len(pred_positions))
        score = 1.0 - dist / max_len if max_len > 0 else 1.0

        total_dist += dist
        total_score += score

        gold_str = _annotate_morphs(morphs, is_root)
        pred_roots = [bool(p) for p in pred_labels]
        pred_str = _annotate_morphs(morphs, pred_roots)
        word_results.append((0.0, dist, gold_str, pred_str))

    word_results.sort(key=lambda x: x[1], reverse=True)
    n_words = len(word_results)
    n_correct = sum(1 for _, d, _, _ in word_results if d == 0)

    return {
        "avg_levenshtein": total_dist / n_words if n_words else 0.0,
        "levenshtein_score": total_score / n_words if n_words else 0.0,
        "avg_loss": 0.0,
        "word_accuracy": n_correct / n_words if n_words else 0.0,
        "n_words": n_words,
        "n_correct": n_correct,
        "word_results": word_results,
    }


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

RESULT_COLUMNS = [
    "lang", "n_words", "avg_levenshtein", "levenshtein_score",
    "word_accuracy", "n_correct",
]


def _write_result_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Mode: train-eval
# ---------------------------------------------------------------------------


def cmd_train_eval(args: argparse.Namespace) -> None:
    """Leave-language-out cross-validation."""
    print(f"Loading training data from {args.train_dir} ...")
    all_data = load_all_training_data(args.train_dir, args.freq_dir)
    langs = sorted(all_data.keys())

    if len(langs) < 2:
        print("Need at least 2 training languages for leave-one-out evaluation.")
        sys.exit(1)

    print(f"\nFound {len(langs)} training languages: {', '.join(langs)}")
    print(f"Running leave-language-out cross-validation ...\n")

    seg_rows: List[dict] = []
    root_rows: List[dict] = []

    for held_out in langs:
        train_langs = [l for l in langs if l != held_out]
        n_train = sum(len(all_data[l]["seg_words"]) for l in train_langs)
        n_test = len(all_data[held_out]["seg_words"])
        print(f"  Hold out {held_out}: train on {len(train_langs)} langs "
              f"({n_train} words), test on {n_test} words")

        # Pool training features
        seg_X, seg_y = pool_seg_examples(all_data, train_langs)
        root_X, root_y = pool_root_examples(all_data, train_langs)

        # Train
        seg_model = train_seg_from_pool(seg_X, seg_y)
        root_model = train_root_from_pool(root_X, root_y)

        # Evaluate on held-out
        held = all_data[held_out]
        seg_metrics = evaluate_seg_on_lang(
            seg_model, held["seg_words"], held["freq_map"])
        root_metrics = evaluate_root_on_lang(
            root_model, held["root_data"], held["freq_map"])

        print(f"    seg:  lev_score={seg_metrics['levenshtein_score']:.4f}  "
              f"word_acc={seg_metrics['word_accuracy']:.4f}")
        print(f"    root: lev_score={root_metrics['levenshtein_score']:.4f}  "
              f"word_acc={root_metrics['word_accuracy']:.4f}")

        if args.n_worst > 0:
            seg_worst = seg_metrics["word_results"][:args.n_worst]
            if seg_worst:
                print(f"    worst segmentations:")
                for _, dist, gold, pred in seg_worst:
                    if dist > 0:
                        print(f"      {gold:<30s} → {pred}")

        seg_rows.append({
            "lang": held_out,
            "n_words": seg_metrics["n_words"],
            "avg_levenshtein": round(seg_metrics["avg_levenshtein"], 4),
            "levenshtein_score": round(seg_metrics["levenshtein_score"], 4),
            "word_accuracy": round(seg_metrics["word_accuracy"], 4),
            "n_correct": seg_metrics["n_correct"],
        })
        root_rows.append({
            "lang": held_out,
            "n_words": root_metrics["n_words"],
            "avg_levenshtein": round(root_metrics["avg_levenshtein"], 4),
            "levenshtein_score": round(root_metrics["levenshtein_score"], 4),
            "word_accuracy": round(root_metrics["word_accuracy"], 4),
            "n_correct": root_metrics["n_correct"],
        })
        print()

    # Aggregate
    n_total = sum(r["n_words"] for r in seg_rows)
    avg_seg_score = sum(r["levenshtein_score"] * r["n_words"] for r in seg_rows) / n_total if n_total else 0
    avg_root_score = sum(r["levenshtein_score"] * r["n_words"] for r in root_rows) / n_total if n_total else 0
    print(f"Aggregate (weighted by word count):")
    print(f"  Seg  levenshtein_score: {avg_seg_score:.4f}")
    print(f"  Root levenshtein_score: {avg_root_score:.4f}")

    # Write CSVs
    seg_path = os.path.join(args.result_dir, "result_segment_universal.csv")
    root_path = os.path.join(args.result_dir, "result_root_universal.csv")
    _write_result_csv(seg_path, seg_rows)
    _write_result_csv(root_path, root_rows)
    print(f"\nResults written to:")
    print(f"  {seg_path}")
    print(f"  {root_path}")


# ---------------------------------------------------------------------------
# Mode: train-save
# ---------------------------------------------------------------------------


def cmd_train_save(args: argparse.Namespace) -> None:
    """Train universal models on all data and save."""
    print(f"Loading training data from {args.train_dir} ...")
    all_data = load_all_training_data(args.train_dir, args.freq_dir)
    langs = sorted(all_data.keys())

    total_words = sum(len(d["seg_words"]) for d in all_data.values())
    print(f"\nPooling {total_words} words from {len(langs)} languages ...")

    seg_path = os.path.join(args.model_dir, "universal_seg.pkl")
    root_path = os.path.join(args.model_dir, "universal_root.pkl")

    # Pool and train segmentation model
    print("Building segmentation features ...")
    seg_X, seg_y = pool_seg_examples(all_data, langs)
    print(f"  {len(seg_X)} gap examples ({int(seg_y.sum())} positive)")
    seg_model = train_seg_from_pool(seg_X, seg_y)

    os.makedirs(os.path.dirname(seg_path), exist_ok=True)
    save_seg_model(seg_path, seg_model)
    print(f"Segmentation model saved to {seg_path}")

    # Pool and train root model
    print("Building root identification features ...")
    root_X, root_y = pool_root_examples(all_data, langs)
    print(f"  {len(root_X)} morph examples ({int(root_y.sum())} positive)")
    root_model = train_root_from_pool(root_X, root_y)

    os.makedirs(os.path.dirname(root_path), exist_ok=True)
    save_root_model(root_path, root_model)
    print(f"Root model saved to {root_path}")


# ---------------------------------------------------------------------------
# Mode: segment
# ---------------------------------------------------------------------------


def segment_word(
    word: str,
    seg_model: MorphSegmenter,
    root_model: RootIdentifier,
) -> str:
    """Segment a word and annotate roots. Returns e.g. 'za @hrad a'."""
    segmented = seg_model.segment(word)
    morphs = segmented.split()
    annotated = root_model.annotate(morphs)
    return annotated


def process_annotated_csv(
    lang: str,
    annotated_dir: str,
    freq_dir: str,
    seg_model: MorphSegmenter,
    root_model: RootIdentifier,
) -> int:
    """Process one 2_annotated CSV: segment words in-place. Returns word count."""
    csv_path = os.path.join(annotated_dir, f"{lang}.csv")
    freq_map = load_freq_map_if_exists(lang, freq_dir)

    # Swap in this language's freq_map
    seg_model.freq_map = freq_map
    root_model.freq_map = freq_map

    # Read
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)

    # Segment each word
    for row in rows:
        if len(row) >= 1:
            original_word = row[0].replace("@", "").replace(" ", "")
            if original_word:
                row[0] = segment_word(original_word, seg_model, root_model)

    # Write back
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    return len(rows)


def cmd_segment(args: argparse.Namespace) -> None:
    """Apply universal (and optionally per-language) models to 2_annotated CSVs."""
    seg_path = os.path.join(args.model_dir, "universal_seg.pkl")
    root_path = os.path.join(args.model_dir, "universal_root.pkl")

    # Check model files exist
    if not os.path.isfile(seg_path):
        print(f"Universal segmentation model not found: {seg_path}")
        print("Run 'train-save' first.")
        sys.exit(1)
    if not os.path.isfile(root_path):
        print(f"Universal root model not found: {root_path}")
        print("Run 'train-save' first.")
        sys.exit(1)

    # Discover languages to process
    if args.langs:
        target_langs = args.langs
    else:
        target_langs = discover_annotated_langs(args.annotated_dir)

    if not target_langs:
        print(f"No CSV files found in {args.annotated_dir}")
        sys.exit(1)

    print(f"Segment mode: {len(target_langs)} languages to process")

    # Discover which languages have per-language models (for --improve)
    trained_langs = discover_training_langs(args.train_dir)

    # --- Phase 1: Universal model on all languages ---
    print(f"\nLoading universal models ...")
    seg_model = load_seg_model(seg_path)
    root_model = load_root_model(root_path)
    print(f"  Loaded {seg_path}")
    print(f"  Loaded {root_path}")

    print(f"\nProcessing {len(target_langs)} languages with universal model ...")
    for lang in target_langs:
        csv_path = os.path.join(args.annotated_dir, f"{lang}.csv")
        if not os.path.isfile(csv_path):
            print(f"  [SKIP] {lang}: no file {csv_path}")
            continue
        n = process_annotated_csv(
            lang, args.annotated_dir, args.freq_dir, seg_model, root_model)
        freq_map = load_freq_map_if_exists(lang, args.freq_dir)
        freq_info = f"{len(freq_map)} freq" if freq_map else "no freq"
        print(f"  {lang}: {n} words segmented ({freq_info})")

    # --- Phase 2: --improve with per-language models ---
    if args.improve:
        improve_langs = [l for l in trained_langs
                         if l in set(target_langs)]
        if improve_langs:
            print(f"\n--improve: re-processing {len(improve_langs)} languages "
                  f"with per-language models: {', '.join(improve_langs)}")
            for lang in improve_langs:
                lang_seg_path = os.path.join(args.model_dir, f"{lang}_seg.pkl")
                lang_root_path = os.path.join(args.model_dir, f"{lang}_root.pkl")
                if not os.path.isfile(lang_seg_path) or not os.path.isfile(lang_root_path):
                    print(f"  [SKIP] {lang}: per-language models not found")
                    continue
                lang_seg = load_seg_model(lang_seg_path)
                lang_root = load_root_model(lang_root_path)
                n = process_annotated_csv(
                    lang, args.annotated_dir, args.freq_dir, lang_seg, lang_root)
                print(f"  {lang}: {n} words re-segmented with per-language model")
        else:
            print("\n--improve: no per-language models overlap with target languages")

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Universal morphological segmentation + root identification.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # -- Shared arguments --
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--train-dir",
        default=DEFAULT_TRAIN_DIR,
        help=f"Directory with .morf training files (default: {DEFAULT_TRAIN_DIR})",
    )
    shared.add_argument(
        "--freq-dir",
        default=DEFAULT_FREQ_DIR,
        help=f"Directory with frequency CSVs (default: {DEFAULT_FREQ_DIR})",
    )
    shared.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Directory for model files (default: {DEFAULT_MODEL_DIR})",
    )
    shared.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)",
    )

    # -- train-eval --
    p_eval = sub.add_parser(
        "train-eval",
        parents=[shared],
        help="Leave-language-out cross-validation",
    )
    p_eval.add_argument(
        "--result-dir",
        default=DEFAULT_RESULT_DIR,
        help=f"Directory for result CSVs (default: {DEFAULT_RESULT_DIR})",
    )
    p_eval.add_argument(
        "--n-worst", type=int, default=20,
        help="Number of worst predictions to show (default: 20)",
    )

    # -- train-save --
    p_train = sub.add_parser(
        "train-save",
        parents=[shared],
        help="Train on all data and save universal models",
    )

    # -- segment --
    p_seg = sub.add_parser(
        "segment",
        parents=[shared],
        help="Apply universal models to 2_annotated CSVs",
    )
    p_seg.add_argument(
        "--annotated-dir",
        default=DEFAULT_ANNOTATED_DIR,
        help=f"Directory with annotated CSVs (default: {DEFAULT_ANNOTATED_DIR})",
    )
    p_seg.add_argument(
        "--improve",
        action="store_true",
        help="Re-process languages that have per-language models with those (better) models",
    )
    p_seg.add_argument(
        "--langs",
        nargs="*",
        default=None,
        help="Only process these language codes (default: all)",
    )

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    if args.command == "train-eval":
        cmd_train_eval(args)
    elif args.command == "train-save":
        cmd_train_save(args)
    elif args.command == "segment":
        cmd_segment(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
