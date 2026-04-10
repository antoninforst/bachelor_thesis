#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Universal morphological segmentation + root identification pipeline.

Trains a single model on all available annotated languages (.morf files)
and applies it to any language that has a frequency word list.

Modes
-----
  train-eval   Leave-language-out cross-validation across training languages.
  train-save   Train on all training data and save universal models.
  segment      Apply saved universal models to data/2_annotated CSVs.

Shared options (all modes)
--------------------------
  --train-dir DIR    Directory with .morf training files
                     (default: data/2_1_segmentation/training_data)
  --freq-dir DIR     Directory with frequency CSVs
                     (default: data/1_aggregated)
  --model-dir DIR    Directory for model .pkl files
                     (default: data/2_1_segmentation/models)
  --seed N           Random seed (default: 42)

train-eval options
------------------
  --result-dir DIR   Directory for result CSVs (default: results/2_segmentation)
  --n-worst N        Number of worst predictions to print (default: 20)

segment options
---------------
  --annotated-dir DIR   Directory with annotated CSVs (default: data/2_annotated)
  --langs CODE ...      Only process these language codes (default: all)
  --improve             Use per-language models for languages that have them
                        instead of the universal model (not in addition to)
  --workers N           Number of parallel workers (default: min(4, CPUs);
                        1 = sequential)
  --skip-segmented      Skip languages whose data is already segmented

Usage examples
--------------
  python generate_universal.py train-eval
  python generate_universal.py train-eval --n-worst 5
  python generate_universal.py train-save
  python generate_universal.py segment
  python generate_universal.py segment --improve
  python generate_universal.py segment --langs ces eng deu
  python generate_universal.py segment --skip-segmented --workers 2
"""

import argparse
import csv
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from segmentation import (
    MorphSegmenter,
    load_frequency_map,
    load_affix_map,
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
DEFAULT_RESULT_DIR = "results/2_segmentation"
DEFAULT_ANNOTATED_DIR = "data/2_annotated"
DEFAULT_AFFIX_DIR = "results/1_preprocessing/affixes"


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


def load_affix_maps_if_exist(
    lang: str, freq_dir: str, affix_dir: Optional[str],
) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, int]]]:
    """Load prefix/suffix affix maps, auto-generating if freq file exists."""
    if affix_dir is None:
        return None, None
    freq_path = os.path.join(freq_dir, f"{lang}.csv")
    if not os.path.isfile(freq_path):
        return None, None
    prep_path = os.path.join(affix_dir, f"{lang}.prep")
    post_path = os.path.join(affix_dir, f"{lang}.post")
    if not os.path.isfile(prep_path) or not os.path.isfile(post_path):
        import subprocess, sys as _sys
        gen_script = os.path.join(os.path.dirname(__file__), "..", "1_preprocessing", "generate_affixes.py")
        if os.path.isfile(gen_script):
            os.makedirs(affix_dir, exist_ok=True)
            subprocess.run(
                [_sys.executable, gen_script, "-freq", freq_path, "-out_dir", affix_dir],
                check=True,
            )
        else:
            return None, None
    prefix_map = load_affix_map(prep_path) if os.path.isfile(prep_path) else None
    suffix_map = load_affix_map(post_path) if os.path.isfile(post_path) else None
    return prefix_map, suffix_map


def load_all_training_data(
    train_dir: str,
    freq_dir: str,
    affix_dir: Optional[str] = None,
) -> Dict[str, dict]:
    """Discover all .morf files and load their data + optional freq maps.

    Returns {lang: {"seg_words": [...], "root_data": [...], "freq_map": ..., "prefix_map": ..., "suffix_map": ...}}
    """
    langs = discover_training_langs(train_dir)
    result: Dict[str, dict] = {}
    for lang in langs:
        morf_path = os.path.join(train_dir, f"{lang}.morf")
        seg_words = load_segmented_words(morf_path)
        root_data = load_words_with_roots(morf_path)
        freq_map = load_freq_map_if_exists(lang, freq_dir)
        prefix_map, suffix_map = load_affix_maps_if_exist(lang, freq_dir, affix_dir)
        result[lang] = {
            "seg_words": seg_words,
            "root_data": root_data,
            "freq_map": freq_map,
            "prefix_map": prefix_map,
            "suffix_map": suffix_map,
        }
        freq_info = f"{len(freq_map)} entries" if freq_map else "no freq file"
        affix_info = f", {len(prefix_map)} pfx" if prefix_map else ""
        print(f"  {lang}: {len(seg_words)} words ({freq_info}{affix_info})")
    return result


# ---------------------------------------------------------------------------
# Pooling: build features from multiple languages (each with own freq_map)
# ---------------------------------------------------------------------------


def pool_seg_examples(
    all_data: Dict[str, dict],
    langs: List[str],
) -> Tuple[List[Dict[str, object]], np.ndarray, np.ndarray]:
    """Pool segmentation features from multiple languages.

    Returns (X_dicts, y, sample_weights) where each language is weighted
    equally regardless of how many examples it contributes.
    """
    X_all: List[Dict[str, object]] = []
    y_all: List[int] = []
    w_all: List[float] = []
    for lang in langs:
        d = all_data[lang]
        X, y = build_seg_examples(d["seg_words"], d["freq_map"], d.get("prefix_map"), d.get("suffix_map"))
        n = len(X)
        w = 1.0 / n if n > 0 else 0.0
        X_all.extend(X)
        y_all.extend(y.tolist())
        w_all.extend([w] * n)
    return X_all, np.array(y_all, dtype=int), np.array(w_all, dtype=float)


def pool_root_examples(
    all_data: Dict[str, dict],
    langs: List[str],
) -> Tuple[List[Dict[str, object]], np.ndarray, np.ndarray]:
    """Pool root identification features from multiple languages.

    Returns (X_dicts, y, sample_weights) where each language is weighted
    equally regardless of how many examples it contributes.
    """
    X_all: List[Dict[str, object]] = []
    y_all: List[int] = []
    w_all: List[float] = []
    for lang in langs:
        d = all_data[lang]
        X, y = build_root_examples(d["root_data"], d["freq_map"], d.get("prefix_map"), d.get("suffix_map"))
        n = len(X)
        w = 1.0 / n if n > 0 else 0.0
        X_all.extend(X)
        y_all.extend(y.tolist())
        w_all.extend([w] * n)
    return X_all, np.array(y_all, dtype=int), np.array(w_all, dtype=float)


def train_seg_from_pool(
    X_dicts: List[Dict[str, object]],
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> MorphSegmenter:
    """Train a MorphSegmenter from pre-built feature dicts."""
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dicts)
    clf = LogisticRegression(
        solver="lbfgs", class_weight="balanced", C=1.0, max_iter=5000,
    )
    clf.fit(X, y, sample_weight=sample_weight)
    return MorphSegmenter(vec, clf, freq_map=None)


def train_root_from_pool(
    X_dicts: List[Dict[str, object]],
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> RootIdentifier:
    """Train a RootIdentifier from pre-built feature dicts."""
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dicts)
    clf = LogisticRegression(
        solver="lbfgs", class_weight="balanced", C=1.0, max_iter=5000,
    )
    clf.fit(X, y, sample_weight=sample_weight)
    return RootIdentifier(vec, clf, freq_map=None)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_seg_on_lang(
    seg_model: MorphSegmenter,
    seg_words: List[str],
    freq_map: Optional[Dict[str, int]],
    prefix_map: Optional[Dict[str, int]] = None,
    suffix_map: Optional[Dict[str, int]] = None,
) -> dict:
    """Evaluate a trained segmentation model on one language's words."""
    seg_model.freq_map = freq_map
    seg_model.prefix_map = prefix_map
    seg_model.suffix_map = suffix_map
    word_results = []
    total_dist = 0
    total_score = 0.0

    for seg in seg_words:
        raw, boundaries = segmented_to_raw_and_boundaries(seg)
        if len(raw) < 2:
            continue
        true_labels = build_boundary_labels(raw, boundaries)
        feats = [extract_gap_features(raw, i, freq_map, prefix_map, suffix_map) for i in range(len(raw) - 1)]
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
    prefix_map: Optional[Dict[str, int]] = None,
    suffix_map: Optional[Dict[str, int]] = None,
) -> dict:
    """Evaluate a trained root model on one language's words."""
    root_model.freq_map = freq_map
    root_model.prefix_map = prefix_map
    root_model.suffix_map = suffix_map
    word_results = []
    total_dist = 0
    total_score = 0.0

    for morphs, is_root in root_data:
        true_labels = [1 if r else 0 for r in is_root]
        feats = [extract_morph_features(morphs, i, freq_map, prefix_map, suffix_map) for i in range(len(morphs))]
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
    affix_dir = getattr(args, "affix_dir", None)
    all_data = load_all_training_data(args.train_dir, args.freq_dir, affix_dir)
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
        seg_X, seg_y, seg_w = pool_seg_examples(all_data, train_langs)
        root_X, root_y, root_w = pool_root_examples(all_data, train_langs)

        # Train
        seg_model = train_seg_from_pool(seg_X, seg_y, seg_w)
        root_model = train_root_from_pool(root_X, root_y, root_w)

        # Evaluate on held-out
        held = all_data[held_out]
        seg_metrics = evaluate_seg_on_lang(
            seg_model, held["seg_words"], held["freq_map"],
            held.get("prefix_map"), held.get("suffix_map"))
        root_metrics = evaluate_root_on_lang(
            root_model, held["root_data"], held["freq_map"],
            held.get("prefix_map"), held.get("suffix_map"))

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
    affix_dir = getattr(args, "affix_dir", None)
    all_data = load_all_training_data(args.train_dir, args.freq_dir, affix_dir)
    langs = sorted(all_data.keys())

    total_words = sum(len(d["seg_words"]) for d in all_data.values())
    print(f"\nPooling {total_words} words from {len(langs)} languages ...")

    seg_path = os.path.join(args.model_dir, "universal_seg.pkl")
    root_path = os.path.join(args.model_dir, "universal_root.pkl")

    # Pool and train segmentation model
    print("Building segmentation features ...")
    seg_X, seg_y, seg_w = pool_seg_examples(all_data, langs)
    print(f"  {len(seg_X)} gap examples ({int(seg_y.sum())} positive)")
    seg_model = train_seg_from_pool(seg_X, seg_y, seg_w)

    os.makedirs(os.path.dirname(seg_path), exist_ok=True)
    save_seg_model(seg_path, seg_model)
    print(f"Segmentation model saved to {seg_path}")

    # Pool and train root model
    print("Building root identification features ...")
    root_X, root_y, root_w = pool_root_examples(all_data, langs)
    print(f"  {len(root_X)} morph examples ({int(root_y.sum())} positive)")
    root_model = train_root_from_pool(root_X, root_y, root_w)

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


def _is_already_segmented(word_field: str) -> bool:
    """Check if a CSV word field contains segmentation markers."""
    return " " in word_field or "@" in word_field


def _check_already_segmented(csv_path: str) -> bool:
    """Peek at the first few data rows of a CSV to check if it's already segmented."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for _, row in zip(range(10), reader):
            if len(row) >= 1 and _is_already_segmented(row[0]):
                return True
    return False


def process_annotated_csv(
    lang: str,
    annotated_dir: str,
    freq_dir: str,
    seg_model: MorphSegmenter,
    root_model: RootIdentifier,
    affix_dir: Optional[str] = None,
    batch_size: int = 4096,
) -> Tuple[int, bool, str]:
    """Process one 2_annotated CSV: segment words via a temp file.

    Reads rows in batches, segments each batch, writes to a temp file,
    then replaces the original. Keeps memory bounded.

    Returns (word_count, was_already_segmented, freq_info).
    """
    csv_path = os.path.join(annotated_dir, f"{lang}.csv")
    tmp_path = csv_path + ".tmp"
    freq_map = load_freq_map_if_exists(lang, freq_dir)
    freq_info = f"{len(freq_map)} freq" if freq_map else "no freq"

    seg_model.freq_map = freq_map
    root_model.freq_map = freq_map

    # Load affix maps
    prefix_map, suffix_map = load_affix_maps_if_exist(lang, freq_dir, affix_dir)
    seg_model.prefix_map = prefix_map
    seg_model.suffix_map = suffix_map
    root_model.prefix_map = prefix_map
    root_model.suffix_map = suffix_map

    n_rows = 0
    was_segmented = _check_already_segmented(csv_path)

    with open(csv_path, "r", encoding="utf-8") as fin, \
         open(tmp_path, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(header)

        batch_rows: List[List[str]] = []

        for row in reader:
            batch_rows.append(row)
            if len(batch_rows) >= batch_size:
                _segment_and_write_batch(
                    batch_rows, writer, seg_model, root_model)
                n_rows += len(batch_rows)
                batch_rows = []

        # Last partial batch
        if batch_rows:
            _segment_and_write_batch(
                batch_rows, writer, seg_model, root_model)
            n_rows += len(batch_rows)

    os.replace(tmp_path, csv_path)

    seg_model.freq_map = None
    root_model.freq_map = None
    seg_model.prefix_map = None
    seg_model.suffix_map = None
    root_model.prefix_map = None
    root_model.suffix_map = None

    return n_rows, was_segmented, freq_info


def _segment_and_write_batch(
    rows: List[List[str]],
    writer: csv.writer,
    seg_model: MorphSegmenter,
    root_model: RootIdentifier,
) -> None:
    """Segment a batch of CSV rows and write them out."""
    # Collect words to segment
    raw_words: List[str] = []
    word_row_indices: List[int] = []
    for i, row in enumerate(rows):
        if len(row) >= 1:
            w = row[0].replace("@", "").replace(" ", "")
            if w:
                raw_words.append(w)
                word_row_indices.append(i)

    if raw_words:
        segmented = seg_model.segment_batch(raw_words)
        morph_lists = [s.split() for s in segmented]
        annotated = root_model.annotate_batch(morph_lists)
        for idx, ann in zip(word_row_indices, annotated):
            rows[idx][0] = ann

    writer.writerows(rows)


# ---------------------------------------------------------------------------
# Parallel worker with per-process model caching
# ---------------------------------------------------------------------------

import multiprocessing

_worker_seg_model: Optional[MorphSegmenter] = None
_worker_root_model: Optional[RootIdentifier] = None
_worker_id: int = 0
_worker_counter: Optional[multiprocessing.Value] = None
_worker_affix_dir: Optional[str] = None


def _worker_init(seg_model_path: str, root_model_path: str,
                 counter: multiprocessing.Value,
                 affix_dir: Optional[str] = None) -> None:
    """Called once per worker process to load models into process-global variables."""
    global _worker_seg_model, _worker_root_model, _worker_id, _worker_affix_dir
    with counter.get_lock():
        counter.value += 1
        _worker_id = counter.value
    _worker_affix_dir = affix_dir
    print(f"  [worker {_worker_id}] loading models ...", flush=True)
    _worker_seg_model = load_seg_model(seg_model_path)
    _worker_root_model = load_root_model(root_model_path)


def _process_one_lang(
    lang: str,
    annotated_dir: str,
    freq_dir: str,
) -> Tuple[str, int, bool, str]:
    """Worker function for parallel processing. Returns (lang, n, was_seg, freq_info)."""
    print(f"  [worker {_worker_id}] {lang}: segmenting ...", flush=True)
    n, was_seg, freq_info = process_annotated_csv(
        lang, annotated_dir, freq_dir, _worker_seg_model, _worker_root_model,
        affix_dir=_worker_affix_dir)
    return lang, n, was_seg, freq_info


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

    # When --improve, find languages that actually have per-language model
    # files so we can skip them in the universal pass.
    improve_langs_set: set = set()
    if args.improve:
        for lang in trained_langs:
            lang_seg = os.path.join(args.model_dir, f"{lang}_seg.pkl")
            lang_root = os.path.join(args.model_dir, f"{lang}_root.pkl")
            if os.path.isfile(lang_seg) and os.path.isfile(lang_root):
                improve_langs_set.add(lang)

    # --- Phase 1: Universal model on all languages ---
    print(f"\nLoading universal models ...")
    seg_model = load_seg_model(seg_path)
    root_model = load_root_model(root_path)
    print(f"  Loaded {seg_path}")
    print(f"  Loaded {root_path}")

    # Filter to existing files; exclude --improve languages from universal pass
    valid_langs = []
    for lang in target_langs:
        csv_path = os.path.join(args.annotated_dir, f"{lang}.csv")
        if not os.path.isfile(csv_path):
            print(f"  [SKIP] {lang}: no file {csv_path}")
        elif lang in improve_langs_set:
            pass  # will be handled in Phase 2 with per-language model
        else:
            valid_langs.append(lang)

    workers = getattr(args, "workers", None)
    skip_segmented = getattr(args, "skip_segmented", False)
    n_reseg = 0
    n_skipped = 0

    # Pre-filter: skip already-segmented languages (cheap: reads 2 lines)
    if skip_segmented:
        todo_langs = []
        for lang in valid_langs:
            csv_path = os.path.join(args.annotated_dir, f"{lang}.csv")
            if _check_already_segmented(csv_path):
                n_skipped += 1
            else:
                todo_langs.append(lang)
        valid_langs = todo_langs

    if not valid_langs:
        print("Nothing to process.")
    elif workers == 1 or len(valid_langs) <= 3:
        # Sequential
        print(f"\nProcessing {len(valid_langs)} languages with universal model ...")
        for lang in valid_langs:
            n, was_seg, freq_info = process_annotated_csv(
                lang, args.annotated_dir, args.freq_dir, seg_model, root_model,
                affix_dir=getattr(args, 'affix_dir', None))
            reseg_tag = " (re-segmented)" if was_seg else ""
            if was_seg:
                n_reseg += 1
            print(f"  {lang}: {n} words ({freq_info}){reseg_tag}")
    else:
        # Parallel — cap default workers to limit memory
        n_workers = workers or min(4, os.cpu_count() or 1, len(valid_langs))
        print(f"\nProcessing {len(valid_langs)} languages with universal model "
              f"({n_workers} workers) ...")

        # Each worker loads models once via _worker_init
        counter = multiprocessing.Value("i", 0)

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(seg_path, root_path, counter,
                      getattr(args, 'affix_dir', None)),
        ) as pool:
            futures = {}
            for lang in valid_langs:
                futures[pool.submit(
                    _process_one_lang,
                    lang, args.annotated_dir, args.freq_dir,
                )] = lang
            for future in as_completed(futures):
                lang, n, was_seg, freq_info = future.result()
                reseg_tag = " (re-segmented)" if was_seg else ""
                if was_seg:
                    n_reseg += 1
                print(f"  {lang}: {n} words ({freq_info}){reseg_tag}")

    if n_skipped > 0:
        print(f"\nSkipped {n_skipped} already-segmented language(s).")
    if n_reseg > 0:
        print(f"\nNote: {n_reseg} file(s) contained previously segmented data "
              f"and were re-segmented from scratch.")

    # --- Phase 2: --improve with per-language models ---
    if args.improve and improve_langs_set:
        improve_langs = [l for l in trained_langs
                         if l in improve_langs_set and l in set(target_langs)]
        if improve_langs:
            print(f"\n--improve: processing {len(improve_langs)} languages "
                  f"with per-language models: {', '.join(improve_langs)}")
            for lang in improve_langs:
                lang_seg_path = os.path.join(args.model_dir, f"{lang}_seg.pkl")
                lang_root_path = os.path.join(args.model_dir, f"{lang}_root.pkl")
                lang_seg = load_seg_model(lang_seg_path)
                lang_root = load_root_model(lang_root_path)
                n, _, _ = process_annotated_csv(
                    lang, args.annotated_dir, args.freq_dir, lang_seg, lang_root,
                    affix_dir=getattr(args, 'affix_dir', None))
                print(f"  {lang}: {n} words segmented with per-language model")
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
    shared.add_argument(
        "--affix-dir",
        default=DEFAULT_AFFIX_DIR,
        help=f"Directory with .prep/.post affix files (default: {DEFAULT_AFFIX_DIR})",
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
    p_seg.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: min(4, CPUs); 1 = sequential)",
    )
    p_seg.add_argument(
        "--skip-segmented",
        action="store_true",
        help="Skip languages whose data is already segmented",
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
