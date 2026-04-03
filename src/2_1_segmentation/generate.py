#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Morphological analysis pipeline — main interface.

Orchestrates two tasks:
  1. Segmentation — split a word into morphs (segmentation.py)
  2. Root identification — mark which morph(s) are roots (root_identification.py)

Usage:
  Train + evaluate segmentation:
    python generate.py -source cze.morf -freq ces-mixed2012.csv -test 0.2

  Train + save model:
    python generate.py -source cze.morf -freq ces-mixed2012.csv -test 0 -save_model model.pkl

  Load model + interactive:
    python generate.py -model model.pkl -source cze.morf -freq ces-mixed2012.csv

  Load model + segment text file:
    python generate.py -model model.pkl -freq ces-mixed2012.csv -text input.txt -save_to output.morf
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

from segmentation import (
    MorphSegmenter,
    load_frequency_map,
    load_segmented_words,
    load_model as load_seg_model,
    save_model as save_seg_model,
    train as train_seg,
    evaluate as evaluate_seg,
    print_evaluation as print_seg_evaluation,
)
from root_identification import (
    RootIdentifier,
    load_words_with_roots,
    load_model as load_root_model,
    save_model as _save_root_model,
    train as train_root,
    evaluate as evaluate_root,
    print_evaluation as print_root_evaluation,
)


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------


def _format_word_with_proba(word: str, gap_proba: List[Tuple[int, float]]) -> str:
    """Build a display string showing each letter with boundary probabilities."""
    parts: List[str] = []
    for i, ch in enumerate(word):
        parts.append(ch)
        if i < len(word) - 1:
            idx, prob = gap_proba[i]
            if prob >= 0.5:
                parts.append(f" |{prob:.3f}| ")
            elif prob >= 0.2:
                parts.append(f"({prob:.3f})")
    return "".join(parts)


def interactive_loop(
    seg_model: MorphSegmenter,
    root_model: RootIdentifier,
    source_path: Optional[str] = None,
) -> None:
    """Ask the user for words, show segmentation + root predictions."""
    print("Interactive mode. Enter a word to segment (empty line to quit).")
    while True:
        word = input("WORD> ").strip()
        if not word:
            break
        # Step 1: segmentation
        segmented = seg_model.segment(word)
        morphs = segmented.split()
        gap_proba = seg_model.predict_boundaries_with_proba(word)

        # Step 2: root identification
        annotated = root_model.annotate(morphs)

        print(f"  segmentation: {segmented}")
        print(f"  with roots:   {annotated}")
        print(f"  proba:        {_format_word_with_proba(word, gap_proba)}")

        if source_path is not None:
            correction = input("  CORRECT (Enter to accept)> ").strip()
            if correction:
                joined = "".join(re.split(r"\s+", correction.replace("@", "")))
                if joined != word:
                    print(f"  [ERR] Correction joins to '{joined}', expected '{word}'. Skipped.")
                    continue
                with open(source_path, "a", encoding="utf-8") as f:
                    f.write(correction + "\n")
                print(f"  [OK] Saved to {source_path}")


# ---------------------------------------------------------------------------
# Segment a text file
# ---------------------------------------------------------------------------


def segment_text_file(
    seg_model: MorphSegmenter,
    root_model: RootIdentifier,
    text_path: str,
    output_path: str,
) -> None:
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = re.findall(r"\b\w+\b", text.lower())
    print(f"Found {len(tokens)} word tokens, segmenting unique words...")

    unique_words = list(set(tokens))
    cache: Dict[str, str] = {}
    for i, word in enumerate(unique_words):
        segmented = seg_model.segment(word)
        morphs = segmented.split()
        annotated = root_model.annotate(morphs)
        cache[word] = annotated
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(unique_words)} unique words processed")
    print(f"Processed {len(unique_words)} unique words.")

    with open(output_path, "w", encoding="utf-8") as f:
        for word in tokens:
            f.write(cache[word] + "\n")
    print(f"Written {len(tokens)} tokens → {output_path}")


# ---------------------------------------------------------------------------
# Pipeline (callable from other modules)
# ---------------------------------------------------------------------------


def run_pipeline(
    source: str,
    freq: Optional[str] = None,
    folds: int = 5,
    n: int = 20,
    seed: int = 42,
    save_model: Optional[str] = None,
    save_root_model: Optional[str] = None,
    eval_only: bool = False,
    quiet: bool = False,
) -> dict:
    """Run training / evaluation pipeline and return statistics.

    Parameters mirror the CLI arguments. When *quiet* is True, stdout
    output is suppressed (useful when called from generate_multiple).

    Returns a dict with keys:
        lang            – language code derived from the source filename
        n_words         – number of annotated words
        seg_metrics     – segmentation evaluation dict (None if folds < 2)
        root_metrics    – root evaluation dict (None if folds < 2)
    """
    import io, contextlib

    # Validate paths
    for label, path in [("-source", source), ("-freq", freq)]:
        if path is not None and not os.path.isfile(path):
            raise FileNotFoundError(f"{label} file not found: {path}")

    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()

    with ctx:
        freq_map = None
        if freq:
            freq_map = load_frequency_map(freq)
            print(f"Loaded {len(freq_map)} frequency entries.")

        all_seg_words = load_segmented_words(source)
        all_root_data = load_words_with_roots(source)
        print(f"Loaded {len(all_seg_words)} annotated words from {source}")

        seg_metrics = None
        root_metrics = None

        # -- K-fold evaluation --
        if folds >= 2:
            print(f"\nSegmentation — {folds}-fold cross-validated evaluation:")
            seg_metrics = evaluate_seg(all_seg_words, freq_map, n_folds=folds, seed=seed)
            print_seg_evaluation(seg_metrics, n_worst=n)

            print(f"\nRoot identification — {folds}-fold cross-validated evaluation:")
            root_metrics = evaluate_root(all_root_data, freq_map, n_folds=folds, seed=seed)
            print_root_evaluation(root_metrics, n_worst=n)

        if not eval_only:
            seg_model = train_seg(all_seg_words, freq_map)
            print(f"\nTrained final segmentation model on all {len(all_seg_words)} words.")

            root_model_trained = train_root(all_root_data, freq_map)
            print(f"Trained final root identification model on all {len(all_root_data)} words.")

            if save_model:
                os.makedirs(os.path.dirname(save_model), exist_ok=True)
                save_seg_model(save_model, seg_model)
                print(f"Segmentation model saved to {save_model}")

            if save_root_model:
                os.makedirs(os.path.dirname(save_root_model), exist_ok=True)
                _save_root_model(save_root_model, root_model_trained)
                print(f"Root model saved to {save_root_model}")

    lang = os.path.splitext(os.path.basename(source))[0]

    return {
        "lang": lang,
        "n_words": len(all_seg_words),
        "has_freq": freq is not None,
        "seg_metrics": seg_metrics,
        "root_metrics": root_metrics,
        "log": buf.getvalue() if quiet else None,
    }


# ---------------------------------------------------------------------------
# Main (CLI)
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Morphological analysis pipeline (segmentation + root identification).")
    ap.add_argument("-source", help="Annotated source file (segmented words, one per line).")
    ap.add_argument("-freq", default=None, help="Corpus frequency CSV (Item,Frequency).")
    ap.add_argument("-folds", type=int, default=5, help="Number of cross-validation folds for evaluation (default: 5).")
    ap.add_argument("-n", type=int, default=20, help="Number of worst predictions to show (default: 20).")
    ap.add_argument("-seed", type=int, default=42, help="Random seed (default: 42).")
    ap.add_argument("-model", default=None, help="Load a saved segmentation model (pickle).")
    ap.add_argument("-save_model", default=None, help="Save trained segmentation model to this path.")
    ap.add_argument("-root_model", default=None, help="Load a saved root identification model (pickle).")
    ap.add_argument("-save_root_model", default=None, help="Save trained root model to this path.")
    ap.add_argument("-text", default=None, help="Segment words from a text file (requires -save_to).")
    ap.add_argument("-save_to", default=None, help="Output path for segmented text.")
    ap.add_argument("-eval_only", action="store_true", help="Only evaluate (K-fold), do not train a final model.")
    args = ap.parse_args()

    # --- Validate file paths early ---
    for label, path in [
        ("-source", args.source),
        ("-freq", args.freq),
        ("-model", args.model),
        ("-root_model", args.root_model),
        ("-text", args.text),
    ]:
        if path is not None and not os.path.isfile(path):
            ap.error(f"{label} file not found: {path}")

    # Load frequency map
    freq_map = None
    if args.freq:
        freq_map = load_frequency_map(args.freq)
        print(f"Loaded {len(freq_map)} frequency entries.")

    # --- Load or create root model ---
    if args.root_model is not None:
        root_model = load_root_model(args.root_model)
        if freq_map is not None:
            root_model.freq_map = freq_map
        print("Root identification model loaded.")
    else:
        root_model = RootIdentifier()  # untrained placeholder

    # --- Load existing segmentation model ---
    if args.model is not None:
        seg_model = load_seg_model(args.model)
        if freq_map is not None:
            seg_model.freq_map = freq_map
        print("Segmentation model loaded.")

        if args.text and args.save_to:
            segment_text_file(seg_model, root_model, args.text, args.save_to)
        else:
            interactive_loop(seg_model, root_model, source_path=args.source)
        return

    # --- Training mode → delegate to run_pipeline ---
    if args.source is None:
        ap.error("-source is required for training / evaluation.")

    run_pipeline(
        source=args.source,
        freq=args.freq,
        folds=args.folds,
        n=args.n,
        seed=args.seed,
        save_model=args.save_model,
        save_root_model=args.save_root_model,
        eval_only=args.eval_only,
        quiet=False,
    )


if __name__ == "__main__":
    main()
