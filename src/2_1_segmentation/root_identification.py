#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Root identification module.

Given an already-segmented word (list of morphs), identifies which morph(s)
are root(s). In the training data, roots are marked with '@' at the beginning
of the morph (e.g. 'za @hrad a' means 'hrad' is the root).

Uses per-morph LogisticRegression with features such as morph length,
position, frequency in a corpus, and character n-grams.
"""

import math
import pickle
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOWELS = set(
    # Latin (lower + upper, incl. diacritics)
    "aeiouyàáâãäåāăąæèéêëēĕėęěìíîïĩīĭįıòóôõöøōŏőœùúûüũūŭůűųýÿ"
    "AEIOUYÀÁÂÃÄÅĀĂĄÆÈÉÊËĒĔĖĘĚÌÍÎÏĨĪĬĮÒÓÔÕÖØŌŎŐŒÙÚÛÜŨŪŬŮŰŲÝŸ"
    # Greek
    "αεηιουωάέήίόύώϊϋΐΰΑΕΗΙΟΥΩΆΈΉΊΌΎΏ"
    # Cyrillic
    "аеёиоуыэюяіієїәөүАЕЁИОУЫЭЮЯІЄЇӘӨҮ"
    # Devanagari (independent vowels + matras)
    "अआइईउऊऋएऐओऔािीुूृेैोौ"
    # Telugu (independent vowels + matras)
    "అఆఇఈఉఊఋఎఏఐఒఓఔాిీుూృెేైొోౌ"
    # Georgian
    "აეიოუ"
    # Armenian
    "աէըիոԱԷԸԻՈ"
    # Japanese kana vowels
    "あいうえおアイウエオ"
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_words_with_roots(path: str) -> List[Tuple[List[str], List[bool]]]:
    """Load annotated file and extract morph lists with root labels.

    Each line is a segmented word with morphs separated by spaces.
    Root morphs are prefixed with '@'.

    Returns a list of (morphs, is_root) tuples, where:
      - morphs: list of morph strings (without '@')
      - is_root: list of booleans indicating whether each morph is a root
    """
    result = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            raw_morphs = re.split(r"\s+", line)
            morphs = []
            is_root = []
            for m in raw_morphs:
                if m.startswith("@"):
                    morphs.append(m[1:])
                    is_root.append(True)
                else:
                    morphs.append(m)
                    is_root.append(False)
            result.append((morphs, is_root))
    return result


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_morph_features(
    morphs: List[str],
    morph_idx: int,
    freq_map: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    """Extract features for a single morph within a segmented word."""
    morph = morphs[morph_idx]
    n_morphs = len(morphs)
    word = "".join(morphs)
    word_len = len(word)

    feats: Dict[str, object] = {}

    # --- morph identity / length ---
    feats["morph_len"] = len(morph)
    feats["morph_len_ratio"] = len(morph) / word_len if word_len > 0 else 0.0

    # --- positional features ---
    feats["n_morphs"] = n_morphs
    feats["morph_idx"] = morph_idx
    feats["norm_pos"] = round(morph_idx / max(n_morphs - 1, 1), 2)
    feats["is_first"] = morph_idx == 0
    feats["is_last"] = morph_idx == n_morphs - 1
    feats["is_only"] = n_morphs == 1
    # character offset of morph start within the word
    char_offset = sum(len(morphs[j]) for j in range(morph_idx))
    feats["char_offset"] = char_offset
    feats["norm_char_offset"] = round(char_offset / max(word_len - 1, 1), 2)

    # --- character n-gram features ---
    for k in (1, 2, 3):
        feats[f"first{k}"] = morph[:k] if len(morph) >= k else morph
        feats[f"last{k}"] = morph[-k:] if len(morph) >= k else morph

    # --- vowel content ---
    n_vowels = sum(1 for ch in morph if ch in VOWELS)
    feats["n_vowels"] = n_vowels
    feats["has_vowel"] = n_vowels > 0
    feats["vowel_ratio"] = n_vowels / len(morph) if len(morph) > 0 else 0.0

    # --- is this the longest morph? (language-agnostic root signal) ---
    max_morph_len = max(len(m) for m in morphs)
    feats["is_longest"] = len(morph) == max_morph_len
    feats["len_vs_max"] = len(morph) / max_morph_len if max_morph_len > 0 else 0.0

    # --- context: neighbour lengths ---
    if morph_idx > 0:
        prev_len = len(morphs[morph_idx - 1])
        feats["prev_len"] = prev_len
        feats["longer_than_prev"] = len(morph) > prev_len
    else:
        feats["prev_len"] = 0
        feats["longer_than_prev"] = False

    if morph_idx < n_morphs - 1:
        next_len = len(morphs[morph_idx + 1])
        feats["next_len"] = next_len
        feats["longer_than_next"] = len(morph) > next_len
    else:
        feats["next_len"] = 0
        feats["longer_than_next"] = False

    # --- frequency-based features ---
    if freq_map is not None:
        morph_freq = freq_map.get(morph.lower(), 0)
        word_freq = freq_map.get(word.lower(), 0)
        feats["log_morph_freq"] = math.log1p(morph_freq)
        feats["morph_in_wordlist"] = morph_freq > 0
        feats["log_word_freq"] = math.log1p(word_freq)
        # frequency of the morph relative to how big it is — roots should be
        # recognisable words in the corpus
        if len(morph) >= 3:
            feats["freq_per_char"] = math.log1p(morph_freq) / len(morph)
        else:
            feats["freq_per_char"] = 0.0

    return feats


# ---------------------------------------------------------------------------
# Building training data
# ---------------------------------------------------------------------------


def build_examples(
    data: List[Tuple[List[str], List[bool]]],
    freq_map: Optional[Dict[str, int]] = None,
) -> Tuple[List[Dict[str, object]], np.ndarray]:
    """Convert loaded root data into (feature_dicts, labels) for all morphs."""
    X: List[Dict[str, object]] = []
    y: List[int] = []
    for morphs, is_root in data:
        for idx in range(len(morphs)):
            X.append(extract_morph_features(morphs, idx, freq_map))
            y.append(1 if is_root[idx] else 0)
    return X, np.array(y, dtype=int)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class RootIdentifier:
    """Root identification model.

    Takes a list of morphs (from segmentation) and predicts which are roots.
    When no model is loaded (vec/clf are None), returns all-False (no roots).
    """

    def __init__(
        self,
        vec: Optional[DictVectorizer] = None,
        clf: Optional[LogisticRegression] = None,
        freq_map: Optional[Dict[str, int]] = None,
    ):
        self.vec = vec
        self.clf = clf
        self.freq_map = freq_map

    @property
    def is_trained(self) -> bool:
        return self.vec is not None and self.clf is not None

    def predict_roots(self, morphs: List[str]) -> List[bool]:
        """Predict which morphs are roots. Guarantees at least one root."""
        if len(morphs) == 0:
            return []
        if not self.is_trained:
            # Untrained fallback: mark the longest morph as root
            max_len = max(len(m) for m in morphs)
            return [len(m) == max_len for m in morphs]
        feats = [extract_morph_features(morphs, i, self.freq_map) for i in range(len(morphs))]
        X = self.vec.transform(feats)
        preds = list(self.clf.predict(X))
        # Guarantee at least one root: if none predicted, pick highest-proba morph
        if not any(p == 1 for p in preds):
            proba = self.clf.predict_proba(X)
            cls1 = list(self.clf.classes_).index(1)
            best = int(np.argmax(proba[:, cls1]))
            preds[best] = 1
        return [bool(p) for p in preds]

    def predict_roots_with_proba(self, morphs: List[str]) -> List[Tuple[int, float]]:
        """Return list of (morph_idx, probability_of_root) for all morphs."""
        if not self.is_trained or len(morphs) == 0:
            return [(i, 0.0) for i in range(len(morphs))]
        feats = [extract_morph_features(morphs, i, self.freq_map) for i in range(len(morphs))]
        X = self.vec.transform(feats)
        proba = self.clf.predict_proba(X)
        cls1 = list(self.clf.classes_).index(1)
        return [(i, proba[i, cls1]) for i in range(len(morphs))]

    def annotate(self, morphs: List[str]) -> str:
        """Return segmented word string with root morphs marked by '@'."""
        roots = self.predict_roots(morphs)
        parts = []
        for morph, is_root in zip(morphs, roots):
            if is_root:
                parts.append("@" + morph)
            else:
                parts.append(morph)
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    data: List[Tuple[List[str], List[bool]]],
    freq_map: Optional[Dict[str, int]] = None,
) -> RootIdentifier:
    X_dicts, y = build_examples(data, freq_map)
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dicts)
    clf = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        C=1.0,
        max_iter=5000,
    )
    clf.fit(X, y)
    return RootIdentifier(vec, clf, freq_map)


# ---------------------------------------------------------------------------
# Evaluation (K-fold cross-validated)
# ---------------------------------------------------------------------------


def _annotate_morphs(morphs: List[str], is_root: List[bool]) -> str:
    """Build annotation string: root morphs prefixed with '@'."""
    return " ".join(("@" + m if r else m) for m, r in zip(morphs, is_root))


def _levenshtein(a: List[int], b: List[int]) -> int:
    """Levenshtein distance between two sequences of positions."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def evaluate(
    data: List[Tuple[List[str], List[bool]]],
    freq_map: Optional[Dict[str, int]] = None,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """K-fold cross-validated evaluation of root identification.

    Every word is scored exactly once while held out.  Quality is measured
    by Levenshtein distance between gold and predicted root-position
    sequences: score = 1 \u2212 dist / max(|gold|, |pred|).

    Returns aggregate metrics and a per-word list of
    (distance, gold, predicted) sorted by descending distance.
    """
    data_arr = np.array(data, dtype=object)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    word_results: List[Tuple[float, int, str, str]] = []
    total_dist = 0
    total_score = 0.0
    total_loss = 0.0

    for train_idx, test_idx in kf.split(data_arr):
        train_data = [data[i] for i in train_idx]
        test_data = [data[i] for i in test_idx]

        X_dicts, y = build_examples(train_data, freq_map)
        vec = DictVectorizer(sparse=True)
        X = vec.fit_transform(X_dicts)
        clf = LogisticRegression(
            solver="lbfgs", class_weight="balanced", C=1.0, max_iter=5000,
        )
        clf.fit(X, y)
        cls1 = list(clf.classes_).index(1)

        for morphs, is_root in test_data:
            true_labels = [1 if r else 0 for r in is_root]
            feats = [extract_morph_features(morphs, i, freq_map) for i in range(len(morphs))]
            Xw = vec.transform(feats)
            proba = clf.predict_proba(Xw)
            pred_labels = (proba[:, cls1] >= 0.5).astype(int).tolist()

            # Guarantee at least one root
            if not any(p == 1 for p in pred_labels):
                best = int(np.argmax(proba[:, cls1]))
                pred_labels[best] = 1

            gold_positions = [i for i, v in enumerate(true_labels) if v == 1]
            pred_positions = [i for i, v in enumerate(pred_labels) if v == 1]
            dist = _levenshtein(gold_positions, pred_positions)
            max_len = max(len(gold_positions), len(pred_positions))
            score = 1.0 - dist / max_len if max_len > 0 else 1.0

            # Per-morph cross-entropy loss (for outlier ranking)
            eps = 1e-15
            loss = 0.0
            for m_i, label in enumerate(true_labels):
                p_true = proba[m_i, cls1] if label == 1 else 1 - proba[m_i, cls1]
                loss += -math.log(max(p_true, eps))
            avg_loss = loss / len(true_labels) if true_labels else 0.0

            total_dist += dist
            total_score += score
            total_loss += avg_loss

            gold_str = _annotate_morphs(morphs, is_root)
            pred_roots = [bool(p) for p in pred_labels]
            pred_str = _annotate_morphs(morphs, pred_roots)
            word_results.append((avg_loss, dist, gold_str, pred_str))

    word_results.sort(key=lambda x: x[0], reverse=True)

    n_words = len(word_results)
    n_correct = sum(1 for _, d, _, _ in word_results if d == 0)

    return {
        "avg_levenshtein": total_dist / n_words if n_words else 0.0,
        "levenshtein_score": total_score / n_words if n_words else 0.0,
        "avg_loss": total_loss / n_words if n_words else 0.0,
        "word_accuracy": n_correct / n_words if n_words else 0.0,
        "n_words": n_words,
        "n_correct": n_correct,
        "word_results": word_results,
    }


def print_evaluation(metrics: dict, n_worst: int = 20) -> None:
    n = metrics["n_words"]
    n_correct = metrics["n_correct"]
    n_wrong = n - n_correct
    print(f"  Words evaluated : {n}")
    print(f"  Avg Levenshtein : {metrics['avg_levenshtein']:.4f}")
    print(f"  Lev score       : {metrics['levenshtein_score']:.4f}")
    print(f"  Avg loss        : {metrics['avg_loss']:.4f}")
    print(f"  Word accuracy   : {metrics['word_accuracy']:.4f}")
    print(f"  Correct / Wrong : {n_correct} / {n_wrong}")

    word_results = metrics["word_results"]
    wrong = [(loss, dist, gold, pred) for loss, dist, gold, pred in word_results if dist > 0]
    show = min(n_worst, len(wrong))
    if show > 0:
        print(f"\n  Top {show} most uncertain wrong predictions (by cross-entropy loss):\n")
        print(f"  {'Loss':>8s}  {'Dist':>4s}  {'Gold':<30s}  Suggested")
        print(f"  {'----':>8s}  {'----':>4s}  {'----':<30s}  ---------")
        for loss, dist, gold, pred in wrong[:show]:
            print(f"  {loss:8.4f}  {dist:4d}  {gold:<30s}  {pred}")


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_model(path: str, model: RootIdentifier) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> RootIdentifier:
    with open(path, "rb") as f:
        return pickle.load(f)
