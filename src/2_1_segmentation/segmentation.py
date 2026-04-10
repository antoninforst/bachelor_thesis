#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Morphological segmentation module.

Formulates segmentation as binary classification of gaps between letters.
Uses character-context features and corpus frequency features.
"""

import csv
import math
import pickle
import re
import unicodedata
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# Fast DictVectorizer transform (bypasses sklearn validation overhead)
# ---------------------------------------------------------------------------


def _fast_transform(vec: DictVectorizer, feature_dicts: List[Dict[str, object]]) -> csr_matrix:
    """Build sparse matrix from feature dicts using vec.vocabulary_ directly.

    ~3-5x faster than vec.transform() for large batches because it skips
    sklearn's type checks, tag resolution, and Python-level isinstance overhead.
    """
    vocab = vec.vocabulary_
    dtype = vec.dtype
    n_features = len(vocab)
    n_samples = len(feature_dicts)
    data_list: List[float] = []
    idx_list: List[int] = []
    indptr = np.empty(n_samples + 1, dtype=np.int32)
    indptr[0] = 0
    da = data_list.append
    ia = idx_list.append
    vg = vocab.get  # local ref

    for row_i, fd in enumerate(feature_dicts):
        for key, val in fd.items():
            if val.__class__ is str:
                idx = vg(f"{key}={val}")
                if idx is not None:
                    da(1.0); ia(idx)
            elif val.__class__ is bool:
                idx = vg(f"{key}={val}")
                if idx is not None:
                    da(1.0); ia(idx)
            else:
                idx = vg(key)
                if idx is not None:
                    da(val); ia(idx)
        indptr[row_i + 1] = len(idx_list)

    return csr_matrix(
        (np.array(data_list, dtype=dtype),
         np.array(idx_list, dtype=np.int32),
         indptr),
        shape=(n_samples, n_features),
    )
# Constants
# ---------------------------------------------------------------------------

_BASE_VOWELS = set(
    # Latin base vowels (+ ligatures / special letters that don't decompose)
    "aeiouyæœı"
    # Greek
    "αεηιουω"
    # Cyrillic
    "аеиоуыэюяіієәөү"
    # Devanagari (independent vowels + matras)
    "अआइईउऊऋएऐओऔािीुूृेैोौ"
    # Telugu (independent vowels + matras)
    "అఆఇఈఉఊఋఎఏఐఒఓఔాిీుూృెేైొోౌ"
    # Georgian
    "აეიოუ"
    # Armenian
    "աէըիո"
    # Japanese kana vowels
    "あいうえおアイウエオ"
)


@lru_cache(maxsize=4096)
def _strip_diacritics(ch: str) -> str:
    """Remove combining marks from a character, returning the base letter."""
    return "".join(c for c in unicodedata.normalize("NFD", ch)
                   if unicodedata.category(c) != "Mn")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_segmented_words(path: str) -> List[str]:
    """Load annotated file where each line is a segmented word (morphs separated by spaces).
    Lines starting with '#' and blank lines are skipped.
    Root markers '@' are stripped so the data is clean for segmentation training."""
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                # Strip root markers — not relevant for segmentation
                line = line.replace("@", "")
                words.append(line)
    return words


def load_frequency_map(path: str) -> Dict[str, int]:
    """Load a CSV wordlist with columns Item,Frequency. Returns {word: freq}."""
    freq: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                word = row[0].strip()
                try:
                    freq[word] = int(row[1].strip())
                except ValueError:
                    continue
    return freq


def load_affix_map(path: str) -> Dict[str, int]:
    """Load a .prep or .post CSV (affix,score). Returns {affix: score}."""
    result: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2:
                try:
                    result[row[0]] = int(row[1])
                except ValueError:
                    continue
    return result


# ---------------------------------------------------------------------------
# Segmented word → raw word + boundary labels
# ---------------------------------------------------------------------------


def segmented_to_raw_and_boundaries(segmented: str) -> Tuple[str, List[int]]:
    """Convert a segmented word like 'běl o s kv ouc í' into
    raw word 'běloskvoucí' and a list of boundary positions (0-indexed gap ids).

    Gap i sits between raw[i] and raw[i+1], so there are len(raw)-1 gaps.
    A boundary label of 1 at gap i means there is a morph boundary after raw[i].
    """
    morphs = re.split(r"\s+", segmented)
    raw = "".join(morphs)
    boundaries: List[int] = []
    pos = 0
    for morph in morphs[:-1]:
        pos += len(morph)
        # gap index = pos - 1 (gap between raw[pos-1] and raw[pos])
        if 1 <= pos <= len(raw) - 1:
            boundaries.append(pos - 1)
    return raw, boundaries


def build_boundary_labels(raw: str, boundaries: List[int]) -> List[int]:
    """Return a list of 0/1 labels for each of the len(raw)-1 gaps."""
    bset = set(boundaries)
    return [1 if i in bset else 0 for i in range(len(raw) - 1)]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _safe_char(word: str, idx: int) -> str:
    if idx < 0:
        return "^"
    if idx >= len(word):
        return "$"
    return word[idx]


@lru_cache(maxsize=4096)
def _is_vowel(ch: str) -> bool:
    return _strip_diacritics(ch).lower() in _BASE_VOWELS


@lru_cache(maxsize=4096)
def _vc_pattern(ch: str) -> str:
    """Return 'V' for vowels, 'C' for consonants, or the char itself for boundary markers."""
    if ch in ("^", "$"):
        return ch
    return "V" if _is_vowel(ch) else "C"


def extract_gap_features(
    word: str,
    gap_idx: int,
    freq_map: Optional[Dict[str, int]] = None,
    prefix_map: Optional[Dict[str, int]] = None,
    suffix_map: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    """Extract features for the gap between word[gap_idx] and word[gap_idx+1].

    gap_idx is in range [0, len(word)-2].
    """
    n = len(word)

    # Characters around the gap (wider window: 3 left, 3 right)
    l3 = _safe_char(word, gap_idx - 2)
    l2 = _safe_char(word, gap_idx - 1)
    l1 = _safe_char(word, gap_idx)
    r1 = _safe_char(word, gap_idx + 1)
    r2 = _safe_char(word, gap_idx + 2)
    r3 = _safe_char(word, gap_idx + 3)

    feats: Dict[str, object] = {}

    # --- character identity features (wider window) ---
    feats["l3"] = l3
    feats["l2"] = l2
    feats["l1"] = l1
    feats["r1"] = r1
    feats["r2"] = r2
    feats["r3"] = r3

    # bigrams
    feats["bi_left"] = l2 + l1
    feats["bi_cross"] = l1 + r1
    feats["bi_right"] = r1 + r2

    # trigrams
    feats["tri_center"] = l1 + r1 + r2
    feats["tri_left"] = l2 + l1 + r1
    feats["tri_wide"] = l3 + l2 + l1
    feats["tri_right"] = r1 + r2 + r3

    # vowel/consonant pattern around gap
    vc = _vc_pattern(l2) + _vc_pattern(l1) + _vc_pattern(r1) + _vc_pattern(r2)
    feats["vc_pattern"] = vc
    feats["l1_vowel"] = _is_vowel(l1)
    feats["r1_vowel"] = _is_vowel(r1)
    feats["vc_transition"] = _vc_pattern(l1) + _vc_pattern(r1)

    # prefix/suffix length features (NOT raw strings — those cause feature explosion)
    prefix_len = gap_idx + 1
    suffix_len = n - gap_idx - 1
    feats["prefix_len"] = prefix_len
    feats["suffix_len"] = suffix_len

    # short suffix/prefix n-grams (generalizable, unlike full prefix/suffix)
    prefix = word[: gap_idx + 1]
    suffix = word[gap_idx + 1 :]
    for k in (1, 2, 3):
        feats[f"prefix_last{k}"] = prefix[-k:] if len(prefix) >= k else prefix
        feats[f"suffix_first{k}"] = suffix[:k] if len(suffix) >= k else suffix

    # positional features
    feats["word_len"] = n
    feats["norm_pos"] = round(gap_idx / (n - 1), 2) if n > 1 else 0.0
    feats["at_start"] = gap_idx == 0
    feats["at_end"] = gap_idx == n - 2
    feats["near_start"] = gap_idx <= 2
    feats["near_end"] = gap_idx >= n - 4

    # --- frequency-based features ---
    if freq_map is not None:
        whole_freq = freq_map.get(word.lower(), 0)
        left_freq = freq_map.get(prefix.lower(), 0)
        right_freq = freq_map.get(suffix.lower(), 0)

        feats["log_whole_freq"] = math.log1p(whole_freq)
        feats["log_left_freq"] = math.log1p(left_freq)
        feats["log_right_freq"] = math.log1p(right_freq)
        feats["left_in_wordlist"] = left_freq > 0
        feats["right_in_wordlist"] = right_freq > 0
        if whole_freq > 0:
            feats["freq_ratio"] = math.log1p(left_freq + right_freq) / math.log1p(whole_freq)
        else:
            feats["freq_ratio"] = math.log1p(left_freq + right_freq)

    # --- affix score features ---
    if prefix_map is not None:
        feats["prefix_score"] = prefix_map.get(prefix.lower(), 0)
    if suffix_map is not None:
        feats["suffix_score"] = suffix_map.get(suffix.lower(), 0)
    if prefix_map is not None and suffix_map is not None:
        ps = feats["prefix_score"]
        ss = feats["suffix_score"]
        feats["affix_sum"] = ps + ss
        feats["affix_diff"] = ps - ss

    return feats


# ---------------------------------------------------------------------------
# Building training data
# ---------------------------------------------------------------------------


def build_examples(
    segmented_words: List[str],
    freq_map: Optional[Dict[str, int]] = None,
    prefix_map: Optional[Dict[str, int]] = None,
    suffix_map: Optional[Dict[str, int]] = None,
) -> Tuple[List[Dict[str, object]], np.ndarray]:
    """Convert a list of segmented words into (feature_dicts, labels) for all gaps."""
    X: List[Dict[str, object]] = []
    y: List[int] = []
    for seg in segmented_words:
        raw, boundaries = segmented_to_raw_and_boundaries(seg)
        if len(raw) < 2:
            continue
        labels = build_boundary_labels(raw, boundaries)
        for gap_idx in range(len(raw) - 1):
            X.append(extract_gap_features(raw, gap_idx, freq_map, prefix_map, suffix_map))
            y.append(labels[gap_idx])
    return X, np.array(y, dtype=int)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class MorphSegmenter:
    def __init__(self, vec: DictVectorizer, clf: LogisticRegression,
                 freq_map: Optional[Dict[str, int]] = None,
                 prefix_map: Optional[Dict[str, int]] = None,
                 suffix_map: Optional[Dict[str, int]] = None):
        self.vec = vec
        self.clf = clf
        self.freq_map = freq_map
        self.prefix_map = prefix_map
        self.suffix_map = suffix_map

    def predict_boundaries(self, word: str) -> List[int]:
        """Return list of gap indices predicted as morph boundaries."""
        if len(word) < 2:
            return []
        feats = [extract_gap_features(word, i, self.freq_map, self.prefix_map, self.suffix_map) for i in range(len(word) - 1)]
        X = self.vec.transform(feats)
        preds = self.clf.predict(X)
        return [i for i, flag in enumerate(preds) if flag == 1]

    def predict_boundaries_with_proba(self, word: str) -> List[Tuple[int, float]]:
        """Return list of (gap_idx, probability) for all gaps."""
        if len(word) < 2:
            return []
        feats = [extract_gap_features(word, i, self.freq_map, self.prefix_map, self.suffix_map) for i in range(len(word) - 1)]
        X = self.vec.transform(feats)
        proba = self.clf.predict_proba(X)
        # column index for class 1
        cls1 = list(self.clf.classes_).index(1)
        return [(i, proba[i, cls1]) for i in range(len(word) - 1)]

    def segment(self, word: str) -> str:
        """Return the segmented form of a word (morphs separated by spaces)."""
        boundaries = self.predict_boundaries(word)
        parts: List[str] = []
        prev = 0
        for b in sorted(boundaries):
            parts.append(word[prev : b + 1])
            prev = b + 1
        parts.append(word[prev:])
        return " ".join(parts)

    def segment_batch(self, words: List[str]) -> List[str]:
        """Segment a list of words in one vectorize+predict call."""
        if not words:
            return []
        # Build features for all gaps of all words, track word boundaries
        all_feats: List[Dict[str, object]] = []
        word_gap_counts: List[int] = []  # how many gaps per word
        for word in words:
            n_gaps = max(len(word) - 1, 0)
            word_gap_counts.append(n_gaps)
            for i in range(n_gaps):
                all_feats.append(extract_gap_features(word, i, self.freq_map, self.prefix_map, self.suffix_map))

        if not all_feats:
            return list(words)

        X = _fast_transform(self.vec, all_feats)
        scores = X @ self.clf.coef_.T + self.clf.intercept_
        all_preds = (scores.ravel() > 0).astype(int) if scores.shape[1] == 1 else np.argmax(scores, axis=1)

        # Split predictions back per word
        results: List[str] = []
        offset = 0
        for word, n_gaps in zip(words, word_gap_counts):
            if n_gaps == 0:
                results.append(word)
            else:
                preds = all_preds[offset : offset + n_gaps]
                parts: List[str] = [word[0]]
                for i, flag in enumerate(preds):
                    if flag:
                        parts.append(" ")
                    parts.append(word[i + 1])
                results.append("".join(parts))
            offset += n_gaps
        return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    segmented_words: List[str],
    freq_map: Optional[Dict[str, int]] = None,
    prefix_map: Optional[Dict[str, int]] = None,
    suffix_map: Optional[Dict[str, int]] = None,
) -> MorphSegmenter:
    X_dicts, y = build_examples(segmented_words, freq_map, prefix_map, suffix_map)
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dicts)
    clf = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        C=1.0,
        max_iter=5000,
    )
    clf.fit(X, y)
    return MorphSegmenter(vec, clf, freq_map, prefix_map, suffix_map)


# ---------------------------------------------------------------------------
# Evaluation (K-fold cross-validated)
# ---------------------------------------------------------------------------


def _boundaries_to_segmented(raw: str, pred_labels: List[int]) -> str:
    """Reconstruct segmented string from raw word and 0/1 gap labels."""
    parts: List[str] = [raw[0]]
    for i, label in enumerate(pred_labels):
        if label:
            parts.append(" ")
        parts.append(raw[i + 1])
    return "".join(parts)


def _levenshtein(a: List[int], b: List[int]) -> int:
    """Levenshtein distance between two sequences of boundary positions."""
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
    segmented_words: List[str],
    freq_map: Optional[Dict[str, int]] = None,
    prefix_map: Optional[Dict[str, int]] = None,
    suffix_map: Optional[Dict[str, int]] = None,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """K-fold cross-validated evaluation of segmentation.

    Every word is scored exactly once while held out. The segmentation
    quality is measured by Levenshtein distance between gold and predicted
    boundary-position sequences: score = 1 − dist / max(|gold|, |pred|).

    Returns aggregate metrics and a per-word list of
    (distance, gold, predicted) sorted by descending distance.
    """
    words = np.array(segmented_words, dtype=object)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    word_results: List[Tuple[float, int, str, str]] = []  # (loss, lev_dist, gold, predicted)
    total_dist = 0
    total_score = 0.0
    total_loss = 0.0

    for train_idx, test_idx in kf.split(words):
        train_words = words[train_idx].tolist()
        test_words = words[test_idx].tolist()

        X_dicts, y = build_examples(train_words, freq_map, prefix_map, suffix_map)
        vec = DictVectorizer(sparse=True)
        X = vec.fit_transform(X_dicts)
        clf = LogisticRegression(
            solver="lbfgs", class_weight="balanced", C=1.0, max_iter=5000,
        )
        clf.fit(X, y)
        cls1 = list(clf.classes_).index(1)

        for seg in test_words:
            raw, boundaries = segmented_to_raw_and_boundaries(seg)
            if len(raw) < 2:
                continue
            true_labels = build_boundary_labels(raw, boundaries)
            feats = [extract_gap_features(raw, i, freq_map, prefix_map, suffix_map) for i in range(len(raw) - 1)]
            Xw = vec.transform(feats)
            proba = clf.predict_proba(Xw)
            pred_labels = (proba[:, cls1] >= 0.5).astype(int).tolist()

            gold_bounds = [i for i, v in enumerate(true_labels) if v == 1]
            pred_bounds = [i for i, v in enumerate(pred_labels) if v == 1]
            dist = _levenshtein(gold_bounds, pred_bounds)
            max_len = max(len(gold_bounds), len(pred_bounds))
            score = 1.0 - dist / max_len if max_len > 0 else 1.0

            # Per-gap cross-entropy loss (for outlier ranking)
            eps = 1e-15
            loss = 0.0
            for gap_i, label in enumerate(true_labels):
                p_true = proba[gap_i, cls1] if label == 1 else 1 - proba[gap_i, cls1]
                loss += -math.log(max(p_true, eps))
            avg_loss = loss / len(true_labels)

            total_dist += dist
            total_score += score
            total_loss += avg_loss

            pred_seg = _boundaries_to_segmented(raw, pred_labels)
            word_results.append((avg_loss, dist, seg, pred_seg))

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


def save_model(path: str, model: MorphSegmenter) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> MorphSegmenter:
    with open(path, "rb") as f:
        return pickle.load(f)
