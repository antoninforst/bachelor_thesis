"""
Compute per-file quality metrics for aggregated word-frequency files.

For each CSV in the input directory, reports:
  - foreign_script_pct: % of word types whose dominant Unicode script differs
                        from the script declared in the filename (e.g. Latin
                        words in a Cyrillic file)
  - long_outlier_pct : % of word types longer than the boxplot upper fence
                       (Q3 + 3·IQR on all word lengths — catches sentences
                       without spaces, URLs, etc.)
  - top20_long_pct   : % of the 20 most-frequent word types that are length
                       outliers within those 20 (Q3 + 1.5·IQR — tighter check)
  - punct_char_pct   : share of known punctuation characters out of ALL
                       characters (frequency-weighted), not per word type
  - program_pct      : % of word types that look like programming / web
                       artifacts ("www", "http", braces, backslash, "@",
                       multiple dots, multiple commas)

Results are written to results/1_process/3_truncate/quality.csv

Usage:
    python src/1_process/3_truncate/quality_check.py
    python src/1_process/3_truncate/quality_check.py --input-dir data/1_aggregated
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "data" / "2_annotated"
DEFAULT_OUTPUT = ROOT / "results" / "1_process" / "3_truncate" / "quality.csv"
SCRIPTS_CSV = ROOT / "metadata" / "scripts.csv"

sys.path.insert(0, str(ROOT / "src" / "1_process" / "1_filter"))
from script_check import ScriptDetector

csv.field_size_limit(10 * 1024 * 1024)

# Characters counted as "known punctuation" for the char-level metric
_KNOWN_PUNCT = set('".,?!;:/\\@#$%^&*~`<>|+=_()[]{}')


def _percentile(sorted_data: list[int | float], p: float) -> float:
    """Linear-interpolation percentile on *already sorted* data (0 ≤ p ≤ 100)."""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_data[0])
    k = (n - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, n - 1)
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def _boxplot_upper_fence(sorted_lengths: list[int], multiplier: float) -> float:
    """Return Q3 + multiplier·IQR on *already sorted* lengths."""
    q1 = _percentile(sorted_lengths, 25)
    q3 = _percentile(sorted_lengths, 75)
    iqr = q3 - q1
    return q3 + multiplier * iqr


def _load_iso_to_unicode_script(scripts_csv: Path) -> dict[str, str]:
    """Map ISO 15924 4-letter code -> Unicode script name (e.g. 'Latn' -> 'Latin')."""
    mapping: dict[str, str] = {}
    with open(scripts_csv, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2:
                mapping[row[1].strip()] = row[0].strip()
    return mapping


def _dominant_script(word: str, detector: ScriptDetector) -> str:
    """Return the most common Unicode script name in *word*."""
    counts: dict[str, int] = {}
    for ch in word:
        if not ch.isalpha():
            continue
        script = detector.classify(ch)
        if script == "Other":
            continue
        counts[script] = counts.get(script, 0) + 1
    if not counts:
        return "Common"
    return max(counts, key=counts.get)  # type: ignore[arg-type]


def _is_program_artifact(word: str) -> bool:
    """True if the word looks like a web / programming artefact."""
    wl = word.lower()
    if "www" in wl or "http" in wl:
        return True
    if "{" in word or "}" in word:
        return True
    if "\\" in word:
        return True
    if "@" in word:
        return True
    if word.count(".") > 1:
        return True
    if word.count(",") > 1:
        return True
    return False


# English stopwords used as a cross-language contamination signal
_ENG_STOPWORDS = frozenset({
    "the", "with", "there", "that", "will", "which", "after",
    "would", "people", "other", "where", "years", "because",
    "against", "however",
})


# ── Individual metric checks (each takes a ctx dict, returns metrics) ──

def _check_foreign_script(ctx):
    count = sum(ctx["is_foreign"])
    return {"foreign_script_pct": round(100.0 * count / ctx["n_types"], 4)}


def _check_long_outliers(ctx):
    return {
        "long_outlier_pct": round(100.0 * sum(ctx["is_long"]) / ctx["n_types"], 4),
        "long_threshold": round(ctx["long_threshold"], 1),
    }


def _check_top20_long(ctx):
    n_types, freqs, lengths = ctx["n_types"], ctx["freqs"], ctx["lengths"]
    top_n = min(20, n_types)
    top_indices = sorted(range(n_types), key=lambda i: -freqs[i])[:top_n]
    top_sorted = sorted(lengths[i] for i in top_indices)
    threshold = _boxplot_upper_fence(top_sorted, 1.5)
    count = sum(1 for i in top_indices if lengths[i] > threshold)
    return {
        "top20_long_pct": round(100.0 * count / top_n, 4),
        "top20_threshold": round(threshold, 1),
    }


def _check_punct_chars(ctx):
    total_chars = 0
    punct_chars = 0
    for word, freq in zip(ctx["words"], ctx["freqs"]):
        total_chars += len(word) * freq
        punct_chars += sum(1 for ch in word if ch in _KNOWN_PUNCT) * freq
    return {
        "punct_char_pct": round(100.0 * punct_chars / total_chars, 4) if total_chars else 0.0,
    }


def _check_program_artifacts(ctx):
    count = sum(ctx["is_program"])
    return {"program_pct": round(100.0 * count / ctx["n_types"], 4)}


def _check_corrupted(ctx):
    count = 0
    freq_sum = 0
    for i in range(ctx["n_types"]):
        if ctx["is_long"][i] or ctx["is_foreign"][i] or ctx["is_program"][i]:
            count += 1
            freq_sum += ctx["freqs"][i]
    total_freq = ctx["total_freq"]
    return {
        "corrupted_pct": round(100.0 * count / ctx["n_types"], 4),
        "corrupted_freq_pct": round(100.0 * freq_sum / total_freq, 4) if total_freq else 0.0,
    }


def _check_eng_stopwords(ctx):
    if ctx["lang_code"] == "eng":
        return {"eng_stopwords": 0}
    word_set = {w.lower() for w in ctx["words"]}
    return {"eng_stopwords": len(_ENG_STOPWORDS & word_set)}


_CHECKS = [
    _check_foreign_script,
    _check_long_outliers,
    _check_top20_long,
    _check_punct_chars,
    _check_program_artifacts,
    _check_corrupted,
    _check_eng_stopwords,
]

_EMPTY_CTX = {
    "words": ["x"], "freqs": [1], "lengths": [1],
    "is_foreign": [False], "is_program": [False], "is_long": [False],
    "n_types": 1, "total_freq": 1, "long_threshold": 0, "lang_code": "eng",
}


def _analyse_file(path: Path, expected_script_name: str | None, lang_code: str,
                  detector: ScriptDetector) -> dict[str, object]:
    """Return quality metrics for one word-frequency file."""
    words: list[str] = []
    freqs: list[int] = []
    total_freq = 0

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            try:
                freq = int(row[1])
            except ValueError:
                continue
            words.append(row[0])
            freqs.append(freq)
            total_freq += freq

    n_types = len(words)

    if n_types == 0:
        empty: dict[str, object] = {"total_types": 0, "total_frequency": 0}
        for check in _CHECKS:
            for key in check(_EMPTY_CTX):
                empty[key] = 0
        return empty

    # Pre-compute per-word arrays
    lengths = [len(w) for w in words]
    sorted_lengths = sorted(lengths)
    long_threshold = _boxplot_upper_fence(sorted_lengths, 3.0)

    is_foreign = [False] * n_types
    if expected_script_name:
        for i, w in enumerate(words):
            script = _dominant_script(w, detector)
            is_foreign[i] = script not in ("Common", expected_script_name)

    ctx = {
        "words": words, "freqs": freqs, "lengths": lengths,
        "is_foreign": is_foreign,
        "is_program": [_is_program_artifact(w) for w in words],
        "is_long": [l > long_threshold for l in lengths],
        "n_types": n_types, "total_freq": total_freq,
        "long_threshold": long_threshold, "lang_code": lang_code,
    }

    result = {"total_types": n_types, "total_frequency": total_freq}
    for check in _CHECKS:
        result.update(check(ctx))
    return result


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute per-file quality metrics.")
    p.add_argument(
        "langs", nargs="*", metavar="LANG",
        help="Language codes to limit processing (default: all files).",
    )
    p.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT,
        help=f"Directory with word-frequency CSVs (default: {DEFAULT_INPUT.relative_to(ROOT)}).",
    )
    p.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT.relative_to(ROOT)}).",
    )
    return p.parse_args(argv)


def _extract_script_from_stem(stem: str) -> str | None:
    """'eng_Latn' -> 'Latn', 'jpn' -> None."""
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and len(parts[1]) == 4 and parts[1][0].isupper():
        return parts[1]
    return None


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    input_dir: Path = args.input_dir
    output: Path = args.output

    files = sorted(input_dir.glob("*.csv"))
    if args.langs:
        lang_filter = {l.lower() for l in args.langs}
        files = [f for f in files if f.stem.rsplit("_", 1)[0].lower() in lang_filter]
    if not files:
        print(f"No CSV files found in {input_dir}")
        sys.exit(1)

    iso_to_script = _load_iso_to_unicode_script(SCRIPTS_CSV)
    detector = ScriptDetector.from_csv(SCRIPTS_CSV)

    rows: list[dict[str, object]] = []
    for path in tqdm(files, desc="Analysing", unit="file"):
        stem = path.stem
        iso_code = _extract_script_from_stem(stem)
        lang_code = stem.rsplit("_", 1)[0] if iso_code else stem
        expected_script = iso_to_script.get(iso_code, None) if iso_code else None

        metrics = _analyse_file(path, expected_script, lang_code, detector)
        metrics["file"] = path.name
        metrics["language"] = lang_code
        metrics["script"] = iso_code or ""
        rows.append(metrics)

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file", "language", "script",
        "foreign_script_pct",
        "long_outlier_pct", "top20_long_pct", "punct_char_pct", "program_pct",
        "corrupted_pct",
        "corrupted_freq_pct",
        "eng_stopwords",
        "long_threshold", "top20_threshold",
        "total_types", "total_frequency",
    ]
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
