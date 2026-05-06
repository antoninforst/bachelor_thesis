from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

SEGMENTED_DIR = _PROJECT_ROOT / "data" / "2_segmented"
RESULTS_DIR = _PROJECT_ROOT / "results" / "3_analyze"

COLUMNS = ["word", "frequency", "segmentation", "annotation"]


@dataclass(frozen=True)
class Row:
    word: str
    frequency: float
    segmentation: list[str]
    annotation: list[str]


class Frequency:
    def __init__(self):
        self.data: dict[str, tuple[float, int]] = {}
        self.total_count: float = 0
        self.record_count: int = 0
        self.fhepax_count: int = 0
        self.hepax_count: int = 0

    def add(self, key: str, count: float = 1):
        if key in self.data:
            f, c = self.data[key]
            if f == 1:
                self.fhepax_count -= 1
            if c == 1:
                self.hepax_count -= 1
            self.data[key] = (f + count, c + 1)
        else:
            self.data[key] = (count, 1)
            if count == 1:
                self.fhepax_count += 1
            self.hepax_count += 1
        self.total_count += count
        self.record_count += 1

    def get_unique_count(self) -> int:
        return len(self.data)

    def get_total_count(self) -> float:
        return self.total_count

    def get_entropy(self) -> float:
        total = self.total_count
        if total == 0:
            return 0.0
        s = 0.0
        log2 = math.log2
        for f, _ in self.data.values():
            p = f / total
            s -= p * log2(p)
        return s

    def get_type_entropy(self) -> float:
        """Entropy based on number of word types each key appears in."""
        total = self.record_count
        if total == 0:
            return 0.0
        s = 0.0
        log2 = math.log2
        for _, c in self.data.values():
            p = c / total
            s -= p * log2(p)
        return s

    def get_hepax_count(self) -> int:
        return self.hepax_count

    def get_freq_hepax_count(self) -> int:
        return self.fhepax_count


MetricFn = Callable[[Frequency], Any]


# --- Frequency metrics ---

def metric_count(freq: Frequency) -> int:
    return freq.get_unique_count()


def metric_total_frequency(freq: Frequency) -> float:
    return freq.get_total_count()


def metric_ttr(freq: Frequency) -> float:
    total = freq.get_total_count()
    return freq.get_unique_count() / total if total else 0.0


def metric_entropy(freq: Frequency) -> float:
    return freq.get_entropy()


def metric_perplexity(freq: Frequency) -> float:
    return 2 ** freq.get_entropy()


def metric_type_entropy(freq: Frequency) -> float:
    return freq.get_type_entropy()


def metric_type_perplexity(freq: Frequency) -> float:
    return 2 ** freq.get_type_entropy()


def metric_hapax_count(freq: Frequency) -> int:
    return freq.get_hepax_count()


def metric_hapax_ratio(freq: Frequency) -> float:
    unique = freq.get_unique_count()
    return freq.get_hepax_count() / unique if unique else 0.0


def metric_freq_hapax_count(freq: Frequency) -> int:
    return freq.get_freq_hepax_count()


def metric_freq_hapax_ratio(freq: Frequency) -> float:
    unique = freq.get_unique_count()
    return freq.get_freq_hepax_count() / unique if unique else 0.0


def metric_avg_length(freq: Frequency) -> float:
    keys = freq.data.keys()
    if not keys:
        return float("nan")
    return sum(len(w) for w in keys) / len(keys)


def metric_avg_length_weighted(freq: Frequency) -> float:
    total = freq.get_total_count()
    if not total:
        return 0.0
    return sum(len(w) * f for w, (f, _) in freq.data.items()) / total


def metric_zipf_slope(freq: Frequency) -> float:
    if not freq.data:
        return float("nan")
    freqs = sorted((f for f, _ in freq.data.values()), reverse=True)
    n = len(freqs)
    if n < 2:
        return float("nan")
    log = math.log
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_xy = 0.0
    for rank, f in enumerate(freqs, 1):
        if f <= 0:
            continue
        x = log(rank)
        y = log(f)
        sum_x += x
        sum_y += y
        sum_xx += x * x
        sum_xy += x * y
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return float("nan")
    return (n * sum_xy - sum_x * sum_y) / denom


METRICS: dict[str, MetricFn] = {
    "count": metric_count,
    "total_frequency": metric_total_frequency,
    "ttr": metric_ttr,
    "zipf_slope": metric_zipf_slope,
    "hapax_count": metric_hapax_count,
    "hapax_ratio": metric_hapax_ratio,
    "freq_hapax_count": metric_freq_hapax_count,
    "freq_hapax_ratio": metric_freq_hapax_ratio,
    "avg_length": metric_avg_length,
    "avg_length_weighted": metric_avg_length_weighted,
    "entropy": metric_entropy,
    "perplexity": metric_perplexity,
    "type_entropy": metric_type_entropy,
    "type_perplexity": metric_type_perplexity,
}


# --- Morph-type composition metrics ---

class MorphStats:
    def __init__(self):
        self.word_count: int = 0
        self.total_freq: float = 0
        self.total_morphs: int = 0
        self.total_roots: int = 0
        self.total_affixes: int = 0
        self.total_prefixes: int = 0
        self.total_suffixes: int = 0
        self.weighted_morphs: float = 0
        self.weighted_roots: float = 0
        self.weighted_affixes: float = 0
        self.weighted_prefixes: float = 0
        self.weighted_suffixes: float = 0
        # Per-word sums for thesis formulas
        self.compounding_sum: float = 0.0
        self.compounding_sum_weighted: float = 0.0
        self.affix_deviation_sum: float = 0.0
        self.affix_deviation_count: int = 0
        self.affix_deviation_sum_weighted: float = 0.0
        self.affix_deviation_freq_weighted: float = 0.0
        # Separate frequency tables for entropy by morph type
        self.root_freq = Frequency()
        self.affix_freq = Frequency()
        self.prefix_freq = Frequency()
        self.suffix_freq = Frequency()

    def add(self, row: Row):
        morphs = row.segmentation
        annots = row.annotation
        freq = row.frequency
        n = len(morphs)

        roots = sum(1 for a in annots if a == "R")
        affixes = sum(1 for a in annots if a == "A")

        # Determine prefixes and suffixes based on position relative to first root
        root_indices = [i for i, a in enumerate(annots) if a == "R"]
        if root_indices:
            first_root = root_indices[0]
            last_root = root_indices[-1]
            prefixes = sum(1 for i, a in enumerate(annots) if a == "A" and i < first_root)
            suffixes = sum(1 for i, a in enumerate(annots) if a == "A" and i > last_root)
        else:
            prefixes = 0
            suffixes = 0

        # Per-word compounding index: roots_w / morphs_w
        if n > 0:
            ci = roots / n
            self.compounding_sum += ci
            self.compounding_sum_weighted += ci * freq

        # Per-word affix deviation: (suffix_w - pref_w) / min(pref_w, suffix_w)
        min_ps = min(prefixes, suffixes)
        if min_ps > 0:
            dev = (suffixes - prefixes) / min_ps
            self.affix_deviation_sum += dev
            self.affix_deviation_count += 1
            self.affix_deviation_sum_weighted += dev * freq
            self.affix_deviation_freq_weighted += freq

        self.word_count += 1
        self.total_freq += freq
        self.total_morphs += n
        self.total_roots += roots
        self.total_affixes += affixes
        self.total_prefixes += prefixes
        self.total_suffixes += suffixes
        self.weighted_morphs += n * freq
        self.weighted_roots += roots * freq
        self.weighted_affixes += affixes * freq
        self.weighted_prefixes += prefixes * freq
        self.weighted_suffixes += suffixes * freq

        # Track morph-type frequencies for entropy by type
        for i, (morph, ann) in enumerate(zip(morphs, annots)):
            if ann == "R":
                self.root_freq.add(morph, freq)
            elif ann == "A":
                self.affix_freq.add(morph, freq)
                if root_indices and i < root_indices[0]:
                    self.prefix_freq.add(morph, freq)
                elif root_indices and i > root_indices[-1]:
                    self.suffix_freq.add(morph, freq)

    def get_metrics(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.word_count:
            out["avg_morphs_per_word"] = self.total_morphs / self.word_count
            out["avg_roots_per_word"] = self.total_roots / self.word_count
            out["avg_affixes_per_word"] = self.total_affixes / self.word_count
            out["avg_prefixes_per_word"] = self.total_prefixes / self.word_count
            out["avg_suffixes_per_word"] = self.total_suffixes / self.word_count
            # Compounding index (thesis formula): per-word average of roots_w / morphs_w
            out["compounding_index"] = self.compounding_sum / self.word_count
            # Affix deviation (thesis formula): per-word average
            if self.affix_deviation_count > 0:
                out["affix_deviation"] = self.affix_deviation_sum / self.affix_deviation_count
            else:
                out["affix_deviation"] = float("nan")
        if self.total_freq:
            out["avg_morphs_per_word_weighted"] = self.weighted_morphs / self.total_freq
            out["avg_roots_per_word_weighted"] = self.weighted_roots / self.total_freq
            out["avg_affixes_per_word_weighted"] = self.weighted_affixes / self.total_freq
            out["avg_prefixes_per_word_weighted"] = self.weighted_prefixes / self.total_freq
            out["avg_suffixes_per_word_weighted"] = self.weighted_suffixes / self.total_freq
            out["compounding_index_weighted"] = self.compounding_sum_weighted / self.total_freq
            if self.affix_deviation_freq_weighted > 0:
                out["affix_deviation_weighted"] = self.affix_deviation_sum_weighted / self.affix_deviation_freq_weighted
            else:
                out["affix_deviation_weighted"] = float("nan")
        # Entropy by morph type
        out["root_entropy"] = self.root_freq.get_entropy()
        out["affix_entropy"] = self.affix_freq.get_entropy()
        out["prefix_entropy"] = self.prefix_freq.get_entropy()
        out["suffix_entropy"] = self.suffix_freq.get_entropy()
        out["root_count"] = self.root_freq.get_unique_count()
        out["affix_count"] = self.affix_freq.get_unique_count()
        out["prefix_count"] = self.prefix_freq.get_unique_count()
        out["suffix_count"] = self.suffix_freq.get_unique_count()
        out["root_type_entropy"] = self.root_freq.get_type_entropy()
        out["affix_type_entropy"] = self.affix_freq.get_type_entropy()
        out["prefix_type_entropy"] = self.prefix_freq.get_type_entropy()
        out["suffix_type_entropy"] = self.suffix_freq.get_type_entropy()
        return out


# --- I/O ---

def read_tsv_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        word_idx = header.index("word")
        freq_idx = header.index("frequency")
        seg_idx = header.index("segmentation")
        ann_idx = header.index("annotation")
        for r in reader:
            if not r:
                continue
            segmentation = r[seg_idx].split("+")
            annotation = r[ann_idx].split("+")
            rows.append(Row(
                word=r[word_idx],
                frequency=float(r[freq_idx]),
                segmentation=segmentation,
                annotation=annotation,
            ))
    return rows


def compute_metrics_for_file(path: Path) -> dict[str, Any]:
    rows = read_tsv_rows(path)

    whole_words = Frequency()
    morphs = Frequency()
    morph_stats = MorphStats()

    for row in rows:
        whole_words.add(row.word, row.frequency)
        for morph in row.segmentation:
            morphs.add(morph, row.frequency)
        morph_stats.add(row)

    code = path.stem
    parts = code.split("-")
    lang_code = parts[0].split("_")[0] if "_" in parts[0] else parts[0][:3]

    out: dict[str, Any] = {
        "file": code,
        "lang": lang_code,
    }

    for name, fn in METRICS.items():
        out["word_" + name] = fn(whole_words)
        out["morph_" + name] = fn(morphs)

    out.update(morph_stats.get_metrics())

    return out


def iter_files(folder: Path, pattern: str) -> list[Path]:
    return sorted(folder.glob(pattern))


def build_table(folder: Path, pattern: str) -> pd.DataFrame:
    files = iter_files(folder, pattern)
    print(f"folder={folder.resolve()}")
    print(f"pattern={pattern}")
    print(f"foundFiles={len(files)}")

    records: list[dict[str, Any]] = []
    pbar = tqdm(files, desc="Processing", unit="file")
    for p in pbar:
        pbar.set_postfix_str(p.stem)
        try:
            records.append(compute_metrics_for_file(p))
        except Exception as e:
            tqdm.write(f"ERROR file={p.name} type={type(e).__name__} msg={e}")

    return pd.DataFrame(records)


def save_outputs(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=SEGMENTED_DIR)
    parser.add_argument("--pattern", default="*.tsv")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "morphs.csv")
    args = parser.parse_args()

    folder = args.folder
    if not folder.exists():
        print(f"ERROR: folder does not exist: {folder}", file=sys.stderr)
        raise SystemExit(2)
    if not folder.is_dir():
        print(f"ERROR: not a directory: {folder}", file=sys.stderr)
        raise SystemExit(2)

    df = build_table(folder, args.pattern)
    if df.empty:
        print("ERROR: no rows in output table (no matching files or all files failed).", file=sys.stderr)
        raise SystemExit(3)

    save_outputs(df, args.out)
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
