from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any
import pandas as pd
import math
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

ANNOTATED_DIR = _PROJECT_ROOT / "data" / "2_annotated"
RESULTS_DIR = _PROJECT_ROOT / "results" / "3_analyze"
STATISTICS_PATH = _PROJECT_ROOT / "results" / "0_data_processing" / "statistics.csv"
SHORTCUTS_PATH = _PROJECT_ROOT / "src" / "0_data_processing" / "corpora" / "leipzig" / "lepzig_shortcuts.csv"

COLUMNS = ["word", "frequency", "ppm"]

@dataclass(frozen=True)
class Row:
    word: str
    frequency: float
    ppm: float | None = None

class Frequency:
    def __init__(self):
        self.data = {}
        self.total_count = 0
        self.ppm_total_count = 0
        self.record_count = 0
        self.fhepax_count = 0
        self.hepax_count = 0

    def add(self, key, count=1, ppm: float | None = None):
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
        self.ppm_total_count += count if ppm is None else ppm
        self.record_count += 1
    
    def add_range(self, keys):
        for key in keys:
            self.add(key)

    @classmethod
    def from_rows(cls, rows: list[Row]) -> Frequency:
        freq = cls()
        for r in rows:
            freq.add(r.word, r.frequency)
        return freq

    def get(self, key):
        if key in self.data:
            return self.data[key]
        return 0
    
    def get_probability(self, key):
        if key in self.data:
            return self.data[key][0] / self.total_count
        return 0
    
    def get_avg(self):
        if not self.data:
            return 0
        return self.total_count / len(self.data)
    
    def get_unique_count(self):
        return len(self.data)
    
    def get_total_count(self):
        return self.total_count

    def get_ppm_total_count(self):
        return self.ppm_total_count
    
    def get_entropy(self):
        total = self.total_count
        if total == 0:
            return 0.0
        s = 0.0
        log2 = math.log2
        for f, _ in self.data.values():
            p = f / total
            s -= p * log2(p)
        return s

    def get_hepax_count(self):
        return self.hepax_count
    
    def get_freq_hepax_count(self):
        return self.fhepax_count

    def print(self, n: int):
        sorted_items = sorted(self.data.items(), key=lambda x: x[1], reverse=True)
        for key, value in sorted_items[:n]:
            print(f"{key}: {value}")



MetricFn = Callable[[Frequency], Any]


def load_shortcuts() -> dict[str, str]:
    mapping: dict[str, str] = {}
    if SHORTCUTS_PATH.exists():
        with open(SHORTCUTS_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping[row["code"]] = row["language"]
    return mapping


def load_statistics() -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    if STATISTICS_PATH.exists():
        df = pd.read_csv(STATISTICS_PATH)
        for _, row in df.iterrows():
            stats[row["file"]] = {
                "original_distinct_words": int(row["distinct_words"]),
                "original_total_frequency": int(row["total_frequency"]),
            }
    return stats


def read_csv_rows(path: Path) -> list[Row]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        word_idx = header.index("word")
        freq_idx = header.index("frequency")
        ppm_idx = header.index("ppm") if "ppm" in header else None
        for r in reader:
            if r:
                rows.append(Row(
                    word=r[word_idx],
                    frequency=float(r[freq_idx]),
                    ppm=float(r[ppm_idx]) if ppm_idx is not None and r[ppm_idx] else None,
                ))
    return rows


def safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


# Frequency metrics

def metric_num_rows(freq: Frequency) -> int:
    return freq.get_unique_count()

def metric_total_frequency(freq: Frequency) -> int:
    return freq.get_total_count()

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

def metric_avg_word_len(freq: Frequency) -> float:
    keys = freq.data.keys()
    if not keys:
        return float("nan")
    return sum(len(w) for w in keys) / len(keys)

def metric_avg_word_len_weighted(freq: Frequency) -> float:
    return sum(len(w) * f for w, (f, _) in freq.data.items()) / freq.get_total_count() if freq.get_total_count() else 0.0

def metric_frequency_entropy(freq: Frequency) -> float:
    return freq.get_entropy()

def metric_frequency_perplexity(freq: Frequency) -> float:
    return 2 ** freq.get_entropy()

def metric_ttr(freq: Frequency) -> float:
    total = freq.get_ppm_total_count()
    return freq.get_unique_count() / total if total else 0.0

def metric_zipf_slope(freq: Frequency) -> float:
    """Slope of linear regression on log(rank) vs log(frequency)."""
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

class MorphStats:
    def __init__(self):
        self.word_count = 0
        self.total_freq = 0
        self.total_roots = 0
        self.total_prefixes = 0
        self.total_suffixes = 0
        self.total_interfixes = 0
        self.weighted_roots = 0
        self.weighted_prefixes = 0
        self.weighted_suffixes = 0
        self.weighted_interfixes = 0

    def add(self, row: Row):
        morphemes = row.word.split()
        freq = row.frequency

        root_indices = [i for i, m in enumerate(morphemes) if m.startswith("@")]

        if not root_indices:
            roots = 1
            prefixes = 0
            suffixes = 0
            interfixes = 0
        else:
            roots = len(root_indices)
            first_root = root_indices[0]
            last_root = root_indices[-1]
            prefixes = first_root
            suffixes = len(morphemes) - last_root - 1
            interfixes = sum(
                1 for i in range(first_root + 1, last_root)
                if not morphemes[i].startswith("@")
            )

        self.word_count += 1
        self.total_freq += freq
        self.total_roots += roots
        self.total_prefixes += prefixes
        self.total_suffixes += suffixes
        self.total_interfixes += interfixes
        self.weighted_roots += roots * freq
        self.weighted_prefixes += prefixes * freq
        self.weighted_suffixes += suffixes * freq
        self.weighted_interfixes += interfixes * freq

    def get_metrics(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.word_count:
            out["avg_root_count"] = self.total_roots / self.word_count
            out["avg_prefix_count"] = self.total_prefixes / self.word_count
            out["avg_suffix_count"] = self.total_suffixes / self.word_count
            out["avg_interfix_count"] = self.total_interfixes / self.word_count
        if self.total_freq:
            out["avg_root_count_weighted"] = self.weighted_roots / self.total_freq
            out["avg_prefix_count_weighted"] = self.weighted_prefixes / self.total_freq
            out["avg_suffix_count_weighted"] = self.weighted_suffixes / self.total_freq
            out["avg_interfix_count_weighted"] = self.weighted_interfixes / self.total_freq
        return out


METRICS: dict[str, MetricFn] = {
    "count": metric_num_rows,
    "total_frequency": metric_total_frequency,
    "ttr": metric_ttr,
    "zipf_slope": metric_zipf_slope,
    "hapax_count": metric_hapax_count,
    "hapax_ratio": metric_hapax_ratio,
    "freq_hapax_count": metric_freq_hapax_count,
    "freq_hapax_ratio": metric_freq_hapax_ratio,
    "avg_length": metric_avg_word_len,
    "avg_length_weighted": metric_avg_word_len_weighted,
    "frequency_entropy": metric_frequency_entropy,
    "frequency_perplexity": metric_frequency_perplexity,
}

def break_word(word: str) -> tuple[str, list[str]]:
    segmentation = word.split()
    segmentation = [s[1:] if len(s) > 1 and s.startswith("@") else s for s in segmentation]
    return "".join(segmentation), segmentation

def compute_metrics_for_file(path: Path, lang_names: dict[str, str], statistics: dict[str, dict[str, int]]) -> dict[str, Any]:
    rows = read_csv_rows(path)
    whole_words = Frequency()
    morphs = Frequency()
    morph_stats = MorphStats()
    for row in rows:
        word, segmentation = break_word(row.word)
        whole_words.add(word, row.frequency, row.ppm)
        for morph in segmentation:
            morphs.add(morph, row.frequency, row.ppm)
        morph_stats.add(row)
    code = path.stem
    lang_code = code.split("_")[0]
    script_code = code[-4:] if len(code) >= 4 else code
    out: dict[str, Any] = {
        "language": lang_names.get(lang_code, code),
        "lang": lang_code[:3],
        "script": script_code,
    }

    # join statistics
    if code in statistics:
        out.update(statistics[code])

    for name, fn in METRICS.items():
        out["word_"+name] = fn(whole_words)
        out["morph_"+name] = fn(morphs)

    out.update(morph_stats.get_metrics())

    # percentage of original distinct words retained after cutoff
    if "original_distinct_words" in out and out["original_distinct_words"]:
        out["cutoff_word_pct"] = out["word_count"] / out["original_distinct_words"] * 100

    return out


def iter_files(folder: Path, pattern: str) -> list[Path]:
    return sorted(folder.glob(pattern))


def build_table(folder: Path, pattern: str, lang_names: dict[str, str], statistics: dict[str, dict[str, int]]) -> pd.DataFrame:
    files = iter_files(folder, pattern)
    print(f"folder={folder.resolve()}")
    print(f"pattern={pattern}")
    print(f"foundFiles={len(files)}")

    records: list[dict[str, Any]] = []
    pbar = tqdm(files, desc="Processing", unit="file")
    for p in pbar:
        pbar.set_postfix_str(p.stem)
        try:
            records.append(compute_metrics_for_file(p, lang_names, statistics))
        except Exception as e:
            tqdm.write(f"ERROR file={p.name} type={type(e).__name__} msg={e}")

    return pd.DataFrame(records)


def save_outputs(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=ANNOTATED_DIR)
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "results.csv")
    args = parser.parse_args()

    folder = args.folder
    if not folder.exists():
        print(f"ERROR: folder does not exist: {folder}", file=sys.stderr)
        raise SystemExit(2)
    if not folder.is_dir():
        print(f"ERROR: not a directory: {folder}", file=sys.stderr)
        raise SystemExit(2)

    lang_names = load_shortcuts()
    statistics = load_statistics()

    df = build_table(folder, args.pattern, lang_names, statistics)
    if df.empty:
        print("ERROR: no rows in output table (no matching files or all files failed).", file=sys.stderr)
        raise SystemExit(3)

    save_outputs(df, args.out)
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()