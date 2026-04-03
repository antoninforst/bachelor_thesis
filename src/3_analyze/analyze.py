from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any
import pandas as pd
import math

ANNOTATED_DIR = Path("data/2_annotated")
RESULTS_DIR = Path("data/3_results")
STATISTICS_PATH = RESULTS_DIR / "statistics.csv"
SHORTCUTS_PATH = Path(__file__).resolve().parent.parent / "0_data_processing" / "leipzig" / "lepzig_shortcuts.csv"

COLUMNS = ["word", "frequency"]

@dataclass(frozen=True)
class Row:
    word: str
    frequency: int

class Frequency:
    def __init__(self):
        self.data = {}
        self.total_count = 0
        self.hepax_count = 0

    def add(self, key):
        if key in self.data:
            d = self.data[key]
            if d == 1:
                self.hepax_count -= 1
            self.data[key] = d + 1
        else:
            self.data[key] = 1
            self.hepax_count += 1
        self.total_count += 1
    
    def add_range(self, keys):
        for key in keys:
            self.add(key)

    def get(self, key):
        if key in self.data:
            return self.data[key]
        return 0
    
    def get_probability(self, key):
        if key in self.data:
            return self.data[key] / self.total_count
        return 0
    
    def get_avg(self):
        if not self.data:
            return 0
        return self.total_count / len(self.data)
    
    def get_unique_count(self):
        return len(self.data)
    
    def get_total_count(self):
        return self.total_count
    
    def get_entropy(self):
        s = 0
        for key in self.data:
            p = self.get_probability(key)
            s -= p * math.log2(p)
        return s

    def get_hepax_count(self):
        return self.hepax_count

    def print(self, n: int):
        sorted_items = sorted(self.data.items(), key=lambda x: x[1], reverse=True)
        for key, value in sorted_items[:n]:
            print(f"{key}: {value}")



MetricFn = Callable[[list[Row], Frequency], Any]


def load_shortcuts() -> dict[str, str]:
    mapping: dict[str, str] = {}
    if SHORTCUTS_PATH.exists():
        with open(SHORTCUTS_PATH, encoding="utf-8") as f:
            for row in csv.DictReader(f):
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
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = []
    for _, r in df.iterrows():
        rows.append(Row(
            word=r["word"],
            frequency=int(r["frequency"]),
        ))
    return rows


def safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


# ── Frequency metrics ──────────────────────────────────────

def metric_num_rows(rows: list[Row], _) -> int:
    return len(rows)

def metric_total_frequency(rows: list[Row], _) -> int:
    return sum(r.frequency for r in rows)

def metric_hapax_count(rows: list[Row], _) -> int:
    return sum(1 for r in rows if r.frequency == 1)

def metric_hapax_ratio(rows: list[Row], _) -> float:
    hapax = sum(1 for r in rows if r.frequency == 1)
    return hapax / len(rows) if rows else 0.0

def metric_avg_word_len(rows: list[Row], _) -> float:
    return safe_mean([len(r.word) for r in rows])

def metric_avg_word_len_weighted(rows: list[Row], _) -> float:
    total_freq = sum(r.frequency for r in rows)
    if total_freq == 0:
        return float("nan")
    return sum(len(r.word) * r.frequency for r in rows) / total_freq

def metric_frequency_entropy(rows: list[Row], _) -> float:
    total = sum(r.frequency for r in rows)
    if total == 0:
        return 0.0
    s = 0.0
    for r in rows:
        p = r.frequency / total
        if p > 0:
            s -= p * math.log2(p)
    return s

def metric_frequency_perplexity(rows: list[Row], _) -> float:
    return 2 ** metric_frequency_entropy(rows, _)

def metric_ttr(rows: list[Row], _) -> float:
    total = sum(r.frequency for r in rows)
    return len(rows) / total if total else 0.0


# ── Metric registry ───────────────────────────────────────

METRICS: dict[str, MetricFn] = {
    "word_count": metric_num_rows,
    "total_frequency": metric_total_frequency,
    "ttr": metric_ttr,
    "hapax_count": metric_hapax_count,
    "hapax_ratio": metric_hapax_ratio,
    "avg_word_length": metric_avg_word_len,
    "avg_word_length_weighted": metric_avg_word_len_weighted,
    "frequency_entropy": metric_frequency_entropy,
    "frequency_perplexity": metric_frequency_perplexity,
}


def compute_metrics_for_file(path: Path, lang_names: dict[str, str], statistics: dict[str, dict[str, int]]) -> dict[str, Any]:
    rows = read_csv_rows(path)
    code = path.stem
    out: dict[str, Any] = {"language": lang_names.get(code, code)}

    # join statistics
    if code in statistics:
        out.update(statistics[code])

    for name, fn in METRICS.items():
        out[name] = fn(rows, None)

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
    for i, p in enumerate(files, 1):
        print(f"[{i}/{len(files)}] processing={p.name}", flush=True)
        try:
            records.append(compute_metrics_for_file(p, lang_names, statistics))
        except Exception as e:
            print(f"ERROR file={p.name} type={type(e).__name__} msg={e}", file=__import__("sys").stderr)

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