from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any
import pandas as pd
import math

COLUMNS = ["term", "lemma", "pos", "separation", "detail"]

@dataclass(frozen=True)
class Row:
    term: str
    lemma: str
    pos: str
    separation: str
    detail: str

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


def read_tsv_rows(path: Path) -> list[Row]:
    df = pd.read_csv(path, sep="\t", header=None, names=COLUMNS, dtype=str, keep_default_na=False)
    return [Row(**rec) for rec in df.to_dict(orient="records")]


def safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")

def metric_avg_morph_count(rows: list[Row], frequency : Frequency) -> float:
    return frequency.total_count / len(rows)

def metric_num_rows(rows: list[Row], _) -> int:
    return len(rows)

def get_morph_frequency(rows: list[Row]) -> Frequency:
    f = Frequency()
    for r in rows:
        morphs = r.separation.split("+")
        morphs = [m.strip() for m in morphs]
        f.add_range(morphs)
    return f

def metric_avg_morph_use(rows: list[Row], frequency : Frequency) -> float:
    return frequency.get_avg()

def metric_total_morph_type_count_per_word(rows: list[Row], frequency : Frequency) -> float:
    return frequency.get_unique_count() / len(rows)

def metric_morph_type_token_ratio(rows: list[Row], frequency : Frequency) -> float:
    return frequency.get_unique_count() / frequency.get_total_count()

def metric_morph_entropy(rows: list[Row], frequency : Frequency) -> float:
    return frequency.get_entropy()

def metric_morph_perplexity(rows: list[Row], frequency : Frequency) -> float:
    return 2 ** frequency.get_entropy()

def metric_hepax_to_words(rows: list[Row], frequency : Frequency) -> float:
    return frequency.get_hepax_count() / len(rows)

def metric_hepax_to_morph_types(rows: list[Row], frequency : Frequency) -> float:
    return frequency.get_hepax_count() / frequency.get_unique_count()

def metric_avg_word_len(rows: list[Row], _) -> float:
    return safe_mean([len(r.term) for r in rows])

METRICS: dict[str, MetricFn] = {
    "nRows": metric_num_rows,
    "avgMorphCount": metric_avg_morph_count,
    "avgWordLen": metric_avg_word_len,
    #"avgMorphUse": metric_avg_morph_use,
    "morphTypeCountPerWord": metric_total_morph_type_count_per_word,
    "morphTTR" : metric_morph_type_token_ratio,
    #"morphEntropy" : metric_morph_entropy,
    "morphPerplexity" : metric_morph_perplexity,
    "hepaxToWords" : metric_hepax_to_words,
    "hepaxToMorphTypes" : metric_hepax_to_morph_types,
}

def compute_metrics_for_file(path: Path, metrics: dict[str, MetricFn]) -> dict[str, Any]:
    rows = read_tsv_rows(path)
    out: dict[str, Any] = {"file": path.name}
    fq = get_morph_frequency(rows)
    for name, fn in metrics.items():
        out[name] = fn(rows, fq)
    return out


def iter_files(folder: Path, pattern: str) -> list[Path]:
    return sorted(folder.glob(pattern))


def build_table(folder: Path, pattern: str, metrics: dict[str, MetricFn]) -> pd.DataFrame:
    files = iter_files(folder, pattern)
    print(f"folder={folder.resolve()}")
    print(f"pattern={pattern}")
    print(f"foundFiles={len(files)}")

    records: list[dict[str, Any]] = []
    for i, p in enumerate(files, 1):
        print(f"[{i}/{len(files)}] processing={p.name}", flush=True)
        try:
            records.append(compute_metrics_for_file(p, metrics))
        except Exception as e:
            print(f"ERROR file={p.name} type={type(e).__name__} msg={e}", file=__import__("sys").stderr)

    cols = ["file"] + list(metrics.keys())
    df = pd.DataFrame(records)
    return df.reindex(columns=cols)


def save_outputs(df: pd.DataFrame, out_prefix: Path) -> None:
    df.to_csv(out_prefix.with_suffix(".tsv"), sep="\t", index=False)


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--pattern", default="*.tsv")
    parser.add_argument("--out", type=Path, default=Path("metrics_out"))
    args = parser.parse_args()

    folder = args.folder
    if not folder.exists():
        print(f"ERROR: folder does not exist: {folder}", file=sys.stderr)
        raise SystemExit(2)
    if not folder.is_dir():
        print(f"ERROR: not a directory: {folder}", file=sys.stderr)
        raise SystemExit(2)

    df = build_table(folder, args.pattern, METRICS)
    if df.empty:
        print("ERROR: no rows in output table (no matching files or all files failed).", file=sys.stderr)
        raise SystemExit(3)

    save_outputs(df, args.out)
    print(f"saved={args.out.with_suffix('.tsv')}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()