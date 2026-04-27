"""
Truncate aggregated word-frequency files using coverage statistics.


"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from tqdm import tqdm


SOURCE_DIR = Path("data/1_aggregated")
TARGET_DIR = Path("data/2_annotated")
STATISTICS_PATH = Path("results/1_process/3_truncate/statistics.csv")
DEFAULT_COVERAGE = 94

csv.field_size_limit(10 * 1024 * 1024)


def _coverage_column(coverage: str | int | float) -> str:
    value = str(coverage).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return f"cov_{value.replace('.', '_')}_types"


def _load_statistics(path: Path, coverage: str | int | float) -> dict[str, dict[str, int]]:
    """Return {lang: {keep_types, total_frequency}} from statistics.csv."""
    column = _coverage_column(coverage)
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"file", "total_frequency", column} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing column(s) in {path}: {', '.join(sorted(missing))}")
        return {
            Path(row["file"].strip()).stem.lower(): {
                "keep_types": int(row[column]),
                "total_frequency": int(row["total_frequency"]),
            }
            for row in reader
            if row.get("file", "").strip()
        }


def _valid_row(row: list[str]) -> tuple[list[str], int] | None:
    if len(row) < 2:
        return None
    try:
        return row[:2], int(row[1])
    except ValueError:
        return None


def truncate_file(source_path: Path, target_path: Path, keep_types: int, rng: random.Random) -> int:
    """Write a truncated two-column copy and return the number of data rows written."""
    with open(source_path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, ["word", "frequency"])[:2]
        if keep_types <= 0:
            return _write_rows(target_path, header, [])

        kept: list[list[str]] = []
        band_start = 0        # index where the current frequency band begins
        cur_freq: int | None = None

        # Read rows up to the cut, tracking where each frequency band starts.
        for row in reader:
            parsed = _valid_row(row)
            if parsed is None:
                continue
            if parsed[1] != cur_freq:
                band_start = len(kept)
                cur_freq = parsed[1]
            kept.append(parsed[0])
            if len(kept) == keep_types:
                break
        else:
            return _write_rows(target_path, header, kept)

        # Collect any extra rows tied at the boundary frequency.
        tied_below: list[list[str]] = []
        for row in reader:
            parsed = _valid_row(row)
            if parsed is None:
                continue
            if parsed[1] != cur_freq:
                break
            tied_below.append(parsed[0])

    if not tied_below:
        return _write_rows(target_path, header, kept)

    above = kept[:band_start]
    band = kept[band_start:] + tied_below
    needed = keep_types - len(above)
    chosen = sorted(rng.sample(range(len(band)), needed))
    rows = above + [band[i] for i in chosen]
    return _write_rows(target_path, header, rows)


def _write_rows(path: Path, header: list[str], rows: list[list[str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return len(rows)


def _parse_tokens(tokens: list[str]) -> tuple[list[str], int | None]:
    langs: list[str] = []
    top_n = None
    for token in tokens:
        if token.startswith("n="):
            top_n = int(token[2:])
        else:
            langs.append(token.lower())
    return langs, top_n


def _select_languages(stats: dict[str, dict[str, int]], langs: list[str], top_n: int | None) -> list[str]:
    selected = set(langs) if langs else set(stats)
    if top_n is not None:
        selected &= {
            lang
            for lang, _ in sorted(
                stats.items(), key=lambda item: (-item[1]["total_frequency"], item[0])
            )[:top_n]
        }
    return sorted(selected)


def run(
    *,
    source_dir: Path = SOURCE_DIR,
    target_dir: Path = TARGET_DIR,
    statistics_path: Path = STATISTICS_PATH,
    coverage: str | int | float = DEFAULT_COVERAGE,
    langs: list[str] | None = None,
    language_limit: int | None = None,
    seed: int | None = None,
) -> int:
    stats = _load_statistics(statistics_path, coverage)
    selected = _select_languages(stats, langs or [], language_limit)
    rng = random.Random(seed)
    processed = 0
    iterator = tqdm(selected, desc="Truncating")

    for lang in iterator:
        source_path = source_dir / f"{lang}.csv"
        if not source_path.exists():
            continue
        truncate_file(source_path, target_dir / source_path.name, stats[lang]["keep_types"], rng)
        processed += 1
        iterator.set_postfix_str(lang)
    return processed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Truncate aggregated CSVs using precomputed coverage statistics.")
    parser.add_argument("tokens", nargs="*", metavar="LANG|n=N")
    parser.add_argument("--src", dest="source_dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--out", dest="target_dir", type=Path, default=TARGET_DIR)
    parser.add_argument("--stats", dest="statistics_path", type=Path, default=STATISTICS_PATH)
    parser.add_argument("--coverage", default=str(DEFAULT_COVERAGE))
    parser.add_argument("--top-n", "-n", dest="top_n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = _parse_args(argv)
    langs, token_top_n = _parse_tokens(args.tokens)
    run(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        statistics_path=args.statistics_path,
        coverage=args.coverage,
        langs=langs,
        language_limit=token_top_n if token_top_n is not None else args.top_n,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()