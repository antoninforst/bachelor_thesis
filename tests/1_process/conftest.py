"""
Shared fixtures for 1_process integration tests.

Provides a temporary directory populated with small raw CSV files
for two fake languages: "aaa" and "bbb".
"""

import csv
import pytest
from pathlib import Path


def _write_csv(path: Path, header: list[str], rows: list[list]):
    """Write a simple CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """
    Create a tiny data/0_raw/ with 5 files for 2 languages.

    aaa: 3 files
      - aaa_source1.csv  (Latin, 10 words, some overlap with source2)
      - aaa_source2.csv  (Latin, 8 words, heavy overlap with source1)
      - aaa_source3.csv  (Cyrillic, 6 words — wrong script)

    bbb: 2 files
      - bbb_source1.csv  (Latin, 6 words)
      - bbb_source2.csv  (Latin, 5 words, independent)
    """
    d = tmp_path / "raw"
    d.mkdir()

    # --- aaa: Latin files with heavy overlap ---
    _write_csv(d / "aaa_source1.csv", ["word", "frequency"], [
        ["hello", 100],
        ["world", 80],
        ["apple", 1],   # hapax
        ["banana", 1],  # hapax
        ["cherry", 1],  # hapax
        ["dog", 1],     # hapax
        ["eagle", 1],   # hapax
        ["frog", 1],    # hapax
        ["grape", 1],   # hapax
        ["house", 1],   # hapax
    ])

    # source2 is a subset — most hapaxes overlap with source1
    _write_csv(d / "aaa_source2.csv", ["word", "frequency"], [
        ["hello", 50],
        ["world", 40],
        ["apple", 1],   # hapax, overlaps
        ["banana", 1],  # hapax, overlaps
        ["cherry", 1],  # hapax, overlaps
        ["dog", 1],     # hapax, overlaps
        ["eagle", 1],   # hapax, overlaps
        ["frog", 1],    # hapax, overlaps
    ])

    # source3 is Cyrillic — should be flagged by script check
    _write_csv(d / "aaa_source3.csv", ["word", "frequency"], [
        ["\u043f\u0440\u0438\u0432\u0435\u0442", 60],   # привет
        ["\u043c\u0438\u0440", 40],                       # мир
        ["\u0434\u043e\u043c", 20],                       # дом
        ["\u043a\u043e\u0442", 1],                         # кот
        ["\u043b\u0435\u0441", 1],                         # лес
        ["\u0441\u043e\u043d", 1],                         # сон
    ])

    # --- bbb: two independent Latin files ---
    _write_csv(d / "bbb_source1.csv", ["word", "frequency"], [
        ["alpha", 200],
        ["beta", 150],
        ["gamma", 50],
        ["delta", 1],    # hapax
        ["epsilon", 1],  # hapax
        ["zeta", 1],     # hapax
    ])

    _write_csv(d / "bbb_source2.csv", ["word", "frequency"], [
        ["one", 100],
        ["two", 80],
        ["three", 30],
        ["four", 1],     # hapax
        ["five", 1],     # hapax
    ])

    return d


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    """Empty output directory for results."""
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.fixture
def agg_dir(tmp_path: Path) -> Path:
    """
    Pre-built aggregated directory with two small language files.
    Simulates output of aggregate.py.
    """
    d = tmp_path / "aggregated"
    d.mkdir()

    # aaa.csv: 8 words, total freq = 188
    _write_csv(d / "aaa.csv", ["word", "frequency"], [
        ["hello", 150],
        ["world", 120],
        ["apple", 2],
        ["banana", 2],
        ["cherry", 2],
        ["dog", 2],
        ["eagle", 2],
        ["frog", 2],
    ])

    # bbb.csv: 8 words, total freq = 414
    _write_csv(d / "bbb.csv", ["word", "frequency"], [
        ["alpha", 200],
        ["beta", 150],
        ["gamma", 50],
        ["one", 10],
        ["two", 2],
        ["three", 1],
        ["four", 1],
        ["five", 1],
    ])

    return d


@pytest.fixture
def stats_csv(tmp_path: Path, agg_dir: Path) -> Path:
    """
    Fake statistics.csv with a cov_94_types column.

    aaa: keep 4 of 8 → boundary freq=2 with a tie (6 at freq=2, pick 2)
    bbb: keep 5 of 8 → boundary freq=2, no tie
    """
    p = tmp_path / "statistics.csv"
    _write_csv(
        p,
        ["file", "distinct_words", "total_frequency", "cov_94_types", "cov_100_types"],
        [
            ["aaa", 8, 282, 4, 8],
            ["bbb", 8, 414, 5, 8],
        ],
    )
    return p
