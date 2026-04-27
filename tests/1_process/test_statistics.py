"""Tests for statistics.py — compute_file_stats and CLI."""

import csv
from pathlib import Path

import pytest

from statistics import compute_file_stats, _cov_column, main


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_freq_csv(path: Path, rows: list[tuple[str, int]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["word", "frequency"])
        for word, freq in rows:
            w.writerow([word, freq])


@pytest.fixture
def simple_csv(tmp_path: Path) -> Path:
    """10 words, total frequency = 1000, descending."""
    p = tmp_path / "lang.csv"
    _write_freq_csv(p, [
        ("a", 500), ("b", 200), ("c", 100), ("d", 80), ("e", 50),
        ("f", 30), ("g", 20), ("h", 10), ("i", 5), ("j", 5),
    ])
    return p


# ---------------------------------------------------------------------------
# compute_file_stats
# ---------------------------------------------------------------------------

class TestComputeFileStats:

    def test_basic_counts(self, simple_csv: Path):
        row = compute_file_stats(simple_csv)
        assert row["distinct_words"] == 10
        assert row["total_frequency"] == 1000

    def test_coverage_50(self, simple_csv: Path):
        """50% of 1000 = 500; cumsum[0]=500 ≥ 500 → 1 type."""
        row = compute_file_stats(simple_csv, coverage_levels=[50])
        assert row["cov_50_types"] == 1
        assert row["cov_50_tokens"] == 500

    def test_coverage_100(self, simple_csv: Path):
        row = compute_file_stats(simple_csv, coverage_levels=[100])
        assert row["cov_100_types"] == 10
        assert row["cov_100_tokens"] == 1000

    def test_coverage_70(self, simple_csv: Path):
        """70% of 1000 = 700; cumsum: 500, 700 → idx=1, types=2."""
        row = compute_file_stats(simple_csv, coverage_levels=[70])
        assert row["cov_70_types"] == 2
        assert row["cov_70_tokens"] == 700

    def test_freq_cutoff_100(self, simple_csv: Path):
        """freq >= 100: a(500), b(200), c(100) → 3 types, 800 tokens."""
        row = compute_file_stats(simple_csv)
        assert row["freq100_types"] == 3
        assert row["freq100_tokens"] == 800

    def test_freq_cutoff_10(self, simple_csv: Path):
        """freq >= 10: first 8 words → 8 types, 990 tokens."""
        row = compute_file_stats(simple_csv)
        assert row["freq10_types"] == 8
        assert row["freq10_tokens"] == 990

    def test_rank_5pct(self, simple_csv: Path):
        """5% of 10 → max(1, int(0.5))=1 type."""
        row = compute_file_stats(simple_csv)
        assert row["rank5pct_types"] == 1
        assert row["rank5pct_tokens"] == 500

    def test_rank_30pct(self, simple_csv: Path):
        """30% of 10 = 3 types."""
        row = compute_file_stats(simple_csv)
        assert row["rank30pct_types"] == 3
        assert row["rank30pct_tokens"] == 800

    def test_single_word_file(self, tmp_path: Path):
        p = tmp_path / "one.csv"
        _write_freq_csv(p, [("only", 42)])
        row = compute_file_stats(p, coverage_levels=[50, 100])
        assert row["distinct_words"] == 1
        assert row["total_frequency"] == 42
        assert row["freq100_types"] == 0
        assert row["freq10_types"] == 1


# ---------------------------------------------------------------------------
# _cov_column
# ---------------------------------------------------------------------------

class TestCovColumn:
    def test_integer(self):
        assert _cov_column(90.0) == "cov_90"

    def test_half(self):
        assert _cov_column(90.5) == "cov_90_5"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:

    def test_output_csv_created(self, tmp_path: Path, simple_csv: Path):
        out = tmp_path / "stats_out.csv"
        main(["--agg-dir", str(simple_csv.parent), "--output", str(out)])
        assert out.exists()
        with open(out, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["file"] == "lang"
        assert rows[0]["distinct_words"] == "10"

    def test_output_has_coverage_columns(self, tmp_path: Path, simple_csv: Path):
        out = tmp_path / "stats.csv"
        main(["--agg-dir", str(simple_csv.parent), "--output", str(out)])
        with open(out, encoding="utf-8") as fh:
            header = fh.readline().strip().split(",")
        assert "cov_94_types" in header
        assert "cov_94_tokens" in header

    def test_no_files_exits(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(SystemExit):
            main(["--agg-dir", str(empty)])
