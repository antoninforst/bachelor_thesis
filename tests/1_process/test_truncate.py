"""
Integration tests for the truncation step.

Tests: truncate.py reads n_types from statistics.csv and cuts aggregated files.
"""

import csv
from pathlib import Path

from truncate import main as truncate_main


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class TestTruncate:
    """truncate.py reads coverage from statistics.csv and cuts correctly."""

    def test_creates_output(self, agg_dir, out_dir, stats_csv):
        truncate_main([
            "--src", str(agg_dir),
            "--out", str(out_dir),
            "--stats", str(stats_csv),
            "--seed", "42",
        ])
        assert (out_dir / "aaa.csv").exists()
        assert (out_dir / "bbb.csv").exists()

    def test_preserves_two_frequency_columns(self, agg_dir, out_dir, stats_csv):
        truncate_main([
            "--src", str(agg_dir),
            "--out", str(out_dir),
            "--stats", str(stats_csv),
            "--seed", "42",
        ])
        rows = _read_csv(out_dir / "aaa.csv")
        assert list(rows[0]) == ["word", "frequency"]

    def test_keeps_exact_count(self, agg_dir, out_dir, stats_csv):
        """Output has exactly as many rows as statistics.csv says."""
        truncate_main([
            "--src", str(agg_dir),
            "--out", str(out_dir),
            "--stats", str(stats_csv),
            "--seed", "42",
        ])
        assert len(_read_csv(out_dir / "aaa.csv")) == 4
        assert len(_read_csv(out_dir / "bbb.csv")) == 5

    def test_cutoff_100_keeps_all(self, agg_dir, out_dir, stats_csv):
        """Cutoff 100 reads the cov_100_types column → keeps everything."""
        truncate_main([
            "--src", str(agg_dir),
            "--out", str(out_dir),
            "--stats", str(stats_csv),
            "--coverage", "100",
            "--seed", "42",
        ])
        original = _read_csv(agg_dir / "bbb.csv")
        truncated = _read_csv(out_dir / "bbb.csv")
        assert len(truncated) == len(original)

    def test_tie_random_selection(self, agg_dir, out_dir, stats_csv):
        """aaa: keep 4 → top 2 always kept, pick 2 of 6 at freq=2."""
        truncate_main([
            "--src", str(agg_dir),
            "--out", str(out_dir),
            "--stats", str(stats_csv),
            "--seed", "42",
        ])
        rows = _read_csv(out_dir / "aaa.csv")
        words = [r["word"] for r in rows]
        # Top 2 (hello, world) are always present
        assert "hello" in words
        assert "world" in words
        # Exactly 2 more from the freq=2 band
        band_words = {"apple", "banana", "cherry", "dog", "eagle", "frog"}
        chosen = [w for w in words if w in band_words]
        assert len(chosen) == 2

    def test_no_tie_fast_path(self, agg_dir, out_dir, stats_csv):
        """bbb: keep 5 → boundary at freq=2, no tie below."""
        truncate_main([
            "--src", str(agg_dir),
            "--out", str(out_dir),
            "--stats", str(stats_csv),
            "--seed", "42",
        ])
        rows = _read_csv(out_dir / "bbb.csv")
        words = [r["word"] for r in rows]
        assert words == ["alpha", "beta", "gamma", "one", "two"]

    def test_reproducible_with_seed(self, agg_dir, stats_csv, tmp_path):
        """Same seed produces identical output."""
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        for d in (out1, out2):
            truncate_main([
                "--src", str(agg_dir),
                "--out", str(d),
                "--stats", str(stats_csv),
                "--seed", "123",
            ])
        assert (out1 / "aaa.csv").read_text() == (out2 / "aaa.csv").read_text()
        assert (out1 / "bbb.csv").read_text() == (out2 / "bbb.csv").read_text()

    def test_different_seeds_can_differ(self, agg_dir, stats_csv, tmp_path):
        """Different seeds may produce different tie-breaking for aaa."""
        results = set()
        for seed in range(20):
            d = tmp_path / f"out_{seed}"
            truncate_main([
                "--src", str(agg_dir),
                "--out", str(d),
                "--stats", str(stats_csv),
                "--seed", str(seed),
            ])
            results.add((d / "aaa.csv").read_text())
        # With 6-choose-2 = 15 combos, 20 seeds should hit at least 2
        assert len(results) > 1

    def test_filter_by_lang(self, agg_dir, out_dir, stats_csv):
        """Passing language codes limits which files are processed."""
        truncate_main([
            "aaa",
            "--src", str(agg_dir),
            "--out", str(out_dir),
            "--stats", str(stats_csv),
            "--seed", "42",
        ])
        assert (out_dir / "aaa.csv").exists()
        assert not (out_dir / "bbb.csv").exists()

    def test_top_n_languages_by_token_count(self, agg_dir, out_dir, stats_csv):
        """n=N selects languages with the largest total_frequency values."""
        truncate_main([
            "n=1",
            "--src", str(agg_dir),
            "--out", str(out_dir),
            "--stats", str(stats_csv),
            "--seed", "42",
        ])
        assert not (out_dir / "aaa.csv").exists()
        assert (out_dir / "bbb.csv").exists()
