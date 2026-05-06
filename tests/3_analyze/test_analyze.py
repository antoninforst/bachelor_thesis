"""Tests for core analysis metrics: TTR, entropy, word type count."""
import math
import pytest
from analyze import Frequency, Row, metric_ttr, metric_frequency_entropy, metric_num_rows, metric_total_frequency, metric_hapax_count, metric_hapax_ratio, compute_metrics_for_file, break_word
from pathlib import Path
import tempfile
import csv


# --- Frequency class tests ---

class TestFrequency:
    def test_empty(self):
        freq = Frequency()
        assert freq.get_unique_count() == 0
        assert freq.get_total_count() == 0
        assert freq.get_entropy() == 0.0

    def test_single_word(self):
        freq = Frequency()
        freq.add("hello", 5)
        assert freq.get_unique_count() == 1
        assert freq.get_total_count() == 5
        # Single word => entropy = 0
        assert freq.get_entropy() == 0.0

    def test_uniform_distribution(self):
        """Uniform distribution of n items gives entropy = log2(n)."""
        freq = Frequency()
        freq.add("a", 10)
        freq.add("b", 10)
        freq.add("c", 10)
        freq.add("d", 10)
        assert freq.get_unique_count() == 4
        assert freq.get_total_count() == 40
        assert math.isclose(freq.get_entropy(), math.log2(4), rel_tol=1e-9)

    def test_skewed_distribution(self):
        """Skewed distribution has lower entropy than uniform."""
        freq_uniform = Frequency()
        freq_uniform.add("a", 25)
        freq_uniform.add("b", 25)
        freq_uniform.add("c", 25)
        freq_uniform.add("d", 25)

        freq_skewed = Frequency()
        freq_skewed.add("a", 97)
        freq_skewed.add("b", 1)
        freq_skewed.add("c", 1)
        freq_skewed.add("d", 1)

        assert freq_skewed.get_entropy() < freq_uniform.get_entropy()

    def test_hapax(self):
        freq = Frequency()
        freq.add("rare", 1)
        freq.add("common", 100)
        freq.add("also_rare", 1)
        assert freq.get_hepax_count() == 3  # all added once via add()
        assert freq.get_freq_hepax_count() == 2  # "rare" and "also_rare" have frequency=1

    def test_from_rows(self):
        rows = [Row("cat", 10), Row("dog", 5), Row("cat", 3)]
        freq = Frequency.from_rows(rows)
        # "cat" appears twice in the list
        assert freq.get_unique_count() == 2
        assert freq.get_total_count() == 18


# --- Metric function tests ---

class TestMetricFunctions:
    def _make_freq(self, items: list[tuple[str, int]]) -> Frequency:
        freq = Frequency()
        for word, count in items:
            freq.add(word, count)
        return freq

    def test_ttr_basic(self):
        # 3 types, 30 tokens
        freq = self._make_freq([("a", 10), ("b", 10), ("c", 10)])
        assert math.isclose(metric_ttr(freq), 3 / 30)

    def test_ttr_single_type(self):
        freq = self._make_freq([("only", 100)])
        assert math.isclose(metric_ttr(freq), 1 / 100)

    def test_ttr_uses_ppm_denominator_when_available(self):
        freq = Frequency()
        freq.add("a", 100, 25.0)
        freq.add("b", 50, 12.5)
        assert math.isclose(metric_ttr(freq), 2 / 37.5)
        assert math.isclose(metric_total_frequency(freq), 150)

    def test_num_rows(self):
        freq = self._make_freq([("x", 5), ("y", 3), ("z", 1)])
        assert metric_num_rows(freq) == 3

    def test_total_frequency(self):
        freq = self._make_freq([("x", 5), ("y", 3), ("z", 1)])
        assert metric_total_frequency(freq) == 9

    def test_entropy_known_value(self):
        # 2 equally likely items => entropy = 1 bit
        freq = self._make_freq([("a", 50), ("b", 50)])
        assert math.isclose(metric_frequency_entropy(freq), 1.0, rel_tol=1e-9)

    def test_hapax_count(self):
        freq = self._make_freq([("a", 1), ("b", 1), ("c", 5)])
        assert metric_hapax_count(freq) == 3  # all added once

    def test_hapax_ratio(self):
        freq = self._make_freq([("a", 1), ("b", 1), ("c", 5)])
        assert math.isclose(metric_hapax_ratio(freq), 1.0)  # 3/3


# --- break_word tests ---

class TestBreakWord:
    def test_plain_word(self):
        word, segs = break_word("hello")
        assert word == "hello"
        assert segs == ["hello"]

    def test_segmented_word(self):
        word, segs = break_word("un @break able")
        assert word == "unbreakable"
        assert segs == ["un", "break", "able"]

    def test_multiple_roots(self):
        word, segs = break_word("@foot @ball")
        assert word == "football"
        assert segs == ["foot", "ball"]


# --- Integration test with file ---

class TestComputeMetricsForFile:
    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a small CSV file matching the expected format."""
        p = tmp_path / "test_lang.csv"
        with open(p, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["word", "frequency", "ppm"])
            writer.writerow(["the", "100", "40"])
            writer.writerow(["a", "80", "32"])
            writer.writerow(["cat", "20", "8"])
            writer.writerow(["dog", "15", "6"])
            writer.writerow(["run", "10", "4"])
        return p

    def test_basic_metrics(self, sample_csv):
        result = compute_metrics_for_file(sample_csv, {}, {})
        # 5 word types
        assert result["word_count"] == 5
        # total frequency = 100+80+20+15+10 = 225
        assert result["word_total_frequency"] == 225
        # TTR uses the PPM denominator when available: 5/(40+32+8+6+4)
        assert math.isclose(result["word_ttr"], 5 / 90)
        # Entropy > 0 (not uniform, not degenerate)
        assert result["word_frequency_entropy"] > 0
        # Perplexity = 2^entropy
        assert math.isclose(
            result["word_frequency_perplexity"],
            2 ** result["word_frequency_entropy"],
        )

    @pytest.fixture
    def segmented_csv(self, tmp_path):
        """CSV with morphologically segmented words."""
        p = tmp_path / "seg_lang.csv"
        with open(p, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["word", "frequency"])
            writer.writerow(["un @break able", "50"])
            writer.writerow(["re @build", "30"])
            writer.writerow(["@cat s", "20"])
        return p

    def test_morph_metrics(self, segmented_csv):
        result = compute_metrics_for_file(segmented_csv, {}, {})
        # 3 word types
        assert result["word_count"] == 3
        # morph types: un, break, able, re, build, cat, s = 7
        assert result["morph_count"] == 7
        # morph TTR = 7 / morph_total_frequency
        # morph tokens: un(50)+break(50)+able(50)+re(30)+build(30)+cat(20)+s(20) = 250
        assert result["morph_total_frequency"] == 250
        assert math.isclose(result["morph_ttr"], 7 / 250)
        # morph entropy should be > 0
        assert result["morph_frequency_entropy"] > 0


# --- Test with real data (optional, skipped if file missing) ---

REAL_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "2_annotated" / "eng_Latn.csv"


@pytest.mark.skipif(not REAL_FILE.exists(), reason="Real data file not available")
class TestRealData:
    def test_english_sanity(self):
        result = compute_metrics_for_file(REAL_FILE, {}, {})
        # English should have many word types
        assert result["word_count"] > 1000
        # PPM-normalized TTR should stay in a compact range for a large corpus
        assert result["word_ttr"] < 0.1
        # Entropy should be positive and reasonable (typically 8-12 bits for natural language)
        assert 5 < result["word_frequency_entropy"] < 20
