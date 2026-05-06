"""Tests for morph-level analysis metrics (morphs.py)."""
import math
import csv
import pytest
from pathlib import Path

from morphs import (
    Frequency,
    Row,
    MorphStats,
    metric_count,
    metric_total_frequency,
    metric_ttr,
    metric_entropy,
    metric_perplexity,
    metric_hapax_count,
    metric_hapax_ratio,
    metric_freq_hapax_count,
    metric_freq_hapax_ratio,
    metric_avg_length,
    metric_avg_length_weighted,
    metric_zipf_slope,
    read_tsv_rows,
    compute_metrics_for_file,
)


# ---- Frequency class ----

class TestFrequency:
    def test_empty(self):
        freq = Frequency()
        assert freq.get_unique_count() == 0
        assert freq.get_total_count() == 0
        assert freq.get_entropy() == 0.0

    def test_single_key(self):
        freq = Frequency()
        freq.add("a", 10)
        assert freq.get_unique_count() == 1
        assert freq.get_total_count() == 10
        assert freq.get_entropy() == 0.0

    def test_uniform_entropy(self):
        freq = Frequency()
        freq.add("a", 25)
        freq.add("b", 25)
        freq.add("c", 25)
        freq.add("d", 25)
        assert freq.get_unique_count() == 4
        assert freq.get_total_count() == 100
        assert math.isclose(freq.get_entropy(), math.log2(4), rel_tol=1e-9)

    def test_skewed_has_lower_entropy(self):
        uniform = Frequency()
        for k in "abcd":
            uniform.add(k, 25)

        skewed = Frequency()
        skewed.add("a", 97)
        skewed.add("b", 1)
        skewed.add("c", 1)
        skewed.add("d", 1)

        assert skewed.get_entropy() < uniform.get_entropy()

    def test_duplicate_key_merges(self):
        freq = Frequency()
        freq.add("x", 3)
        freq.add("x", 7)
        assert freq.get_unique_count() == 1
        assert freq.get_total_count() == 10

    def test_hapax_count(self):
        freq = Frequency()
        freq.add("once", 5)
        freq.add("twice", 3)
        freq.add("twice", 2)
        # "once" added 1 time, "twice" added 2 times => hapax = 1
        assert freq.get_hepax_count() == 1

    def test_freq_hapax(self):
        freq = Frequency()
        freq.add("rare", 1)
        freq.add("common", 100)
        # Both added once (hapax=2), but only "rare" has frequency==1
        assert freq.get_hepax_count() == 2
        assert freq.get_freq_hepax_count() == 1


# ---- Metric functions ----

class TestMetricFunctions:
    def _freq(self, items: list[tuple[str, float]]) -> Frequency:
        f = Frequency()
        for k, c in items:
            f.add(k, c)
        return f

    def test_count(self):
        assert metric_count(self._freq([("a", 1), ("b", 2)])) == 2

    def test_total_frequency(self):
        assert metric_total_frequency(self._freq([("a", 3), ("b", 7)])) == 10

    def test_ttr(self):
        freq = self._freq([("a", 10), ("b", 10), ("c", 10)])
        assert math.isclose(metric_ttr(freq), 3 / 30)

    def test_ttr_empty(self):
        assert metric_ttr(Frequency()) == 0.0

    def test_entropy_two_equal(self):
        freq = self._freq([("a", 50), ("b", 50)])
        assert math.isclose(metric_entropy(freq), 1.0, rel_tol=1e-9)

    def test_perplexity(self):
        freq = self._freq([("a", 50), ("b", 50)])
        assert math.isclose(metric_perplexity(freq), 2.0, rel_tol=1e-9)

    def test_hapax_count(self):
        freq = self._freq([("a", 1), ("b", 1), ("c", 5)])
        assert metric_hapax_count(freq) == 3

    def test_hapax_ratio(self):
        freq = self._freq([("a", 1), ("b", 1), ("c", 5)])
        assert math.isclose(metric_hapax_ratio(freq), 1.0)

    def test_freq_hapax_count(self):
        freq = self._freq([("a", 1), ("b", 1), ("c", 5)])
        assert metric_freq_hapax_count(freq) == 2

    def test_freq_hapax_ratio(self):
        freq = self._freq([("a", 1), ("b", 1), ("c", 5)])
        assert math.isclose(metric_freq_hapax_ratio(freq), 2 / 3)

    def test_avg_length(self):
        freq = self._freq([("ab", 1), ("cdef", 1)])
        assert math.isclose(metric_avg_length(freq), 3.0)

    def test_avg_length_weighted(self):
        # "ab"(len=2, freq=80), "cdef"(len=4, freq=20) => (2*80+4*20)/100 = 2.4
        freq = self._freq([("ab", 80), ("cdef", 20)])
        assert math.isclose(metric_avg_length_weighted(freq), 2.4)

    def test_zipf_slope_negative(self):
        freq = self._freq([("a", 1000), ("b", 100), ("c", 10)])
        slope = metric_zipf_slope(freq)
        assert slope < 0

    def test_zipf_slope_single_item(self):
        freq = self._freq([("a", 5)])
        assert math.isnan(metric_zipf_slope(freq))


# ---- MorphStats ----

class TestMorphStats:
    def _row(self, word, freq, seg, ann):
        return Row(word=word, frequency=freq, segmentation=seg, annotation=ann)

    def test_single_root(self):
        ms = MorphStats()
        ms.add(self._row("cat", 10, ["cat"], ["R"]))
        m = ms.get_metrics()
        assert m["avg_morphs_per_word"] == 1.0
        assert m["avg_roots_per_word"] == 1.0
        assert m["avg_affixes_per_word"] == 0.0
        assert math.isclose(m["compounding_index"], 1.0)

    def test_prefix_suffix(self):
        """un+break+able => A R A: 1 prefix, 1 suffix."""
        ms = MorphStats()
        ms.add(self._row("unbreakable", 10, ["un", "break", "able"], ["A", "R", "A"]))
        m = ms.get_metrics()
        assert m["avg_morphs_per_word"] == 3.0
        assert m["avg_roots_per_word"] == 1.0
        assert m["avg_affixes_per_word"] == 2.0
        assert m["avg_prefixes_per_word"] == 1.0
        assert m["avg_suffixes_per_word"] == 1.0
        # compounding: 1 root / 3 morphs
        assert math.isclose(m["compounding_index"], 1 / 3)
        # affix deviation: (1-1)/min(1,1) = 0
        assert math.isclose(m["affix_deviation"], 0.0)

    def test_compound_word(self):
        """foot+ball => R R: 2 roots, 0 affixes."""
        ms = MorphStats()
        ms.add(self._row("football", 50, ["foot", "ball"], ["R", "R"]))
        m = ms.get_metrics()
        assert m["avg_roots_per_word"] == 2.0
        assert m["avg_affixes_per_word"] == 0.0
        assert math.isclose(m["compounding_index"], 1.0)  # 2/2
        assert math.isnan(m["affix_deviation"])  # no affixes

    def test_compounding_index_per_word_average(self):
        """Verify compounding index is per-word average of roots/morphs, not global ratio."""
        ms = MorphStats()
        # word1: 1 root, 3 morphs => ci = 1/3
        ms.add(self._row("w1", 1, ["a", "b", "c"], ["A", "R", "A"]))
        # word2: 2 roots, 2 morphs => ci = 1.0
        ms.add(self._row("w2", 1, ["d", "e"], ["R", "R"]))
        m = ms.get_metrics()
        expected = (1 / 3 + 1.0) / 2
        assert math.isclose(m["compounding_index"], expected, rel_tol=1e-9)

    def test_affix_deviation_formula(self):
        """Test affix deviation: (pref - suf) / min(pref, suf) averaged over qualifying words."""
        ms = MorphStats()
        # word1: A A R A => 2 prefixes, 1 suffix => dev = (2-1)/1 = 1
        ms.add(self._row("w1", 1, ["a", "b", "c", "d"], ["A", "A", "R", "A"]))
        # word2: A R A A => 1 prefix, 2 suffixes => dev = (1-2)/1 = -1
        ms.add(self._row("w2", 1, ["e", "f", "g", "h"], ["A", "R", "A", "A"]))
        m = ms.get_metrics()
        # Average: (1 + (-1)) / 2 = 0
        assert math.isclose(m["affix_deviation"], 0.0, abs_tol=1e-9)

    def test_affix_deviation_skipped_when_no_both(self):
        """Words with only prefixes or only suffixes are excluded from affix deviation."""
        ms = MorphStats()
        # A R => 1 prefix, 0 suffixes => min=0 => skip
        ms.add(self._row("w1", 1, ["a", "b"], ["A", "R"]))
        m = ms.get_metrics()
        assert math.isnan(m["affix_deviation"])

    def test_weighted_metrics(self):
        ms = MorphStats()
        ms.add(self._row("w1", 100, ["a", "b", "c"], ["A", "R", "A"]))
        ms.add(self._row("w2", 1, ["d", "e"], ["R", "R"]))
        m = ms.get_metrics()
        # weighted morphs: (3*100 + 2*1) / 101
        assert math.isclose(m["avg_morphs_per_word_weighted"], 302 / 101, rel_tol=1e-9)

    def test_root_entropy(self):
        ms = MorphStats()
        ms.add(self._row("w1", 50, ["root1"], ["R"]))
        ms.add(self._row("w2", 50, ["root2"], ["R"]))
        m = ms.get_metrics()
        # 2 equally frequent roots => entropy = 1 bit
        assert math.isclose(m["root_entropy"], 1.0, rel_tol=1e-9)

    def test_affix_entropy(self):
        ms = MorphStats()
        ms.add(self._row("w1", 50, ["un", "root", "ly"], ["A", "R", "A"]))
        ms.add(self._row("w2", 50, ["re", "root", "ed"], ["A", "R", "A"]))
        m = ms.get_metrics()
        # 4 affix types (un, ly, re, ed), each with freq 50 => entropy = log2(4)
        assert math.isclose(m["affix_entropy"], math.log2(4), rel_tol=1e-9)
        assert m["affix_count"] == 4
        assert m["prefix_count"] == 2
        assert m["suffix_count"] == 2

    def test_prefix_suffix_entropy_separate(self):
        ms = MorphStats()
        # 2 prefixes each freq 50 => prefix entropy = 1
        ms.add(self._row("w1", 50, ["pre1", "root", "suf"], ["A", "R", "A"]))
        ms.add(self._row("w2", 50, ["pre2", "root", "suf"], ["A", "R", "A"]))
        m = ms.get_metrics()
        assert math.isclose(m["prefix_entropy"], 1.0, rel_tol=1e-9)
        # 1 suffix type (suf, freq 100) => suffix entropy = 0
        assert math.isclose(m["suffix_entropy"], 0.0, abs_tol=1e-9)


# ---- read_tsv_rows ----

class TestReadTsvRows:
    def test_basic(self, tmp_path):
        p = tmp_path / "test.tsv"
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["word", "frequency", "segmentation", "annotation"])
            w.writerow(["running", "50", "run+ning", "R+A"])
            w.writerow(["cat", "100", "cat", "R"])
        rows = read_tsv_rows(p)
        assert len(rows) == 2
        assert rows[0].word == "running"
        assert rows[0].frequency == 50.0
        assert rows[0].segmentation == ["run", "ning"]
        assert rows[0].annotation == ["R", "A"]
        assert rows[1].segmentation == ["cat"]
        assert rows[1].annotation == ["R"]

    def test_empty_rows_skipped(self, tmp_path):
        p = tmp_path / "test.tsv"
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["word", "frequency", "segmentation", "annotation"])
            w.writerow(["cat", "10", "cat", "R"])
            w.writerow([])
        rows = read_tsv_rows(p)
        assert len(rows) == 1


# ---- compute_metrics_for_file integration ----

class TestComputeMetricsForFile:
    @pytest.fixture
    def sample_tsv(self, tmp_path):
        p = tmp_path / "eng_MorphTest-eng_Latn.tsv"
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["word", "frequency", "segmentation", "annotation"])
            w.writerow(["unbreakable", "50", "un+break+able", "A+R+A"])
            w.writerow(["rebuild", "30", "re+build", "A+R"])
            w.writerow(["cats", "20", "cat+s", "R+A"])
            w.writerow(["football", "10", "foot+ball", "R+R"])
        return p

    def test_output_keys(self, sample_tsv):
        result = compute_metrics_for_file(sample_tsv)
        assert result["file"] == "eng_MorphTest-eng_Latn"
        assert result["lang"] == "eng"
        assert "word_count" in result
        assert "morph_count" in result
        assert "compounding_index" in result
        assert "affix_deviation" in result
        assert "root_entropy" in result
        assert "affix_entropy" in result
        assert "prefix_entropy" in result
        assert "suffix_entropy" in result

    def test_word_count(self, sample_tsv):
        result = compute_metrics_for_file(sample_tsv)
        assert result["word_count"] == 4

    def test_morph_count(self, sample_tsv):
        result = compute_metrics_for_file(sample_tsv)
        # morphs: un, break, able, re, build, cat, s, foot, ball = 9 unique
        assert result["morph_count"] == 9

    def test_total_frequencies(self, sample_tsv):
        result = compute_metrics_for_file(sample_tsv)
        assert result["word_total_frequency"] == 110
        # morph tokens: un(50)+break(50)+able(50)+re(30)+build(30)+cat(20)+s(20)+foot(10)+ball(10) = 270
        assert result["morph_total_frequency"] == 270

    def test_morph_ttr(self, sample_tsv):
        result = compute_metrics_for_file(sample_tsv)
        assert math.isclose(result["morph_ttr"], 9 / 270)

    def test_entropy_positive(self, sample_tsv):
        result = compute_metrics_for_file(sample_tsv)
        assert result["word_entropy"] > 0
        assert result["morph_entropy"] > 0

    def test_perplexity_matches_entropy(self, sample_tsv):
        result = compute_metrics_for_file(sample_tsv)
        assert math.isclose(
            result["morph_perplexity"],
            2 ** result["morph_entropy"],
            rel_tol=1e-9,
        )


# ---- Real data test (skipped if not available) ----

REAL_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "2_segmented" / "eng_MorphoLex-eng_Latn.tsv"


@pytest.mark.skipif(not REAL_FILE.exists(), reason="Real segmented data not available")
class TestRealData:
    def test_english_sanity(self):
        result = compute_metrics_for_file(REAL_FILE)
        assert result["word_count"] > 1000
        assert result["morph_count"] > 100
        assert result["word_ttr"] < 0.1
        assert result["morph_entropy"] > 0
        assert result["compounding_index"] > 0
        assert result["root_entropy"] > 0
        assert result["affix_entropy"] > 0
