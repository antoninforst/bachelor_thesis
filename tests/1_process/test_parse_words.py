"""Tests for parse_words: frequency conservation and segmentation logic."""
import csv
from pathlib import Path

import pytest

from parse_words import (
    _read_freq_file,
    _write_freq_file,
    segment_file,
    _script_code_from_stem,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_csv(tmp_path: Path, rows: list[tuple[str, int]]) -> Path:
    """Write a word-frequency CSV and return its path."""
    p = tmp_path / "test_Hani.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["word", "frequency"])
        w.writerows(rows)
    return p


def _read_result(path: Path) -> dict[str, int]:
    """Read back a word-frequency CSV into a dict."""
    return _read_freq_file(path)


def _dummy_segmenter(word: str) -> list[str]:
    """Split every 2 chars: 'abcd' -> ['ab', 'cd']."""
    if len(word) <= 2:
        return [word]
    return [word[i:i + 2] for i in range(0, len(word), 2)]


# ---------------------------------------------------------------------------
# _script_code_from_stem
# ---------------------------------------------------------------------------

class TestScriptCodeFromStem:
    def test_hani(self):
        assert _script_code_from_stem("zho_Hani") == "Hani"

    def test_thai(self):
        assert _script_code_from_stem("tha_Thai") == "Thai"

    def test_no_script(self):
        assert _script_code_from_stem("eng_Latn") is None

    def test_plain(self):
        assert _script_code_from_stem("eng") is None


# ---------------------------------------------------------------------------
# _read / _write round-trip
# ---------------------------------------------------------------------------

class TestReadWrite:
    def test_round_trip(self, tmp_path):
        data = {"hello": 100, "world": 50}
        p = tmp_path / "rw.csv"
        _write_freq_file(p, data)
        result = _read_freq_file(p)
        assert result == data

    def test_sorted_descending(self, tmp_path):
        data = {"a": 1, "b": 999, "c": 50}
        p = tmp_path / "sorted.csv"
        _write_freq_file(p, data)
        with open(p, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)
        assert rows[0][0] == "b"
        assert rows[-1][0] == "a"


# ---------------------------------------------------------------------------
# segment_file — frequency conservation
# ---------------------------------------------------------------------------

class TestSegmentFile:
    def test_no_segmentation_single_char(self, tmp_path):
        """Single-char words skip segmentation entirely."""
        rows = [("a", 10), ("b", 20)]
        p = _make_csv(tmp_path, rows)
        before, after, seg_count = segment_file(p, _dummy_segmenter)
        result = _read_result(p)
        assert result == {"a": 10, "b": 20}
        assert seg_count == 0

    def test_no_split_short_word(self, tmp_path):
        """Two-char word: dummy segmenter returns it as-is."""
        rows = [("ab", 50)]
        p = _make_csv(tmp_path, rows)
        before, after, seg_count = segment_file(p, _dummy_segmenter)
        result = _read_result(p)
        assert result["ab"] == 50
        assert seg_count == 0

    def test_split_distributes_frequency(self, tmp_path):
        """'abcd' (freq 100) -> 'ab' + 'cd', each gets 100."""
        rows = [("abcd", 100)]
        p = _make_csv(tmp_path, rows)
        before, after, seg_count = segment_file(p, _dummy_segmenter)
        result = _read_result(p)
        assert result["ab"] == 100
        assert result["cd"] == 100
        assert seg_count == 1

    def test_split_merges_parts(self, tmp_path):
        """Two words that share a segment part should merge frequencies."""
        rows = [("abcd", 100), ("abef", 50)]
        p = _make_csv(tmp_path, rows)
        before, after, seg_count = segment_file(p, _dummy_segmenter)
        result = _read_result(p)
        # ab appears from both: 100 + 50
        assert result["ab"] == 150
        assert result["cd"] == 100
        assert result["ef"] == 50
        assert seg_count == 2

    def test_total_frequency_conserved_simple(self, tmp_path):
        """Total frequency in == total frequency out for non-splitting words."""
        rows = [("ab", 30), ("cd", 70)]
        p = _make_csv(tmp_path, rows)
        total_in = sum(f for _, f in rows)
        segment_file(p, _dummy_segmenter)
        result = _read_result(p)
        total_out = sum(result.values())
        assert total_out == total_in

    def test_single_char_passthrough_with_multi(self, tmp_path):
        """Mix of single-char (skipped) and multi-char (segmented)."""
        rows = [("x", 10), ("abcd", 200)]
        p = _make_csv(tmp_path, rows)
        before, after, seg_count = segment_file(p, _dummy_segmenter)
        result = _read_result(p)
        assert result["x"] == 10
        assert result["ab"] == 200
        assert result["cd"] == 200
        assert seg_count == 1
        assert before == 2

    def test_duplicate_input_rows_summed(self, tmp_path):
        """If input has duplicate words, their freqs are summed first."""
        rows = [("abcd", 60), ("abcd", 40)]
        p = _make_csv(tmp_path, rows)
        segment_file(p, _dummy_segmenter)
        result = _read_result(p)
        assert result["ab"] == 100
        assert result["cd"] == 100

    def test_types_before_after(self, tmp_path):
        """Check returned type counts."""
        rows = [("abcd", 10), ("abef", 20), ("xy", 5)]
        p = _make_csv(tmp_path, rows)
        before, after, seg_count = segment_file(p, _dummy_segmenter)
        assert before == 3  # abcd, abef, xy
        result = _read_result(p)
        assert after == len(result)

    def test_empty_file(self, tmp_path):
        """Empty file doesn't crash."""
        rows: list[tuple[str, int]] = []
        p = _make_csv(tmp_path, rows)
        before, after, seg_count = segment_file(p, _dummy_segmenter)
        assert before == 0
        assert after == 0
        assert seg_count == 0

    def test_segmenter_returns_empty(self, tmp_path):
        """If segmenter returns empty parts, word is preserved."""
        def empty_seg(w):
            return ["", " "]
        rows = [("ab", 10)]
        p = _make_csv(tmp_path, rows)
        segment_file(p, empty_seg)
        result = _read_result(p)
        assert result.get("ab", 0) == 10


# ---------------------------------------------------------------------------
# segment_file — real Chinese via jieba
# ---------------------------------------------------------------------------

jieba = pytest.importorskip("jieba")
from parse_words import _segment_chinese


class TestSegmentFileChinese:
    def test_shared_prefix_merges(self, tmp_path):
        """中国人(100) + 中国话(50) -> 中国=150, 人=100, 话=50."""
        rows = [("中国人", 100), ("中国话", 50)]
        p = _make_csv(tmp_path, rows)
        before, after, seg_count = segment_file(p, _segment_chinese)
        result = _read_result(p)
        assert result["中国"] == 150
        assert result["人"] == 100
        assert result["话"] == 50
        assert seg_count == 2

    def test_single_char_skipped(self, tmp_path):
        """Single Chinese character passes through without segmenter call."""
        rows = [("人", 200)]
        p = _make_csv(tmp_path, rows)
        before, after, seg_count = segment_file(p, _segment_chinese)
        result = _read_result(p)
        assert result["人"] == 200
        assert seg_count == 0

    def test_unsplittable_word(self, tmp_path):
        """A two-char word that jieba keeps intact."""
        rows = [("中国", 300)]
        p = _make_csv(tmp_path, rows)
        before, after, seg_count = segment_file(p, _segment_chinese)
        result = _read_result(p)
        assert result["中国"] == 300
        assert seg_count == 0

    def test_total_frequency_conserved(self, tmp_path):
        """Total frequency must not change after segmentation."""
        rows = [("中国人民", 100), ("人民日报", 80), ("中国", 50)]
        p = _make_csv(tmp_path, rows)
        total_in = sum(f for _, f in rows)
        segment_file(p, _segment_chinese)
        result = _read_result(p)
        total_out = sum(result.values())
        # Each split word distributes its freq to all parts, so total
        # out >= total in. But no frequency should be lost.
        assert total_out >= total_in
