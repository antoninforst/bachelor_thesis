"""Tests for quality_check.py metrics."""

import csv
import pytest
from pathlib import Path

from quality_check import (
    _percentile,
    _boxplot_upper_fence,
    _dominant_script,
    _is_program_artifact,
    _analyse_file,
    main,
    SCRIPTS_CSV,
)
from script_check import ScriptDetector

_DETECTOR = ScriptDetector.from_csv(SCRIPTS_CSV)


# ── Helpers ──────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[list[str]]) -> Path:
    """Write a CSV with header ['word', 'frequency']."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["word", "frequency"])
        w.writerows(rows)
    return path


# ── _percentile ──────────────────────────────────────────────────────

class TestPercentile:
    def test_empty(self):
        assert _percentile([], 50) == 0.0

    def test_single(self):
        assert _percentile([7], 25) == 7.0

    def test_median_odd(self):
        assert _percentile([1, 2, 3, 4, 5], 50) == 3.0

    def test_q1_q3(self):
        data = list(range(1, 101))
        assert _percentile(data, 25) == pytest.approx(25.75)
        assert _percentile(data, 75) == pytest.approx(75.25)


# ── _boxplot_upper_fence ─────────────────────────────────────────────

class TestBoxplotUpperFence:
    def test_uniform_lengths(self):
        # All same length → IQR = 0 → fence = Q3
        data = [5, 5, 5, 5]
        assert _boxplot_upper_fence(data, 1.5) == 5.0

    def test_known_values(self):
        # Q1=2, Q3=4 → IQR=2, fence = 4 + 1.5*2 = 7
        data = [1, 2, 3, 4, 5]
        fence = _boxplot_upper_fence(data, 1.5)
        assert fence == pytest.approx(7.0)

    def test_multiplier_3(self):
        data = [1, 2, 3, 4, 5]
        fence = _boxplot_upper_fence(data, 3.0)
        assert fence == pytest.approx(10.0)


# ── _dominant_script ─────────────────────────────────────────────────

class TestDominantScript:
    def test_latin(self):
        assert _dominant_script("hello", _DETECTOR) == "Latin"

    def test_cyrillic(self):
        assert _dominant_script("привет", _DETECTOR) == "Cyrillic"

    def test_mixed_returns_majority(self):
        # 3 Latin + 1 Cyrillic → Latin
        assert _dominant_script("helло", _DETECTOR) == "Latin"

    def test_only_common(self):
        # Characters outside all known script ranges → Common
        assert _dominant_script("\u2600\u2601", _DETECTOR) == "Common"

    def test_arabic(self):
        assert _dominant_script("مرحبا", _DETECTOR) == "Arabic"


# ── _is_program_artifact ─────────────────────────────────────────────

class TestIsProgramArtifact:
    @pytest.mark.parametrize("word", [
        "www.example.com",
        "http://x",
        "https://x",
        "{json}",
        "path\\file",
        "user@email",
        "a.b.c",
        "1,2,3",
    ])
    def test_detected(self, word):
        assert _is_program_artifact(word) is True

    @pytest.mark.parametrize("word", [
        "hello",
        "world",
        "one.two",
        "a,b",
        "self-aware",
    ])
    def test_not_detected(self, word):
        assert _is_program_artifact(word) is False


# ── _analyse_file ────────────────────────────────────────────────────

class TestAnalyseFile:
    def test_empty_file(self, tmp_path):
        p = _write_csv(tmp_path / "empty.csv", [])
        m = _analyse_file(p, "Latin", "zzz", _DETECTOR)
        assert m["total_types"] == 0
        assert m["total_frequency"] == 0
        assert m["corrupted_pct"] == 0.0
        assert m["corrupted_freq_pct"] == 0.0
        assert m["eng_stopwords"] == 0

    def test_clean_latin(self, tmp_path):
        rows = [["the", "100"], ["of", "80"], ["and", "60"], ["to", "50"],
                ["in", "40"], ["a", "30"], ["is", "20"], ["it", "10"]]
        p = _write_csv(tmp_path / "clean.csv", rows)
        m = _analyse_file(p, "Latin", "eng", _DETECTOR)
        assert m["foreign_script_pct"] == 0.0
        assert m["program_pct"] == 0.0
        assert m["corrupted_pct"] == 0.0
        assert m["total_types"] == 8
        assert m["total_frequency"] == 390
        # English dataset → always 0
        assert m["eng_stopwords"] == 0

    def test_foreign_script_detected(self, tmp_path):
        # File claims Latin, but one word is Cyrillic
        rows = [["hello", "100"], ["world", "80"], ["привет", "50"]]
        p = _write_csv(tmp_path / "mixed.csv", rows)
        m = _analyse_file(p, "Latin", "zzz", _DETECTOR)
        assert m["foreign_script_pct"] > 0
        assert m["corrupted_pct"] > 0

    def test_no_foreign_without_expected_script(self, tmp_path):
        rows = [["hello", "100"], ["привет", "50"]]
        p = _write_csv(tmp_path / "noexpect.csv", rows)
        m = _analyse_file(p, None, "zzz", _DETECTOR)
        assert m["foreign_script_pct"] == 0.0

    def test_long_outlier_detected(self, tmp_path):
        # 9 short words + 1 very long word
        rows = [["a", "10"]] * 9 + [["a" * 100, "1"]]
        p = _write_csv(tmp_path / "long.csv", rows)
        m = _analyse_file(p, "Latin", "zzz", _DETECTOR)
        assert m["long_outlier_pct"] > 0
        assert m["corrupted_pct"] > 0

    def test_program_artifact_counted(self, tmp_path):
        rows = [["hello", "100"], ["www.google.com", "5"], ["user@mail", "3"]]
        p = _write_csv(tmp_path / "prog.csv", rows)
        m = _analyse_file(p, "Latin", "zzz", _DETECTOR)
        assert m["program_pct"] > 0
        assert m["corrupted_pct"] > 0
        assert m["corrupted_freq_pct"] > 0
        # corrupted freq = 5+3=8 out of 108 total
        assert m["corrupted_freq_pct"] == pytest.approx(100.0 * 8 / 108, abs=0.01)

    def test_punct_char_pct(self, tmp_path):
        # "a!" has 1 punct char out of 2, freq=100 → 50% of chars are punct
        rows = [["a!", "100"]]
        p = _write_csv(tmp_path / "punct.csv", rows)
        m = _analyse_file(p, "Latin", "zzz", _DETECTOR)
        assert m["punct_char_pct"] == pytest.approx(50.0)

    def test_corrupted_is_union(self, tmp_path):
        # A word that is BOTH long AND a program artifact should only be counted once
        rows = [["a", "10"]] * 9 + [["http://" + "x" * 200, "1"]]
        p = _write_csv(tmp_path / "union.csv", rows)
        m = _analyse_file(p, "Latin", "zzz", _DETECTOR)
        # corrupted_pct should equal max(long_outlier_pct, program_pct), not sum
        assert m["corrupted_pct"] <= m["long_outlier_pct"] + m["program_pct"]

    def test_top20_long(self, tmp_path):
        # 20 words: 19 short (len 2-3), 1 long (len 15) — all high frequency
        rows = [[f"w{i}", str(100 - i)] for i in range(19)]
        rows.append(["averylongishword", "90"])  # 16 chars among 2-3 char words
        p = _write_csv(tmp_path / "top20.csv", rows)
        m = _analyse_file(p, "Latin", "zzz", _DETECTOR)
        assert m["top20_long_pct"] > 0

    def test_eng_stopwords_non_english(self, tmp_path):
        rows = [["the", "100"], ["with", "50"], ["however", "30"], ["slovo", "80"]]
        p = _write_csv(tmp_path / "ces.csv", rows)
        m = _analyse_file(p, "Latin", "ces", _DETECTOR)
        assert m["eng_stopwords"] == 3

    def test_eng_stopwords_english_always_zero(self, tmp_path):
        rows = [["the", "100"], ["with", "50"], ["however", "30"]]
        p = _write_csv(tmp_path / "eng.csv", rows)
        m = _analyse_file(p, "Latin", "eng", _DETECTOR)
        assert m["eng_stopwords"] == 0


# ── main (integration) ──────────────────────────────────────────────

class TestMain:
    def test_writes_output(self, tmp_path):
        in_dir = tmp_path / "input"
        in_dir.mkdir()
        _write_csv(in_dir / "eng_Latn.csv", [["the", "100"], ["of", "80"]])

        out = tmp_path / "quality.csv"

        # Create a minimal scripts.csv
        scripts_csv = tmp_path / "scripts.csv"
        scripts_csv.write_text("Latin,Latn\nCyrillic,Cyrl\n", encoding="utf-8")

        import quality_check
        orig = quality_check.SCRIPTS_CSV
        quality_check.SCRIPTS_CSV = scripts_csv
        try:
            main(["--input-dir", str(in_dir), "--output", str(out)])
        finally:
            quality_check.SCRIPTS_CSV = orig

        assert out.exists()
        with open(out, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["language"] == "eng"
        assert rows[0]["script"] == "Latn"
        assert "corrupted_pct" in rows[0]
        assert "corrupted_freq_pct" in rows[0]

    def test_lang_filter(self, tmp_path):
        in_dir = tmp_path / "input"
        in_dir.mkdir()
        _write_csv(in_dir / "eng_Latn.csv", [["the", "100"]])
        _write_csv(in_dir / "fra_Latn.csv", [["le", "100"]])

        out = tmp_path / "quality.csv"
        scripts_csv = tmp_path / "scripts.csv"
        scripts_csv.write_text("Latin,Latn\n", encoding="utf-8")

        import quality_check
        orig = quality_check.SCRIPTS_CSV
        quality_check.SCRIPTS_CSV = scripts_csv
        try:
            main(["eng", "--input-dir", str(in_dir), "--output", str(out)])
        finally:
            quality_check.SCRIPTS_CSV = orig

        with open(out, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["language"] == "eng"
