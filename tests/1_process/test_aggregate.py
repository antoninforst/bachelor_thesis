"""
Integration tests for the aggregation step.

Tests: aggregate.py merging raw files, cleaning, ignoring, and report generation.
"""

import csv
from pathlib import Path

import aggregate
from aggregate import main as agg_main
from lib import load_script_codes


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, header: list[str], rows: list[list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def _write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class TestAggregate:
    """aggregate.py merges and cleans raw frequency files."""

    def test_creates_aggregated_files(self, raw_dir, out_dir, tmp_path):
        agg_out = tmp_path / "agg"
        agg_main(["--raw-dir", str(raw_dir), "--out-dir", str(agg_out)])
        assert (agg_out / "aaa.csv").exists()
        assert (agg_out / "bbb.csv").exists()

    def test_merged_frequencies(self, raw_dir, tmp_path):
        """Frequencies from multiple files are summed."""
        agg_out = tmp_path / "agg"
        agg_main(["--raw-dir", str(raw_dir), "--out-dir", str(agg_out)])
        rows = _read_csv(agg_out / "aaa.csv")
        freq_map = {r["word"]: int(r["frequency"]) for r in rows}
        # "hello" appears in source1(100) + source2(50) = 150
        # source3 has Cyrillic "привет" not "hello"
        assert freq_map["hello"] == 150

    def test_sorted_by_frequency(self, raw_dir, tmp_path):
        agg_out = tmp_path / "agg"
        agg_main(["--raw-dir", str(raw_dir), "--out-dir", str(agg_out)])
        rows = _read_csv(agg_out / "aaa.csv")
        freqs = [int(r["frequency"]) for r in rows]
        assert freqs == sorted(freqs, reverse=True)

    def test_punctuation_stripped(self, tmp_path):
        """Words with leading/trailing punctuation are cleaned."""
        raw = tmp_path / "raw_punct"
        raw.mkdir()
        _write_csv(raw / "xxx_test.csv", ["word", "frequency"], [
            ['"hello"', 10],
            ["hello", 5],
            ["...world...", 3],
            ["world", 2],
        ])
        agg_out = tmp_path / "agg_punct"
        agg_main(["--raw-dir", str(raw), "--out-dir", str(agg_out)])
        rows = _read_csv(agg_out / "xxx.csv")
        freq_map = {r["word"]: int(r["frequency"]) for r in rows}
        assert freq_map["hello"] == 15
        assert freq_map["world"] == 5

    def test_lowercase_merge(self, tmp_path):
        """Upper and lowercase forms are merged."""
        raw = tmp_path / "raw_case"
        raw.mkdir()
        _write_csv(raw / "yyy_test.csv", ["word", "frequency"], [
            ["Hello", 10],
            ["hello", 5],
            ["HELLO", 3],
        ])
        agg_out = tmp_path / "agg_case"
        agg_main(["--raw-dir", str(raw), "--out-dir", str(agg_out)])
        rows = _read_csv(agg_out / "yyy.csv")
        assert len(rows) == 1
        assert rows[0]["word"] == "hello"
        assert int(rows[0]["frequency"]) == 18

    def test_non_words_filtered(self, tmp_path):
        """Entries without letters are removed."""
        raw = tmp_path / "raw_nonword"
        raw.mkdir()
        _write_csv(raw / "zzz_test.csv", ["word", "frequency"], [
            ["123", 50],
            ["...", 30],
            ["hello", 10],
        ])
        agg_out = tmp_path / "agg_nonword"
        agg_main(["--raw-dir", str(raw), "--out-dir", str(agg_out)])
        rows = _read_csv(agg_out / "zzz.csv")
        words = {r["word"] for r in rows}
        assert "hello" in words
        assert "123" not in words
        assert "..." not in words

    def test_ignore_flag(self, raw_dir, tmp_path):
        """Files in the ignore list are excluded from aggregation."""
        ignore_csv = tmp_path / "ignored.csv"
        _write_csv(ignore_csv, ["filename", "lang", "reason"], [
            ["aaa_source2.csv", "aaa", "test"],
            ["aaa_source3.csv", "aaa", "test"],
        ])
        agg_out = tmp_path / "agg_ignore"
        agg_main([
            "--raw-dir", str(raw_dir),
            "--out-dir", str(agg_out),
            "--ignore", str(ignore_csv),
        ])
        rows = _read_csv(agg_out / "aaa.csv")
        freq_map = {r["word"]: int(r["frequency"]) for r in rows}
        # Only source1 remains: hello=100, no Cyrillic words
        assert freq_map["hello"] == 100
        cyrillic = [w for w in freq_map if any(ord(c) > 0x400 for c in w)]
        assert cyrillic == []

    def test_filter_by_lang(self, raw_dir, tmp_path):
        """Passing language codes limits which languages are processed."""
        agg_out = tmp_path / "agg_lang"
        agg_main(["bbb", "--raw-dir", str(raw_dir), "--out-dir", str(agg_out)])
        assert (agg_out / "bbb.csv").exists()
        assert not (agg_out / "aaa.csv").exists()

    def test_no_report_matches_default_output(self, raw_dir, tmp_path):
        """--no-report skips report metrics without changing aggregate CSVs."""
        normal_out = tmp_path / "agg_normal"
        no_report_out = tmp_path / "agg_no_report"
        agg_main(["--raw-dir", str(raw_dir), "--out-dir", str(normal_out)])
        agg_main(["--raw-dir", str(raw_dir), "--out-dir", str(no_report_out), "--no-report"])
        assert _read_csv(normal_out / "aaa.csv") == _read_csv(no_report_out / "aaa.csv")
        assert _read_csv(normal_out / "bbb.csv") == _read_csv(no_report_out / "bbb.csv")

    def test_load_script_codes_accepts_comma_and_semicolon(self, tmp_path):
        comma = tmp_path / "scripts_comma.csv"
        semicolon = tmp_path / "scripts_semicolon.csv"
        _write_text(comma, "Latin,Latn\nThai,Thai\n")
        _write_text(semicolon, "Latin;Latn\nThai;Thai\n")

        assert load_script_codes(comma) == {"Latin": "Latn", "Thai": "Thai"}
        assert load_script_codes(semicolon) == {"Latin": "Latn", "Thai": "Thai"}

    def test_skip_existing_uses_expected_script_suffix(self, tmp_path):
        """An old fallback lang.csv must not skip the expected lang_Script.csv."""
        raw = tmp_path / "raw_suffix"
        raw.mkdir()
        _write_csv(raw / "lll_source.csv", ["word", "frequency"], [["hello", 10]])

        agg_out = tmp_path / "agg_suffix"
        agg_out.mkdir()
        _write_csv(agg_out / "lll.csv", ["word", "frequency"], [["old", 1]])

        scripts_csv = tmp_path / "scripts.csv"
        overview_csv = tmp_path / "language_overview.csv"
        _write_text(scripts_csv, "Latin,Latn\n")
        _write_csv(overview_csv, ["used_shortcut", "primary_script"], [["lll", "Latin"]])

        agg_main([
            "--raw-dir", str(raw),
            "--out-dir", str(agg_out),
            "--scripts-csv", str(scripts_csv),
            "--lang-overview", str(overview_csv),
            "--skip-existing",
            "--no-report",
        ])

        assert (agg_out / "lll.csv").exists()
        assert (agg_out / "lll_Latn.csv").exists()

    def test_repair_mode(self, tmp_path):
        """--repair re-cleans an already-aggregated file."""
        agg = tmp_path / "agg_repair"
        agg.mkdir()
        # Write a file with dirty entries
        _write_csv(agg / "rrr.csv", ["word", "frequency"], [
            ['"hello"', 10],
            ["hello", 5],
            ["123", 3],
        ])
        agg_main(["--repair", "--out-dir", str(agg)])
        rows = _read_csv(agg / "rrr.csv")
        freq_map = {r["word"]: int(r["frequency"]) for r in rows}
        assert freq_map["hello"] == 15
        assert "123" not in freq_map

    def test_handles_metadata_header(self, tmp_path):
        """Leipzig-style files with metadata rows before the header are parsed."""
        raw = tmp_path / "raw_meta"
        raw.mkdir()
        meta_file = raw / "mmm_leipzig.csv"
        with open(meta_file, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["corpus", "mmm_mixed_2013"])
            w.writerow(["subcorpus", "-"])
            w.writerow(["Item", "Frequency"])
            w.writerow(["word", 100])
            w.writerow(["test", 50])
        agg_out = tmp_path / "agg_meta"
        agg_main(["--raw-dir", str(raw), "--out-dir", str(agg_out)])
        rows = _read_csv(agg_out / "mmm.csv")
        freq_map = {r["word"]: int(r["frequency"]) for r in rows}
        assert freq_map["word"] == 100
        assert freq_map["test"] == 50
