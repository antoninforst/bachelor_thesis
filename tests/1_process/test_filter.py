"""Tests for the raw-file filtering pipeline."""

import csv
from pathlib import Path

from hapax_overlap import main as hapax_main
from script_check import main as script_main
from generate_ignore_list import main as ignore_main


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class TestHapaxOverlap:
    """hapax_overlap.py produces correct overlap CSV."""

    def test_creates_output(self, raw_dir, out_dir):
        hapax_main(["--raw-dir", str(raw_dir), "--out-dir", str(out_dir)])
        assert (out_dir / "hapax_overlap.csv").exists()

    def test_correct_columns(self, raw_dir, out_dir):
        hapax_main(["--raw-dir", str(raw_dir), "--out-dir", str(out_dir)])
        rows = _read_csv(out_dir / "hapax_overlap.csv")
        expected = {"lang", "file_a", "file_b", "hapaxes_a", "hapaxes_b",
                    "overlap", "share_a_pct", "share_b_pct"}
        assert set(rows[0].keys()) == expected

    def test_detects_aaa_overlap(self, raw_dir, out_dir):
        hapax_main(["--raw-dir", str(raw_dir), "--out-dir", str(out_dir)])
        rows = _read_csv(out_dir / "hapax_overlap.csv")
        aaa_pairs = [r for r in rows if r["lang"] == "aaa"]
        assert len(aaa_pairs) == 3

    def test_high_overlap_source1_source2(self, raw_dir, out_dir):
        """source2 is smaller and all its hapaxes also occur in source1."""
        hapax_main(["--raw-dir", str(raw_dir), "--out-dir", str(out_dir)])
        rows = _read_csv(out_dir / "hapax_overlap.csv")
        pair = [r for r in rows
                if {r["file_a"], r["file_b"]} == {"aaa_source1.csv", "aaa_source2.csv"}]
        assert len(pair) == 1
        assert float(pair[0]["share_b_pct"]) == 100.0

    def test_bbb_no_overlap(self, raw_dir, out_dir):
        """bbb files have no overlapping hapaxes."""
        hapax_main(["--raw-dir", str(raw_dir), "--out-dir", str(out_dir)])
        rows = _read_csv(out_dir / "hapax_overlap.csv")
        bbb = [r for r in rows if r["lang"] == "bbb"]
        assert len(bbb) == 1
        assert int(bbb[0]["overlap"]) == 0

    def test_filter_by_lang(self, raw_dir, out_dir):
        """Passing specific language codes limits output."""
        hapax_main(["bbb", "--raw-dir", str(raw_dir), "--out-dir", str(out_dir)])
        rows = _read_csv(out_dir / "hapax_overlap.csv")
        assert all(r["lang"] == "bbb" for r in rows)


class TestGenerateIgnoreList:
    """generate_ignore_list.py flags the right files."""

    def _run_hapax(self, raw_dir, out_dir):
        hapax_main(["--raw-dir", str(raw_dir), "--out-dir", str(out_dir)])

    def _run_hapax_ignore(self, raw_dir, out_dir):
        self._run_hapax(raw_dir, out_dir)
        ignore_main([
            "--overlap-csv", str(out_dir / "hapax_overlap.csv"),
            "--out-dir", str(out_dir),
        ])

    def test_creates_ignore_csv(self, raw_dir, out_dir):
        self._run_hapax_ignore(raw_dir, out_dir)
        assert (out_dir / "ignored_files.csv").exists()

    def test_ignores_source2_not_source1(self, raw_dir, out_dir):
        """source2 has 100% overlap, so the smaller overlapping file is ignored."""
        self._run_hapax_ignore(raw_dir, out_dir)
        rows = _read_csv(out_dir / "ignored_files.csv")
        ignored_names = {r["filename"] for r in rows}
        assert "aaa_source2.csv" in ignored_names
        assert "aaa_source1.csv" not in ignored_names

    def test_hapax_only_does_not_add_script_ignores(self, raw_dir, out_dir):
        """generate_ignore_list stays hapax-only unless --script-csv is passed."""
        self._run_hapax(raw_dir, out_dir)
        script_main(["--raw-dir", str(raw_dir), "--out-dir", str(out_dir)])
        ignore_main([
            "--overlap-csv", str(out_dir / "hapax_overlap.csv"),
            "--out-dir", str(out_dir),
        ])

        rows = _read_csv(out_dir / "ignored_files.csv")
        ignored_names = {r["filename"] for r in rows}
        assert "aaa_source2.csv" in ignored_names
        assert "aaa_source3.csv" not in ignored_names

    def test_bbb_not_ignored(self, raw_dir, out_dir):
        """bbb files have 0 overlap, so neither bbb file is ignored."""
        self._run_hapax_ignore(raw_dir, out_dir)
        rows = _read_csv(out_dir / "ignored_files.csv")
        ignored_names = {r["filename"] for r in rows}
        assert not any(n.startswith("bbb") for n in ignored_names)


class TestScriptCheck:
    """script_check.py detects wrong-script files."""

    def test_creates_output(self, raw_dir, out_dir):
        script_main(["--raw-dir", str(raw_dir), "--out-dir", str(out_dir)])
        assert (out_dir / "script_check.csv").exists()

    def test_aaa_source3_flagged(self, raw_dir, out_dir):
        """Cyrillic file in a Latin-dominant language gets low language-script share."""
        script_main(["--raw-dir", str(raw_dir), "--out-dir", str(out_dir)])
        rows = _read_csv(out_dir / "script_check.csv")
        source3 = [r for r in rows if r["file"] == "aaa_source3.csv"]
        assert len(source3) == 1
        assert source3[0]["language_script"] == "Latin"
        assert source3[0]["file_script"] == "Cyrillic"
        assert float(source3[0]["language_script_share"]) < 5.0

    def test_output_columns_are_explicit(self, raw_dir, out_dir):
        script_main(["--raw-dir", str(raw_dir), "--out-dir", str(out_dir)])
        rows = _read_csv(out_dir / "script_check.csv")
        expected = {
            "file", "language", "language_script", "file_script",
            "language_script_share", "file_script_share",
            "second_language_script_share",
        }
        assert expected.issubset(rows[0].keys())

    def test_unlisted_letter_script_is_kept_as_other(self, tmp_path, out_dir):
        raw_dir = tmp_path / "raw_other"
        raw_dir.mkdir()
        path = raw_dir / "xxx_other.csv"
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["word", "frequency"])
            writer.writerow(["𐎠𐎡𐎢", 10])
            writer.writerow(["𐎣𐎤𐎥", 5])

        script_main(["--raw-dir", str(raw_dir), "--out-dir", str(out_dir)])
        rows = _read_csv(out_dir / "script_check.csv")
        assert rows[0]["file_script"] == "Other"
        assert rows[0]["Other"] == "100.0"

    def test_respects_ignore_csv(self, raw_dir, out_dir):
        """Files listed in --ignore-csv are skipped."""
        ignore_path = out_dir / "ignored_files.csv"
        with open(ignore_path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["filename", "lang", "reason"])
            w.writerow(["aaa_source2.csv", "aaa", "test"])
        script_main([
            "--raw-dir", str(raw_dir),
            "--out-dir", str(out_dir),
            "--ignore-csv", str(ignore_path),
        ])
        rows = _read_csv(out_dir / "script_check.csv")
        files = {r["file"] for r in rows}
        assert "aaa_source2.csv" not in files


class TestFullFilterPipeline:
    """End-to-end: the 4-step filter pipeline produces a correct final ignore list."""

    def test_full_pipeline(self, raw_dir, out_dir):
        hapax_main(["--raw-dir", str(raw_dir), "--out-dir", str(out_dir)])

        ignore_main([
            "--overlap-csv", str(out_dir / "hapax_overlap.csv"),
            "--out-dir", str(out_dir),
        ])

        script_main([
            "--raw-dir", str(raw_dir),
            "--out-dir", str(out_dir),
            "--ignore-csv", str(out_dir / "ignored_files.csv"),
        ])

        ignore_main([
            "--overlap-csv", str(out_dir / "hapax_overlap.csv"),
            "--script-csv", str(out_dir / "script_check.csv"),
            "--out-dir", str(out_dir),
        ])

        rows = _read_csv(out_dir / "ignored_files.csv")
        ignored = {r["filename"] for r in rows}

        assert "aaa_source2.csv" in ignored
        assert "aaa_source3.csv" in ignored
        assert "aaa_source1.csv" not in ignored
        assert not any(n.startswith("bbb") for n in ignored)
