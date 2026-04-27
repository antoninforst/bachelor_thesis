"""
Convert Bible XML files into frequency-list CSVs compatible with data/0_raw/.

Reads XML files from data/0_1_bible/bibles/, extracts words from <seg> verses,
counts word-type frequencies, and writes CSVs to data/0_raw/ named as
{iso639_code}-bible.csv using codes from lepzig_shortcuts.csv.

Only files whose iso639 attribute matches a code in lepzig_shortcuts.csv are
processed. When multiple XMLs map to the same code, the one producing the
most word types is kept.

Usage:
    python src/1_process/0_download/bible/get_bible.py
    python src/1_process/0_download/bible/get_bible.py --dry-run
"""

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BIBLE_DIR = Path("data/0_1_bible/bibles")
RAW_DIR = Path("data/0_raw")
SHORTCUTS_CSV = Path("src/1_process/0_download/leipzig/lepzig_shortcuts.csv")

_HAS_LETTER = re.compile(r"[^\W\d_]").search

# ISO 639-2/B (bibliographic) → ISO 639-3/T codes used in Leipzig shortcuts
_ISO_ALIAS: dict[str, str] = {
    "alb": "sqi",   # Albanian
    "arm": "hye",   # Armenian
    "baq": "eus",   # Basque
    "chi": "cmn",   # Chinese → Mandarin
    "cze": "ces",   # Czech
    "gre": "ell",   # Greek
    "jap": "jpn",   # Japanese
    "kor": "kor",   # Korean (same)
    "mao": "mri",   # Maori
    "rum": "ron",   # Romanian
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_codes(path: Path) -> set[str]:
    """Load valid 3-letter codes from lepzig_shortcuts.csv."""
    codes: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            codes.add(row["code"])
    return codes


def _get_iso639_and_name(xml_path: Path) -> tuple[Optional[str], str]:
    """Extract (iso639 code, language name) from a Bible XML.

    The code is normalized via the alias map.
    """
    tree = ET.parse(xml_path)
    lang_el = tree.find(".//language")
    if lang_el is not None:
        code = lang_el.get("iso639")
        name = (lang_el.text or "").strip() or "?"
        if code:
            return _ISO_ALIAS.get(code, code), name
    return None, "?"


def _extract_frequencies(xml_path: Path) -> Counter:
    """Parse a Bible XML and return word-type frequency counts."""
    tree = ET.parse(xml_path)
    freq: Counter = Counter()

    for seg in tree.iter("seg"):
        text = seg.text
        if not text:
            continue
        for word in text.split():
            # Strip leading/trailing punctuation but keep internal
            word = word.strip(".,;:!?\"'()[]{}«»""''—–-…·").lower()
            if word and _HAS_LETTER(word):
                freq[word] += 1

    return freq


def _write_csv(out_path: Path, freq: Counter) -> None:
    """Write a frequency CSV in word,frequency format."""
    sorted_items = freq.most_common()
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["word", "frequency"])
        writer.writerows(sorted_items)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Bible XMLs to frequency-list CSVs."
    )
    parser.add_argument(
        "--bible-dir", type=Path, default=BIBLE_DIR,
        help=f"Directory with Bible XML files (default: {BIBLE_DIR}).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=RAW_DIR,
        help=f"Output directory (default: {RAW_DIR}).",
    )
    parser.add_argument(
        "--shortcuts", type=Path, default=SHORTCUTS_CSV,
        help=f"Path to lepzig_shortcuts.csv (default: {SHORTCUTS_CSV}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without writing files.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    valid_codes = _load_codes(args.shortcuts)
    xml_files = sorted(args.bible_dir.glob("*.xml"))

    if not xml_files:
        print(f"No XML files found in {args.bible_dir}")
        sys.exit(1)

    print(f"Found {len(xml_files)} Bible XML files")
    print(f"Valid language codes: {len(valid_codes)}")
    print(f"{'=' * 60}")

    # Group XML files by matching code
    # {code: [(xml_path, iso), ...]}
    code_to_xmls: dict[str, list[Path]] = {}
    skipped = []

    for xml_path in xml_files:
        iso, lang_name = _get_iso639_and_name(xml_path)
        if iso and iso in valid_codes:
            code_to_xmls.setdefault(iso, []).append(xml_path)
        else:
            skipped.append((xml_path.name, iso or "?", lang_name))

    print(f"Matched: {sum(len(v) for v in code_to_xmls.values())} files "
          f"-> {len(code_to_xmls)} codes")
    print(f"Skipped: {len(skipped)} files (no matching code)")

    if skipped:
        for name, iso, lang_name in skipped:
            print(f"  SKIP {name} (iso639={iso}, {lang_name})")

    # Process each code -- if multiple XMLs, pick the one with most word types
    written = 0
    for code in sorted(code_to_xmls):
        xmls = code_to_xmls[code]
        best_freq: Counter = Counter()
        best_name = ""

        for xml_path in xmls:
            freq = _extract_frequencies(xml_path)
            if len(freq) > len(best_freq):
                best_freq = freq
                best_name = xml_path.name

        out_path = args.out_dir / f"{code}_bible.csv"
        extra = f" (from {best_name})" if len(xmls) > 1 else ""
        if args.dry_run:
            print(f"  {code}: {len(best_freq):,} words{extra} [DRY RUN]")
        else:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            _write_csv(out_path, best_freq)
            print(f"  {code}: {len(best_freq):,} words{extra} -> {out_path.name}")
        written += 1

    print(f"Done. Wrote {written} frequency files.")


if __name__ == "__main__":
    main()
