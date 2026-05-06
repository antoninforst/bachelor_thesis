"""Rename files in data/1_aggregated/ to include script suffix.

eng.csv -> eng_Latn.csv  (if primary_script is Latin)
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AGGREGATED = ROOT / "data" / "1_aggregated"
LANG_OVERVIEW = ROOT / "results" / "1_process" / "2_aggregate" / "language_overview.csv"
SCRIPTS_CSV = ROOT / "src" / "1_process" / "1_filter" / "scripts.csv"
FALLBACK = "xxxx"


def load_script_codes() -> dict[str, str]:
    mapping = {}
    with open(SCRIPTS_CSV, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                mapping[row[0].strip()] = row[1].strip()
    return mapping


def load_lang_scripts(script_codes: dict[str, str]) -> dict[str, str]:
    mapping = {}
    with open(LANG_OVERVIEW, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lang = row["used_shortcut"].strip()
            script_name = row["primary_script"].strip()
            mapping[lang] = script_codes.get(script_name, FALLBACK)
    return mapping


def main() -> None:
    script_codes = load_script_codes()
    lang_scripts = load_lang_scripts(script_codes)

    renamed = 0
    skipped = 0
    for path in sorted(AGGREGATED.glob("*.csv")):
        lang = path.stem
        # skip files with script suffix
        if "_" in lang:
            skipped += 1
            continue
        code = lang_scripts.get(lang, FALLBACK)
        if code == FALLBACK:
            skipped += 1
            continue
        new_name = f"{lang}_{code}.csv"
        new_path = path.parent / new_name
        if new_path.exists():
            print(f"SKIP {path.name} -> {new_name} (already exists)")
            skipped += 1
            continue
        path.rename(new_path)
        print(f"{path.name} -> {new_name}")
        renamed += 1

    print(f"\nRenamed: {renamed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
