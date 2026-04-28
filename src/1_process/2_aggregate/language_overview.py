"""
Generate a comprehensive language overview CSV – a single source of truth.

Combines metadata from Leipzig shortcuts, UD mappings, TIL mappings,
WALS / lang_families, Glot500, aggregated statistics, and script-check
results into one CSV per language used in this repository.

Usage:
    python src/1_process/2_aggregate/language_overview.py
"""

import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
AGGREGATED_DIR = Path("data/1_aggregated")
RAW_DIR = Path("data/0_raw")
STATISTICS_CSV = Path("results/1_process/3_truncate/statistics.csv")
SCRIPT_CHECK_CSV = Path("results/1_process/1_filter/script_check.csv")
LANG_FAMILIES_CSV = Path("data/5_other/lang_families.csv")
LEIPZIG_SHORTCUTS_CSV = Path("src/1_process/0_download/leipzig/lepzig_shortcuts.csv")
UD_MAPPING_CSV = Path("src/1_process/0_download/ud/ud_language_mapping.csv")
OUTPUT_CSV = Path("results/1_process/2_aggregate/language_overview.csv")

# ---------------------------------------------------------------------------
# TIL code -> ISO 639-3 (copied from til/download_til.py)
# ---------------------------------------------------------------------------
TIL_TO_ISO3 = {
    "alt": "alt", "az": "aze", "ba": "bak", "cjs": "cjs", "crh": "crh",
    "cv": "chv", "gag": "gag", "kaa": "kaa", "kjh": "kjh", "kk": "kaz",
    "krc": "krc", "kum": "kum", "ky": "kir", "sah": "sah", "slr": "slr",
    "tk": "tuk", "tr": "tur", "tt": "tat", "tyv": "tyv", "ug": "uig",
    "uum": "uum", "uz": "uzb",
}
# Reverse: ISO 639-3 -> TIL short code
_ISO3_TO_TIL = {v: k for k, v in TIL_TO_ISO3.items()}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_leipzig_shortcuts() -> dict[str, str]:
    """Return {code: language_name}."""
    mapping: dict[str, str] = {}
    with LEIPZIG_SHORTCUTS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["code"].strip()] = row["language"].strip()
    return mapping


def _load_ud_mapping() -> dict[str, str]:
    """Return {iso_code: ud_name}."""
    mapping: dict[str, str] = {}
    with UD_MAPPING_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["code"].strip()] = row["ud_name"].strip()
    return mapping


def _load_lang_families() -> dict[str, dict]:
    """Return {iso_code: {name, family, genus, macroarea, lat, lon, countries}}.

    Some ISO codes appear multiple times; we keep the first occurrence.
    """
    mapping: dict[str, dict] = {}
    with LANG_FAMILIES_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = row["iso_code"].strip()
            if code not in mapping:
                mapping[code] = {
                    "wals_name": row["name"].strip(),
                    "family": row["family"].strip(),
                    "genus": row["genus"].strip(),
                    "macroarea": row["macroarea"].strip(),
                    "latitude": row["lat"].strip(),
                    "longitude": row["lon"].strip(),
                    "countries": row["countries"].strip(),
                }
    return mapping


def _load_statistics() -> dict[str, dict]:
    """Return {lang_code: {distinct_words, total_frequency}}."""
    mapping: dict[str, dict] = {}
    with STATISTICS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = row["file"].strip()
            mapping[code] = {
                "distinct_words": row["distinct_words"].strip(),
                "total_frequency": row["total_frequency"].strip(),
            }
    return mapping


def _load_script_check() -> dict[str, dict[str, float]]:
    """Aggregate per-file script data to per-language.

    For each language, compute mean percentages across its files for every
    script column, then determine dominant and secondary scripts.

    Returns {lang_code: {dominant_script, dominant_pct, secondary_script, secondary_pct}}.
    """
    # Collect per-language: list of dicts {script_name: pct}
    lang_rows: dict[str, list[dict[str, float]]] = {}
    script_columns: list[str] = []

    with SCRIPT_CHECK_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        # Script columns start after the 4 metadata columns
        script_columns = header[4:]

        for row in reader:
            lang = row["language"].strip()
            lang_rows.setdefault(lang, [])
            script_vals = {}
            for sc in script_columns:
                try:
                    script_vals[sc] = float(row[sc])
                except (ValueError, KeyError):
                    script_vals[sc] = 0.0
            lang_rows[lang].append(script_vals)

    result: dict[str, dict] = {}
    for lang, rows in lang_rows.items():
        n = len(rows)
        avg: dict[str, float] = {}
        for sc in script_columns:
            avg[sc] = sum(r[sc] for r in rows) / n

        # Sort scripts by average share descending
        sorted_scripts = sorted(avg.items(), key=lambda x: x[1], reverse=True)
        dominant = sorted_scripts[0] if sorted_scripts else ("", 0.0)
        secondary = sorted_scripts[1] if len(sorted_scripts) > 1 and sorted_scripts[1][1] > 0.5 else ("", 0.0)

        result[lang] = {
            "primary_script": dominant[0] if dominant[1] > 0 else "",
            "primary_script_pct": round(dominant[1], 2),
            "secondary_script": secondary[0] if secondary[1] > 0.5 else "",
            "secondary_script_pct": round(secondary[1], 2) if secondary[1] > 0.5 else 0.0,
        }
    return result


def _load_raw_sources() -> dict[str, set[str]]:
    """Determine which data sources exist for each language in data/0_raw/.

    Returns {lang_code: {source_labels...}} where source labels are like
    'wikipedia', 'news', 'bible', 'glot500', 'til', 'community', etc.
    """
    sources: dict[str, set[str]] = {}
    for path in RAW_DIR.glob("*.csv"):
        name = path.stem
        # Files: {lang}_{source}_{rest} or {lang}_{source}.csv
        parts = name.split("_", 2)
        if len(parts) < 2:
            continue
        lang = parts[0]
        # Handle lang codes with region suffix (e.g., afr-za)
        base_lang = lang.split("-")[0] if "-" in lang else lang
        source = parts[1]
        sources.setdefault(base_lang, set()).add(source)
    return sources


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load all data sources
    leipzig = _load_leipzig_shortcuts()
    ud_map = _load_ud_mapping()
    families = _load_lang_families()
    stats = _load_statistics()
    scripts = _load_script_check()
    raw_sources = _load_raw_sources()

    # Enumerate languages from aggregated directory
    # Filenames are {lang}_{Script}.csv; extract the lang code
    languages = sorted(set(
        p.stem.split("_")[0] for p in AGGREGATED_DIR.glob("*.csv")
    ))

    columns = [
        "used_shortcut",
        "name",
        "alternative_name",
        "ud_name",
        "leipzig_shortcut",
        "ud_shortcut",
        "til_shortcut",
        "wals_shortcut",
        "iso_code",
        "primary_script",
        "primary_script_pct",
        "secondary_script",
        "secondary_script_pct",
        "distinct_words",
        "total_frequency",
        "family",
        "genus",
        "macroarea",
        "latitude",
        "longitude",
        "countries",
        "data_sources",
    ]

    rows: list[dict[str, str]] = []
    for lang in languages:
        leipzig_name = leipzig.get(lang, "")
        fam = families.get(lang, {})
        wals_name = fam.get("wals_name", "")
        alt_name = wals_name if wals_name and wals_name != leipzig_name else ""
        ud_name = ud_map.get(lang, "")
        # UD shortcut: the UD treebank language name (same as ud_name if overridden,
        # otherwise try Leipzig name since UD normally uses the same name)
        ud_shortcut = ud_name if ud_name else ""
        til_shortcut = _ISO3_TO_TIL.get(lang, "")
        wals_shortcut = lang if lang in families else ""
        sc = scripts.get(lang, {})
        st = stats.get(lang, {})
        src_set = raw_sources.get(lang, set())

        rows.append({
            "used_shortcut": lang,
            "name": leipzig_name,
            "alternative_name": alt_name,
            "ud_name": ud_name,
            "leipzig_shortcut": lang if lang in leipzig else "",
            "ud_shortcut": ud_shortcut,
            "til_shortcut": til_shortcut,
            "wals_shortcut": wals_shortcut,
            "iso_code": lang,
            "primary_script": sc.get("primary_script", ""),
            "primary_script_pct": sc.get("primary_script_pct", ""),
            "secondary_script": sc.get("secondary_script", ""),
            "secondary_script_pct": sc.get("secondary_script_pct", ""),
            "distinct_words": st.get("distinct_words", ""),
            "total_frequency": st.get("total_frequency", ""),
            "family": fam.get("family", ""),
            "genus": fam.get("genus", ""),
            "macroarea": fam.get("macroarea", ""),
            "latitude": fam.get("latitude", ""),
            "longitude": fam.get("longitude", ""),
            "countries": fam.get("countries", ""),
            "data_sources": ", ".join(sorted(src_set)),
        })

    # Write output
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} languages to {OUTPUT_CSV}")

    # Summary stats
    n_leipzig = sum(1 for r in rows if r["leipzig_shortcut"])
    n_ud = sum(1 for r in rows if r["ud_shortcut"])
    n_til = sum(1 for r in rows if r["til_shortcut"])
    n_wals = sum(1 for r in rows if r["wals_shortcut"])
    n_family = sum(1 for r in rows if r["family"])
    print(f"  Leipzig match: {n_leipzig}, UD match: {n_ud}, "
          f"TIL match: {n_til}, WALS match: {n_wals}, Family info: {n_family}")


if __name__ == "__main__":
    main()
