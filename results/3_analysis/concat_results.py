import pandas as pd
from pathlib import Path

results_dir = Path(__file__).parent
shortcuts_path = results_dir / ".." / ".." / "src" / "0_data_processing" / "leipzig" / "lepzig_shortcuts.csv"
families_path = results_dir / ".." / "5_other" / "lang_families.csv"

# load language code <-> name mappings
shortcuts = pd.read_csv(shortcuts_path)
lang_to_code = dict(zip(shortcuts["language"], shortcuts["code"]))
code_to_lang = dict(zip(shortcuts["code"], shortcuts["language"]))

files = {
    "segment_true": ("result_segment_true.csv", "lang"),
    "segment_universal": ("result_segment_universal.csv", "lang"),
    "root_true": ("result_root_true.csv", "lang"),
    "root_universal": ("result_root_universal.csv", "lang"),
    "results": ("results.csv", "language"),
    "statistics": ("statistics.csv", "file"),
}

dfs = {}
for name, (filename, lang_col) in files.items():
    path = results_dir / filename
    if not path.exists():
        print(f"Skipping {filename} (not found)")
        continue
    df = pd.read_csv(path)
    df = df.rename(columns={lang_col: "lang"})
    # map full language names to codes (for results.csv)
    if filename == "results.csv":
        df["lang"] = df["lang"].map(lang_to_code)
        unmapped = df["lang"].isna().sum()
        if unmapped:
            print(f"  Warning: {unmapped} unmapped languages in {filename}")
        df = df.dropna(subset=["lang"])
    # add suffix to non-key columns to avoid collisions
    df = df.rename(columns={c: f"{c}_{name}" for c in df.columns if c != "lang"})
    dfs[name] = df
    print(f"Loaded {filename}: {len(df)} rows")

merged = None
for name, df in dfs.items():
    if merged is None:
        merged = df
    else:
        merged = merged.merge(df, on="lang", how="outer")

# add full language name and reorder so lang + language are first
merged["language"] = merged["lang"].map(code_to_lang)

# merge with language families (deduplicate by iso_code, keep first)
families = pd.read_csv(families_path)
families = families.drop_duplicates(subset="iso_code", keep="first")
families = families.rename(columns={"iso_code": "lang"})
family_cols = ["family", "genus", "macroarea", "lat", "lon", "countries"]
merged = merged.merge(families[["lang"] + family_cols], on="lang", how="left")

# report languages without family info
no_family = merged[merged["family"].isna()]["lang"].tolist()
if no_family:
    print(f"\nLanguages without family info ({len(no_family)}):")
    for code in no_family:
        name = code_to_lang.get(code, code)
        print(f"  {code} ({name})")

cols = ["lang", "language"] + family_cols + [c for c in merged.columns if c not in ["lang", "language"] + family_cols]
merged = merged[cols]

print(f"\nMerged: {len(merged)} rows, {len(merged.columns)} columns")

out_path = results_dir / "all_results.csv"
merged.to_csv(out_path, index=False)
#merged.to_csv(out_path, index=False, sep=";", decimal=",")
print(f"Saved to {out_path}")
