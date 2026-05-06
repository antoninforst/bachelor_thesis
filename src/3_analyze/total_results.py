"""
Merge all analysis results into a single total_results.csv file.

Sources:
  - results/3_analyze/results.csv                       (corpus-level metrics from analyze.py)
  - results/3_analyze/morphs.csv                        (morph-level metrics from morphs.py)
  - results/3_analyze/compress100k.csv                   (compression metrics)
  - results/1_process/2_aggregate/language_overview.csv  (token counts, dominant script)
  - metadata/languages.csv                               (name, family, genus, typology)
  - data/5_other/script_types.csv                        (script type classification)
  - data/2_annotated/*.csv                               (last type frequency per language)

Output:
  - results/3_analyze/total_results.csv
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- Source paths ---
RESULTS_PATH = ROOT / "results" / "3_analyze" / "results.csv"
MORPHS_PATH = ROOT / "results" / "3_analyze" / "morphs.csv"
COMPRESS_PATH = ROOT / "results" / "3_analyze" / "compress100k.csv"
OVERVIEW_PATH = ROOT / "results" / "1_process" / "2_aggregate" / "language_overview.csv"
LANG_META_PATH = ROOT / "metadata" / "languages.csv"
SCRIPT_TYPES_PATH = ROOT / "data" / "5_other" / "script_types.csv"
ANNOTATED_DIR = ROOT / "data" / "2_annotated"

OUTPUT_PATH = ROOT / "results" / "3_analyze" / "total_results.csv"

# Column renames applied to results.csv (corpus-level metrics)
CORPUS_METRIC_RENAMES = {
    "word_count": "type_count",
    "word_total_frequency": "token_count",
    "word_ttr": "type_token_ratio",
    "word_zipf_slope": "corpus_zipf_slope",
    "word_hapax_count": "hapax_type_count",
    "word_hapax_ratio": "hapax_type_ratio",
    "word_freq_hapax_count": "single_token_type_count",
    "word_freq_hapax_ratio": "single_token_type_ratio",
    "word_avg_length": "avg_type_length",
    "word_avg_length_weighted": "avg_token_length",
    "word_frequency_entropy": "token_frequency_entropy",
    "word_frequency_perplexity": "token_frequency_perplexity",
}

# Final output column order
OUT_COLS = [
    # Identifiers
    "lang", "script", "script_type", "segmentor", "language_name", "family", "genus", "typology",
    # Corpus-level counts
    "dominant_script_pct",
    "original_token_count", "original_type_count",
    "current_token_count", "current_type_count",
    "last_type_frequency",
    # Corpus metrics
    "type_token_ratio", "corpus_zipf_slope",
    "compress_ratio", "compress_reduction_pct",
    "token_frequency_entropy", "token_frequency_perplexity",
    "avg_type_length", "avg_token_length",
    "hapax_type_ratio", "single_token_type_ratio",
    # Word metrics (from segmented data)
    "word_count", "word_total_frequency",
    "word_ttr", "word_entropy", "word_perplexity",
    "word_zipf_slope", "morph_zipf_slope",
    "word_hapax_ratio", "morph_hapax_ratio",
    "word_avg_length", "word_avg_length_weighted",
    # Morph metrics
    "morph_count", "morph_total_frequency",
    "morph_ttr", "morph_entropy", "morph_perplexity",
    "morph_type_entropy", "morph_type_perplexity",
    "morph_avg_length", "morph_avg_length_weighted",
    # Morphological composition (unweighted)
    "avg_morphs_per_word", "avg_roots_per_word",
    "avg_affixes_per_word", "avg_prefixes_per_word", "avg_suffixes_per_word",
    "compounding_index", "affix_deviation",
    # Morphological composition (weighted)
    "avg_morphs_per_word_weighted", "avg_roots_per_word_weighted",
    "avg_affixes_per_word_weighted", "avg_prefixes_per_word_weighted",
    "avg_suffixes_per_word_weighted",
    "compounding_index_weighted", "affix_deviation_weighted",
    # Morph-type entropies & counts
    "root_entropy", "affix_entropy", "prefix_entropy", "suffix_entropy",
    "root_type_entropy", "affix_type_entropy", "prefix_type_entropy", "suffix_type_entropy",
    "root_count", "affix_count", "prefix_count", "suffix_count",
]


def load_corpus_results() -> pd.DataFrame:
    """Load results.csv and rename corpus-level metrics."""
    res = pd.read_csv(RESULTS_PATH, keep_default_na=False, na_values=[""])
    res = res.rename(columns={"language": "language_code"})

    # Drop morph columns from analyze.py (they come from morphs.csv instead)
    morph_cols = [c for c in res.columns if c.startswith("morph_")
                  or c.startswith("avg_root_") or c.startswith("avg_prefix_")
                  or c.startswith("avg_suffix_") or c.startswith("avg_interfix_")]
    res = res.drop(columns=morph_cols)

    res = res.rename(columns=CORPUS_METRIC_RENAMES)
    return res


def load_overview() -> pd.DataFrame:
    """Load language_overview.csv for original token/type counts and dominant script."""
    overview = pd.read_csv(OVERVIEW_PATH, keep_default_na=False, na_values=[""])
    overview = overview.drop_duplicates(subset="used_shortcut", keep="first")
    overview = overview.rename(columns={
        "used_shortcut": "language_code",
        "distinct_words": "original_type_count",
        "total_frequency": "original_token_count",
        "primary_script": "dominant_script",
        "primary_script_pct": "dominant_script_pct",
    })
    return overview[["language_code", "dominant_script", "dominant_script_pct",
                      "original_type_count", "original_token_count"]]


def load_morphs() -> pd.DataFrame:
    """Load morphs.csv and extract segmentor and lang_script."""
    morphs = pd.read_csv(MORPHS_PATH)
    morphs["segmentor"] = morphs["file"].str.split("-").str[0].str.split("_", n=1).str[1]
    morphs["lang_script"] = morphs["file"].str.split("-").str[1]
    return morphs.drop(columns=["lang", "file"])


def load_compress() -> pd.DataFrame:
    """Load compress100k.csv if it exists."""
    if not COMPRESS_PATH.exists():
        return pd.DataFrame()
    compress = pd.read_csv(COMPRESS_PATH)
    compress = compress.rename(columns={
        "language": "language_code",
        "ratio": "compress_ratio",
        "reduction_pct": "compress_reduction_pct",
    })
    return compress[["language_code", "compress_ratio", "compress_reduction_pct"]]


def load_lang_metadata() -> pd.DataFrame:
    """Load metadata/languages.csv for name, family, genus, typology."""
    meta = pd.read_csv(LANG_META_PATH, keep_default_na=False, na_values=[""])
    return meta.drop_duplicates(subset=["iso_code"], keep="first")


def compute_last_type_frequency(language_codes: list[str]) -> dict[str, int]:
    """For each language, find the minimum frequency in its annotated file."""
    last_freq = {}
    for lang_script in language_codes:
        fpath = ANNOTATED_DIR / f"{lang_script}.csv"
        if fpath.exists():
            df_ann = pd.read_csv(fpath)
            last_freq[lang_script] = int(df_ann["frequency"].min())
    return last_freq


def main():
    # --- Load all sources ---
    corpus = load_corpus_results()
    overview = load_overview()
    morphs = load_morphs()
    compress = load_compress()
    lang_meta = load_lang_metadata()
    script_types = pd.read_csv(SCRIPT_TYPES_PATH)

    # --- Merge corpus results with overview (original counts + script info) ---
    # Extract lang and script from language_code first (overview uses short codes)
    corpus["lang"] = corpus["language_code"].str.split("_").str[0]
    corpus["script"] = corpus["language_code"].str.split("_").str[1]

    merged = corpus.merge(overview, left_on="lang", right_on="language_code",
                          how="left", suffixes=("", "_overview"))
    if "language_code_overview" in merged.columns:
        merged = merged.drop(columns=["language_code_overview"])

    # Rename type_count/token_count to current_* (post-truncation counts)
    merged = merged.rename(columns={
        "type_count": "current_type_count",
        "token_count": "current_token_count",
    })

    # --- Merge compression data ---
    if not compress.empty:
        merged = merged.merge(compress, on="language_code", how="left")

    # --- Merge morphs (segmented subset, left join) ---
    merged = merged.merge(morphs, left_on="language_code", right_on="lang_script", how="left")

    # --- Merge language metadata (name, family, genus, typology) ---
    merged = merged.merge(
        lang_meta[["iso_code", "name", "family", "genus", "typology"]],
        left_on="lang", right_on="iso_code", how="left",
    )
    merged = merged.rename(columns={"name": "language_name"})

    # --- Merge script types ---
    merged = merged.merge(script_types, on="script", how="left")
    merged = merged.rename(columns={"type": "script_type"})

    # --- Last type frequency ---
    last_freq = compute_last_type_frequency(merged["language_code"].unique().tolist())
    merged["last_type_frequency"] = merged["language_code"].map(last_freq)

    # --- Select and order columns ---
    out = merged[OUT_COLS]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(out)} rows x {len(out.columns)} columns -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
