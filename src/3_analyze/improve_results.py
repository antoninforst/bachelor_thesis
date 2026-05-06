from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RESULTS_PATH = Path("results/3_analyze/results.csv")
LANGUAGE_OVERVIEW_PATH = Path("results/1_process/2_aggregate/language_overview.csv")
COMPRESS_PATH = Path("results/3_analyze/compress100k.csv")
OUTPUT_PATH = Path("results/3_analyze/results_improved.csv")

METRIC_RENAMES = {
	"original_distinct_words": "original_type_count",
	"original_total_frequency": "original_token_count",
	"word_count": "type_count",
	"word_total_frequency": "token_count",
	"word_ttr": "type_token_ratio",
	"word_hapax_count": "hapax_type_count",
	"word_hapax_ratio": "hapax_type_ratio",
	"word_freq_hapax_count": "single_token_type_count",
	"word_freq_hapax_ratio": "single_token_type_ratio",
	"word_avg_length": "avg_type_length",
	"word_avg_length_weighted": "avg_token_length",
	"word_frequency_entropy": "token_frequency_entropy",
	"word_frequency_perplexity": "token_frequency_perplexity",
}


def first_non_empty(row: pd.Series, columns: list[str], fallback: str) -> str:
	for column in columns:
		value = row.get(column)
		if pd.notna(value) and str(value).strip():
			return str(value).strip()
	return fallback


def load_language_overview(path: Path) -> pd.DataFrame:
	overview = pd.read_csv(path, keep_default_na=False, na_values=[""])
	overview = overview.drop_duplicates(subset="used_shortcut", keep="first")
	overview["language_name"] = overview.apply(
		lambda row: first_non_empty(
			row,
			["name", "alternative_name", "ud_name"],
			str(row["used_shortcut"]),
		),
		axis=1,
	)
	overview = overview.rename(
		columns={
			"used_shortcut": "language_code",
			"distinct_words": "original_distinct_words",
			"total_frequency": "original_total_frequency",
			"primary_script": "dominant_script",
			"primary_script_pct": "dominant_script_pct",
		}
	)
	return overview[
		[
			"language_code",
			"language_name",
			"dominant_script",
			"dominant_script_pct",
			"secondary_script",
			"secondary_script_pct",
			"original_distinct_words",
			"original_total_frequency",
		]
	]


def improve_results(results_path: Path, overview_path: Path, compress_path: Path = COMPRESS_PATH) -> pd.DataFrame:
	results = pd.read_csv(results_path, keep_default_na=False, na_values=[""])
	if "language" not in results.columns:
		raise ValueError(f"Missing 'language' column in {results_path}")

	results = results.rename(columns={"language": "language_code"})
	overview = load_language_overview(overview_path)

	# The overview uses short codes (e.g. "abk") while results use full codes (e.g. "abk_Cyrl").
	# Derive 'lang' (first 3 chars) and 'script' (last 4 chars) for joining.
	if "lang" not in results.columns:
		results["lang"] = results["language_code"].str[:3]
	if "script" not in results.columns:
		results["script"] = results["language_code"].str[-4:]

	merged = results.merge(
		overview,
		left_on="lang",
		right_on="language_code",
		how="left",
		suffixes=("", "_overview"),
	)
	# Drop the overview's language_code (short form), keep ours (full form)
	if "language_code_overview" in merged.columns:
		merged = merged.drop(columns=["language_code_overview"])

	missing = merged[merged["language_name"].isna()]["language_code"].tolist()
	if missing:
		print(f"WARNING: {len(missing)} languages missing from overview: {', '.join(missing)}")
		merged["language_name"] = merged["language_name"].fillna(merged["language_code"])

	merged = merged.rename(columns={"language_name": "language"})
	morphology_columns = [
		column for column in merged.columns
		if column.startswith("morph_")
		or column in {
			"morph_count",
			"morph_total_frequency",
			"avg_root_count",
			"avg_prefix_count",
			"avg_suffix_count",
			"avg_interfix_count",
			"avg_root_count_weighted",
			"avg_prefix_count_weighted",
			"avg_suffix_count_weighted",
			"avg_interfix_count_weighted",
		}
	]
	merged = merged.drop(columns=morphology_columns)
	merged = merged.rename(columns=METRIC_RENAMES)

	if compress_path.exists():
		compress = pd.read_csv(compress_path)
		compress = compress.rename(columns={
			"language": "language_code",
			"ratio": "compress_ratio",
			"reduction_pct": "compress_reduction_pct",
		})
		compress = compress[["language_code", "unique_chars", "bin_size", "zip_size", "rand_zip_size", "compress_ratio", "compress_reduction_pct"]]
		merged = merged.merge(compress, on="language_code", how="left")

	if "type_count" in merged.columns and "original_type_count" in merged.columns:
		merged["cutoff_type_pct"] = merged["type_count"] / merged["original_type_count"] * 100

	first_columns = [
		"language_code",
		"language",
		"dominant_script",
		"dominant_script_pct",
		"secondary_script",
		"secondary_script_pct",
		"original_type_count",
		"original_token_count",
	]
	remaining_columns = [column for column in merged.columns if column not in first_columns]
	return merged[first_columns + remaining_columns]


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--results", type=Path, default=RESULTS_PATH)
	parser.add_argument("--overview", type=Path, default=LANGUAGE_OVERVIEW_PATH)
	parser.add_argument("--compress", type=Path, default=COMPRESS_PATH)
	parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
	args = parser.parse_args()

	improved = improve_results(args.results, args.overview, args.compress)
	args.out.parent.mkdir(parents=True, exist_ok=True)
	improved.to_csv(args.out, index=False)
	print(f"saved={args.out}")


if __name__ == "__main__":
	main()
