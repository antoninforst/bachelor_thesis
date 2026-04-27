# Data preprocessing

## 1. Download

- The scripts for downloading the files are located [here](0_download/).
- Since this is meant to be done once, there is no make file.
- Created files should land in [here](../../data/0_raw/).
  - All files must be prefixed with the three-letter ISO code of the language.

## Make actions

Run from [src/1_process](.) with `make ACTION`.

| Action | Meaning |
| --- | --- |
| `all` | Run raw-to-annotated preprocessing. |
| `filter` | Run hapax filtering and script filtering. |
| `filter_hapax` | Create overlap report and hapax-only ignore list. |
| `filter_script` | Run script check and add script reasons to ignore list. |
| `aggregate` | Create cleaned aggregated language files. |
| `statistics` | Compute coverage statistics. |
| `truncate` | Create annotated/truncated files. |
| `clear` | Remove aggregated and annotated files. |
| `clear_aggregated` | Remove aggregated files only. |
| `clear_annotated` | Remove annotated files only. |

Useful variables: `LANGS="ces eng"`, `COVERAGE=94`, `PYTHON=python`.

## 2. File filtering

This step finds raw files that should not be used later. The final result is [ignored_files.csv](../../results/1_process/1_filter/ignored_files.csv).

The filtering pipeline has four commands:

```bash
python src/1_process/1_filter/hapax_overlap.py
python src/1_process/1_filter/generate_ignore_list.py
python src/1_process/1_filter/script_check.py --ignore-csv results/1_process/1_filter/ignored_files.csv
python src/1_process/1_filter/generate_ignore_list.py --script-csv results/1_process/1_filter/script_check.csv
```

The first `generate_ignore_list.py` command creates a hapax-only ignore list. The last command rewrites the same file and adds script-check reasons.

### 2.1 Overlapping files

[hapax_overlap.py](1_filter/hapax_overlap.py) compares raw files inside each language. It extracts hapax legomena (`frequency == 1`) from every raw CSV file and measures how much two files overlap.

This does not ignore anything by itself. It only creates the overlap report.

#### How to run it

```bash
python src/1_process/1_filter/hapax_overlap.py
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `LANG ...` | all languages | Optional three-letter language codes to process, for example `ces eng`. |
| `--raw-dir` | `data/0_raw` | Directory with raw CSV files. |
| `--out-dir` | `results/1_process/1_filter` | Directory where `hapax_overlap.csv` is written. |
| `--max-pairs` | `16` | Maximum number of highest-overlap file pairs kept per language. |

#### Output

Output file: [hapax_overlap.csv](../../results/1_process/1_filter/hapax_overlap.csv)

Columns:

| Column | Meaning |
| --- | --- |
| `lang` | Three-letter language code. |
| `file_a`, `file_b` | Compared raw files. |
| `hapaxes_a`, `hapaxes_b` | Number of hapax words in each file. |
| `overlap` | Number of shared hapax words. |
| `share_a_pct`, `share_b_pct` | Overlap as a percentage of each file's hapaxes. |

#### Create ignore file from overlap

[generate_ignore_list.py](1_filter/generate_ignore_list.py) reads `hapax_overlap.csv` and creates the first [ignored_files.csv](../../results/1_process/1_filter/ignored_files.csv). This first ignore list uses only overlap information.

```bash
python src/1_process/1_filter/generate_ignore_list.py
```

For pairs above `33.3%`, the script ignores the file with the larger overlap share. If a file is already ignored, later pairs involving that file are skipped.

Options used in this step:

| Option | Default | Meaning |
| --- | --- | --- |
| `--overlap-csv` | `results/1_process/1_filter/hapax_overlap.csv` | Input file created by `hapax_overlap.py`. |
| `--out-dir` | `results/1_process/1_filter` | Directory where `ignored_files.csv` is written. |
| `--threshold` | `33.3` | Hapax-overlap percentage above which one file from the pair is ignored. |

At this point, `ignored_files.csv` contains only overlap-based ignores. Script-check results are not added unless `--script-csv` is passed later.

### 2.2 Script check

[script_check.py](1_filter/script_check.py) checks whether each raw file uses the expected writing script for its language. It uses Unicode script properties through the `regex` package and shows progress with `tqdm`.

The script writes one column per observed script. Letters from scripts that are not listed in the script checker are counted as `Other`. If many sampled letters are `Other`, the script prints a warning so the script list can be extended if needed.

This does not update `ignored_files.csv` by itself. It creates `script_check.csv`; the final ignore-list command uses that file.

#### How to run it

Prerequisite: run the overlap ignore-list step first, so that script check can skip files already ignored for overlap.

```bash
python src/1_process/1_filter/script_check.py \
  --ignore-csv results/1_process/1_filter/ignored_files.csv
```

Examples:

```bash
python src/1_process/1_filter/script_check.py ces eng \
  --ignore-csv results/1_process/1_filter/ignored_files.csv

python src/1_process/1_filter/script_check.py \
  --raw-dir data/0_raw \
  --out-dir results/1_process/1_filter \
  --ignore-csv results/1_process/1_filter/ignored_files.csv
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `LANG ...` | all languages | Optional three-letter language codes to process, for example `ces eng`. |
| `--raw-dir` | `data/0_raw` | Directory with raw CSV files. |
| `--out-dir` | `results/1_process/1_filter` | Directory where `script_check.csv` is written. |
| `--ignore-csv` | not used | Path to `ignored_files.csv`; files listed there are skipped. |
| `--ignore` | same as `--ignore-csv` | Shorter alias kept for compatibility. |

Internal warning settings:

| Setting | Value | Meaning |
| --- | --- | --- |
| `DEFAULT_THRESHOLD` | `95.0` | Used only for console warnings about files that are not mostly one script. It does not decide the final ignore list. |
| `OTHER_THRESHOLD` | `20.0` | Prints a warning when many sampled letters fall into `Other`. |
| `WORD_STEP` | `5` | Samples every fifth word from each raw file. |
| `LETTER_STEP` | `3` | Samples every third letter from each sampled word. |
| `MAX_SAMPLED_WORDS` | `10_000` | Stops reading a file after enough sampled words for script detection. |

#### Output

Output file: [script_check.csv](../../results/1_process/1_filter/script_check.csv)

Important columns:

| Column | Meaning |
| --- | --- |
| `file` | Raw file name. |
| `language` | Three-letter language code. |
| `language_script` | Main script detected for the whole language. |
| `file_script` | Main script detected for this file. |
| `language_script_share` | Percentage of this file that matches the language's main script. |
| `file_script_share` | Percentage of this file that matches its own main script. |
| `second_language_script_share` | Percentage of this file that matches the language's second most common script. |
| script columns, for example `Latin`, `Cyrillic`, `Other` | Percentage of sampled letters in that script. |

#### Add script results to ignore file

Run [generate_ignore_list.py](1_filter/generate_ignore_list.py) again, this time with `--script-csv`. This rewrites [ignored_files.csv](../../results/1_process/1_filter/ignored_files.csv) with both overlap and script-based reasons.

```bash
python src/1_process/1_filter/generate_ignore_list.py \
  --script-csv results/1_process/1_filter/script_check.csv
```

A file is added from script check when `language_script_share` is below `75%`.

Options used in this step:

| Option | Default | Meaning |
| --- | --- | --- |
| `--overlap-csv` | `results/1_process/1_filter/hapax_overlap.csv` | Input file created by `hapax_overlap.py`. Used again so overlap ignores are preserved. |
| `--out-dir` | `results/1_process/1_filter` | Directory where `ignored_files.csv` is written. |
| `--threshold` | `33.3` | Hapax-overlap cutoff. Keep the same value as in the first ignore-list run unless intentionally changing it. |
| `--script-csv` | not used | Path to `script_check.csv`. Required to add script-based ignores. |
| `--script-threshold` | `75.0` | Minimum required share of the language's main script. |

## 3. Aggregate

This step creates one cleaned frequency list per language. The main result is one CSV file per language in [data/1_aggregated](../../data/1_aggregated/).

The aggregation pipeline has two commands:

```bash
python src/1_process/2_aggregate/aggregate.py --ignore results/1_process/1_filter/ignored_files.csv
python src/1_process/2_aggregate/language_overview.py
```

The first command reads raw files, skips ignored files, cleans word types, and aggregates frequencies. The second command creates a language-level overview from the aggregated data and metadata.

### 3.1 Word-type filtering

[aggregate.py](2_aggregate/aggregate.py) cleans word types before merging files. It strips selected punctuation from word boundaries, lowercases words, removes entries without letters, and merges duplicates created by those changes.

This does not create a separate filtered-word report by itself. The filtering is part of aggregation.

#### How to run it

```bash
python src/1_process/2_aggregate/aggregate.py \
  --ignore results/1_process/1_filter/ignored_files.csv
```

Examples:

```bash
python src/1_process/2_aggregate/aggregate.py ces eng \
  --ignore results/1_process/1_filter/ignored_files.csv

python src/1_process/2_aggregate/aggregate.py \
  --raw-dir data/0_raw \
  --out-dir data/1_aggregated \
  --ignore results/1_process/1_filter/ignored_files.csv
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `LANG ...` | all languages | Optional three-letter language codes to process, for example `ces eng`. |
| `--raw-dir` | `data/0_raw` | Directory with raw CSV files. |
| `--out-dir` | `data/1_aggregated` | Directory where aggregated language files are written. |
| `--ignore` | not used | Path to `ignored_files.csv`; files listed there are skipped. |
| `--repair` | off | Re-cleans files already in `--out-dir` instead of reading raw files. |

Internal cleaning settings:

| Setting | Value | Meaning |
| --- | --- | --- |
| `STRIP_CHARS` | punctuation list in the script | Characters stripped from the start and end of words. |
| `_HAS_LETTER` | Unicode-letter regex | Keeps only entries that contain at least one letter. |

#### Output

Output folder: [data/1_aggregated](../../data/1_aggregated/)

Each language file has two columns:

| Column | Meaning |
| --- | --- |
| `word` | Cleaned word type. |
| `frequency` | Frequency summed across all non-ignored raw files for the language. |

The script also writes [aggregation_report.csv](../../results/1_process/aggregation_report.csv).

Important report columns:

| Column | Meaning |
| --- | --- |
| `lng_shortcut` | Three-letter language code. |
| `file_name` | Raw file included in aggregation. |
| `total_tokens` | Token count before cleaning. |
| `total_tokens_after_ignoring` | Token count after cleaning. |
| `total_types(rows)` | Number of remaining word types in that file. |
| `ignored_word_count` | Number of removed word types. |
| `avg_deleted_punc_per_token` | Average number of stripped boundary punctuation characters per token. |
| `new_types_from_previous_file` | Number of cleaned types not seen in earlier files for the same language. |

### 3.2 Aggregation

[language_overview.py](2_aggregate/language_overview.py) creates one metadata row per aggregated language. It combines language names, script information, aggregated statistics, family information, coordinates, countries, and data-source labels.

Prerequisites: run aggregation first. For complete columns, also run script check and statistics before creating the overview.

#### How to run it

```bash
python src/1_process/2_aggregate/language_overview.py
```

Options:

This script has no command-line options. Paths are constants inside the file.

Internal paths:

| Setting | Value | Meaning |
| --- | --- | --- |
| `AGGREGATED_DIR` | `data/1_aggregated` | Languages included in the overview. |
| `RAW_DIR` | `data/0_raw` | Used to list data-source labels. |
| `STATISTICS_CSV` | `results/1_process/3_truncate/statistics.csv` | Adds distinct-word and total-frequency columns. |
| `SCRIPT_CHECK_CSV` | `results/1_process/1_filter/script_check.csv` | Adds primary and secondary script columns. |
| `OUTPUT_CSV` | `results/1_process/2_aggregate/language_overview.csv` | Output file. |

#### Output

Output file: [language_overview.csv](../../results/1_process/2_aggregate/language_overview.csv)

Important columns:

| Column | Meaning |
| --- | --- |
| `used_shortcut` | Three-letter language code used by this project. |
| `name` | Main language name. |
| `alternative_name` | Alternative name when available. |
| `ud_name` | Universal Dependencies language name when mapped. |
| `leipzig_shortcut`, `ud_shortcut`, `til_shortcut`, `wals_shortcut` | Matching identifiers in external sources. |
| `iso_code` | ISO code, normally the same as `used_shortcut`. |
| `primary_script`, `primary_script_pct` | Main script and its average share. |
| `secondary_script`, `secondary_script_pct` | Second script when it is large enough to report. |
| `distinct_words`, `total_frequency` | Statistics from aggregated files. |
| `family`, `genus`, `macroarea` | Language-family metadata. |
| `latitude`, `longitude`, `countries` | Geographic metadata. |
| `data_sources` | Raw source labels found for the language. |

## 4. Truncation

This step computes vocabulary coverage statistics and then creates shorter word-frequency lists. The final result is one truncated CSV per language in [data/2_annotated](../../data/2_annotated/).

The truncation pipeline has two commands:

```bash
python src/1_process/3_truncate/statistics.py
python src/1_process/3_truncate/truncate.py --coverage 94
```

The first command decides how many word types are needed for each coverage level. The second command uses those counts to copy only the selected top part of each aggregated file.

### 4.1 Truncation methods

[statistics.py](3_truncate/statistics.py) computes coverage, frequency, and rank statistics for every aggregated language file.

This does not truncate anything by itself. It creates the statistics table used by `truncate.py`.

#### How to run it

```bash
python src/1_process/3_truncate/statistics.py
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--agg-dir` | `data/1_aggregated` | Directory with aggregated language CSV files. |
| `--output` | `results/1_process/3_truncate/statistics.csv` | Output statistics file. |

Internal coverage settings:

| Setting | Value | Meaning |
| --- | --- | --- |
| `COVERAGE_LEVELS` | `50` to `100` with smaller steps near high coverage | Coverage percentages for which type counts and token counts are computed. |

#### Output

Output file: [statistics.csv](../../results/1_process/3_truncate/statistics.csv)

Important columns:

| Column | Meaning |
| --- | --- |
| `file` | Language code. |
| `distinct_words` | Number of word types in the aggregated file. |
| `total_frequency` | Total token frequency in the aggregated file. |
| `cov_<N>_types` | Number of top word types needed to reach `N%` token coverage. |
| `cov_<N>_tokens` | Number of tokens covered at `N%` coverage. |
| `freq100_types`, `freq100_tokens` | Types and tokens with frequency at least `100`. |
| `freq10_types`, `freq10_tokens` | Types and tokens with frequency at least `10`. |
| `rank5pct_types`, `rank5pct_tokens` | Top `5%` of word types and their tokens. |
| `rank30pct_types`, `rank30pct_tokens` | Top `30%` of word types and their tokens. |

### 4.2 Truncate

[truncate.py](3_truncate/truncate.py) reads [statistics.csv](../../results/1_process/3_truncate/statistics.csv), chooses the number of word types needed for the requested coverage, and writes truncated CSV files.

If several words have the same frequency at the cutoff boundary, the script samples within that tied frequency band. Use `--seed` when you need the tie choice to be repeatable.

#### How to run it

```bash
python src/1_process/3_truncate/truncate.py --coverage 94
```

Examples:

```bash
python src/1_process/3_truncate/truncate.py ces eng --coverage 94

python src/1_process/3_truncate/truncate.py n=100 --coverage 94 --seed 1

python src/1_process/3_truncate/truncate.py \
  --src data/1_aggregated \
  --out data/2_annotated \
  --stats results/1_process/3_truncate/statistics.csv \
  --coverage 94
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `LANG ...` | all languages | Optional three-letter language codes to truncate, for example `ces eng`. |
| `n=N` | not used | Positional shortcut for truncating only the top `N` languages by total frequency. |
| `--src` | `data/1_aggregated` | Directory with aggregated CSV files. |
| `--out` | `data/2_annotated` | Directory where truncated CSV files are written. |
| `--stats` | `results/1_process/3_truncate/statistics.csv` | Statistics file created by `statistics.py`. |
| `--coverage` | `94` | Coverage percentage to use, matching one of the `cov_<N>_types` columns. |
| `--top-n`, `-n` | not used | Truncate only the top `N` languages by total frequency. |
| `--seed` | not used | Random seed for tie-breaking at the cutoff frequency. |

#### Output

Output folder: [data/2_annotated](../../data/2_annotated/)

Each truncated file has the same two columns as the aggregated files:

| Column | Meaning |
| --- | --- |
| `word` | Word type kept after truncation. |
| `frequency` | Frequency from the aggregated file. |

The number of rows in each output file is taken from the selected coverage column in [statistics.csv](../../results/1_process/3_truncate/statistics.csv).

