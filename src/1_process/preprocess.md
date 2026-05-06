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
| `aggregate_continue` | Continue aggregation by skipping existing files and not writing the report. |
| `parse_words` | Segment Chinese and Thai words in aggregated files. |
| `statistics` | Compute coverage statistics. |
| `truncate` | Create annotated/truncated files with PPM column. |
| `clear` | Remove aggregated and annotated files. |
| `clear_aggregated` | Remove aggregated files only. |
| `clear_annotated` | Remove annotated files only. |

Useful variables: `LANGS="ces eng"`, `COVERAGE=94`, `TOP_N=100`, `WORKERS=4`, `PYTHON=python`.

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
python src/1_process/2_aggregate/parse_words.py
```

The first command reads raw files, skips ignored files, cleans word types, aggregates frequencies, and appends a script suffix to the output filename (e.g. `eng_Latn.csv`). The second command segments compound words for Chinese and Thai languages in-place.

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

python src/1_process/2_aggregate/aggregate.py \
  --ignore results/1_process/1_filter/ignored_files.csv \
  --skip-existing \
  --no-report

python src/1_process/2_aggregate/aggregate.py ces eng \
  --ignore results/1_process/1_filter/ignored_files.csv \
  --no-report \
  --jobs 2
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `LANG ...` | all languages | Optional three-letter language codes to process, for example `ces eng`. |
| `--raw-dir` | `data/0_raw` | Directory with raw CSV files. |
| `--out-dir` | `data/1_aggregated` | Directory where aggregated language files are written. |
| `--ignore` | not used | Path to `ignored_files.csv`; files listed there are skipped. |
| `--repair` | off | Re-cleans files already in `--out-dir` instead of reading raw files. |
| `--skip-existing` | off | Skip languages that already have an output file in `--out-dir`. |
| `--jobs` | `1` | Number of languages to process in parallel. Use `0` for all available CPUs. |
| `--no-report` | off | Skip `aggregation_report.csv` and report-only metrics. |

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

The full aggregation run writes [aggregation_report.csv](../../results/1_process/aggregation_report.csv). Continue runs should use `--skip-existing --no-report`, because the report would contain only newly processed files and would overwrite the full-run report.

The make targets follow this rule:

```bash
make aggregate
make aggregate_continue
make aggregate WORKERS=4
make aggregate_continue WORKERS=4
make aggregate_continue WORKERS=8
```

`make aggregate` writes the report. `make aggregate_continue` skips existing output files and does not write the report. `WORKERS` is passed to `aggregate.py` as `--jobs`, so `WORKERS=0` uses all available CPUs and `WORKERS=1` keeps the run sequential.

`WORKERS` is a maximum. With several large languages, memory-aware scheduling may run fewer workers at once. The scheduler starts the largest missing language that fits the current memory estimate, then fills remaining worker and memory headroom with smaller languages. If no estimate fits and no worker is active, it starts the smallest missing language so the continue run still makes progress. The biggest memory costs are the per-language frequency dictionaries, the temporary cleaned dictionary for the current raw file, and the sorted output list created just before writing. The memory limit and scheduling multiplier are constants in [aggregate.py](2_aggregate/aggregate.py). The multiplier estimates RAM cost from raw input size: for example, a 500 MB raw input is treated as roughly 4 GB of RAM when the multiplier is `8`. This is conservative because Python dictionaries store each word as objects, hashes, pointers, and integer values, so they use much more memory than the compact CSV text on disk.

Use `--no-report` when the aggregation report is not needed. It skips report-only metric collection and does not write [aggregation_report.csv](../../results/1_process/aggregation_report.csv).

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

### 3.2 Word segmentation

[parse_words.py](2_aggregate/parse_words.py) segments compound words in aggregated files for languages that lack whitespace word boundaries. It reads each word-frequency pair, splits the word into sub-words using a language-specific tokenizer, assigns the original frequency to each sub-word, re-aggregates duplicates, re-applies the standard cleaning pipeline, and overwrites the aggregated file.

Supported language groups:

| Group | Languages | Tokenizer |
| --- | --- | --- |
| Chinese | `zho` `cmn` `yue` `wuu` `lzh` | jieba |
| Thai | `tha` | pythainlp (newmm engine) |

Romanized Chinese variants (`nan`, `hak`, `cdo`) are excluded because jieba destructively splits Latin-script text character by character.

Prerequisite: run aggregation first. The script modifies aggregated files in-place.

#### How to run it

```bash
python src/1_process/2_aggregate/parse_words.py
```

Examples:

```bash
python src/1_process/2_aggregate/parse_words.py zho tha

python src/1_process/2_aggregate/parse_words.py \
  --agg-dir data/1_aggregated
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `LANG ...` | all known segmentation languages | Optional three-letter language codes to process. |
| `--agg-dir` | `data/1_aggregated` | Directory with aggregated CSV files to modify in-place. |

#### Output

The script overwrites the existing aggregated CSV files. The format stays the same (two columns: `word`, `frequency`). The number of word types typically increases because compound entries are split into their components.

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
| `ppm` | Parts per million relative to the original corpus total frequency. |

The number of rows in each output file is taken from the selected coverage column in [statistics.csv](../../results/1_process/3_truncate/statistics.csv).

### 4.3 Quality check

[quality_check.py](3_truncate/quality_check.py) computes per-file quality metrics for truncated word-frequency files. It reuses `ScriptDetector` from the filtration step for script classification.

Each metric is an independent check function. Adding a new metric means writing one function and appending it to the `_CHECKS` list.

#### How to run it

```bash
python src/1_process/3_truncate/quality_check.py
python src/1_process/3_truncate/quality_check.py ces eng
python src/1_process/3_truncate/quality_check.py --input-dir data/1_aggregated
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `LANG ...` | all files | Optional language codes to limit processing. |
| `--input-dir` | `data/2_annotated` | Directory with word-frequency CSVs. |
| `--output` | `results/1_process/3_truncate/quality.csv` | Output CSV path. |

#### Output

Output file: [quality.csv](../../results/1_process/3_truncate/quality.csv)

| Column | Meaning |
| --- | --- |
| `file` | Input file name. |
| `language` | Three-letter language code. |
| `script` | ISO 15924 script code from the filename. |
| `foreign_script_pct` | % of word types in a different script than expected. |
| `long_outlier_pct` | % of word types longer than Q3 + 3·IQR. |
| `top20_long_pct` | % of the 20 most frequent words that are length outliers. |
| `punct_char_pct` | Share of punctuation characters (frequency-weighted). |
| `program_pct` | % of word types that look like web/programming artifacts. |
| `corrupted_pct` | % of word types flagged by any of the above. |
| `corrupted_freq_pct` | Token share of corrupted word types. |
| `eng_stopwords` | Number of English stopwords found (0 for English files). |

