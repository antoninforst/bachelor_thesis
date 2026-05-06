# Analysis

Computes frequency, morphological, and compression metrics for annotated and segmented word lists.

## Make actions

Run from [src/3_analyze](.) with `make ACTION`.

| Action | Meaning |
| --- | --- |
| `all` | Run the full analysis pipeline. |
| `analyze` | Compute corpus frequency metrics on annotated files. |
| `morphs` | Compute morphological metrics on segmented files. |
| `compress` | Compute compression metrics. |
| `results` | Merge all metrics into a single total results file. |
| `print` | Generate a LaTeX table from total results. |

## 1. Corpus metrics

[analyze.py](analyze.py) computes frequency-based metrics for each annotated language file. Each file produces one row of metrics.

### How to run it

```bash
python src/3_analyze/analyze.py
python src/3_analyze/analyze.py --folder data/2_annotated --out results/3_analyze/results.csv
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--folder` | `data/2_annotated` | Input directory with word-frequency CSVs. |
| `--pattern` | `*.csv` | Glob pattern for input files. |
| `--out` | `results/3_analyze/results.csv` | Output CSV path. |

### Output

Output file: [results.csv](../../results/3_analyze/results.csv)

| Column | Meaning |
| --- | --- |
| `num_rows` | Number of unique word types. |
| `total_frequency` | Total token count. |
| `hapax_count` | Words with frequency 1. |
| `hapax_ratio` | Hapax count / total types. |
| `freq_hapax_count` | First-hapax by frequency threshold. |
| `freq_hapax_ratio` | Frequency-based hapax ratio. |
| `avg_word_len` | Average word length (unweighted). |
| `avg_word_len_weighted` | Average word length weighted by frequency. |
| `frequency_entropy` | Shannon entropy of the frequency distribution. |
| `frequency_perplexity` | Perplexity (2^entropy). |
| `ttr` | Type-token ratio. |
| `zipf_slope` | Slope of log-rank vs log-frequency regression. |

## 2. Morphological metrics

[morphs.py](morphs.py) computes word-level and morph-level frequency metrics plus morphological composition statistics from segmented data.

### How to run it

```bash
python src/3_analyze/morphs.py
python src/3_analyze/morphs.py --folder data/2_segmented --out results/3_analyze/morphs.csv
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--folder` | `data/2_segmented` | Input directory with segmented TSV files. |
| `--pattern` | `*.tsv` | Glob pattern for input files. |
| `--out` | `results/3_analyze/morphs.csv` | Output CSV path. |

Input files are TSV with columns: `word`, `frequency`, `segmentation`, `annotation`. Segmentation uses spaces between morphemes; annotation labels each morpheme as root (`R`), prefix (`P`), suffix (`S`), or interfix (`I`).

### Output

Output file: [morphs.csv](../../results/3_analyze/morphs.csv)

Word-level and morph-level variants of the same metrics as `analyze.py` (prefixed `word_` and `morph_`), plus:

| Column | Meaning |
| --- | --- |
| `avg_morphs_per_word` | Average number of morphemes per word. |
| `avg_roots_per_word` | Average root count per word. |
| `avg_affixes_per_word` | Average affix count per word. |
| `avg_prefixes_per_word` | Average prefix count per word. |
| `avg_suffixes_per_word` | Average suffix count per word. |
| `compounding_index` | Ratio of roots to total morphemes (weighted). |
| `affix_deviation` | Asymmetry between suffixes and prefixes. |
| `root_entropy`, `affix_entropy` | Entropy by morph type. |
| `root_count`, `affix_count` | Number of distinct morphs by type. |

## 3. Compression metrics

[compressing/compress.py](compressing/compress.py) measures how well word lists compress, as a proxy for orthographic regularity.

It reads annotated files from `data/2_annotated/`, compresses them, and writes results to `results/3_analyze/compress1M.csv`.

### How to run it

```bash
python src/3_analyze/compressing/compress.py
```

This script has no command-line options.

## 4. Total results

[total_results.py](total_results.py) merges corpus metrics, morphological metrics, compression metrics, language metadata, and script classification into one file.

### How to run it

```bash
python src/3_analyze/total_results.py
```

This script has no command-line options. It reads from fixed paths.

### Inputs

| Source | Path |
| --- | --- |
| Corpus metrics | `results/3_analyze/results.csv` |
| Morph metrics | `results/3_analyze/morphs.csv` |
| Compression | `results/3_analyze/compress100k.csv` |
| Language overview | `results/1_process/2_aggregate/language_overview.csv` |
| Language metadata | `metadata/languages.csv` |
| Script types | `data/5_other/script_types.csv` |

### Output

Output file: [total_results.csv](../../results/3_analyze/total_results.csv)

Combines identifiers (`lang`, `script`, `family`, `genus`, `typology`), corpus counts, all frequency and morphological metrics, and compression ratio into 47 columns.

## 5. LaTeX table

[print_table.py](print_table.py) generates a LaTeX table from total results for thesis inclusion.

### How to run it

```bash
python src/3_analyze/print_table.py
```

### Output

Output file: [print_table.tex](../../results/3_analyze/print_table.tex)

## Pipeline

```
data/2_annotated/*.csv ──→ analyze.py ──→ results.csv
                       └─→ compress.py ─→ compress1M.csv
data/2_segmented/*.tsv ──→ morphs.py ──→ morphs.csv
                                              ↓
              metadata + language_overview ──→ total_results.py ──→ total_results.csv
                                                                        ↓
                                                              print_table.py ──→ print_table.tex
```
