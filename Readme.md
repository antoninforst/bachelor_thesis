# The Pipeline

A Makefile-driven pipeline for downloading, aggregating, segmenting, and analysing word-frequency data from the Leipzig Corpora Collection.

## Prerequisites

```bash
pip install pandas
pip install regex   # optional – improves Unicode letter detection
```

## Quick start

```bash
make all    # runs: aggregate → preprocess → segment → analyze
make help   # list all targets and current variable values
```

## Pipeline overview

| Step | Make target | Description |
|------|-------------|-------------|
| 0 | `download` | Download raw CSVs from Leipzig Corpora |
| 1 | `aggregate` | Merge raw CSVs per language, remove non-words |
| 2 | `preprocess` | Cut at cumulative-frequency cutoff, prepare for annotation |
| 2.1 | `segment` | Segment words with universal morphological model |
| 2.1+ | `segment-improve` | Re-segment with per-language models where available |
| 2.1+ | `train-universal` | Train and save universal segmentation models |
| 2.5 | `sample` | Sample shorter word lists from annotated data |
| 3 | `analyze` | Compute measures, write results table |
| 3+ | `analyze-improve` | Compute measures from improved annotations |
| — | `data-statistics` | Compute raw data statistics |

## Makefile variables

All variables can be overridden on the command line, e.g. `make preprocess CUTOFF=0.99`.

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHON` | `python3` | Python interpreter |
| `CUTOFF` | `0.95` | Cumulative frequency cutoff ratio |
| `SAMPLE_SIZE` | `5000` | Number of words to sample |
| `DOWNLOAD_N` | `50` | Number of files to download per run |
| `DOWNLOAD_DELAY` | `5` | Seconds to wait between downloads |
| `LNG` | *(empty = all)* | Language code(s) to process (e.g. `LNG=eng`) |

---

## Step 0 – Download (`make download`)

Downloads word-frequency CSVs from the Leipzig Corpora Collection. The list of corpora to download is defined in `src/0_data_processing/corpora/leipzig/to_download.tsv`. Already-downloaded entries are tracked in `already.tsv` so the command is safe to re-run.

```bash
make download                          # download 50 files, 5 s delay
make download DOWNLOAD_N=200           # download 200 files
make download DOWNLOAD_DELAY=10        # slower rate between requests
```

**Standalone:**

```bash
python src/0_data_processing/corpora/leipzig/lepzig_download.py -n 50 --delay 5
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-n` | int | **required** | Number of files to download in this run |
| `--delay` | float | `5.0` | Delay in seconds between downloads |

---

## Step 1 – Aggregation (`make aggregate`)

Groups every CSV in `data/0_raw/` by its first three characters (language code), merges duplicate words, sums their frequencies, removes non-word entries, and writes the result sorted by frequency to `data/1_aggregated/<lang>.csv`.

```bash
make aggregate          # aggregate all language codes
make clean-aggregate    # remove aggregated files
```

**Standalone:**

```bash
python src/0_data_processing/aggregate.py              # all languages
python src/0_data_processing/aggregate.py ces eng       # selected codes
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `langs` | positional | all | Three-letter language codes to process |
| `--raw-dir` | path | `data/0_raw` | Directory with raw CSV files |
| `--out-dir` | path | `data/1_aggregated` | Output directory |

---

## Step 2 – Preprocessing (`make preprocess`)

Cuts each aggregated CSV at a cumulative frequency cutoff (default 95 %) and adds an empty `lemma` column, preparing the file for downstream annotation. Output goes to `data/2_annotated/<lang>.csv`.

```bash
make preprocess                # default 95 % cutoff
make preprocess CUTOFF=0.99    # override cutoff
make clean-preprocess          # remove preprocessed files
```

**Standalone:**

```bash
python src/1_preprocessing/preprocess.py                    # all, default cutoff
python src/1_preprocessing/preprocess.py --cutoff 0.99      # custom cutoff
python src/1_preprocessing/preprocess.py ces eng             # selected codes
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `langs` | positional | all | Three-letter language codes to process |
| `--cutoff` | float | `0.95` | Cumulative frequency cutoff ratio |
| `--agg-dir` | path | `data/1_aggregated` | Directory with aggregated CSVs |
| `--out-dir` | path | `data/2_annotated` | Output directory |

---

## Step 2.1 – Segmentation (`make segment`)

Applies the universal morphological model to segment words in `data/2_annotated/` and identify roots.

```bash
make segment              # segment all languages with universal model
make segment-improve      # improve with per-language models (see below)
make train-universal      # train and save universal models
```

`segment-improve` always works with `data/2_2_annotated_improved/`. For each language that has per-language models (`.morf` training data), it checks whether the CSV already exists in that folder. Missing files are copied from `data/2_annotated/` first, then re-segmented using the per-language models. Files already present in `data/2_2_annotated_improved/` are not overwritten before re-segmentation.

**Standalone** (subcommands of `generate_universal.py`):

```bash
# Segment all languages
python src/2_1_segmentation/generate_universal.py segment

# Segment specific languages with improvement
python src/2_1_segmentation/generate_universal.py segment --langs ces deu --improve

# Train and save models
python src/2_1_segmentation/generate_universal.py train-save

# Cross-validation evaluation
python src/2_1_segmentation/generate_universal.py train-eval
```

**Shared options** (all subcommands):

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train-dir` | str | `data/2_1_segmentation/training_data` | Directory with `.morf` training files |
| `--freq-dir` | str | `data/1_aggregated` | Directory with frequency CSVs |
| `--model-dir` | str | `data/2_1_segmentation/models` | Directory for model `.pkl` files |
| `--seed` | int | `42` | Random seed |

**`segment` options:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--annotated-dir` | str | `data/2_annotated` | Directory with annotated CSVs |
| `--improve` | flag | `False` | Use per-language models where available |
| `--langs` | str | all | Only process these language codes |
| `--workers` | int | `min(4, CPUs)` | Number of parallel workers |
| `--skip-segmented` | flag | `False` | Skip already-segmented languages |

**`train-eval` options:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--result-dir` | str | `results/2_segmentation` | Directory for result CSVs |
| `--n-worst` | int | `20` | Number of worst predictions to show |

---

## Step 2.5 – Sampling (`make sample`)

Samples a shorter word list weighted by log-rank (uniform density in log-space) from annotated data.

```bash
make sample                           # sample 5000 words, all languages
make sample SAMPLE_SIZE=1000          # smaller sample
make sample LNG=eng                   # only English
```

**Standalone:**

```bash
python src/1_preprocessing/sample_wordlist.py -n 5000
python src/1_preprocessing/sample_wordlist.py -n 1000 eng ces --plot --seed 42
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `langs` | positional | all | Three-letter language codes |
| `-n` / `--size` | int | `5000` | Number of words to sample |
| `--seed` | int | `None` | Random seed for reproducibility |
| `--plot` | flag | `False` | Generate a log-log verification plot per language |
| `--ann-dir` | path | `data/2_annotated` | Directory with annotated CSVs |
| `--out-dir` | path | `results/1_preprocessing/samples` | Output directory |

---

## Step 3 – Analysis (`make analyze`)

Computes measures on the annotated/segmented files and writes a combined results table to `results/3_analysis/results.csv` (one row per language).

```bash
make analyze              # standard analysis
make analyze-improve      # analysis from improved annotations
```

**Standalone:**

```bash
python src/3_analyze/analyze.py                                              # all files
python src/3_analyze/analyze.py --folder data/2_2_annotated_improved         # improved data
python src/3_analyze/analyze.py --out results/3_analysis/custom.csv              # custom output
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--folder` | path | `data/2_annotated` | Input folder with CSVs |
| `--pattern` | str | `*.csv` | Glob pattern for input files |
| `--out` | path | `results/3_analysis/results.csv` | Output path for results CSV |

Measures are organised into three **categories** based on what data they need:

| Category | Required columns |
|----------|------------------|
| FREQUENCY | `word`, `frequency` |
| SEGMENTATION | `separation` (morphs joined by `+`) |
| LEMMA | `lemma` (filled) |

The runner auto-detects which categories are available in each file and skips measures whose data is missing.

### Adding a new measure

1. Open the file for the appropriate category under `src/3_analyze/measures/`
   (e.g. `frequency.py`, `segmentation.py`, `lemma.py`).
2. Create a class inheriting from `Measure`.
3. Set `name` (used as CSV column) and `category`.
4. Implement `compute(self, data: LanguageData) -> float | int`.
5. The measure is auto-registered – no other changes needed.

---

## Utilities

### Data statistics (`make data-statistics`)

Prints summary statistics (distinct words, total frequency, cutoff info) for aggregated frequency files.

```bash
make data-statistics
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--agg-dir` | path | `data/1_aggregated` | Directory with aggregated CSVs |
| `--output` | path | `None` | Path to save results as CSV |

### Cleanup targets

```bash
make clean-aggregate    # remove aggregated files
make clean-preprocess   # remove preprocessed files
make clean-sample       # remove sampled files and plots
```