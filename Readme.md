# The pipeline

1. Download the data from Leipzig Corpus
2. Aggregate the data to language specific datasets. It should also filter out non-words and headers
3. Prepare the data into .morph file format
   1. Restrict the data to 95 % of use
   2. Segment the data
   3. Find the root
   4. Annotate (find lemma)
4. Apply measures and return results

---

## Step 2 – Aggregation (`src/0_data_processing/aggregate.py`)

Groups every CSV in `data/0_raw/` by its **first three characters** (language
code), merges duplicate words, sums their frequencies, removes non-word entries
(anything without a Unicode letter), and writes the result sorted by frequency
to `data/1_aggregated/<lang>.csv`.

### Prerequisites

```
pip install pandas
pip install regex   # optional – improves Unicode letter detection
```

### Standalone usage

```bash
# Process all language codes found in data/0_raw/
python src/0_data_processing/aggregate.py

# Process only selected codes
python src/0_data_processing/aggregate.py ces eng
```

Custom directories can be specified with `--raw-dir` and `--out-dir`.

### Via Makefile

```bash
make aggregate          # aggregate all language codes
make clean-aggregate    # remove aggregated files
```

---

## Step 3 – Preprocessing (`src/0_data_processing/preprocess.py`)

Cuts each aggregated CSV at a **cumulative frequency cutoff** (default 95 %)
and adds an empty `lemma` column, preparing the file for downstream annotation.
Output is written to `data/2_annotated/<lang>.csv`.

### Standalone usage

```bash
# All files, default 95 % cutoff
python src/0_data_processing/preprocess.py

# All files, 99 % cutoff
python src/0_data_processing/preprocess.py --cutoff 0.99

# Selected language codes
python src/0_data_processing/preprocess.py ces eng
```

### Via Makefile

```bash
make preprocess                # default 95 % cutoff
make preprocess CUTOFF=0.99    # override cutoff
make clean-preprocess           # remove preprocessed files
```

---

## Step 4 – Analysis (`src/3_analyze/analyze.py`)

Computes measures on the annotated files and writes a combined results table
to `data/3_results/results.csv` (one row per language).

Measures are organised into three **categories** based on what data they need:

| Category | Required columns | Available now? |
|---|---|---|
| FREQUENCY | `word`, `frequency` | Yes |
| SEGMENTATION | `separation` (morphs joined by `+`) | No – needs segmentation step |
| LEMMA | `lemma` (filled) | No – needs annotation step |

The runner auto-detects which categories are available in each file and skips
measures whose data is missing.

### Current frequency measures

`word_count`, `total_frequency`, `hapax_count`, `hapax_ratio`,
`avg_word_length`, `avg_word_length_weighted`, `frequency_entropy`,
`frequency_perplexity`

### Adding a new measure

1. Open the file for the appropriate category under `src/3_analyze/measures/`
   (e.g. `frequency.py`, `segmentation.py`, `lemma.py`).
2. Create a class inheriting from `Measure`.
3. Set `name` (used as CSV column) and `category`.
4. Implement `compute(self, data: LanguageData) -> float | int`.
5. The measure is auto-registered – no other changes needed.

### Standalone usage

```bash
python src/3_analyze/analyze.py              # all files
python src/3_analyze/analyze.py ces eng       # selected codes
```

### Via Makefile

```bash
make analyze          # full pipeline: aggregate → preprocess → analyze
make clean-analyze    # remove results
```