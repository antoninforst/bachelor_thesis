# Morphological Analysis Pipeline

Morphological analysis tool that segments words into morphemes and identifies root morphemes. The pipeline consists of two stages:

1. **Segmentation** – splits a word into morphs (e.g. `přípravuje` → `při prav uj e`)
2. **Root identification** – marks which morph(s) are roots (e.g. `při @prav uj e`)

Both stages use LogisticRegression classifiers trained on annotated data. Evaluation uses K-fold cross-validation so every word is scored exactly once while held out. The worst predictions (ranked by Levenshtein distance) are shown first — useful for spotting mislabeled training data.

## Files

| File | Role |
|------|------|
| `generate.py` | Main interface — CLI for single-language training, evaluation, interactive mode, file processing |
| `generate_multiple.py` | Batch processing — evaluate/train all languages in parallel, write result CSVs |
| `segmentation.py` | Segmentation model — feature extraction, training, prediction, K-fold evaluation |
| `root_identification.py` | Root identification model — feature extraction, training, prediction, K-fold evaluation |

## Directory structure

```
data/
├── 1_aggregated/              # Word frequency CSVs (ces.csv, eng.csv, …)
├── 2_1_segmentation/
│   ├── training_data/         # Annotated .morf files (ces.morf, eng.morf, …)
│   └── models/                # Saved model pickles (*_seg.pkl, *_root.pkl)
└── 3_results/                 # Batch result CSVs (result_segment_true.csv, …)
```

## Data format

**Annotated source file** (e.g. `ces.morf`): one word per line, morphs separated by spaces, root morphs prefixed with `@`:

```
za @hrad a
@zlat o @kop k a
pro @da v ač k a
```

**Frequency file** (e.g. `ces.csv`): CSV with columns `Item,Frequency`:

```
slovo,12345
dům,6789
```

## Evaluation metrics

Both segmentation and root identification are evaluated using the same set of metrics based on **Levenshtein (edit) distance** between gold and predicted position sequences.

### How it works

For **segmentation**, each word has a sequence of boundary positions (gap indices where morph splits occur). For example the word `při prav uj e` (raw: `připravuje`) has boundaries at positions `[2, 6, 8]`. The model predicts its own set of boundary positions, and the Levenshtein distance between the two sequences is computed.

For **root identification**, each word has a sequence of root morph indices. For example `při @prav uj e` has root positions `[1]` (the second morph). The model predicts its own root positions, and the Levenshtein distance between the two sequences is computed.

### Reported metrics

| Metric | Description |
|--------|-------------|
| **Avg Levenshtein** | Mean Levenshtein distance between gold and predicted position sequences across all words. Lower is better (0 = perfect). |
| **Lev score** | Mean normalized score per word: `1 − dist / max(\|gold\|, \|pred\|)`. Higher is better (1.0 = perfect). If both sequences are empty the score is 1. |
| **Word accuracy** | Fraction of words with Levenshtein distance = 0 (exact match). |
| **Correct / Wrong** | Absolute count of exact-match vs. mismatched words. |

### Worst predictions

After the summary, the top-N worst predictions are printed sorted by descending Levenshtein distance. This helps identify:
- Training data annotation errors (mislabeled words)
- Systematically hard word patterns

## Usage

All commands assume you run from `src/2_1_segmentation/`.

### Evaluate and train (single language)

```bash
python generate.py -source ../../data/2_1_segmentation/training_data/ces.morf \
                   -freq ../../data/1_aggregated/ces.csv
```

Runs K-fold cross-validated evaluation on both segmentation and root identification, prints the top-N worst predictions, then trains final models on all data.

```bash
python generate.py -source ../../data/2_1_segmentation/training_data/ces.morf \
                   -freq ../../data/1_aggregated/ces.csv \
                   -folds 10 -n 30 \
                   -save_model ../../data/2_1_segmentation/models/ces_seg.pkl \
                   -save_root_model ../../data/2_1_segmentation/models/ces_root.pkl
```

### Interactive mode (load saved models)

```bash
python generate.py -model ../../data/2_1_segmentation/models/ces_seg.pkl \
                   -root_model ../../data/2_1_segmentation/models/ces_root.pkl \
                   -freq ../../data/1_aggregated/ces.csv
```

Enter words at the `WORD>` prompt to see segmentation and root predictions. Add `-source ...ces.morf` to enable saving corrections back to the source file.

### Segment a text file

```bash
python generate.py -model ../../data/2_1_segmentation/models/ces_seg.pkl \
                   -root_model ../../data/2_1_segmentation/models/ces_root.pkl \
                   -freq ../../data/1_aggregated/ces.csv \
                   -text input.txt -save_to output.morf
```

### Batch: evaluate/train all languages

```bash
python generate_multiple.py
```

Discovers all `.morf` files in the training-data directory, runs evaluation and training for each language **in parallel**, and writes aggregate result CSVs to `results/2_segmentation/`.

```bash
python generate_multiple.py -train_dir ../../data/2_1_segmentation/training_data \
                            -freq_dir ../../data/1_aggregated \
                            -result_dir ../../results/2_segmentation \
                            -folds 10 -workers 4
```

**Output files:**

| File | Content |
|------|---------|
| `result_segment_true.csv` | Per-language segmentation metrics (avg Levenshtein, Lev score, word accuracy) |
| `result_root_true.csv` | Per-language root identification metrics (avg Levenshtein, Lev score, word accuracy) |

## Makefile

The Makefile provides shorthand targets. Run from `src/2_1_segmentation/`:

```bash
make help                        # show available targets and languages
make eval LNG=ces                # K-fold evaluate + train Czech
make eval LNG=eng FOLDS=10       # 10-fold for English
make eval-all                    # evaluate all languages sequentially
make batch                       # evaluate all languages in parallel, write result CSVs
make batch WORKERS=4 FOLDS=10    # parallel with 4 workers, 10 folds
make train LNG=ces               # train without evaluation
make interactive LNG=ces         # interactive mode
make segment LNG=ces TEXT=in.txt OUTPUT=out.morf
```

| Variable | Default | Description |
|----------|---------|-------------|
| `LNG` | `ces` | Language code |
| `FOLDS` | `5` | K-fold cross-validation folds |
| `N_WORST` | `20` | Number of worst predictions to show |
| `SEED` | `42` | Random seed |
| `WORKERS` | (all CPUs) | Max parallel workers for `batch` |
| `PYTHON` | `python3` | Python executable |

## Arguments reference (`generate.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-source` | str | – | Annotated source file (segmented words, one per line) |
| `-freq` | str | – | Corpus frequency CSV (`Item,Frequency`) |
| `-folds` | int | 5 | K-fold cross-validation folds (0 = skip evaluation) |
| `-n` | int | 20 | Number of worst predictions to show |
| `-seed` | int | 42 | Random seed |
| `-eval_only` | flag | – | Only evaluate, skip training final models |
| `-model` | str | – | Load a saved segmentation model (pickle) |
| `-save_model` | str | – | Save trained segmentation model to this path |
| `-root_model` | str | – | Load a saved root identification model (pickle) |
| `-save_root_model` | str | – | Save trained root model to this path |
| `-text` | str | – | Segment words from a text file (requires `-save_to`) |
| `-save_to` | str | – | Output path for segmented text |

## Arguments reference (`generate_multiple.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-train_dir` | str | `../../data/2_1_segmentation/training_data` | Directory with `.morf` training files |
| `-freq_dir` | str | `../../data/1_aggregated` | Directory with frequency CSVs |
| `-model_dir` | str | `../../data/2_1_segmentation/models` | Directory to save models into |
| `-result_dir` | str | `../../results/2_segmentation` | Directory to write result CSVs |
| `-folds` | int | 5 | K-fold cross-validation folds |
| `-seed` | int | 42 | Random seed |
| `-workers` | int | CPU count | Max parallel worker processes |
| `-eval_only` | flag | – | Only evaluate, do not save models |
