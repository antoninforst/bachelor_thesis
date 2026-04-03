# Plan: Universal Morphological Segmentation Model

## TL;DR

Create `generate_universal.py` — a new script that trains a single universal segmentation+root model from all 11 annotated languages, then applies it to all 73+ languages that only have frequency data. Uses leave-language-out cross-validation for evaluation. Integrates into the root Makefile between `preprocess` and `analyze`.

## Context

- **11 languages** have .morf training data (ces, deu, ell, eng, fre, hin, jap, lat, rus, spa, tel)
- **73+ languages** have aggregated frequency data in `data/1_aggregated/` (more being generated; eventually all languages will have freq data)
- Language codes are **consistent** across all files (.morf, 1_aggregated, 2_annotated) — no code mapping needed
- The existing `segmentation.py` and `root_identification.py` models use character-level features (language-agnostic) + optional corpus frequency features (language-specific)
- The universal model pools character-pattern training from all languages; at inference time, swaps in the target language's freq_map
- **2_annotated CSV format:** 3 columns — `word` (will be overwritten with segmented form), `frequency`, `lemma` (empty for now, handled by a different model later)
- **Segmented word format:** morphs separated by **spaces**, roots marked with `@` prefix (e.g., `za @hrad a`). Original word reconstructed by stripping spaces and `@`

## Steps

### Phase 1: Create `generate_universal.py`

1. **New file:** `src/2_1_segmentation/generate_universal.py`
   - Handle 3 CLI modes: `train-eval`, `train-save`, `segment`
   - Shared: `load_all_training_data()` function that discovers all .morf files, loads morf words + root data + freq_maps (from 1_aggregated, same code)
   - If a freq file doesn't exist yet for a training language, train with freq_map=None

2. **`train-eval` mode** — Leave-language-out cross-validation
   - For each of the 11 languages: train on the other 10, test on the held-out one
   - Each language's training examples use its own freq_map (or None if aggregated file not yet available)
   - Report per-language metrics (levenshtein_score, word_accuracy) + aggregate
   - Write results to `data/3_results/result_segment_universal.csv` and `result_root_universal.csv`

3. **`train-save` mode** — Train final models on all data
   - Pool all 11 languages' examples (each with its own freq_map)
   - Train one `MorphSegmenter` and one `RootIdentifier`
   - Save to `data/2_1_segmentation/models/universal_seg.pkl` and `universal_root.pkl`

4. **`segment` mode** — Apply universal models to all 2_annotated CSVs
   - Load `universal_seg.pkl` and `universal_root.pkl`
   - For each CSV in `data/2_annotated/`:
     - Load the word list (columns: word, frequency, lemma)
     - Reconstruct original word from `word` column (strip spaces and `@` if already segmented — but on first run these are plain words)
     - Load freq_map from `data/1_aggregated/<lang>.csv` (same 3-letter code; skip freq features if file missing)
     - Set `seg_model.freq_map = freq_map` and `root_model.freq_map = freq_map`
     - For each word: segment → identify roots → produce annotated form (e.g., `za @hrad a`)
     - **Replace** the `word` column with the segmented+root form (morphs separated by **spaces**, roots prefixed with `@`)
     - Frequency and lemma columns preserved unchanged
     - Write back to the same CSV file
   - Processes **all** languages by default
   - **`--improve` flag:** additionally re-process languages that have per-language .morf models using their dedicated (better) models, overwriting the universal results for those languages

### Phase 2: Makefile Integration

5. **Update root `makefile`**
   - Add `segment` phony target after `preprocess`
   - `segment` depends on `preprocess` (needs 2_annotated CSVs to exist)
   - Calls `python src/2_1_segmentation/generate_universal.py segment`
   - Add `train-universal` target that runs train-save mode
   - Update `all` target: `all: aggregate preprocess segment analyze`

6. **Update `src/2_1_segmentation/makefile`**
   - Add `universal-eval`, `universal-train`, `universal-segment` targets

## Relevant files

- `src/2_1_segmentation/generate_universal.py` — **NEW** main script (3 modes)
- `src/2_1_segmentation/segmentation.py` — Reused unchanged: `build_examples()`, `MorphSegmenter`, `train()`, `load_model()`, `save_model()`, `load_frequency_map()`, `load_segmented_words()`, `evaluate()`
- `src/2_1_segmentation/root_identification.py` — Reused unchanged: `build_examples()`, `RootIdentifier`, `train()`, `load_model()`, `save_model()`, `load_words_with_roots()`, `evaluate()`
- `makefile` (root) — Add `segment` and `train-universal` targets
- `src/2_1_segmentation/makefile` — Add universal targets
- `data/2_1_segmentation/models/universal_seg.pkl` — Output model (segmentation)
- `data/2_1_segmentation/models/universal_root.pkl` — Output model (root)
- `data/2_annotated/*.csv` — Modified in-place by segment mode
- `data/1_aggregated/*.csv` — Read-only frequency source

## Decisions

- **No code mapping needed:** Language codes are consistent across .morf files, 1_aggregated, and 2_annotated
- **Word column format:** Segment mode replaces the `word` column with segmented+root form (`za @hrad a` — morphs separated by spaces, roots prefixed with `@`). Original word reconstructed by stripping spaces and `@`
- **Lemma column untouched** — a separate model handles that later
- **All 2_annotated CSVs are processed** by default; `--improve` flag additionally overwrites trained languages with their per-language models; `--langs` flag can limit to specific codes
- **No changes to segmentation.py or root_identification.py** — the universal model reuses existing `build_examples()` and model classes; pooling happens at the feature-dict level

## What has been done

### Prototype infrastructure (completed)

- [x] **Created `generate_universal.py`** — Full CLI with 3 subcommands (`train-eval`, `train-save`, `segment`), all arguments parsed, data discovery helpers implemented, stub logic for all modes (loads files, prints what it would do, no actual model training/prediction)
- [x] **Updated root `makefile`** — Added `segment` (depends on `preprocess`) and `train-universal` targets, updated `all: aggregate preprocess segment analyze`, added to `.PHONY` and help text
- [x] **Updated `src/2_1_segmentation/makefile`** — Added `universal-eval`, `universal-train`, `universal-segment` targets with correct paths, added `ANN_DIR` variable, updated `.PHONY` and help text
- [x] **Created this plan file** in `src/2_1_segmentation/PLAN_universal.md`

### Still TODO

- [ ] **Implement `train-eval`** — Leave-language-out CV: pool `build_seg_examples()` / `build_root_examples()` from N-1 languages, train, evaluate on held-out, collect metrics, write result CSVs
- [ ] **Implement `train-save`** — Pool all training data, train final `MorphSegmenter` + `RootIdentifier`, save as `universal_seg.pkl` + `universal_root.pkl`
- [ ] **Implement `segment`** — Load universal models, iterate 2_annotated CSVs, call `segment_word()` + `root_model.annotate()` per word, write back
- [ ] **Implement `--improve`** — Load per-language models for trained languages, re-segment those CSVs

## Verification

1. Run `python src/2_1_segmentation/generate_universal.py train-eval` — verify leave-language-out metrics printed
2. Run `python src/2_1_segmentation/generate_universal.py train-save` — verify models saved
3. Run `python src/2_1_segmentation/generate_universal.py segment` — verify 2_annotated CSVs updated
4. Run `make segment` from root — verify it invokes correctly
5. Run `make all` — verify aggregate → preprocess → segment → analyze runs end-to-end

## Further Considerations

1. **Frequency data availability:** Aggregated freq files are being generated and will eventually cover all languages. The script handles missing freq files gracefully (freq_map=None). As more become available, model quality improves automatically.
