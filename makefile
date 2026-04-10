PYTHON = python3
CUTOFF = 0.95
SAMPLE_SIZE = 5000
DOWNLOAD_N = 50
DOWNLOAD_DELAY = 5
LNG =

RAW_DIR      = data/0_raw
AGG_DIR      = data/1_aggregated
ANNOTATED_DIR = data/2_annotated
ANNOTATED2_DIR = data/2_2_annotated_improved
RESULTS_DIR  = results/3_analysis
SAMPLES_DIR  = results/1_preprocessing/samples
TRAIN_DIR    = data/2_1_segmentation/training_data
MODEL_DIR    = data/2_1_segmentation/models

# Collect all raw CSVs and derive 3-letter language codes
RAW_FILES  = $(wildcard $(RAW_DIR)/*.csv)
LANG_CODES = $(sort $(foreach f,$(notdir $(RAW_FILES)),$(shell echo $(f) | cut -c1-3)))
AGG_TARGETS = $(patsubst %,$(AGG_DIR)/%.csv,$(LANG_CODES))
ANN_TARGETS = $(patsubst %,$(ANNOTATED_DIR)/%.csv,$(LANG_CODES))

# ── Phony targets ──────────────────────────────────────────
.PHONY: all download aggregate preprocess segment segment-improve train-universal analyze analyze-improve sample data-statistics clean-aggregate clean-preprocess clean-sample help

all: aggregate preprocess segment analyze

# ── Step 0: Download raw data from Leipzig Corpora ──────────
download:
	$(PYTHON) src/0_data_processing/corpora/leipzig/lepzig_download.py -n $(DOWNLOAD_N) --delay $(DOWNLOAD_DELAY)

# ── Step 1: Aggregate raw data ─────────────────────────────
aggregate: $(AGG_TARGETS)

$(AGG_DIR)/%.csv: $(RAW_DIR)/%*
	$(PYTHON) src/0_data_processing/aggregate.py $*

clean-aggregate:
	rm -f $(AGG_DIR)/*.csv

# ── Step 2: Preprocess (cut at frequency cutoff) ───────────
preprocess: $(ANN_TARGETS)

$(ANNOTATED_DIR)/%.csv: $(AGG_DIR)/%.csv
	$(PYTHON) src/1_preprocessing/preprocess.py --cutoff $(CUTOFF) $*

clean-preprocess:
	rm -f $(ANNOTATED_DIR)/*.csv

# ── Step 2.1: Segment annotated data (universal model) ──────
segment:
	$(PYTHON) src/2_1_segmentation/generate_universal.py segment \
		--train-dir $(TRAIN_DIR) --freq-dir $(AGG_DIR) \
		--model-dir $(MODEL_DIR) --annotated-dir $(ANNOTATED_DIR)

TRAIN_LANGS = $(sort $(basename $(notdir $(wildcard $(TRAIN_DIR)/*.morf))))

segment-improve:
	mkdir -p $(ANNOTATED2_DIR)
	$(foreach l,$(TRAIN_LANGS),test -f $(ANNOTATED2_DIR)/$(l).csv || cp $(ANNOTATED_DIR)/$(l).csv $(ANNOTATED2_DIR)/$(l).csv 2>/dev/null || true;)
	$(PYTHON) src/2_1_segmentation/generate_universal.py segment \
		--train-dir $(TRAIN_DIR) --freq-dir $(AGG_DIR) \
		--model-dir $(MODEL_DIR) --annotated-dir $(ANNOTATED2_DIR) \
		--langs $(TRAIN_LANGS) \
		--improve

train-universal:
	$(PYTHON) src/2_1_segmentation/generate_universal.py train-save \
		--train-dir $(TRAIN_DIR) --freq-dir $(AGG_DIR) \
		--model-dir $(MODEL_DIR)

# ── Step 2.5: Sample word lists ─────────────────────────────
sample:
	$(PYTHON) src/1_preprocessing/sample_wordlist.py -n $(SAMPLE_SIZE) $(LNG)

clean-sample:
	rm -f $(SAMPLES_DIR)/*.csv $(SAMPLES_DIR)/*.png

# ── Step 3: Analyse annotated data ───────────────────────
analyze:
	$(PYTHON) src/3_analyze/analyze.py

analyze-improve:
	$(PYTHON) src/3_analyze/analyze.py --folder $(ANNOTATED2_DIR) --out $(RESULTS_DIR)/results_improved.csv

# ── Data statistics ──────────────────────────────────────
data-statistics:
	$(PYTHON) src/1_preprocessing/statistics.py --output results/0_data_processing/statistics.csv

# ── Help ───────────────────────────────────────────────────
help:
	@echo "Targets:"
	@echo "  download         – download raw CSVs from Leipzig Corpora"
	@echo "  aggregate        – aggregate raw CSVs by language code"
	@echo "  preprocess       – cut aggregated files at frequency cutoff"
	@echo "  segment          – segment words with universal model"
	@echo "  segment-improve  – segment + improve with per-language models"
	@echo "  train-universal  – train universal segmentation models"
	@echo "  analyze          – compute measures and write results"
	@echo "  analyze-improve  – compute measures from improved annotations"
	@echo "  sample           – sample shorter word lists from annotated data"
	@echo "  data-statistics  – compute raw data statistics"
	@echo "  clean-aggregate  – remove aggregated files"
	@echo "  clean-preprocess – remove preprocessed files"
	@echo "  clean-sample     – remove sampled files"
	@echo ""
	@echo "Variables:"
	@echo "  CUTOFF=$(CUTOFF)  (cumulative frequency ratio)"
	@echo "  SAMPLE_SIZE=$(SAMPLE_SIZE)  (number of words to sample)"
	@echo "  DOWNLOAD_N=$(DOWNLOAD_N)  (number of files to download per run)"
	@echo "  DOWNLOAD_DELAY=$(DOWNLOAD_DELAY)  (seconds between downloads)"
	@echo "  LNG=          (language code(s) for sample, e.g. LNG=eng)"
	@echo ""
	@echo "Detected language codes: $(LANG_CODES)"
