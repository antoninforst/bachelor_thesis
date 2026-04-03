PYTHON = python3
CUTOFF = 0.95
SAMPLE_SIZE = 5000
LNG =

RAW_DIR      = data/0_raw
AGG_DIR      = data/1_aggregated
ANNOTATED_DIR = data/2_annotated
RESULTS_DIR  = data/3_results
SAMPLES_DIR  = data/4_samples

# Collect all raw CSVs and derive 3-letter language codes
RAW_FILES  = $(wildcard $(RAW_DIR)/*.csv)
LANG_CODES = $(sort $(foreach f,$(notdir $(RAW_FILES)),$(shell echo $(f) | cut -c1-3)))
AGG_TARGETS = $(patsubst %,$(AGG_DIR)/%.csv,$(LANG_CODES))
ANN_TARGETS = $(patsubst %,$(ANNOTATED_DIR)/%.csv,$(LANG_CODES))

# ── Phony targets ──────────────────────────────────────────
.PHONY: all aggregate preprocess analyze sample data-statistics clean-aggregate clean-preprocess clean-sample help

all: aggregate preprocess analyze

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

# ── Step 2.5: Sample word lists ─────────────────────────────
sample:
	$(PYTHON) src/1_preprocessing/sample_wordlist.py -n $(SAMPLE_SIZE) $(LNG)

clean-sample:
	rm -f $(SAMPLES_DIR)/*.csv $(SAMPLES_DIR)/*.png

# ── Step 3: Analyse annotated data ───────────────────────
analyze:
	$(PYTHON) src/3_analyze/analyze.py

# ── Data statistics ──────────────────────────────────────
data-statistics:
	$(PYTHON) src/0_data_processing/statistics.py --output $(RESULTS_DIR)/statistics.csv

# ── Help ───────────────────────────────────────────────────
help:
	@echo "Targets:"
	@echo "  aggregate        – aggregate raw CSVs by language code"
	@echo "  preprocess       – cut aggregated files at frequency cutoff"
	@echo "  analyze          – compute measures and write results"
	@echo "  sample           – sample shorter word lists from annotated data"
	@echo "  data-statistics  – compute raw data statistics"
	@echo "  clean-aggregate  – remove aggregated files"
	@echo "  clean-preprocess – remove preprocessed files"
	@echo "  clean-sample     – remove sampled files"
	@echo ""
	@echo "Variables:"
	@echo "  CUTOFF=$(CUTOFF)  (cumulative frequency ratio)"
	@echo "  SAMPLE_SIZE=$(SAMPLE_SIZE)  (number of words to sample)"
	@echo "  LNG=          (language code(s) for sample, e.g. LNG=eng)"
	@echo ""
	@echo "Detected language codes: $(LANG_CODES)"
