PYTHON   ?= python3
ROOT     ?= ../..
LNG      ?= ces

# Directories (relative to repo root)
TRAIN_DIR = $(ROOT)/data/2_1_segmentation/training_data
MODEL_DIR = $(ROOT)/data/2_1_segmentation/models
FREQ_DIR  = $(ROOT)/data/1_aggregated
RESULT_DIR= $(ROOT)/data/3_results

# Derived paths
SOURCE    = $(TRAIN_DIR)/$(LNG).morf
SEG_MODEL = $(MODEL_DIR)/$(LNG)_seg.pkl
ROOT_MODEL= $(MODEL_DIR)/$(LNG)_root.pkl

# Auto-detect frequency file (empty if not found)
FREQ_FILE = $(wildcard $(FREQ_DIR)/$(LNG).csv)
FREQ_FLAG = $(if $(FREQ_FILE),-freq $(FREQ_FILE),)

# Evaluation
N_WORST ?= 20
FOLDS   ?= 5
SEED    ?= 42

ALL_LANGS = $(notdir $(basename $(wildcard $(TRAIN_DIR)/*.morf)))

# ── Phony targets ─────────────────────────────────────────
.PHONY: help eval eval-all train interactive segment batch

help:
	@echo "Usage: make <target> LNG=<code> [options]"
	@echo ""
	@echo "Targets:"
	@echo "  eval         K-fold evaluate + train final models (LNG=ces FOLDS=5 N_WORST=20)"
	@echo "  eval-all     Evaluate all languages (sequential)"
	@echo "  batch        Evaluate all languages in parallel, write result CSVs"
	@echo "  train        Train and save models without evaluation (LNG=ces)"
	@echo "  interactive  Interactive mode with saved models (LNG=ces)"
	@echo "  segment      Segment a text file (LNG=ces TEXT=input.txt OUTPUT=out.morf)"
	@echo ""
	@echo "Available languages: $(ALL_LANGS)"
	@echo "Frequency file:      $(or $(FREQ_FILE),(not found for $(LNG)))"

# ── Evaluate + train ──────────────────────────────────────
eval: $(SOURCE)
	$(PYTHON) generate.py -source $(SOURCE) $(FREQ_FLAG) \
		-folds $(FOLDS) -n $(N_WORST) -seed $(SEED) \
		-save_model $(SEG_MODEL) -save_root_model $(ROOT_MODEL)

# ── Evaluate all languages ────────────────────────────────
eval-all:
	@for lang in $(ALL_LANGS); do \
		echo "=== $$lang ==="; \
		$(MAKE) --no-print-directory eval LNG=$$lang; \
		echo ""; \
	done

# ── Train only (no evaluation) ────────────────────────────
train: $(SOURCE) | $(MODEL_DIR)
	$(PYTHON) generate.py -source $(SOURCE) $(FREQ_FLAG) \
		-folds 0 -seed $(SEED) \
		-save_model $(SEG_MODEL) -save_root_model $(ROOT_MODEL)

$(MODEL_DIR):
	mkdir -p $@

# ── Interactive mode ──────────────────────────────────────
interactive: $(SEG_MODEL) $(ROOT_MODEL)
	$(PYTHON) generate.py -model $(SEG_MODEL) -root_model $(ROOT_MODEL) \
		$(FREQ_FLAG) -source $(SOURCE)

# ── Segment a text file ──────────────────────────────────
segment: $(SEG_MODEL) $(ROOT_MODEL)
ifndef TEXT
	$(error TEXT is required, e.g. make segment LNG=ces TEXT=input.txt OUTPUT=out.morf)
endif
ifndef OUTPUT
	$(error OUTPUT is required, e.g. make segment LNG=ces TEXT=input.txt OUTPUT=out.morf)
endif
	$(PYTHON) generate.py -model $(SEG_MODEL) -root_model $(ROOT_MODEL) \
		$(FREQ_FLAG) -text $(TEXT) -save_to $(OUTPUT)

# ── Batch: all languages in parallel ─────────────────────
WORKERS ?=

batch:
	$(PYTHON) generate_multiple.py \
		-train_dir $(TRAIN_DIR) -freq_dir $(FREQ_DIR) \
		-model_dir $(MODEL_DIR) -result_dir $(RESULT_DIR) \
		-folds $(FOLDS) -seed $(SEED) \
		$(if $(WORKERS),-workers $(WORKERS),)
