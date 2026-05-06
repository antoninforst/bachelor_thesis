PYTHON = python -u
ROOT = ../..

ANNOTATED_DIR = data/2_annotated
SEGMENTED_DIR = data/2_segmented
RESULTS_DIR = results/3_analyze

.PHONY: all help analyze morphs compress results print

all: analyze morphs compress results print

analyze:
	$(PYTHON) analyze.py --folder $(ROOT)/$(ANNOTATED_DIR) --out $(ROOT)/$(RESULTS_DIR)/results.csv

morphs:
	$(PYTHON) morphs.py --folder $(ROOT)/$(SEGMENTED_DIR) --out $(ROOT)/$(RESULTS_DIR)/morphs.csv

compress:
	$(PYTHON) compressing/compress.py

results: analyze morphs
	$(PYTHON) total_results.py

print: total_results
	$(PYTHON) print_table.py