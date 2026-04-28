PYTHON = python -u
LANGS =
COVERAGE = 94
TOP_N =
TOP_N_ARG = $(if $(TOP_N),--top-n $(TOP_N),)
WORKERS =
WORKERS_ARG = $(if $(WORKERS),--jobs $(WORKERS),)

RAW_DIR = data/0_raw
AGG_DIR = data/1_aggregated
ANNOTATED_DIR = data/2_annotated
FILTER_DIR = results/1_process/1_filter
TRUNCATE_DIR = results/1_process/3_truncate
ROOT = ../..

.PHONY: all help annotated filter filter_hapax filter_script aggregate aggregate_continue parse_words statistics truncate language_overview clear_aggregated clear_annotated

all: filter aggregate parse_words statistics truncate

filter: filter_hapax filter_script

clear: clear_aggregated clear_annotated

filter_hapax:
	$(PYTHON) 1_filter/hapax_overlap.py $(LANGS) --raw-dir $(ROOT)/$(RAW_DIR) --out-dir $(ROOT)/$(FILTER_DIR)
	$(PYTHON) 1_filter/generate_ignore_list.py --overlap-csv $(ROOT)/$(FILTER_DIR)/hapax_overlap.csv --out-dir $(ROOT)/$(FILTER_DIR)

filter_script:
	$(PYTHON) 1_filter/script_check.py $(LANGS) --raw-dir $(ROOT)/$(RAW_DIR) --out-dir $(ROOT)/$(FILTER_DIR) --ignore-csv $(ROOT)/$(FILTER_DIR)/ignored_files.csv --scripts-csv $(ROOT)/src/1_process/1_filter/scripts.csv
	$(PYTHON) 1_filter/generate_ignore_list.py --overlap-csv $(ROOT)/$(FILTER_DIR)/hapax_overlap.csv --out-dir $(ROOT)/$(FILTER_DIR) --script-csv $(ROOT)/$(FILTER_DIR)/script_check.csv

aggregate:
	$(PYTHON) 2_aggregate/aggregate.py $(LANGS) --raw-dir $(ROOT)/$(RAW_DIR) --out-dir $(ROOT)/$(AGG_DIR) --ignore $(ROOT)/$(FILTER_DIR)/ignored_files.csv --scripts-csv $(ROOT)/src/1_process/1_filter/scripts.csv --lang-overview $(ROOT)/results/1_process/2_aggregate/language_overview.csv $(WORKERS_ARG)

aggregate_continue:
	$(PYTHON) 2_aggregate/aggregate.py $(LANGS) --raw-dir $(ROOT)/$(RAW_DIR) --out-dir $(ROOT)/$(AGG_DIR) --ignore $(ROOT)/$(FILTER_DIR)/ignored_files.csv --scripts-csv $(ROOT)/src/1_process/1_filter/scripts.csv --lang-overview $(ROOT)/results/1_process/2_aggregate/language_overview.csv --skip-existing --no-report $(WORKERS_ARG)

parse_words:
	$(PYTHON) 2_aggregate/parse_words.py $(LANGS) --agg-dir $(ROOT)/$(AGG_DIR)

statistics:
	$(PYTHON) 3_truncate/statistics.py --agg-dir $(ROOT)/$(AGG_DIR) --output $(ROOT)/$(TRUNCATE_DIR)/statistics.csv

truncate:
	$(PYTHON) 3_truncate/truncate.py $(LANGS) --src $(ROOT)/$(AGG_DIR) --out $(ROOT)/$(ANNOTATED_DIR) --stats $(ROOT)/$(TRUNCATE_DIR)/statistics.csv --coverage $(COVERAGE) $(TOP_N_ARG)

clear_aggregated:
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('$(ROOT)/$(AGG_DIR)').glob('*.csv')]"

clear_annotated:
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('$(ROOT)/$(ANNOTATED_DIR)').glob('*.csv')]"

help:
	@echo "annotated filter filter_hapax filter_script aggregate aggregate_continue parse_words statistics truncate language_overview clear_aggregated clear_annotated"
