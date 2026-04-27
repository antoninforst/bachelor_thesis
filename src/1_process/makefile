PYTHON = python
LANGS =
COVERAGE = 94

RAW_DIR = data/0_raw
AGG_DIR = data/1_aggregated
ANNOTATED_DIR = data/2_annotated
FILTER_DIR = results/1_process/1_filter
TRUNCATE_DIR = results/1_process/3_truncate
ROOT = ../..

.PHONY: all annotated filter filter_hapax filter_script aggregate statistics truncate language_overview clear_aggregated clear_annotated

all: filter aggregate statistics truncate

filter: filter_hapax filter_script

clear: clear_aggregated clear_annotated

filter_hapax:
	$(PYTHON) 1_filter/hapax_overlap.py $(LANGS) --raw-dir $(ROOT)/$(RAW_DIR) --out-dir $(ROOT)/$(FILTER_DIR)
	$(PYTHON) 1_filter/generate_ignore_list.py --overlap-csv $(ROOT)/$(FILTER_DIR)/hapax_overlap.csv --out-dir $(ROOT)/$(FILTER_DIR)

filter_script:
	$(PYTHON) 1_filter/script_check.py $(LANGS) --raw-dir $(ROOT)/$(RAW_DIR) --out-dir $(ROOT)/$(FILTER_DIR) --ignore-csv $(ROOT)/$(FILTER_DIR)/ignored_files.csv
	$(PYTHON) 1_filter/generate_ignore_list.py --overlap-csv $(ROOT)/$(FILTER_DIR)/hapax_overlap.csv --out-dir $(ROOT)/$(FILTER_DIR) --script-csv $(ROOT)/$(FILTER_DIR)/script_check.csv

aggregate:
	$(PYTHON) 2_aggregate/aggregate.py $(LANGS) --raw-dir $(ROOT)/$(RAW_DIR) --out-dir $(ROOT)/$(AGG_DIR) --ignore $(ROOT)/$(FILTER_DIR)/ignored_files.csv

statistics:
	$(PYTHON) 3_truncate/statistics.py --agg-dir $(ROOT)/$(AGG_DIR) --output $(ROOT)/$(TRUNCATE_DIR)/statistics.csv

truncate:
	$(PYTHON) 3_truncate/truncate.py $(LANGS) --src $(ROOT)/$(AGG_DIR) --out $(ROOT)/$(ANNOTATED_DIR) --stats $(ROOT)/$(TRUNCATE_DIR)/statistics.csv --coverage $(COVERAGE)

clear_aggregated:
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('$(ROOT)/$(AGG_DIR)').glob('*.csv')]"

clear_annotated:
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('$(ROOT)/$(ANNOTATED_DIR)').glob('*.csv')]"
