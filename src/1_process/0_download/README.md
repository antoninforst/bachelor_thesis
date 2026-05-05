# Download Scripts

Scripts for downloading raw frequency data from different sources.

## bible/get_bible.py

Converts Bible XML files (from `data/0_1_bible/bibles/`) into word-frequency CSVs.
Picks the richest translation when multiple exist for one language.

```bash
python src/1_process/0_download/bible/get_bible.py
python src/1_process/0_download/bible/get_bible.py --dry-run
```

## glot500/get_glot500.py

Downloads monolingual data from the Glot500 Hugging Face dataset and tokenizes it into frequency CSVs.
Uses the HF Parquet API (no auth needed).

```bash
python src/1_process/0_download/glot500/get_glot500.py --all
python src/1_process/0_download/glot500/get_glot500.py --langs alt chv krc
python src/1_process/0_download/glot500/get_glot500.py --skip-existing
python src/1_process/0_download/glot500/get_glot500.py --list-configs
```

## leipzig/get_leipzig.py

Downloads wordlists from the Leipzig Corpora Collection.
Reads corpus names from `to_download.tsv` and tracks progress in `already.tsv`.

```bash
python src/1_process/0_download/leipzig/get_leipzig.py -n 10
python src/1_process/0_download/leipzig/get_leipzig.py -n 5 --delay 10
```

## ud/get_ud.py

Extracts word frequencies from a Universal Dependencies release archive (`.tgz`).
Merges train/dev/test splits per treebank and writes one CSV per language variant.

```bash
python src/1_process/0_download/ud/get_ud.py
```

## wiki/get_wiki_pages.py

Fetches Wikipedia language pages to get typological info (morphological type, word order, etc.).

```bash
python src/1_process/0_download/wiki/get_wiki_pages.py
```
