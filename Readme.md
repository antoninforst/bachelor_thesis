# Morphological Complexity of Languages

This repository contains code and data for my bachelor's thesis on morphological complexity of languages.

The project is divided into top-level folders:

| Folder | Purpose |
| --- | --- |
| `src/` | Python programs used in the project. |
| `data/` | Input data and intermediate data files. |
| `results/` | Output files created by the programs. |
| `notebooks/` | Exploration notebooks and graph notebooks. |
| `tests/` | Tests for the implemented programs. |
| `doc/` | Thesis text and related documents. |

The main work folders are divided into phases:

| Phase | Meaning |
| --- | --- |
| `1_process` | Download, filter, aggregate, and truncate language frequency lists. |
| `2_segment` | Segment words and work with morphological annotation. |
| `3_analyze` | Analyze the processed and annotated data. |

Python dependencies are listed in [requirements.txt](requirements.txt). Install them before running the programs:

```bash
pip install -r requirements.txt
```
*Install pyarrow for dowloading glot500*

Currently, the implemented part is the processing phase. Its documentation is here:

[src/1_process/preprocess.md](src/1_process/preprocess.md)
