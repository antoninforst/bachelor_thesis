# Morphological Complexity of Languages

This repository contains code and data for my bachelor's thesis on morphological complexity of languages.

## Folder structure

| Folder | Purpose |
| --- | --- |
| `src/` | Python programs used in the project. |
| `data/` | Input data and intermediate data files. |
| `results/` | Output files created by the programs. |
| `metadata/` | Language and script metadata tables. |
| `notebooks/` | Exploration and graph notebooks. |
| `tests/` | Tests for the implemented programs. |
| `thesis/` | LaTeX source of the thesis. |
| `doc/` | Supporting documents (word count, prompts). |

## Pipeline

| Phase | Folder | Meaning |
| --- | --- | --- |
| 1 — Process | `src/1_process/` | Download, filter, aggregate, and truncate language frequency lists. |
| 3 — Analyze | `src/3_analyze/` | Compute frequency, morphological, and compression metrics. |

Detailed documentation for each phase:

- [src/1_process/preprocess.md](src/1_process/preprocess.md) — filtering, aggregation, truncation
- [src/3_analyze/analyze.md](src/3_analyze/analyze.md) — corpus metrics, morphology, compression, results

## Setup

Python dependencies are listed in [requirements.txt](requirements.txt):

```bash
pip install -r requirements.txt
```

Install `pyarrow` additionally if downloading Glot500 data.
