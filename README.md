# ViTALBench

ViTALBench is a Vietnamese traffic-law visual question answering dataset and research codebase for multimodal traffic-scene understanding and legal reasoning.

The repository contains the released images, multiple-choice questions, Vietnamese traffic-law corpora, baseline inference scripts, and the paper source.

ViTALBench contains two top-level tiers: detection and law.
The law tier is divided into violation identification, regulation application, and multi-step legal reasoning.
The dataset is constructed from publicly accessible traffic images and Vietnamese traffic regulations through LLM-assisted drafting and human verification.

## Release statistics

The statistics below are computed from the canonical files in this repository.

| Resource | Count |
|---|---:|
| Images | 551 |
| Image categories | 11 |
| Total questions | 4,536 |
| Detection questions | 2,122 |
| Law questions | 2,414 |
| Legal corpus chunks | 990 |

The law questions are distributed across the three categories used in the paper:

| Category | Count |
|---|---:|
| Violation identification | 482 |
| Regulation Application | 1,601 |
| Multi-step legal Reasoning | 331 |

The canonical release files are `detection.json`, `law.json`, `law_categories.json`, `law_db.json`, and the images under `data/`.

Run the release audit after changing these files:

```bash
python scripts/check_release_stats.py
```

Regenerate Fig. 3 from the same counts with:

```bash
python scripts/plot_dataset_statistics.py
```

## Repository layout

```text
data/                 Released traffic images grouped by category.
detection.json         Detection questions.
law.json               Law questions with legal references.
law_categories.json    Law-question category labels.
law_db.json            Canonical 990-chunk legal corpus.
scripts/               Dataset, evaluation, and release-audit scripts.
tests/                 Release validation tests.
paper/                 ViTALBench paper source.
```

## Installation

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Some model and retrieval dependencies are large and may require a compatible GPU, Java installation, or external model downloads.

## Quick checks

Run the release audit and tests:

```bash
python scripts/check_release_stats.py
PYTHONPATH=. pytest -q
```

The audit is dependency-free and verifies the question counts, image count, category count, law-category counts, and legal-corpus size.

## Running inference

The top-level scripts `answer_vqa.py`, `annotate_law.py`, and `get_law.py` provide the original VQA and law-processing workflows.

Inspect each script's command-line help before running a model experiment:

```bash
python answer_vqa.py --help
python annotate_law.py --help
python get_law.py --help
```

Model inference requires downloading the selected model separately and may require substantial GPU memory.

## Paper and reproducibility

The paper source uses the name ViTALBench and the repository-derived dataset statistics above.

The legal corpus is built from QCVN 41:2024/BGTVT, Law No. 36/2024/QH15, and Decree No. 168/2024/ND-CP.

The paper figure can be regenerated from the canonical files with `python scripts/plot_dataset_statistics.py`.

Generated logs, caches, indexes, and model outputs are not included in the public release and should be regenerated locally when needed.

The release audit validates the canonical dataset inputs.

It does not independently reproduce every historical model result reported in the paper.

## Data and legal-corpus notice

The images were collected from publicly accessible sources and are included for research use.

Users should verify image provenance, copyright, privacy, and redistribution requirements before publishing a mirror or using the images commercially.

The legal corpus contains Vietnamese traffic-law text for research and reference.

It is not legal advice, and current official sources should be consulted for legal decisions.
