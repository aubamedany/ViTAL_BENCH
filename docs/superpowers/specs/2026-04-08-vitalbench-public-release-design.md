# ViTALBench Public Release Design

## Goal

Prepare the repository for public release under the dataset name ViTALBench.

The repository is the source of truth for dataset and corpus statistics.

## Canonical release contents

The release retains the 551 images under `data/`.

The canonical question files are `detection.json` and `law.json`.

The canonical legal corpus is `law_db.json` with 990 chunks.

Legacy corpora and experimental snapshots under `backup/` remain clearly labeled as historical material and are not used for headline statistics.

## Canonical statistics

The release documents 4,536 questions in total.

The release documents 2,122 detection questions and 2,414 law questions.

The law questions contain 482 Violation Identification questions, 1,601 Regulation Application questions, and 331 Multi-step Legal Reasoning questions.

The image collection contains 551 images in 11 categories.

## Documentation alignment

`README.md` will describe the actual repository layout, canonical files, measured statistics, installation, smoke tests, and reproduction limitations.

The paper source will use the ViTALBench name and the repository-authoritative dataset and corpus statistics where it makes dataset claims.

Experimental result claims will not be changed without evidence from repository outputs.

Unverifiable or stale claims will be identified rather than silently presented as reproducible results.

## Repository cleanup

Generated logs, scheduler output, Python caches, IDE metadata, temporary Office files, and generated retrieval indexes will be excluded from the public release where they are not required by the documented workflow.

The image dataset, source code, canonical JSON files, paper source, useful scripts, and reproducibility materials will be retained.

No files under `data/` will be modified.

Existing user worktree changes will not be reset or overwritten.

## Validation

A lightweight audit command will verify the canonical question counts, image count, category count, law-category counts, and corpus chunk count.

The final verification will include the audit command, syntax checks for changed Python files, README reference checks, secret/path scans, and a git diff review.
