# VNTrafficVQA / VNTraffic-Agent

Experiment code for **VNTrafficVQA**, an option-blind agentic benchmark for
grounded Vietnamese traffic-law reasoning over images (AAAI 2026), and
**VNTraffic-Agent**, a training-free option-blind two-phase agent.

This repository implements the full programme in `EXPERIMENT_PLAN.md`: the data
gates, the E1–E4 ablation ladder, a ≤14B model ladder, citation-grounding and
leakage metrics, ablations, and the paper tables T1–T6.

- **Dataset** — `data.json`: 2,475 MCQ items (1,156 detection + 1,319 law) over
  ~300 Vietnamese traffic images, 11 categories. Law items carry gold citations.
- **Statute corpus** — `legal_traffic_chunks.json`: 777 clause-level chunks
  (Luật 36/2024, NĐ 168/2024, QCVN 41:2024).

## Claims → conditions → tables

| Claim | Conditions | Headline |
|---|---|---|
| C1 perception–legal gap | E1, both tiers, full ladder | T1 |
| C2 retrieval helps but leaks | E2 vs E3 | T2 |
| C3 option exposure = leakage | E2/E3/E4 decomposition | **T3** |
| C4 option-blind agent is best-grounded | E4 vs E1–E3 | T4, T5 |

Conditions (one variable changes per rung):

| ID | Retrieval | Query | Options at retrieval |
|----|-----------|-------|----------------------|
| E1 | none | — | — |
| E2 | single-shot | question only | hidden |
| E3 | single-shot | question + options | **exposed** (leakage probe) |
| E4 | iterative agent | agent-formulated | hidden (Phase 1) |

## Install

```bash
pip install -r requirements.txt
```

The **data gates, metrics, stats, and tables run with only PyYAML** (everything
else is pure Python). `torch`/`transformers`/`sentence-transformers`/`faiss` are
needed only for real model inference and dense retrieval.

## Pipeline

### 1. Data gates → freeze dataset v1 (`EXPERIMENT_PLAN.md §7`)

```bash
python scripts/run_gates.py
```

Runs, in order: dedup (§7.5) → resolver coverage (§7.1) → gradeability / Path B
(§7.2) → detection rebalance to ~25/25/25/25 (§7.3) → freeze. Verifies the plan's
frozen numbers (coverage 6/23/78/72, 974 gradeable) and writes
`artifacts/dataset_v1/` (`data.json`, `manifest.json`, `coverage.json`,
`gradeability.json`). The original `data.json` is never mutated.

Optional subtype tags (§7.4) — proposed by a transparent rule tagger, applied
**only after human review**:

```bash
python scripts/tag_subtypes.py propose --out artifacts/subtype_review.csv
# edit the 'reviewed_subtype' column, then:
python scripts/run_gates.py --reviewed-subtypes artifacts/subtype_review.csv
```

### 2. Build the retrieval index (`§6`)

```bash
python scripts/build_index.py                 # multilingual-e5-base by default
```

### 3. Pilot go/no-go (`§9.2`)

```bash
BACKEND=hf MODEL=internvl3-8b bash scripts/run_pilot.sh
```

Samples a stratified 200-item law set, runs E1–E4, and checks the leakage
signature (E3 acc > E2, E3 F1 ≈ E2, E4 F1 > E3). **Stop and debug E3 if the
signature is absent.**

### 4. Main sweep + ablations + tables (`§9.3–§9.6`)

```bash
python scripts/gen_configs.py                 # write the full run matrix → configs/
BACKEND=hf bash scripts/run_main_sweep.sh     # E1/E4 full ladder + E2/E3 subset
BACKEND=hf bash scripts/run_ablations.sh      # A1/A2/A4 + CircularEval, then tables
python scripts/make_tables.py                 # T1–T6 + bootstrap CIs + McNemar
```

Tables land in `paper_tables/` (markdown + LaTeX) with `stats.json`.

## Backends

Set the backend per run (`--backend` on `run_condition.py`, or `BACKEND=` for the
shell wrappers):

- `hf` (default) — local HuggingFace Transformers; per-family adapters for
  InternVL3, Qwen2.5-VL, Llama-3.2-Vision.
- `openai` — OpenAI-compatible server; point `base_url` at `vllm serve` for
  ~5–10× faster sweeps (E4 is tens of thousands of calls).
- `mock` — deterministic, dependency-free; for dry runs.

The retriever is selectable too: `retriever: faiss` (default, dense e5) or
`retriever: lexical` (pure-Python TF-IDF; a zero-setup smoke-test + lexical
baseline).

### Dry run (no GPU, no downloads)

```bash
python scripts/run_gates.py
python scripts/run_condition.py \
  --config <(python - <<'PY'
from vntraffic.config import RunConfig
print(RunConfig(run_id="dry__E4", condition="E4", model_key="internvl3-8b",
                backend="mock", retriever="lexical",
                data="artifacts/dataset_v1/data.json", limit=5).to_dict())
PY
) --backend mock
```

(Or construct a `RunConfig(..., backend="mock", retriever="lexical")` in Python
and call `vntraffic.runner.run`.)

## QCVN 41:2024 ingestion (Path A, `§7.2`)

The shipped corpus resolves 974/1,319 law items (Path B). To close the gap when
you have the official QCVN text:

```bash
python scripts/qcvn_ingest.py skeleton --out artifacts/qcvn_skeleton.yaml
# fill the 'content' fields, then:
python scripts/qcvn_ingest.py merge --skeleton artifacts/qcvn_skeleton.yaml \
    --out-corpus artifacts/legal_traffic_chunks_v2.json
```

Merging grows the corpus + an alias map, changes the corpus version (invalidating
caches), and flips previously-ungradeable items to gradeable — picked up
automatically the next time `run_gates.py` runs.

## Layout

```
vntraffic/          package: dataset, corpus, resolver, gradeability, retriever,
                    embedder, models/ (hf|openai|mock), prompts, agent, conditions,
                    runner, metrics, circular, stats, tables, gates, subtags, qcvn_ingest
scripts/            run_gates, build_index, make_pilot_sample, gen_configs,
                    run_condition, check_pilot_gonogo, make_tables, tag_subtypes,
                    qcvn_ingest, run_{pilot,main_sweep,ablations}.sh
data/               images (data/<category>/<n>.jpg)
data.json           QA items
legal_traffic_chunks.json   statute corpus
```

## Reproducibility

Fixed seeds; greedy decoding (temperature 0); one config file per
model × condition (no manual prompt edits between runs); per-item JSONL logs +
run manifests (model id, corpus version, resolver coverage, decoding params);
SHA-keyed retrieval cache. Generated outputs (`artifacts/`, `configs/`, `runs/`,
`cache/`, `paper_tables/`) are git-ignored and regenerate deterministically from
source.
