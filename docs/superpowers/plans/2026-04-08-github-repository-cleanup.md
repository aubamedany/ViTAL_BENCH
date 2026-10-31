# GitHub Repository Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use supo-subagent-driven-development (recommended) or supo-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean the public ViTALBench repository, group JSON resources under `datasets/`, update documentation and paths, and exclude agent-related files.

**Architecture:** Keep released images under the protected `data/` directory.
Move question datasets to `datasets/questions/` and legal resources to `datasets/legal/`.
Update consumers to resolve these repository-relative paths explicitly.

**Tech Stack:** Python, JSON, Markdown, Git, pytest.

## Global Constraints

- Use ViTALBench consistently.
- Do not modify anything under `data/` or `checkpoints/`.
- Do not track or publish any `AGENTS.md` file, `.idea/`, or `backup/` directory.
- Do not mention or include the agent pipeline.
- Do not edit `CHANGELOG.md` or generated files.
- Preserve dataset contents while moving only the specified root JSON files.
- Do not push until validation is complete and credentials are available.

---

### Task 1: Move JSON resources into the public dataset layout

**Files:**
- Move: `detection.json` -> `datasets/questions/detection.json`
- Move: `detection_e.json` -> `datasets/questions/detection_e.json`
- Move: `law.json` -> `datasets/questions/law.json`
- Move: `law_e.json` -> `datasets/questions/law_e.json`
- Move: `law_categories.json` -> `datasets/legal/law_categories.json`
- Move: `law_db.json` -> `datasets/legal/law_db.json`

**Interfaces:**
- Produces the canonical paths consumed by all later tasks.

- [ ] **Step 1: Create destination directories.**

```bash
mkdir -p datasets/questions datasets/legal
```

- [ ] **Step 2: Move files with Git-aware renames.**

```bash
git mv detection.json datasets/questions/detection.json
git mv detection_e.json datasets/questions/detection_e.json
git mv law.json datasets/questions/law.json
git mv law_e.json datasets/questions/law_e.json
git mv law_categories.json datasets/legal/law_categories.json
git mv law_db.json datasets/legal/law_db.json
```

- [ ] **Step 3: Confirm only the six intended JSON paths changed.**

```bash
git status --short
find datasets -maxdepth 2 -type f -name '*.json' -print | sort
```

Expected: six JSON files under `datasets/questions/` and `datasets/legal/`, with no root copies.

- [ ] **Step 4: Commit the file layout change.**

```bash
git add datasets
git commit -m "chore: group release JSON resources"
```

---

### Task 2: Update code and tests for the new paths

**Files:**
- Modify: `annotate_law.py`
- Modify: `scripts/check_release_stats.py`
- Modify: `scripts/plot_dataset_statistics.py`
- Modify: `scripts/clean_qcvn_html.py`
- Modify: `utils/make_e_dataset.py`
- Modify: `notebooks/draw_fig3.ipynb`
- Modify: `tests/test_release_stats.py`

**Interfaces:**
- Consumers continue to load the same JSON structures from `datasets/questions/` and `datasets/legal/`.
- The Fig. 3 notebook continues to write only the existing paper figure outputs.

- [ ] **Step 1: Find every stale root JSON reference.**

```bash
rg -n "(detection|law)(_e)?\\.json|law_categories\\.json|law_db\\.json" --glob '!docs/superpowers/**' .
```

- [ ] **Step 2: Update canonical paths.**

Use these exact replacements:

```text
ROOT / "detection.json" -> ROOT / "datasets/questions/detection.json"
ROOT / "law.json" -> ROOT / "datasets/questions/law.json"
ROOT / "law_categories.json" -> ROOT / "datasets/legal/law_categories.json"
ROOT / "law_db.json" -> ROOT / "datasets/legal/law_db.json"
```

Update command-line defaults in `utils/make_e_dataset.py` to:

```text
--in datasets/questions/law.json
--out datasets/questions/law_e.json
```

Update the notebook’s JSON input paths to the same canonical paths while preserving the user’s editable category labels and figure styling.

- [ ] **Step 3: Update tests to use the new paths.**

The release-statistics test must read the canonical files from `datasets/questions/` and `datasets/legal/` rather than the repository root.

- [ ] **Step 4: Search again for stale paths.**

```bash
rg -n "(^|[\\\"/ ])(detection|law)(_e)?\\.json|law_categories\\.json|law_db\\.json" --glob '!docs/superpowers/**' .
```

Expected: only intentional documentation strings with the new `datasets/` paths remain.

- [ ] **Step 5: Run focused validation.**

```bash
python scripts/check_release_stats.py
PYTHONPATH=. pytest -q tests/test_release_stats.py
```

- [ ] **Step 6: Commit path updates.**

```bash
git add annotate_law.py scripts utils tests notebooks/draw_fig3.ipynb
git commit -m "fix: update consumers for grouped JSON resources"
```

---

### Task 3: Clean ignore rules and public documentation

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**
- The published Fig. 3 notebook remains force-tracked despite the broad local-notebook ignore rule.
- README paths match the canonical `datasets/` layout.

- [ ] **Step 1: Add explicit forbidden-path rules.**

Ensure `.gitignore` includes rules covering:

```gitignore
AGENTS.md
**/AGENTS.md
.idea/
**/.idea/
backup/
**/backup/
```

Keep the existing `notebooks/` ignore rule if desired, because the published notebook is already tracked explicitly.

- [ ] **Step 2: Update README canonical paths and layout.**

Replace root JSON references with:

```text
datasets/questions/detection.json
datasets/questions/law.json
datasets/legal/law_categories.json
datasets/legal/law_db.json
```

Document `detection_e.json` and `law_e.json` under `datasets/questions/`.

Keep the authoritative release statistics unchanged.

- [ ] **Step 3: Check README and ignore rules for forbidden terminology.**

```bash
rg -ni "agent|backup|\\.idea" README.md .gitignore
```

Expected: `.gitignore` may contain exclusion rules, while README contains no agent-pipeline discussion.

- [ ] **Step 4: Commit documentation and ignore updates.**

```bash
git add .gitignore README.md
git commit -m "docs: clean public repository guidance"
```

---

### Task 4: Verify the public release tree

**Files:**
- Verify: all tracked files
- Verify: `datasets/questions/`
- Verify: `datasets/legal/`

- [ ] **Step 1: Confirm forbidden paths are not tracked.**

```bash
git ls-files | grep -Ei '(^|/)(AGENTS\\.md|\\.idea|backup)(/|$)'
```

Expected: no output.

- [ ] **Step 2: Confirm no root JSON files remain.**

```bash
find . -maxdepth 1 -type f -name '*.json' -print
```

Expected: no output.

- [ ] **Step 3: Run release validation.**

```bash
python scripts/check_release_stats.py
PYTHONPATH=. pytest -q
```

Expected: the audit passes and the full test suite passes.

- [ ] **Step 4: Review the final diff and status.**

```bash
git diff origin/main...HEAD --stat
git status --short --branch
git log --oneline -6
```

Expected: only the cleanup commits and the earlier local commits are ahead, with no untracked or unintended files.

- [ ] **Step 5: Do not push automatically.**

Push only after explicit approval and valid GitHub authentication are available.
