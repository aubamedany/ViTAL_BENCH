"""Frozen constants for the VNTrafficVQA experiment (EXPERIMENT_PLAN.md §2, §4).

Anything that must stay identical across runs lives here so that no headline
number depends on a magic value buried in a script.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Paths (relative to repository root; overridable via env for CI/alt-checkouts)
# --------------------------------------------------------------------------- #
REPO_ROOT = os.environ.get(
    "VNTRAFFIC_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DATA_JSON = os.path.join(REPO_ROOT, "data.json")
CORPUS_JSON = os.path.join(REPO_ROOT, "legal_traffic_chunks.json")
IMAGE_ROOT = REPO_ROOT  # image paths in data.json are "/data/..." → repo-root relative
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts")
DATASET_V1_DIR = os.path.join(ARTIFACTS_DIR, "dataset_v1")
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
CACHE_DIR = os.path.join(REPO_ROOT, "cache")
INDEX_DIR = os.path.join(ARTIFACTS_DIR, "index")
CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")
TABLES_DIR = os.path.join(REPO_ROOT, "paper_tables")

# --------------------------------------------------------------------------- #
# Tiers, options, categories
# --------------------------------------------------------------------------- #
TIER_DETECTION = "detection"
TIER_LAW = "law"
TIERS = (TIER_DETECTION, TIER_LAW)

OPTION_LETTERS = ("A", "B", "C", "D")

# The 11 image categories (== image folder names == type_image values).
CATEGORIES = (
    "cong_kenh",
    "di_vao_duong_cam",
    "du_lieu_khong_loi",
    "dung_qua_vach",
    "khong_mu_bao_hiem",
    "khong_the_xac_dinh",
    "khong_tuan_thu_den_tin_hieu",
    "nguoc_chieu",
    "sai_bien_bao",
    "sai_lan",
    "tong_hop",
)

# Calibration categories used for abstention / false-violation analysis (§5.4).
NO_VIOLATION_CATEGORY = "du_lieu_khong_loi"
INDETERMINATE_CATEGORY = "khong_the_xac_dinh"
CALIBRATION_CATEGORIES = (NO_VIOLATION_CATEGORY, INDETERMINATE_CATEGORY)

# --------------------------------------------------------------------------- #
# Answer-key priors (mandatory baselines, EXPERIMENT_PLAN.md §2)
# These are frozen expectations from the *shipped* data.json. run_gates.py
# recomputes the live values and warns if they drift.
# --------------------------------------------------------------------------- #
RANDOM_BASELINE = 0.25  # 4-way MCQ

# majority-label baseline per tier (fraction of the most common GT letter)
MAJORITY_BASELINE = {
    TIER_DETECTION: 0.50,  # B-skew on the shipped (un-rebalanced) detection tier
    TIER_LAW: 0.29,        # C is the modal law answer
}
MAJORITY_LABEL = {
    TIER_DETECTION: "B",
    TIER_LAW: "C",
}

# --------------------------------------------------------------------------- #
# Expected dataset shape (used by run_gates.py sanity checks; NOT load-bearing)
# --------------------------------------------------------------------------- #
EXPECTED_TOTAL_ITEMS = 2475
EXPECTED_DETECTION_ITEMS = 1156
EXPECTED_LAW_ITEMS = 1319
EXPECTED_UNIQUE_GOLD_CITATIONS = 179
EXPECTED_GRADEABLE_LAW_ITEMS = 974  # Path B pre-registered subset (current DB + resolver)
EXPECTED_CORPUS_CHUNKS = 777

# Resolver coverage over the 179 distinct gold citations (EXPERIMENT_PLAN.md §7.1).
EXPECTED_COVERAGE = {
    "exact": 6,
    "case_only": 23,
    "point_fallback": 78,
    "unresolved": 72,
}

# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
GLOBAL_SEED = 20260706        # dataset-freeze / rebalance / sampling seed
DEFAULT_DECODING_SEED = 0     # model decoding seed (greedy anyway, but logged)

# --------------------------------------------------------------------------- #
# Retrieval defaults (EXPERIMENT_PLAN.md §6)
# --------------------------------------------------------------------------- #
DEFAULT_EMBEDDER = "intfloat/multilingual-e5-base"
DEFAULT_TOP_K = 5
DEFAULT_AGENT_BUDGET = 3      # E4 REFINE budget B

# --------------------------------------------------------------------------- #
# Conditions (EXPERIMENT_PLAN.md §3)
# --------------------------------------------------------------------------- #
COND_E1 = "E1"  # parametric-only
COND_E2 = "E2"  # RAG-Q  (question-only retrieval, options hidden)
COND_E3 = "E3"  # RAG-QO (question+options retrieval — leakage probe)
COND_E4 = "E4"  # VNTraffic-Agent (option-blind two-phase)
CONDITIONS = (COND_E1, COND_E2, COND_E3, COND_E4)
RETRIEVAL_CONDITIONS = (COND_E2, COND_E3, COND_E4)  # conditions that use the retriever
