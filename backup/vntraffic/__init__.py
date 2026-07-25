"""VNTrafficVQA / VNTraffic-Agent experiment package.

Implements the evaluation programme described in EXPERIMENT_PLAN.md:
data gates, the E1-E4 ablation ladder, the option-blind two-phase agent,
grounding / leakage metrics, and the paper tables.

The package is intentionally dependency-light at import time: heavy optional
dependencies (torch/transformers, sentence-transformers, faiss) are imported
lazily inside the modules that need them, so the data gates and metrics can run
on a CPU-only machine with no model downloads.
"""

__version__ = "1.0.0"
