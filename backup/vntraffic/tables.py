"""Paper tables T1-T6 from run outputs (EXPERIMENT_PLAN.md §10).

Reads every run under ``runs/`` (each has manifest.json + results.jsonl), joins
with the frozen dataset + gradeability id sets, recomputes metrics, and emits
markdown + LaTeX tables:

  T1  E1 accuracy, detection vs law, full ladder + random/majority + CircularEval
  T2  E1-E4 law accuracy + grounding F1 (leakage subset models)
  T3  leakage decomposition 2x2 (E2/E3/E4) — headline
  T4  grounded accuracy across the ladder, E4 vs E1
  T5  A1 budget curve (accuracy & grounding vs B)
  T6  appendix: per-category, per-subtype, abstention

Metrics come from metrics.py; nothing is recomputed by hand.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from . import constants as C
from . import metrics as M
from .corpus import load_corpus
from .dataset import index_by_id, load_items
from .circular import circular_eval


# --------------------------------------------------------------------------- #
# Context (frozen dataset + gradeability)
# --------------------------------------------------------------------------- #
@dataclass
class Context:
    items_by_id: Dict[int, object]
    corpus: object
    gradeable_ids: set
    f1_gradeable_ids: set

    @classmethod
    def load(cls, dataset_dir: str = C.DATASET_V1_DIR,
             corpus_path: str = C.CORPUS_JSON,
             corpus_aliases: Optional[str] = None) -> "Context":
        data_path = os.path.join(dataset_dir, "data.json")
        items = load_items(data_path if os.path.exists(data_path) else C.DATA_JSON)
        corpus = load_corpus(corpus_path, corpus_aliases)
        grade_path = os.path.join(dataset_dir, "gradeability.json")
        gradeable = f1 = set()
        if os.path.exists(grade_path):
            with open(grade_path, "r", encoding="utf-8") as f:
                gd = json.load(f)
            gradeable = set(gd.get("gradeable_ids", []))
            f1 = set(gd.get("f1_gradeable_ids", []))
        return cls(index_by_id(items), corpus, gradeable, f1)


# --------------------------------------------------------------------------- #
# Run discovery
# --------------------------------------------------------------------------- #
@dataclass
class RunView:
    run_dir: str
    manifest: dict
    _records: Optional[List[dict]] = None

    @property
    def run_id(self) -> str:
        return self.manifest.get("run_id", os.path.basename(self.run_dir))

    @property
    def condition(self) -> str:
        return self.manifest.get("condition", "")

    @property
    def tier(self) -> str:
        return self.manifest.get("tier", "")

    @property
    def model_key(self) -> str:
        return self.manifest.get("model", {}).get("model_key", "")

    @property
    def circular(self) -> bool:
        return bool(self.manifest.get("circular"))

    @property
    def variant(self) -> str:
        """Trailing '__variant' token of the run_id (e.g. 'b1', 'k10', 'circular')."""
        parts = self.run_id.split("__")
        return parts[3] if len(parts) >= 4 else ""

    def records(self) -> List[dict]:
        if self._records is None:
            path = os.path.join(self.run_dir, "results.jsonl")
            self._records = M.load_records(path) if os.path.exists(path) else []
        return self._records


def discover_runs(runs_dir: str = C.RUNS_DIR) -> List[RunView]:
    out = []
    if not os.path.isdir(runs_dir):
        return out
    for name in sorted(os.listdir(runs_dir)):
        man = os.path.join(runs_dir, name, "manifest.json")
        if os.path.exists(man):
            with open(man, "r", encoding="utf-8") as f:
                out.append(RunView(os.path.join(runs_dir, name), json.load(f)))
    return out


def _find(runs: Sequence[RunView], tier: str, condition: str, model: str,
          variant: str = "", circular: bool = False) -> Optional[RunView]:
    for r in runs:
        if (r.tier == tier and r.condition == condition and r.model_key == model
                and r.variant == variant and r.circular == circular):
            return r
    return None


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join([line, sep, body])


def _latex_table(headers: Sequence[str], rows: Sequence[Sequence[str]],
                 caption: str, label: str) -> str:
    cols = "l" + "r" * (len(headers) - 1)
    head = " & ".join(headers) + r" \\"
    body = " \\\\\n".join(" & ".join(str(c) for c in row) for row in rows) + r" \\"
    return ("\\begin{table}[t]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}")


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def table_t1(runs: Sequence[RunView], ctx: Context,
             ladder: Sequence[str]) -> Tuple[str, str]:
    headers = ["Model", "Det acc", "Det bal-acc", "Det circ",
               "Law acc", "Law bal-acc", "Law circ"]
    rows = []
    for m in ladder:
        det = _find(runs, C.TIER_DETECTION, C.COND_E1, m)
        law = _find(runs, C.TIER_LAW, C.COND_E1, m)
        det_c = _find(runs, C.TIER_DETECTION, C.COND_E1, m, "circular", True)
        law_c = _find(runs, C.TIER_LAW, C.COND_E1, m, "circular", True)
        det_a = M.accuracy(det.records()) if det else None
        law_a = M.accuracy(law.records()) if law else None
        rows.append([
            m,
            _pct(det_a.accuracy) if det_a else "—",
            _pct(det_a.balanced_accuracy) if det_a else "—",
            _pct(circular_eval(det_c.records()).circular_accuracy) if det_c else "—",
            _pct(law_a.accuracy) if law_a else "—",
            _pct(law_a.balanced_accuracy) if law_a else "—",
            _pct(circular_eval(law_c.records()).circular_accuracy) if law_c else "—",
        ])
    # baseline rows
    rows.append(["_random_", _pct(C.RANDOM_BASELINE), "—", "—",
                 _pct(C.RANDOM_BASELINE), "—", "—"])
    rows.append([f"_majority ({C.MAJORITY_LABEL[C.TIER_DETECTION]}/"
                 f"{C.MAJORITY_LABEL[C.TIER_LAW]})_",
                 _pct(C.MAJORITY_BASELINE[C.TIER_DETECTION]), "—", "—",
                 _pct(C.MAJORITY_BASELINE[C.TIER_LAW]), "—", "—"])
    cap = ("T1: E1 accuracy by tier across the model ladder, with random/majority "
           "baselines and CircularEval. Detection accuracy sits far above its 50% "
           "prior; law accuracy sits on its 29% prior (Claim 1).")
    return _md_table(headers, rows), _latex_table(headers, rows, cap, "tab:t1")


def table_t2(runs: Sequence[RunView], ctx: Context,
             models: Sequence[str]) -> Tuple[str, str]:
    headers = ["Model", "Cond", "Acc", "Grounding F1", "n(F1)"]
    rows = []
    for m in models:
        for cond in C.CONDITIONS:
            rv = _find(runs, C.TIER_LAW, cond, m)
            if rv is None:
                continue
            acc = M.accuracy(rv.records())
            if cond == C.COND_E1:
                f1s, n = "—", "—"
            else:
                g = M.grounding(rv.records(), ctx.items_by_id, ctx.corpus,
                                ctx.f1_gradeable_ids)
                f1s, n = _pct(g.f1), g.n
            rows.append([m, cond, _pct(acc.accuracy), f1s, n])
    cap = ("T2: Law-tier answer accuracy and citation-grounding F1 across E1-E4 "
           "for the leakage-subset models (Claims 2, 3).")
    return _md_table(headers, rows), _latex_table(headers, rows, cap, "tab:t2")


def table_t3(runs: Sequence[RunView], ctx: Context,
             models: Sequence[str]) -> Tuple[str, str]:
    headers = ["Model", "Cond", "✅grounded", "⚠️leaked", "🔎mis", "❌wrong",
               "Grounded-acc", "Ans-acc"]
    rows = []
    for m in models:
        for cond in (C.COND_E2, C.COND_E3, C.COND_E4):
            rv = _find(runs, C.TIER_LAW, cond, m)
            if rv is None:
                continue
            lk = M.leakage_decomposition(rv.records(), ctx.items_by_id, ctx.corpus,
                                         ctx.f1_gradeable_ids)
            rows.append([m, cond, lk.grounded_correct, lk.leaked_correct,
                         lk.grounded_misselect, lk.fully_wrong,
                         _pct(lk.grounded_accuracy), _pct(lk.answer_accuracy)])
    cap = ("T3 (headline): leakage decomposition. Claim 3 predicts E3-E2 accuracy "
           "gain concentrates in the leaked cell; Claim 4 predicts E4 dominates the "
           "grounded cell.")
    return _md_table(headers, rows), _latex_table(headers, rows, cap, "tab:t3")


def table_t4(runs: Sequence[RunView], ctx: Context,
             ladder: Sequence[str]) -> Tuple[str, str]:
    headers = ["Model", "E1 acc", "E4 acc", "E4 grounded-acc", "E4 grounding-F1"]
    rows = []
    for m in ladder:
        e1 = _find(runs, C.TIER_LAW, C.COND_E1, m)
        e4 = _find(runs, C.TIER_LAW, C.COND_E4, m)
        e1a = _pct(M.accuracy(e1.records()).accuracy) if e1 else "—"
        if e4:
            e4a = M.accuracy(e4.records()).accuracy
            lk = M.leakage_decomposition(e4.records(), ctx.items_by_id, ctx.corpus,
                                         ctx.f1_gradeable_ids)
            g = M.grounding(e4.records(), ctx.items_by_id, ctx.corpus,
                            ctx.f1_gradeable_ids)
            rows.append([m, e1a, _pct(e4a), _pct(lk.grounded_accuracy), _pct(g.f1)])
        else:
            rows.append([m, e1a, "—", "—", "—"])
    cap = ("T4: grounded accuracy across the ladder, E4 vs E1 (Claim 4).")
    return _md_table(headers, rows), _latex_table(headers, rows, cap, "tab:t4")


def table_t5(runs: Sequence[RunView], ctx: Context, best_model: str) -> Tuple[str, str]:
    headers = ["Budget B", "Acc", "Grounded-acc", "Grounding F1", "mean rounds"]
    rows = []
    # B=1,5 come from variant runs; B=3 is the main E4 run
    variant_for_b = {1: "b1", 3: "", 5: "b5"}
    for b in (1, 3, 5):
        rv = _find(runs, C.TIER_LAW, C.COND_E4, best_model, variant_for_b[b])
        if rv is None:
            rows.append([b, "—", "—", "—", "—"])
            continue
        recs = rv.records()
        acc = M.accuracy(recs)
        lk = M.leakage_decomposition(recs, ctx.items_by_id, ctx.corpus, ctx.f1_gradeable_ids)
        g = M.grounding(recs, ctx.items_by_id, ctx.corpus, ctx.f1_gradeable_ids)
        rounds = [r.get("budget_used") for r in M.only_identity(recs)
                  if r.get("budget_used") is not None]
        mean_rounds = round(sum(rounds) / len(rounds), 2) if rounds else "—"
        rows.append([b, _pct(acc.accuracy), _pct(lk.grounded_accuracy),
                     _pct(g.f1), mean_rounds])
    cap = ("T5: agent budget ablation A1 on the best model (loop earns its compute).")
    return _md_table(headers, rows), _latex_table(headers, rows, cap, "tab:t5")


def table_t6_categories(runs: Sequence[RunView], ctx: Context,
                        model: str, condition: str = C.COND_E4) -> Tuple[str, str]:
    rv = _find(runs, C.TIER_LAW, condition, model)
    headers = ["Category", "n", "Acc"]
    rows = []
    if rv is not None:
        recs = M.only_identity(rv.records())
        by_cat: Dict[str, List[dict]] = {}
        for r in recs:
            by_cat.setdefault(r.get("category", "?"), []).append(r)
        for cat in sorted(by_cat):
            a = M.accuracy(by_cat[cat])
            rows.append([cat, a.n, _pct(a.accuracy)])
    cap = f"T6a: per-category accuracy ({condition}, {model})."
    return _md_table(headers, rows), _latex_table(headers, rows, cap, "tab:t6cat")


def table_t6_abstention(runs: Sequence[RunView], ctx: Context,
                        models: Sequence[str]) -> Tuple[str, str]:
    headers = ["Model", "Cond", "Category", "Abstain@GT", "False-violation"]
    rows = []
    for m in models:
        for cond in (C.COND_E1, C.COND_E4):
            rv = _find(runs, C.TIER_LAW, cond, m)
            if rv is None:
                continue
            for cat in C.CALIBRATION_CATEGORIES:
                ab = M.abstention(rv.records(), ctx.items_by_id, cat)
                rows.append([m, cond, cat, _pct(ab.abstain_when_gt),
                             _pct(ab.false_violation_rate)])
    cap = "T6b: abstention / false-violation on no-violation & indeterminate images (§5.4)."
    return _md_table(headers, rows), _latex_table(headers, rows, cap, "tab:t6abs")


def table_t6_subtypes(runs: Sequence[RunView], ctx: Context,
                      model: str, condition: str = C.COND_E4) -> Optional[Tuple[str, str]]:
    """Per-subtype accuracy — only if the dataset carries law_subtype tags."""
    rv = _find(runs, C.TIER_LAW, condition, model)
    if rv is None:
        return None
    recs = M.only_identity(rv.records())
    by_sub: Dict[str, List[dict]] = {}
    for r in recs:
        item = ctx.items_by_id.get(r["id"])
        sub = getattr(item, "law_subtype", None) if item else None
        if sub:
            by_sub.setdefault(sub, []).append(r)
    if not by_sub:
        return None  # tags absent → drop the subtype table (per §7.4)
    headers = ["Subtype", "n", "Acc"]
    rows = [[s, M.accuracy(by_sub[s]).n, _pct(M.accuracy(by_sub[s]).accuracy)]
            for s in sorted(by_sub)]
    cap = f"T6c: per-subtype accuracy ({condition}, {model})."
    return _md_table(headers, rows), _latex_table(headers, rows, cap, "tab:t6sub")
