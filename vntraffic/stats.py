"""Statistical reporting (EXPERIMENT_PLAN.md §5.6).

* Bootstrap 95% CIs (10k resamples) on headline accuracies / grounded accuracy.
* Paired McNemar tests for E-condition comparisons on identical items
  (E3 vs E4 answer accuracy; E4 vs E2 grounded accuracy).

Pure-Python (no numpy/scipy) so it runs anywhere. McNemar uses the exact
binomial test; given K2's exploratory-adjacent nature the plan asks for large
effects surviving p < .001, so an exact test is preferred over the χ² approx.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from . import constants as C


@dataclass
class CI:
    mean: float
    lo: float
    hi: float
    n: int

    def to_dict(self) -> dict:
        return {"mean": round(self.mean, 4), "lo": round(self.lo, 4),
                "hi": round(self.hi, 4), "n": self.n}


def bootstrap_ci(values: Sequence[float], n_resamples: int = 10000,
                 alpha: float = 0.05, seed: int = C.GLOBAL_SEED) -> CI:
    """Percentile bootstrap CI for the mean of *values* (0/1 or floats)."""
    n = len(values)
    if n == 0:
        return CI(0.0, 0.0, 0.0, 0)
    rng = random.Random(seed)
    vals = list(values)
    total = float(sum(vals))
    means = []
    for _ in range(n_resamples):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return CI(mean=total / n, lo=lo, hi=hi, n=n)


# --------------------------------------------------------------------------- #
# Paired McNemar
# --------------------------------------------------------------------------- #
@dataclass
class McNemarResult:
    n_pairs: int
    b: int          # A correct, B wrong
    c: int          # A wrong, B correct
    p_value: float
    acc_a: float
    acc_b: float

    def to_dict(self) -> dict:
        return {"n_pairs": self.n_pairs, "b": self.b, "c": self.c,
                "p_value": self.p_value, "acc_a": round(self.acc_a, 4),
                "acc_b": round(self.acc_b, 4)}


def _binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided binomial p for McNemar with n=b+c, p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided: 2 * P(X <= k) capped at 1
    cdf = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * cdf)


def paired_from_ids(ids_a: Sequence[int], vals_a: Sequence[int],
                    ids_b: Sequence[int], vals_b: Sequence[int]
                    ) -> Tuple[List[int], List[int]]:
    """Align two (ids, values) series on common ids → (paired_a, paired_b)."""
    ma = dict(zip(ids_a, vals_a))
    mb = dict(zip(ids_b, vals_b))
    common = sorted(set(ma) & set(mb))
    return [ma[i] for i in common], [mb[i] for i in common]


def mcnemar(paired_a: Sequence[int], paired_b: Sequence[int]) -> McNemarResult:
    assert len(paired_a) == len(paired_b)
    b = sum(1 for a, bb in zip(paired_a, paired_b) if a and not bb)
    c = sum(1 for a, bb in zip(paired_a, paired_b) if (not a) and bb)
    n = len(paired_a)
    return McNemarResult(
        n_pairs=n, b=b, c=c, p_value=_binom_two_sided(b, c),
        acc_a=(sum(paired_a) / n if n else 0.0),
        acc_b=(sum(paired_b) / n if n else 0.0),
    )


def mcnemar_by_ids(ids_a, vals_a, ids_b, vals_b) -> McNemarResult:
    pa, pb = paired_from_ids(ids_a, vals_a, ids_b, vals_b)
    return mcnemar(pa, pb)
