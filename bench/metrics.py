"""Retrieval metrics, with the confidence intervals the original chart lacked.

With 735 questions a single percentage point is well inside the noise floor,
so every reported figure carries a bootstrap interval. That is the difference
between "89% beats 88%" and "we cannot tell these two apart".
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def rank_of(ranked_ids: list[str], gold_id: str) -> int | None:
    """1-based rank of the gold chunk, or None if it never showed up."""
    try:
        return ranked_ids.index(gold_id) + 1
    except ValueError:
        return None


def recall_at_k(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None, k: int = 10) -> float:
    return 1.0 / rank if rank is not None and rank <= k else 0.0


def ndcg_at_k(rank: int | None, k: int = 10) -> float:
    """Single relevant document, binary relevance: IDCG is 1."""
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def bootstrap_ci(
    values: list[float], *, iterations: int = 2000, alpha: float = 0.05, seed: int = 20261027
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of `values`."""
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iterations):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * iterations)]
    hi = means[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return (lo, hi)


@dataclass
class Score:
    n: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    recall_at_3_ci: tuple[float, float]

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["recall_at_3_ci"] = list(self.recall_at_3_ci)
        return d


def score(ranks: list[int | None]) -> Score:
    n = len(ranks)
    if n == 0:
        return Score(0, 0, 0, 0, 0, 0, 0, (0.0, 0.0))
    r3 = [recall_at_k(r, 3) for r in ranks]
    mean = lambda xs: sum(xs) / n
    return Score(
        n=n,
        recall_at_1=mean([recall_at_k(r, 1) for r in ranks]),
        recall_at_3=mean(r3),
        recall_at_5=mean([recall_at_k(r, 5) for r in ranks]),
        recall_at_10=mean([recall_at_k(r, 10) for r in ranks]),
        mrr_at_10=mean([reciprocal_rank(r) for r in ranks]),
        ndcg_at_10=mean([ndcg_at_k(r) for r in ranks]),
        recall_at_3_ci=bootstrap_ci(r3),
    )


def paired_delta_ci(
    a_ranks: list[int | None], b_ranks: list[int | None], k: int = 3,
    *, iterations: int = 2000, alpha: float = 0.05, seed: int = 20261027
) -> tuple[float, float, float]:
    """Paired bootstrap on recall@k difference (b - a), over the same questions.

    Pairing matters: the two pipelines answered the *same* questions, so the
    paired interval is much tighter than comparing two independent intervals.
    Returns (delta, lo, hi).
    """
    assert len(a_ranks) == len(b_ranks), "paired comparison needs the same question set"
    diffs = [recall_at_k(b, k) - recall_at_k(a, k) for a, b in zip(a_ranks, b_ranks)]
    n = len(diffs)
    if n == 0:
        return (0.0, 0.0, 0.0)
    delta = sum(diffs) / n
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iterations))
    lo = means[int((alpha / 2) * iterations)]
    hi = means[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return (delta, lo, hi)
