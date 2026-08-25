"""Retrieval quality metrics, defined precisely enough to be hand-checked.

Every function here is exactly testable against a fixture computed by hand,
and the tests do exactly that. This matters more than it sounds: "recall@k"
is used in the literature for at least three different quantities, and a repo
that reports a recall curve without saying which one it means has not reported
anything. The definitions used here:

- :func:`recall_at_k` -- ANN recall. The fraction of the *true* top-k that the
  approximate index returned. The reference set is itself of size k, so this
  is the standard ANN-benchmark recall@k, and it is 1.0 for an exact index by
  construction.
- :func:`mrr` -- reciprocal rank of the first relevant item, averaged.
- :func:`ndcg_at_k` -- binary-gain NDCG with the standard log2(i+1) discount.

All three take *ranked id lists*, never scores, so they cannot be accidentally
sensitive to a score scale -- which is precisely the bug :mod:`.fusion`
demonstrates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[int], truth: Sequence[int], k: int) -> float:
    """Fraction of the true top-k that appears in the retrieved top-k.

    Both lists are truncated to ``k`` before comparison. Truncating the truth
    list too is what makes this ANN recall rather than "recall against an
    arbitrarily large relevant set" -- with a k-sized reference, a perfect
    index scores exactly 1.0.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    reference = {int(x) for x in truth[:k]}
    if not reference:
        raise ValueError("truth set is empty; recall is undefined")
    hits = sum(1 for doc in retrieved[:k] if int(doc) in reference)
    return hits / len(reference)


def mean_recall_at_k(
    retrieved: Sequence[Sequence[int]], truth: Sequence[Sequence[int]], k: int
) -> float:
    """Mean of :func:`recall_at_k` over a query set."""
    if len(retrieved) != len(truth):
        raise ValueError(f"query counts differ: {len(retrieved)} vs {len(truth)}")
    if len(retrieved) == 0:
        raise ValueError("cannot average over zero queries")
    return sum(
        recall_at_k(r, t, k) for r, t in zip(retrieved, truth, strict=True)
    ) / len(retrieved)


def per_query_recall(
    retrieved: Sequence[Sequence[int]], truth: Sequence[Sequence[int]], k: int
) -> list[float]:
    """Per-query recall, kept unaggregated so it can be fed to a bootstrap CI.

    Averaging first and bootstrapping the average is not possible; the CI needs
    the sample. Returning the vector rather than the mean is what lets the
    results script attach an interval to every recall number it reports.
    """
    if len(retrieved) != len(truth):
        raise ValueError(f"query counts differ: {len(retrieved)} vs {len(truth)}")
    return [recall_at_k(r, t, k) for r, t in zip(retrieved, truth, strict=True)]


def reciprocal_rank(retrieved: Sequence[int], relevant: Sequence[int]) -> float:
    """1/rank of the first relevant item, 0.0 if none is present."""
    relevant_set = set(relevant)
    for position, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            return 1.0 / position
    return 0.0


def mrr(retrieved: Sequence[Sequence[int]], relevant: Sequence[Sequence[int]]) -> float:
    """Mean reciprocal rank."""
    if len(retrieved) != len(relevant):
        raise ValueError(f"query counts differ: {len(retrieved)} vs {len(relevant)}")
    if len(retrieved) == 0:
        raise ValueError("cannot average over zero queries")
    return sum(
        reciprocal_rank(r, g) for r, g in zip(retrieved, relevant, strict=True)
    ) / len(retrieved)


def ndcg_at_k(retrieved: Sequence[int], relevant: Sequence[int], k: int) -> float:
    """Binary-gain NDCG@k with the log2(i+1) discount.

    The ideal ranking places ``min(k, len(relevant))`` relevant items first, so
    a query with fewer relevant documents than ``k`` can still score 1.0.
    Normalising against a full-k ideal instead would cap such queries below 1
    and quietly penalise the retriever for the labelling, not the ranking.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("relevant set is empty; NDCG is undefined")

    gain = sum(
        1.0 / math.log2(position + 1)
        for position, doc in enumerate(retrieved[:k], start=1)
        if doc in relevant_set
    )
    ideal = sum(
        1.0 / math.log2(position + 1)
        for position in range(1, min(k, len(relevant_set)) + 1)
    )
    return gain / ideal
