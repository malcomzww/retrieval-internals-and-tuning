"""The filtered-ANN problem: why metadata filters break vector search.

The failure is easy to state and easy to miss. You ask for the 10 nearest
neighbours *that match a filter*. The obvious implementation asks the index for
10 neighbours and then drops the ones that fail the filter. When the filter is
broad this is fine. When the filter is selective it is catastrophic: if 2% of
the corpus matches, then of 10 unfiltered neighbours roughly 0.2 survive, and
the caller receives an almost-empty result set for a query that has thousands
of perfectly good answers.

What makes this genuinely under-appreciated is that it does not fail loudly.
The system returns *some* results, ranked plausibly, with no error. Recall has
collapsed to a fraction of what the same index delivers unfiltered, and nothing
in the response says so. Teams discover it when a user asks why filtering by a
rare category returns three documents.

Three strategies are implemented and measured:

- :func:`post_filter` -- retrieve k, then filter. The naive approach.
- :func:`overfetch_filter` -- retrieve k x multiplier, filter, truncate to k.
  The usual mitigation. It buys headroom, not a fix: the required multiplier
  scales as 1/selectivity, so at 1% selectivity you fetch 100x to expect k
  survivors, and the latency saving that motivated the ANN index is gone.
- :func:`pre_filter_exact` -- restrict the candidate set first, then scan it
  exactly. Always recall 1.0, and on a selective filter it is also *fast*,
  because the surviving subset is tiny. This is the strategy that wins where
  intuition says it should lose.

The last point is the practical one, and it is why the ADR on Postgres exists:
at high selectivity, a filtered exact scan over the matching rows beats a
filtered ANN search on both recall and latency, and a plain relational database
does that natively.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .corpus import Document
from .index import FlatIndex, HnswIndex, single_threaded
from .metrics import mean_recall_at_k, per_query_recall

Predicate = Callable[[Document], bool]


def matching_ids(docs: Sequence[Document], predicate: Predicate) -> np.ndarray:
    """Row ids passing the predicate. Relies on ``doc_id == row index``."""
    return np.array([d.doc_id for d in docs if predicate(d)], dtype=np.int64)


def selectivity(docs: Sequence[Document], predicate: Predicate) -> float:
    """Fraction of the corpus the filter keeps.

    Named the way the database literature names it: *lower* means *more*
    selective, i.e. fewer surviving rows. The collapse this module measures is
    a function of this number and nothing else.
    """
    if not docs:
        raise ValueError("cannot compute selectivity of an empty corpus")
    return len(matching_ids(docs, predicate)) / len(docs)


def filtered_ground_truth(
    doc_vectors: np.ndarray, query_vectors: np.ndarray, allowed: np.ndarray, k: int
) -> np.ndarray:
    """Exact top-k *within the filtered subset*.

    This is the only defensible ground truth for a filtered query. Measuring
    filtered retrieval against the *unfiltered* top-k would score a correct
    filtered result as a failure, since the unfiltered neighbours mostly do not
    satisfy the filter. Getting this reference wrong is the second-most common
    error in this area, after the collapse itself.
    """
    if allowed.size == 0:
        raise ValueError("filter matches no documents; filtered top-k is undefined")
    subset = np.ascontiguousarray(doc_vectors[allowed])
    local = FlatIndex(subset).search(query_vectors, min(k, subset.shape[0]))
    return allowed[local]


def post_filter(
    hnsw: HnswIndex, query_vectors: np.ndarray, allowed: np.ndarray, k: int
) -> list[list[int]]:
    """Retrieve k from the index, then discard non-matching results.

    Returns short lists when the filter bites -- that shortfall *is* the bug.
    Padding the result to length k would hide the failure behind whatever
    filler was chosen, which is precisely how this ships to production
    unnoticed.
    """
    allowed_set = {int(x) for x in allowed}
    return [
        [int(doc) for doc in row if int(doc) in allowed_set]
        for row in hnsw.search(query_vectors, k)
    ]


def overfetch_filter(
    hnsw: HnswIndex,
    query_vectors: np.ndarray,
    allowed: np.ndarray,
    k: int,
    *,
    multiplier: int = 10,
) -> list[list[int]]:
    """Fetch ``k * multiplier`` candidates, filter, keep the best k.

    Capped at the corpus size, because asking an index for more neighbours than
    it holds is an error in FAISS rather than a clamp.
    """
    if multiplier < 1:
        raise ValueError(f"multiplier must be at least 1, got {multiplier}")
    allowed_set = {int(x) for x in allowed}
    fetch = min(k * multiplier, hnsw._index.ntotal)
    return [
        [int(doc) for doc in row if int(doc) in allowed_set][:k]
        for row in hnsw.search(query_vectors, fetch)
    ]


def pre_filter_exact(
    doc_vectors: np.ndarray, query_vectors: np.ndarray, allowed: np.ndarray, k: int
) -> list[list[int]]:
    """Restrict to matching rows, then scan them exactly.

    Recall is 1.0 by construction -- this computes the same thing
    :func:`filtered_ground_truth` computes. It is listed as a *strategy* rather
    than dismissed as the reference because its cost is what makes it a real
    option: the scan is over the filtered subset, so a filter keeping 1% of the
    corpus makes it roughly 100x cheaper than a full exact scan.
    """
    return [[int(x) for x in row] for row in filtered_ground_truth(
        doc_vectors, query_vectors, allowed, k
    )]


@dataclass(frozen=True)
class FilterResult:
    """Measured outcome of one strategy at one selectivity."""

    strategy: str
    selectivity: float
    recall: float
    per_query: list[float]
    mean_returned: float
    latency_ms: float
    empty_rate: float

    @property
    def fill_rate(self) -> float:
        """Fraction of the requested k actually returned, averaged over queries."""
        return self.mean_returned


def evaluate_strategies(
    docs: Sequence[Document],
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    hnsw: HnswIndex,
    predicate: Predicate,
    *,
    k: int = 10,
    overfetch_multipliers: tuple[int, ...] = (10, 50),
    latency_queries: int = 40,
) -> list[FilterResult]:
    """Measure every strategy against the exact filtered top-k.

    ``empty_rate`` -- the fraction of queries returning *nothing* -- is
    reported alongside recall because it is the number a user actually feels.
    A mean recall of 0.2 sounds survivable; "one query in three returns an
    empty page" does not, and they can be the same measurement.
    """
    import time

    allowed = matching_ids(docs, predicate)
    sel = len(allowed) / len(docs)
    truth = filtered_ground_truth(doc_vectors, query_vectors, allowed, k)
    truth_lists = [[int(x) for x in row] for row in truth]

    def measure(
        strategy: str, run: Callable[[np.ndarray], list[list[int]]]
    ) -> FilterResult:
        retrieved = run(query_vectors)
        with single_threaded():
            sample = query_vectors[:latency_queries]
            run(sample[:1])  # warm-up, excluded from the timing
            start = time.perf_counter()
            for row in range(sample.shape[0]):
                run(sample[row : row + 1])
            latency = (time.perf_counter() - start) / sample.shape[0] * 1000.0
        return FilterResult(
            strategy=strategy,
            selectivity=sel,
            recall=mean_recall_at_k(retrieved, truth_lists, k),
            per_query=per_query_recall(retrieved, truth_lists, k),
            mean_returned=sum(len(r) for r in retrieved) / len(retrieved) / k,
            latency_ms=latency,
            empty_rate=sum(1 for r in retrieved if not r) / len(retrieved),
        )

    results = [
        measure("post-filter", lambda q: post_filter(hnsw, q, allowed, k)),
    ]
    for multiplier in overfetch_multipliers:
        results.append(
            measure(
                f"over-fetch {multiplier}x",
                lambda q, mult=multiplier: overfetch_filter(
                    hnsw, q, allowed, k, multiplier=mult
                ),
            )
        )
    results.append(
        measure("pre-filter exact", lambda q: pre_filter_exact(doc_vectors, q, allowed, k))
    )
    return results


def required_overfetch(selectivity_value: float, k: int) -> int:
    """Candidates needed so that ``k`` survive a filter in expectation.

    ``k / selectivity``, which is the whole argument against over-fetching as a
    general fix: the cost is inversely proportional to selectivity, so the
    filters that need help most are exactly the ones where help is most
    expensive. At 1% selectivity and k=10 this is 1,000 candidates -- and
    fetching 1,000 neighbours from a 20,000-document index is most of an exact
    scan, at which point the ANN index has stopped earning its complexity.

    Expectation only: it says nothing about the variance, and a query whose
    matching documents cluster far from it can still come back short.
    """
    if not 0.0 < selectivity_value <= 1.0:
        raise ValueError(f"selectivity must be in (0,1], got {selectivity_value}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    import math

    return math.ceil(k / selectivity_value)
