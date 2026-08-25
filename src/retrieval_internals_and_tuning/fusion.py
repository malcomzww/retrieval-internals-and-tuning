"""Hybrid fusion, and why naive score addition gets the wrong answer.

Combining a dense retriever with BM25 is standard practice, and the obvious
implementation -- add the scores, maybe with a weight -- is wrong in a way that
is easy to miss because it usually *almost* works.

The problem is that the two scores are not on the same scale, and not even on
scales that a linear rescaling can reconcile:

- Cosine similarity is bounded in [-1, 1]. A great match scores 0.85, a poor
  one 0.30. The spread between them is small.
- BM25 is unbounded above and depends on corpus statistics, query length, and
  document length. A great match might score 18, a poor one 2. The spread is
  large and its magnitude varies per query.

Add them and BM25 dominates by sheer magnitude: the dense retriever's opinion
is rounded away. Min-max normalise per query first and a different failure
appears, the one :func:`weighted_score_fusion` is built to expose -- min-max
maps the best result of *every* list to exactly 1.0, including a list whose
best result is bad. A retriever that found nothing useful gets its top item
promoted to a perfect score, and it can outvote a retriever that found the
right answer.

:func:`reciprocal_rank_fusion` avoids the whole class of problem by discarding
score magnitudes and using only ranks. That loses information -- an emphatic
first place and a marginal one count the same -- and RRF is still the right
default, because the information it throws away is precisely the part that was
not comparable between retrievers.

:func:`pitfall_case` constructs a small, exact case where score fusion ranks a
non-relevant document first and RRF does not. It is a fixture, not a
measurement: the point is that such cases exist and are ordinary, and a
hand-checkable example proves it more convincingly than an aggregate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoredList:
    """One retriever's output: ids with scores, best first."""

    name: str
    doc_ids: list[int]
    scores: list[float]

    def __post_init__(self) -> None:
        if len(self.doc_ids) != len(self.scores):
            raise ValueError(
                f"{self.name}: {len(self.doc_ids)} ids but {len(self.scores)} scores"
            )

    @property
    def ranking(self) -> list[int]:
        return list(self.doc_ids)


def min_max_normalise(scores: Sequence[float]) -> list[float]:
    """Rescale to [0, 1] per list.

    Returns all-1.0 when every score is identical. That degenerate case is not
    a curiosity: a single-result list normalises to [1.0], so a retriever
    returning one mediocre document is credited with a perfect score.
    """
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high == low:
        return [1.0] * len(scores)
    return [(s - low) / (high - low) for s in scores]


def weighted_score_fusion(
    lists: Sequence[ScoredList], *, weights: Sequence[float] | None = None,
    normalise: bool = True,
) -> list[int]:
    """Fuse by summing (optionally normalised, optionally weighted) scores.

    Included as the *wrong* answer, implemented properly so the comparison is
    fair. With ``normalise=False`` it demonstrates scale domination; with
    ``normalise=True`` it demonstrates the min-max promotion failure. Both are
    real bugs found in real hybrid search code.
    """
    if weights is None:
        weights = [1.0] * len(lists)
    if len(weights) != len(lists):
        raise ValueError(f"{len(weights)} weights for {len(lists)} lists")

    totals: dict[int, float] = {}
    for scored, weight in zip(lists, weights, strict=True):
        values = min_max_normalise(scored.scores) if normalise else list(scored.scores)
        for doc_id, value in zip(scored.doc_ids, values, strict=True):
            totals[doc_id] = totals.get(doc_id, 0.0) + weight * value
    # Ties broken by ascending id so the output is deterministic; a fusion that
    # reorders tied documents between runs cannot be regression-tested.
    return [doc for doc, _ in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))]


def reciprocal_rank_fusion(
    lists: Sequence[ScoredList], *, k: float = 60.0, weights: Sequence[float] | None = None
) -> list[int]:
    """Fuse by summing ``weight / (k + rank)`` over the lists containing each document.

    ``k=60`` is the value from the original Cormack et al. paper and is not
    arbitrary in effect: it flattens the difference between the top few ranks,
    so a document ranked 1st by one retriever and 3rd by another is not
    dominated by a document ranked 1st by one and absent from the other.
    Lowering ``k`` sharpens the emphasis on first place; raising it approaches
    a plain vote over set membership.

    Only ranks are read. Scores are ignored entirely, which is the property
    that makes this immune to the scale problems above.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if weights is None:
        weights = [1.0] * len(lists)
    if len(weights) != len(lists):
        raise ValueError(f"{len(weights)} weights for {len(lists)} lists")

    totals: dict[int, float] = {}
    for scored, weight in zip(lists, weights, strict=True):
        for rank, doc_id in enumerate(scored.doc_ids, start=1):
            totals[doc_id] = totals.get(doc_id, 0.0) + weight / (k + rank)
    return [doc for doc, _ in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))]


@dataclass(frozen=True)
class PitfallCase:
    """A hand-checkable case where score fusion and RRF disagree on the winner."""

    dense: ScoredList
    lexical: ScoredList
    relevant: list[int]

    @property
    def score_fusion_winner(self) -> int:
        return weighted_score_fusion([self.dense, self.lexical], normalise=True)[0]

    @property
    def raw_score_fusion_winner(self) -> int:
        return weighted_score_fusion([self.dense, self.lexical], normalise=False)[0]

    @property
    def rrf_winner(self) -> int:
        return reciprocal_rank_fusion([self.dense, self.lexical])[0]


def pitfall_case() -> PitfallCase:
    """The demonstration: score fusion promotes a non-relevant document to rank 1.

    Document 7 is the relevant one. Both retrieval styles do a reasonable job
    of finding it -- the dense retriever ranks it first, BM25 ranks it second.

    The construction that breaks score fusion:

    - Dense scores are tightly packed, as cosine similarities on a real
      embedding model are. Doc 7 leads doc 3 by 0.86 to 0.84 -- a real but
      *narrow* margin.
    - BM25 scores are widely spread, as they always are. Doc 3 scores 30.0
      against doc 7's 12.0, because doc 3 happens to repeat a query term
      several times. Term repetition is exactly what BM25 rewards and what
      makes it useful; it is also why its scores are not comparable to cosine.

    Min-max normalise each list and the margins are destroyed in opposite
    directions. Dense (0.86, 0.85, 0.84) -> doc 7 = 1.00, doc 5 = 0.50,
    doc 3 = 0.00: a 0.02 spread is stretched to the full unit interval, so a
    near-tie is reported as a landslide. Lexical (30.0, 12.0) -> doc 3 = 1.00,
    doc 7 = 0.00: doc 7's genuine *second place out of two* is flattened to
    zero, because min-max only knows the extremes of the list it was handed.

    Summing gives doc 3 = 0.00 + 1.00 = 1.00 and doc 7 = 1.00 + 0.00 = 1.00, a
    tie that the id tie-break hands to **doc 3 -- the non-relevant document**.
    Adding the raw scores instead gives doc 3 = 30.84 against doc 7 = 12.86,
    so BM25's magnitude decides the ranking outright and doc 3 wins by a
    distance. Both spellings of score fusion get it wrong.

    RRF reads only ranks. Doc 7 is 1st and 2nd: 1/61 + 1/62 = 0.03252. Doc 3
    is 3rd and 1st: 1/63 + 1/61 = 0.03227. **Doc 7 wins**, because two strong
    placements beat one strong and one weak regardless of how emphatic any
    individual score was.
    """
    dense = ScoredList(
        name="dense-cosine",
        doc_ids=[7, 5, 3],
        scores=[0.86, 0.85, 0.84],
    )
    lexical = ScoredList(
        name="bm25",
        doc_ids=[3, 7],
        scores=[30.0, 12.0],
    )
    return PitfallCase(dense=dense, lexical=lexical, relevant=[7])
