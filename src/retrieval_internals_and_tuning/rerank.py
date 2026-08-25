"""Re-ranking, and the latency budget at which it stops paying.

This is the module the repo exists for. The question is not "does re-ranking
improve quality" -- it does, reliably, and saying so is not a result. The
question is whether it improves quality *more than spending the same
milliseconds on the retriever would*, because that is the choice an engineer
with a latency budget actually faces.

Framing it that way changes the answer. The usual comparison is
"ANN at ef=64" versus "ANN at ef=64 plus a re-ranker", which is not a fair
fight: the second arm is given extra time the first was never offered. The
honest comparison holds *total* latency fixed and asks which allocation of it
wins. A budget spent entirely on retrieval buys a larger ef_search or an exact
scan; a budget split buys a cheaper retrieval pass plus re-ranking of its
candidates. :func:`budget_frontier` runs that comparison.

Two re-rankers are provided:

- :class:`LexicalReranker` -- BM25 over the candidate set only. Microseconds,
  no model, and it is a genuinely different signal from the embedding, which
  is why it can add anything at all.
- :class:`CrossEncoderReranker` -- ``ms-marco-MiniLM-L-6-v2`` scoring each
  (query, document) pair jointly. Far stronger and far slower, and its cost
  grows linearly in the candidate count, which is what creates a crossover
  rather than a uniform win.

The quality metric here is deliberately *not* ANN recall. Re-ranking reorders a
candidate set; it cannot add a document the retriever missed, so its recall
against the *ANN* ground truth can only fall. Quality is measured instead
against :func:`~.corpus.term_relevance` -- documents that genuinely discuss
both terms the query asks about -- which is a property of the corpus text and
independent of any embedding.

That independence is not a detail. An earlier version of this module defined
relevance as the exact embedding ranking, and every retrieval arm scored NDCG
1.0000 by construction: the retriever was graded against the ordering it
exists to approximate, so re-ranking could only ever lose. The conclusion
"re-ranking never pays at any budget" fell straight out of the setup and had
nothing to do with re-ranking.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .index import FlatIndex, HnswIndex, HnswParams, single_threaded
from .metrics import mrr, ndcg_at_k

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def embedding_rank_labels(
    doc_vectors: np.ndarray, query_vectors: np.ndarray, *, depth: int = 20
) -> list[list[int]]:
    """The exact top-``depth`` embedding neighbours of each query.

    Kept, but **not** for scoring the re-ranking comparison -- see the module
    docstring for why doing that produces a tautology. Its legitimate use is as
    the ANN ground truth for :mod:`.sweep`, where the question genuinely is
    "did the approximate index find what exhaustive search would have found".
    """
    return [[int(x) for x in row] for row in FlatIndex(doc_vectors).search(query_vectors, depth)]


class LexicalReranker:
    """BM25 over the candidate set.

    Scores only the candidates, not the corpus, so cost scales with the
    candidate count rather than the index size. IDF is computed from the
    *corpus* though: deriving it from a 50-document candidate set would make
    every term look rare and produce a scoring function that changes meaning
    with the candidate count.
    """

    name = "bm25-lexical"

    def __init__(self, texts: Sequence[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tokens = [tokenise(t) for t in texts]
        self.lengths = np.array([len(t) for t in self.tokens], dtype=np.float32)
        self.avg_length = float(self.lengths.mean()) if len(self.lengths) else 0.0

        document_frequency: dict[str, int] = {}
        for token_list in self.tokens:
            for term in set(token_list):
                document_frequency[term] = document_frequency.get(term, 0) + 1
        n = len(self.tokens)
        # Robertson/Sparck-Jones IDF with the +0.5 smoothing, floored at zero.
        # Unfloored it goes negative for terms in over half the corpus, which
        # lets a common term actively subtract from a score.
        self.idf = {
            term: max(0.0, math.log((n - freq + 0.5) / (freq + 0.5) + 1.0))
            for term, freq in document_frequency.items()
        }

    def score(self, query: str, candidates: Sequence[int]) -> np.ndarray:
        query_terms = tokenise(query)
        scores = np.zeros(len(candidates), dtype=np.float32)
        for position, doc_id in enumerate(candidates):
            tokens = self.tokens[doc_id]
            if not tokens:
                continue
            length = self.lengths[doc_id]
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            total = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(self.avg_length, 1e-9)
                )
                total += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
            scores[position] = total
        return scores

    def rerank(self, query: str, candidates: Sequence[int]) -> list[int]:
        order = np.argsort(-self.score(query, candidates), kind="stable")
        return [int(candidates[i]) for i in order]


class CrossEncoderReranker:
    """A cross-encoder scoring (query, document) pairs jointly.

    The expensive option, and the reason it is expensive is also the reason it
    works: a bi-encoder compresses the document to a vector before it has seen
    the query, while a cross-encoder attends over both together. That extra
    power costs a full transformer forward pass *per candidate*, so latency is
    linear in the candidate count where the retriever's is logarithmic.
    """

    name = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: Sequence[int], texts: Sequence[str]) -> list[int]:
        if not candidates:
            return []
        scores = self.model.predict(
            [(query, texts[doc_id]) for doc_id in candidates], show_progress_bar=False
        )
        order = np.argsort(-np.asarray(scores), kind="stable")
        return [int(candidates[i]) for i in order]


@dataclass(frozen=True)
class Arm:
    """One end-to-end configuration and what it cost.

    ``latency_ms`` is the *total* per query -- retrieval plus re-ranking. That
    total is the only figure a budget can be compared against, and reporting
    the re-ranker's cost separately is how re-ranking comes to look free.
    """

    name: str
    latency_ms: float
    ndcg: float
    mrr_score: float
    per_query_ndcg: list[float]
    candidates: int
    reranker: str


def _score_arm(
    name: str,
    rankings: list[list[int]],
    relevant: list[list[int]],
    latency_ms: float,
    *,
    k: int,
    candidates: int,
    reranker: str,
) -> Arm:
    # The full relevance set is passed to both metrics, never a truncation of
    # it. `ndcg_at_k` already caps its ideal ranking at k, so truncating the
    # labels first would only corrupt the *gain* term by discarding relevant
    # documents that the ranking legitimately found.
    per_query = [
        ndcg_at_k(ranking, labels, k)
        for ranking, labels in zip(rankings, relevant, strict=True)
    ]
    return Arm(
        name=name,
        latency_ms=latency_ms,
        ndcg=sum(per_query) / len(per_query),
        mrr_score=mrr(rankings, relevant),
        per_query_ndcg=per_query,
        candidates=candidates,
        reranker=reranker,
    )


def retrieval_only_arm(
    hnsw: HnswIndex,
    query_vectors: np.ndarray,
    relevant: list[list[int]],
    ef_search: int,
    *,
    k: int = 10,
    latency_queries: int = 40,
) -> Arm:
    """Spend the whole budget on retrieval: raise ef_search, return the top k."""
    hnsw.set_ef_search(ef_search)
    rankings = [[int(x) for x in row] for row in hnsw.search(query_vectors, k)]
    with single_threaded():
        sample = query_vectors[:latency_queries]
        hnsw.search(sample[:1], k)
        start = time.perf_counter()
        for row in range(sample.shape[0]):
            hnsw.search(sample[row : row + 1], k)
        latency = (time.perf_counter() - start) / sample.shape[0] * 1000.0
    return _score_arm(
        f"retrieve-only ef={ef_search}", rankings, relevant, latency,
        k=k, candidates=k, reranker="none",
    )


def reranked_arm(
    hnsw: HnswIndex,
    query_vectors: np.ndarray,
    queries: Sequence[str],
    texts: Sequence[str],
    relevant: list[list[int]],
    ef_search: int,
    candidates: int,
    reranker: LexicalReranker | CrossEncoderReranker,
    *,
    k: int = 10,
    latency_queries: int = 40,
) -> Arm:
    """Retrieve ``candidates``, re-rank them, return the top ``k``.

    Latency covers both stages, measured on the same queries the quality
    numbers come from.
    """
    hnsw.set_ef_search(max(ef_search, candidates))
    retrieved = hnsw.search(query_vectors, candidates)

    def run(index_of_query: int) -> list[int]:
        pool = [int(x) for x in retrieved[index_of_query]]
        if isinstance(reranker, CrossEncoderReranker):
            return reranker.rerank(queries[index_of_query], pool, texts)[:k]
        return reranker.rerank(queries[index_of_query], pool)[:k]

    rankings = [run(i) for i in range(len(queries))]

    with single_threaded():
        count = min(latency_queries, len(queries))
        run(0)  # warm-up
        start = time.perf_counter()
        for i in range(count):
            hnsw.search(query_vectors[i : i + 1], candidates)
            run(i)
        latency = (time.perf_counter() - start) / count * 1000.0

    return _score_arm(
        f"{reranker.name} c={candidates} ef={ef_search}", rankings, relevant, latency,
        k=k, candidates=candidates, reranker=reranker.name,
    )


def exact_arm(
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    relevant: list[list[int]],
    *,
    k: int = 10,
    latency_queries: int = 40,
) -> Arm:
    """Spend the budget on an exhaustive scan. The quality ceiling for the retriever.

    Included because it is the arm most often left out, and it frequently wins:
    on a corpus of this size an exact scan is only a few milliseconds, so any
    re-ranker costing more than that must beat *perfect* retrieval to justify
    itself.
    """
    flat = FlatIndex(doc_vectors)
    rankings = [[int(x) for x in row] for row in flat.search(query_vectors, k)]
    with single_threaded():
        sample = query_vectors[:latency_queries]
        flat.search(sample[:1], k)
        start = time.perf_counter()
        for row in range(sample.shape[0]):
            flat.search(sample[row : row + 1], k)
        latency = (time.perf_counter() - start) / sample.shape[0] * 1000.0
    return _score_arm(
        "exact scan", rankings, relevant, latency, k=k, candidates=k, reranker="none"
    )


def budget_frontier(arms: Sequence[Arm], budgets_ms: Sequence[float]) -> list[tuple[float, Arm]]:
    """Best arm fitting each latency budget.

    The whole answer to the repo's question is the shape of this list: read
    down it and note where the winning arm stops being a re-ranked one. That
    boundary is the crossover, and it is a *ranking* of arms rather than a set
    of millisecond values -- which is what makes it portable to a machine with
    different absolute timings.
    """
    frontier: list[tuple[float, Arm]] = []
    for budget in budgets_ms:
        affordable = [a for a in arms if a.latency_ms <= budget]
        if affordable:
            frontier.append((budget, max(affordable, key=lambda a: a.ndcg)))
    return frontier


def crossover_budget(arms: Sequence[Arm], budgets_ms: Sequence[float]) -> float | None:
    """Smallest budget at which a re-ranked arm becomes the best affordable one.

    ``None`` means no budget tested was ever won by re-ranking. That is a real
    possible answer and the code returns it rather than reaching for the
    nearest number that could be called a crossover.
    """
    for budget, winner in budget_frontier(arms, sorted(budgets_ms)):
        if winner.reranker != "none":
            return budget
    return None


def build_arms(
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    queries: Sequence[str],
    texts: Sequence[str],
    relevant: list[list[int]],
    *,
    k: int = 10,
    ef_values: Sequence[int] = (16, 32, 64, 128, 256),
    candidate_counts: Sequence[int] = (25, 50, 100),
    use_cross_encoder: bool = True,
    latency_queries: int = 40,
) -> list[Arm]:
    """Assemble every arm of the budget comparison on one shared graph.

    One graph serves all arms so that differences between them are the
    strategy and never the index. Rebuilding per arm would fold graph-build
    variance into a latency comparison that is supposed to isolate query cost.
    """
    hnsw = HnswIndex(doc_vectors, HnswParams(m=16, ef_construction=200))
    arms: list[Arm] = [
        retrieval_only_arm(hnsw, query_vectors, relevant, ef, k=k,
                           latency_queries=latency_queries)
        for ef in ef_values
    ]
    arms.append(exact_arm(doc_vectors, query_vectors, relevant, k=k,
                          latency_queries=latency_queries))

    lexical = LexicalReranker(texts)
    rerankers: list[LexicalReranker | CrossEncoderReranker] = [lexical]
    if use_cross_encoder:
        try:
            rerankers.append(CrossEncoderReranker())
        except Exception:
            # No model, no network: the lexical arm still answers the question,
            # and the results header records which re-rankers actually ran.
            pass

    for reranker in rerankers:
        for count in candidate_counts:
            arms.append(
                reranked_arm(hnsw, query_vectors, queries, texts, relevant,
                             ef_search=64, candidates=count, reranker=reranker,
                             k=k, latency_queries=latency_queries)
            )
    return arms
