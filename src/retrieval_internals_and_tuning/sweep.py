"""Parameter sweeps producing recall-vs-latency curves.

The sweep exists to answer a question tuning guides usually skip: given a
recall target, which configuration reaches it *cheapest*? That is not the same
as which configuration reaches the highest recall. A large-``M`` graph beats a
small one at every ef_search, but it also costs more per query and more memory,
and past a certain recall target the extra edges buy nothing that a larger
ef_search on a cheaper graph would not.

Answering it requires the Pareto frontier, not a table of numbers, so
:func:`pareto_frontier` is the real output of this module. A configuration that
is both slower *and* less accurate than another is strictly dominated, and no
one should ever deploy it; the frontier is what remains after those are struck
out.

The IVF-PQ sweep is deliberately kept separate rather than merged into one
combined table. Its recall loss has two independent causes -- ``nprobe``
(search-time, recoverable) and the quantiser width (build-time, permanent) --
and a single mixed table cannot show which one a given number came from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .index import (
    FlatIndex,
    HnswIndex,
    HnswParams,
    IvfPqIndex,
    IvfPqParams,
    measure_query_latency_ms,
    single_threaded,
)
from .metrics import mean_recall_at_k, per_query_recall


@dataclass(frozen=True)
class SweepPoint:
    """One measured operating point."""

    label: str
    recall: float
    latency_ms: float
    per_query: list[float]
    build_seconds: float = 0.0
    bytes_per_vector: float = 0.0

    @property
    def qps(self) -> float:
        """Single-core queries per second implied by the median latency."""
        return 1000.0 / self.latency_ms if self.latency_ms > 0 else float("inf")


def sweep_hnsw(
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    truth: np.ndarray,
    *,
    k: int = 10,
    m_values: tuple[int, ...] = (8, 16, 32),
    ef_construction_values: tuple[int, ...] = (40, 200),
    ef_search_values: tuple[int, ...] = (10, 20, 50, 100, 200, 400),
    latency_queries: int = 60,
    repeats: int = 3,
) -> list[SweepPoint]:
    """Sweep ``M`` x ``ef_construction`` x ``ef_search``.

    The loop nests ef_search innermost on purpose: it is the only one of the
    three that can be changed without rebuilding, so one graph serves the whole
    inner loop. Rebuilding per ef_search would multiply the sweep cost by six
    and measure nothing extra -- and the fact that it *would* be wasteful is
    itself the practical lesson about which knob to reach for in production.
    """
    import time

    points: list[SweepPoint] = []
    for m in m_values:
        for ef_construction in ef_construction_values:
            start = time.perf_counter()
            hnsw = HnswIndex(doc_vectors, HnswParams(m=m, ef_construction=ef_construction))
            build_seconds = time.perf_counter() - start

            # Graph memory: M bidirectional links per node at layer 0 (FAISS
            # allocates 2*M there), 4-byte ids, plus the full float32 vector,
            # which IndexHNSWFlat stores uncompressed. This is why HNSW is a
            # memory decision before it is a latency decision.
            dim = doc_vectors.shape[1]
            bytes_per_vector = dim * 4 + 2 * m * 4

            for ef_search in ef_search_values:
                hnsw.set_ef_search(ef_search)
                retrieved = hnsw.search(query_vectors, k)
                with single_threaded():
                    latency = measure_query_latency_ms(
                        hnsw, query_vectors[:latency_queries], k, repeats=repeats
                    )
                points.append(
                    SweepPoint(
                        label=f"M={m},efc={ef_construction},ef={ef_search}",
                        recall=mean_recall_at_k(retrieved, truth, k),
                        latency_ms=latency,
                        per_query=per_query_recall(retrieved, truth, k),
                        build_seconds=build_seconds,
                        bytes_per_vector=bytes_per_vector,
                    )
                )
    return points


def sweep_ivfpq(
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    truth: np.ndarray,
    *,
    k: int = 10,
    nlist: int = 256,
    m_sub_values: tuple[int, ...] = (16, 48, 96),
    nprobe_values: tuple[int, ...] = (1, 4, 16, 64),
    latency_queries: int = 60,
    repeats: int = 3,
    seed: int = 0,
) -> list[SweepPoint]:
    """Sweep the quantiser width against ``nprobe``.

    ``m_sub`` is the compression axis: at dim 384, m_sub=16 stores 16 bytes per
    vector (96x), m_sub=96 stores 96 bytes (16x). Sweeping both axes is what
    separates "recall lost to visiting too few partitions" from "recall lost to
    throwing away the vector's precision".
    """
    points: list[SweepPoint] = []
    dim = doc_vectors.shape[1]
    for m_sub in m_sub_values:
        if dim % m_sub != 0:
            continue
        params = IvfPqParams(nlist=nlist, m_sub=m_sub, nprobe=nprobe_values[0])
        ivf = IvfPqIndex(doc_vectors, params, seed=seed)
        for nprobe in nprobe_values:
            ivf.set_nprobe(nprobe)
            retrieved = ivf.search(query_vectors, k)
            with single_threaded():
                latency = measure_query_latency_ms(
                    ivf, query_vectors[:latency_queries], k, repeats=repeats
                )
            points.append(
                SweepPoint(
                    label=f"m_sub={m_sub},nprobe={nprobe}",
                    recall=mean_recall_at_k(retrieved, truth, k),
                    latency_ms=latency,
                    per_query=per_query_recall(retrieved, truth, k),
                    bytes_per_vector=float(m_sub),
                )
            )
    return points


def measure_flat(
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    truth: np.ndarray,
    *,
    k: int = 10,
    latency_queries: int = 60,
    repeats: int = 3,
) -> SweepPoint:
    """The exact baseline: recall 1.0 by definition, and the latency to beat.

    Worth measuring rather than assuming. On a corpus small enough, a flat scan
    is faster than an HNSW traversal *and* exact, which makes every approximate
    index on this page a pessimisation. Knowing where that crossover sits is
    the first question to ask before building an ANN index at all.
    """
    flat = FlatIndex(doc_vectors)
    with single_threaded():
        latency = measure_query_latency_ms(
            flat, query_vectors[:latency_queries], k, repeats=repeats
        )
    retrieved = flat.search(query_vectors, k)
    return SweepPoint(
        label="flat (exact)",
        recall=mean_recall_at_k(retrieved, truth, k),
        latency_ms=latency,
        per_query=per_query_recall(retrieved, truth, k),
        bytes_per_vector=float(doc_vectors.shape[1] * 4),
    )


def pareto_frontier(points: list[SweepPoint]) -> list[SweepPoint]:
    """Configurations not dominated on both recall and latency.

    A point is dominated when another is at least as fast *and* at least as
    accurate, with one of the two strictly better. Everything dominated is a
    configuration nobody should deploy: there is a strictly better one
    available at no cost. Publishing the full sweep without the frontier
    invites exactly that mistake, because a big table makes a bad row look like
    a legitimate trade-off.
    """
    frontier: list[SweepPoint] = []
    for candidate in points:
        dominated = any(
            other.latency_ms <= candidate.latency_ms
            and other.recall >= candidate.recall
            and (other.latency_ms < candidate.latency_ms or other.recall > candidate.recall)
            for other in points
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda p: p.latency_ms)


def cheapest_for_recall(points: list[SweepPoint], target: float) -> SweepPoint | None:
    """Lowest-latency configuration meeting a recall target, or ``None``.

    This is the function the decision guide's selection table is built from.
    Returning ``None`` rather than the best available is deliberate: silently
    handing back a 0.93 config when 0.99 was asked for is how a recall target
    quietly stops being a target.
    """
    qualifying = [p for p in points if p.recall >= target]
    if not qualifying:
        return None
    return min(qualifying, key=lambda p: p.latency_ms)
