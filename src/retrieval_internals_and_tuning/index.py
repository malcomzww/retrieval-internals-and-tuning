"""Vector indexes: exact flat, HNSW, and IVF-PQ, behind one interface.

Three deliberate choices govern this module.

**Inner product on normalised vectors.** Every index uses the same metric, so
recall comparisons between them mean something. Mixing L2 and cosine across a
sweep produces curves that cannot be read against each other.

**Single-threaded search.** FAISS defaults to every core, and on a 24-core box
that turns a 0.4 ms query into a 0.02 ms query -- which flatters the ANN index
so much that re-ranking never appears to pay. Real serving runs many queries
concurrently, so each query gets roughly one core. Measuring one query on 24
cores answers a question nobody has. :func:`single_threaded` pins it.

**Latency is per-query and taken as a median of repeated single-query calls.**
Batched search amortises overhead that a live serving path does not get, and
the mean is hostage to one GC pause. The median of individual calls is the
closest honest analogue of a served request.

If FAISS is unavailable the module degrades to an exact numpy scan, so the
recall machinery and its tests still run -- see :data:`FAISS_AVAILABLE`.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - import-time environment probe
    import faiss

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore[assignment]
    FAISS_AVAILABLE = False


@contextlib.contextmanager
def single_threaded() -> Iterator[None]:
    """Pin FAISS to one thread for the duration of the block.

    Restores the previous setting afterwards so that index *construction*,
    which is embarrassingly parallel and not what we are measuring, can still
    use the whole machine.
    """
    if not FAISS_AVAILABLE:
        yield
        return
    previous = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        yield
    finally:
        faiss.omp_set_num_threads(previous)


class FlatIndex:
    """Exhaustive inner-product scan. The ground truth every recall is measured against.

    Uses FAISS when present purely for speed; the numpy path returns identical
    ids for identical input, because both are exact. That equivalence is
    asserted in the tests rather than assumed.
    """

    name = "flat"

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index = None
        if FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(self.vectors.shape[1])
            self._index.add(self.vectors)

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if self._index is not None:
            _, ids = self._index.search(queries, k)
            return ids
        scores = queries @ self.vectors.T
        # argpartition then sort only the top-k slice: an O(n) selection beats
        # a full O(n log n) sort of 20k rows per query by a wide margin.
        top = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
        ordered = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
        return np.take_along_axis(top, ordered, axis=1)


@dataclass(frozen=True)
class HnswParams:
    """The three knobs that define an HNSW operating point.

    ``M`` and ``ef_construction`` are baked into the graph and can only be
    changed by rebuilding. ``ef_search`` is free to change per query. That
    asymmetry is the single most useful fact about tuning HNSW, and the sweep
    in :mod:`.sweep` is organised around it.
    """

    m: int = 16
    ef_construction: int = 100
    ef_search: int = 64


class HnswIndex:
    """FAISS ``IndexHNSWFlat``.

    Note ``ef_search`` is set on the shared graph object, so it is mutated in
    place by :meth:`set_ef_search` rather than requiring a rebuild -- which is
    exactly why an ef_search sweep is cheap and an M sweep is not.
    """

    name = "hnsw"

    def __init__(self, vectors: np.ndarray, params: HnswParams) -> None:
        if not FAISS_AVAILABLE:  # pragma: no cover
            raise RuntimeError("HNSW requires faiss-cpu")
        self.params = params
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index = faiss.IndexHNSWFlat(vectors.shape[1], params.m,
                                          faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = params.ef_construction
        self._index.add(vectors)
        self._index.hnsw.efSearch = params.ef_search

    def set_ef_search(self, ef_search: int) -> None:
        self._index.hnsw.efSearch = ef_search

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        _, ids = self._index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        return ids


@dataclass(frozen=True)
class IvfPqParams:
    """IVF-PQ configuration.

    ``m_sub`` sub-quantisers x ``nbits`` each replaces a ``dim``-dimensional
    float32 vector with ``m_sub * nbits / 8`` bytes. At dim=384 and m_sub=48,
    nbits=8 that is 1536 bytes down to 48 -- a 32x reduction, and the reason
    anyone tolerates the recall loss this module measures.
    """

    nlist: int = 256
    m_sub: int = 48
    nbits: int = 8
    nprobe: int = 16

    def compression_ratio(self, dim: int) -> float:
        """Raw float32 bytes per vector divided by PQ code bytes per vector.

        Excludes the coarse-quantiser id and the centroid table, which are
        real but do not scale with the corpus -- at 20k vectors the table is
        noise, and at 20M it is invisible.
        """
        return (dim * 4) / (self.m_sub * self.nbits / 8)


class IvfPqIndex:
    """FAISS ``IndexIVFPQ``: coarse partitioning plus product quantisation.

    Two independent sources of recall loss live here, and the sweep separates
    them: ``nprobe`` controls how many partitions are visited (a search-time
    knob, recoverable), while ``m_sub``/``nbits`` control how badly the vector
    is compressed (a build-time knob, permanent). Reporting one IVF-PQ recall
    number without saying which knob produced the loss is uninterpretable.
    """

    name = "ivfpq"

    def __init__(self, vectors: np.ndarray, params: IvfPqParams, *, seed: int = 0) -> None:
        if not FAISS_AVAILABLE:  # pragma: no cover
            raise RuntimeError("IVF-PQ requires faiss-cpu")
        self.params = params
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        dim = vectors.shape[1]
        if dim % params.m_sub != 0:
            raise ValueError(f"m_sub={params.m_sub} must divide dim={dim}")

        quantiser = faiss.IndexFlatIP(dim)
        self._index = faiss.IndexIVFPQ(quantiser, dim, params.nlist, params.m_sub,
                                       params.nbits, faiss.METRIC_INNER_PRODUCT)
        self._index.cp.seed = seed
        self._index.train(vectors)
        self._index.add(vectors)
        self._index.nprobe = params.nprobe

    def set_nprobe(self, nprobe: int) -> None:
        self._index.nprobe = nprobe

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        _, ids = self._index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        return ids


def measure_query_latency_ms(
    index: FlatIndex | HnswIndex | IvfPqIndex,
    queries: np.ndarray,
    k: int,
    *,
    repeats: int = 3,
) -> float:
    """Median single-query latency in milliseconds, measured one query at a time.

    ``repeats`` passes over the query set are taken and the median of *all*
    individual timings is returned. A single pass on a laptop picks up
    scheduler noise on a handful of queries; the median across repeats is
    stable to a few percent between runs, which is the most that can be asked
    of a wall-clock measurement on a non-realtime OS.

    The caller is responsible for wrapping this in :func:`single_threaded`.
    Doing it here would hide the choice, and the choice materially changes the
    answer.
    """
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    queries = np.ascontiguousarray(queries, dtype=np.float32)

    # One untimed pass: the first query pays for lazy allocation inside FAISS
    # and for pulling the graph into cache. Charging that to the measurement
    # would inflate the fastest configurations most, since they have the least
    # real work to hide it behind.
    index.search(queries[:1], k)

    timings: list[float] = []
    for _ in range(repeats):
        for row in range(queries.shape[0]):
            one = queries[row : row + 1]
            start = time.perf_counter()
            index.search(one, k)
            timings.append((time.perf_counter() - start) * 1000.0)
    timings.sort()
    return timings[len(timings) // 2]
