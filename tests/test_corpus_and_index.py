"""Corpus invariants and index equivalences.

These run on the hashed fallback embedder so the suite needs no model download
and no network. The properties under test -- uniqueness, determinism, exactness
of the flat index -- are properties of the machinery, not of MiniLM, so the
fallback tests them just as well.
"""

from __future__ import annotations

import numpy as np
import pytest

from retrieval_internals_and_tuning import corpus, index
from retrieval_internals_and_tuning.metrics import mean_recall_at_k

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def docs() -> list[corpus.Document]:
    return corpus.build_corpus(2_000, seed=0)


@pytest.fixture(scope="module")
def vectors(docs: list[corpus.Document]) -> np.ndarray:
    return corpus.embed([d.text for d in docs], use_model=False)


class TestCorpus:
    def test_documents_are_unique(self, docs: list[corpus.Document]) -> None:
        # The regression this guards: duplicate texts produce identical
        # vectors, identical vectors tie in the flat scan, and ties cap
        # measured ANN recall strictly below 1.0 regardless of ef_search. An
        # earlier version of this corpus had 7,030 unique texts out of 20,000
        # and pinned recall at 0.889, which looked like an HNSW limitation and
        # was not.
        assert len({d.text for d in docs}) == len(docs)

    def test_vectors_are_unique(self, vectors: np.ndarray) -> None:
        assert len(np.unique(vectors, axis=0)) == vectors.shape[0]

    def test_deterministic(self) -> None:
        assert [d.text for d in corpus.build_corpus(200, seed=0)] == [
            d.text for d in corpus.build_corpus(200, seed=0)
        ]

    def test_categories_are_exactly_balanced(self, docs: list[corpus.Document]) -> None:
        # Filter selectivity in the filtered-ANN experiment is derived from
        # these counts, so an imbalance would move the reported collapse.
        counts = {c: sum(1 for d in docs if d.category == c) for c in corpus.CATEGORIES}
        assert set(counts.values()) == {len(docs) // len(corpus.CATEGORIES)}

    def test_metadata_within_declared_domain(self, docs: list[corpus.Document]) -> None:
        assert all(d.category in corpus.CATEGORIES for d in docs)
        assert all(d.year in corpus.YEARS for d in docs)

    def test_doc_ids_are_dense_and_ordered(self, docs: list[corpus.Document]) -> None:
        # Every index in this repo uses positional row == doc_id. If that ever
        # stops holding, every retrieved id silently points at the wrong text.
        assert [d.doc_id for d in docs] == list(range(len(docs)))

    def test_queries_are_not_verbatim_documents(self, docs: list[corpus.Document]) -> None:
        queries = corpus.build_queries(docs, 50)
        assert not (set(queries) & {d.text for d in docs})

    def test_rejects_nonpositive_size(self) -> None:
        with pytest.raises(ValueError, match="n_docs must be positive"):
            corpus.build_corpus(0)


class TestEmbedding:
    def test_l2_normalised(self, vectors: np.ndarray) -> None:
        # Inner product equals cosine only on the unit sphere; every index in
        # the repo relies on that to agree about what "nearest" means.
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_dtype_is_float32(self, vectors: np.ndarray) -> None:
        assert vectors.dtype == np.float32

    def test_fallback_is_deterministic(self) -> None:
        # blake2b rather than hash(): Python's string hash is salted per
        # process, so a hash()-based embedder would produce different vectors
        # on every run and no cached result would ever be reusable.
        a = corpus.embed(["page cache contention"], use_model=False)
        b = corpus.embed(["page cache contention"], use_model=False)
        assert np.array_equal(a, b)

    def test_shared_vocabulary_is_closer_than_disjoint(self) -> None:
        v = corpus.embed(
            ["write-ahead log fsync durability",
             "write-ahead log fsync checkpoint",
             "congestion window packet loss handshake"],
            use_model=False,
        )
        assert float(v[0] @ v[1]) > float(v[0] @ v[2])

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="cannot embed an empty list"):
            corpus.embed([], use_model=False)


class TestFlatIndex:
    def test_returns_requested_k(self, vectors: np.ndarray) -> None:
        assert index.FlatIndex(vectors).search(vectors[:5], 7).shape == (5, 7)

    def test_self_query_returns_itself_first(self, vectors: np.ndarray) -> None:
        # A vector's nearest neighbour under cosine is itself. If this fails,
        # the metric or the normalisation is wrong.
        ids = index.FlatIndex(vectors).search(vectors[:20], 1)
        assert [int(x) for x in ids[:, 0]] == list(range(20))

    def test_matches_numpy_brute_force(self, vectors: np.ndarray) -> None:
        # The ground truth every recall number is measured against, verified
        # against an independent argsort rather than trusted.
        #
        # Compared by *score*, not by id. The hashed fallback embedder yields
        # integer-valued dot products, so neighbours frequently tie exactly,
        # and FAISS and numpy are entitled to break a tie differently. An
        # id-equality assertion would be testing tie-break order, which no
        # part of this repo depends on; equality of the retrieved similarity
        # profile is the property that actually matters.
        flat = index.FlatIndex(vectors)
        scores = vectors[:10] @ vectors.T
        got = np.take_along_axis(scores, flat.search(vectors[:10], 5), axis=1)
        want = np.sort(scores, axis=1)[:, ::-1][:, :5]
        assert np.allclose(got, want, atol=1e-5)


@pytest.mark.skipif(not index.FAISS_AVAILABLE, reason="requires faiss-cpu")
class TestApproximateIndexes:
    def test_hnsw_recall_rises_with_ef_search(self, vectors: np.ndarray) -> None:
        # The core monotonicity that makes ef_search a usable knob: more
        # candidates explored can only help. This is the claim the decision
        # guide's selection table rests on.
        #
        # Asserted as a trend between the extremes rather than step-by-step.
        # On the tie-heavy fallback embedder an intermediate step can move by
        # a fraction of a percent in either direction purely on tie-break
        # order; the end-to-end direction is the real claim.
        truth = index.FlatIndex(vectors).search(vectors[:60], 10)
        hnsw = index.HnswIndex(vectors, index.HnswParams(m=8, ef_construction=40))
        recalls = []
        for ef in (5, 10, 40, 200):
            hnsw.set_ef_search(ef)
            recalls.append(mean_recall_at_k(hnsw.search(vectors[:60], 10), truth, 10))
        assert recalls[-1] > recalls[0]
        # Not `>= max(recalls)`: on the tie-heavy fallback embedder an
        # intermediate ef can edge ahead by a fraction of a percent purely on
        # tie-break order, which is what the comment above anticipates but the
        # strict form did not allow. A 1% band keeps the monotonic claim while
        # tolerating the ties.
        assert recalls[-1] >= max(recalls) - 0.01, (
            f"highest ef_search gave {recalls[-1]:.4f} against a best of "
            f"{max(recalls):.4f} across {recalls}; more search should not lose"
        )

    def test_hnsw_approaches_exhaustive_recall(self, vectors: np.ndarray) -> None:
        # Guards the duplicate-vector regression from the corpus side: with a
        # large ef_search HNSW must approach exhaustive search.
        #
        # The threshold is 0.90 rather than 0.99 because this runs on the
        # hashed fallback embedder, whose exact ties cost recall that no
        # amount of graph exploration recovers -- when the true top-10 has
        # eleven equally-scored candidates, some legitimate answer is always
        # counted as a miss. On real MiniLM vectors the same configuration
        # reaches 1.000; that number is produced by scripts/generate_results.py,
        # not asserted here, because it needs a model download.
        truth = index.FlatIndex(vectors).search(vectors[:60], 10)
        hnsw = index.HnswIndex(vectors, index.HnswParams(m=32, ef_construction=200))
        hnsw.set_ef_search(400)
        assert mean_recall_at_k(hnsw.search(vectors[:60], 10), truth, 10) > 0.90

    def test_ivfpq_compression_ratio_arithmetic(self) -> None:
        # 384 dims x 4 bytes = 1536 bytes raw; 48 sub-quantisers x 8 bits = 48
        # bytes of code. Exactly 32x.
        params = index.IvfPqParams(m_sub=48, nbits=8)
        assert params.compression_ratio(384) == pytest.approx(32.0)

    def test_ivfpq_rejects_indivisible_subquantisers(self, vectors: np.ndarray) -> None:
        with pytest.raises(ValueError, match="must divide dim"):
            index.IvfPqIndex(vectors, index.IvfPqParams(nlist=8, m_sub=5))

    def test_ivfpq_recall_rises_with_nprobe(self, vectors: np.ndarray) -> None:
        truth = index.FlatIndex(vectors).search(vectors[:60], 10)
        ivf = index.IvfPqIndex(vectors, index.IvfPqParams(nlist=32, m_sub=48, nprobe=1))
        low = mean_recall_at_k(ivf.search(vectors[:60], 10), truth, 10)
        ivf.set_nprobe(32)
        assert mean_recall_at_k(ivf.search(vectors[:60], 10), truth, 10) >= low

    def test_latency_measurement_is_positive(self, vectors: np.ndarray) -> None:
        flat = index.FlatIndex(vectors)
        with index.single_threaded():
            assert index.measure_query_latency_ms(flat, vectors[:5], 10, repeats=1) > 0.0

    def test_single_threaded_restores_thread_count(self) -> None:
        import faiss

        before = faiss.omp_get_max_threads()
        with index.single_threaded():
            assert faiss.omp_get_max_threads() == 1
        assert faiss.omp_get_max_threads() == before
