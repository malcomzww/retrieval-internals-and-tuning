"""The filtered-ANN failure, and the arithmetic behind it.

The headline claim -- naive post-filtering collapses when the filter is
selective -- is tested here as a *relationship* rather than as a stored
number: post-filter recall tracks selectivity, and pre-filtering does not.
That form of the claim survives a change of corpus, embedder or machine, which
a recorded 0.008 would not.
"""

from __future__ import annotations

import numpy as np
import pytest

from retrieval_internals_and_tuning import corpus, filtering, index

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def docs() -> list[corpus.Document]:
    return corpus.build_corpus(2_000, seed=0)


@pytest.fixture(scope="module")
def vectors(docs: list[corpus.Document]) -> np.ndarray:
    return corpus.embed([d.text for d in docs], use_model=False)


@pytest.fixture(scope="module")
def queries(docs: list[corpus.Document]) -> np.ndarray:
    return corpus.embed(corpus.build_queries(docs, 40), use_model=False)


class TestSelectivity:
    def test_full_corpus_predicate(self, docs: list[corpus.Document]) -> None:
        assert filtering.selectivity(docs, lambda d: True) == 1.0

    def test_empty_predicate(self, docs: list[corpus.Document]) -> None:
        assert filtering.selectivity(docs, lambda d: False) == 0.0

    def test_category_is_one_fifth(self, docs: list[corpus.Document]) -> None:
        assert filtering.selectivity(
            docs, lambda d: d.category == "storage"
        ) == pytest.approx(0.2)

    def test_tenant_is_independent_of_category(self, docs: list[corpus.Document]) -> None:
        # The reason the tenant field exists. `doc_id % 5` is perfectly aliased
        # to category, because categories are assigned round-robin over five
        # values -- so a filter built from it would confound selectivity with
        # topic and misreport the collapse. Tenants must not do that.
        joint = filtering.selectivity(docs, lambda d: d.tenant < 8 and d.category == "storage")
        product = filtering.selectivity(docs, lambda d: d.tenant < 8) * filtering.selectivity(
            docs, lambda d: d.category == "storage"
        )
        assert joint == pytest.approx(product, abs=0.01)

    def test_doc_id_mod_five_is_aliased_to_category(self, docs: list[corpus.Document]) -> None:
        # Documents the trap explicitly, so nobody reintroduces it.
        assert filtering.selectivity(
            docs, lambda d: d.doc_id % 5 == 0 and d.category == "storage"
        ) == pytest.approx(0.2)

    def test_matching_ids_are_row_indices(self, docs: list[corpus.Document]) -> None:
        ids = filtering.matching_ids(docs, lambda d: d.tenant == 0)
        assert all(docs[int(i)].tenant == 0 for i in ids)

    def test_rejects_empty_corpus(self) -> None:
        with pytest.raises(ValueError, match="empty corpus"):
            filtering.selectivity([], lambda d: True)


class TestRequiredOverfetch:
    def test_full_selectivity_needs_exactly_k(self) -> None:
        assert filtering.required_overfetch(1.0, 10) == 10

    def test_one_percent_needs_a_thousand(self) -> None:
        # The argument against over-fetching as a general fix, in one number:
        # k/selectivity, so the filters needing most help cost the most to help.
        assert filtering.required_overfetch(0.01, 10) == 1_000

    def test_scales_inversely_with_selectivity(self) -> None:
        assert filtering.required_overfetch(0.05, 10) == 2 * filtering.required_overfetch(
            0.10, 10
        )

    def test_rejects_zero_selectivity(self) -> None:
        with pytest.raises(ValueError, match="selectivity must be in"):
            filtering.required_overfetch(0.0, 10)

    def test_rejects_nonpositive_k(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            filtering.required_overfetch(0.5, 0)


class TestFilteredGroundTruth:
    def test_only_returns_matching_documents(
        self, docs: list[corpus.Document], vectors: np.ndarray, queries: np.ndarray
    ) -> None:
        allowed = filtering.matching_ids(docs, lambda d: d.tenant < 4)
        truth = filtering.filtered_ground_truth(vectors, queries, allowed, 10)
        assert set(truth.ravel().tolist()) <= {int(x) for x in allowed}

    def test_differs_from_unfiltered_top_k(
        self, docs: list[corpus.Document], vectors: np.ndarray, queries: np.ndarray
    ) -> None:
        # The reference-set mistake this guards against: scoring a filtered
        # result against the *unfiltered* top-k would mark a perfectly correct
        # filtered answer as a miss, because most unfiltered neighbours do not
        # satisfy the filter.
        allowed = filtering.matching_ids(docs, lambda d: d.tenant < 2)
        filtered = filtering.filtered_ground_truth(vectors, queries, allowed, 10)
        unfiltered = index.FlatIndex(vectors).search(queries, 10)
        assert not np.array_equal(filtered, unfiltered)

    def test_rejects_empty_filter(self, vectors: np.ndarray, queries: np.ndarray) -> None:
        with pytest.raises(ValueError, match="matches no documents"):
            filtering.filtered_ground_truth(
                vectors, queries, np.array([], dtype=np.int64), 10
            )


@pytest.mark.skipif(not index.FAISS_AVAILABLE, reason="requires faiss-cpu")
class TestStrategies:
    @pytest.fixture(scope="class")
    def hnsw(self, vectors: np.ndarray) -> index.HnswIndex:
        return index.HnswIndex(vectors, index.HnswParams(m=16, ef_construction=100,
                                                         ef_search=64))

    def test_post_filter_returns_only_allowed(
        self, docs: list[corpus.Document], hnsw: index.HnswIndex, queries: np.ndarray
    ) -> None:
        allowed = filtering.matching_ids(docs, lambda d: d.tenant < 4)
        allowed_set = {int(x) for x in allowed}
        results = filtering.post_filter(hnsw, queries, allowed, 10)
        assert all(doc in allowed_set for row in results for doc in row)

    def test_post_filter_returns_short_lists_when_filter_bites(
        self, docs: list[corpus.Document], hnsw: index.HnswIndex, queries: np.ndarray
    ) -> None:
        # The failure mode itself. Short lists are not a cosmetic defect: they
        # are the user-visible symptom, and the code must not pad them away.
        allowed = filtering.matching_ids(docs, lambda d: d.tenant < 2)
        results = filtering.post_filter(hnsw, queries, allowed, 10)
        assert min(len(r) for r in results) < 10

    def test_pre_filter_is_always_complete(
        self, docs: list[corpus.Document], vectors: np.ndarray, queries: np.ndarray
    ) -> None:
        allowed = filtering.matching_ids(docs, lambda d: d.tenant < 2)
        results = filtering.pre_filter_exact(vectors, queries, allowed, 10)
        assert all(len(r) == 10 for r in results)

    def test_overfetch_rejects_zero_multiplier(
        self, docs: list[corpus.Document], hnsw: index.HnswIndex, queries: np.ndarray
    ) -> None:
        allowed = filtering.matching_ids(docs, lambda d: d.tenant < 4)
        with pytest.raises(ValueError, match="multiplier must be at least 1"):
            filtering.overfetch_filter(hnsw, queries, allowed, 10, multiplier=0)

    def test_overfetch_beats_post_filter(
        self, docs: list[corpus.Document], vectors: np.ndarray,
        hnsw: index.HnswIndex, queries: np.ndarray
    ) -> None:
        from retrieval_internals_and_tuning.metrics import mean_recall_at_k

        predicate = filtering.matching_ids(docs, lambda d: d.tenant < 4)
        truth = [
            [int(x) for x in row]
            for row in filtering.filtered_ground_truth(vectors, queries, predicate, 10)
        ]
        naive = filtering.post_filter(hnsw, queries, predicate, 10)
        wide = filtering.overfetch_filter(hnsw, queries, predicate, 10, multiplier=50)
        assert mean_recall_at_k(wide, truth, 10) > mean_recall_at_k(naive, truth, 10)

    def test_post_filter_recall_falls_as_filter_tightens(
        self, docs: list[corpus.Document], vectors: np.ndarray,
        hnsw: index.HnswIndex, queries: np.ndarray
    ) -> None:
        # The headline claim, stated portably. Post-filter recall is bounded by
        # roughly the selectivity, because of ten unfiltered neighbours only
        # about `selectivity * 10` survive. So tightening the filter must drive
        # recall down, and the drop is arithmetic rather than incidental.
        results = filtering.evaluate_strategies(
            docs, vectors, queries, hnsw, lambda d: d.tenant < 16, k=10, latency_queries=5
        )
        broad = next(r for r in results if r.strategy == "post-filter")

        results = filtering.evaluate_strategies(
            docs, vectors, queries, hnsw, lambda d: d.tenant < 1, k=10, latency_queries=5
        )
        narrow = next(r for r in results if r.strategy == "post-filter")

        assert narrow.recall < broad.recall
        assert narrow.empty_rate > broad.empty_rate
        # Recall cannot materially exceed the fraction of the corpus that
        # survives the filter.
        assert narrow.recall <= narrow.selectivity * 3 + 0.05

    def test_pre_filter_always_reaches_perfect_recall(
        self, docs: list[corpus.Document], vectors: np.ndarray,
        hnsw: index.HnswIndex, queries: np.ndarray
    ) -> None:
        for tenants in (16, 4, 1):
            results = filtering.evaluate_strategies(
                docs, vectors, queries, hnsw,
                lambda d, n=tenants: d.tenant < n, k=10, latency_queries=5,
            )
            exact = next(r for r in results if r.strategy == "pre-filter exact")
            assert exact.recall == pytest.approx(1.0)
            assert exact.empty_rate == 0.0
