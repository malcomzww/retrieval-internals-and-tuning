"""Metrics tested against fixtures computed by hand.

Every expected value in this file was worked out on paper and is written as
the arithmetic that produces it, not as a decimal literal. A test that asserts
``== 0.5693`` proves only that the code still does what it did when someone
pasted the output in; writing ``(1/log2(2) + 1/log2(4)) / ideal`` proves the
implementation matches the definition.
"""

from __future__ import annotations

import math

import pytest

from retrieval_internals_and_tuning.metrics import (
    mean_recall_at_k,
    mrr,
    ndcg_at_k,
    per_query_recall,
    recall_at_k,
    reciprocal_rank,
)


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        assert recall_at_k([1, 2, 3], [1, 2, 3], k=3) == 1.0

    def test_order_does_not_matter(self) -> None:
        # Recall is a set measure. A retriever that finds all three true
        # neighbours but ranks them backwards has recall 1.0; that is what
        # separates recall from NDCG, and conflating the two is a common way
        # to report a ranking improvement that is really a retrieval one.
        assert recall_at_k([3, 2, 1], [1, 2, 3], k=3) == 1.0

    def test_partial_recall(self) -> None:
        # Two of the true top-3 present.
        assert recall_at_k([1, 2, 9], [1, 2, 3], k=3) == pytest.approx(2 / 3)

    def test_zero_recall(self) -> None:
        assert recall_at_k([7, 8, 9], [1, 2, 3], k=3) == 0.0

    def test_truncates_both_lists_to_k(self) -> None:
        # Retrieved id 3 sits at rank 4 and must not count at k=3; truth is
        # also cut to {1,2,3}, so the denominator is 3 and only 1 and 2 hit.
        assert recall_at_k([1, 2, 9, 3], [1, 2, 3, 4], k=3) == pytest.approx(2 / 3)

    def test_extra_retrieved_beyond_k_ignored(self) -> None:
        assert recall_at_k([1, 2, 3, 4, 5, 6], [1, 2, 3], k=3) == 1.0

    def test_rejects_nonpositive_k(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            recall_at_k([1], [1], k=0)

    def test_rejects_empty_truth(self) -> None:
        with pytest.raises(ValueError, match="truth set is empty"):
            recall_at_k([1], [], k=3)


class TestMeanAndPerQueryRecall:
    def test_mean_of_two_queries(self) -> None:
        retrieved = [[1, 2], [5, 9]]
        truth = [[1, 2], [5, 6]]
        # 1.0 and 0.5 -> 0.75
        assert mean_recall_at_k(retrieved, truth, k=2) == pytest.approx(0.75)

    def test_per_query_is_unaggregated(self) -> None:
        retrieved = [[1, 2], [5, 9]]
        truth = [[1, 2], [5, 6]]
        assert per_query_recall(retrieved, truth, k=2) == [1.0, 0.5]

    def test_per_query_mean_matches_mean_recall(self) -> None:
        retrieved = [[1, 2, 3], [4, 9, 8], [7, 8, 9]]
        truth = [[1, 2, 3], [4, 5, 6], [7, 8, 1]]
        each = per_query_recall(retrieved, truth, k=3)
        assert sum(each) / len(each) == pytest.approx(mean_recall_at_k(retrieved, truth, 3))

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="query counts differ"):
            mean_recall_at_k([[1]], [[1], [2]], k=1)

    def test_rejects_zero_queries(self) -> None:
        with pytest.raises(ValueError, match="cannot average over zero queries"):
            mean_recall_at_k([], [], k=1)


class TestReciprocalRank:
    def test_first_position(self) -> None:
        assert reciprocal_rank([5, 1, 2], [5]) == 1.0

    def test_third_position(self) -> None:
        assert reciprocal_rank([1, 2, 5], [5]) == pytest.approx(1 / 3)

    def test_absent_scores_zero(self) -> None:
        assert reciprocal_rank([1, 2, 3], [5]) == 0.0

    def test_uses_first_relevant_only(self) -> None:
        # Both 2 and 3 are relevant; the rank of the *first* is what counts.
        assert reciprocal_rank([1, 2, 3], [2, 3]) == pytest.approx(1 / 2)

    def test_mrr_averages(self) -> None:
        # 1/1 and 1/3 -> 2/3
        assert mrr([[5, 1], [1, 2, 5]], [[5], [5]]) == pytest.approx((1 + 1 / 3) / 2)


class TestNdcgAtK:
    def test_perfect_ranking(self) -> None:
        assert ndcg_at_k([1, 2, 3], [1, 2, 3], k=3) == 1.0

    def test_single_relevant_at_rank_two(self) -> None:
        # DCG = 1/log2(3); ideal has one relevant item at rank 1 -> 1/log2(2) = 1.
        assert ndcg_at_k([9, 1, 8], [1], k=3) == pytest.approx(1 / math.log2(3))

    def test_reversed_ranking_scores_below_perfect(self) -> None:
        # Same three relevant docs, worst order. Recall would be 1.0 here.
        gain = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
        assert ndcg_at_k([3, 2, 1], [1, 2, 3], k=3) == pytest.approx(gain / gain)

    def test_ranking_order_changes_ndcg_when_recall_is_equal(self) -> None:
        # The pair that shows NDCG and recall are not interchangeable. Both
        # rankings contain exactly the same true top-3 -- so recall@3 is 1.0
        # for both -- yet NDCG separates them, because only NDCG looks at
        # where in the list the relevant items landed.
        best = [1, 2, 3]
        worst = [3, 2, 1]
        assert recall_at_k(best, [1, 2, 3], k=3) == recall_at_k(worst, [1, 2, 3], k=3)
        assert ndcg_at_k([1, 9, 2], [1, 2], k=3) > ndcg_at_k([9, 1, 2], [1, 2], k=3)

    def test_ideal_uses_min_of_k_and_relevant_count(self) -> None:
        # Two relevant docs, k=5, both at the top: the ideal DCG is over two
        # items, not five, so this is exactly 1.0.
        assert ndcg_at_k([1, 2, 7, 8, 9], [1, 2], k=5) == pytest.approx(1.0)

    def test_partial_credit_computed_by_hand(self) -> None:
        # Relevant {1,2}. Retrieved ranks: 1 at position 1, 2 at position 3.
        gain = 1 / math.log2(2) + 1 / math.log2(4)
        ideal = 1 / math.log2(2) + 1 / math.log2(3)
        assert ndcg_at_k([1, 9, 2], [1, 2], k=3) == pytest.approx(gain / ideal)

    def test_rejects_empty_relevant(self) -> None:
        with pytest.raises(ValueError, match="relevant set is empty"):
            ndcg_at_k([1], [], k=1)
