"""Fusion arithmetic, including the pitfall case worked out by hand.

The pitfall test is the one that matters: it asserts that naive score fusion
puts a *non-relevant* document first while RRF puts the relevant one first, on
a fixture small enough to verify with a calculator. That is a stronger form of
the claim than any aggregate over a corpus, because there is nowhere for a
confound to hide.
"""

from __future__ import annotations

import pytest

from retrieval_internals_and_tuning.fusion import (
    ScoredList,
    min_max_normalise,
    pitfall_case,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)


class TestMinMaxNormalise:
    def test_maps_extremes_to_zero_and_one(self) -> None:
        assert min_max_normalise([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]

    def test_identical_scores_all_become_one(self) -> None:
        # The degenerate case that causes the pitfall: a list whose scores are
        # all equal -- including a single-element list -- has its every member
        # promoted to a perfect 1.0.
        assert min_max_normalise([3.0, 3.0]) == [1.0, 1.0]

    def test_single_element_becomes_one(self) -> None:
        assert min_max_normalise([0.01]) == [1.0]

    def test_empty(self) -> None:
        assert min_max_normalise([]) == []

    def test_destroys_scale_information(self) -> None:
        # A 0.02-wide spread and a 20-wide spread normalise identically. This
        # is exactly the information loss that makes min-max fusion unsafe:
        # after normalising, a near-tie is indistinguishable from a landslide.
        assert min_max_normalise([0.84, 0.85, 0.86]) == min_max_normalise([10.0, 20.0, 30.0])


class TestScoredList:
    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="2 ids but 1 scores"):
            ScoredList(name="x", doc_ids=[1, 2], scores=[1.0])


class TestReciprocalRankFusion:
    def test_single_list_preserves_order(self) -> None:
        one = ScoredList(name="a", doc_ids=[5, 2, 9], scores=[0.0, 0.0, 0.0])
        assert reciprocal_rank_fusion([one]) == [5, 2, 9]

    def test_ignores_scores_entirely(self) -> None:
        # The defining property. Two lists with the same ranking but wildly
        # different scores must fuse identically -- that immunity to scale is
        # the reason RRF is the safe default.
        modest = ScoredList(name="a", doc_ids=[1, 2], scores=[0.51, 0.50])
        emphatic = ScoredList(name="a", doc_ids=[1, 2], scores=[9999.0, 0.001])
        other = ScoredList(name="b", doc_ids=[2, 1], scores=[1.0, 0.5])
        assert reciprocal_rank_fusion([modest, other]) == reciprocal_rank_fusion(
            [emphatic, other]
        )

    def test_agreement_between_lists_wins(self) -> None:
        # Doc 1 is 1st and 1st; doc 2 is 2nd and 2nd.
        first = ScoredList(name="a", doc_ids=[1, 2], scores=[1.0, 0.9])
        second = ScoredList(name="b", doc_ids=[1, 2], scores=[1.0, 0.9])
        assert reciprocal_rank_fusion([first, second])[0] == 1

    def test_hand_computed_scores(self) -> None:
        # Doc 1: rank 1 in a, rank 2 in b -> 1/61 + 1/62.
        # Doc 2: rank 2 in a, rank 1 in b -> 1/62 + 1/61. Exactly tied, so the
        # id tie-break decides and doc 1 leads.
        a = ScoredList(name="a", doc_ids=[1, 2], scores=[1.0, 0.9])
        b = ScoredList(name="b", doc_ids=[2, 1], scores=[1.0, 0.9])
        assert reciprocal_rank_fusion([a, b]) == [1, 2]

    def test_k_controls_top_rank_emphasis(self) -> None:
        # Small k sharpens first place enough that one 1st beats a 2nd+3rd.
        a = ScoredList(name="a", doc_ids=[1], scores=[1.0])
        b = ScoredList(name="b", doc_ids=[2, 1], scores=[1.0, 0.5])
        assert reciprocal_rank_fusion([a, b], k=0.5)[0] == 1

    def test_weights_are_applied(self) -> None:
        a = ScoredList(name="a", doc_ids=[1, 2], scores=[1.0, 0.9])
        b = ScoredList(name="b", doc_ids=[2, 1], scores=[1.0, 0.9])
        assert reciprocal_rank_fusion([a, b], weights=[1.0, 10.0])[0] == 2

    def test_rejects_nonpositive_k(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            reciprocal_rank_fusion([ScoredList("a", [1], [1.0])], k=0)

    def test_rejects_weight_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="1 weights for 2 lists"):
            reciprocal_rank_fusion(
                [ScoredList("a", [1], [1.0]), ScoredList("b", [2], [1.0])],
                weights=[1.0],
            )


class TestWeightedScoreFusion:
    def test_raw_sum_is_dominated_by_the_larger_scale(self) -> None:
        # Cosine cannot outvote BM25 when the two are added raw: doc 2 loses on
        # the dense list by 0.5 and wins on the lexical list by 90.
        dense = ScoredList(name="dense", doc_ids=[1, 2], scores=[1.0, 0.5])
        lexical = ScoredList(name="bm25", doc_ids=[2, 1], scores=[100.0, 10.0])
        assert weighted_score_fusion([dense, lexical], normalise=False)[0] == 2

    def test_normalising_changes_the_winner(self) -> None:
        dense = ScoredList(name="dense", doc_ids=[1, 2], scores=[1.0, 0.5])
        lexical = ScoredList(name="bm25", doc_ids=[2, 1], scores=[100.0, 10.0])
        raw = weighted_score_fusion([dense, lexical], normalise=False)
        normalised = weighted_score_fusion([dense, lexical], normalise=True)
        # Same inputs, different answer, purely from the normalisation choice --
        # which is the sign that neither answer is trustworthy.
        assert raw != normalised

    def test_rejects_weight_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="1 weights for 2 lists"):
            weighted_score_fusion(
                [ScoredList("a", [1], [1.0]), ScoredList("b", [2], [1.0])],
                weights=[1.0],
            )


class TestPitfallCase:
    def test_score_fusion_ranks_a_non_relevant_document_first(self) -> None:
        # The headline claim of the fusion section.
        case = pitfall_case()
        assert case.score_fusion_winner not in case.relevant

    def test_raw_score_fusion_also_gets_it_wrong(self) -> None:
        # Both spellings fail, so "just don't normalise" is not the fix.
        case = pitfall_case()
        assert case.raw_score_fusion_winner not in case.relevant

    def test_rrf_ranks_the_relevant_document_first(self) -> None:
        case = pitfall_case()
        assert case.rrf_winner in case.relevant

    def test_the_two_methods_disagree(self) -> None:
        case = pitfall_case()
        assert case.score_fusion_winner != case.rrf_winner

    def test_pitfall_arithmetic_by_hand(self) -> None:
        case = pitfall_case()
        # Dense (0.86, 0.85, 0.84) -> (1.0, 0.5, 0.0) for docs (7, 5, 3).
        assert min_max_normalise(case.dense.scores) == [1.0, 0.5, 0.0]
        # Lexical (30.0, 12.0) -> (1.0, 0.0) for docs (3, 7): doc 7's real
        # second-of-two place is flattened to zero.
        assert min_max_normalise(case.lexical.scores) == [1.0, 0.0]
        # Both totals are 1.0; the id tie-break awards it to doc 3.
        assert weighted_score_fusion([case.dense, case.lexical], normalise=True)[:2] == [3, 7]
        # RRF: doc 7 = 1/61 + 1/62 > doc 3 = 1/63 + 1/61.
        assert 1 / 61 + 1 / 62 > 1 / 63 + 1 / 61
        assert reciprocal_rank_fusion([case.dense, case.lexical])[0] == 7

    def test_relevant_document_is_found_by_both_retrievers(self) -> None:
        # Guards against a strawman fixture. The pitfall must not depend on one
        # retriever failing outright: doc 7 is ranked 1st by dense and 2nd by
        # lexical, so any sane fusion has the evidence it needs to rank it top.
        case = pitfall_case()
        for doc in case.relevant:
            assert doc in case.dense.doc_ids
            assert doc in case.lexical.doc_ids
