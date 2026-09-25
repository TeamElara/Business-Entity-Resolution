"""Phase 9 metric definitions and fixed-candidate reranking checks."""

import polars as pl

from src.normalize.evaluate import (
    _add_query_counts,
    _empty_rank_counts,
    _final_rank_counts,
    rank_one_query,
    score_true_pairs,
    token_jaccard,
)


def test_jaccard_empty_and_overlap():
    assert token_jaccard("", "") == 0
    assert token_jaccard("acme private", "acme limited") == 1 / 3


def test_true_pair_metrics_ignore_missing_postcodes():
    frame = pl.DataFrame({
        "country": ["India"],
        "s1_raw": ["ACME Pvt Ltd"], "cand_raw": ["Acme Private Limited"],
        "s1_core": ["acme"], "cand_core": ["acme"],
        "s1_postcode": [None], "cand_postcode": ["411001"],
        "s1_script": ["latin"], "cand_script": ["latin"],
        "s1_raw_tokens": ["acme pvt ltd"],
        "cand_raw_tokens": ["acme private limited"],
    })
    result = score_true_pairs(frame)
    overall = result["overall"]
    assert overall["true_pairs"] == 1
    assert overall["raw_exact_name_equal"] == 0
    assert overall["raw_casefold_name_equal"] == 0
    assert overall["normalized_core_equal"] == 1
    assert overall["postcode_both_present"] == 0
    assert overall["postcode_equal_given_both"] is None
    assert overall["raw_token_jaccard_mean"] == 1 / 5
    assert overall["normalized_core_jaccard_mean"] == 1


def test_rank_same_candidate_pool_and_counts():
    true = {"match"}
    result = rank_one_query(
        "Acme", "acme", ["other", "match"],
        ["Other", "Acme"], ["other", "acme"],
        [0.9, 0.4], true,
    )
    assert result["retrieved_true_pairs"] == 1
    assert result["raw_top5_pairs"] == 1
    assert result["core_top10_pairs"] == 1
    assert result["raw_true_ranks"] == [1]
    assert result["core_true_ranks"] == [1]
    counts = _empty_rank_counts()
    _add_query_counts(counts, true, result)
    _add_query_counts(counts, set(), result)
    final = _final_rank_counts(counts)
    assert final["queries"] == 2
    assert final["positive_queries"] == 1
    assert final["core_top10_hit_queries_rate"] == 1
