"""Phase 10 pair and same-pool rank metric regressions."""

import polars as pl

from src.normalize.evaluate_transliteration import rank_same_pool, score_devanagari_pairs


def test_pair_metrics_on_transliterated_true_match():
    pairs = pl.DataFrame({
        "s1_id": ["S1-1"], "cand_id": ["S2-1"],
        "s1_core": ["sky"], "s1_latin": ["sky"],
        "cand_core": ["स्काई"], "cand_latin": ["sky"],
        "cand_script": ["devanagari"],
    })
    report = score_devanagari_pairs(pairs, {"स्काई": "sky"})
    assert report["true_devanagari_pairs"] == 1
    assert report["core_exact_rate"] == 0
    assert report["latin_exact_rate"] == 1
    assert report["mapped_devanagari_token_instance_rate"] == 1
    assert report["nonlatin_output_pairs"] == 0


def test_rank_changes_only_hindi_candidate_score():
    out = rank_same_pool(
        "sky", "sky", ["other", "match"],
        ["sky corp", "स्काई"], ["sky corp", "sky"],
        ["latin", "devanagari"], [0.9, 0.5],
        {"match"}, {"match"},
    )
    assert out["retrieved_dev_pairs"] == 1
    assert out["baseline_dev_true_ranks"] == [2]
    assert out["latin_dev_true_ranks"] == [1]
