"""Phase 12 feature-delta and F0.5 threshold regressions."""

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from src.common.metrics import macro_f05_fast
from src.matching import baseline_v0 as base
from src.normalize.evaluate_generalization import (
    _macro_numpy, _sample_s1, prepare_with_latin,
)


def test_latin_variant_only_changes_devanagari_name_fields():
    source = pl.DataFrame({
        "entity_id": ["S1-1", "S2-2"],
        "country": ["India", "India"],
        "business_name": ["Sky Infra Pvt Ltd", "स्काई इंफ्रा प्राइवेट लिमिटेड"],
        "business_address": ["Pune", "Pune"],
    })
    baseline = base.prepare(source)
    variant = prepare_with_latin(source)
    assert_frame_equal(baseline.head(1), variant.head(1))
    assert baseline["name_key"][1] != variant["name_key"][1]
    assert variant["name_key"][1] == "skyinfra"
    assert_frame_equal(
        baseline.tail(1).drop("name_core", "name_key"),
        variant.tail(1).drop("name_core", "name_key"),
    )
    fallback = prepare_with_latin(source, token_map={})
    assert_frame_equal(baseline.head(1), fallback.head(1))
    assert fallback["name_key"][1] != variant["name_key"][1]


def test_numpy_f05_agrees_with_shared_reference():
    truth = pl.DataFrame({
        "s1_id": ["a", "b", "c"],
        "matched_ids": [["1", "2"], ["3"], []],
    })
    cand = pl.DataFrame({
        "s1_id": ["a", "a", "b", "c"],
        "cand_id": ["1", "x", "3", "y"],
        "block_rank": [1, 2, 1, 1],
        "prob": [0.8, 0.4, 0.6, 0.2],
    })
    qi = np.array([0, 0, 1, 2])
    rank = cand["block_rank"].to_numpy()
    prob = cand["prob"].to_numpy()
    label = np.array([True, False, True, False])
    n_true = truth["matched_ids"].list.len().to_numpy()
    got = _macro_numpy(qi, rank, prob, label, n_true, 0.7, 0.5)
    reference = macro_f05_fast(base.decide(cand, 0.7, 0.5), truth)
    assert abs(got - reference) < 1e-12


def test_us_fit_excludes_validation_and_india_labels():
    s1 = pl.DataFrame({
        "entity_id": ["u1", "u2", "u3", "u4", "u5", "i1", "i2"],
        "country": ["US"] * 5 + ["India"] * 2,
    })
    truth = pl.DataFrame({
        "s1_id": s1["entity_id"],
        "matched_ids": [[f"m{x}"] for x in range(7)],
        "is_val": [False, False, False, False, True, False, True],
    })
    selected, labels = _sample_s1(s1, truth, fit_n=3, val_n=None)
    assert selected["fit"].height == 3
    assert set(selected["fit"]["entity_id"]).issubset({"u1", "u2", "u3", "u4"})
    assert selected["us_val"]["entity_id"].to_list() == ["u5"]
    assert selected["india_val"]["entity_id"].to_list() == ["i2"]
    for name, frame in selected.items():
        assert set(labels[name]["s1_id"]) == set(frame["entity_id"])
