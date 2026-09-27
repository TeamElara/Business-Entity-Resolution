"""Small join check for the validation-pair agreement report."""

import polars as pl

from src.normalize.evaluate_house_no2 import full_change_counts, pair_agreement


SCHEMA = {
    "entity_id": pl.String, "country": pl.String, "address_raw": pl.String,
    "postcode": pl.String, "house_no": pl.String, "addr_norm": pl.String,
}


def write(path, rows):
    pl.DataFrame(rows, schema=SCHEMA).write_parquet(path)


def test_pair_agreement_uses_same_candidate_ids(tmp_path):
    write(tmp_path / "train_s1.parquet", [{
        "entity_id": "S1-1", "country": "France", "address_raw": "001511 Rue de Paris",
        "postcode": None, "house_no": "00151", "addr_norm": "001511 rue de paris",
    }])
    write(tmp_path / "train_s2.parquet", [{
        "entity_id": "S2-2", "country": "France", "address_raw": "1511 Rue de Paris",
        "postcode": None, "house_no": "1511", "addr_norm": "1511 rue de paris",
    }])
    write(tmp_path / "train_s3.parquet", [{
        "entity_id": "S3-3", "country": "France", "address_raw": "10 Rue de Paris",
        "postcode": None, "house_no": "10", "addr_norm": "10 rue de paris",
    }])
    pairs = pl.DataFrame({
        "s1_id": ["S1-1", "S1-1"], "cand_id": ["S2-2", "S3-3"],
        "label": [1, 0], "prob": [0.4, 0.8],
    })
    result = pair_agreement(pairs, tmp_path)
    assert result.height == 2
    assert result["old_agree"].to_list() == [False, False]
    assert result["new_agree"].to_list() == [True, False]
    changes = full_change_counts(tmp_path)
    assert changes["rows"].sum() == 3
    assert changes["house_changed"].sum() == 1
