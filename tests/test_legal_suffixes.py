import polars as pl
import pytest
from polars.testing import assert_frame_equal

from src.normalize import normalize_df


@pytest.mark.parametrize("name, normalized, core, legal", [
    ("Acme Pvt. Ltd.", "acme private limited", "acme", "private limited"),
    ("Acme Private Limited", "acme private limited", "acme", "private limited"),
    ("Acme Ltd.", "acme limited", "acme", "limited"),
    ("Acme Corp.", "acme corporation", "acme", "corporation"),
    ("Acme Inc.", "acme incorporated", "acme", "incorporated"),
    ("Acme L.L.C.", "acme llc", "acme", "llc"),
    ("Acme LLC", "acme llc", "acme", "llc"),
    ("Acme Limited Liability Company", "acme limited liability company", "acme", "llc"),
    ("Acme L.L.P.", "acme llp", "acme", "llp"),
    ("Acme S.A.R.L.", "acme sarl", "acme", "sarl"),
    ("Acme S.A.S.U.", "acme sasu", "acme", "sasu"),
    ("Acme SAS", "acme sas", "acme", "sas"),
    ("Acme SA", "acme sa", "acme", "sa"),
    ("Acme EURL", "acme eurl", "acme", "eurl"),
    ("Acme SCI", "acme sci", "acme", "sci"),
    ("Acme SNC", "acme snc", "acme", "snc"),
    ("Acme OPC", "acme opc", "acme", "opc"),
    ("Limited", "limited", "limited", None),
    ("Acme Private Limited and Sons", "acme private limited and sons", "acme private limited and sons", None),
    ("Acme Sarlene", "acme sarlene", "acme sarlene", None),
    ("ACME", "acme", "acme", None),
    (None, "", "", None),
])
def test_legal_form(name, normalized, core, legal):
    df = pl.DataFrame({"business_name": [name], "business_address": ["1 Road"], "country": ["Unseen"]})
    result = normalize_df(df)
    assert (result["name_norm"][0], result["name_core"][0], result["legal_suffix"][0]) == (
        normalized, core, legal
    )
    assert result["name_raw"][0] == name
    assert_frame_equal(normalize_df(result), result)


def test_multiple_records_and_raw_values():
    source = pl.DataFrame({
        "entity_id": ["S1-1", "S1-2", "S1-3"],
        "business_name": ["A Ltd", "B", "S.A.R.L."],
        "business_address": [None, "12 St", ""],
    })
    result = normalize_df(source)
    assert result["name_core"].to_list() == ["a", "b", "sarl"]
    assert result["legal_suffix"].to_list() == ["limited", None, None]
    assert_frame_equal(result.select(source.columns), source)
