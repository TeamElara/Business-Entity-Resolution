import polars as pl
import pytest
from polars.testing import assert_frame_equal

from src.normalize import normalize_df
from src.normalize.basic import clean_text


@pytest.mark.parametrize("raw, expected", [
    (" << ACME & Sons, Ltd. -- ", "acme and sons ltd"),
    ("ＡＣＭＥ ＆ Sons", "acme and sons"),
    ("A_12/B-3\t Road\n", "a 12 b 3 road"),
    ("ÉCOLE Café", "école café"),
    ("प्राइवेट लिमिटेड", "प्राइवेट लिमिटेड"),
    ("சதர்ன்", "சதர்ன்"),
    ("సన్", "సన్"),
    ("ಸಿಲ್ವರ್", "ಸಿಲ್ವರ್"),
    (None, ""), ("", ""), ("--- << >>", ""),
    ("NA", "na"),
])
def test_text_cases(raw, expected):
    df = pl.DataFrame({"business_name": [raw], "business_address": [raw]})
    result = normalize_df(df)
    assert df.select(clean_text(pl.col("business_name"))).item() == expected
    assert df.select(clean_text(pl.col("business_address"))).item() == expected
    assert result["name_raw"][0] == raw
    assert result["address_raw"][0] == raw


def test_identity_and_repeatability():
    df = pl.DataFrame({"entity_id": ["S1-1", "S1-2"], "country": ["France", "Unknown"],
                       "business_name": ["ACME & Co", None], "business_address": [" 1 Rd.", ""]})
    result = normalize_df(df)
    assert_frame_equal(result.select(df.columns), df)
    assert_frame_equal(normalize_df(result), result)


def test_missing_columns():
    with pytest.raises(ValueError, match="business_address"):
        normalize_df(pl.DataFrame({"business_name": ["a"]}))


def test_empty_frame():
    result = normalize_df(pl.DataFrame(schema={"business_name": pl.String, "business_address": pl.String}))
    assert result.height == 0
    assert result.schema["name_norm"] == pl.String
