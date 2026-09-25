"""Phase 10 learning, fallback, and dataframe-order regressions."""

import polars as pl
from polars.testing import assert_frame_equal

from src.normalize import normalize_df
from src.normalize.transliteration import (
    build_token_map,
    has_devanagari,
    romanize_token,
    transliterate_name,
)


def test_learn_aligned_tokens_and_ignore_ambiguous():
    names = pl.DataFrame({
        "latin_name": ["sky private limited", "sky private limited", "red private limited"],
        "dev_name": ["स्काई प्राइवेट लिमिटेड", "स्काई प्राइवेट लिमिटेड", "रेड प्राइवेट लिमिटेड"],
    })
    mapping, stats = build_token_map(names)
    assert stats["equal_token_count_pairs"] == 3
    assert mapping["स्काई"] == "sky"
    assert mapping["प्राइवेट"] == "private"
    assert mapping["लिमिटेड"] == "limited"


def test_transliteration_fallback_and_suffix_strip():
    mapping = {"स्काई": "sky", "प्राइवेट": "private", "लिमिटेड": "limited"}
    assert transliterate_name("स्काई प्राइवेट लिमिटेड", mapping) == "sky"
    assert transliterate_name("स्काई infra private limited", mapping) == "sky infra"
    unknown = romanize_token("भारत")
    assert unknown and unknown.isascii() and not has_devanagari(unknown)
    assert romanize_token("ॐ").isascii()


def test_normalize_df_keeps_order_and_raw_script():
    source = pl.DataFrame({
        "entity_id": ["1", "2", "3"], "country": ["India", "India", "India"],
        "business_name": ["स्काई प्राइवेट लिमिटेड", "ACME Ltd", "தமிழ் கடை"],
        "business_address": ["Pune", "Delhi", "Chennai"],
    })
    mapping = {"स्काई": "sky", "प्राइवेट": "private", "लिमिटेड": "limited"}
    out = normalize_df(source, transliteration_map=mapping)
    assert out["entity_id"].to_list() == ["1", "2", "3"]
    assert out["name_latin"].to_list() == ["sky", "acme", ""]
    assert out["script"].to_list() == ["devanagari", "latin", "other"]
    assert_frame_equal(normalize_df(out, transliteration_map=mapping), out)
