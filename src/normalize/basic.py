"""Deterministic basic cleaning and Phase 4 abbreviation expansion."""

import polars as pl

from .abbreviations import (
    normalize_address_expr,
    normalize_address_v2_expr,
    normalize_name_expr,
)
from .address import address_fields, house_number_v2_expr
from .basic_text import clean_text, strip_accents
from .legal import legal_suffix_and_core
from .script import script_expr, with_latin_names
from .transliteration import load_token_map


def normalize_df(
    df: pl.DataFrame, transliteration_map: dict[str, str] | None = None
) -> pl.DataFrame:
    """Preserve inputs and add raw/cleaned columns for names and addresses.

    Raw columns preserve original strings, including nulls. Devanagari tokens
    use the non-validation training map when available, plus a deterministic
    fallback. Other non-Latin scripts remain empty in name_latin. Repeated
    calls recompute from original business columns, so are idempotent.
    """
    required = {"business_name", "business_address"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    country = pl.col("country") if "country" in df.columns else pl.lit("")
    france = country.cast(pl.String).fill_null("").str.to_lowercase() == "france"
    raw_name_norm = normalize_name_expr(pl.col("business_name"))
    # French accents are folded for matching; name_raw keeps the exact input.
    name_norm = pl.when(france).then(strip_accents(raw_name_norm)).otherwise(raw_name_norm)
    base = df.with_columns(
        pl.col("business_name").alias("name_raw"),
        pl.col("business_address").alias("address_raw"),
        name_norm.alias("name_norm"),
        normalize_address_expr(pl.col("business_address"), country).alias("addr_norm"),
        normalize_address_v2_expr(pl.col("business_address"), country).alias("addr_norm2"),
    )
    name_core, legal_suffix = legal_suffix_and_core(pl.col("name_norm"), country)
    postcode, city, house_no = address_fields(
        pl.col("business_address"), country
    )
    with_fields = base.with_columns(
        name_core.alias("name_core"),
        legal_suffix.alias("legal_suffix"),
        postcode.alias("postcode"),
        city.alias("city"),
        house_no.alias("house_no"),
        house_number_v2_expr(pl.col("business_address"), postcode, country).alias("house_no2"),
        script_expr(pl.col("business_name")).alias("script"),
    )
    token_map = load_token_map() if transliteration_map is None else transliteration_map
    return with_latin_names(with_fields, token_map)
