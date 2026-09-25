"""Deterministic basic cleaning and Phase 4 abbreviation expansion."""

import polars as pl

from .abbreviations import (
    ADDRESS_ABBREVIATIONS,
    expand_tokens,
    normalize_name_expr,
)
from .address import address_fields
from .basic_text import clean_text
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
    name_norm = normalize_name_expr(pl.col("business_name"))
    base = df.with_columns(
        pl.col("business_name").alias("name_raw"),
        pl.col("business_address").alias("address_raw"),
        name_norm.alias("name_norm"),
        expand_tokens(clean_text(pl.col("business_address")), ADDRESS_ABBREVIATIONS).alias("addr_norm"),
    )
    name_core, legal_suffix = legal_suffix_and_core(pl.col("name_norm"))
    postcode, city, house_no = address_fields(
        pl.col("business_address"), pl.col("country") if "country" in df.columns else pl.lit("")
    )
    with_fields = base.with_columns(
        name_core.alias("name_core"),
        legal_suffix.alias("legal_suffix"),
        postcode.alias("postcode"),
        city.alias("city"),
        house_no.alias("house_no"),
        script_expr(pl.col("business_name")).alias("script"),
    )
    token_map = load_token_map() if transliteration_map is None else transliteration_map
    return with_latin_names(with_fields, token_map)
