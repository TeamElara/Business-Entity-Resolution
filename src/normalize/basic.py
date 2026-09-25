"""Deterministic basic cleaning and Phase 4 abbreviation expansion."""

import polars as pl

from .abbreviations import (
    ADDRESS_ABBREVIATIONS,
    NAME_ABBREVIATIONS,
    expand_tokens,
    normalize_dotted_legal_forms,
)
from .address import address_fields
from .basic_text import clean_text
from .legal import legal_suffix_and_core
from .script import interim_latin_name, script_expr


def normalize_df(df: pl.DataFrame) -> pl.DataFrame:
    """Preserve inputs and add raw/cleaned columns for names and addresses.

    Raw columns preserve original strings, including nulls. Transliterating
    non-Latin names is deferred to Phase 10; their interim name_latin is empty.
    Repeated calls recompute from original business columns, so are idempotent.
    """
    required = {"business_name", "business_address"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    name_norm = normalize_dotted_legal_forms(
        expand_tokens(clean_text(pl.col("business_name")), NAME_ABBREVIATIONS)
    )
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
    return with_fields.with_columns(
        interim_latin_name(pl.col("name_core"), pl.col("script")).alias("name_latin")
    )
