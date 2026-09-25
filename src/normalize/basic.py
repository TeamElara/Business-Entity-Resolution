"""Deterministic basic cleaning and Phase 4 abbreviation expansion."""

import polars as pl

from .abbreviations import (
    ADDRESS_ABBREVIATIONS,
    NAME_ABBREVIATIONS,
    expand_tokens,
    normalize_dotted_legal_forms,
)
from .legal import legal_suffix_and_core


def clean_text(expr: pl.Expr) -> pl.Expr:
    """Clean text while preserving Unicode letters, digits and combining marks.

    Missing text becomes empty. Punctuation and symbols become boundaries;
    marks must survive because Indic vowel signs carry meaning.
    """
    return (
        expr.cast(pl.String)
        .fill_null("")
        .str.normalize("NFKC")
        .str.to_lowercase()
        .str.replace_all("&", " and ", literal=True)
        .str.replace_all(r"[^\p{L}\p{M}\p{N}\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def normalize_df(df: pl.DataFrame) -> pl.DataFrame:
    """Preserve inputs and add raw/cleaned columns for names and addresses.

    Raw columns preserve original strings, including nulls. This Phase 5
    implementation does not yet produce the full final normalization contract.
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
    return base.with_columns(name_core.alias("name_core"), legal_suffix.alias("legal_suffix"))
