"""Phase 3: deterministic basic cleaning without abbreviation expansion."""

import polars as pl


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

    Raw columns preserve original strings, including nulls. This Phase 3
    implementation does not yet produce the full final normalization contract.
    Repeated calls recompute from original business columns, so are idempotent.
    """
    required = {"business_name", "business_address"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    return df.with_columns(
        pl.col("business_name").alias("name_raw"),
        pl.col("business_address").alias("address_raw"),
        clean_text(pl.col("business_name")).alias("name_norm"),
        clean_text(pl.col("business_address")).alias("addr_norm"),
    )
