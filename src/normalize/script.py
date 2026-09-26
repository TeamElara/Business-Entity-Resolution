"""Unicode script detection for business names."""

import polars as pl

from .transliteration import transliterate_name


def script_expr(name: pl.Expr) -> pl.Expr:
    """Classify names as latin, devanagari, or other.

    Devanagari takes precedence for mixed names; other non-Latin letters are
    classified as other. Empty and digit-only names use latin as the neutral
    fallback. This is independent of country labels.
    """
    raw = name.cast(pl.String).fill_null("")
    return (
        pl.when(raw.str.contains(r"\p{Devanagari}"))
        .then(pl.lit("devanagari"))
        .when(raw.str.contains(r"[\p{L}&&\P{Latin}]"))
        .then(pl.lit("other"))
        .otherwise(pl.lit("latin"))
    )


def interim_latin_name(name_core: pl.Expr, script: pl.Expr) -> pl.Expr:
    """Latin core with accents removed; other scripts need a later pass.

    Devanagari is filled by `with_latin_names` below. Other non-Latin scripts
    retain an empty value rather than being mislabelled as Latin.
    """
    latin = name_core.str.normalize("NFD").str.replace_all(r"\p{M}", "")
    return pl.when(script == "latin").then(latin).otherwise(pl.lit(""))


def with_latin_names(frame: pl.DataFrame, token_map: dict[str, str]) -> pl.DataFrame:
    """Fill name_latin, evaluating Python only on Devanagari rows."""
    base = frame.with_columns(
        interim_latin_name(pl.col("name_core"), pl.col("script")).alias("name_latin")
    )
    if base.is_empty() or not base.select((pl.col("script") == "devanagari").any()).item():
        return base
    indexed = base.with_row_index("__normalize_row")
    hindi = indexed.filter(pl.col("script") == "devanagari").select(
        "__normalize_row",
        pl.col("name_core").map_elements(
            lambda value: transliterate_name(value, token_map), return_dtype=pl.String
        ).alias("__hindi_latin"),
    )
    return (
        indexed.join(hindi, on="__normalize_row", how="left")
        .with_columns(
            pl.when(pl.col("script") == "devanagari")
            .then(pl.col("__hindi_latin"))
            .otherwise(pl.col("name_latin"))
            .alias("name_latin")
        )
        .sort("__normalize_row")
        .drop("__normalize_row", "__hindi_latin")
    )
