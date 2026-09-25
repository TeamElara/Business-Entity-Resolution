"""Shared Unicode text cleaning used by names and address components."""

import polars as pl


def clean_text(expr: pl.Expr) -> pl.Expr:
    """Normalize text while preserving Unicode letters, digits, and marks."""
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


def strip_accents(expr: pl.Expr) -> pl.Expr:
    """Fold accents and common French ligatures without changing raw text."""
    return (
        expr.str.normalize("NFD")
        .str.replace_all(r"\p{M}", "")
        .str.replace_all("œ", "oe", literal=True)
        .str.replace_all("æ", "ae", literal=True)
    )
