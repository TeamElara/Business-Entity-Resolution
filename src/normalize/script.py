"""Unicode script detection for business names."""

import polars as pl


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
    """Latin core with accents removed; other scripts await Phase 10.

    The schema requires a string `name_latin` now. An empty value represents
    an unavailable transliteration rather than falsely labelling Indic text
    as Latin. Later phases will fill these values.
    """
    latin = name_core.str.normalize("NFD").str.replace_all(r"\p{M}", "")
    return pl.when(script == "latin").then(latin).otherwise(pl.lit(""))
