"""Detect legal forms without changing the full normalized name."""

import re

import polars as pl


# Longest forms first; values are the canonical form retained in legal_suffix.
# The cleaned input is lower-case and punctuation has become spaces.
LEGAL_FORMS = (
    ("private limited", "private limited"),
    ("limited liability company", "llc"),
    ("limited liability partnership", "llp"),
    ("public limited company", "plc"),
    ("one person company", "opc"),
    ("limited", "limited"),
    ("incorporated", "incorporated"),
    ("corporation", "corporation"),
    ("company", "company"),
    ("llc", "llc"),
    ("llp", "llp"),
    ("plc", "plc"),
    ("opc", "opc"),
    ("sarl", "sarl"),
    ("sasu", "sasu"),
    ("sas", "sas"),
    ("eurl", "eurl"),
    ("sci", "sci"),
    ("snc", "snc"),
    ("sa", "sa"),
)

FRANCE_PREFIX_FORMS = ("sarl", "sas", "sasu", "sa", "eurl", "sci", "snc")


def legal_suffix_and_core(
    name_norm: pl.Expr, country: pl.Expr | None = None
) -> tuple[pl.Expr, pl.Expr]:
    """Return `(name_core, legal_suffix)` for names with a legal form.

    A form only counts when at least one name token remains before it. Thus a
    business literally named `Limited` is not reduced to an empty core.
    France also recognizes leading forms when no trailing form was found.
    """
    variants = "|".join(
        re.escape(phrase).replace(r"\ ", r"\s+") for phrase, _ in LEGAL_FORMS
    )
    suffix = name_norm.str.extract(rf"^.+?\s+({variants})$", 1).replace(dict(LEGAL_FORMS))
    core = name_norm.str.replace(rf"^(.+?)\s+(?:{variants})$", "${1}")
    if country is None:
        return core, suffix
    french = country.cast(pl.String).fill_null("").str.to_lowercase() == "france"
    prefix_variants = "|".join(FRANCE_PREFIX_FORMS)
    prefix = name_norm.str.extract(rf"^({prefix_variants})\s+.+$", 1)
    prefix_core = name_norm.str.replace(rf"^(?:{prefix_variants})\s+(.+)$", "${1}")
    use_prefix = french & suffix.is_null() & prefix.is_not_null()
    return (
        pl.when(use_prefix).then(prefix_core).otherwise(core),
        pl.when(use_prefix).then(prefix).otherwise(suffix),
    )
