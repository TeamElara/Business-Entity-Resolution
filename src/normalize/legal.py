"""Detect trailing legal forms without changing the full normalized name."""

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


def legal_suffix_and_core(name_norm: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """Return `(name_core, legal_suffix)` for names with a trailing form.

    A form only counts when at least one name token remains before it. Thus a
    business literally named `Limited` is not reduced to an empty core.
    """
    variants = "|".join(
        re.escape(phrase).replace(r"\ ", r"\s+") for phrase, _ in LEGAL_FORMS
    )
    suffix = name_norm.str.extract(rf"^.+?\s+({variants})$", 1).replace(dict(LEGAL_FORMS))
    core = name_norm.str.replace(rf"^(.+?)\s+(?:{variants})$", "${1}")
    return core, suffix
