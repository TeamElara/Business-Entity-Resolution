"""Whole-token, field-specific abbreviation maps for Phase 4.

Rules are country-independent so unseen country labels receive the same fallback.
Ambiguous short forms (co, in, sa, no) are deliberately retained. Address 'st'
follows the agreed street convention; it can also mean Saint, so raw text remains
available to downstream matching. French-specific rules belong to Phase 11.
"""

import polars as pl

from .basic_text import clean_text

NAME_ABBREVIATIONS = {
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "inc": "incorporated",
}

ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "hwy": "highway",
    "bldg": "building",
    "flr": "floor",
    "apt": "apartment",
}

# Punctuation cleaning splits dotted legal acronyms into separate letters.
# Reassemble only complete, space-delimited sequences; never change letters
# embedded inside another word. Long forms precede their shorter prefixes.
DOTTED_LEGAL_ACRONYMS = (
    ("s a s u", "sasu"),
    ("s a r l", "sarl"),
    ("e u r l", "eurl"),
    ("l l c", "llc"),
    ("l l p", "llp"),
    ("p l c", "plc"),
    ("s c i", "sci"),
    ("s n c", "snc"),
    ("s a s", "sas"),
    ("s a", "sa"),
)


def expand_tokens(expr: pl.Expr, mapping: dict[str, str]) -> pl.Expr:
    """Replace complete space-delimited tokens in already-cleaned text."""
    return expr.str.split(" ").list.eval(pl.element().replace(mapping)).list.join(" ")


def normalize_dotted_legal_forms(expr: pl.Expr) -> pl.Expr:
    """Reassemble legal acronyms split by periods in the original text."""
    for spaced, joined in DOTTED_LEGAL_ACRONYMS:
        expr = expr.str.replace_all(
            rf"(^| ){spaced}($| )", rf"${{1}}{joined}${{2}}"
        )
    return expr


def normalize_name_expr(raw: pl.Expr) -> pl.Expr:
    """Shared name expression for the normalizer and raw-only map training."""
    return normalize_dotted_legal_forms(
        expand_tokens(clean_text(raw), NAME_ABBREVIATIONS)
    )
