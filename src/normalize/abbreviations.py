"""Whole-token, field-specific abbreviation maps for Phase 4.

Rules are country-independent so unseen country labels receive the same fallback.
Ambiguous short forms (co, in, sa, no) are deliberately retained. Address 'st'
follows the agreed street convention; it can also mean Saint, so raw text remains
available to downstream matching. French-specific rules belong to Phase 11.
"""

import polars as pl

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


def expand_tokens(expr: pl.Expr, mapping: dict[str, str]) -> pl.Expr:
    """Replace complete space-delimited tokens in already-cleaned text."""
    return expr.str.split(" ").list.eval(pl.element().replace(mapping)).list.join(" ")
