"""Validation split, identical for the whole team (master plan §5).

Never use Python's built-in hash() here: it is randomized per process.
"""
import zlib

import polars as pl


def is_val(s1_id: str) -> bool:
    return zlib.crc32(s1_id.encode()) % 10 == 0


def add_is_val(df: pl.DataFrame, col: str = "s1_id") -> pl.DataFrame:
    """Return df with a boolean is_val column computed from the S1 id in `col`."""
    flags = [is_val(x) for x in df[col].to_list()]
    return df.with_columns(pl.Series("is_val", flags, dtype=pl.Boolean))
