"""Readers for the raw challenge TSVs (master plan §4).

Every column is read as a string with quoting disabled (names contain quotes and
literal strings like "NA"). Empty fields and nulls are the same thing: both come
back as null.

Paths default to data/raw (a symlink to student_resource/dataset); override with
the DATA_RAW environment variable.
"""
import os
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path(os.environ.get("DATA_RAW", REPO_ROOT / "data" / "raw"))
SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def _empty_to_null(frame):
    """Turn "" into null in every column so null and "" are treated the same."""
    return frame.with_columns(
        pl.when(pl.col(c) == "").then(None).otherwise(pl.col(c)).alias(c)
        for c in frame.collect_schema().names()
    )


def read_tsv(path) -> pl.DataFrame:
    """Read a TSV eagerly: all columns str, quoting off, "" -> null."""
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0)
    return _empty_to_null(df)


def scan_tsv(path) -> pl.LazyFrame:
    """Lazy version of read_tsv, for filtering (e.g. per country) before loading."""
    lf = pl.scan_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return _empty_to_null(lf)


def source_path(split: str, source: int) -> Path:
    """split = "train" | "test", source = 1 | 2 | 3."""
    return RAW_DIR / split / f"{split}_source{source}.tsv"


def scan_source(split: str, source: int, country=None, columns=None) -> pl.LazyFrame:
    """Lazy scan of one source file, optionally filtered to one country / columns."""
    lf = scan_tsv(source_path(split, source))
    if country is not None:
        lf = lf.filter(pl.col("country") == country)
    if columns is not None:
        lf = lf.select(columns)
    return lf


def load_source(split: str, source: int, country=None, columns=None) -> pl.DataFrame:
    return scan_source(split, source, country, columns).collect()


def load_train_s1(**kw): return load_source("train", 1, **kw)
def load_train_s2(**kw): return load_source("train", 2, **kw)
def load_train_s3(**kw): return load_source("train", 3, **kw)
def load_test_s1(**kw): return load_source("test", 1, **kw)
def load_test_s2(**kw): return load_source("test", 2, **kw)
def load_test_s3(**kw): return load_source("test", 3, **kw)


def load_ground_truth() -> pl.DataFrame:
    """Train labels as (s1_id: str, matched_ids: list[str]); empty list = no match."""
    df = read_tsv(RAW_DIR / "train" / "train_ground_truth.tsv")
    return df.select(
        pl.col("source1_entity_id").alias("s1_id"),
        pl.col("matched_entity_ids")
        .fill_null("")
        .str.split(",")
        .list.eval(pl.element().str.strip_chars().filter(pl.element() != ""))
        .alias("matched_ids"),
    )


def truth_pairs(truth: pl.DataFrame) -> pl.DataFrame:
    """Ground truth exploded to one row per true (s1_id, cand_id) pair."""
    return (
        truth.select("s1_id", pl.col("matched_ids").alias("cand_id"))
        .filter(pl.col("cand_id").list.len() > 0)
        .explode("cand_id", empty_as_null=False)
    )


def count_rows(split: str, source: int) -> int:
    return scan_source(split, source).select(pl.len()).collect().item()
