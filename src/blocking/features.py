"""Cheap pair features for the stage-2 pruner.

All features are country-agnostic: string similarities of the prepared (cleaned, Latin) name
and address, shared house/street numbers, block scores and ranks, and a few record-level flags.
Similarities are computed with rapidfuzz's multithreaded element-wise `cpdist`.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

BLOCK_COLS = ["s_word", "r_word", "s_skel", "r_skel", "s_noaddr", "r_noaddr", "s_namenum", "r_namenum", "s_concat", "r_concat"]
FEATURES = [
    *BLOCK_COLS, "n_blocks", "s_word_rel", "s_skel_rel", "n_cands",
    "name_tsr", "name_ratio", "name_partial", "name_skel_tsr",
    "addr_tsr", "addr_ratio", "num_jacc", "num_shared", "s1_nums", "c_nums",
    "c_addr_empty", "c_name_len", "s1_name_len", "c_nonlatin", "c_is_s3",
]


def _sim(a: pl.Series, b: pl.Series, scorer) -> np.ndarray:
    return cpdist(a.fill_null("").to_list(), b.fill_null("").to_list(), scorer=scorer,
                  workers=-1, dtype=np.float32)


def add_features(pairs: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    """pairs: s1_id, cand_id + BLOCK_COLS (nulls allowed). s1 / pool: entity_id, name, addr,
    skel_name, nonlatin, addr_empty. Returns pairs with FEATURES added."""
    rec = ["entity_id", "name", "addr", "skel_name"]
    d = (
        pairs
        .join(s1.select(rec).rename(lambda c: f"q_{c}"), left_on="s1_id", right_on="q_entity_id", how="left")
        .join(pool.select(*rec, "nonlatin", "addr_empty").rename(lambda c: f"c_{c}"),
              left_on="cand_id", right_on="c_entity_id", how="left")
    )
    num = r"\d+"
    d = d.with_columns(
        pl.col("q_addr").str.extract_all(num).list.unique().alias("_qn"),
        pl.col("c_addr").str.extract_all(num).list.unique().alias("_cn"),
    )
    d = d.with_columns(
        pl.col("_qn").list.set_intersection("_cn").list.len().alias("num_shared"),
        pl.col("_qn").list.len().alias("s1_nums"),
        pl.col("_cn").list.len().alias("c_nums"),
    ).with_columns(
        (pl.col("num_shared") / (pl.col("s1_nums") + pl.col("c_nums") - pl.col("num_shared")).clip(1))
        .alias("num_jacc")
    )
    d = d.with_columns(
        name_tsr=_sim(d["q_name"], d["c_name"], fuzz.token_set_ratio),
        name_ratio=_sim(d["q_name"], d["c_name"], fuzz.ratio),
        name_partial=_sim(d["q_name"], d["c_name"], fuzz.partial_ratio),
        name_skel_tsr=_sim(d["q_skel_name"], d["c_skel_name"], fuzz.token_set_ratio),
        addr_tsr=_sim(d["q_addr"], d["c_addr"], fuzz.token_set_ratio),
        addr_ratio=_sim(d["q_addr"], d["c_addr"], fuzz.ratio),
    )
    d = d.with_columns(
        pl.sum_horizontal(pl.col(c).is_not_null() for c in ("s_word", "s_skel", "s_noaddr", "s_namenum", "s_concat")).alias("n_blocks"),
        (pl.col("s_word") / pl.col("s_word").max().over("s1_id")).alias("s_word_rel"),
        (pl.col("s_skel") / pl.col("s_skel").max().over("s1_id")).alias("s_skel_rel"),
        pl.len().over("s1_id").alias("n_cands"),
        pl.col("c_addr_empty").cast(pl.Int8),
        pl.col("c_nonlatin").cast(pl.Int8),
        pl.col("c_name").str.len_chars().alias("c_name_len"),
        pl.col("q_name").str.len_chars().alias("s1_name_len"),
        pl.col("cand_id").str.starts_with("S3").cast(pl.Int8).alias("c_is_s3"),
    )
    return d.select("s1_id", "cand_id", *FEATURES)
