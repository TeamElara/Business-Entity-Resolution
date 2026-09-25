"""Blocking v0: top-k S2/S3 candidates per S1 by word TF-IDF cosine on business name + address.

Quick first version for the early handoff to matching. Uses temporary cleaning (lowercase,
punctuation to spaces) until the normalized parquet files are available. Runs one country at a
time; countries are read from the S1 file, never hard-coded, so test-only countries are handled
like any other.

Output: data/cand/{split}.parquet with columns s1_id, cand_id, sources, block_score, block_rank.

Usage (from the repo root):
    python -m src.blocking.tfidf_v0 --split train --k 20
    python -m src.blocking.tfidf_v0 --split train --countries India --val-only   # dev run
    python -m src.blocking.tfidf_v0 --split test --k 20

Why name + address, word tokens: on 3k India val S1, name-only char 3-grams reach 54% pair
recall at top-20, name + address words 92% (US 97%), and word tokens are ~10x faster.
max_df drops tokens found in more than that share of the pool (street types, "private",
"limited"): 0.01 keeps search at ~1.5 ms/query for a ~2 point recall loss vs 0.05.
"""
import argparse
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from src.common.io import REPO_ROOT, load_ground_truth, scan_source
from src.common.metrics import blocking_report
from src.common.split import is_val

CAND_DIR = REPO_ROOT / "data" / "cand"


def clean_text(col: pl.Expr) -> pl.Expr:
    """Temporary cleaning: lowercase, non letter/digit runs to one space.

    Letters of any script (Devanagari, accented Latin) and combining marks are kept.
    """
    return (
        col.fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[^\p{L}\p{N}\p{M}]+", " ")
        .str.strip_chars()
    )


def load_country(split: str, source: int, country: str) -> pl.DataFrame:
    """entity_id + cleaned "name address" text for one source file and one country."""
    return (
        scan_source(split, source, country=country, columns=["entity_id", "business_name", "business_address"])
        .select(
            "entity_id",
            (clean_text(pl.col("business_name")) + " " + clean_text(pl.col("business_address")))
            .str.strip_chars().alias("text"),
        )
        .collect()
    )


def list_countries(split: str) -> list[str]:
    """All country labels present in S1 of the split (null country kept as its own group)."""
    return (
        scan_source(split, 1, columns=["country"])
        .select(pl.col("country").drop_nulls().unique())
        .collect()["country"]
        .to_list()
    )


def topk_cosine(q_texts: list[str], p_texts: list[str], k: int, chunk: int, n_threads: int,
                max_df: float = 0.01):
    """(query_idx, pool_idx, score) for the top-k cosine matches of every query text.

    Word TF-IDF is fitted on the pool (S2+S3 of one country). Tokens in more than `max_df` of
    the pool are dropped: they rarely separate businesses and dominate the cost of the sparse
    product. Queries are scored in chunks to bound memory.
    """
    vec = TfidfVectorizer(
        analyzer="word", token_pattern=r"\S+", min_df=2, max_df=max_df,
        sublinear_tf=True, dtype=np.float32,
    )
    P = vec.fit_transform(p_texts)
    Q = vec.transform(q_texts)
    PT = P.T.tocsr()
    del P
    qi, pi, sc = [], [], []
    for start in range(0, Q.shape[0], chunk):
        R = sp_matmul_topn(Q[start:start + chunk], PT, top_n=k, threshold=0.0, n_threads=n_threads).tocoo()
        qi.append(R.row.astype(np.int64) + start)
        pi.append(R.col.astype(np.int64))
        sc.append(R.data.astype(np.float32))
    if not qi:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.float32)
    return np.concatenate(qi), np.concatenate(pi), np.concatenate(sc)


def peak_ram_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6  # bytes on macOS, KB on Linux


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--k", type=int, default=20, help="candidates kept per S1")
    ap.add_argument("--max-df", type=float, default=0.01, help="drop tokens in more than this share of the pool")
    ap.add_argument("--chunk", type=int, default=20000, help="S1 queries scored per batch")
    ap.add_argument("--countries", nargs="*", help="restrict to these countries (default: all in S1)")
    ap.add_argument("--val-only", action="store_true", help="only score val S1 (train split, dev runs)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=None, help="output parquet (default data/cand/{split}.parquet)")
    args = ap.parse_args()

    t0 = time.time()
    countries = args.countries or list_countries(args.split)
    print(f"countries: {countries}")

    parts, s1_seen = [], []
    for country in countries:
        t1 = time.time()
        q = load_country(args.split, 1, country)
        if args.val_only:
            q = q.filter(pl.Series([is_val(x) for x in q["entity_id"].to_list()], dtype=pl.Boolean))
        p = pl.concat([load_country(args.split, s, country) for s in (2, 3)])
        s1_seen.append(q.select(pl.col("entity_id").alias("s1_id"), pl.lit(country).alias("country")))
        if q.height == 0 or p.height == 0:
            print(f"[{country}] skipped (S1 {q.height:,}, pool {p.height:,})")
            continue
        qi, pi, sc = topk_cosine(q["text"].to_list(), p["text"].to_list(), args.k, args.chunk,
                                 args.threads, args.max_df)
        parts.append(pl.DataFrame({
            "s1_id": q["entity_id"].gather(qi),
            "cand_id": p["entity_id"].gather(pi),
            "block_score": sc,
        }))
        print(f"[{country}] S1 {q.height:,} x pool {p.height:,} -> {len(qi):,} pairs "
              f"in {time.time() - t1:.0f}s (peak RAM {peak_ram_gb():.1f} GB)")
        del q, p

    cands = (
        pl.concat(parts)
        .with_columns(sources=pl.lit("nameaddr_tfidf"))
        .with_columns(block_rank=pl.col("block_score").rank("ordinal", descending=True).over("s1_id").cast(pl.Int32))
        .select("s1_id", "cand_id", "sources", "block_score", "block_rank")
        .sort("s1_id", "block_rank")
    )
    out = Path(args.out) if args.out else CAND_DIR / f"{args.split}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    cands.write_parquet(out)
    s1_all = pl.concat(s1_seen)
    print(f"wrote {out}: {cands.height:,} pairs, {cands['s1_id'].n_unique():,}/{s1_all.height:,} S1 with candidates | "
          f"{time.time() - t0:.0f}s total, peak RAM {peak_ram_gb():.1f} GB")

    if args.split == "train":
        val = s1_all.filter(pl.Series([is_val(x) for x in s1_all["s1_id"].to_list()], dtype=pl.Boolean))
        truth = load_ground_truth().join(val.select("s1_id"), on="s1_id", how="semi")
        blocking_report(cands, truth, s1_country=val, name=f"tfidf_v0 k={args.k}")


if __name__ == "__main__":
    main()
