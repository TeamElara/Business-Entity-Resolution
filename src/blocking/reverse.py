"""Reverse search: for every S2/S3 record, its top-k S1 records of the same country.

Uses the two strongest stage-1 blocks (word, namenum), with TF-IDF fitted on the S1 side, and
queries every S2/S3 record in chunks. The two lists are unioned and ranked by the best of the two
cosines. Feeds the matching "competition" features (is this S1 the record's best S1?) and the
orphan model.

Output (streamed, bounded memory): data/cand/rev_{split}.parquet with
rec_id, s1_id, score, rank, s_word, s_namenum, country.

Usage:
    python -m src.blocking.reverse --split train
    python -m src.blocking.reverse --split test
"""
import argparse
import os
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sparse_dot_topn import sp_matmul_topn

from .stage1 import BLOCKS, CAND_DIR, fit_tfidf, prepare
from .text import load_script_map
from .tfidf_v0 import list_countries, peak_ram_gb

REV_BLOCKS = ("word", "namenum")


def reverse_chunk(p: pl.DataFrame, fitted: dict, k: int, n_threads: int) -> pl.DataFrame:
    """Top-k S1 per pool record for one chunk: union of the blocks, ranked by the best score."""
    out = None
    for name, (vec, S1T, s1_ids, col) in fitted.items():
        R = sp_matmul_topn(vec.transform(p[col].to_list()), S1T, top_n=k, threshold=0.0,
                           n_threads=n_threads).tocoo()
        b = pl.DataFrame({
            "rec_id": p["entity_id"].gather(R.row.astype(np.int64)),
            "s1_id": s1_ids.gather(R.col.astype(np.int64)),
            f"s_{name}": R.data.astype(np.float32),
        })
        out = b if out is None else out.join(b, on=["rec_id", "s1_id"], how="full", coalesce=True)
    return (
        out.with_columns(pl.max_horizontal(*[f"s_{b}" for b in fitted]).alias("score"))
        .with_columns(pl.col("score").rank("ordinal", descending=True).over("rec_id").cast(pl.Int16).alias("rank"))
        .filter(pl.col("rank") <= k)
        .select("rec_id", "s1_id", "score", "rank", *[f"s_{b}" for b in fitted])
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--k", type=int, default=10, help="S1 kept per S2/S3 record")
    ap.add_argument("--chunk", type=int, default=200000, help="pool records per query batch")
    ap.add_argument("--countries", nargs="*", help="restrict to these countries (default: all in S1)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()

    t0 = time.time()
    tmap = load_script_map()
    countries = args.countries or list_countries(args.split)
    path = CAND_DIR / f"rev_{args.split}.parquet"
    tmp = path.with_suffix(".parquet.partial")
    writer, n_rows = None, 0
    for country in countries:
        t1 = time.time()
        q = prepare(args.split, 1, country, tmap)
        fitted = {}
        for name in REV_BLOCKS:
            col, _, _, max_df = BLOCKS[name]
            vec, S1T = fit_tfidf(q[col].to_list(), max_df)
            fitted[name] = (vec, S1T, q["entity_id"], col)
        n_pool = 0
        for source in (2, 3):
            p = prepare(args.split, source, country, tmap).select("entity_id", *[BLOCKS[b][0] for b in REV_BLOCKS])
            n_pool += p.height
            for start in range(0, p.height, args.chunk):
                part = reverse_chunk(p.slice(start, args.chunk), fitted, args.k, args.threads) \
                    .with_columns(pl.lit(country).alias("country"))
                table = part.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(tmp, table.schema)
                writer.write_table(table)
                n_rows += part.height
            del p
        print(f"[{country}] S1 {q.height:,}, S2+S3 {n_pool:,} queried in {time.time() - t1:.0f}s, "
              f"peak RAM {peak_ram_gb():.1f} GB, rows so far {n_rows:,}", flush=True)
        del q, fitted
    writer.close()
    tmp.replace(path)
    print(f"wrote {path}: {n_rows:,} rows in {time.time() - t0:.0f}s, peak RAM {peak_ram_gb():.1f} GB", flush=True)


if __name__ == "__main__":
    main()
