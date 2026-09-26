"""Stage 1 (wide, internal): union of three TF-IDF blocks per country.

Blocks, all fitted on the S2+S3 pool of one country and queried with that country's S1:
- word:   word TF-IDF on prepared "name address" text (main signal; the address is the most
          stable field across sources).
- skel:   the same text as consonant skeletons (absorbs typos and romanization differences).
- noaddr: name skeleton against only the pool records without an address, which the two
          blocks above rank poorly.
- namenum: name + the numbers of the address only. A long S1 address dilutes the word block;
          same name + same house number is found here (India val: +2.2 pts stage-1 recall).

Countries come from the S1 file, never from a fixed list. Outputs (gitignored):
- data/cand/stage1_{split}.parquet: s1_id, cand_id, s_word, r_word, s_skel, r_skel, s_noaddr, r_noaddr
- data/cand/prep_{split}.parquet:   prepared text per record, reused by the pruner features

Usage:
    python -m src.blocking.stage1 --split train
    python -m src.blocking.stage1 --split train --countries India --val-only   # dev run
"""
import argparse
import os
import time
import zlib

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from src.common.io import REPO_ROOT, scan_source
from src.common.split import is_val
from .text import NON_LATIN, clean_expr, latinize, load_script_map, skeletonize
from .tfidf_v0 import list_countries, peak_ram_gb

CAND_DIR = REPO_ROOT / "data" / "cand"
BLOCKS = {  # name: (text column, pool filter column or None, top-k, max_df)
    "word": ("text", None, 20, 0.01),
    "skel": ("skel", None, 20, 0.01),
    "noaddr": ("skel_name", "addr_empty", 10, 0.05),
    "namenum": ("namenum", None, 10, 0.05),
}


def prepare(split: str, source: int, country: str, tmap: dict) -> pl.DataFrame:
    """Prepared text for one source file and one country.

    Columns: entity_id, name, addr, text, skel, skel_name, namenum, addr_empty, nonlatin.
    """
    d = (
        scan_source(split, source, country=country, columns=["entity_id", "business_name", "business_address"])
        .select(
            "entity_id",
            clean_expr(pl.col("business_name")).alias("name"),
            clean_expr(pl.col("business_address")).alias("addr"),
            pl.col("business_name").fill_null("").str.contains(NON_LATIN).alias("nonlatin"),
        )
        .collect()
    )
    d = d.with_columns(name=latinize(d["name"], tmap), addr=latinize(d["addr"], tmap))
    d = d.with_columns(
        (pl.col("name") + " " + pl.col("addr")).str.strip_chars().alias("text"),
        (pl.col("addr") == "").alias("addr_empty"),
        (pl.col("name") + " " + pl.col("addr").str.extract_all(r"\d+").list.join(" ")).str.strip_chars().alias("namenum"),
    )
    return d.with_columns(skel=skeletonize(d["text"]), skel_name=skeletonize(d["name"]))


def fit_tfidf(texts: list[str], max_df: float):
    """Word TF-IDF fitted on pool texts -> (vectorizer, transposed pool matrix).

    Falls back to keeping every token when max_df/min_df would leave none (tiny pools).
    """
    for mn, mx in ((min(2, len(texts)), max_df if len(texts) > 100 else 1.0), (1, 1.0)):
        vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=mn, max_df=mx,
                              sublinear_tf=True, dtype=np.float32)
        try:
            return vec, vec.fit_transform(texts).T.tocsr()
        except ValueError:
            continue
    raise ValueError("no tokens in pool texts")


def topk_block(q: pl.DataFrame, p: pl.DataFrame, col: str, k: int, max_df: float,
               chunk: int, n_threads: int) -> pl.DataFrame:
    """Top-k cosine pool matches of every query on column `col`: s1_id, cand_id, score, rank."""
    if q.height == 0 or p.height == 0:
        return pl.DataFrame(schema={"s1_id": pl.String, "cand_id": pl.String,
                                    "score": pl.Float32, "rank": pl.UInt32})
    vec, PT = fit_tfidf(p[col].to_list(), max_df)
    Q = vec.transform(q[col].to_list())
    qi, pi, sc = [], [], []
    for start in range(0, Q.shape[0], chunk):
        R = sp_matmul_topn(Q[start:start + chunk], PT, top_n=k, threshold=0.0, n_threads=n_threads).tocoo()
        qi.append(R.row.astype(np.int64) + start)
        pi.append(R.col.astype(np.int64))
        sc.append(R.data.astype(np.float32))
    qi, pi, sc = np.concatenate(qi), np.concatenate(pi), np.concatenate(sc)
    return (
        pl.DataFrame({"s1_id": q["entity_id"].gather(qi), "cand_id": p["entity_id"].gather(pi), "score": sc})
        .with_columns(pl.col("score").rank("ordinal", descending=True).over("s1_id").alias("rank"))
    )


def run_country(split: str, country: str, tmap: dict, s1_filter, chunk: int, n_threads: int):
    """Stage-1 pairs and prepared records (S1 + pool) for one country.

    s1_filter: optional function s1_id -> bool selecting the S1 records to query.
    """
    q = prepare(split, 1, country, tmap)
    if s1_filter is not None:
        q = q.filter(pl.Series([s1_filter(x) for x in q["entity_id"].to_list()], dtype=pl.Boolean))
    p = pl.concat([prepare(split, s, country, tmap) for s in (2, 3)])
    pairs = None
    for name, (col, flt, k, max_df) in BLOCKS.items():
        pp = p if flt is None else p.filter(pl.col(flt))
        b = topk_block(q, pp, col, k, max_df, chunk, n_threads).rename({"score": f"s_{name}", "rank": f"r_{name}"})
        pairs = b if pairs is None else pairs.join(b, on=["s1_id", "cand_id"], how="full", coalesce=True)
    return pairs, q, p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--countries", nargs="*", help="restrict to these countries (default: all in S1)")
    ap.add_argument("--val-only", action="store_true", help="only query val S1 (train split, dev runs)")
    ap.add_argument("--dev-sample", action="store_true",
                    help="val S1 plus a 10%% non-val sample (crc32 %% 10 == 1) to train the pruner quickly")
    ap.add_argument("--crc-mod", type=int, nargs="*",
                    help="only S1 with crc32(id) %% 10 in these values (e.g. 1 = a 10%% non-val sample)")
    ap.add_argument("--chunk", type=int, default=20000, help="S1 queries scored per batch")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _dev")
    args = ap.parse_args()

    t0 = time.time()
    tmap = load_script_map()
    countries = args.countries or list_countries(args.split)
    s1_filter = None
    if args.val_only:
        s1_filter = is_val
    elif args.dev_sample:
        s1_filter = lambda x: zlib.crc32(x.encode()) % 10 in (0, 1)
    elif args.crc_mod:
        mods = set(args.crc_mod)
        s1_filter = lambda x: zlib.crc32(x.encode()) % 10 in mods
    pairs, preps = [], []
    for country in countries:
        t1 = time.time()
        pr, q, p = run_country(args.split, country, tmap, s1_filter, args.chunk, args.threads)
        pairs.append(pr)
        preps.append(pl.concat([q, p]).drop("text", "skel", "namenum"))
        print(f"[{country}] S1 {q.height:,} x pool {p.height:,} -> {pr.height:,} pairs "
              f"({pr.height / max(q.height, 1):.1f}/S1) in {time.time() - t1:.0f}s, peak RAM {peak_ram_gb():.1f} GB",
              flush=True)
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    out = CAND_DIR / f"stage1_{args.split}{args.tag}.parquet"
    pl.concat(pairs).write_parquet(out)
    pl.concat(preps).write_parquet(CAND_DIR / f"prep_{args.split}{args.tag}.parquet")
    print(f"wrote {out} in {time.time() - t0:.0f}s, peak RAM {peak_ram_gb():.1f} GB")


if __name__ == "__main__":
    main()
