"""One command for the final candidate set: stage 1 blocks -> pair features -> pruner, per chunk.

For each country (read from S1, never hard-coded) the three TF-IDF blocks of stage1.py are fitted
once on the S2+S3 pool; S1 queries are then processed in chunks: top-k per block, union, pair
features, LightGBM pruner, adaptive cutoff. Only the pruned pairs are kept, so memory stays bounded
on the full splits.

Needs data/cand/pruner_lgb.txt and pruner_cutoff.json (from `python -m src.blocking.pruner train`).

Output: data/cand/{split}.parquet (s1_id, cand_id, sources, block_score, block_rank), the final
candidate set that the matching model scores and candidate_pairs.tsv is written from.
With --save-stage1, the unpruned stage-1 pairs are also written (data/cand/stage1_{split}/).

Usage:
    python -m src.blocking.run --split train
    python -m src.blocking.run --split test
"""
import argparse
import json
import os
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn  # before lightgbm: OpenMP clash hangs on macOS otherwise

import lightgbm as lgb

from src.common.io import load_ground_truth
from src.common.metrics import blocking_report
from src.common.split import is_val
from .features import FEATURES, add_features
from .pruner import CUTOFF_PATH, MODEL_PATH, apply_cutoff
from .stage1 import BLOCKS, CAND_DIR, prepare
from .text import load_script_map
from .tfidf_v0 import list_countries, peak_ram_gb


def fit_blocks(p: pl.DataFrame) -> dict:
    """Fit every block's TF-IDF on the pool once: name -> (vectorizer, pool^T, pool ids)."""
    fitted = {}
    for name, (col, flt, k, max_df) in BLOCKS.items():
        pp = p if flt is None else p.filter(pl.col(flt))
        if pp.height == 0:
            continue
        vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=min(2, pp.height),
                              max_df=max_df if pp.height > 100 else 1.0, sublinear_tf=True, dtype=np.float32)
        PT = vec.fit_transform(pp[col].to_list()).T.tocsr()
        fitted[name] = (vec, PT, pp["entity_id"], col, k)
    return fitted


def stage1_chunk(q: pl.DataFrame, fitted: dict, n_threads: int) -> pl.DataFrame:
    """Union of the block top-k lists for one chunk of S1 queries, one row per pair."""
    pairs = None
    for name, (vec, PT, ids, col, k) in fitted.items():
        R = sp_matmul_topn(vec.transform(q[col].to_list()), PT, top_n=k, threshold=0.0, n_threads=n_threads).tocoo()
        b = pl.DataFrame({
            "s1_id": q["entity_id"].gather(R.row.astype(np.int64)),
            "cand_id": ids.gather(R.col.astype(np.int64)),
            f"s_{name}": R.data.astype(np.float32),
        }).with_columns(pl.col(f"s_{name}").rank("ordinal", descending=True).over("s1_id").alias(f"r_{name}"))
        pairs = b if pairs is None else pairs.join(b, on=["s1_id", "cand_id"], how="full", coalesce=True)
    for name in BLOCKS:  # a block can be absent when its pool subset is empty
        if f"s_{name}" not in pairs.columns:
            pairs = pairs.with_columns(pl.lit(None, pl.Float32).alias(f"s_{name}"),
                                       pl.lit(None, pl.UInt32).alias(f"r_{name}"))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--countries", nargs="*", help="restrict to these countries (default: all in S1)")
    ap.add_argument("--val-only", action="store_true", help="only query val S1 (train split, dev runs)")
    ap.add_argument("--chunk", type=int, default=50000, help="S1 queries per chunk")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--save-stage1", action="store_true", help="also write unpruned stage-1 pairs")
    ap.add_argument("--out", default=None, help="output parquet (default data/cand/{split}.parquet)")
    args = ap.parse_args()

    t0 = time.time()
    tmap = load_script_map()
    model = lgb.Booster(model_file=str(MODEL_PATH))
    cut = json.loads(CUTOFF_PATH.read_text())
    countries = args.countries or list_countries(args.split)
    stage1_dir = CAND_DIR / f"stage1_{args.split}"
    if args.save_stage1:
        stage1_dir.mkdir(parents=True, exist_ok=True)

    kept, s1_seen, n_stage1 = [], [], 0
    for country in countries:
        t1 = time.time()
        q_all = prepare(args.split, 1, country, tmap)
        if args.val_only:
            q_all = q_all.filter(pl.Series([is_val(x) for x in q_all["entity_id"].to_list()], dtype=pl.Boolean))
        p = pl.concat([prepare(args.split, s, country, tmap) for s in (2, 3)])
        s1_seen.append(q_all.select(pl.col("entity_id").alias("s1_id"), pl.lit(country).alias("country")))
        if q_all.height == 0 or p.height == 0:
            print(f"[{country}] skipped (S1 {q_all.height:,}, pool {p.height:,})", flush=True)
            continue
        fitted = fit_blocks(p)
        rec = ["entity_id", "name", "addr", "skel_name", "nonlatin", "addr_empty"]
        pool_rec = p.select(rec)
        for i, start in enumerate(range(0, q_all.height, args.chunk)):
            q = q_all.slice(start, args.chunk)
            pairs = stage1_chunk(q, fitted, args.threads)
            n_stage1 += pairs.height
            if args.save_stage1:
                pairs.write_parquet(stage1_dir / f"{country}_{i:04d}.parquet")
            f = add_features(pairs, q.select(rec), pool_rec)
            scored = f.select(
                "s1_id", "cand_id",
                pl.concat_str([pl.when(pl.col(f"s_{b}").is_not_null()).then(pl.lit(b)) for b in BLOCKS],
                              separator="|", ignore_nulls=True).alias("sources"),
            ).with_columns(prob=pl.Series(model.predict(f.select(FEATURES).to_numpy()), dtype=pl.Float32))
            kept.append(apply_cutoff(scored, cut["min_keep"], cut["min_prob"], cut["rel"], cut["k_max"]))
        print(f"[{country}] S1 {q_all.height:,} x pool {p.height:,} in {time.time() - t1:.0f}s, "
              f"peak RAM {peak_ram_gb():.1f} GB", flush=True)
        del p, pool_rec, fitted

    out = (
        pl.concat(kept)
        .select("s1_id", "cand_id", "sources", pl.col("prob").alias("block_score"),
                pl.col("block_rank").cast(pl.Int32))
        .sort("s1_id", "block_rank")
    )
    path = CAND_DIR / f"{args.split}.parquet" if args.out is None else args.out
    out.write_parquet(path)
    s1_all = pl.concat(s1_seen)
    print(f"wrote {path}: {out.height:,} pairs for {out['s1_id'].n_unique():,}/{s1_all.height:,} S1 "
          f"({out.height / max(s1_all.height, 1):.2f} cands/S1; stage 1 had {n_stage1 / max(s1_all.height, 1):.1f}) | "
          f"{time.time() - t0:.0f}s total, peak RAM {peak_ram_gb():.1f} GB", flush=True)

    if args.split == "train":
        val = s1_all.filter(pl.Series([is_val(x) for x in s1_all["s1_id"].to_list()], dtype=pl.Boolean))
        truth = load_ground_truth().join(val.select("s1_id"), on="s1_id", how="semi")
        blocking_report(out, truth, s1_country=val, name="final (stage 1 + pruner)")


if __name__ == "__main__":
    main()
