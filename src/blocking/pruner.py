"""Stage 2 (pruner): score stage-1 pairs with a small LightGBM model and keep a short,
adaptive list per S1. The result is the final candidate set (data/cand/{split}.parquet).

Model: binary LightGBM on the country-agnostic pair features of features.py, trained on
non-validation train S1 only. Cutoff per S1: rank by probability, always keep the top
`min_keep`, then keep further candidates while prob >= `min_prob` and
prob >= best_prob * `rel`, up to `k_max`.

Usage:
    python -m src.blocking.pruner train --tag _dev          # fit on non-val pairs, grid on val
    python -m src.blocking.pruner apply --split train       # write data/cand/train.parquet
    python -m src.blocking.pruner apply --split test        # write data/cand/test.parquet
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from src.common.io import REPO_ROOT, load_ground_truth, truth_pairs
from src.common.split import is_val
from .features import FEATURES, add_features
from .stage1 import BLOCKS

CAND_DIR = REPO_ROOT / "data" / "cand"
MODEL_PATH = CAND_DIR / "pruner_lgb.txt"
CUTOFF_PATH = CAND_DIR / "pruner_cutoff.json"
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=0)


def featurize(split: str, tag: str, s1_ids=None, chunk_s1: int = 40000):
    """Yield feature frames for stage-1 pairs, processed in chunks of S1 to bound memory."""
    stage1 = pl.read_parquet(CAND_DIR / f"stage1_{split}{tag}.parquet")
    prep = pl.read_parquet(CAND_DIR / f"prep_{split}{tag}.parquet")
    if s1_ids is not None:
        stage1 = stage1.join(pl.DataFrame({"s1_id": s1_ids}), on="s1_id", how="semi")
    ids = stage1["s1_id"].unique(maintain_order=True)
    for start in range(0, len(ids), chunk_s1):
        part = stage1.join(pl.DataFrame({"s1_id": ids[start:start + chunk_s1]}), on="s1_id", how="semi")
        yield add_features(part, prep, prep)


def label(frame: pl.DataFrame, tp: pl.DataFrame) -> pl.DataFrame:
    return frame.join(tp.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1_id", "cand_id"], how="left") \
        .with_columns(pl.col("y").fill_null(0))


def apply_cutoff(scored: pl.DataFrame, min_keep: int, min_prob: float, rel: float, k_max: int) -> pl.DataFrame:
    """scored: s1_id, cand_id, prob. Returns the kept pairs with block_rank (1 = best)."""
    d = scored.with_columns(
        pl.col("prob").rank("ordinal", descending=True).over("s1_id").alias("block_rank"),
        pl.col("prob").max().over("s1_id").alias("_best"),
    )
    keep = (pl.col("block_rank") <= min_keep) | (
        (pl.col("block_rank") <= k_max) & (pl.col("prob") >= min_prob) & (pl.col("prob") >= pl.col("_best") * rel)
    )
    return d.filter(keep).drop("_best")


def oracle_stats(kept: pl.DataFrame, truth: pl.DataFrame, tp: pl.DataFrame) -> dict:
    """Avg candidates per S1, pair recall and oracle F0.5 ceiling over all truth S1."""
    hits = kept.join(tp, on=["s1_id", "cand_id"], how="semi").group_by("s1_id").len("tp")
    per = (
        truth.select("s1_id", pl.col("matched_ids").list.len().alias("n_true"))
        .join(kept.group_by("s1_id").len("n"), on="s1_id", how="left")
        .join(hits, on="s1_id", how="left")
        .with_columns(pl.col("n", "tp").fill_null(0))
    )
    r = pl.col("tp") / pl.col("n_true")
    oracle = pl.when(pl.col("n_true") == 0).then(1.0).when(pl.col("tp") == 0).then(0.0).otherwise(1.25 * r / (0.25 + r))
    return {
        "avg_cands": per["n"].mean(),
        "p95_cands": per["n"].quantile(0.95, "nearest"),
        "pair_recall": per["tp"].sum() / per["n_true"].sum(),
        "oracle": per.select(oracle.mean()).item(),
    }


def cmd_train(args) -> None:
    t0 = time.time()
    truth = load_ground_truth()
    tp = truth_pairs(truth)
    frames = [label(f, tp) for f in featurize("train", args.tag)]
    data = pl.concat(frames)
    val_mask = pl.Series([is_val(x) for x in data["s1_id"].to_list()], dtype=pl.Boolean)
    tr, va = data.filter(~val_mask), data.filter(val_mask)
    print(f"features: {data.height:,} pairs ({tr.height:,} train / {va.height:,} val), "
          f"positives {tr['y'].mean():.3f}, {time.time() - t0:.0f}s", flush=True)

    dtr = lgb.Dataset(tr.select(FEATURES).to_numpy(), tr["y"].to_numpy(), feature_name=FEATURES)
    dva = lgb.Dataset(va.select(FEATURES).to_numpy(), va["y"].to_numpy(), reference=dtr)
    model = lgb.train(PARAMS, dtr, num_boost_round=args.rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)])
    model.save_model(str(MODEL_PATH))
    imp = sorted(zip(FEATURES, model.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(n, round(g / sum(v for _, v in imp), 3)) for n, g in imp[:10]])

    scored = va.select("s1_id", "cand_id").with_columns(
        prob=pl.Series(model.predict(va.select(FEATURES).to_numpy()), dtype=pl.Float32))
    # all val S1 queried in stage 1 count, including those left with no pairs
    vtruth = truth.filter(pl.Series([is_val(x) for x in truth["s1_id"].to_list()], dtype=pl.Boolean)) \
        .join(pl.read_parquet(CAND_DIR / f"prep_train{args.tag}.parquet", columns=["entity_id"])
              .rename({"entity_id": "s1_id"}), on="s1_id", how="semi")
    base = oracle_stats(va.select("s1_id", "cand_id"), vtruth, tp)
    print(f"stage 1 (val): avg {base['avg_cands']:.1f} cands/S1, recall {base['pair_recall']:.4f}, "
          f"oracle {base['oracle']:.4f}")

    rows = []
    for k_max in (8, 10, 12, 15, 20):
        for min_prob in (0.005, 0.01, 0.02, 0.05, 0.1):
            for rel in (0.0, 0.01, 0.03, 0.1):
                for min_keep in (1, 2):
                    st = oracle_stats(apply_cutoff(scored, min_keep, min_prob, rel, k_max), vtruth, tp)
                    rows.append({"k_max": k_max, "min_prob": min_prob, "rel": rel, "min_keep": min_keep, **st})
    grid = pl.DataFrame(rows).sort("avg_cands")
    grid.write_csv(CAND_DIR / "pruner_grid.csv")
    # Pareto front: best oracle for each list size
    front = grid.filter(pl.col("oracle") > pl.col("oracle").shift(1).cum_max().fill_null(-1))
    with pl.Config(tbl_rows=60, tbl_width_chars=200):
        print(front)
    print(f"done in {time.time() - t0:.0f}s")


def cmd_apply(args) -> None:
    t0 = time.time()
    cut = json.loads(CUTOFF_PATH.read_text())
    model = lgb.Booster(model_file=str(MODEL_PATH))
    parts = []
    for f in featurize(args.split, args.tag):
        prob = model.predict(f.select(FEATURES).to_numpy())
        scored = f.select(
            "s1_id", "cand_id",
            pl.concat_str(
                [pl.when(pl.col(f"s_{b}").is_not_null()).then(pl.lit(b)) for b in BLOCKS],
                separator="|", ignore_nulls=True).alias("sources"),
        ).with_columns(prob=pl.Series(prob, dtype=pl.Float32))
        parts.append(apply_cutoff(scored, cut["min_keep"], cut["min_prob"], cut["rel"], cut["k_max"]))
        print(f"  {sum(p.height for p in parts):,} kept pairs, {time.time() - t0:.0f}s", flush=True)
    out = (
        pl.concat(parts)
        .select("s1_id", "cand_id", "sources", pl.col("prob").alias("block_score"),
                pl.col("block_rank").cast(pl.Int32))
        .sort("s1_id", "block_rank")
    )
    path = CAND_DIR / f"{args.split}{args.tag}.parquet"
    out.write_parquet(path)
    print(f"wrote {path}: {out.height:,} pairs, {out['s1_id'].n_unique():,} S1, "
          f"{out.height / out['s1_id'].n_unique():.2f} cands/S1, {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--tag", default="_dev")
    t.add_argument("--rounds", type=int, default=400)
    a = sub.add_parser("apply")
    a.add_argument("--split", choices=["train", "test"], required=True)
    a.add_argument("--tag", default="")
    args = ap.parse_args()
    cmd_train(args) if args.cmd == "train" else cmd_apply(args)


if __name__ == "__main__":
    main()
