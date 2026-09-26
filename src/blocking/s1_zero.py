"""S1-side model: probability that an S1 record has no match at all (5.6% of train S1).

Features (label-free, no country feature):
- the S1 record: name / address lengths and token counts, digits, legal form, compound house number,
  how many S1 of the country share the cleaned name, how many pool records have that exact name;
- reverse search seen from the S1 side (rev_{split}.parquet): how many pool records have this S1
  in their top 10 / as their best S1, best and second-best score of those records, how many are
  above 0.7 / 0.9, best namenum score;
- orphan model on the records that pick this S1 as their best S1: expected number of matched
  records (sum of 1 - orphan_prob), lowest orphan_prob, how many below 0.5. Train uses the
  S1-grouped out-of-fold orphan scores (orphan_train_grouped.parquet), test uses orphan_test.

Train S1 get 5-fold out-of-fold scores with folds by crc32(s1_id) % 5 (the same S1 grouping as
the grouped orphan OOF); test S1 are scored by one model fitted on train S1.

Output: data/cand/s1_zero_{split}.parquet (s1_id, p_zero) and data/cand/s1_zero_auc.json.

Usage: python -m src.blocking.s1_zero
"""
import json
import time
import zlib

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from src.common.io import load_ground_truth, scan_source
from .orphan import LEGAL, prior_shift, prob_summary
from .stage1 import CAND_DIR, HOUSE_RE  # imports sparse_dot_topn, which must precede lightgbm on macOS
from .text import clean_expr

import lightgbm as lgb

FEATURES = [
    "name_len", "name_tokens", "name_digits", "name_legal", "addr_len", "addr_parts", "addr_digits",
    "house_compound", "s1_name_twins", "pool_name_count",
    "in_top10", "in_rank1", "in_best", "in_second", "in_n70", "in_n90", "in_r1_n70", "in_best_namenum",
    "r1_exp_matched", "r1_min_orphan", "r1_n_orphan_lt50",
]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=300,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=0)


def s1_frame(split: str) -> pl.DataFrame:
    name, addr = pl.col("business_name").fill_null(""), pl.col("business_address").fill_null("")
    s1 = scan_source(split, 1, columns=["entity_id", "business_name", "business_address", "country"]).select(
        pl.col("entity_id").alias("s1_id"), "country", clean_expr(pl.col("business_name")).alias("cname"),
        name.str.len_chars().alias("name_len"),
        name.str.count_matches(r"\d").alias("name_digits"),
        name.str.contains(LEGAL).cast(pl.Int8).alias("name_legal"),
        addr.str.len_chars().alias("addr_len"),
        addr.str.split(",").list.len().alias("addr_parts"),
        addr.str.count_matches(r"\d").alias("addr_digits"),
        addr.str.to_lowercase().str.contains(HOUSE_RE).cast(pl.Int8).alias("house_compound"),
    ).collect()
    pool_names = pl.concat([
        scan_source(split, s, columns=["business_name", "country"])
        .select("country", clean_expr(pl.col("business_name")).alias("cname")).collect() for s in (2, 3)
    ]).group_by("country", "cname").len("pool_name_count")
    return (
        s1.with_columns(pl.col("cname").str.split(" ").list.len().alias("name_tokens"),
                        (pl.len().over("country", "cname") - 1).alias("s1_name_twins"))
        .join(pool_names, on=["country", "cname"], how="left")
        .with_columns(pl.col("pool_name_count").fill_null(0))
        .drop("cname")
    )


def reverse_side(split: str, orphan_file: str) -> pl.DataFrame:
    rev = pl.scan_parquet(CAND_DIR / f"rev_{split}.parquet")
    agg = (rev.group_by("s1_id").agg(
        pl.len().alias("in_top10"),
        (pl.col("rank") == 1).sum().alias("in_rank1"),
        pl.col("score").max().alias("in_best"),
        pl.col("score").sort(descending=True).get(1, null_on_oob=True).alias("in_second"),
        (pl.col("score") >= 0.7).sum().alias("in_n70"),
        (pl.col("score") >= 0.9).sum().alias("in_n90"),
        ((pl.col("rank") == 1) & (pl.col("score") >= 0.7)).sum().alias("in_r1_n70"),
        pl.col("s_namenum").max().alias("in_best_namenum"),
    ).collect(engine="streaming"))
    orphan = pl.scan_parquet(CAND_DIR / orphan_file)
    r1 = (rev.filter(pl.col("rank") == 1).select("rec_id", "s1_id").join(orphan, on="rec_id", how="left")
          .group_by("s1_id").agg((1 - pl.col("orphan_prob")).sum().alias("r1_exp_matched"),
                                 pl.col("orphan_prob").min().alias("r1_min_orphan"),
                                 (pl.col("orphan_prob") < 0.5).sum().alias("r1_n_orphan_lt50"))
          .collect(engine="streaming"))
    return agg.join(r1, on="s1_id", how="left")


def build(split: str, orphan_file: str) -> pl.DataFrame:
    t = time.time()
    d = s1_frame(split).join(reverse_side(split, orphan_file), on="s1_id", how="left")
    print(f"[{split}] {d.height:,} S1, features in {time.time() - t:.0f}s", flush=True)
    return d


def main() -> None:
    t0 = time.time()
    tr = build("train", "orphan_train_grouped.parquet")
    gt = load_ground_truth().select("s1_id", (pl.col("matched_ids").list.len() == 0).cast(pl.Int8).alias("y"))
    tr = tr.join(gt, on="s1_id", how="left").with_columns(pl.col("y").fill_null(1))
    fold = np.array([zlib.crc32(x.encode()) % 5 for x in tr["s1_id"].to_list()])
    X, y = tr.select(FEATURES).to_numpy().astype(np.float32), tr["y"].to_numpy()
    rate = float(y.mean())
    print(f"train S1 with no match {rate:.4f}", flush=True)
    oof = np.zeros(len(y), dtype=np.float32)
    for f in range(5):
        fit, idx = np.where(fold != f)[0], np.where(fold == f)[0]
        m = lgb.train(PARAMS, lgb.Dataset(X[fit], y[fit], feature_name=FEATURES), num_boost_round=400)
        oof[idx] = m.predict(X[idx])
        print(f"  fold {f}: AUC {roc_auc_score(y[idx], oof[idx]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    country = tr["country"].to_numpy()
    auc = {"all": roc_auc_score(y, oof), **{c: roc_auc_score(y[country == c], oof[country == c])
                                            for c in sorted(set(country))}}
    print("OOF AUC:", {k: round(v, 4) for k, v in auc.items()}, flush=True)
    dist = {"zero": prob_summary(oof[y == 1]), "matched": prob_summary(oof[y == 0])}
    for k, v in dist.items():
        print(f"OOF p_zero, S1 {k}:", {a: round(b, 3) for a, b in v.items()}, flush=True)
    pl.DataFrame({"s1_id": tr["s1_id"], "p_zero": oof}).write_parquet(CAND_DIR / "s1_zero_train.parquet")

    full = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=FEATURES), num_boost_round=400)
    imp = sorted(zip(FEATURES, full.feature_importance("gain")), key=lambda x: -x[1])
    tot = sum(g for _, g in imp)
    print("top features:", [(n, round(g / tot, 3)) for n, g in imp[:10]], flush=True)
    te = build("test", "orphan_test.parquet")
    prob = full.predict(te.select(FEATURES).to_numpy().astype(np.float32)).astype(np.float32)
    pl.DataFrame({"s1_id": te["s1_id"], "p_zero": prob}).write_parquet(CAND_DIR / "s1_zero_test.parquet")
    tc = te["country"].to_numpy()
    test = {}
    for c in sorted(set(tc)):
        pc = prob[tc == c].astype(np.float64)
        test[c] = {"mean_p_zero": float(pc.mean()), "em_zero_share": prior_shift(pc, rate), **prob_summary(pc)}
        print(f"test {c}: mean p_zero {pc.mean():.4f}, EM zero-match share {test[c]['em_zero_share']:.4f}, "
              f"share > 0.5 {test[c]['share_gt_0.5']:.4f}", flush=True)
    (CAND_DIR / "s1_zero_auc.json").write_text(json.dumps(
        {"oof_auc": auc, "train_zero_rate": rate,
         "train_zero_rate_by_country": {c: float(y[country == c].mean()) for c in sorted(set(country))},
         "train_prob_dist": dist, "test": test, "features": FEATURES,
         "feature_gain_share": [(n, round(g / tot, 4)) for n, g in imp]}, indent=1))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
