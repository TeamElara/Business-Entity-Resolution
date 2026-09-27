"""Orphan model: probability that an S2/S3 record matches no S1 record.

Train label: the record appears in no ground-truth match list. Features (no country feature):
- record only: name / address lengths and token counts, domain-like or one-token name, phone, ID,
  "doing business as", junk prefix, digits in the name, legal form, non-Latin script, blank address,
  compound house number, source (S3).
- relation to the split's own inputs (label-free): share of name tokens found in the S1 names of the
  same country, rarest token frequency, exact cleaned name present in S1 (and how often), pool twins
  (same cleaned name; same name + address numbers).
- reverse search (rev_{split}.parquet): best S1 score, second best, gap, number of S1 above 0.5 /
  0.7 / 0.9, best word and namenum scores.

Train records get 5-fold out-of-fold scores (folds by crc32 of the record id); test records are
scored by a model fitted on all train records.

Output: data/cand/orphan_{split}.parquet with rec_id, orphan_prob. AUC is printed and saved.

--group-by-s1: folds grouped by the owner S1 instead (all records of one S1 in the same fold,
crc32(s1_id) % 5; orphans by crc32(rec_id) % 5), so an S2/S3 twin of a held-out record is never in
the training folds. Train only: writes orphan_train_grouped.parquet and orphan_auc_grouped.json.

Usage: python -m src.blocking.orphan [--group-by-s1]
"""
import argparse
import json
import time
import zlib

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from src.common.io import load_ground_truth, scan_source, truth_pairs
from .stage1 import CAND_DIR, HOUSE_RE  # imports sparse_dot_topn, which must precede lightgbm on macOS
from .text import NON_LATIN, clean_expr

import lightgbm as lgb

LEGAL = r"(?i)\b(private|limited|pvt|ltd|llc|l\.l\.c|llp|inc|corp|corporation|company|co\.|plc|sarl|sas|sasu|eurl|sa|sci|snc)\b"
FEATURES = [
    "name_len", "name_tokens", "name_one_token", "name_web", "name_phone", "name_id", "name_dba",
    "name_junk", "name_digits", "name_legal", "nonlatin", "addr_blank", "addr_len", "addr_parts",
    "addr_digits", "house_compound", "is_s3",
    "tok_in_s1_share", "tok_min_s1_df", "name_in_s1", "name_s1_count", "pool_name_twins", "pool_key_twins",
    "rev_best", "rev_second", "rev_gap", "rev_n50", "rev_n70", "rev_n90", "rev_best_word", "rev_best_namenum", "rev_n",
]
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=500,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1,
              num_threads=8, deterministic=True, force_row_wise=True)  # same result on every machine


def record_frame(split: str) -> pl.DataFrame:
    """Record-level and label-free relation features for all S2/S3 records of a split."""
    def base(source):
        name, addr = pl.col("business_name").fill_null(""), pl.col("business_address").fill_null("")
        return scan_source(split, source, columns=["entity_id", "business_name", "business_address", "country"]).select(
            pl.col("entity_id").alias("rec_id"), "country",
            clean_expr(pl.col("business_name")).alias("cname"),
            addr.str.extract_all(r"\d+").list.eval(pl.element().str.replace(r"^0+(\d)", "$1")).list.sort().list.join(" ").alias("digits"),
            name.str.len_chars().alias("name_len"),
            name.str.contains(r"(?i)\.(com|in|net|org|fr)\b|www\.").cast(pl.Int8).alias("name_web"),
            name.str.contains(r"\d{7,}").cast(pl.Int8).alias("name_phone"),
            name.str.contains(r"(?i)\(id:|#\d{3,}").cast(pl.Int8).alias("name_id"),
            name.str.contains(r"(?i)doing business as|\bdba\b|\bt/a\b").cast(pl.Int8).alias("name_dba"),
            name.str.contains(r"^\s*[-<>#*@._~]+").cast(pl.Int8).alias("name_junk"),
            name.str.count_matches(r"\d").alias("name_digits"),
            name.str.contains(LEGAL).cast(pl.Int8).alias("name_legal"),
            name.str.contains(NON_LATIN).cast(pl.Int8).alias("nonlatin"),
            (addr.str.strip_chars() == "").cast(pl.Int8).alias("addr_blank"),
            addr.str.len_chars().alias("addr_len"),
            addr.str.split(",").list.len().alias("addr_parts"),
            addr.str.count_matches(r"\d").alias("addr_digits"),
            addr.str.to_lowercase().str.contains(HOUSE_RE).cast(pl.Int8).alias("house_compound"),
            pl.lit(1 if source == 3 else 0, pl.Int8).alias("is_s3"),
        ).collect()

    pool = pl.concat([base(2), base(3)])
    pool = pool.with_columns(
        pl.col("cname").str.split(" ").list.len().alias("name_tokens"),
        (pl.col("cname").str.split(" ").list.len() == 1).cast(pl.Int8).alias("name_one_token"),
    )
    s1 = scan_source(split, 1, columns=["business_name", "country"]).select(
        "country", clean_expr(pl.col("business_name")).alias("cname")).collect()
    s1_names = s1.group_by("country", "cname").len("name_s1_count")
    s1_tok = (s1.select("country", pl.col("cname").str.split(" ").alias("tok")).explode("tok")
              .group_by("country", "tok").len("tok_df"))
    tok = (pool.select("rec_id", "country", pl.col("cname").str.split(" ").alias("tok")).explode("tok")
           .join(s1_tok, on=["country", "tok"], how="left")
           .group_by("rec_id").agg(pl.col("tok_df").is_not_null().mean().alias("tok_in_s1_share"),
                                   pl.col("tok_df").fill_null(0).min().alias("tok_min_s1_df")))
    pool = (
        pool.join(tok, on="rec_id", how="left")
        .join(s1_names, on=["country", "cname"], how="left")
        .with_columns(pl.col("name_s1_count").fill_null(0),
                      (pl.col("name_s1_count").fill_null(0) > 0).cast(pl.Int8).alias("name_in_s1"))
        .with_columns((pl.len().over("country", "cname") - 1).alias("pool_name_twins"),
                      (pl.len().over("country", "cname", "digits") - 1).alias("pool_key_twins"))
    )
    return pool.drop("cname", "digits")


def reverse_stats(split: str) -> pl.DataFrame:
    """Per-record summary of the reverse-search top-10 S1 list."""
    return (
        pl.scan_parquet(CAND_DIR / f"rev_{split}.parquet")
        .group_by("rec_id")
        .agg(
            pl.col("score").max().alias("rev_best"),
            pl.col("score").filter(pl.col("rank") == 2).first().alias("rev_second"),
            (pl.col("score") >= 0.5).sum().alias("rev_n50"),
            (pl.col("score") >= 0.7).sum().alias("rev_n70"),
            (pl.col("score") >= 0.9).sum().alias("rev_n90"),
            pl.col("s_word").max().alias("rev_best_word"),
            pl.col("s_namenum").max().alias("rev_best_namenum"),
            pl.len().alias("rev_n"),
        )
        .with_columns((pl.col("rev_best") - pl.col("rev_second").fill_null(0)).alias("rev_gap"))
        .collect(engine="streaming")
    )


def prob_summary(p: np.ndarray) -> dict:
    """Mean, quantiles and high-probability shares of a set of orphan probabilities."""
    qs = (0.1, 0.25, 0.5, 0.75, 0.9)
    out = {"n": int(len(p)), "mean": float(p.mean())}
    out.update({f"q{int(q * 100)}": float(v) for q, v in zip(qs, np.quantile(p, qs))})
    out.update({"share_gt_0.5": float((p > 0.5).mean()), "share_gt_0.8": float((p > 0.8).mean())})
    return out


def prior_shift(p: np.ndarray, train_rate: float, iters: int = 200) -> float:
    """Orphan share of an unlabeled set, re-estimated by EM from probabilities calibrated on train
    (Saerens et al. 2002). The raw mean stays pulled towards the train rate when the shares differ."""
    pi = float(p.mean())
    for _ in range(iters):
        a = p * (pi / train_rate)
        b = (1 - p) * ((1 - pi) / (1 - train_rate))
        pi = float((a / (a + b)).mean())
    return pi


def build(split: str) -> pl.DataFrame:
    t = time.time()
    d = record_frame(split).join(reverse_stats(split), on="rec_id", how="left")
    print(f"[{split}] {d.height:,} records, features in {time.time() - t:.0f}s", flush=True)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group-by-s1", action="store_true", help="OOF folds grouped by owner S1 (train only)")
    args = ap.parse_args()
    t0 = time.time()
    tr = build("train")
    # every record belongs to at most one S1 in the ground truth
    owner = truth_pairs(load_ground_truth()).select(pl.col("cand_id").alias("rec_id"), "s1_id").unique("rec_id")
    tr = tr.join(owner, on="rec_id", how="left").with_columns(pl.col("s1_id").is_null().cast(pl.Int8).alias("y"))
    fold_key = tr["s1_id"].fill_null(tr["rec_id"]) if args.group_by_s1 else tr["rec_id"]
    fold = np.array([zlib.crc32(x.encode()) % 5 for x in fold_key.to_list()])
    X, y = tr.select(FEATURES).to_numpy().astype(np.float32), tr["y"].to_numpy()
    print(f"train orphans {y.mean():.3f}", flush=True)
    oof = np.zeros(len(y), dtype=np.float32)
    rng = np.random.default_rng(0)
    for f in range(5):
        fit = np.where((fold != f) & (rng.random(len(y)) < 0.5))[0]  # half of the other folds, for speed
        m = lgb.train(PARAMS, lgb.Dataset(X[fit], y[fit], feature_name=FEATURES), num_boost_round=300)
        idx = np.where(fold == f)[0]
        oof[idx] = m.predict(X[idx])
        print(f"  fold {f}: AUC {roc_auc_score(y[idx], oof[idx]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    auc = {"all": roc_auc_score(y, oof)}
    for c in tr["country"].unique().to_list():
        msk = (tr["country"] == c).to_numpy()
        auc[c] = roc_auc_score(y[msk], oof[msk])
    print("OOF AUC:", {k: round(v, 4) for k, v in auc.items()}, flush=True)
    tr_country = tr["country"].to_numpy()
    train_dist = {"orphans": prob_summary(oof[y == 1]), "matched": prob_summary(oof[y == 0])}
    for k, v in train_dist.items():
        print(f"OOF orphan_prob, {k}:", {a: round(b, 3) for a, b in v.items()}, flush=True)
    if args.group_by_s1:
        pl.DataFrame({"rec_id": tr["rec_id"], "orphan_prob": oof}).write_parquet(CAND_DIR / "orphan_train_grouped.parquet")
        (CAND_DIR / "orphan_auc_grouped.json").write_text(json.dumps(
            {"folds": "grouped by owner S1 (orphans by rec_id)", "oof_auc": auc,
             "train_mean_oof_prob": {c: float(oof[tr_country == c].mean()) for c in sorted(set(tr_country))},
             "train_prob_dist": train_dist}, indent=1))
        print(f"done (train only) in {time.time() - t0:.0f}s", flush=True)
        return
    pl.DataFrame({"rec_id": tr["rec_id"], "orphan_prob": oof}).write_parquet(CAND_DIR / "orphan_train.parquet")

    fit = np.where(rng.random(len(y)) < 0.6)[0]
    full = lgb.train(PARAMS, lgb.Dataset(X[fit], y[fit], feature_name=FEATURES), num_boost_round=300)
    imp = sorted(zip(FEATURES, full.feature_importance("gain")), key=lambda x: -x[1])
    tot = sum(g for _, g in imp)
    print("top features:", [(n, round(g / tot, 3)) for n, g in imp[:12]], flush=True)
    del tr, X
    te = build("test")
    prob = full.predict(te.select(FEATURES).to_numpy().astype(np.float32))
    pl.DataFrame({"rec_id": te["rec_id"], "orphan_prob": prob.astype(np.float32)}).write_parquet(CAND_DIR / "orphan_test.parquet")
    rate = float(y.mean())
    te_country = te["country"].to_numpy()
    test_dist, test_em = {}, {}
    for c in sorted(set(te_country)):
        pc = prob[te_country == c]
        test_dist[c], test_em[c] = prob_summary(pc), prior_shift(pc, rate)
        print(f"test {c}: mean orphan_prob {pc.mean():.3f}, EM orphan share {test_em[c]:.3f}, "
              f"share > 0.8 {test_dist[c]['share_gt_0.8']:.3f}", flush=True)
    countries = sorted(set(tr_country))
    (CAND_DIR / "orphan_auc.json").write_text(json.dumps(
        {"oof_auc": auc, "train_orphan_rate": rate,
         "train_orphan_rate_by_country": {c: float(y[tr_country == c].mean()) for c in countries},
         "train_mean_oof_prob": {c: float(oof[tr_country == c].mean()) for c in countries},
         "train_prob_dist": train_dist,
         "test_mean_prob": {c: d["mean"] for c, d in test_dist.items()},
         "test_em_orphan_share": test_em, "test_prob_dist": test_dist,
         "features": FEATURES, "feature_gain_share": [(n, round(g / tot, 4)) for n, g in imp]}, indent=1))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
