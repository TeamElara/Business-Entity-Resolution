"""Label-free check of a test submission (matching_results.tsv), per country.

For the predicted (S1, record) pairs of a split without labels:
- matches per S1, S1 without a match, and the label-free target from the orphan model
  (records that belong to an S1 = pool x (1 - EM orphan share), see docs/orphan_model.md);
- records claimed by 2+ S1 (at most one S1 is right for each);
- orphan_prob of the predicted records (mean, quantiles, share > 0.5 / 0.8);
- reverse search: is the S1 the record's best S1 (rank 1), in its top 10, or not in it at all;
- "strong" pairs: name token-set similarity >= 90 and address token-set similarity >= 70 on the
  cleaned, latinized text (same idea as the France check in docs/blocking.md).

With --val (train split, validation S1 only) the same numbers are computed for a validation
prediction, plus the real precision of each group, to show what the label-free numbers mean.

Usage:
    python -m src.blocking.v3_check --matches output/matching_results.tsv            # test
    python -m src.blocking.v3_check --matches val_matching_results.tsv --val          # calibration
"""
import argparse
import json

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from src.common.io import load_ground_truth, scan_source, truth_pairs
from src.common.split import is_val
from .stage1 import CAND_DIR
from .text import clean_expr, latinize, load_script_map


def read_matches(path: str) -> pl.DataFrame:
    """matching_results.tsv -> (s1_id, cand_id) pairs."""
    m = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return (m.select(pl.col("source1_entity_id").alias("s1_id"),
                     pl.col("matched_entity_ids").fill_null("").str.split(",").alias("cand_id"))
            .explode("cand_id").filter(pl.col("cand_id") != ""))


def texts(split: str, source: int, tmap: dict) -> pl.DataFrame:
    d = scan_source(split, source, columns=["entity_id", "business_name", "business_address", "country"]).select(
        "entity_id", "country", clean_expr(pl.col("business_name")).alias("name"),
        clean_expr(pl.col("business_address")).alias("addr")).collect()
    return d.with_columns(name=latinize(d["name"], tmap), addr=latinize(d["addr"], tmap))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--val", action="store_true", help="train split, validation S1 only, with precision")
    ap.add_argument("--out", default=None, help="json summary path (default data/cand/v3_check_<split>.json)")
    args = ap.parse_args()

    if args.val:
        args.split = "train"
    tmap = load_script_map()
    s1 = texts(args.split, 1, tmap)
    if args.val:
        s1 = s1.filter(pl.Series([is_val(x) for x in s1["entity_id"].to_list()], dtype=pl.Boolean))
    pool = pl.concat([texts(args.split, s, tmap) for s in (2, 3)])
    pred = read_matches(args.matches)
    s1_ids = s1.select(pl.col("entity_id").alias("s1_id"))
    bad = pred.join(s1_ids, on="s1_id", how="anti").height  # e.g. non-val S1 with --val: dropped
    pred = pred.join(s1_ids, on="s1_id", how="semi")
    orphan = pl.read_parquet(CAND_DIR / f"orphan_{args.split}.parquet")
    rev = pl.scan_parquet(CAND_DIR / f"rev_{args.split}.parquet").select("rec_id", "s1_id", "rank")
    em = json.load(open(CAND_DIR / "orphan_auc.json")).get("test_em_orphan_share", {}) if args.split == "test" else {}

    p = (pred.join(s1.select(pl.col("entity_id").alias("s1_id"), "country",
                             pl.col("name").alias("n1"), pl.col("addr").alias("a1")), on="s1_id", how="left")
         .join(pool.select(pl.col("entity_id").alias("cand_id"), pl.col("name").alias("n2"), pl.col("addr").alias("a2")),
               on="cand_id", how="left")
         .join(orphan.rename({"rec_id": "cand_id"}), on="cand_id", how="left")
         .join(pred.select("s1_id", "cand_id").lazy().join(rev.rename({"rec_id": "cand_id"}), on=["cand_id", "s1_id"])
               .collect(engine="streaming"), on=["s1_id", "cand_id"], how="left")
         .with_columns(pl.len().over("cand_id").alias("n_claims")))
    ns = process.cpdist(p["n1"].to_list(), p["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    as_ = process.cpdist(p["a1"].to_list(), p["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    p = p.with_columns(strong=pl.Series((np.asarray(ns) >= 90) & (np.asarray(as_) >= 70)))
    if args.val:
        tp = truth_pairs(load_ground_truth()).select("s1_id", "cand_id").with_columns(pl.lit(True).alias("ok"))
        p = p.join(tp, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("ok").fill_null(False))
        n_true = tp.join(s1.select(pl.col("entity_id").alias("s1_id"), "country"), on="s1_id")

    n_s1 = s1.group_by("country").len("S1")
    n_pool = pool.group_by("country").len("pool")
    summary = {"matches_file": args.matches, "pairs": p.height, "pairs_dropped_other_S1": bad}
    rows = []
    for c in sorted(n_s1["country"].to_list()):
        q = p.filter(pl.col("country") == c)
        S1 = n_s1.filter(pl.col("country") == c)["S1"].item()
        P = n_pool.filter(pl.col("country") == c)["pool"].item()
        op = q["orphan_prob"].drop_nulls().to_numpy()
        r = {
            "country": c, "S1": S1, "pairs": q.height, "matches_per_S1": q.height / S1,
            "S1_no_match_%": 100 * (1 - q["s1_id"].n_unique() / S1),
            "target_matches_per_S1_EM": P * (1 - em[c]) / S1 if c in em else None,
            "records_claimed_2plus": int(q.filter(pl.col("n_claims") > 1)["cand_id"].n_unique()),
            "orphan_prob_mean": float(op.mean()) if len(op) else None,
            "orphan_prob_q50": float(np.median(op)) if len(op) else None,
            "orphan_prob_q90": float(np.quantile(op, 0.9)) if len(op) else None,
            "orphan_prob_gt_0.5_%": 100 * float((op > 0.5).mean()) if len(op) else None,
            "orphan_prob_gt_0.8_%": 100 * float((op > 0.8).mean()) if len(op) else None,
            "expected_orphans_in_pred": float(op.sum()),
            "rev_rank1_%": 100 * float((q["rank"] == 1).fill_null(False).mean()),
            "rev_top10_%": 100 * float(q["rank"].is_not_null().mean()),
            "strong_%": 100 * float(q["strong"].mean()),
        }
        if args.val:
            prec = lambda f: float(q.filter(f)["ok"].mean()) if q.filter(f).height else None  # noqa: E731
            r.update({
                "true_matches_per_S1": n_true.filter(pl.col("country") == c).height / S1,
                "precision": prec(pl.lit(True)),
                "precision_strong": prec(pl.col("strong")), "precision_not_strong": prec(~pl.col("strong")),
                "precision_orphan_gt_0.8": prec(pl.col("orphan_prob") > 0.8),
                "precision_orphan_le_0.8": prec(pl.col("orphan_prob") <= 0.8),
                "precision_rev_rank1": prec(pl.col("rank") == 1),
                "precision_not_in_rev_top10": prec(pl.col("rank").is_null()),
                "precision_claimed_2plus": prec(pl.col("n_claims") > 1),
            })
        rows.append(r)
    summary["by_country"] = rows
    t = pl.DataFrame(rows)
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=250, float_precision=3):
        print(t.transpose(include_header=True, header_name="stat", column_names="country"))
    out = args.out or str(CAND_DIR / f"v3_check_{'val' if args.val else args.split}.json")
    json.dump(summary, open(out, "w"), indent=1)
    print(f"pairs {p.height:,} (dropped, S1 outside the checked set: {bad:,}) -> {out}")


if __name__ == "__main__":
    main()
