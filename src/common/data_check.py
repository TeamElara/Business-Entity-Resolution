"""STEP 2 sanity check on the real data: python -m src.common.data_check"""
import time

import polars as pl

from .io import load_ground_truth, scan_source, truth_pairs
from .metrics import blocking_report, macro_f05
from .split import add_is_val


def main():
    t0 = time.time()
    print("== row counts and country distribution ==")
    ids = {}
    for split in ("train", "test"):
        for src in (1, 2, 3):
            df = scan_source(split, src, columns=["entity_id", "country"]).collect()
            ids[(split, src)] = df["entity_id"]
            dist = df["country"].fill_null("<null>").value_counts(sort=True)
            dist_s = ", ".join(f"{c} {n:,} ({n / df.height:.1%})" for c, n in dist.iter_rows())
            print(f"  {split}_source{src}: {df.height:>10,} rows, {df['entity_id'].n_unique():>10,} unique ids | {dist_s}")
            if src == 1 and split == "train":
                s1_country = df.rename({"entity_id": "s1_id"})

    truth = load_ground_truth()
    print(f"  train_ground_truth: {truth.height:>10,} rows, {truth['s1_id'].n_unique():,} unique s1_id")
    same = set(truth["s1_id"].to_list()) == set(ids[("train", 1)].to_list())
    print(f"  ground truth S1 ids == train S1 ids: {same}")

    pairs = truth_pairs(truth)
    pool = pl.concat([ids[("train", 2)], ids[("train", 3)]]).to_frame("cand_id")
    unknown = pairs.join(pool, on="cand_id", how="anti").height
    print(f"  true pairs {pairs.height:,}: S2 {pairs.filter(pl.col('cand_id').str.starts_with('S2-')).height:,}, "
          f"S3 {pairs.filter(pl.col('cand_id').str.starts_with('S3-')).height:,}, not found in train S2/S3: {unknown:,}")
    multi = pairs.group_by("cand_id").len().filter(pl.col("len") > 1).height
    print(f"  S2/S3 ids matched to more than one S1: {multi:,}")

    print("\n== match-count distribution (train ground truth) ==")
    mc = truth.select(pl.col("matched_ids").list.len().alias("n")).group_by("n").len().sort("n")
    for n, c in mc.iter_rows():
        print(f"  {n:>3} matches: {c:>9,} ({c / truth.height:.2%})")

    truth = add_is_val(truth)
    val = truth.filter("is_val").drop("is_val")
    print(f"\n== val split ==\n  val S1: {val.height:,} of {truth.height:,} ({val.height / truth.height:.2%})")
    vc = val.join(s1_country, on="s1_id").group_by("country").len().sort("len", descending=True)
    print("  val by country: " + ", ".join(f"{c} {n:,}" for c, n in vc.iter_rows()))
    print(f"  val singletons: {(val['matched_ids'].list.len() == 0).mean():.4%}")

    print("\n== scorer check on val ==")
    print(f"  ground truth vs itself: {macro_f05(val, val):.6f}")
    print(f"  all-empty predictions:  {macro_f05({}, val):.6f}")

    print("\n== blocking_report with candidates = val truth (oracle must be 1.0) ==")
    cand = (pairs.join(val.select("s1_id"), on="s1_id", how="semi")
            .with_columns(pl.lit("truth").alias("sources"), pl.lit(1.0).alias("block_score"),
                          pl.lit(1).alias("block_rank")))
    blocking_report(cand, val, s1_country=s1_country, name="(val truth as candidates)")
    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
