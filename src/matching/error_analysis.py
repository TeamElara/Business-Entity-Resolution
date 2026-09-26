"""Error analysis of a scored validation candidate set (v1 and later).

Splits the macro F0.5 loss into: false positives, true matches rejected by the final model,
true matches cut by the pruner, and true matches never retrieved by stage 1; then puts each
error into a category using the pair features, and prints examples with the raw records.

    python -m src.matching.error_analysis --tag pm                 # v1: data/feat/v1_val_scored_pm.parquet
    python -m src.matching.error_analysis --version v2 --tag k6    # v2: data/feat/v2/val_scored_k6.parquet
"""
import argparse
import json

import polars as pl

from src.common import add_is_val, load_ground_truth, macro_f05_fast, truth_pairs
from src.common.io import REPO_ROOT

FEAT_DIR = REPO_ROOT / "data" / "feat"
MODEL_DIR = REPO_ROOT / "data" / "models"
NORM_DIR = REPO_ROOT / "data" / "norm"

CTX = ["name_ts", "name_ratio", "addr_ts", "house_tol", "house_full_eq", "num_jacc", "p_nonlatin",
       "p_other_script", "p_addr_missing", "q_addr_missing", "q_only_n", "p_only_n", "q_only_idf",
       "p_only_idf", "diff_ratio", "city_sim", "p_is_s3", "in_name", "in_addr"]


def decide(c, t, t1):
    best = pl.col("prob") == pl.col("prob").max().over("s1_id")
    return c.filter((pl.col("prob") >= t) | (best & (pl.col("prob") >= t1)))


def fn_category():
    """First matching reason, most specific first."""
    return (
        pl.when(pl.col("p_nonlatin") == 1).then(pl.lit("cross-script name"))
        .when((pl.col("p_addr_missing") == 1) | (pl.col("q_addr_missing") == 1)).then(pl.lit("address missing"))
        .when(pl.col("house_tol") == -1).then(pl.lit("house number differs (2+ edits)"))
        .when(pl.col("house_tol") == 0.5).then(pl.lit("house number 1 edit"))
        .when(pl.col("name_ts") < 0.6).then(pl.lit("name very different (<0.6)"))
        .when(pl.col("addr_ts") < 0.5).then(pl.lit("address very different (<0.5)"))
        .when(pl.col("name_ts") < 0.85).then(pl.lit("name partly different"))
        .otherwise(pl.lit("name+address similar"))
    )


def fp_category():
    return (
        pl.when(pl.col("p_nonlatin") == 1).then(pl.lit("cross-script name"))
        .when((pl.col("name_ts") >= 0.95) & (pl.col("addr_ts") >= 0.95) & (pl.col("house_tol") >= 0))
        .then(pl.lit("near-identical record (label noise?)"))
        .when((pl.col("addr_ts") >= 0.85) & (pl.col("name_ts") < 0.85)).then(pl.lit("same address, different name"))
        .when((pl.col("name_ts") >= 0.9) & (pl.col("house_tol") == -1)).then(pl.lit("same name, other house number"))
        .when((pl.col("name_ts") >= 0.9) & (pl.col("house_tol") == 0.5)).then(pl.lit("same name, house 1 edit"))
        .when((pl.col("name_ts") >= 0.9) & (pl.col("addr_ts") < 0.7)).then(pl.lit("same name, different address"))
        .when((pl.col("p_addr_missing") == 1) | (pl.col("q_addr_missing") == 1)).then(pl.lit("address missing"))
        .otherwise(pl.lit("other"))
    )


def table(df, col, total_pairs, title):
    t = df.group_by(col).agg(pl.len().alias("pairs"), pl.col("prob").mean().alias("mean_prob"),
                             pl.col("s1_id").n_unique().alias("s1")).sort("pairs", descending=True)
    print(f"\n{title} ({total_pairs:,} pairs)")
    for r in t.iter_rows(named=True):
        print(f"  {r[col]:<40} {r['pairs']:>8,}  {r['pairs'] / total_pairs:6.1%}  S1 {r['s1']:>7,}"
              f"  mean prob {r['mean_prob'] if r['mean_prob'] is not None else float('nan'):.2f}")
    return t


def text_lookup(ids: pl.Series) -> pl.DataFrame:
    parts = []
    for s in (1, 2, 3):
        parts.append(pl.scan_parquet(NORM_DIR / f"train_s{s}.parquet")
                     .select("entity_id", "name_raw", "address_raw")
                     .join(pl.LazyFrame({"entity_id": ids.unique()}), on="entity_id", how="semi").collect())
    return pl.concat(parts)


def examples(df, cat_col, cats, n, txt, header):
    print(f"\n{header}")
    for cat in cats:
        ex = df.filter(pl.col(cat_col) == cat).sample(min(n, df.filter(pl.col(cat_col) == cat).height), seed=0)
        if not ex.height:
            continue
        print(f"  [{cat}]")
        ex = (ex.join(txt.rename({"entity_id": "s1_id", "name_raw": "qn", "address_raw": "qa"}), on="s1_id", how="left")
              .join(txt.rename({"entity_id": "cand_id", "name_raw": "pn", "address_raw": "pa"}), on="cand_id", how="left"))
        for r in ex.iter_rows(named=True):
            p = "-" if r.get("prob") is None else f"{r['prob']:.2f}"
            print(f"    p={p} {r['country']:<6} {str(r['qn'])[:38]:<38} | {str(r['qa'])[:40]}")
            print(f"    {'':<13} {str(r['pn'])[:38]:<38} | {str(r['pa'])[:40]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2"], default="v1")
    ap.add_argument("--tag", default="pm")
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    prm = json.load(open(MODEL_DIR / f"{args.version}_params{sfx}.json"))
    t, t1 = prm["t"], prm["t1"]
    s1c = pl.read_parquet(NORM_DIR / "train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    truth = add_is_val(load_ground_truth()).filter("is_val").drop("is_val").join(s1c, on="s1_id")
    tp = truth_pairs(truth.drop("country")).join(s1c, on="s1_id")
    if args.version == "v1":
        cand = pl.read_parquet(FEAT_DIR / f"v1_val_scored{sfx}.parquet")
        stage1 = pl.read_parquet(FEAT_DIR / "v1_val.parquet", columns=["s1_id", "cand_id", *CTX])
    else:
        cand = pl.read_parquet(FEAT_DIR / "v2" / f"val_scored{sfx}.parquet")
        stage1 = pl.concat([pl.read_parquet(f, columns=["s1_id", "cand_id", *CTX])
                            for f in sorted((FEAT_DIR / "v2" / "val").glob("*.parquet"))])
    cand = cand.join(stage1, on=["s1_id", "cand_id"], how="left").join(s1c, on="s1_id")
    pred = decide(cand, t, t1)
    f = macro_f05_fast(pred, truth)
    print(f"thresholds t={t} t1={t1}: val macro F0.5 {f:.4f}, {cand.height / truth.height:.2f} cands/S1")

    # ---------------------------------------------------------------- loss decomposition
    fp = pred.filter(pl.col("label") == 0)
    fn_model = cand.filter(pl.col("label") == 1).join(pred, on=["s1_id", "cand_id"], how="anti")
    missing = tp.join(cand, on=["s1_id", "cand_id"], how="anti")
    in_stage1 = missing.join(stage1.select("s1_id", "cand_id"), on=["s1_id", "cand_id"], how="semi")
    fn_pruned = in_stage1.join(stage1, on=["s1_id", "cand_id"], how="left")
    fn_stage1 = missing.join(stage1.select("s1_id", "cand_id"), on=["s1_id", "cand_id"], how="anti")
    good = pred.filter(pl.col("label") == 1).select("s1_id", "cand_id")
    fixes = {
        "false positives removed": good,
        "model-rejected true matches accepted": pl.concat([pred.select("s1_id", "cand_id"),
                                                           fn_model.select("s1_id", "cand_id")]),
        "+ pruner-cut true matches (on top of the line above)": pl.concat([
            pred.select("s1_id", "cand_id"), fn_model.select("s1_id", "cand_id"),
            fn_pruned.select("s1_id", "cand_id")]),
        "+ stage-1-missed true matches (all fixed = oracle 1.0)": pl.concat([
            good, fn_model.select("s1_id", "cand_id"), missing.select("s1_id", "cand_id")]),
    }
    print(f"\npredicted pairs {pred.height:,}, correct {pred.height - fp.height:,}; true pairs {tp.height:,}")
    print(f"  false positives                 {fp.height:>8,}")
    print(f"  true matches rejected by model  {fn_model.height:>8,}")
    print(f"  true matches cut by the pruner  {fn_pruned.height:>8,}")
    print(f"  true matches missed by stage 1  {fn_stage1.height:>8,}")
    print("\nmacro F0.5 if fixed:")
    for k, p in fixes.items():
        print(f"  {k:<55} {macro_f05_fast(p, truth):.4f}")

    print("\nby country:")
    for ctry in sorted(truth["country"].unique().to_list()):
        tc = truth.filter(pl.col("country") == ctry)
        pc = pred.filter(pl.col("country") == ctry)
        print(f"  {ctry:<6} F0.5 {macro_f05_fast(pc, tc):.4f} | FP {fp.filter(pl.col('country') == ctry).height:>7,}"
              f" | FN model {fn_model.filter(pl.col('country') == ctry).height:>7,}"
              f" | FN pruner {fn_pruned.filter(pl.col('country') == ctry).height:>6,}"
              f" | FN stage 1 {fn_stage1.filter(pl.col('country') == ctry).height:>6,}"
              f" | if FP fixed {macro_f05_fast(fixes['false positives removed'], tc):.4f}"
              f" | if FN model fixed {macro_f05_fast(fixes['model-rejected true matches accepted'], tc):.4f}")

    # ---------------------------------------------------------------- per-S1 view
    n_pred = pred.group_by("s1_id").agg(pl.len().alias("n_pred"), (pl.col("label") == 1).sum().alias("tp"))
    per = (truth.select("s1_id", "country", pl.col("matched_ids").list.len().alias("n_true"))
           .join(n_pred, on="s1_id", how="left").with_columns(pl.col("n_pred", "tp").fill_null(0)))
    prec, rec = pl.col("tp") / pl.col("n_pred"), pl.col("tp") / pl.col("n_true")
    per = per.with_columns(
        pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
        .when(pl.col("tp") == 0).then(0.0).otherwise(1.25 * prec * rec / (0.25 * prec + rec)).alias("f"))
    n = per.height
    print("\nloss by number of true matches (share of the total macro F0.5 loss):")
    loss = (1 - per["f"]).sum()
    for r in (per.with_columns(pl.col("n_true").clip(0, 7).alias("k")).group_by("k")
              .agg(pl.len().alias("s1"), (1 - pl.col("f")).sum().alias("loss"), pl.col("f").mean().alias("f"),
                   (pl.col("n_pred").cast(pl.Int64) - pl.col("n_true").cast(pl.Int64)).mean().alias("pred_minus_true"))
              .sort("k").iter_rows(named=True)):
        print(f"  {r['k']}{'+' if r['k'] == 7 else ' '}  S1 {r['s1']:>7,}  F0.5 {r['f']:.4f}  "
              f"loss share {r['loss'] / loss:6.1%}  (pred - true) per S1 {r['pred_minus_true']:+.2f}")
    kinds = per.with_columns(
        pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") > 0)).then(pl.lit("singleton given a match"))
        .when((pl.col("n_true") > 0) & (pl.col("n_pred") == 0)).then(pl.lit("has matches, predicted none"))
        .when((pl.col("n_true") > 0) & (pl.col("tp") == 0)).then(pl.lit("only wrong matches"))
        .when(pl.col("f") < 1).then(pl.lit("partly right"))
        .otherwise(pl.lit("perfect")).alias("kind"))
    print("\nS1 outcome:")
    for r in (kinds.group_by("kind").agg(pl.len().alias("s1"), (1 - pl.col("f")).sum().alias("loss"))
              .sort("loss", descending=True).iter_rows(named=True)):
        print(f"  {r['kind']:<30} S1 {r['s1']:>7,} ({r['s1'] / n:5.1%})  loss share {r['loss'] / loss:6.1%}")

    # ---------------------------------------------------------------- categories
    fn_model = fn_model.with_columns(fn_category().alias("cat"))
    fp = fp.with_columns(fp_category().alias("cat"))
    table(fn_model, "cat", fn_model.height, "true matches rejected by the model, by reason")
    print("  prob of rejected true matches: " + ", ".join(
        f"{a}-{b}: {fn_model.filter((pl.col('prob') >= a) & (pl.col('prob') < b)).height:,}"
        for a, b in ((0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, t), (t, 1.01))))
    table(fp, "cat", fp.height, "false positives, by reason")
    fp_ctx = fp.join(per.select("s1_id", "n_true", "tp"), on="s1_id").with_columns(
        pl.when(pl.col("n_true") == 0).then(pl.lit("S1 is a singleton"))
        .when(pl.col("tp") == 0).then(pl.lit("S1 got no true match"))
        .otherwise(pl.lit("extra next to true matches")).alias("ctx"))
    table(fp_ctx, "ctx", fp.height, "false positives, by what else the S1 got")
    fn_pruned = fn_pruned.with_columns(fn_category().alias("cat"), pl.lit(None, pl.Float32).alias("prob"))
    table(fn_pruned, "cat", fn_pruned.height, "true matches cut by the pruner, by reason")
    s1c_txt = fn_stage1.with_columns(pl.col("cand_id").str.slice(0, 2).alias("src"), pl.lit(None, pl.Float32).alias("prob"))
    table(s1c_txt, "src", s1c_txt.height, "true matches missed by stage 1, by source")

    # ---------------------------------------------------------------- threshold by country
    print("\nbest thresholds per country (diagnostic only, France has no labels):")
    for ctry in sorted(truth["country"].unique().to_list()):
        tc, cc = truth.filter(pl.col("country") == ctry), cand.filter(pl.col("country") == ctry)
        best = max((macro_f05_fast(decide(cc, a / 100, b / 100), tc), a / 100, b / 100)
                   for a in range(50, 91, 5) for b in range(30, a + 1, 5))
        print(f"  {ctry:<6} best {best[0]:.4f} at t={best[1]} t1={best[2]} vs {macro_f05_fast(decide(cc, t, t1), tc):.4f}"
              f" at the global t={t} t1={t1}")

    # ---------------------------------------------------------------- examples
    ids = pl.concat([fn_model["s1_id"], fn_model["cand_id"], fp["s1_id"], fp["cand_id"],
                     fn_stage1["s1_id"], fn_stage1["cand_id"], fn_pruned["s1_id"], fn_pruned["cand_id"]])
    txt = text_lookup(ids)
    examples(fn_model, "cat", fn_model["cat"].value_counts(sort=True)["cat"].to_list(), args.examples, txt,
             "EXAMPLES: true matches rejected by the model (S1 line, then candidate line)")
    examples(fp, "cat", fp["cat"].value_counts(sort=True)["cat"].to_list(), args.examples, txt,
             "EXAMPLES: false positives")
    examples(fn_stage1.with_columns(pl.lit("missed").alias("cat"), pl.lit(None, pl.Float32).alias("prob")),
             "cat", ["missed"], 2 * args.examples, txt, "EXAMPLES: true matches missed by stage 1")


if __name__ == "__main__":
    main()
