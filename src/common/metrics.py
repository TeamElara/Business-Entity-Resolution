"""Scoring and blocking diagnostics (master plan §6).

Conventions:
- truth: polars DataFrame (s1_id, matched_ids: list[str]) as returned by
  io.load_ground_truth(), usually filtered to the val split. A dict {s1_id: ids}
  is also accepted.
- cand_df: the §8 candidate contract (s1_id, cand_id, sources, block_score,
  block_rank). block_score is "higher = more similar"; block_rank 1 = best.
- All averages run over ALL truth S1 ids. An S1 with no prediction / no candidate
  counts as an empty set; candidates for S1 ids outside truth are ignored.
"""
import polars as pl

from .io import load_source, scan_source, truth_pairs


def f05_entity(pred: set, truth: set) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


# ---------------------------------------------------------------- conversions

def _as_truth_df(truth) -> pl.DataFrame:
    if isinstance(truth, dict):
        return pl.DataFrame(
            {"s1_id": list(truth), "matched_ids": [sorted(v) for v in truth.values()]},
            schema={"s1_id": pl.String, "matched_ids": pl.List(pl.String)},
        )
    return truth.select("s1_id", "matched_ids")


def to_set_dict(obj, key: str = "s1_id") -> dict:
    """{s1_id: set(ids)} from a dict, a (s1_id, list) frame or a (s1_id, cand_id) pairs frame."""
    if isinstance(obj, dict):
        return {k: set(v) for k, v in obj.items()}
    list_cols = [c for c, t in obj.schema.items() if c != key and isinstance(t, pl.List)]
    if list_cols:
        col = list_cols[0]
    else:
        col = "cand_id"
        obj = (
            obj.drop_nulls(col)
            .group_by(key)
            .agg(pl.col(col))
        )
    return {k: set(v or []) for k, v in zip(obj[key].to_list(), obj[col].to_list())}


# ---------------------------------------------------------------- final metric

def macro_f05(pred, truth, return_per_entity: bool = False):
    """Mean f05_entity over ALL truth S1 ids (missing prediction = empty set)."""
    P, T = to_set_dict(pred), to_set_dict(truth)
    scores = {s1: f05_entity(P.get(s1, set()), t) for s1, t in T.items()}
    score = sum(scores.values()) / len(scores) if scores else float("nan")
    return (score, scores) if return_per_entity else score


# ---------------------------------------------------------------- blocking

def _oracle_expr(tp: str = "tp", n_true: str = "n_true") -> pl.Expr:
    """f05_entity(truth & cands, truth) vectorised: precision is 1, recall = tp / n_true."""
    r = pl.col(tp) / pl.col(n_true)
    return (
        pl.when(pl.col(n_true) == 0).then(1.0)
        .when(pl.col(tp) == 0).then(0.0)
        .otherwise(1.25 * r / (0.25 + r))
    )


def _per_s1(cands: pl.DataFrame, t: pl.DataFrame, tp_pairs: pl.DataFrame) -> pl.DataFrame:
    """One row per truth S1: n_true, n_cand, tp, oracle."""
    hits = cands.join(tp_pairs, on=["s1_id", "cand_id"], how="semi")
    return (
        t.select("s1_id", pl.col("matched_ids").list.len().alias("n_true"))
        .join(cands.group_by("s1_id").len("n_cand"), on="s1_id", how="left")
        .join(hits.group_by("s1_id").len("tp"), on="s1_id", how="left")
        .with_columns(pl.col("n_cand", "tp").fill_null(0).cast(pl.Int64))
        .with_columns(_oracle_expr().alias("oracle"))
    )


def _restrict(cand_df: pl.DataFrame, t: pl.DataFrame) -> pl.DataFrame:
    return cand_df.join(t.select("s1_id"), on="s1_id", how="semi")


def _pool_size(split: str, countries) -> int:
    """Number of S2+S3 records of `split`, restricted to `countries` when given."""
    n = 0
    for src in (2, 3):
        lf = scan_source(split, src, columns=["country"])
        if countries is not None:
            lf = lf.filter(pl.col("country").is_in(list(countries)))
        n += lf.select(pl.len()).collect().item()
    return n


def blocking_report(cand_df: pl.DataFrame, truth, s1_country="train",
                    pool_size=None, split: str = "train", name: str = "",
                    verbose: bool = True) -> dict:
    """Print and return blocking quality for cand_df against truth.

    s1_country: DataFrame (s1_id, country), or "train"/"test" to load S1
        countries from that split, or None to skip the per-country breakdown.
    pool_size: number of S2+S3 records used for the reduction ratio. Default:
        S2+S3 rows of `split` in the countries of the evaluated S1 (so an
        India-only run is compared with the India pool), or all rows when
        s1_country is None.
    """
    t = _as_truth_df(truth)
    tp_pairs = truth_pairs(t)
    n_in = cand_df.height
    c = _restrict(cand_df, t).select("s1_id", "cand_id")
    n_dup = c.height - c.unique().height
    c = c.unique()
    per = _per_s1(c, t, tp_pairs)

    if isinstance(s1_country, str):
        s1_country = load_source(s1_country, 1, columns=["entity_id", "country"]) \
            .rename({"entity_id": "s1_id"})
    if s1_country is not None:
        per = per.join(s1_country.select("s1_id", "country"), on="s1_id", how="left") \
            .with_columns(pl.col("country").fill_null("<missing>"))

    n_s1 = per.height
    total_pairs = int(per["n_cand"].sum())
    n_true = int(per["n_true"].sum())
    tp = int(per["tp"].sum())
    pool_countries = None
    if pool_size is None:
        if s1_country is not None:
            pool_countries = sorted(per["country"].unique().to_list())
        pool_size = _pool_size(split, pool_countries)
    nc = per["n_cand"]
    out = {
        "name": name,
        "n_s1": n_s1,
        "total_pairs": total_pairs,
        "mean_cands": nc.mean(),
        "median_cands": nc.median(),
        "p95_cands": nc.quantile(0.95, "nearest"),
        "max_cands": nc.max(),
        "s1_zero_cands": int((nc == 0).sum()),
        "pair_recall": tp / n_true if n_true else float("nan"),
        "oracle_ceiling": per["oracle"].mean(),
        "reduction_ratio": 1 - total_pairs / (n_s1 * pool_size) if pool_size else float("nan"),
        "dup_pairs": n_dup,
        "ignored_pairs": n_in - n_dup - c.height,
        "pool_size": pool_size,
        "pool_countries": pool_countries,
    }

    # recall by source prefix of the true id
    src = (
        tp_pairs.with_columns(pl.col("cand_id").str.slice(0, 2).alias("src"))
        .join(c.with_columns(pl.lit(True).alias("hit")), on=["s1_id", "cand_id"], how="left")
        .group_by("src").agg(pl.len().alias("n_true"), pl.col("hit").fill_null(False).sum().alias("tp"))
        .join(
            c.with_columns(pl.col("cand_id").str.slice(0, 2).alias("src"))
            .group_by("src").len("n_cand"),
            on="src", how="full", coalesce=True,
        )
        .with_columns(pl.col("n_true", "tp", "n_cand").fill_null(0))
        .with_columns(
            (pl.col("tp") / pl.col("n_true")).alias("pair_recall"),
            (pl.col("n_cand") / n_s1).alias("cands_per_s1"),
        )
        .sort("src")
    )
    out["by_source"] = src

    by_country = None
    if s1_country is not None:
        by_country = (
            per.group_by("country")
            .agg(
                pl.len().alias("n_s1"),
                pl.col("n_cand").mean().alias("mean_cands"),
                (pl.col("tp").sum() / pl.col("n_true").sum()).alias("pair_recall"),
                pl.col("oracle").mean().alias("oracle"),
            )
            .sort("n_s1", descending=True)
        )
    out["by_country"] = by_country

    # by the S1's number of true matches (7+ grouped)
    out["by_n_true"] = (
        per.with_columns(pl.col("n_true").clip(upper_bound=7).alias("n_true_grp"))
        .group_by("n_true_grp")
        .agg(
            pl.len().alias("n_s1"),
            pl.col("n_cand").mean().alias("mean_cands"),
            (pl.col("tp").sum() / pl.col("n_true").sum()).alias("pair_recall"),
            pl.col("oracle").mean().alias("oracle"),
        )
        .sort("n_true_grp")
    )
    out["per_s1"] = per

    if verbose:
        _print_report(out)
    return out


def _print_report(o: dict) -> None:
    print(f"=== blocking report {o['name']} ===")
    print(f"  S1 evaluated        {o['n_s1']:,}")
    print(f"  total pairs         {o['total_pairs']:,}")
    print(f"  cands/S1 mean {o['mean_cands']:.2f} | median {o['median_cands']:.1f} "
          f"| p95 {o['p95_cands']} | max {o['max_cands']} | S1 with 0 cands {o['s1_zero_cands']:,}")
    print(f"  pair recall         {o['pair_recall']:.4f}")
    print(f"  oracle ceiling      {o['oracle_ceiling']:.4f}")
    pool_note = "all countries" if o["pool_countries"] is None else "/".join(o["pool_countries"])
    print(f"  reduction ratio     {o['reduction_ratio']:.8f}  (pool {o['pool_size']:,} S2+S3, {pool_note})")
    if o["dup_pairs"] or o["ignored_pairs"]:
        print(f"  (dropped {o['dup_pairs']:,} duplicate pairs, "
              f"ignored {o['ignored_pairs']:,} pairs for S1 outside truth)")
    print("  by source:")
    for r in o["by_source"].iter_rows(named=True):
        print(f"    {r['src']}  pair recall {r['pair_recall']:.4f}  "
              f"({r['tp']:,}/{r['n_true']:,})  cands/S1 {r['cands_per_s1']:.2f}")
    if o["by_country"] is not None:
        print("  by country:")
        for r in o["by_country"].iter_rows(named=True):
            print(f"    {r['country']:<10} n_s1 {r['n_s1']:>9,}  cands/S1 {r['mean_cands']:6.2f}  "
                  f"pair recall {r['pair_recall']:.4f}  oracle {r['oracle']:.4f}")
    print("  by number of true matches:")
    for r in o["by_n_true"].iter_rows(named=True):
        label = f"{r['n_true_grp']}+" if r["n_true_grp"] == 7 else str(r["n_true_grp"])
        print(f"    {label:<3} n_s1 {r['n_s1']:>9,}  cands/S1 {r['mean_cands']:6.2f}  "
              f"pair recall {r['pair_recall']:.4f}  oracle {r['oracle']:.4f}")


DEFAULT_CUTOFFS = (
    [("all", None)]
    + [("topk", k) for k in (50, 30, 20, 15, 10, 8, 5, 3, 1)]
    + [("gap", g) for g in (0.3, 0.2, 0.1, 0.05)]
)


def _apply_cutoff(c: pl.DataFrame, kind: str, val) -> pl.DataFrame:
    """kind: all | topk (block_rank <= k) | gap (block_score >= best - g)
    | min_score (block_score >= s) | topk_gap ((k, g): both)."""
    if kind == "all":
        return c
    if kind == "topk":
        return c.filter(pl.col("block_rank") <= val)
    if kind == "gap":
        return c.filter(pl.col("block_score") >= pl.col("best_score") - val)
    if kind == "min_score":
        return c.filter(pl.col("block_score") >= val)
    if kind == "topk_gap":
        k, g = val
        return c.filter((pl.col("block_rank") <= k) & (pl.col("block_score") >= pl.col("best_score") - g))
    raise ValueError(f"unknown cutoff kind {kind!r}")


def tradeoff_table(cand_df: pl.DataFrame, truth, cutoffs=None, verbose: bool = True) -> pl.DataFrame:
    """For each cutoff: avg cands/S1 | oracle ceiling | pair recall (over all truth S1)."""
    t = _as_truth_df(truth)
    tp_pairs = truth_pairs(t)
    c = (
        _restrict(cand_df, t)
        .select("s1_id", "cand_id", "block_score", "block_rank")
        .unique(["s1_id", "cand_id"])
        .with_columns(pl.col("block_score").max().over("s1_id").alias("best_score"))
    )
    n_true = t["matched_ids"].list.len().sum()
    rows = []
    for kind, val in cutoffs or DEFAULT_CUTOFFS:
        per = _per_s1(_apply_cutoff(c, kind, val), t, tp_pairs)
        rows.append({
            "cutoff": kind if val is None else f"{kind}={val}",
            "avg_cands": per["n_cand"].mean(),
            "p95_cands": per["n_cand"].quantile(0.95, "nearest"),
            "oracle": per["oracle"].mean(),
            "pair_recall": per["tp"].sum() / n_true if n_true else float("nan"),
            "total_pairs": int(per["n_cand"].sum()),
        })
    table = pl.DataFrame(rows)
    if verbose:
        print(f"{'cutoff':<16} {'avg cands/S1':>12} {'p95':>5} {'oracle ceiling':>15} "
              f"{'pair recall':>12} {'total pairs':>12}")
        for r in rows:
            print(f"{r['cutoff']:<16} {r['avg_cands']:>12.2f} {r['p95_cands']:>5} "
                  f"{r['oracle']:>15.4f} {r['pair_recall']:>12.4f} {r['total_pairs']:>12,}")
    return table
