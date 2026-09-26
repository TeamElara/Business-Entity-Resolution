"""Compare legacy and proposed house agreement on v3 validation pairs.

Example::

    python -m src.normalize.evaluate_house_no2 \
      --pairs path/to/val_pairs_v3.parquet --norm-dir data/norm --full-counts

The script reads existing train normalization files. It does not overwrite
them or change v3's candidate set or predictions.
"""

import argparse
from pathlib import Path

import polars as pl

from src.common.io import REPO_ROOT
from .abbreviations import normalize_address_v2_expr
from .address import house_number_v2_expr


FIELDS = ["entity_id", "country", "address_raw", "postcode", "house_no", "addr_norm"]


def with_v2(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame.with_columns(
        house_number_v2_expr(
            pl.col("address_raw"), pl.col("postcode"), pl.col("country")
        ).alias("house_no2"),
        normalize_address_v2_expr(
            pl.col("address_raw"), pl.col("country")
        ).alias("addr_norm2"),
    )


def read_needed(path: Path, ids: pl.DataFrame) -> pl.DataFrame:
    frame = pl.scan_parquet(path).select(FIELDS).join(ids.lazy(), on="entity_id", how="semi")
    return with_v2(frame).collect(engine="streaming")


def full_change_counts(norm_dir: Path) -> pl.DataFrame:
    reports = []
    for source in (1, 2, 3):
        frame = with_v2(pl.scan_parquet(norm_dir / f"train_s{source}.parquet").select(FIELDS))
        reports.append(frame.group_by("country").agg(
            pl.len().alias("rows"),
            (pl.col("house_no").fill_null("<null>") !=
             pl.col("house_no2").fill_null("<null>")).sum().alias("house_changed"),
            (pl.col("addr_norm") != pl.col("addr_norm2")).sum().alias("address_changed"),
        ).with_columns(pl.lit(f"S{source}").alias("source")).collect(engine="streaming"))
    return pl.concat(reports).select("country", "source", "rows", "house_changed", "address_changed")


def pair_agreement(pairs: pl.DataFrame, norm_dir: Path) -> pl.DataFrame:
    needed_s1 = pairs.select(pl.col("s1_id").alias("entity_id")).unique()
    needed_cand = pairs.select(pl.col("cand_id").alias("entity_id")).unique()
    q = read_needed(norm_dir / "train_s1.parquet", needed_s1).select(
        pl.col("entity_id").alias("s1_id"),
        pl.col("country"),
        pl.col("house_no").alias("q_old"),
        pl.col("house_no2").alias("q_new"),
    )
    p = pl.concat([
        read_needed(norm_dir / f"train_s{source}.parquet", needed_cand)
        for source in (2, 3)
    ]).select(
        pl.col("entity_id").alias("cand_id"),
        pl.col("country").alias("cand_country"),
        pl.col("house_no").alias("p_old"),
        pl.col("house_no2").alias("p_new"),
    )
    if q.height != needed_s1.height or p.height != needed_cand.height:
        raise ValueError(
            f"Unmatched IDs: S1 {needed_s1.height - q.height}; candidates {needed_cand.height - p.height}"
        )
    joined = pairs.join(q, on="s1_id", how="left").join(p, on="cand_id", how="left")
    if joined.height != pairs.height or joined.select(pl.col("country").is_null().sum()).item():
        raise ValueError("Pair join lost or duplicated rows")
    if joined.filter(pl.col("country") != pl.col("cand_country")).height:
        raise ValueError("Found cross-country validation pairs")
    return joined.with_columns(
        (pl.col("q_old").is_not_null() & pl.col("p_old").is_not_null()
         & (pl.col("q_old") == pl.col("p_old"))).alias("old_agree"),
        (pl.col("q_new").is_not_null() & pl.col("p_new").is_not_null()
         & (pl.col("q_new") == pl.col("p_new"))).alias("new_agree"),
    )


def report(frame: pl.DataFrame) -> None:
    groups = [
        ("positive", pl.col("label") == 1),
        ("negative", pl.col("label") == 0),
        ("high-prob negative", (pl.col("label") == 0) & (pl.col("prob") >= 0.5)),
        ("low-prob positive", (pl.col("label") == 1) & (pl.col("prob") < 0.5)),
    ]
    print("| Group | Pairs | Old agree | New agree | Delta (pp) |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name, predicate in groups:
        subset = frame.filter(predicate)
        n = subset.height
        old = subset["old_agree"].sum() or 0
        new = subset["new_agree"].sum() or 0
        old_rate = 100 * old / n if n else 0
        new_rate = 100 * new / n if n else 0
        print(f"| {name} | {n:,} | {old:,} ({old_rate:.3f}%) | "
              f"{new:,} ({new_rate:.3f}%) | {new_rate - old_rate:+.3f} |")
    print("\nBy country:")
    for country in sorted(frame["country"].unique().to_list()):
        print(f"\n{country}")
        report_country = frame.filter(pl.col("country") == country)
        for name, predicate in groups:
            subset = report_country.filter(predicate)
            n = subset.height
            if not n:
                continue
            old = subset["old_agree"].sum() or 0
            new = subset["new_agree"].sum() or 0
            print(f"  {name}: {n:,} pairs, old {old/n:.4%}, new {new/n:.4%}, "
                  f"delta {(new-old)/n:+.4%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--norm-dir", type=Path, default=REPO_ROOT / "data" / "norm")
    parser.add_argument("--full-counts", action="store_true")
    args = parser.parse_args()
    pairs = pl.read_parquet(args.pairs).select("s1_id", "cand_id", "label", "prob")
    pairs = pairs.with_columns(
        pl.col("label").cast(pl.Int8), pl.col("prob").cast(pl.Float64)
    )
    print(f"Validation pairs: {pairs.height:,}")
    joined = pair_agreement(pairs, args.norm_dir)
    report(joined)
    if args.full_counts:
        print("\nNew-field differences from existing train normalization:")
        print("| Country | Source | Rows | House changed | Address changed |")
        print("| --- | --- | ---: | ---: | ---: |")
        for row in full_change_counts(args.norm_dir).sort("country", "source").iter_rows(named=True):
            print(f"| {row['country']} | {row['source']} | {row['rows']:,} | "
                  f"{row['house_changed']:,} | {row['address_changed']:,} |")


if __name__ == "__main__":
    main()
