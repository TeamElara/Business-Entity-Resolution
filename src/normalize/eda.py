"""Reproducible EDA for normalization design.

This module only reads the provided challenge data. It creates local, gitignored
inspection files under ``data/eda`` and a compact machine-readable summary used
to maintain ``docs/eda_notes.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import polars as pl
from rapidfuzz.fuzz import ratio


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
COUNTRIES = {"train": ("India", "US"), "test": ("France", "India", "US")}


def scan_tsv(path: Path) -> pl.LazyFrame:
    """Read challenge TSV data with the shared, all-string contract."""
    return pl.scan_csv(
        path,
        separator="\t",
        quote_char=None,
        infer_schema_length=0,
        empty_string_is_null=True,
    ).select(
        [pl.col(column).cast(pl.String).fill_null("") for column in SOURCE_COLUMNS]
    )


def deterministic_sample(
    frame: pl.LazyFrame, country: str, size: int, seed: int
) -> pl.DataFrame:
    """Select a stable pseudo-random sample without loading the full file."""
    return (
        frame.filter(pl.col("country") == country)
        .with_columns(pl.col("entity_id").hash(seed=seed).alias("_sample_key"))
        .sort("_sample_key")
        .limit(size)
        .drop("_sample_key")
        .collect(engine="streaming")
    )


def summarize_source(frame: pl.LazyFrame) -> pl.DataFrame:
    """Compute exact data-quality counts by country in a streaming query."""
    name = pl.col("business_name")
    address = pl.col("business_address")
    return (
        frame.group_by("country")
        .agg(
            pl.len().alias("rows"),
            (name.str.strip_chars() == "").sum().alias("blank_name"),
            (address.str.strip_chars() == "").sum().alias("blank_address"),
            name.str.contains(r"[\u0900-\u097f]").sum().alias("devanagari_name"),
            address.str.contains(r"[\u0900-\u097f]").sum().alias("devanagari_address"),
            name.str.contains(r"(?i)^\s*(?:--+|<<+|>>+|[|_/\\]+)").sum().alias(
                "junk_prefix_name"
            ),
            name.str.contains(r"&").sum().alias("ampersand_name"),
            name.str.contains(r"(?i)\b(?:www\.|[a-z0-9-]+\.(?:com|in|org|net))\b").sum().alias(
                "domain_like_name"
            ),
        )
        .sort("country")
        .collect(engine="streaming")
    )


def _simple_clean(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold().replace("&", " and ")
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _token_jaccard(left: str, right: str) -> float:
    a, b = set(_simple_clean(left).split()), set(_simple_clean(right).split())
    return len(a & b) / len(a | b) if a or b else 1.0


def sample_true_pairs(data_dir: Path, size: int, seed: int) -> pl.DataFrame:
    """Return stable, fully hydrated S1-to-S2/S3 true matched pairs."""
    train_dir = data_dir / "train"
    pairs = (
        pl.scan_csv(
            train_dir / "train_ground_truth.tsv",
            separator="\t",
            quote_char=None,
            infer_schema_length=0,
            empty_string_is_null=True,
        )
        .with_columns(pl.col("matched_entity_ids").cast(pl.String).fill_null(""))
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids", empty_as_null=True)
        .filter(pl.col("matched_entity_ids") != "")
        .rename({"matched_entity_ids": "candidate_id"})
        .with_columns(
            pl.concat_str(["source1_entity_id", "candidate_id"], separator="|")
            .hash(seed=seed)
            .alias("_sample_key")
        )
        .sort("_sample_key")
        .limit(size)
        .drop("_sample_key")
        .collect(engine="streaming")
    )

    s1_ids = pairs["source1_entity_id"].to_list()
    candidate_ids = pairs["candidate_id"].to_list()
    s1 = (
        scan_tsv(train_dir / "train_source1.tsv")
        .filter(pl.col("entity_id").is_in(s1_ids))
        .rename(
            {
                "entity_id": "source1_entity_id",
                "business_name": "s1_name",
                "business_address": "s1_address",
                "country": "s1_country",
            }
        )
        .collect(engine="streaming")
    )

    candidates = []
    for source in (2, 3):
        candidate = (
            scan_tsv(train_dir / f"train_source{source}.tsv")
            .filter(pl.col("entity_id").is_in(candidate_ids))
            .rename(
                {
                    "entity_id": "candidate_id",
                    "business_name": "candidate_name",
                    "business_address": "candidate_address",
                    "country": "candidate_country",
                }
            )
            .collect(engine="streaming")
        )
        candidates.append(candidate)

    hydrated = pairs.join(s1, on="source1_entity_id", how="left").join(
        pl.concat(candidates), on="candidate_id", how="left"
    )
    return hydrated.with_columns(
        pl.struct(["s1_name", "candidate_name"])
        .map_elements(
            lambda row: _simple_clean(row["s1_name"]) == _simple_clean(row["candidate_name"]),
            return_dtype=pl.Boolean,
        )
        .alias("simple_name_equal"),
        pl.struct(["s1_address", "candidate_address"])
        .map_elements(
            lambda row: _simple_clean(row["s1_address"])
            == _simple_clean(row["candidate_address"]),
            return_dtype=pl.Boolean,
        )
        .alias("simple_address_equal"),
        pl.struct(["s1_name", "candidate_name"])
        .map_elements(
            lambda row: _token_jaccard(row["s1_name"], row["candidate_name"]),
            return_dtype=pl.Float64,
        )
        .alias("name_token_jaccard"),
        pl.struct(["s1_name", "candidate_name"])
        .map_elements(
            lambda row: ratio(_simple_clean(row["s1_name"]), _simple_clean(row["candidate_name"]))
            / 100.0,
            return_dtype=pl.Float64,
        )
        .alias("name_fuzzy_ratio"),
    )


def run(data_dir: Path, output_dir: Path, sample_size: int, pair_size: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_rows = []
    source_summaries = []

    for split, countries in COUNTRIES.items():
        for source in (1, 2, 3):
            path = data_dir / split / f"{split}_source{source}.tsv"
            frame = scan_tsv(path)
            summary = summarize_source(frame).with_columns(
                pl.lit(split).alias("split"), pl.lit(source).alias("source")
            )
            source_summaries.append(summary)
            for country in countries:
                sample = deterministic_sample(frame, country, sample_size, seed).with_columns(
                    pl.lit(split).alias("split"), pl.lit(source).alias("source")
                )
                sampled_rows.append(sample)

    samples = pl.concat(sampled_rows, how="diagonal_relaxed").select(
        ["split", "source", *SOURCE_COLUMNS]
    )
    samples.write_csv(output_dir / "random_rows.tsv", separator="\t")

    summaries = pl.concat(source_summaries, how="diagonal_relaxed").select(
        [
            "split",
            "source",
            "country",
            "rows",
            "blank_name",
            "blank_address",
            "devanagari_name",
            "devanagari_address",
            "junk_prefix_name",
            "ampersand_name",
            "domain_like_name",
        ]
    )
    summaries.write_csv(output_dir / "source_summary.tsv", separator="\t")

    pairs = sample_true_pairs(data_dir, pair_size, seed)
    pairs.write_csv(output_dir / "true_match_pairs.tsv", separator="\t")
    pair_summary = {
        "sampled_pairs": pairs.height,
        "country_agreement_rate": float(
            (pairs["s1_country"] == pairs["candidate_country"]).mean()
        ),
        "simple_name_equality_rate": float(pairs["simple_name_equal"].mean()),
        "simple_address_equality_rate": float(pairs["simple_address_equal"].mean()),
        "mean_name_token_jaccard": float(pairs["name_token_jaccard"].mean()),
        "mean_name_fuzzy_ratio": float(pairs["name_fuzzy_ratio"].mean()),
    }
    (output_dir / "pair_summary.json").write_text(
        json.dumps(pair_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(pair_summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/eda"))
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--pair-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.output_dir, args.sample_size, args.pair_size, args.seed)
