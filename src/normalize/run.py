"""Bounded-memory normalization of the challenge TSV sources.

Run ``python -m src.normalize.run --split train|test|all`` from the repo root.
Outputs are written atomically to data/norm/{split}_s{1,2,3}.parquet.
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import polars as pl
import pyarrow.parquet as pq

from src.common.io import REPO_ROOT, source_path
from .basic import normalize_df
from .transliteration import load_token_map, train_and_write_map


CONTRACT = (
    "entity_id", "country", "name_raw", "address_raw", "name_norm",
    "name_core", "legal_suffix", "name_latin", "addr_norm", "postcode",
    "city", "house_no", "house_no2", "addr_norm2", "script",
)


def _batch_counts(frame: pl.DataFrame, stats: dict) -> None:
    grouped = frame.group_by("country").agg(
        pl.len().alias("rows"),
        pl.col("postcode").is_not_null().sum().alias("postcode"),
        pl.col("city").is_not_null().sum().alias("city"),
        pl.col("house_no").is_not_null().sum().alias("house_no"),
        (pl.col("script") == "devanagari").sum().alias("devanagari"),
        ((pl.col("script") == "devanagari") & (pl.col("name_latin") != ""))
        .sum().alias("devanagari_latin_filled"),
    )
    for row in grouped.iter_rows(named=True):
        country = row["country"] or "<missing>"
        for field in (
            "rows", "postcode", "city", "house_no", "devanagari",
            "devanagari_latin_filled",
        ):
            stats[country][field] += row[field]


def normalize_source(split: str, source: int, output_dir: Path, batch_size: int) -> dict:
    path = source_path(split, source)
    if not path.is_file():
        raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{split}_s{source}.parquet"
    temp = dest.with_suffix(".parquet.partial")
    batches = pl.scan_csv(
        path, separator="\t", quote_char=None, infer_schema=False,
        empty_string_is_null=True,
    ).collect_batches(chunk_size=batch_size)
    stats = defaultdict(lambda: defaultdict(int))
    token_map = load_token_map(output_dir / "hi_latin_map.json")
    print(f"{split} S{source}: Hindi map {len(token_map)} tokens", flush=True)
    rows = 0
    writer = None
    started = time.perf_counter()
    try:
        for batch in batches:
            normalized = normalize_df(batch, transliteration_map=token_map).select(
                pl.col(column).cast(pl.String) for column in CONTRACT
            )
            table = normalized.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
            writer.write_table(table)
            _batch_counts(normalized, stats)
            rows += len(batch)
            if rows // 1_000_000 != (rows - len(batch)) // 1_000_000:
                print(f"{split} S{source}: {rows:,} rows", flush=True)
        if writer is None:
            raise ValueError(f"Input has no rows: {path}")
        writer.close()
        writer = None
        os.replace(temp, dest)
    except BaseException:
        if writer is not None:
            writer.close()
        temp.unlink(missing_ok=True)
        raise
    elapsed = time.perf_counter() - started
    result = {
        "split": split, "source": source, "rows": rows,
        "seconds": round(elapsed, 2), "path": str(dest),
        "transliteration_map_tokens": len(token_map),
        "by_country": {country: dict(counts) for country, counts in sorted(stats.items())},
    }
    print(f"{split} S{source}: done, {rows:,} rows in {elapsed:.1f}s -> {dest}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test", "all"), required=True)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "norm")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    splits = ("train", "test") if args.split == "all" else (args.split,)
    map_path = args.output_dir / "hi_latin_map.json"
    if not map_path.is_file():
        print("Training Hindi token map from non-validation raw pairs", flush=True)
        stats = train_and_write_map(map_path)
        print(f"Learned {stats['accepted_token_mappings']} Hindi tokens", flush=True)
    results = [
        normalize_source(split, source, args.output_dir, args.batch_size)
        for split in splits for source in (1, 2, 3)
    ]
    report = args.output_dir / "normalization_report.json"
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Run report: {report}", flush=True)


if __name__ == "__main__":
    main()
