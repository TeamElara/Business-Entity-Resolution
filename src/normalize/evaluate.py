"""Held-out Phase 9 evaluation of normalization and name ranking.

Run ``python -m src.normalize.evaluate`` after generating train Parquet files.
The fixed retrieval pool uses raw name+address word TF-IDF (top 50); raw and
normalized names are re-ranked over the *same* candidates with RapidFuzz
WRatio. Validation labels are used only to score, never to fit TF-IDF.
"""

import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import polars as pl
import psutil
from rapidfuzz import fuzz
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from src.common.io import REPO_ROOT, load_ground_truth, truth_pairs
from src.common.split import add_is_val
from .basic_text import clean_text


NORM_DIR = REPO_ROOT / "data" / "norm"
DEFAULT_REPORT = REPO_ROOT / "data" / "eda" / "phase09_metrics.json"


def token_jaccard(left: str | None, right: str | None) -> float:
    """Set-token Jaccard; two empty names have similarity zero, not one."""
    a, b = set((left or "").split()), set((right or "").split())
    return len(a & b) / len(a | b) if a or b else 0.0


def _empty_pair_counts() -> dict:
    return defaultdict(float)


def score_true_pairs(pairs: pl.DataFrame) -> dict:
    """Aggregate equality and Jaccard on known true pairs by country/script."""
    counts = defaultdict(_empty_pair_counts)
    cols = [
        "country", "s1_raw", "cand_raw", "s1_core", "cand_core",
        "s1_postcode", "cand_postcode", "s1_script", "cand_script",
        "s1_raw_tokens", "cand_raw_tokens",
    ]
    for row in pairs.select(cols).iter_rows():
        (country, raw1, raw2, core1, core2, pc1, pc2,
         script1, script2, tokens1, tokens2) = row
        cross_script = script1 != script2
        groups = ("overall", f"country:{country}",
                  "cross_script" if cross_script else "same_script")
        raw_equal = bool(raw1 and raw2 and raw1 == raw2)
        raw_casefold_equal = bool(raw1 and raw2 and raw1.casefold() == raw2.casefold())
        core_equal = bool(core1 and core2 and core1 == core2)
        pc_both = bool(pc1 and pc2)
        pc_equal = pc_both and pc1 == pc2
        raw_j = token_jaccard(tokens1, tokens2)
        core_j = token_jaccard(core1, core2)
        for group in groups:
            c = counts[group]
            c["pairs"] += 1
            c["raw_name_equal"] += raw_equal
            c["raw_casefold_name_equal"] += raw_casefold_equal
            c["core_name_equal"] += core_equal
            c["postcode_both"] += pc_both
            c["postcode_equal"] += pc_equal
            c["raw_jaccard_sum"] += raw_j
            c["core_jaccard_sum"] += core_j
    out = {}
    for group, c in counts.items():
        n = int(c["pairs"])
        pc_n = int(c["postcode_both"])
        out[group] = {
            "true_pairs": n,
            "raw_exact_name_equal": int(c["raw_name_equal"]),
            "raw_exact_name_equal_rate": c["raw_name_equal"] / n,
            "raw_casefold_name_equal": int(c["raw_casefold_name_equal"]),
            "raw_casefold_name_equal_rate": c["raw_casefold_name_equal"] / n,
            "normalized_core_equal": int(c["core_name_equal"]),
            "normalized_core_equal_rate": c["core_name_equal"] / n,
            "postcode_both_present": pc_n,
            "postcode_equal_given_both": (c["postcode_equal"] / pc_n) if pc_n else None,
            "raw_token_jaccard_mean": c["raw_jaccard_sum"] / n,
            "normalized_core_jaccard_mean": c["core_jaccard_sum"] / n,
        }
    return out


def load_pair_frame(val: pl.DataFrame, norm_dir: Path) -> pl.DataFrame:
    """Join every held-out true pair with its S1 and S2/S3 fields."""
    positives = truth_pairs(val)
    s1 = pl.scan_parquet(norm_dir / "train_s1.parquet").select(
        pl.col("entity_id").alias("s1_id"), "country",
        pl.col("name_raw").alias("s1_raw"),
        pl.col("name_core").alias("s1_core"),
        pl.col("name_latin").alias("s1_latin"),
        pl.col("postcode").alias("s1_postcode"),
        pl.col("script").alias("s1_script"),
    )
    pool = pl.concat([
        pl.scan_parquet(norm_dir / f"train_s{source}.parquet")
        .select(
            pl.col("entity_id").alias("cand_id"),
            pl.col("name_raw").alias("cand_raw"),
            pl.col("name_core").alias("cand_core"),
            pl.col("name_latin").alias("cand_latin"),
            pl.col("postcode").alias("cand_postcode"),
            pl.col("script").alias("cand_script"),
        ) for source in (2, 3)
    ])
    frame = (
        positives.lazy()
        .join(s1, on="s1_id", how="inner")
        .join(pool, on="cand_id", how="inner")
        .with_columns(
            clean_text(pl.col("s1_raw")).alias("s1_raw_tokens"),
            clean_text(pl.col("cand_raw")).alias("cand_raw_tokens"),
        )
        .collect(engine="streaming")
    )
    if frame.height != positives.height:
        raise ValueError(
            f"True-pair join returned {frame.height:,} rows for "
            f"{positives.height:,} ground-truth pairs; check IDs and data files"
        )
    return frame


def _retrieval_text(name: pl.Expr, address: pl.Expr) -> pl.Expr:
    def clean(value: pl.Expr) -> pl.Expr:
        return (
            value.fill_null("").str.to_lowercase()
            .str.replace_all(r"[^\p{L}\p{N}\p{M}]+", " ")
            .str.strip_chars()
        )
    return (clean(name) + " " + clean(address)).str.strip_chars()


def _load_country(country: str, source: int, norm_dir: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(norm_dir / f"train_s{source}.parquet")
        .filter(pl.col("country") == country)
        .select(
            "entity_id", "name_raw", "name_core",
            _retrieval_text(pl.col("name_raw"), pl.col("address_raw")).alias("retrieval_text"),
        )
        .collect(engine="streaming")
    )


def rank_one_query(
    query_raw: str | None,
    query_core: str | None,
    candidate_ids: list[str],
    candidate_raw: list[str | None],
    candidate_core: list[str | None],
    retrieval_scores: list[float],
    true_ids: set[str],
) -> dict:
    """Rank the same candidate IDs twice; ties use fixed TF-IDF then ID."""
    raw_q = (query_raw or "").casefold()
    core_q = query_core or ""
    scored = []
    for cid, raw, core, tfidf in zip(
        candidate_ids, candidate_raw, candidate_core, retrieval_scores, strict=True
    ):
        raw_score = fuzz.WRatio(raw_q, (raw or "").casefold()) if raw_q and raw else 0.0
        core_score = fuzz.WRatio(core_q, core or "") if core_q and core else 0.0
        scored.append((cid, raw_score, core_score, tfidf))
    raw_sorted = sorted(scored, key=lambda x: (-x[1], -x[3], x[0]))
    core_sorted = sorted(scored, key=lambda x: (-x[2], -x[3], x[0]))
    return {
        "retrieved_true_pairs": sum(cid in true_ids for cid in candidate_ids),
        "raw_top5_pairs": sum(cid in true_ids for cid, *_ in raw_sorted[:5]),
        "raw_top10_pairs": sum(cid in true_ids for cid, *_ in raw_sorted[:10]),
        "core_top5_pairs": sum(cid in true_ids for cid, *_ in core_sorted[:5]),
        "core_top10_pairs": sum(cid in true_ids for cid, *_ in core_sorted[:10]),
        "raw_true_ranks": [i for i, row in enumerate(raw_sorted, 1) if row[0] in true_ids],
        "core_true_ranks": [i for i, row in enumerate(core_sorted, 1) if row[0] in true_ids],
    }


def _empty_rank_counts() -> dict:
    return defaultdict(int)


def _add_query_counts(counts: dict, true_ids: set[str], result: dict) -> None:
    counts["queries"] += 1
    if not true_ids:
        return
    counts["positive_queries"] += 1
    counts["true_pairs"] += len(true_ids)
    counts["retrieved_true_pairs"] += result["retrieved_true_pairs"]
    counts["retrieval_hit_queries"] += bool(result["retrieved_true_pairs"])
    for level in ("top5", "top10"):
        for variant in ("raw", "core"):
            hits = result[f"{variant}_{level}_pairs"]
            counts[f"{variant}_{level}_pairs"] += hits
            counts[f"{variant}_{level}_hit_queries"] += bool(hits)
    if result["core_top10_pairs"] and not result["raw_top10_pairs"]:
        counts["core_wins_top10_queries"] += 1
    if result["raw_top10_pairs"] and not result["core_top10_pairs"]:
        counts["core_losses_top10_queries"] += 1


def _final_rank_counts(counts: dict) -> dict:
    result = dict(counts)
    positive = counts["positive_queries"]
    pairs = counts["true_pairs"]
    for key in ("retrieval_hit_queries", "raw_top5_hit_queries", "raw_top10_hit_queries",
                "core_top5_hit_queries", "core_top10_hit_queries"):
        result[key + "_rate"] = counts[key] / positive if positive else None
    for key in ("retrieved_true_pairs", "raw_top5_pairs", "raw_top10_pairs",
                "core_top5_pairs", "core_top10_pairs"):
        result[key + "_recall"] = counts[key] / pairs if pairs else None
    return result


def evaluate_rank(
    val: pl.DataFrame, norm_dir: Path, top_k: int = 50,
    chunk_size: int = 2_000, threads: int = 4,
    max_queries_per_country: int | None = None,
) -> dict:
    """Raw TF-IDF candidate retrieval; fixed-pool raw/core name reranking."""
    truth = {sid: set(ids or []) for sid, ids in zip(
        val["s1_id"].to_list(), val["matched_ids"].to_list(), strict=True
    )}
    val_ids = val.select("s1_id")
    by_country = {}
    process = psutil.Process()
    started = time.perf_counter()
    countries = sorted(
        pl.scan_parquet(norm_dir / "train_s1.parquet")
        .select(pl.col("country").drop_nulls().unique())
        .collect()["country"].to_list()
    )
    for country in countries:
        country_started = time.perf_counter()
        print(f"[{country}] loading normalized Parquet columns", flush=True)
        q = _load_country(country, 1, norm_dir).join(
            val_ids, left_on="entity_id", right_on="s1_id", how="semi"
        ).sort("entity_id")
        if max_queries_per_country is not None:
            q = q.sample(n=min(max_queries_per_country, q.height), seed=42).sort("entity_id")
        p = pl.concat([_load_country(country, source, norm_dir) for source in (2, 3)])
        q_ids, q_raw, q_core, q_text = (
            q[column].to_list() for column in ("entity_id", "name_raw", "name_core", "retrieval_text")
        )
        p_ids, p_raw, p_core, p_text = (
            p[column].to_list() for column in ("entity_id", "name_raw", "name_core", "retrieval_text")
        )
        del q, p
        gc.collect()
        print(f"[{country}] TF-IDF fit: {len(q_ids):,} held-out queries, "
              f"{len(p_ids):,} S2/S3 records; RSS {process.memory_info().rss/1e9:.1f} GB", flush=True)
        vectorizer = TfidfVectorizer(
            analyzer="word", token_pattern=r"\S+", min_df=2, max_df=0.01,
            sublinear_tf=True, dtype=np.float32,
        )
        pool_matrix = vectorizer.fit_transform(p_text)
        query_matrix = vectorizer.transform(q_text)
        pool_t = csr_matrix(pool_matrix.T)
        del pool_matrix, vectorizer, p_text, q_text
        gc.collect()
        print(f"[{country}] scoring; RSS {process.memory_info().rss/1e9:.1f} GB", flush=True)
        counts = _empty_rank_counts()
        examples = {"core_win": [], "core_loss": []}
        for start in range(0, len(q_ids), chunk_size):
            end = min(start + chunk_size, len(q_ids))
            top = sp_matmul_topn(
                query_matrix[start:end], pool_t, top_n=top_k,
                threshold=0.0, n_threads=threads,
            ).tocsr()
            for local in range(end - start):
                qi = start + local
                lo, hi = top.indptr[local], top.indptr[local + 1]
                pi = top.indices[lo:hi]
                result = rank_one_query(
                    q_raw[qi], q_core[qi],
                    [p_ids[i] for i in pi],
                    [p_raw[i] for i in pi],
                    [p_core[i] for i in pi],
                    top.data[lo:hi].tolist(),
                    truth[q_ids[qi]],
                )
                _add_query_counts(counts, truth[q_ids[qi]], result)
                core_win = bool(result["core_top10_pairs"] and not result["raw_top10_pairs"])
                core_loss = bool(result["raw_top10_pairs"] and not result["core_top10_pairs"])
                label = "core_win" if core_win else "core_loss" if core_loss else None
                if label and len(examples[label]) < 5:
                    true_candidates = [
                        {"cand_id": p_ids[i], "name_raw": p_raw[i], "name_core": p_core[i]}
                        for i in pi if p_ids[i] in truth[q_ids[qi]]
                    ]
                    examples[label].append({
                        "s1_id": q_ids[qi], "s1_name_raw": q_raw[qi],
                        "s1_name_core": q_core[qi],
                        "true_candidates_in_top50": true_candidates,
                        "raw_true_ranks": result["raw_true_ranks"],
                        "core_true_ranks": result["core_true_ranks"],
                    })
            if end % 20_000 < chunk_size or end == len(q_ids):
                print(f"[{country}] {end:,}/{len(q_ids):,} queries; "
                      f"RSS {process.memory_info().rss/1e9:.1f} GB", flush=True)
            del top
        by_country[country] = _final_rank_counts(counts)
        by_country[country]["examples"] = examples
        by_country[country]["seconds"] = round(time.perf_counter() - country_started, 2)
        del query_matrix, pool_t, q_ids, q_raw, q_core, p_ids, p_raw, p_core
        gc.collect()
    overall = _empty_rank_counts()
    for result in by_country.values():
        for key, value in result.items():
            if key not in ("seconds", "examples") and not key.endswith(("_rate", "_recall")):
                overall[key] += value
    return {
        "method": "raw name+address word TF-IDF top-50, fixed pool, RapidFuzz WRatio name rerank",
        "top_k": top_k,
        "max_queries_per_country": max_queries_per_country,
        "by_country": by_country,
        "overall": _final_rank_counts(overall),
        "seconds": round(time.perf_counter() - started, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norm-dir", type=Path, default=NORM_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=2_000)
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 4, 4))
    parser.add_argument("--max-queries-per-country", type=int)
    parser.add_argument("--skip-rank", action="store_true")
    args = parser.parse_args()
    if args.top_k < 10 or args.chunk_size < 1 or args.threads < 1:
        parser.error("top-k must be >=10; chunk-size and threads must be positive")
    started = time.perf_counter()
    val = add_is_val(load_ground_truth()).filter(pl.col("is_val")).drop("is_val")
    print(f"held-out validation S1: {val.height:,}", flush=True)
    pairs = load_pair_frame(val, args.norm_dir)
    pair_metrics = score_true_pairs(pairs)
    del pairs
    gc.collect()
    print(f"true validation pairs: {pair_metrics['overall']['true_pairs']:,}", flush=True)
    rank = None if args.skip_rank else evaluate_rank(
        val, args.norm_dir, args.top_k, args.chunk_size,
        args.threads, args.max_queries_per_country,
    )
    result = {
        "validation_s1": val.height,
        "split_rule": "crc32(s1_id) % 10 == 0",
        "pair_metrics": pair_metrics,
        "rank": rank,
        "seconds": round(time.perf_counter() - started, 2),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"report: {args.report}; total {result['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
