"""Phase 10 held-out Devanagari-to-Latin pair and top-10 evaluation.

Uses the same full India S2/S3 raw-text TF-IDF top-50 pool as Phase 9. The
candidate pool is fixed; only Devanagari candidate name scoring changes from
name_core to learned/fallback name_latin. Validation labels only score output.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from src.common.io import REPO_ROOT, load_ground_truth
from src.common.split import add_is_val
from .evaluate import NORM_DIR, _retrieval_text, load_pair_frame, token_jaccard
from .transliteration import has_devanagari, load_token_map


DEFAULT_REPORT = REPO_ROOT / "data" / "eda" / "phase10_metrics.json"


def score_devanagari_pairs(pairs: pl.DataFrame, token_map: dict[str, str]) -> dict:
    """Direct name metrics on held-out Latin-S1/Devanagari-candidate pairs."""
    dev = pairs.filter(pl.col("cand_script") == "devanagari")
    n = dev.height
    counts = {
        "pairs": n, "positive_s1": dev["s1_id"].n_unique(),
        "core_exact": 0, "latin_exact": 0,
        "core_jaccard_sum": 0.0, "latin_jaccard_sum": 0.0,
        "core_wratio_sum": 0.0, "latin_wratio_sum": 0.0,
        "dev_tokens": 0, "mapped_dev_tokens": 0,
        "nonlatin_output_pairs": 0,
    }
    for core, latin, cand_core, cand_latin in dev.select(
        "s1_core", "s1_latin", "cand_core", "cand_latin"
    ).iter_rows():
        core, latin = core or "", latin or ""
        cand_core, cand_latin = cand_core or "", cand_latin or ""
        counts["core_exact"] += bool(core and cand_core and core == cand_core)
        counts["latin_exact"] += bool(latin and cand_latin and latin == cand_latin)
        counts["core_jaccard_sum"] += token_jaccard(core, cand_core)
        counts["latin_jaccard_sum"] += token_jaccard(latin, cand_latin)
        counts["core_wratio_sum"] += fuzz.WRatio(core, cand_core) if core and cand_core else 0
        counts["latin_wratio_sum"] += fuzz.WRatio(latin, cand_latin) if latin and cand_latin else 0
        counts["nonlatin_output_pairs"] += has_devanagari(cand_latin)
        for token in cand_core.split():
            if has_devanagari(token):
                counts["dev_tokens"] += 1
                counts["mapped_dev_tokens"] += token in token_map
    return {
        "true_devanagari_pairs": n,
        "positive_s1_with_devanagari_match": counts["positive_s1"],
        "core_exact_rate": counts["core_exact"] / n if n else None,
        "latin_exact_rate": counts["latin_exact"] / n if n else None,
        "core_token_jaccard_mean": counts["core_jaccard_sum"] / n if n else None,
        "latin_token_jaccard_mean": counts["latin_jaccard_sum"] / n if n else None,
        "core_wratio_mean": counts["core_wratio_sum"] / n if n else None,
        "latin_wratio_mean": counts["latin_wratio_sum"] / n if n else None,
        "mapped_devanagari_token_instances": counts["mapped_dev_tokens"],
        "all_devanagari_token_instances": counts["dev_tokens"],
        "mapped_devanagari_token_instance_rate": (
            counts["mapped_dev_tokens"] / counts["dev_tokens"]
            if counts["dev_tokens"] else None
        ),
        "nonlatin_output_pairs": counts["nonlatin_output_pairs"],
    }


def rank_same_pool(
    query_core: str, query_latin: str, candidate_ids: list[str],
    candidate_core: list[str], candidate_latin: list[str],
    candidate_script: list[str], retrieval_scores: list[float],
    true_dev_ids: set[str], true_all_ids: set[str],
) -> dict:
    """Change only Devanagari candidate name scoring, never the pool."""
    scored = []
    for cid, core, latin, script, retrieval in zip(
        candidate_ids, candidate_core, candidate_latin, candidate_script,
        retrieval_scores, strict=True
    ):
        baseline = fuzz.WRatio(query_core, core) if query_core and core else 0.0
        translit = (
            fuzz.WRatio(query_latin, latin) if query_latin and latin else 0.0
        ) if script == "devanagari" else baseline
        scored.append((cid, baseline, translit, retrieval))
    baseline_ranked = sorted(scored, key=lambda x: (-x[1], -x[3], x[0]))
    translit_ranked = sorted(scored, key=lambda x: (-x[2], -x[3], x[0]))

    def hits(items: list[tuple], truth: set[str], k: int) -> int:
        return sum(row[0] in truth for row in items[:k])

    return {
        "retrieved_dev_pairs": sum(cid in true_dev_ids for cid in candidate_ids),
        "baseline_dev_top5_pairs": hits(baseline_ranked, true_dev_ids, 5),
        "baseline_dev_top10_pairs": hits(baseline_ranked, true_dev_ids, 10),
        "latin_dev_top5_pairs": hits(translit_ranked, true_dev_ids, 5),
        "latin_dev_top10_pairs": hits(translit_ranked, true_dev_ids, 10),
        "baseline_all_top10_pairs": hits(baseline_ranked, true_all_ids, 10),
        "latin_all_top10_pairs": hits(translit_ranked, true_all_ids, 10),
        "baseline_dev_true_ranks": [
            rank for rank, row in enumerate(baseline_ranked, 1) if row[0] in true_dev_ids
        ],
        "latin_dev_true_ranks": [
            rank for rank, row in enumerate(translit_ranked, 1) if row[0] in true_dev_ids
        ],
    }


def _load_country(source: int, norm_dir: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(norm_dir / f"train_s{source}.parquet")
        .filter(pl.col("country") == "India")
        .select(
            "entity_id", "name_core", "name_latin", "script",
            _retrieval_text(pl.col("name_raw"), pl.col("address_raw")).alias("retrieval_text"),
        )
        .collect(engine="streaming")
    )


def evaluate_rank(
    dev_pairs: pl.DataFrame, val: pl.DataFrame, norm_dir: Path,
    top_k: int = 50, chunk_size: int = 2_000, threads: int = 4,
) -> dict:
    """Full India pool; only S1s with at least one true Devanagari match."""
    started = time.perf_counter()
    grouped = dev_pairs.group_by("s1_id").agg(pl.col("cand_id"))
    true_dev = {
        sid: set(ids) for sid, ids in zip(
            grouped["s1_id"].to_list(), grouped["cand_id"].to_list(),
            strict=True,
        )
    }
    true_all = {
        sid: set(ids or []) for sid, ids in zip(
            val["s1_id"].to_list(), val["matched_ids"].to_list(), strict=True
        )
    }
    q = _load_country(1, norm_dir).join(
        pl.DataFrame({"s1_id": list(true_dev)}),
        left_on="entity_id", right_on="s1_id", how="semi"
    ).sort("entity_id")
    p = pl.concat([_load_country(source, norm_dir) for source in (2, 3)])
    q_ids, q_core, q_latin, q_text = (
        q[col].to_list() for col in ("entity_id", "name_core", "name_latin", "retrieval_text")
    )
    p_ids, p_core, p_latin, p_script, p_text = (
        p[col].to_list() for col in (
            "entity_id", "name_core", "name_latin", "script", "retrieval_text"
        )
    )
    del q, p
    gc.collect()
    print(f"India top-{top_k}: {len(q_ids):,} Hindi-positive S1, "
          f"{len(p_ids):,} full S2/S3 pool", flush=True)
    vectorizer = TfidfVectorizer(
        analyzer="word", token_pattern=r"\S+", min_df=2,
        max_df=0.01, sublinear_tf=True, dtype=np.float32,
    )
    p_matrix = vectorizer.fit_transform(p_text)
    q_matrix = vectorizer.transform(q_text)
    p_t = p_matrix.T.tocsr()
    del vectorizer, p_matrix, p_text, q_text
    gc.collect()
    totals = {
        "queries": 0, "dev_true_pairs": 0, "all_true_pairs": 0,
        "retrieved_dev_pairs": 0, "retrieval_dev_hit_queries": 0,
        "baseline_dev_top5_pairs": 0, "baseline_dev_top10_pairs": 0,
        "latin_dev_top5_pairs": 0, "latin_dev_top10_pairs": 0,
        "baseline_dev_top5_hit_queries": 0, "baseline_dev_top10_hit_queries": 0,
        "latin_dev_top5_hit_queries": 0, "latin_dev_top10_hit_queries": 0,
        "baseline_all_top10_pairs": 0, "latin_all_top10_pairs": 0,
        "latin_wins_top10_queries": 0, "latin_losses_top10_queries": 0,
    }
    examples = {"latin_win": [], "latin_loss": []}
    for start in range(0, len(q_ids), chunk_size):
        end = min(start + chunk_size, len(q_ids))
        top = sp_matmul_topn(
            q_matrix[start:end], p_t, top_n=top_k,
            threshold=0.0, n_threads=threads,
        ).tocsr()
        for local in range(end - start):
            qi = start + local
            sid = q_ids[qi]
            lo, hi = top.indptr[local], top.indptr[local + 1]
            pi = top.indices[lo:hi]
            result = rank_same_pool(
                q_core[qi] or "", q_latin[qi] or "",
                [p_ids[i] for i in pi], [p_core[i] or "" for i in pi],
                [p_latin[i] or "" for i in pi], [p_script[i] or "" for i in pi],
                top.data[lo:hi].tolist(), true_dev[sid], true_all[sid],
            )
            totals["queries"] += 1
            totals["dev_true_pairs"] += len(true_dev[sid])
            totals["all_true_pairs"] += len(true_all[sid])
            for key in (
                "retrieved_dev_pairs", "baseline_dev_top5_pairs", "baseline_dev_top10_pairs",
                "latin_dev_top5_pairs", "latin_dev_top10_pairs",
                "baseline_all_top10_pairs", "latin_all_top10_pairs",
            ):
                totals[key] += result[key]
            totals["retrieval_dev_hit_queries"] += bool(result["retrieved_dev_pairs"])
            for method in ("baseline", "latin"):
                for level in ("top5", "top10"):
                    totals[f"{method}_dev_{level}_hit_queries"] += bool(
                        result[f"{method}_dev_{level}_pairs"]
                    )
            win = bool(result["latin_dev_top10_pairs"] and not result["baseline_dev_top10_pairs"])
            loss = bool(result["baseline_dev_top10_pairs"] and not result["latin_dev_top10_pairs"])
            totals["latin_wins_top10_queries"] += win
            totals["latin_losses_top10_queries"] += loss
            label = "latin_win" if win else "latin_loss" if loss else None
            if label and len(examples[label]) < 5:
                examples[label].append({
                    "s1_id": sid, "s1_name_core": q_core[qi],
                    "true_dev_candidates": [
                        {"cand_id": p_ids[i], "name_core": p_core[i], "name_latin": p_latin[i]}
                        for i in pi if p_ids[i] in true_dev[sid]
                    ],
                    "baseline_true_ranks": result["baseline_dev_true_ranks"],
                    "latin_true_ranks": result["latin_dev_true_ranks"],
                })
        if end % 5_000 < chunk_size or end == len(q_ids):
            print(f"India Hindi rank: {end:,}/{len(q_ids):,}", flush=True)
    if totals["queries"] != len(true_dev):
        raise ValueError("Missing a Hindi-positive validation S1 from the rank run")
    result = dict(totals)
    for key in (
        "retrieval_dev_hit_queries", "baseline_dev_top5_hit_queries",
        "baseline_dev_top10_hit_queries", "latin_dev_top5_hit_queries",
        "latin_dev_top10_hit_queries",
    ):
        result[key + "_rate"] = totals[key] / totals["queries"]
    for key in (
        "retrieved_dev_pairs", "baseline_dev_top5_pairs", "baseline_dev_top10_pairs",
        "latin_dev_top5_pairs", "latin_dev_top10_pairs",
    ):
        result[key + "_recall"] = totals[key] / totals["dev_true_pairs"]
    for key in ("baseline_all_top10_pairs", "latin_all_top10_pairs"):
        result[key + "_recall"] = totals[key] / totals["all_true_pairs"]
    result["examples"] = examples
    result["seconds"] = round(time.perf_counter() - started, 2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norm-dir", type=Path, default=NORM_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-rank", action="store_true")
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 4, 4))
    args = parser.parse_args()
    started = time.perf_counter()
    token_map = load_token_map(args.norm_dir / "hi_latin_map.json")
    if not token_map:
        parser.error("train the local map first: python -m src.normalize.train_transliteration")
    val = add_is_val(load_ground_truth()).filter(pl.col("is_val")).drop("is_val")
    pairs = load_pair_frame(val, args.norm_dir)
    pair_metrics = score_devanagari_pairs(pairs, token_map)
    dev_pairs = pairs.filter(pl.col("cand_script") == "devanagari").select("s1_id", "cand_id")
    del pairs
    gc.collect()
    rank = None if args.skip_rank else evaluate_rank(dev_pairs, val, args.norm_dir, threads=args.threads)
    out = {
        "validation_rule": "crc32(s1_id) % 10 == 0",
        "token_map_entries": len(token_map),
        "pair_metrics": pair_metrics,
        "rank": rank,
        "seconds": round(time.perf_counter() - started, 2),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.report}; total {out['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
