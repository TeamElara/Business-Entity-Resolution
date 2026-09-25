"""Phase 12: US-only model fit, then US and India held-out evaluation.

Reuse Arihant's baseline_v0 candidate index, 22 features, LightGBM parameters,
top-8 rule, and macro F0.5. The alternate experiments change only Devanagari
name fields to Latin normalization; the US-trained model and all other fields
stay fixed. The default Latin variant uses a token map learned from India
non-validation matches; --fallback-only uses no India-learned map. Neither
variant trains or tunes the matcher on India validation labels. France has no
labels here.

Run from the repo root::

    python -m src.normalize.evaluate_generalization

Local artifacts are under the gitignored data/eda/phase12 directory. Use
--fit-s1 200 --val-s1 100 --out-dir data/eda/phase12_smoke for a smoke run.
Use a new --out-dir if changing fit-s1, val-s1, or pool-limit: existing US
feature/model caches are reused within that directory.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from src.common.io import REPO_ROOT, load_ground_truth, load_source, scan_source, truth_pairs
from src.common.metrics import blocking_report, macro_f05_fast
from src.common.split import add_is_val
# baseline_v0 imports sparse_dot_topn before LightGBM, required on macOS.
from src.matching import baseline_v0 as base
from .basic import normalize_df

import lightgbm as lgb


DEFAULT_OUT = REPO_ROOT / "data" / "eda" / "phase12"
FIXED_T, FIXED_T1 = 0.65, 0.55  # Arihant's published baseline-v0 thresholds


def prepare_with_latin(
    df: pl.DataFrame, token_map: dict[str, str] | None = None
) -> pl.DataFrame:
    """Baseline-v0 preparation with only Devanagari name comparison changed."""
    norm = normalize_df(df, transliteration_map=token_map)
    norm = norm.with_columns(
        pl.col("name_norm").str.replace_all(base.LEGAL_RE, " ")
        .str.replace_all(r"\s+", " ").str.strip_chars().alias("name_core")
    ).with_columns(
        pl.when(pl.col("name_core") == "")
        .then(pl.col("name_norm")).otherwise(pl.col("name_core"))
        .alias("name_core")
    ).with_columns(
        pl.when((pl.col("script") == "devanagari") & (pl.col("name_latin") != ""))
        .then(pl.col("name_latin")).otherwise(pl.col("name_core"))
        .alias("name_core")
    )
    return (
        norm.with_columns(
            pl.col("name_core").str.replace_all(" ", "").alias("name_key"),
            pl.col("addr_norm").str.extract(r"\b(\d+)\b", 1).alias("house_no"),
            pl.col("addr_norm").str.extract_all(r"\b\d{5,6}\b").list.last().alias("pin"),
            pl.col("addr_norm").str.extract_all(r"\d+").list.unique().alias("nums"),
            (~pl.col("name_norm").str.contains(r"[a-z]")).alias("nonlatin"),
            (pl.col("addr_norm") == "").alias("addr_missing"),
            pl.col("entity_id").str.starts_with("S3-").alias("is_s3"),
        )
        .select("entity_id", "name_core", "name_key", "addr_norm", "house_no",
                "pin", "nums", "nonlatin", "addr_missing", "is_s3")
    )


def _country_features(
    country: str, query_sets: dict[str, pl.DataFrame], on_chunk,
    prepare_fn=base.prepare, chunk: int = 5_000, threads: int = 4,
    pool_limit: int | None = None,
) -> int:
    """Same full-country index/features as baseline_v0, with bounded chunks."""
    started = time.perf_counter()
    if pool_limit is None:
        pool_raw = pl.concat([
            load_source("train", source, country=country) for source in (2, 3)
        ])
    else:
        pool_raw = pl.concat([
            scan_source("train", source, country=country).head(pool_limit).collect()
            for source in (2, 3)
        ])
    pool = prepare_fn(pool_raw)
    del pool_raw
    gc.collect()
    base.THREADS = threads
    index = base.CountryIndex(pool)
    print(f"{country}: full S2/S3 index {pool.height:,} rows in "
          f"{time.perf_counter()-started:.1f}s", flush=True)
    for label, s1 in query_sets.items():
        q_all = prepare_fn(s1)
        for start in range(0, q_all.height, chunk):
            q = q_all.slice(start, chunk)
            on_chunk(label, base.features(index.pairs(q), q, pool))
            if (start // chunk + 1) % 10 == 0 or start + chunk >= q_all.height:
                print(f"{country} {label}: {min(start+chunk, q_all.height):,}"
                      f"/{q_all.height:,} S1", flush=True)
        del q_all
    del index, pool
    gc.collect()
    return round(time.perf_counter() - started, 2)


class _ParquetSink:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.writers = {}
        self.rows = {}

    def write(self, name: str, frame: pl.DataFrame) -> None:
        if frame.is_empty():
            return
        path = self.out_dir / f"{name}.parquet.partial"
        table = frame.to_arrow()
        if name not in self.writers:
            self.writers[name] = pq.ParquetWriter(path, table.schema, compression="zstd")
            self.rows[name] = 0
        self.writers[name].write_table(table)
        self.rows[name] += frame.height

    def close(self) -> None:
        for name, writer in self.writers.items():
            writer.close()
            os.replace(
                self.out_dir / f"{name}.parquet.partial",
                self.out_dir / f"{name}.parquet",
            )


def _sample_s1(s1: pl.DataFrame, truth: pl.DataFrame, fit_n: int, val_n: int | None):
    val = truth.filter(pl.col("is_val")).select("s1_id")
    us = s1.filter(pl.col("country") == "US")
    india = s1.filter(pl.col("country") == "India")
    us_nonval = us.join(val, left_on="entity_id", right_on="s1_id", how="anti")
    us_fit = us_nonval.filter(
        pl.int_range(pl.len()).shuffle(base.SEED) < fit_n
    )
    us_val = us.join(val, left_on="entity_id", right_on="s1_id", how="semi")
    india_val = india.join(val, left_on="entity_id", right_on="s1_id", how="semi")
    if val_n is not None:
        us_val = us_val.sort("entity_id").head(val_n)
        india_val = india_val.sort("entity_id").head(val_n)
    selected = {
        "fit": us_fit, "us_val": us_val, "india_val": india_val,
    }
    truths = {
        key: truth.join(frame.select(pl.col("entity_id").alias("s1_id")),
                        on="s1_id", how="semi").drop("is_val")
        for key, frame in selected.items()
    }
    for key, frame in selected.items():
        print(f"{key}: {frame.height:,} S1", flush=True)
    return selected, truths


def _label(frame: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    positives = truth_pairs(truth).with_columns(pl.lit(1, pl.Int8).alias("label"))
    return frame.join(positives, on=["s1_id", "cand_id"], how="left").with_columns(
        pl.col("label").fill_null(0)
    )


def _train_model(
    fit_path: Path, val_path: Path, fit_truth: pl.DataFrame,
    val_truth: pl.DataFrame, model_path: Path, threads: int,
) -> tuple[lgb.Booster, dict]:
    fit = _label(pl.read_parquet(fit_path), fit_truth)
    # Early stopping uses only a deterministic 20k-S1 subset of US validation.
    stop_ids = val_truth.sort("s1_id").head(20_000)["s1_id"].to_list()
    stop = _label(
        pl.scan_parquet(val_path).filter(pl.col("s1_id").is_in(stop_ids)).collect(),
        val_truth,
    )
    print(f"model fit: {fit.height:,} pairs, {fit['label'].sum():,} positives; "
          f"early-stop US val: {stop.height:,} pairs", flush=True)
    params = dict(
        objective="binary", learning_rate=0.05, num_leaves=63,
        min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.8,
        bagging_freq=1, seed=base.SEED, num_threads=threads, verbose=-1,
    )
    train = lgb.Dataset(
        fit.select(base.FEATURES).to_numpy(), fit["label"].to_numpy(),
        feature_name=base.FEATURES,
    )
    valid = lgb.Dataset(
        stop.select(base.FEATURES).to_numpy(), stop["label"].to_numpy(),
        reference=train,
    )
    model = lgb.train(
        params, train, num_boost_round=600, valid_sets=[valid],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    model.save_model(str(model_path))
    importance = sorted(zip(base.FEATURES, model.feature_importance("gain")),
                        key=lambda item: -item[1])
    info = {
        "fit_pairs": fit.height, "fit_positives": int(fit["label"].sum()),
        "early_stop_pairs": stop.height, "best_iteration": model.best_iteration,
        "top_gain_features": [(name, round(float(gain), 1))
                              for name, gain in importance[:10]],
    }
    del fit, stop, train, valid
    gc.collect()
    return model, info


def _macro_numpy(
    qi: np.ndarray, rank: np.ndarray, prob: np.ndarray, label: np.ndarray,
    n_true: np.ndarray, t: float, t1: float,
) -> float:
    """Fast exact per-S1 F0.5 for a threshold trial over ranked arrays."""
    chosen = (prob >= t) | ((rank == 1) & (prob >= t1))
    predicted = np.bincount(qi, weights=chosen, minlength=len(n_true))
    hits = np.bincount(qi, weights=chosen & label, minlength=len(n_true))
    score = np.zeros(len(n_true), dtype=np.float64)
    empty = n_true == 0
    score[empty] = predicted[empty] == 0
    good = (~empty) & (hits > 0)
    precision = hits[good] / predicted[good]
    recall = hits[good] / n_true[good]
    score[good] = 1.25 * precision * recall / (0.25 * precision + recall)
    return float(score.mean())


def _tune(cand: pl.DataFrame, truth: pl.DataFrame) -> tuple[float, float, float]:
    ids = truth.select("s1_id").with_row_index("qi")
    rows = _label(cand, truth).join(ids, on="s1_id")
    qi = rows["qi"].to_numpy()
    rank = rows["block_rank"].to_numpy()
    prob = rows["prob"].to_numpy()
    label = rows["label"].to_numpy() == 1
    n_true = truth["matched_ids"].list.len().to_numpy()
    best = (-1.0, 0.0, 0.0)
    for t in np.arange(0.20, 0.81, 0.05):
        for t1 in np.arange(0.05, t + 1e-9, 0.05):
            score = _macro_numpy(qi, rank, prob, label, n_true, float(t), float(t1))
            if score > best[0]:
                best = (score, round(float(t), 2), round(float(t1), 2))
    return best


class _Scorer:
    def __init__(self, model: lgb.Booster, truth: pl.DataFrame):
        self.model = model
        self.truth = truth
        self.true_pairs = truth_pairs(truth).select("s1_id", "cand_id")
        self.kept = []
        self.stage1_hits = []
        self.stage1_pairs = 0

    def add(self, frame: pl.DataFrame) -> None:
        if frame.is_empty():
            return
        self.stage1_pairs += frame.height
        hits = frame.select("s1_id", "cand_id").join(
            self.true_pairs, on=["s1_id", "cand_id"], how="semi"
        )
        if not hits.is_empty():
            self.stage1_hits.append(hits)
        probs = self.model.predict(frame.select(base.FEATURES).to_numpy())
        scored = frame.select("s1_id", "cand_id").with_columns(
            pl.Series("prob", probs, dtype=pl.Float32)
        )
        self.kept.append(base.topk_by_prob(scored))

    def finish(self, country: str, tuned: tuple[float, float] | None = None):
        cand = pl.concat(self.kept)
        hits = pl.concat(self.stage1_hits) if self.stage1_hits else self.true_pairs.head(0)
        n_true = int(self.truth["matched_ids"].list.len().sum())
        per = (
            self.truth.select("s1_id", pl.col("matched_ids").list.len().alias("n_true"))
            .join(hits.group_by("s1_id").len("tp"), on="s1_id", how="left")
            .with_columns(pl.col("tp").fill_null(0))
        )
        r = pl.col("tp") / pl.col("n_true")
        oracle = per.select(
            pl.when(pl.col("n_true") == 0).then(1.0)
            .when(pl.col("tp") == 0).then(0.0)
            .otherwise(1.25 * r / (0.25 + r)).mean()
        ).item()
        pool_size = 6_186_873 if country == "US" else 4_133_346
        # These are the full train S2+S3 country pool counts. In a smoke run
        # with --pool-limit, the reduction ratio alone is not meaningful.
        block = blocking_report(
            cand, self.truth, s1_country=None, pool_size=pool_size,
            name=f"Phase 12 {country} top8", verbose=False,
        )
        result = {
            "validation_s1": self.truth.height,
            "true_pairs": n_true,
            "stage1_pairs": self.stage1_pairs,
            "stage1_mean_candidates": self.stage1_pairs / self.truth.height,
            "stage1_pair_recall": hits.height / n_true if n_true else None,
            "stage1_oracle_f05": oracle,
            "top8_mean_candidates": block["mean_cands"],
            "top8_pair_recall": block["pair_recall"],
            "top8_oracle_f05": block["oracle_ceiling"],
            "fixed_threshold_macro_f05": macro_f05_fast(
                base.decide(cand, FIXED_T, FIXED_T1), self.truth
            ),
        }
        if tuned is not None:
            t, t1 = tuned
            result["us_tuned_threshold_macro_f05"] = macro_f05_fast(
                base.decide(cand, t, t1), self.truth
            )
        print(f"{country}: stage1 recall {result['stage1_pair_recall']:.4f}, "
              f"top8 oracle {result['top8_oracle_f05']:.4f}, "
              f"fixed F0.5 {result['fixed_threshold_macro_f05']:.4f}", flush=True)
        return result, cand, hits


def _score_parquet(path: Path, scorer: _Scorer) -> None:
    source = pq.ParquetFile(path)
    for group in range(source.num_row_groups):
        scorer.add(pl.from_arrow(source.read_row_group(group)))
        if (group + 1) % 10 == 0 or group + 1 == source.num_row_groups:
            print(f"US validation scoring: {group+1}/{source.num_row_groups} chunks",
                  flush=True)


def _dev_diagnostic(
    truth: pl.DataFrame, stage1_hits: pl.DataFrame, top8: pl.DataFrame,
) -> dict:
    devanagari = pl.concat([
        pl.scan_parquet(REPO_ROOT / "data" / "norm" / f"train_s{s}.parquet")
        .filter(pl.col("script") == "devanagari")
        .select(pl.col("entity_id").alias("cand_id"))
        for s in (2, 3)
    ]).collect()
    dev_true = truth_pairs(truth).join(devanagari, on="cand_id", how="semi")
    dev_stage1 = stage1_hits.join(dev_true, on=["s1_id", "cand_id"], how="semi")
    dev_top8 = top8.select("s1_id", "cand_id").join(
        dev_true, on=["s1_id", "cand_id"], how="semi"
    )
    return {
        "devanagari_true_pairs": dev_true.height,
        "devanagari_positive_s1": dev_true["s1_id"].n_unique(),
        "devanagari_stage1_pairs": dev_stage1.height,
        "devanagari_top8_pairs": dev_top8.height,
        "devanagari_stage1_pair_recall": dev_stage1.height / dev_true.height,
        "devanagari_top8_pair_recall": dev_top8.height / dev_true.height,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-s1", type=int, default=base.TRAIN_S1_PER_COUNTRY)
    parser.add_argument("--val-s1", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=5_000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pool-limit", type=int, default=None,
                        help="smoke test only: first N rows per pool source")
    parser.add_argument("--skip-latin-variant", action="store_true")
    parser.add_argument("--fallback-only", action="store_true",
                        help="run only code-native Hindi fallback (no India-trained map)")
    parser.add_argument("--skip-india-baseline", action="store_true",
                        help="reuse a previous baseline report; skip its expensive India pass")
    parser.add_argument("--report-name", default="metrics.json")
    args = parser.parse_args()
    if (args.fit_s1 < 1 or args.chunk < 1 or args.threads < 1
            or (args.val_s1 is not None and args.val_s1 < 1)
            or (args.pool_limit is not None and args.pool_limit < 1)):
        parser.error("fit-s1, val-s1, chunk, threads, and pool-limit must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    s1 = load_source("train", 1)
    truth = add_is_val(load_ground_truth())
    selected, truths = _sample_s1(s1, truth, args.fit_s1, args.val_s1)
    del s1, truth
    gc.collect()

    fit_path = args.out_dir / "fit.parquet"
    us_val_path = args.out_dir / "us_val.parquet"
    if not (fit_path.is_file() and us_val_path.is_file()):
        sink = _ParquetSink(args.out_dir)
        elapsed_us = _country_features(
            "US", {"fit": selected["fit"], "us_val": selected["us_val"]},
            sink.write, chunk=args.chunk, threads=args.threads,
            pool_limit=args.pool_limit,
        )
        sink.close()
        print(f"US feature generation {elapsed_us:.1f}s; rows {sink.rows}", flush=True)
    del selected["fit"]
    gc.collect()

    model_path = args.out_dir / "us_only_lgb.txt"
    if model_path.is_file():
        model = lgb.Booster(model_file=str(model_path))
        model_info = {"saved_trees": model.num_trees(), "reused": True}
    else:
        model, model_info = _train_model(
            fit_path, us_val_path, truths["fit"], truths["us_val"],
            model_path, args.threads,
        )
    us_scorer = _Scorer(model, truths["us_val"])
    _score_parquet(us_val_path, us_scorer)
    us_result, us_top8, _ = us_scorer.finish("US")
    tuned_score, t, t1 = _tune(us_top8, truths["us_val"])
    check = macro_f05_fast(base.decide(us_top8, t, t1), truths["us_val"])
    if abs(tuned_score - check) > 1e-6:
        raise AssertionError(f"threshold tuner disagrees with reference scorer: {tuned_score}, {check}")
    us_result["us_tuned_threshold_macro_f05"] = check
    print(f"US-only thresholds: t={t}, t1={t1}, US val F0.5={check:.4f}", flush=True)
    del us_scorer, us_top8
    gc.collect()

    india_result, elapsed_india = None, None
    if not args.skip_india_baseline:
        india_scorer = _Scorer(model, truths["india_val"])
        elapsed_india = _country_features(
            "India", {"india_val": selected["india_val"]},
            lambda _name, frame: india_scorer.add(frame),
            chunk=args.chunk, threads=args.threads, pool_limit=args.pool_limit,
        )
        india_result, india_top8, india_hits = india_scorer.finish("India", (t, t1))
        india_result["devanagari"] = _dev_diagnostic(
            truths["india_val"], india_hits, india_top8
        )
        del india_scorer, india_top8, india_hits
        gc.collect()

    latin_result = None
    if not args.skip_latin_variant:
        latin_scorer = _Scorer(model, truths["india_val"])
        token_map = {} if args.fallback_only else None
        elapsed_latin = _country_features(
            "India", {"india_val": selected["india_val"]},
            lambda _name, frame: latin_scorer.add(frame),
            prepare_fn=lambda df: prepare_with_latin(df, token_map=token_map),
            chunk=args.chunk, threads=args.threads,
            pool_limit=args.pool_limit,
        )
        latin_result, latin_top8, latin_hits = latin_scorer.finish("India", (t, t1))
        latin_result["devanagari"] = _dev_diagnostic(
            truths["india_val"], latin_hits, latin_top8
        )
        del latin_scorer, latin_top8, latin_hits
    else:
        elapsed_latin = None

    report = {
        "validation_rule": "crc32(s1_id) % 10 == 0",
        "training": "US non-validation S1 only; full same-country S2/S3 pools",
        "feature_recipe": "Arihant baseline_v0, 22 country-agnostic features, top-8",
        "fit_s1": args.fit_s1, "val_s1_limit_per_country": args.val_s1,
        "pool_limit_per_source": args.pool_limit,
        "model": model_info, "us_tuned_thresholds": {"t": t, "t1": t1},
        "us_val": us_result, "india_val_baseline": india_result,
        "india_val_latin_candidate_variant": latin_result,
        "latin_variant_uses_learned_india_map": (
            not args.fallback_only if latin_result is not None else None
        ),
        "seconds": round(time.perf_counter() - started, 2),
        "india_feature_seconds": elapsed_india,
        "india_latin_feature_seconds": elapsed_latin,
    }
    path = args.out_dir / args.report_name
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {path}; total {report['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
