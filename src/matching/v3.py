"""Matching v3: v2 + all six of Ojaswi's blocks, pool-size-independent features, cached test
features, test-like decisions.

Changes from v2 (src/matching/v2.py, reused here):

  - stage 1 = v1's two TF-IDF blocks  U  all blocks in src/blocking/stage1.BLOCKS (word, skel,
    noaddr, namenum, concat, namehouse), fitted with Ojaswi's fit_tfidf. Text cleaning includes her
    zero-width-joiner fix and the re-learned script map.
  - features that grow with the pool size are made pool-size independent: absolute name counts are
    dropped, IDF values are divided by log(pool size). (v2's test pools are smaller than the train
    pools, so absolute counts made test names look rarer and the model over-predicted.)
  - optional record-level features joined from files when present: reverse search (top S1 per
    S2/S3 record; is this S1 the record's best S1?) and an orphan score (record matches no S1).
  - test: the pruner's top CACHE_K candidates per S1 are cached with their pair features, so any
    final-set size / final model / decision rule can be re-scored on test in minutes.
  - decisions: thresholds, plus "one record -> at most one S1" (every S2/S3 record belongs to at
    most one S1 in the ground truth): a record claimed by several S1 stays only with the S1 that
    gives it the highest probability.

    python -m src.matching.v3 features
    python -m src.matching.v3 prune
    python -m src.matching.v3 train --final-k 6
    python -m src.matching.v3 test                      # full run + cache
    python -m src.matching.v3 rescore --t 0.85          # new decisions from the cache, minutes
"""
import argparse
import functools
import gc
import json
import time

from sparse_dot_topn import sp_matmul_topn  # isort: skip  (before lightgbm, see v1)

import lightgbm as lgb
import numpy as np
import polars as pl

from src.blocking import stage1 as oj
from src.blocking.text import load_script_map
from src.common import add_is_val, load_ground_truth, truth_pairs
from src.common.io import REPO_ROOT
from src.matching import v1, v2
from src.matching.v1 import CHUNK, SEED, THREADS, countries_of, load_norm, prepare, select_candidates

TRAIN_A_PER_COUNTRY = 80_000
TRAIN_B_PER_COUNTRY = 120_000
FINAL_K = 6
P_MIN = 0.003
CACHE_K = 10

FEAT_DIR = REPO_ROOT / "data" / "feat" / "v3"
MODEL_DIR = REPO_ROOT / "data" / "models"
CAND_DIR = REPO_ROOT / "data" / "cand"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "v3_test"

OJ_BLOCKS = dict(oj.BLOCKS)
OJ_FEATURES = [f"{k}_{b}" for b in OJ_BLOCKS for k in ("s", "r")] + ["n_blocks"]
# pool-size independent versions: IDF / log(pool size); name counts dropped
IDF_COLS = ["q_only_idf", "p_only_idf"]
REC_FEATURES = ["q_tok_minidf", "p_tok_maxidf", "p_tok_minidf"]
PAIR_FEATURES = v1.PAIR_FEATURES + OJ_FEATURES + REC_FEATURES
REL_FEATURES = v2.REL_FEATURES
EXTRA_FEATURES = ["rev_rank", "rev_gap", "rev_n_close", "orphan_prob"]
# S1 no-match probability (Ojaswi's s1_zero model) and legal-form agreement (name_m has legal forms stripped)
CP2_FEATURES = ["p_zero", "legal_eq", "legal_n"]


# ------------------------------------------------------------------ stage 1

class OjIndex(v2.OjIndex):
    """All of Ojaswi's blocks, fitted with her fit_tfidf (falls back on tiny pools)."""

    def __init__(self, split: str, country: str, tmap: dict):
        p = pl.concat([oj.prepare(split, s, country, tmap) for s in (2, 3)])
        self.blocks = {}
        for name, (col, flt, k, max_df) in OJ_BLOCKS.items():
            pp = p if flt is None else p.filter(pl.col(flt))
            if pp.height == 0:
                continue
            vec, PT = oj.fit_tfidf(pp[col].to_list(), max_df)
            self.blocks[name] = (col, k, vec, PT, pp["entity_id"])
        self.q = oj.prepare(split, 1, country, tmap).select("entity_id", *{c for c, *_ in OJ_BLOCKS.values()})


class Stage1(v2.Stage1):
    def __init__(self, split: str, country: str, tmap: dict):
        t0 = time.time()
        self.tmap = tmap
        self.pool = prepare(pl.concat([load_norm(split, 2, country), load_norm(split, 3, country)]), tmap)
        self.idx = v1.CountryIndex(self.pool)
        self.oj = OjIndex(split, country, tmap)
        self.pool_idx = self.pool.select(pl.col("entity_id").alias("cand_id")).with_row_index("pi")
        self.name_cnt = self.pool.group_by("name_m").agg(pl.len().alias("name_cnt"))
        self.p_stats = v2.record_stats(self.pool, self.idx, self.name_cnt)
        print(f"  {split} {country}: indexes on {self.pool.height:,} records in {time.time() - t0:.0f}s", flush=True)

    def features(self, q: pl.DataFrame) -> pl.DataFrame:
        # v2 computes everything for the blocks in v2.OJ_BLOCKS; point it at all of Ojaswi's blocks
        saved = v2.OJ_BLOCKS, v2.PAIR_FEATURES
        v2.OJ_BLOCKS, v2.PAIR_FEATURES = OJ_BLOCKS, v1.PAIR_FEATURES + OJ_FEATURES + v2.REC_FEATURES
        try:
            f = super().features(q)
        finally:
            v2.OJ_BLOCKS, v2.PAIR_FEATURES = saved
        L = float(self.idx.max_idf)
        return f.with_columns(*[(pl.col(c) / L).cast(pl.Float32) for c in IDF_COLS + REC_FEATURES]) \
            .select("s1_id", "cand_id", *PAIR_FEATURES)


# ------------------------------------------------------------------ record-level extras

def extra_features(c: pl.DataFrame, split: str) -> pl.DataFrame:
    """Join reverse-search and orphan features when their files exist (else nulls)."""
    rev_path, orph_path = CAND_DIR / f"rev_{split}.parquet", CAND_DIR / f"orphan_{split}.parquet"
    if rev_path.exists():
        rev = pl.scan_parquet(rev_path).select(pl.col("rec_id").alias("cand_id"), "s1_id",
                                               pl.col("score").cast(pl.Float32), pl.col("rank").cast(pl.Float32))
        keys = c.select("s1_id", "cand_id").lazy()
        mine = rev.join(keys, on=["s1_id", "cand_id"], how="semi").select(
            "s1_id", "cand_id", pl.col("rank").alias("rev_rank"), pl.col("score").alias("rev_score"))
        per_rec = (rev.join(keys.select("cand_id").unique(), on="cand_id", how="semi").group_by("cand_id")
                   .agg(pl.col("score").max().alias("rev_best"), pl.len().cast(pl.Float32).alias("rev_n")))
        c = (c.join(mine.collect(), on=["s1_id", "cand_id"], how="left")
             .join(per_rec.collect(), on="cand_id", how="left")
             .with_columns((pl.col("rev_best") - pl.col("rev_score")).alias("rev_gap"),
                           ((pl.col("rev_best") - pl.col("rev_score")) <= 0.02).cast(pl.Float32).alias("rev_n_close"))
             .drop("rev_score", "rev_best", "rev_n"))
    if orph_path.exists():
        o = pl.read_parquet(orph_path).select(pl.col("rec_id").alias("cand_id"), pl.col("orphan_prob").cast(pl.Float32))
        c = c.join(o, on="cand_id", how="left")
    return c.with_columns([pl.lit(None, pl.Float32).alias(f) for f in EXTRA_FEATURES if f not in c.columns])


@functools.cache
def _legal(split: str) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(v1.NORM_DIR / f"{split}_s{k}.parquet", columns=["entity_id", "legal_suffix"])
                      for k in (1, 2, 3)]).with_columns(pl.col("legal_suffix").fill_null(""))


@functools.cache
def _zero(split: str) -> pl.DataFrame:
    return pl.read_parquet(CAND_DIR / f"s1_zero_{split}.parquet").select("s1_id", pl.col("p_zero").cast(pl.Float32))


def cp2_features(c: pl.DataFrame, split: str) -> pl.DataFrame:
    """p_zero of the S1; legal_eq = 1 same legal form, 0 different, null unless both sides have one."""
    lg = _legal(split)
    c = (c.join(_zero(split), on="s1_id", how="left")
         .join(lg.rename({"entity_id": "s1_id", "legal_suffix": "lq"}), on="s1_id", how="left")
         .join(lg.rename({"entity_id": "cand_id", "legal_suffix": "lp"}), on="cand_id", how="left"))
    has_q, has_p = pl.col("lq").fill_null("") != "", pl.col("lp").fill_null("") != ""
    return c.with_columns(
        pl.when(has_q & has_p).then((pl.col("lq") == pl.col("lp")).cast(pl.Float32)).alias("legal_eq"),
        (has_q.cast(pl.Float32) + has_p.cast(pl.Float32)).alias("legal_n")).drop("lq", "lp")


def orphan_ids() -> pl.DataFrame:
    """cand_id of every train record that appears in some ground-truth match list."""
    return truth_pairs(load_ground_truth()).select("cand_id").unique()


def weights(c: pl.DataFrame, owned: pl.DataFrame, w: float) -> np.ndarray:
    """Sample weight w for negatives whose record is an orphan (test has ~1.6x more of them)."""
    orphan = (pl.col("label") == 0) & ~pl.col("cand_id").is_in(owned["cand_id"].implode())
    return c.select(pl.when(orphan).then(w).otherwise(1.0)).to_series().to_numpy()


def tune_weighted(c: pl.DataFrame, truth: pl.DataFrame, owned: pl.DataFrame, w: float, exclusive: bool = False):
    """Best (F0.5, t, t1) when every orphan false positive counts w times (test-like precision)."""
    n_true = truth.select("s1_id", pl.col("matched_ids").list.len().alias("n_true"))
    c = c.with_columns(((pl.col("label") == 0) & ~pl.col("cand_id").is_in(owned["cand_id"].implode())).alias("orph"))
    best = (0.0, None, None)
    for t in [x / 100 for x in range(50, 97, 5)]:
        for t1 in [x / 100 for x in range(30, int(t * 100) + 1, 5)]:
            p = decide(c, t, t1, exclusive)
            g = p.group_by("s1_id").agg((pl.col("label") == 1).sum().alias("tp"),
                                        ((pl.col("label") == 0) & ~pl.col("orph")).sum().alias("fo"),
                                        pl.col("orph").sum().alias("fr"))
            d = n_true.join(g, on="s1_id", how="left").fill_null(0).with_columns((pl.col("fo") + w * pl.col("fr")).alias("fp"))
            prec, rec = pl.col("tp") / (pl.col("tp") + pl.col("fp")), pl.col("tp") / pl.col("n_true")
            f = d.select(pl.when(pl.col("n_true") == 0).then((pl.col("tp") + pl.col("fp") == 0).cast(pl.Float64))
                         .when(pl.col("tp") == 0).then(0.0).otherwise(1.25 * prec * rec / (0.25 * prec + rec)).mean()).item()
            if f > best[0]:
                best = (f, t, t1)
    return best


def final_features(use_extra: bool, cp2: bool = False):
    return PAIR_FEATURES + REL_FEATURES + (EXTRA_FEATURES if use_extra else []) + (CP2_FEATURES if cp2 else [])


def decide(c: pl.DataFrame, t: float, t1: float, exclusive: bool) -> pl.DataFrame:
    best = pl.col("prob") == pl.col("prob").max().over("s1_id")
    p = c.filter((pl.col("prob") >= t) | (best & (pl.col("prob") >= t1)))
    if exclusive:
        p = p.filter(pl.col("prob") == pl.col("prob").max().over("cand_id")).unique(["cand_id"], keep="first")
    return p


# ------------------------------------------------------------------ feature pass

def cmd_features(_args):
    tmap = load_script_map()
    s1 = load_norm("train", 1)
    truth = add_is_val(load_ground_truth())
    val_ids = truth.filter("is_val").select(pl.col("s1_id").alias("entity_id"))
    order = (s1.join(val_ids, on="entity_id", how="anti")
             .with_columns(pl.int_range(pl.len()).shuffle(SEED).over("country").alias("o")))
    sets = {
        "fitA": order.filter(pl.col("o") < TRAIN_A_PER_COUNTRY).drop("o"),
        "fitB": order.filter((pl.col("o") >= TRAIN_A_PER_COUNTRY)
                             & (pl.col("o") < TRAIN_A_PER_COUNTRY + TRAIN_B_PER_COUNTRY)).drop("o"),
        "val": s1.join(val_ids, on="entity_id", how="semi"),
    }
    labels = truth_pairs(truth).with_columns(pl.lit(1, pl.Int8).alias("label"))
    for name in sets:
        (FEAT_DIR / name).mkdir(parents=True, exist_ok=True)
    text, stats = [], {name: [0, 0, 0] for name in sets}
    for country in countries_of(s1):
        st = Stage1("train", country, tmap)
        text.append(st.pool.select("entity_id", "name_m", "addr", "house"))
        for name, frame in sets.items():
            q_all = prepare(frame.filter(pl.col("country") == country), tmap)
            for i, start in enumerate(range(0, q_all.height, CHUNK)):
                f = st.features(q_all.slice(start, CHUNK)) \
                    .join(labels, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("label").fill_null(0))
                f.write_parquet(FEAT_DIR / name / f"{country}_{i:03d}.parquet")
                stats[name][0] += f.height
                stats[name][1] += f["s1_id"].n_unique()
                stats[name][2] += int(f["label"].sum())
            print(f"  {country} {name}: {q_all.height:,} S1 done", flush=True)
        del st
        gc.collect()
    pl.concat(text).write_parquet(FEAT_DIR / "text_train.parquet")
    for name, (n, s, pos) in stats.items():
        print(f"{name}: {n:,} pairs for {s:,} S1 ({n / max(s, 1):.1f}/S1), {pos:,} positives", flush=True)


# ------------------------------------------------------------------ training

def _parts(name, countries=None):
    files = sorted((FEAT_DIR / name).glob("*.parquet"))
    if countries:
        files = [f for f in files if f.name.rsplit("_", 1)[0] in countries]
    return files


def _paths(tag):
    sfx = f"_{tag}" if tag else ""
    return (str(MODEL_DIR / f"v3_pruner{sfx}_{{}}.txt"), MODEL_DIR / f"v3_final{sfx}.txt",
            MODEL_DIR / f"v3_params{sfx}.json")


def _prune(df, pruners):
    X = df.select(PAIR_FEATURES).to_numpy()
    return pl.Series("prune_prob", np.mean([m.predict(X) for m in pruners], axis=0), dtype=pl.Float32)


def cmd_prune(args):
    from src.common import tradeoff_table
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pruner_path, _, _ = _paths(args.pruner_tag)
    fit = pl.concat([pl.read_parquet(f) for f in _parts("fitA", args.countries)]) \
        .with_columns((pl.col("s1_id").hash(SEED) % 2).alias("fold"))
    for k in (0, 1):
        tr = fit.filter(pl.col("fold") != k)
        d = lgb.Dataset(tr.select(PAIR_FEATURES).to_numpy(), tr["label"].to_numpy(), feature_name=PAIR_FEATURES)
        m = lgb.train(v1.PRUNER_PARAMS, d, num_boost_round=400)
        m.save_model(pruner_path.format(k))
        print(f"pruner fold {k}: {tr.height:,} pairs", flush=True)
    imp = sorted(zip(PAIR_FEATURES, m.feature_importance("gain")), key=lambda x: -x[1])
    print("pruner top features:", ", ".join(f"{f} {g:.0f}" for f, g in imp[:15]))
    del fit, tr, d
    gc.collect()
    pruners = [lgb.Booster(model_file=pruner_path.format(k)) for k in (0, 1)]
    ranked = []
    for f in _parts("val"):
        v = pl.read_parquet(f, columns=["s1_id", "cand_id", *PAIR_FEATURES])
        ranked.append(v.select("s1_id", "cand_id").with_columns(_prune(v, pruners)))
    ranked = select_candidates(pl.concat(ranked), 10_000, 0.0).with_columns(pl.col("prune_prob").alias("block_score"))
    truth = add_is_val(load_ground_truth()).filter("is_val").drop("is_val")
    print("\n-- candidate set size trade-off on val (ranked by pruner prob) --")
    tradeoff_table(ranked, truth, cutoffs=[("all", None)] + [("topk", k) for k in (10, 8, 6, 5)])
    for k in (8, 6, 5):
        for s in (0.003, 0.01):
            t = tradeoff_table(ranked.filter((pl.col("block_rank") <= k) & (pl.col("prune_prob") >= s)),
                               truth, cutoffs=[("all", None)], verbose=False)
            print(f"  top{k} & prob>={s:<6} avg {t['avg_cands'][0]:.2f}  oracle {t['oracle'][0]:.4f}"
                  f"  pair recall {t['pair_recall'][0]:.4f}")


def cmd_train(args):
    from src.common import blocking_report, macro_f05
    pruner_path, final_path, params_path = _paths(args.tag)
    pruners = [lgb.Booster(model_file=_paths(args.pruner_tag)[0].format(k)) for k in (0, 1)]
    text = pl.read_parquet(FEAT_DIR / "text_train.parquet")
    s1c = load_norm("train", 1).select(pl.col("entity_id").alias("s1_id"), "country")
    feats = final_features(args.extra, args.cp2)

    def pruned(name, oof):
        out = []
        for f in _parts(name, args.countries if name != "val" else None):
            d = pl.read_parquet(f)
            if oof:
                X = d.select(PAIR_FEATURES).to_numpy()
                fold = (d["s1_id"].hash(SEED) % 2).to_numpy()
                p = np.where(fold == 0, pruners[1].predict(X), pruners[0].predict(X))
                d = d.with_columns(pl.Series("prune_prob", p, dtype=pl.Float32))
            else:
                d = d.with_columns(_prune(d, pruners))
            out.append(select_candidates(d, args.final_k, args.p_min))
        return pl.concat(out)

    fit_c = v2.rel_features(pl.concat([pruned("fitA", True), pruned("fitB", False)]), text)
    val_c = v2.rel_features(pruned("val", False), text)
    if args.extra:
        fit_c, val_c = extra_features(fit_c, "train"), extra_features(val_c, "train")
    if args.cp2:
        fit_c, val_c = cp2_features(fit_c, "train"), cp2_features(val_c, "train")
    gc.collect()
    truth = add_is_val(load_ground_truth()).filter("is_val").drop("is_val")
    blocking_report(val_c.with_columns(pl.col("prune_prob").alias("block_score")), truth,
                    name=f"v3 final candidates (top{args.final_k}, prob>={args.p_min}, val)")
    if args.countries:
        src_s1 = s1c.filter(pl.col("country").is_in(args.countries))
        truth_src, val_src = truth.join(src_s1, on="s1_id", how="semi"), val_c.join(src_s1, on="s1_id", how="semi")
    else:
        truth_src, val_src = truth, val_c
    owned = orphan_ids()
    wfit = weights(fit_c, owned, args.orphan_weight)
    wval = weights(val_src, owned, args.orphan_weight)
    models = []
    for seed in range(args.seeds):
        params = dict(v1.FINAL_PARAMS, seed=SEED + seed)
        dtrain = lgb.Dataset(fit_c.select(feats).to_numpy(), fit_c["label"].to_numpy(), weight=wfit, feature_name=feats)
        dval = lgb.Dataset(val_src.select(feats).to_numpy(), val_src["label"].to_numpy(), weight=wval, reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=args.rounds, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(1000)])
        m.save_model(str(final_path).replace(".txt", f"_s{seed}.txt"))
        models.append(m)
        print(f"final model seed {seed}: best iteration {m.best_iteration}", flush=True)
    imp = sorted(zip(feats, models[0].feature_importance("gain")), key=lambda x: -x[1])
    print(f"final model: {fit_c.height:,} train pairs ({fit_c['s1_id'].n_unique():,} S1)")
    print("top features:", ", ".join(f"{f} {g:.0f}" for f, g in imp[:20]))
    X = val_c.select(feats).to_numpy()
    val_c = val_c.with_columns(pl.Series("prob", np.mean([m.predict(X) for m in models], axis=0), dtype=pl.Float32))
    f_src, t, t1 = tune_weighted(val_c.join(val_src.select("s1_id").unique(), on="s1_id", how="semi"), truth_src,
                                 owned, args.tune_weight, not args.no_exclusive)
    print(f"test-like tuning (orphan FP x{args.tune_weight}, exclusive={not args.no_exclusive}): F0.5 {f_src:.4f} at t={t}, t1={t1}")
    pred = decide(val_c, t, t1, not args.no_exclusive)
    ref = macro_f05(pred.select("s1_id", "cand_id"), truth)
    per = v1._by_country(pred, truth, s1c)
    print(f"\nthresholds t={t}, t1={t1} (tuned on {'+'.join(args.countries) if args.countries else 'all'} val)")
    print(f"val macro F0.5 all {ref:.4f} | " + " | ".join(f"{k} {v:.4f}" for k, v in per.items()))
    from src.common import macro_f05_fast
    for tt in (0.8, 0.85, 0.9):
        print(f"  t={tt} t1={t1}: {macro_f05_fast(v1.decide(val_c, tt, t1), truth):.4f}")
    n = truth.height
    print(f"val: {val_c.height / n:.2f} cands/S1, {pred.height / n:.2f} matches/S1")
    json.dump({"t": t, "t1": t1, "final_k": args.final_k, "p_min": args.p_min, "val_f05": ref, "extra": args.extra,
               "cp2": args.cp2, "features": feats, "rounds": args.rounds, "orphan_weight": args.orphan_weight, "tune_weight": args.tune_weight, "val_testlike_f05": f_src,
               "seeds": args.seeds, "pruner_tag": args.pruner_tag, "val_by_country": per,
               "countries": args.countries}, open(params_path, "w"), indent=2)
    val_c.select("s1_id", "cand_id", "label", "prune_prob", "block_rank", "prob") \
        .write_parquet(FEAT_DIR / f"val_scored{'_' + args.tag if args.tag else ''}.parquet")


# ------------------------------------------------------------------ test (+ cache) and rescoring

def cmd_test(args):
    """Full test pass: stage 1, pair features, pruner; caches the pruner's top CACHE_K per S1
    (pair features + prune_prob) and the pool text, then scores with rescore()."""
    pruners = [lgb.Booster(model_file=_paths(args.pruner_tag)[0].format(k)) for k in (0, 1)]
    tmap = load_script_map()
    s1 = load_norm("test", 1)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for country in countries_of(s1):
        st = Stage1("test", country, tmap)
        st.pool.select("entity_id", "name_m", "addr", "house").write_parquet(CACHE_DIR / f"text_{country}.parquet")
        q_all = prepare(s1.filter(pl.col("country") == country), tmap)
        for i, start in enumerate(range(0, q_all.height, CHUNK)):
            f = st.features(q_all.slice(start, CHUNK))
            f = f.with_columns(_prune(f, pruners))
            select_candidates(f, CACHE_K, 0.0).write_parquet(CACHE_DIR / f"{country}_{i:03d}.parquet")
        print(f"  test {country}: {q_all.height:,} S1 done", flush=True)
        del st
        gc.collect()
    cmd_rescore(args)


def cmd_rescore(args):
    """Final candidate set + final model + decisions from the test cache (minutes)."""
    from src.common import write_outputs
    _, final_path, params_path = _paths(args.tag)
    prm = json.load(open(params_path))
    feats = prm.get("features") or final_features(prm["extra"])
    models = [lgb.Booster(model_file=str(final_path).replace(".txt", f"_s{s}.txt")) for s in range(prm["seeds"])]
    final_k = args.final_k or prm["final_k"]
    t = args.t if args.t is not None else prm["t"]
    t1 = args.t1 if args.t1 is not None else prm["t1"]
    s1 = load_norm("test", 1)
    kept = []
    for country in countries_of(s1):
        text = pl.read_parquet(CACHE_DIR / f"text_{country}.parquet")
        for f in sorted(CACHE_DIR.glob(f"{country}_*.parquet")):
            c = v2.rel_features(select_candidates(pl.read_parquet(f), final_k, prm["p_min"]), text)
            if prm["extra"]:
                c = extra_features(c, "test")
            if prm.get("cp2"):
                c = cp2_features(c, "test")
            # the final model scores exactly the candidate set c
            c = c.with_columns(pl.Series("prob", np.mean([m.predict(c.select(feats).to_numpy()) for m in models], axis=0),
                                         dtype=pl.Float32))
            kept.append(c.select("s1_id", "cand_id", pl.lit("stage1_union|pruner").alias("sources"),
                                 pl.col("prune_prob").alias("block_score"), "block_rank", "prob"))
    cand = pl.concat(kept)
    out = REPO_ROOT / (args.out or "output")
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    cand.write_parquet(CAND_DIR / f"v3_test{'_' + args.tag if args.tag else ''}.parquet")
    pred = decide(cand, t, t1, not args.no_exclusive)
    write_outputs(pred.select("s1_id", "cand_id", "prob"),
                  cand.select("s1_id", "cand_id", "sources", "block_score", "block_rank"), s1["entity_id"], out)
    n = s1.height
    print(f"test -> {out}: {n:,} S1, {cand.height / n:.2f} cands/S1, {pred.height / n:.2f} matches/S1, "
          f"{(n - pred['s1_id'].n_unique()) / n:.1%} S1 with no match (k={final_k}, t={t}, t1={t1}, "
          f"exclusive={not args.no_exclusive})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["features", "prune", "train", "test", "rescore"])
    ap.add_argument("--countries", nargs="*")
    ap.add_argument("--tag", default="")
    ap.add_argument("--pruner-tag", default="")
    ap.add_argument("--final-k", type=int, default=None)
    ap.add_argument("--p-min", type=float, default=P_MIN)
    ap.add_argument("--extra", action="store_true", help="use reverse-search / orphan features")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--cp2", action="store_true", help="add p_zero and legal-form features")
    ap.add_argument("--rounds", type=int, default=6000, help="max boosting rounds of the final model")
    ap.add_argument("--orphan-weight", type=float, default=1.0, help="training weight of orphan negatives")
    ap.add_argument("--tune-weight", type=float, default=1.6, help="orphan FP weight when tuning thresholds")
    ap.add_argument("--t", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--no-exclusive", action="store_true")
    ap.add_argument("--out", default=None, help="output folder (default output/)")
    args = ap.parse_args()
    if args.command == "train" and args.final_k is None:
        args.final_k = FINAL_K
    t0 = time.time()
    {"features": cmd_features, "prune": cmd_prune, "train": cmd_train, "test": cmd_test,
     "rescore": cmd_rescore}[args.command](args)
    print(f"[{args.command}] finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
