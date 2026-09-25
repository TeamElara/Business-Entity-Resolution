"""Baseline v0: crude cleaning + name/address TF-IDF blocking + small LightGBM.

Temporary end-to-end pipeline for the first uploads, until Mhtv's norm files and
Ojaswi's candidate files are ready. Per country (whatever labels the data has):

  1. clean names/addresses (src.normalize.normalize_df + crude legal-suffix strip)
  2. stage 1 (internal): name char-4gram TF-IDF top WIDE_K  U  address word 1-2gram
     TF-IDF top WIDE_K, among that country's S2+S3 records (~58 per S1)
  3. score every stage-1 pair with a LightGBM model on cheap, country-agnostic
     features; keep the top FINAL_K per S1 -> candidate set (block_score = prob)
  4. match: prob >= t, or the best candidate alone if prob >= t1 (tuned on val)

    python -m src.matching.baseline_v0 features   # train sample + val features (data/feat/)
    python -m src.matching.baseline_v0 train      # fit model, tune on val (data/models/)
    python -m src.matching.baseline_v0 test       # all test S1 -> output/*.tsv
"""
import argparse
import gc
import json
import time

# sparse_dot_topn must be imported before lightgbm: both bundle an OpenMP runtime,
# and loading lightgbm's first makes multithreaded sp_matmul_topn segfault on macOS.
from sparse_dot_topn import sp_matmul_topn  # isort: skip

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer

from src.common import add_is_val, load_ground_truth, load_source, truth_pairs
from src.common.io import REPO_ROOT
from src.normalize import normalize_df

WIDE_K = 30          # per retrieval source
FINAL_K = 8          # candidates kept per S1 (top-8 oracle 0.9792 vs top-10 0.9795 on val)
CHUNK = 50_000       # S1 queries per chunk
THREADS = 8
TRAIN_S1_PER_COUNTRY = 40_000
SEED = 0

FEAT_DIR = REPO_ROOT / "data" / "feat"
MODEL_DIR = REPO_ROOT / "data" / "models"
CAND_DIR = REPO_ROOT / "data" / "cand"

LEGAL = ["private", "limited", "pvt", "ltd", "llp", "llc", "inc", "incorporated", "corp",
         "corporation", "company", "co", "plc", "sarl", "sas", "sasu", "eurl", "sa", "gmbh",
         "www", "com", "net", "org", "http", "https"]
LEGAL_RE = r"\b(?:" + "|".join(LEGAL) + r")\b"

FEATURES = [
    "name_cos", "addr_cos", "name_rank", "addr_rank", "in_name", "in_addr",
    "name_ts", "name_ratio", "name_partial", "addr_ts", "addr_ratio",
    "num_jacc", "num_q", "num_p", "house_eq", "pin_eq", "p_nonlatin", "p_addr_missing",
    "p_is_s3", "len_q", "len_p", "n_cands",
]


# ------------------------------------------------------------------ cleaning

def prepare(df: pl.DataFrame) -> pl.DataFrame:
    """Crude v0 cleaning on top of normalize_df."""
    return (
        normalize_df(df)
        .with_columns(
            pl.col("name_norm").str.replace_all(LEGAL_RE, " ").str.replace_all(r"\s+", " ")
            .str.strip_chars().alias("name_core"),
        )
        .with_columns(
            pl.when(pl.col("name_core") == "").then(pl.col("name_norm")).otherwise(pl.col("name_core"))
            .alias("name_core"),
        )
        .with_columns(
            pl.col("name_core").str.replace_all(" ", "").alias("name_key"),
            pl.col("addr_norm").str.extract(r"\b(\d+)\b", 1).alias("house_no"),
            pl.col("addr_norm").str.extract_all(r"\b\d{5,6}\b").list.last().alias("pin"),
            pl.col("addr_norm").str.extract_all(r"\d+").list.unique().alias("nums"),
            (~pl.col("name_norm").str.contains(r"[a-z]")).alias("nonlatin"),
            (pl.col("addr_norm") == "").alias("addr_missing"),
            pl.col("entity_id").str.starts_with("S3-").alias("is_s3"),
        )
        .select("entity_id", "name_core", "name_key", "addr_norm", "house_no", "pin", "nums",
                "nonlatin", "addr_missing", "is_s3")
    )


# ------------------------------------------------------------------ stage 1

class CountryIndex:
    """TF-IDF indexes over one country's S2+S3 pool."""

    def __init__(self, pool: pl.DataFrame):
        self.pool = pool
        # max_df drops very common terms ("enterprises", "delhi"): they barely change the
        # ranking but dominate the sparse matmul cost (measured: same recall, 2-15x faster)
        self.name_vec = TfidfVectorizer(analyzer="char", ngram_range=(4, 4), min_df=2, max_df=0.02,
                                        dtype=np.float32, sublinear_tf=True)
        self.addr_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+",
                                        min_df=2, max_df=0.005, dtype=np.float32, sublinear_tf=True)
        self.PN = self.name_vec.fit_transform(pool["name_key"].to_list())
        self.PA = self.addr_vec.fit_transform(pool["addr_norm"].to_list())
        self.PNT = self.PN.T.tocsr()
        self.PAT = self.PA.T.tocsr()

    def pairs(self, q: pl.DataFrame) -> pl.DataFrame:
        """Stage-1 pairs for query S1 rows q with retrieval features."""
        QN = self.name_vec.transform(q["name_key"].to_list())
        QA = self.addr_vec.transform(q["addr_norm"].to_list())
        parts = []
        for tag, Q, PT in (("name", QN, self.PNT), ("addr", QA, self.PAT)):
            C = sp_matmul_topn(Q, PT, top_n=WIDE_K, threshold=0.01, sort=True, n_threads=THREADS).tocoo()
            parts.append(
                pl.DataFrame({"qi": C.row.astype(np.uint32), "pi": C.col.astype(np.uint32),
                              "s": C.data.astype(np.float32)})
                .with_columns(pl.col("s").rank("ordinal", descending=True).over("qi")
                              .cast(pl.Float32).alias(f"{tag}_rank"))
                .drop("s")
                .with_columns(pl.lit(1, pl.Int8).alias(f"in_{tag}"))
            )
        pr = parts[0].join(parts[1], on=["qi", "pi"], how="full", coalesce=True) \
            .with_columns(pl.col("in_name", "in_addr").fill_null(0))
        qi, pi = pr["qi"].to_numpy(), pr["pi"].to_numpy()
        return pr.with_columns(
            pl.Series("name_cos", _rowdot(QN, self.PN, qi, pi)),
            pl.Series("addr_cos", _rowdot(QA, self.PA, qi, pi)),
        )


def _rowdot(A, B, ai, bi, step=1_000_000):
    """Cosine of row A[ai[j]] with row B[bi[j]] for every j (rows are L2-normalised)."""
    out = np.empty(len(ai), dtype=np.float32)
    for s in range(0, len(ai), step):
        out[s:s + step] = np.asarray(A[ai[s:s + step]].multiply(B[bi[s:s + step]]).sum(axis=1)).ravel()
    return out


# ------------------------------------------------------------------ features

def features(pr: pl.DataFrame, q: pl.DataFrame, p: pl.DataFrame) -> pl.DataFrame:
    qa = q.with_row_index("qi").rename(lambda n: n if n == "qi" else "q_" + n)
    pa = p.with_row_index("pi").rename(lambda n: n if n == "pi" else "p_" + n)
    df = pr.join(qa, on="qi").join(pa, on="pi")

    def pairwise(a, b, scorer):
        return cpdist(df[a].to_list(), df[b].to_list(), scorer=scorer, workers=-1).astype(np.float32) / 100

    inter = pl.col("q_nums").list.set_intersection(pl.col("p_nums")).list.len()
    union = pl.col("q_nums").list.set_union(pl.col("p_nums")).list.len()
    df = df.with_columns(
        pl.Series("name_ts", pairwise("q_name_core", "p_name_core", fuzz.token_set_ratio)),
        pl.Series("name_ratio", pairwise("q_name_core", "p_name_core", fuzz.ratio)),
        pl.Series("name_partial", pairwise("q_name_key", "p_name_key", fuzz.partial_ratio)),
        pl.Series("addr_ts", pairwise("q_addr_norm", "p_addr_norm", fuzz.token_set_ratio)),
        pl.Series("addr_ratio", pairwise("q_addr_norm", "p_addr_norm", fuzz.ratio)),
    ).with_columns(
        pl.when(union > 0).then(inter / union).otherwise(None).cast(pl.Float32).alias("num_jacc"),
        pl.col("q_nums").list.len().cast(pl.Float32).alias("num_q"),
        pl.col("p_nums").list.len().cast(pl.Float32).alias("num_p"),
        _eq("q_house_no", "p_house_no").alias("house_eq"),
        _eq("q_pin", "p_pin").alias("pin_eq"),
        pl.col("p_nonlatin").cast(pl.Float32),
        pl.col("p_addr_missing").cast(pl.Float32),
        pl.col("p_is_s3").cast(pl.Float32),
        pl.col("q_name_key").str.len_chars().cast(pl.Float32).alias("len_q"),
        pl.col("p_name_key").str.len_chars().cast(pl.Float32).alias("len_p"),
        pl.len().over("qi").cast(pl.Float32).alias("n_cands"),
    )
    return df.select(pl.col("q_entity_id").alias("s1_id"), pl.col("p_entity_id").alias("cand_id"),
                     *[pl.col(f).cast(pl.Float32) for f in FEATURES])


def _eq(a, b):
    return (pl.when(pl.col(a).is_null() | pl.col(b).is_null()).then(0.0)
            .when(pl.col(a) == pl.col(b)).then(1.0).otherwise(-1.0)).cast(pl.Float32)


def country_features(split, country, query_sets, on_chunk):
    """Build the index for one country once, then for each named S1 frame in
    query_sets call on_chunk(name, feature_df) per chunk of CHUNK S1."""
    t0 = time.time()
    pool = prepare(pl.concat([load_source(split, 2, country=country), load_source(split, 3, country=country)]))
    idx = CountryIndex(pool)
    print(f"  {split} {country}: index on {pool.height:,} records in {time.time() - t0:.0f}s", flush=True)
    for name, s1 in query_sets.items():
        q_all = prepare(s1.filter(pl.col("country") == country))
        for start in range(0, q_all.height, CHUNK):
            q = q_all.slice(start, CHUNK)
            on_chunk(name, features(idx.pairs(q), q, pool))
        print(f"  {split} {country} {name}: {q_all.height:,} S1 done at {time.time() - t0:.0f}s", flush=True)
    del idx, pool
    gc.collect()


def countries_of(s1):
    return sorted(s1["country"].drop_nulls().unique().to_list())


# ------------------------------------------------------------------ commands

def cmd_features():
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    s1 = load_source("train", 1)
    truth = add_is_val(load_ground_truth())
    val_ids = truth.filter("is_val").select(pl.col("s1_id").alias("entity_id"))
    fit_ids = (s1.join(val_ids, on="entity_id", how="anti")
               .filter(pl.int_range(pl.len()).shuffle(SEED).over("country") < TRAIN_S1_PER_COUNTRY)
               .select("entity_id"))
    labels = truth_pairs(truth).with_columns(pl.lit(1, pl.Int8).alias("label"))
    sets = {"fit": s1.join(fit_ids, on="entity_id", how="semi"),
            "val": s1.join(val_ids, on="entity_id", how="semi")}
    chunks = {name: [] for name in sets}
    for country in countries_of(s1):
        country_features("train", country, sets, lambda name, f: chunks[name].append(f))
    for name, parts in chunks.items():
        feats = pl.concat(parts).join(labels, on=["s1_id", "cand_id"], how="left") \
            .with_columns(pl.col("label").fill_null(0))
        feats.write_parquet(FEAT_DIR / f"v0_{name}.parquet")
        print(f"{name}: {feats.height:,} pairs for {feats['s1_id'].n_unique():,} S1, "
              f"{feats['label'].sum():,} positives", flush=True)


def topk_by_prob(df: pl.DataFrame, k=FINAL_K) -> pl.DataFrame:
    return (df.with_columns(pl.col("prob").rank("ordinal", descending=True).over("s1_id")
                            .cast(pl.Int32).alias("block_rank"))
            .filter(pl.col("block_rank") <= k)
            .with_columns(pl.col("prob").alias("block_score"), pl.lit("name_tfidf|addr_tfidf").alias("sources")))


def decide(cand: pl.DataFrame, t: float, t1: float) -> pl.DataFrame:
    """prob >= t; if none pass, the best candidate alone when prob >= t1."""
    best = pl.col("block_rank") == 1
    return cand.filter((pl.col("prob") >= t) | (best & (pl.col("prob") >= t1)))


def cmd_train():
    from src.common import blocking_report, macro_f05, macro_f05_fast, tradeoff_table
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fit = pl.read_parquet(FEAT_DIR / "v0_fit.parquet")
    val = pl.read_parquet(FEAT_DIR / "v0_val.parquet")
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, seed=SEED,
                  num_threads=THREADS, verbose=-1)
    dtrain = lgb.Dataset(fit.select(FEATURES).to_numpy(), fit["label"].to_numpy(), feature_name=FEATURES)
    dval = lgb.Dataset(val.select(FEATURES).to_numpy(), val["label"].to_numpy(), reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=600, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    model.save_model(str(MODEL_DIR / "v0_lgb.txt"))
    imp = sorted(zip(FEATURES, model.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", ", ".join(f"{f} {g:.0f}" for f, g in imp[:10]))

    truth = add_is_val(load_ground_truth()).filter("is_val").drop("is_val")
    val = val.with_columns(pl.Series("prob", model.predict(val.select(FEATURES).to_numpy()), dtype=pl.Float32))
    print("\n-- stage 1 (all pairs the model scores internally) --")
    blocking_report(val.with_columns(pl.lit(1).alias("block_rank"), pl.col("prob").alias("block_score")),
                    truth, name="v0 stage 1 (val)")
    ranked = topk_by_prob(val, k=10_000)
    print("\n-- candidate list size trade-off (ranked by model prob) --")
    tradeoff_table(ranked, truth, cutoffs=[("topk", k) for k in (30, 20, 15, 10, 8, 5, 3)])
    cand = ranked.filter(pl.col("block_rank") <= FINAL_K)
    blocking_report(cand, truth, name=f"v0 final candidates top{FINAL_K} (val)")

    best = (0.0, None, None)
    for t in np.arange(0.20, 0.81, 0.05):
        for t1 in np.arange(0.05, t + 1e-9, 0.05):
            f = macro_f05_fast(decide(cand, t, t1), truth)
            if f > best[0]:
                best = (f, round(float(t), 2), round(float(t1), 2))
    ref = macro_f05(decide(cand, best[1], best[2]).select("s1_id", "cand_id"), truth)
    print(f"\nbest val macro F0.5 {ref:.4f} at t={best[1]}, t1={best[2]} (reference scorer)")
    for t in (0.3, 0.4, 0.5, 0.6):
        print(f"  t={t} t1=t: {macro_f05_fast(decide(cand, t, t), truth):.4f}")
    json.dump({"t": best[1], "t1": best[2], "val_f05": best[0], "final_k": FINAL_K,
               "best_iteration": model.best_iteration}, open(MODEL_DIR / "v0_params.json", "w"), indent=2)


def cmd_test():
    from src.common import write_outputs
    model = lgb.Booster(model_file=str(MODEL_DIR / "v0_lgb.txt"))
    prm = json.load(open(MODEL_DIR / "v0_params.json"))
    s1 = load_source("test", 1)
    kept = []

    def on_chunk(_, f):
        f = f.with_columns(pl.Series("prob", model.predict(f.select(FEATURES).to_numpy()), dtype=pl.Float32))
        kept.append(topk_by_prob(f).select("s1_id", "cand_id", "sources", "block_score", "block_rank", "prob"))

    for country in countries_of(s1):
        country_features("test", country, {"test": s1}, on_chunk)
    cand = pl.concat(kept)
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    cand.write_parquet(CAND_DIR / "v0_arihant_test.parquet")
    pred = decide(cand, prm["t"], prm["t1"])
    write_outputs(pred.select("s1_id", "cand_id", "prob"),
                  cand.select("s1_id", "cand_id", "sources", "block_score", "block_rank"),
                  s1["entity_id"], REPO_ROOT / "output")
    n = s1.height
    print(f"test: {n:,} S1, {cand.height / n:.2f} cands/S1, {pred.height / n:.2f} matches/S1, "
          f"{(n - pred['s1_id'].n_unique()) / n:.1%} S1 with no match (t={prm['t']}, t1={prm['t1']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["features", "train", "test"])
    cmd = ap.parse_args().command
    t0 = time.time()
    {"features": cmd_features, "train": cmd_train, "test": cmd_test}[cmd]()
    print(f"[{cmd}] finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
