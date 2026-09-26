"""Matching v1: norm files + name_latin, stage 1 -> pruner -> candidate set -> final model.

Per country (whatever labels the S1 file contains):

  1. prepare: Mhtv's data/norm/{split}_s{1,2,3}.parquet. Match name = name_core (Latin),
     name_latin (Devanagari), or Ojaswi's latinize() for other Indic scripts; Ojaswi's
     clean_expr (leading zeros, digit look-alikes) on names and addresses.
  2. stage 1 (internal, wide): name char-4gram TF-IDF top WIDE_K  U  address word 1-2gram
     TF-IDF top WIDE_K among that country's S2+S3 records (~58 pairs per S1).
  3. pruner: LightGBM on pair features of every stage-1 pair (two fold models, averaged).
     Keep the top FINAL_K per S1 with pruner prob >= P_MIN. This is the final candidate
     set and exactly what goes into candidate_pairs.tsv.
  4. final model: LightGBM on the pair features plus per-S1 relative features computed
     only inside that candidate set (gaps to the best candidate, similarity to the other
     candidates). It scores exactly the candidate set, nothing else.
  5. match: prob >= t, or the best candidate alone if prob >= t1 (tuned on val).

    python -m src.matching.v1 features                  # stage 1 + pair features, train fit sample + val
    python -m src.matching.v1 prune                     # pruner models + candidate-size trade-off on val
    python -m src.matching.v1 train                     # final model + thresholds on val
    python -m src.matching.v1 prune --countries US --tag us   # unseen-country check (then train
    python -m src.matching.v1 train --countries US --tag us   #  reports the other country's val)
    python -m src.matching.v1 test                      # all test S1 -> output/*.tsv
"""
import argparse
import gc
import json
import math
import time

# sparse_dot_topn must be imported before lightgbm: both bundle an OpenMP runtime,
# and loading lightgbm's first makes multithreaded sp_matmul_topn segfault on macOS.
from sparse_dot_topn import sp_matmul_topn  # isort: skip

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer

from src.blocking.text import clean_expr, latinize, load_script_map, skeletonize
from src.common import add_is_val, load_ground_truth, truth_pairs
from src.common.io import REPO_ROOT

WIDE_K = 30          # per retrieval source
FINAL_K = 8          # max candidates per S1 kept by the pruner
P_MIN = 0.0          # min pruner prob for a candidate (set from the val trade-off)
CHUNK = 50_000       # S1 queries per chunk
THREADS = 8
TRAIN_S1_PER_COUNTRY = 80_000
SEED = 0

NORM_DIR = REPO_ROOT / "data" / "norm"
FEAT_DIR = REPO_ROOT / "data" / "feat"
MODEL_DIR = REPO_ROOT / "data" / "models"
CAND_DIR = REPO_ROOT / "data" / "cand"

LEGAL = ["private", "limited", "pvt", "ltd", "llp", "llc", "inc", "incorporated", "corp",
         "corporation", "company", "co", "plc", "sarl", "sas", "sasu", "eurl", "sa", "gmbh",
         "www", "com", "net", "org", "http", "https"]
LEGAL_RE = r"\b(?:" + "|".join(LEGAL) + r")\b"

V0_FEATURES = [
    "name_cos", "addr_cos", "name_rank", "addr_rank", "in_name", "in_addr",
    "name_ts", "name_ratio", "name_partial", "addr_ts", "addr_ratio",
    "num_jacc", "num_q", "num_p", "house_eq", "pin_eq", "p_nonlatin", "p_addr_missing",
    "p_is_s3", "len_q", "len_p", "n_cands",
]
NEW_PAIR_FEATURES = [
    # tolerant house numbers: leading zeros removed, one-digit edits counted as "near"
    "house_tol", "house_lev", "house_full_eq", "house_in_nums", "num_q_only", "num_p_only",
    # name differences: tokens (consonant skeletons) on one side only and how rare they are
    "name_tsort", "skel_ratio", "first_jw", "first_eq", "q_only_n", "p_only_n",
    "q_only_idf", "p_only_idf", "diff_ratio",
    "city_sim", "q_addr_missing", "p_other_script",
]
PAIR_FEATURES = V0_FEATURES + NEW_PAIR_FEATURES
GAP_FEATURES = ["name_ts", "name_ratio", "addr_ts", "num_jacc", "house_tol", "skel_ratio"]
REL_FEATURES = (
    ["prune_prob", "prune_rank", "prune_gap", "n_final", "n_name_hi", "n_house_eq",
     "name_ts_rank", "addr_ts_rank",
     "sib_name_max", "sib_name_w", "sib_addr_max", "sib_addr_w", "sib_support"]
    + [f + "_gap" for f in GAP_FEATURES]
)
FINAL_FEATURES = PAIR_FEATURES + REL_FEATURES


# ------------------------------------------------------------------ prepare

def load_norm(split: str, source: int, country=None) -> pl.DataFrame:
    lf = pl.scan_parquet(NORM_DIR / f"{split}_s{source}.parquet")
    if country is not None:
        lf = lf.filter(pl.col("country") == country)
    return lf.collect()


def _strip_legal(e: pl.Expr) -> pl.Expr:
    return e.str.replace_all(LEGAL_RE, " ").str.replace_all(r"\s+", " ").str.strip_chars()


def prepare(df: pl.DataFrame, tmap: dict) -> pl.DataFrame:
    """Matching text and cheap parsed fields for norm-file records."""
    d = df.select(
        "entity_id",
        pl.when(pl.col("script") == "latin").then(pl.col("name_core"))
        .when(pl.col("script") == "devanagari").then(pl.col("name_latin"))
        .otherwise(pl.col("name_core")).fill_null("").alias("name_m"),
        pl.col("name_norm").fill_null("").alias("name_full"),
        clean_expr(pl.col("addr_norm")).alias("addr"),
        pl.col("house_no"),
        pl.col("city").fill_null(""),
        (pl.col("script") != "latin").alias("nonlatin"),
        (pl.col("script") == "other").alias("other_script"),
        pl.col("entity_id").str.starts_with("S3-").alias("is_s3"),
    )
    # other Indic scripts (and any leftover non-Latin token) -> Latin via Ojaswi's map/romanizer
    d = d.with_columns(name_m=latinize(d["name_m"], tmap), name_full=latinize(d["name_full"], tmap),
                       city=latinize(d["city"], tmap))
    d = d.with_columns(
        _strip_legal(clean_expr(pl.col("name_m"))).alias("name_m"),
        clean_expr(pl.col("name_full")).alias("name_full"),
    ).with_columns(
        pl.when(pl.col("name_m") == "").then(pl.col("name_full")).otherwise(pl.col("name_m")).alias("name_m"),
    )
    d = d.with_columns(skeletonize(d["name_m"]).alias("name_skel"))
    return d.with_columns(
        pl.col("name_m").str.replace_all(" ", "").alias("name_key"),
        pl.col("name_m").str.split(" ").list.first().fill_null("").alias("name_first"),
        # house: first number of the parsed house_no, else the first number in the address
        pl.coalesce(pl.col("house_no").str.extract(r"(\d+)", 1).str.replace(r"^0+(\d)", "$1"),
                    pl.col("addr").str.extract(r"\b(\d+)\b", 1)).fill_null("").alias("house"),
        pl.col("house_no").str.to_lowercase().str.replace_all(r"\b0+(\d)", "$1")
        .str.replace_all(r"[^\p{L}\p{N}]", "").alias("house_full"),
        pl.col("addr").str.extract_all(r"\b\d{5,6}\b").list.last().alias("pin"),
        pl.col("addr").str.extract_all(r"\d+").list.unique().alias("nums"),
        (pl.col("addr") == "").alias("addr_missing"),
    ).select("entity_id", "name_m", "name_key", "name_skel", "name_first", "addr", "house",
             "house_full", "pin", "nums", "city", "nonlatin", "other_script", "addr_missing", "is_s3")


# ------------------------------------------------------------------ stage 1

class CountryIndex:
    """TF-IDF indexes over one country's S2+S3 pool, plus name-token IDF."""

    def __init__(self, pool: pl.DataFrame):
        self.pool = pool
        # max_df drops very common terms ("enterprises", "delhi"): they barely change the
        # ranking but dominate the sparse matmul cost
        self.name_vec = TfidfVectorizer(analyzer="char", ngram_range=(4, 4), min_df=2, max_df=0.02,
                                        dtype=np.float32, sublinear_tf=True)
        self.addr_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+",
                                        min_df=2, max_df=0.005, dtype=np.float32, sublinear_tf=True)
        self.PN = self.name_vec.fit_transform(pool["name_key"].to_list())
        self.PA = self.addr_vec.fit_transform(pool["addr"].to_list())
        self.PNT = self.PN.T.tocsr()
        self.PAT = self.PA.T.tocsr()
        n = pool.height
        self.max_idf = math.log(n + 1)
        self.idf = (
            pool.select(pl.col("name_skel").str.split(" ").list.unique().alias("tok")).explode("tok")
            .group_by("tok").len()
            .select("tok", ((n + 1) / (pl.col("len") + 1)).log().cast(pl.Float32).alias("idf"))
        )

    def pairs(self, q: pl.DataFrame) -> pl.DataFrame:
        """Stage-1 pairs for query S1 rows q with retrieval features."""
        QN = self.name_vec.transform(q["name_key"].to_list())
        QA = self.addr_vec.transform(q["addr"].to_list())
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


# ------------------------------------------------------------------ pair features

def _pw(df, a, b, scorer, scale=100.0):
    return cpdist(df[a].to_list(), df[b].to_list(), scorer=scorer, workers=-1).astype(np.float32) / scale


def _eq(a, b):
    return (pl.when(pl.col(a).is_null() | pl.col(b).is_null()).then(0.0)
            .when(pl.col(a) == pl.col(b)).then(1.0).otherwise(-1.0)).cast(pl.Float32)


def _max_idf(df: pl.DataFrame, col: str, idx: CountryIndex) -> pl.Series:
    """Max IDF over the tokens in list column `col` (0 for an empty list)."""
    ex = (df.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias("row"), pl.col(col).alias("tok"))
          .explode("tok").drop_nulls("tok")
          .join(idx.idf, on="tok", how="left")
          .group_by("row").agg(pl.col("idf").fill_null(idx.max_idf).max()))
    return (pl.DataFrame({"row": pl.int_range(df.height, dtype=pl.UInt32, eager=True)})
            .join(ex, on="row", how="left")["idf"].fill_null(0.0).cast(pl.Float32))


def pair_features(pr: pl.DataFrame, q: pl.DataFrame, p: pl.DataFrame, idx: CountryIndex) -> pl.DataFrame:
    qa = q.with_row_index("qi").rename(lambda n: n if n == "qi" else "q_" + n)
    pa = p.with_row_index("pi").rename(lambda n: n if n == "pi" else "p_" + n)
    df = pr.join(qa, on="qi").join(pa, on="pi")

    q_tok, p_tok = pl.col("q_name_skel").str.split(" "), pl.col("p_name_skel").str.split(" ")
    df = df.with_columns(
        q_tok.list.set_difference(p_tok).alias("q_only"),
        p_tok.list.set_difference(q_tok).alias("p_only"),
    ).with_columns(
        pl.col("q_only").list.join(" ").alias("q_only_s"),
        pl.col("p_only").list.join(" ").alias("p_only_s"),
    )
    df = df.with_columns(
        pl.Series("name_ts", _pw(df, "q_name_m", "p_name_m", fuzz.token_set_ratio)),
        pl.Series("name_ratio", _pw(df, "q_name_m", "p_name_m", fuzz.ratio)),
        pl.Series("name_partial", _pw(df, "q_name_key", "p_name_key", fuzz.partial_ratio)),
        pl.Series("addr_ts", _pw(df, "q_addr", "p_addr", fuzz.token_set_ratio)),
        pl.Series("addr_ratio", _pw(df, "q_addr", "p_addr", fuzz.ratio)),
        pl.Series("name_tsort", _pw(df, "q_name_m", "p_name_m", fuzz.token_sort_ratio)),
        pl.Series("skel_ratio", _pw(df, "q_name_skel", "p_name_skel", fuzz.ratio)),
        pl.Series("first_jw", _pw(df, "q_name_first", "p_name_first", JaroWinkler.normalized_similarity, 1.0)),
        pl.Series("diff_ratio", _pw(df, "q_only_s", "p_only_s", fuzz.ratio)),
        pl.Series("house_lev", _pw(df, "q_house", "p_house", Levenshtein.distance, 1.0)),
        pl.Series("city_sim", _pw(df, "q_city", "p_city", fuzz.ratio)),
        _max_idf(df, "q_only", idx).alias("q_only_idf"),
        _max_idf(df, "p_only", idx).alias("p_only_idf"),
    )

    inter = pl.col("q_nums").list.set_intersection(pl.col("p_nums")).list.len()
    union = pl.col("q_nums").list.set_union(pl.col("p_nums")).list.len()
    no_house = (pl.col("q_house") == "") | (pl.col("p_house") == "")
    df = df.with_columns(
        pl.when(union > 0).then(inter / union).otherwise(None).cast(pl.Float32).alias("num_jacc"),
        pl.col("q_nums").list.len().cast(pl.Float32).alias("num_q"),
        pl.col("p_nums").list.len().cast(pl.Float32).alias("num_p"),
        pl.col("q_nums").list.set_difference(pl.col("p_nums")).list.len().cast(pl.Float32).alias("num_q_only"),
        pl.col("p_nums").list.set_difference(pl.col("q_nums")).list.len().cast(pl.Float32).alias("num_p_only"),
        # v0 feature kept as is: exact first-number agreement
        (pl.when(no_house).then(0.0).when(pl.col("q_house") == pl.col("p_house")).then(1.0)
         .otherwise(-1.0)).cast(pl.Float32).alias("house_eq"),
        # 1 equal, 0.5 one digit inserted/deleted/changed (100 vs 10, 229 vs 228), -1 different
        (pl.when(no_house).then(0.0).when(pl.col("house_lev") == 0).then(1.0)
         .when(pl.col("house_lev") == 1).then(0.5).otherwise(-1.0)).cast(pl.Float32).alias("house_tol"),
        pl.when(no_house).then(None).otherwise(pl.col("house_lev")).alias("house_lev"),
        _eq("q_house_full", "p_house_full").alias("house_full_eq"),
        ((pl.col("p_nums").list.contains(pl.col("q_house")) | pl.col("q_nums").list.contains(pl.col("p_house")))
         & ~no_house).cast(pl.Float32).alias("house_in_nums"),
        _eq("q_pin", "p_pin").alias("pin_eq"),
        (pl.col("q_name_first") == pl.col("p_name_first")).cast(pl.Float32).alias("first_eq"),
        pl.col("q_only").list.len().cast(pl.Float32).alias("q_only_n"),
        pl.col("p_only").list.len().cast(pl.Float32).alias("p_only_n"),
        pl.when((pl.col("q_only_s") == "") & (pl.col("p_only_s") == "")).then(1.0)
        .when((pl.col("q_only_s") == "") | (pl.col("p_only_s") == "")).then(0.0)
        .otherwise(pl.col("diff_ratio")).cast(pl.Float32).alias("diff_ratio"),
        pl.when((pl.col("q_city") == "") | (pl.col("p_city") == "")).then(None)
        .otherwise(pl.col("city_sim")).alias("city_sim"),
        pl.col("p_nonlatin").cast(pl.Float32),
        pl.col("p_other_script").cast(pl.Float32),
        pl.col("p_addr_missing").cast(pl.Float32),
        pl.col("q_addr_missing").cast(pl.Float32),
        pl.col("p_is_s3").cast(pl.Float32),
        pl.col("q_name_key").str.len_chars().cast(pl.Float32).alias("len_q"),
        pl.col("p_name_key").str.len_chars().cast(pl.Float32).alias("len_p"),
        pl.len().over("qi").cast(pl.Float32).alias("n_cands"),
    )
    return df.select(pl.col("q_entity_id").alias("s1_id"), pl.col("p_entity_id").alias("cand_id"),
                     *[pl.col(f).cast(pl.Float32) for f in PAIR_FEATURES])


# ------------------------------------------------------------------ pruner -> candidate set

def prune_prob(df: pl.DataFrame, pruners) -> pl.Series:
    X = df.select(PAIR_FEATURES).to_numpy()
    return pl.Series("prune_prob", np.mean([m.predict(X) for m in pruners], axis=0), dtype=pl.Float32)


def select_candidates(df: pl.DataFrame, final_k: int, p_min: float) -> pl.DataFrame:
    """Final candidate set: top final_k per S1 by pruner prob, prob >= p_min."""
    return (df.with_columns(pl.col("prune_prob").rank("ordinal", descending=True).over("s1_id")
                            .cast(pl.Int32).alias("block_rank"))
            .filter((pl.col("block_rank") <= final_k) & (pl.col("prune_prob") >= p_min)))


def rel_features(c: pl.DataFrame, text: pl.DataFrame) -> pl.DataFrame:
    """Per-S1 relative features, computed only inside the final candidate set c.

    text: entity_id, name_m, addr for (at least) every cand_id in c.
    """
    c = c.with_columns(
        pl.col("block_rank").cast(pl.Float32).alias("prune_rank"),
        (pl.col("prune_prob").max().over("s1_id") - pl.col("prune_prob")).alias("prune_gap"),
        pl.len().over("s1_id").cast(pl.Float32).alias("n_final"),
        (pl.col("name_ts") >= 0.9).sum().over("s1_id").cast(pl.Float32).alias("n_name_hi"),
        (pl.col("house_tol") == 1).sum().over("s1_id").cast(pl.Float32).alias("n_house_eq"),
        pl.col("name_ts").rank("min", descending=True).over("s1_id").cast(pl.Float32).alias("name_ts_rank"),
        pl.col("addr_ts").rank("min", descending=True).over("s1_id").cast(pl.Float32).alias("addr_ts_rank"),
        *[(pl.col(f).max().over("s1_id") - pl.col(f)).alias(f + "_gap") for f in GAP_FEATURES],
    )
    # similarity of each candidate to the other candidates of the same S1: true matches of one
    # S1 are the same business, so they tend to agree with each other
    t = c.select("s1_id", "cand_id", "prune_prob").join(
        text.select(pl.col("entity_id").alias("cand_id"), "name_m", "addr"), on="cand_id", how="left")
    sib = t.join(t, on="s1_id", suffix="_o").filter(pl.col("cand_id") != pl.col("cand_id_o"))
    sib = sib.with_columns(
        pl.Series("sn", _pw(sib, "name_m", "name_m_o", fuzz.token_set_ratio)),
        pl.Series("sa", _pw(sib, "addr", "addr_o", fuzz.token_set_ratio)),
    )
    w = pl.col("prune_prob_o")
    agg = sib.group_by("s1_id", "cand_id").agg(
        pl.col("sn").max().alias("sib_name_max"),
        ((pl.col("sn") * w).sum() / (w.sum() + 1e-6)).alias("sib_name_w"),
        pl.col("sa").max().alias("sib_addr_max"),
        ((pl.col("sa") * w).sum() / (w.sum() + 1e-6)).alias("sib_addr_w"),
        (pl.col("sn") * pl.col("sa") * w).max().alias("sib_support"),
    )
    return c.join(agg, on=["s1_id", "cand_id"], how="left").with_columns(
        *[pl.col(f).cast(pl.Float32) for f in REL_FEATURES])


# ------------------------------------------------------------------ feature pass

def country_features(split, country, query_sets, tmap, on_chunk, keep_text=None):
    """Index one country's pool once; for each named S1 frame call on_chunk(name, q, pool, idx,
    pair_features_df) per chunk of CHUNK S1. keep_text: list to append prepared text to."""
    t0 = time.time()
    pool = prepare(pl.concat([load_norm(split, 2, country), load_norm(split, 3, country)]), tmap)
    idx = CountryIndex(pool)
    print(f"  {split} {country}: index on {pool.height:,} records in {time.time() - t0:.0f}s", flush=True)
    if keep_text is not None:
        keep_text.append(pool.select("entity_id", "name_m", "addr"))
    for name, s1 in query_sets.items():
        q_all = prepare(s1.filter(pl.col("country") == country), tmap)
        for start in range(0, q_all.height, CHUNK):
            q = q_all.slice(start, CHUNK)
            on_chunk(name, q, pool, idx, pair_features(idx.pairs(q), q, pool, idx))
        print(f"  {split} {country} {name}: {q_all.height:,} S1 done at {time.time() - t0:.0f}s", flush=True)
    del idx, pool
    gc.collect()


def countries_of(s1):
    return sorted(s1["country"].drop_nulls().unique().to_list())


def cmd_features(_args):
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    tmap = load_script_map()
    s1 = load_norm("train", 1)
    truth = add_is_val(load_ground_truth())
    val_ids = truth.filter("is_val").select(pl.col("s1_id").alias("entity_id"))
    fit_ids = (s1.join(val_ids, on="entity_id", how="anti")
               .filter(pl.int_range(pl.len()).shuffle(SEED).over("country") < TRAIN_S1_PER_COUNTRY)
               .select("entity_id"))
    labels = truth_pairs(truth).with_columns(pl.lit(1, pl.Int8).alias("label"))
    sets = {"fit": s1.join(fit_ids, on="entity_id", how="semi"),
            "val": s1.join(val_ids, on="entity_id", how="semi")}
    chunks = {name: [] for name in sets}
    text = []
    for country in countries_of(s1):
        country_features("train", country, sets, tmap,
                         lambda name, q, pool, idx, f: chunks[name].append(f), keep_text=text)
    pl.concat(text).write_parquet(FEAT_DIR / "v1_text_train.parquet")
    for name, parts in chunks.items():
        feats = pl.concat(parts).join(labels, on=["s1_id", "cand_id"], how="left") \
            .with_columns(pl.col("label").fill_null(0))
        feats.write_parquet(FEAT_DIR / f"v1_{name}.parquet")
        print(f"{name}: {feats.height:,} pairs for {feats['s1_id'].n_unique():,} S1, "
              f"{feats['label'].sum():,} positives", flush=True)


# ------------------------------------------------------------------ training

PRUNER_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=100,
                     feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, seed=SEED,
                     num_threads=THREADS, verbose=-1)
FINAL_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=50,
                    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                    seed=SEED, num_threads=THREADS, verbose=-1)


def _s1_country():
    return load_norm("train", 1).select(pl.col("entity_id").alias("s1_id"), "country")


def _load_split(name, countries):
    df = pl.read_parquet(FEAT_DIR / f"v1_{name}.parquet")
    if countries:
        df = df.join(_s1_country().filter(pl.col("country").is_in(countries)), on="s1_id", how="semi")
    return df


def _fold():
    return (pl.col("s1_id").hash(SEED) % 2).alias("fold")


def _val_truth():
    return add_is_val(load_ground_truth()).filter("is_val").drop("is_val")


def _paths(tag):
    sfx = f"_{tag}" if tag else ""
    return str(MODEL_DIR / f"v1_pruner{sfx}_{{}}.txt"), MODEL_DIR / f"v1_final{sfx}.txt", MODEL_DIR / f"v1_params{sfx}.json"


def cmd_prune(args):
    """Two pruner models (one per S1 fold of the fit sample) + candidate-size trade-off on val."""
    from src.common import tradeoff_table
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pruner_path, _, _ = _paths(args.tag)
    fit = _load_split("fit", args.countries).with_columns(_fold())
    for k in (0, 1):
        tr = fit.filter(pl.col("fold") != k)
        d = lgb.Dataset(tr.select(PAIR_FEATURES).to_numpy(), tr["label"].to_numpy(), feature_name=PAIR_FEATURES)
        m = lgb.train(PRUNER_PARAMS, d, num_boost_round=400)
        m.save_model(str(pruner_path.format(k)))
        print(f"pruner fold {k}: {tr.height:,} pairs", flush=True)
    imp = sorted(zip(PAIR_FEATURES, m.feature_importance("gain")), key=lambda x: -x[1])
    print("pruner top features:", ", ".join(f"{f} {g:.0f}" for f, g in imp[:12]))
    pruners = [lgb.Booster(model_file=str(pruner_path.format(k))) for k in (0, 1)]
    val = pl.read_parquet(FEAT_DIR / "v1_val.parquet", columns=["s1_id", "cand_id", *PAIR_FEATURES])
    val = val.with_columns(prune_prob(val, pruners))
    ranked = select_candidates(val, 10_000, 0.0).with_columns(pl.col("prune_prob").alias("block_score"))
    truth = _val_truth()
    print("\n-- candidate set size trade-off on val (ranked by pruner prob) --")
    cut = [("all", None)] + [("topk", k) for k in (10, 8, 6, 5, 4, 3)] + \
          [("min_score", s) for s in (0.001, 0.003, 0.01, 0.02, 0.05)]
    tradeoff_table(ranked, truth, cutoffs=cut)
    for k in (8, 6, 5):
        print(f"  top{k} + prob>=s:")
        for s in (0.001, 0.003, 0.01, 0.02):
            tradeoff_table(ranked.filter((pl.col("block_rank") <= k) & (pl.col("prune_prob") >= s)),
                           truth, cutoffs=[("all", None)], verbose=False) \
                .pipe(lambda t: print(f"    s={s:<6} avg {t['avg_cands'][0]:.2f}  oracle {t['oracle'][0]:.4f}"
                                      f"  pair recall {t['pair_recall'][0]:.4f}"))


def decide(cand: pl.DataFrame, t: float, t1: float) -> pl.DataFrame:
    """prob >= t; if none pass, the best candidate alone when prob >= t1."""
    best = pl.col("prob") == pl.col("prob").max().over("s1_id")
    return cand.filter((pl.col("prob") >= t) | (best & (pl.col("prob") >= t1)))


def _tune(cand, truth):
    from src.common import macro_f05_fast
    best = (0.0, None, None)
    for t in np.arange(0.20, 0.86, 0.05):
        for t1 in np.arange(0.05, t + 1e-9, 0.05):
            f = macro_f05_fast(decide(cand, t, t1), truth)
            if f > best[0]:
                best = (f, round(float(t), 2), round(float(t1), 2))
    return best


def _by_country(pred, truth, s1c):
    from src.common import macro_f05_fast
    out = {}
    for ctry in sorted(s1c["country"].unique().to_list()):
        tc = truth.join(s1c.filter(pl.col("country") == ctry), on="s1_id", how="semi")
        if tc.height:
            out[ctry] = macro_f05_fast(pred, tc)
    return out


def cmd_train(args):
    """Final model on the pruned fit candidates (out-of-fold pruner), thresholds on val."""
    from src.common import blocking_report, macro_f05, macro_f05_fast
    pruner_path, final_path, params_path = _paths(args.tag)
    pruners = [lgb.Booster(model_file=str(pruner_path.format(k))) for k in (0, 1)]
    text = pl.read_parquet(FEAT_DIR / "v1_text_train.parquet")
    s1c = _s1_country()

    fit = _load_split("fit", args.countries).with_columns(_fold())
    X = fit.select(PAIR_FEATURES).to_numpy()
    oof = np.where(fit["fold"].to_numpy() == 0, pruners[1].predict(X), pruners[0].predict(X))
    fit = fit.with_columns(pl.Series("prune_prob", oof, dtype=pl.Float32)).drop("fold")
    del X
    fit_c = rel_features(select_candidates(fit, args.final_k, args.p_min), text)
    del fit
    val = pl.read_parquet(FEAT_DIR / "v1_val.parquet")
    val = val.with_columns(prune_prob(val, pruners))
    val_c = rel_features(select_candidates(val, args.final_k, args.p_min), text)
    del val
    gc.collect()
    truth = _val_truth()
    blocking_report(val_c.with_columns(pl.col("prune_prob").alias("block_score")), truth,
                    name=f"v1 final candidates (top{args.final_k}, prob>={args.p_min}, val)")

    # early stopping and thresholds use the training countries' val only
    if args.countries:
        src_s1 = s1c.filter(pl.col("country").is_in(args.countries))
        truth_src = truth.join(src_s1, on="s1_id", how="semi")
        val_src = val_c.join(src_s1, on="s1_id", how="semi")
    else:
        truth_src, val_src = truth, val_c
    dtrain = lgb.Dataset(fit_c.select(FINAL_FEATURES).to_numpy(), fit_c["label"].to_numpy(),
                         feature_name=FINAL_FEATURES)
    dval = lgb.Dataset(val_src.select(FINAL_FEATURES).to_numpy(), val_src["label"].to_numpy(), reference=dtrain)
    model = lgb.train(FINAL_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(250)])
    model.save_model(str(final_path))
    imp = sorted(zip(FINAL_FEATURES, model.feature_importance("gain")), key=lambda x: -x[1])
    print(f"final model: {fit_c.height:,} train pairs, best iteration {model.best_iteration}")
    print("top features:", ", ".join(f"{f} {g:.0f}" for f, g in imp[:15]))

    val_c = val_c.with_columns(pl.Series("prob", model.predict(val_c.select(FINAL_FEATURES).to_numpy()),
                                         dtype=pl.Float32))
    f_src, t, t1 = _tune(val_c.join(val_src.select("s1_id").unique(), on="s1_id", how="semi"), truth_src)
    pred = decide(val_c, t, t1)
    ref = macro_f05(pred.select("s1_id", "cand_id"), truth)
    per = _by_country(pred, truth, s1c)
    print(f"\nthresholds t={t}, t1={t1} (tuned on {'+'.join(args.countries) if args.countries else 'all'} val, "
          f"F0.5 there {f_src:.4f})")
    print(f"val macro F0.5 all {ref:.4f} | " + " | ".join(f"{k} {v:.4f}" for k, v in per.items()))
    n = truth.height
    print(f"val: {val_c.height / n:.2f} cands/S1, {pred.height / n:.2f} matches/S1, "
          f"{(n - pred['s1_id'].n_unique()) / n:.1%} S1 predicted empty "
          f"(true singletons {(truth['matched_ids'].list.len() == 0).mean():.1%})")
    for tt in (0.4, 0.5, 0.6):
        print(f"  t={tt} t1=t: {macro_f05_fast(decide(val_c, tt, tt), truth):.4f}")
    json.dump({"t": t, "t1": t1, "final_k": args.final_k, "p_min": args.p_min, "val_f05": ref,
               "val_by_country": per, "countries": args.countries, "best_iteration": model.best_iteration},
              open(params_path, "w"), indent=2)
    if args.dump_val:
        val_c.select("s1_id", "cand_id", "label", "prune_prob", "block_rank", "prob") \
            .write_parquet(FEAT_DIR / f"v1_val_scored{'_' + args.tag if args.tag else ''}.parquet")


# ------------------------------------------------------------------ test

def cmd_test(args):
    from src.common import write_outputs
    pruner_path, final_path, params_path = _paths(args.tag)
    pruners = [lgb.Booster(model_file=str(pruner_path.format(k))) for k in (0, 1)]
    model = lgb.Booster(model_file=str(final_path))
    prm = json.load(open(params_path))
    tmap = load_script_map()
    s1 = load_norm("test", 1)
    kept = []

    def on_chunk(_, q, pool, idx, f):
        f = f.with_columns(prune_prob(f, pruners))
        c = rel_features(select_candidates(f, prm["final_k"], prm["p_min"]),
                         pool.select("entity_id", "name_m", "addr"))
        c = c.with_columns(pl.Series("prob", model.predict(c.select(FINAL_FEATURES).to_numpy()), dtype=pl.Float32))
        src = (pl.when((pl.col("in_name") == 1) & (pl.col("in_addr") == 1)).then(pl.lit("name_tfidf|addr_tfidf"))
               .when(pl.col("in_name") == 1).then(pl.lit("name_tfidf")).otherwise(pl.lit("addr_tfidf")))
        kept.append(c.select("s1_id", "cand_id", src.alias("sources"), pl.col("prune_prob").alias("block_score"),
                             "block_rank", "prob"))

    for country in countries_of(s1):
        country_features("test", country, {"test": s1}, tmap, on_chunk)
    cand = pl.concat(kept)
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    cand.write_parquet(CAND_DIR / "v1_arihant_test.parquet")
    # the final model scored exactly `cand`; matches are a subset of it
    pred = decide(cand, prm["t"], prm["t1"])
    write_outputs(pred.select("s1_id", "cand_id", "prob"),
                  cand.select("s1_id", "cand_id", "sources", "block_score", "block_rank"),
                  s1["entity_id"], REPO_ROOT / "output")
    n = s1.height
    per = cand.join(s1.select(pl.col("entity_id").alias("s1_id"), "country"), on="s1_id")
    pm = pred.join(s1.select(pl.col("entity_id").alias("s1_id"), "country"), on="s1_id")
    print(f"test: {n:,} S1, {cand.height / n:.2f} cands/S1 (max {cand.group_by('s1_id').len()['len'].max()}), "
          f"{pred.height / n:.2f} matches/S1, {(n - pred['s1_id'].n_unique()) / n:.1%} S1 with no match "
          f"(t={prm['t']}, t1={prm['t1']})")
    for ctry, g in s1.group_by("country"):
        m = g.height
        print(f"  {ctry[0]}: {m:,} S1, {per.filter(pl.col('country') == ctry[0]).height / m:.2f} cands/S1, "
              f"{pm.filter(pl.col('country') == ctry[0]).height / m:.2f} matches/S1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["features", "prune", "train", "test"])
    ap.add_argument("--countries", nargs="*", help="train only on these countries (unseen-country check)")
    ap.add_argument("--tag", default="", help="suffix for model files, e.g. us")
    ap.add_argument("--final-k", type=int, default=FINAL_K)
    ap.add_argument("--p-min", type=float, default=P_MIN)
    ap.add_argument("--dump-val", action="store_true", help="save scored val candidates (error analysis)")
    args = ap.parse_args()
    t0 = time.time()
    {"features": cmd_features, "prune": cmd_prune, "train": cmd_train, "test": cmd_test}[args.command](args)
    print(f"[{args.command}] finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
