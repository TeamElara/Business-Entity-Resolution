"""Matching v2: v1 + Ojaswi's stage-1 blocks + new features + more training data.

Differences from v1 (src/matching/v1.py, reused here):

  - stage 1 = v1's name char-4gram and address word TF-IDF blocks  U  Ojaswi's three blocks
    (src/blocking/stage1.py: word TF-IDF on "name address", consonant skeletons, name skeleton
    against records without an address), computed in the same pass with her code and settings.
  - new pair features: Ojaswi's block scores/ranks, number of blocks that found the pair, how many
    pool records share the S1's / candidate's name, rarest and commonest name-token IDF.
  - new per-S1 features: does the candidate's house number agree with the other candidates'
    (pruner-weighted, and with the best other candidate); how many other candidates share its name.
  - the pruner trains on fit sample A (80k S1 per country, two folds); the final model trains on
    A (out-of-fold pruner) plus fit sample B (120k more S1 per country, scored by the pruner like
    val/test). Features are written to disk in parts, so memory stays bounded.

    python -m src.matching.v2 features
    python -m src.matching.v2 prune
    python -m src.matching.v2 train --p-min 0.003
    python -m src.matching.v2 prune --countries US --tag us ; python -m src.matching.v2 train --countries US --tag us
    python -m src.matching.v2 test
"""
import argparse
import gc
import json
import time

from sparse_dot_topn import sp_matmul_topn  # isort: skip  (before lightgbm, see v1)

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

from src.blocking.stage1 import BLOCKS as _OJ_ALL
from src.blocking.stage1 import prepare as oj_prepare
from src.blocking.text import load_script_map
from src.common import add_is_val, load_ground_truth, truth_pairs
from src.common.io import REPO_ROOT
from src.matching import v1
from src.matching.v1 import (CHUNK, SEED, THREADS, WIDE_K, CountryIndex, _pw, _rowdot, countries_of,
                             decide, load_norm, prepare, select_candidates)

TRAIN_A_PER_COUNTRY = 80_000      # pruner training (+ final model, out of fold)
TRAIN_B_PER_COUNTRY = 120_000     # final model only
FINAL_K = 8
P_MIN = 0.003

# v2 is trained with Ojaswi's first three blocks (commit 3c58950); later blocks go into v3
OJ_BLOCKS = {k: _OJ_ALL[k] for k in ("word", "skel", "noaddr")}

FEAT_DIR = REPO_ROOT / "data" / "feat" / "v2"
MODEL_DIR = REPO_ROOT / "data" / "models"
CAND_DIR = REPO_ROOT / "data" / "cand"

OJ_FEATURES = [f"{k}_{b}" for b in OJ_BLOCKS for k in ("s", "r")] + ["n_blocks"]
REC_FEATURES = ["q_name_cnt", "p_name_cnt", "q_tok_minidf", "p_tok_maxidf", "p_tok_minidf"]
PAIR_FEATURES = v1.PAIR_FEATURES + OJ_FEATURES + REC_FEATURES
SIB2_FEATURES = ["sib_house_w", "sib_house_best", "sib_same_name_n"]
REL_FEATURES = v1.REL_FEATURES + SIB2_FEATURES
FINAL_FEATURES = PAIR_FEATURES + REL_FEATURES


# ------------------------------------------------------------------ stage 1

class OjIndex:
    """Ojaswi's three stage-1 blocks for one country, fitted once (same settings as her stage1.py)."""

    def __init__(self, split: str, country: str, tmap: dict):
        p = pl.concat([oj_prepare(split, s, country, tmap) for s in (2, 3)])
        self.blocks = {}
        for name, (col, flt, k, max_df) in OJ_BLOCKS.items():
            pp = p if flt is None else p.filter(pl.col(flt))
            if pp.height == 0:
                continue
            vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=min(2, pp.height),
                                  max_df=max_df if pp.height > 100 else 1.0, sublinear_tf=True, dtype=np.float32)
            PT = vec.fit_transform(pp[col].to_list()).T.tocsr()
            self.blocks[name] = (col, k, vec, PT, pp["entity_id"])
        self.q = oj_prepare(split, 1, country, tmap).select("entity_id", *{c for c, *_ in OJ_BLOCKS.values()})

    def pairs(self, s1_ids: pl.Series) -> pl.DataFrame:
        q = self.q.join(pl.DataFrame({"entity_id": s1_ids}), on="entity_id", how="semi")
        out = None
        for name, (col, k, vec, PT, ids) in self.blocks.items():
            R = sp_matmul_topn(vec.transform(q[col].to_list()), PT, top_n=k, threshold=0.0,
                               n_threads=THREADS).tocoo()
            b = (pl.DataFrame({"s1_id": q["entity_id"].gather(R.row), "cand_id": ids.gather(R.col),
                               f"s_{name}": R.data.astype(np.float32)})
                 .with_columns(pl.col(f"s_{name}").rank("ordinal", descending=True).over("s1_id")
                               .cast(pl.Float32).alias(f"r_{name}")))
            out = b if out is None else out.join(b, on=["s1_id", "cand_id"], how="full", coalesce=True)
        if out is None:
            return pl.DataFrame(schema={"s1_id": pl.String, "cand_id": pl.String})
        return out


def record_stats(frame: pl.DataFrame, idx: CountryIndex, name_cnt: pl.DataFrame) -> pl.DataFrame:
    """entity_id, name_cnt (pool records with the same match name), max/min name-token IDF."""
    tok = (frame.select("entity_id", pl.col("name_skel").str.split(" ").alias("tok")).explode("tok")
           .join(idx.idf, on="tok", how="left")
           .group_by("entity_id").agg(pl.col("idf").fill_null(idx.max_idf).max().alias("tok_maxidf"),
                                      pl.col("idf").fill_null(idx.max_idf).min().alias("tok_minidf")))
    return (frame.select("entity_id", "name_m").join(name_cnt, on="name_m", how="left")
            .join(tok, on="entity_id", how="left")
            .select("entity_id", pl.col("name_cnt").fill_null(0).cast(pl.Float32),
                    pl.col("tok_maxidf").cast(pl.Float32), pl.col("tok_minidf").cast(pl.Float32)))


class Stage1:
    """v1 TF-IDF blocks + Ojaswi's blocks over one country's pool, and all v2 pair features."""

    def __init__(self, split: str, country: str, tmap: dict):
        t0 = time.time()
        self.tmap = tmap
        self.pool = prepare(pl.concat([load_norm(split, 2, country), load_norm(split, 3, country)]), tmap)
        self.idx = CountryIndex(self.pool)
        self.oj = OjIndex(split, country, tmap)
        self.pool_idx = self.pool.select(pl.col("entity_id").alias("cand_id")).with_row_index("pi")
        self.name_cnt = self.pool.group_by("name_m").agg(pl.len().alias("name_cnt"))
        self.p_stats = record_stats(self.pool, self.idx, self.name_cnt)
        print(f"  {split} {country}: indexes on {self.pool.height:,} records in {time.time() - t0:.0f}s", flush=True)

    def features(self, q: pl.DataFrame) -> pl.DataFrame:
        idx = self.idx
        QN = idx.name_vec.transform(q["name_key"].to_list())
        QA = idx.addr_vec.transform(q["addr"].to_list())
        parts = []
        for tag, Q, PT in (("name", QN, idx.PNT), ("addr", QA, idx.PAT)):
            C = sp_matmul_topn(Q, PT, top_n=WIDE_K, threshold=0.01, sort=True, n_threads=THREADS).tocoo()
            parts.append(
                pl.DataFrame({"qi": C.row.astype(np.uint32), "pi": C.col.astype(np.uint32),
                              "s": C.data.astype(np.float32)})
                .with_columns(pl.col("s").rank("ordinal", descending=True).over("qi")
                              .cast(pl.Float32).alias(f"{tag}_rank"))
                .drop("s").with_columns(pl.lit(1, pl.Int8).alias(f"in_{tag}")))
        oj = self.oj.pairs(q["entity_id"])
        oj = oj.with_columns([pl.lit(None, pl.Float32).alias(f"{k}_{b}") for b in OJ_BLOCKS for k in ("s", "r")
                              if f"{k}_{b}" not in oj.columns])
        q_idx = q.select(pl.col("entity_id").alias("s1_id")).with_row_index("qi")
        ojm = (oj.join(q_idx, on="s1_id").join(self.pool_idx, on="cand_id")
               .select(pl.col("qi").cast(pl.UInt32), pl.col("pi").cast(pl.UInt32)))
        pr = (parts[0].join(parts[1], on=["qi", "pi"], how="full", coalesce=True)
              .join(ojm, on=["qi", "pi"], how="full", coalesce=True)
              .with_columns(pl.col("in_name", "in_addr").fill_null(0)))
        qi, pi = pr["qi"].to_numpy(), pr["pi"].to_numpy()
        pr = pr.with_columns(pl.Series("name_cos", _rowdot(QN, idx.PN, qi, pi)),
                             pl.Series("addr_cos", _rowdot(QA, idx.PA, qi, pi)))
        f = v1.pair_features(pr, q, self.pool, idx)
        q_stats = record_stats(q, idx, self.name_cnt)
        return (
            f.join(oj, on=["s1_id", "cand_id"], how="left")
            .join(q_stats.select(pl.col("entity_id").alias("s1_id"), pl.col("name_cnt").alias("q_name_cnt"),
                                 pl.col("tok_minidf").alias("q_tok_minidf")), on="s1_id", how="left")
            .join(self.p_stats.select(pl.col("entity_id").alias("cand_id"), pl.col("name_cnt").alias("p_name_cnt"),
                                      pl.col("tok_maxidf").alias("p_tok_maxidf"),
                                      pl.col("tok_minidf").alias("p_tok_minidf")), on="cand_id", how="left")
            .with_columns((pl.col("in_name").cast(pl.Float32) + pl.col("in_addr")
                           + sum(pl.col(f"s_{b}").is_not_null().cast(pl.Float32) for b in OJ_BLOCKS))
                          .alias("n_blocks"))
            .select("s1_id", "cand_id", *[pl.col(c).cast(pl.Float32) for c in PAIR_FEATURES])
        )


# ------------------------------------------------------------------ per-S1 features

def rel_features(c: pl.DataFrame, text: pl.DataFrame) -> pl.DataFrame:
    """v1's per-S1 features plus house-number and name agreement with the other candidates."""
    c = v1.rel_features(c, text)
    t = c.select("s1_id", "cand_id", "prune_prob").join(
        text.select(pl.col("entity_id").alias("cand_id"), "house", "name_m"), on="cand_id", how="left")
    sib = t.join(t, on="s1_id", suffix="_o").filter(pl.col("cand_id") != pl.col("cand_id_o"))
    has = (pl.col("house") != "") & (pl.col("house_o") != "")
    same = (has & (pl.col("house") == pl.col("house_o"))).cast(pl.Float32)
    w = pl.col("prune_prob_o")
    agg = sib.group_by("s1_id", "cand_id").agg(
        ((same * w).sum() / (has.cast(pl.Float32) * w).sum()).alias("sib_house_w"),
        same.sort_by(w, descending=True).first().alias("sib_house_best"),
        (pl.col("name_m") == pl.col("name_m_o")).sum().cast(pl.Float32).alias("sib_same_name_n"),
    ).with_columns(pl.col("sib_house_w").fill_nan(None))
    return c.join(agg, on=["s1_id", "cand_id"], how="left").with_columns(
        *[pl.col(f).cast(pl.Float32) for f in SIB2_FEATURES])


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


def _read(name, countries=None, columns=None):
    return pl.concat([pl.read_parquet(f, columns=columns) for f in _parts(name, countries)])


def _paths(tag):
    sfx = f"_{tag}" if tag else ""
    return (str(MODEL_DIR / f"v2_pruner{sfx}_{{}}.txt"), MODEL_DIR / f"v2_final{sfx}.txt",
            MODEL_DIR / f"v2_params{sfx}.json")


def _prune(df, pruners):
    X = df.select(PAIR_FEATURES).to_numpy()
    return pl.Series("prune_prob", np.mean([m.predict(X) for m in pruners], axis=0), dtype=pl.Float32)


def cmd_prune(args):
    from src.common import tradeoff_table
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pruner_path, _, _ = _paths(args.tag)
    fit = _read("fitA", args.countries).with_columns((pl.col("s1_id").hash(SEED) % 2).alias("fold"))
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
    tradeoff_table(ranked, truth, cutoffs=[("all", None)] + [("topk", k) for k in (10, 8, 6, 5)]
                   + [("min_score", s) for s in (0.001, 0.003, 0.01)])
    for k in (10, 8, 6):
        for s in (0.001, 0.003, 0.01):
            t = tradeoff_table(ranked.filter((pl.col("block_rank") <= k) & (pl.col("prune_prob") >= s)),
                               truth, cutoffs=[("all", None)], verbose=False)
            print(f"  top{k} & prob>={s:<6} avg {t['avg_cands'][0]:.2f}  oracle {t['oracle'][0]:.4f}"
                  f"  pair recall {t['pair_recall'][0]:.4f}")


def cmd_train(args):
    from src.common import blocking_report, macro_f05, macro_f05_fast
    pruner_path, final_path, params_path = _paths(args.tag)
    pruners = [lgb.Booster(model_file=pruner_path.format(k)) for k in (0, 1)]
    text = pl.read_parquet(FEAT_DIR / "text_train.parquet")
    s1c = load_norm("train", 1).select(pl.col("entity_id").alias("s1_id"), "country")

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

    fit_c = rel_features(pl.concat([pruned("fitA", True), pruned("fitB", False)]), text)
    val_c = rel_features(pruned("val", False), text)
    gc.collect()
    truth = add_is_val(load_ground_truth()).filter("is_val").drop("is_val")
    blocking_report(val_c.with_columns(pl.col("prune_prob").alias("block_score")), truth,
                    name=f"v2 final candidates (top{args.final_k}, prob>={args.p_min}, val)")
    if args.countries:
        src_s1 = s1c.filter(pl.col("country").is_in(args.countries))
        truth_src, val_src = truth.join(src_s1, on="s1_id", how="semi"), val_c.join(src_s1, on="s1_id", how="semi")
    else:
        truth_src, val_src = truth, val_c
    dtrain = lgb.Dataset(fit_c.select(FINAL_FEATURES).to_numpy(), fit_c["label"].to_numpy(),
                         feature_name=FINAL_FEATURES)
    dval = lgb.Dataset(val_src.select(FINAL_FEATURES).to_numpy(), val_src["label"].to_numpy(), reference=dtrain)
    model = lgb.train(v1.FINAL_PARAMS, dtrain, num_boost_round=4000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(250)])
    model.save_model(str(final_path))
    imp = sorted(zip(FINAL_FEATURES, model.feature_importance("gain")), key=lambda x: -x[1])
    print(f"final model: {fit_c.height:,} train pairs ({fit_c['s1_id'].n_unique():,} S1), "
          f"best iteration {model.best_iteration}")
    print("top features:", ", ".join(f"{f} {g:.0f}" for f, g in imp[:20]))

    val_c = val_c.with_columns(pl.Series("prob", model.predict(val_c.select(FINAL_FEATURES).to_numpy()),
                                         dtype=pl.Float32))
    f_src, t, t1 = v1._tune(val_c.join(val_src.select("s1_id").unique(), on="s1_id", how="semi"), truth_src)
    pred = decide(val_c, t, t1)
    ref = macro_f05(pred.select("s1_id", "cand_id"), truth)
    per = v1._by_country(pred, truth, s1c)
    print(f"\nthresholds t={t}, t1={t1} (tuned on {'+'.join(args.countries) if args.countries else 'all'} val, "
          f"F0.5 there {f_src:.4f})")
    print(f"val macro F0.5 all {ref:.4f} | " + " | ".join(f"{k} {v:.4f}" for k, v in per.items()))
    n = truth.height
    print(f"val: {val_c.height / n:.2f} cands/S1, {pred.height / n:.2f} matches/S1, "
          f"{(n - pred['s1_id'].n_unique()) / n:.1%} S1 predicted empty")
    json.dump({"t": t, "t1": t1, "final_k": args.final_k, "p_min": args.p_min, "val_f05": ref,
               "val_by_country": per, "countries": args.countries, "best_iteration": model.best_iteration},
              open(params_path, "w"), indent=2)
    val_c.select("s1_id", "cand_id", "label", "prune_prob", "block_rank", "prob") \
        .write_parquet(FEAT_DIR / f"val_scored{'_' + args.tag if args.tag else ''}.parquet")


# ------------------------------------------------------------------ test

def cmd_test(args):
    from src.common import write_outputs
    pruner_path, final_path, params_path = _paths(args.tag)
    pruners = [lgb.Booster(model_file=pruner_path.format(k)) for k in (0, 1)]
    model = lgb.Booster(model_file=str(final_path))
    prm = json.load(open(params_path))
    tmap = load_script_map()
    s1 = load_norm("test", 1)
    kept = []
    for country in countries_of(s1):
        st = Stage1("test", country, tmap)
        text = st.pool.select("entity_id", "name_m", "addr", "house")
        q_all = prepare(s1.filter(pl.col("country") == country), tmap)
        for start in range(0, q_all.height, CHUNK):
            f = st.features(q_all.slice(start, CHUNK))
            f = f.with_columns(_prune(f, pruners))
            c = rel_features(select_candidates(f, prm["final_k"], prm["p_min"]), text)
            # the final model scores exactly the candidate set c
            c = c.with_columns(pl.Series("prob", model.predict(c.select(FINAL_FEATURES).to_numpy()),
                                         dtype=pl.Float32))
            kept.append(c.select("s1_id", "cand_id", pl.lit("stage1_union|pruner").alias("sources"),
                                 pl.col("prune_prob").alias("block_score"), "block_rank", "prob"))
        print(f"  test {country}: {q_all.height:,} S1 done", flush=True)
        del st
        gc.collect()
    cand = pl.concat(kept)
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    cand.write_parquet(CAND_DIR / "v2_arihant_test.parquet")
    pred = decide(cand, prm["t"], prm["t1"])
    write_outputs(pred.select("s1_id", "cand_id", "prob"),
                  cand.select("s1_id", "cand_id", "sources", "block_score", "block_rank"),
                  s1["entity_id"], REPO_ROOT / "output")
    n = s1.height
    print(f"test: {n:,} S1, {cand.height / n:.2f} cands/S1, {pred.height / n:.2f} matches/S1, "
          f"{(n - pred['s1_id'].n_unique()) / n:.1%} S1 with no match (t={prm['t']}, t1={prm['t1']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["features", "prune", "train", "test"])
    ap.add_argument("--countries", nargs="*")
    ap.add_argument("--tag", default="")
    ap.add_argument("--final-k", type=int, default=FINAL_K)
    ap.add_argument("--p-min", type=float, default=P_MIN)
    args = ap.parse_args()
    t0 = time.time()
    {"features": cmd_features, "prune": cmd_prune, "train": cmd_train, "test": cmd_test}[args.command](args)
    print(f"[{args.command}] finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
