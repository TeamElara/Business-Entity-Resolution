"""Stage 1 (wide, internal): union of three TF-IDF blocks per country.

Blocks, all fitted on the S2+S3 pool of one country and queried with that country's S1:
- word:   word TF-IDF on prepared "name address" text (main signal; the address is the most
          stable field across sources).
- skel:   the same text as consonant skeletons (absorbs typos and romanization differences).
- noaddr: name skeleton against only the pool records without an address, which the two
          blocks above rank poorly.
- namenum: name + the numbers of the address only. A long S1 address dilutes the word block;
          same name + same house number is found here (India val: +2.2 pts stage-1 recall).
- concat: the name with legal/filler words removed and spaces dropped ("nikolettamoorerhomes"),
          as character 3-grams, against only pool records whose name is one long token (domain-like
          names). 35% of the pairs the other four blocks miss are such names (val: +0.5 pts).
- namehouse: name + whole house-number tokens taken from the raw address ("J-52/4" -> j52_4,
          "70/1/1" -> 70_1_1). Small numbers alone are too frequent to count in TF-IDF; the full
          house number is rare and distinctive (India val: +0.4 pts; few US addresses have them).

Countries come from the S1 file, never from a fixed list. Outputs (gitignored):
- data/cand/stage1_{split}.parquet: s1_id, cand_id, s_word, r_word, s_skel, r_skel, s_noaddr, r_noaddr
- data/cand/prep_{split}.parquet:   prepared text per record, reused by the pruner features

prepare() results can be cached: set BLOCKING_PREP_CACHE to a folder (e.g. data/cand/prep_cache) and
each (split, source, country) is prepared once and read back on later runs. The cache key covers the
code of stage1.py and text.py, the token map and the raw file, so any change prepares again.

Usage:
    python -m src.blocking.stage1 --split train
    python -m src.blocking.stage1 --split train --countries India --val-only   # dev run
"""
import argparse
import hashlib
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from src.common.io import REPO_ROOT, scan_source, source_path
from src.common.split import is_val
from .text import NON_LATIN, clean_expr, latinize, load_script_map, skeletonize
from .tfidf_v0 import list_countries, peak_ram_gb

CAND_DIR = REPO_ROOT / "data" / "cand"
BLOCKS = {  # name: (text column, pool filter column or None, top-k, max_df)
    "word": ("text", None, 20, 0.01),
    "skel": ("skel", None, 20, 0.01),
    "noaddr": ("skel_name", "addr_empty", 10, 0.05),
    "namenum": ("namenum", None, 10, 0.05),
    "concat": ("concat3", "domainlike", 5, 0.05),
    "namehouse": ("namehouse", None, 5, 0.05),
}
# house-number tokens in a raw address: "J-52/4", "70/1/1", "2-1-241/62", "B-3", "39/2475-A"
HOUSE_RE = r"(?:\b[a-z]{1,3}\s?[-#.]?\s?)?\d+(?:\s?[/\-]\s?[a-z0-9]+)+|\b[a-z]{1,3}\s?-\s?\d+[a-z]?\b"


def house_tokens(addr_raw: pl.Expr) -> pl.Expr:
    """Whole house numbers of a raw address as tokens: separators -> '_', leading zeros dropped."""
    return (
        addr_raw.fill_null("").str.to_lowercase().str.extract_all(HOUSE_RE)
        .list.eval(pl.element().str.replace_all(r"\s+", "").str.replace_all(r"[/\-#.]+", "_")
                   .str.replace_all(r"(^|_|[a-z])0+(\d)", "${1}${2}"))
        .list.unique().list.join(" ")
    )
# legal forms and filler words dropped before concatenating a name (English, Indian and French forms)
CONCAT_DROP = (r"\b(private|limited|pvt|ltd|llc|l l c|llp|inc|incorporated|corp|corporation|co|company|plc"
               r"|group|enterprises|services|sarl|sas|sasu|sa|eurl|sci|snc|and|et|of|the|de|la|le|les|des|du"
               r"|m s|mr|mrs|dr|shri|sri|smt)\b")


def char_trigrams(texts: pl.Series) -> pl.Series:
    """Space-separated character 3-grams of each string, so a word TF-IDF acts as a char 3-gram one.

    Strings shorter than 3 characters are kept whole ("" stays "")."""
    k = (texts.str.len_chars().cast(pl.Int64) - 2).clip(lower_bound=1)  # 3-grams per string
    grams = (
        pl.DataFrame({"t": texts, "k": k})
        .select(pl.col("t").repeat_by("k").explode(), pl.int_ranges(0, "k").explode().alias("i"))
        .select(pl.col("t").str.slice(pl.col("i"), 3))["t"]
    )
    offsets = np.zeros(len(texts) + 1, dtype=np.int64)
    np.cumsum(k.to_numpy(), out=offsets[1:])
    return pl.Series("", pa.LargeListArray.from_arrays(pa.array(offsets), grams.rechunk().to_arrow())).list.join(" ")


def _prep_cache_path(split: str, source: int, country: str, tmap: dict) -> Path | None:
    """Cache file of prepare() for these inputs, or None when BLOCKING_PREP_CACHE is not set."""
    folder = os.environ.get("BLOCKING_PREP_CACHE")
    if not folder:
        return None
    h = hashlib.sha1()
    for f in (Path(__file__), Path(__file__).with_name("text.py")):
        h.update(f.read_bytes())
    h.update(json.dumps(tmap, sort_keys=True, ensure_ascii=False).encode())
    raw = source_path(split, source).resolve()
    h.update(f"{raw}|{raw.stat().st_size}|{raw.stat().st_mtime_ns}".encode())
    safe = "".join(ch if ch.isalnum() else "_" for ch in country)
    return Path(folder) / f"prep_{split}_s{source}_{safe}_{h.hexdigest()[:12]}.parquet"


def prepare(split: str, source: int, country: str, tmap: dict) -> pl.DataFrame:
    """Prepared text for one source file and one country.

    Columns: entity_id, name, addr, text, skel, skel_name, namenum, concat3, domainlike, namehouse,
    addr_empty, nonlatin. Cached when BLOCKING_PREP_CACHE is set (see the module docstring).
    """
    path = _prep_cache_path(split, source, country, tmap)
    if path is not None and path.exists():
        return pl.read_parquet(path)
    d = _prepare(split, source, country, tmap)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        d.write_parquet(tmp)
        tmp.replace(path)
    return d


def _prepare(split: str, source: int, country: str, tmap: dict) -> pl.DataFrame:
    d = (
        scan_source(split, source, country=country, columns=["entity_id", "business_name", "business_address"])
        .select(
            "entity_id",
            clean_expr(pl.col("business_name")).alias("name"),
            clean_expr(pl.col("business_address")).alias("addr"),
            pl.col("business_name").fill_null("").str.contains(NON_LATIN).alias("nonlatin"),
            house_tokens(pl.col("business_address")).alias("house"),
        )
        .collect()
    )
    d = d.with_columns(name=latinize(d["name"], tmap), addr=latinize(d["addr"], tmap))
    d = d.with_columns(
        (pl.col("name") + " " + pl.col("addr")).str.strip_chars().alias("text"),
        (pl.col("addr") == "").alias("addr_empty"),
        (pl.col("name") + " " + pl.col("addr").str.extract_all(r"\d+").list.join(" ")).str.strip_chars().alias("namenum"),
    )
    concat = d["name"].str.replace_all(CONCAT_DROP, " ").str.replace_all(r"\s+", "")
    return d.with_columns(
        skel=skeletonize(d["text"]),
        skel_name=skeletonize(d["name"]),
        concat3=char_trigrams(concat),
        domainlike=(d["name"].str.split(" ").list.len() == 1) & (concat.str.len_chars() >= 8),
        namehouse=(d["name"] + " " + d["house"]).str.strip_chars(),
    ).drop("house")


def fit_tfidf(texts: list[str], max_df: float):
    """Word TF-IDF fitted on pool texts -> (vectorizer, transposed pool matrix).

    Falls back to keeping every token when max_df/min_df would leave none (tiny pools).
    """
    for mn, mx in ((min(2, len(texts)), max_df if len(texts) > 100 else 1.0), (1, 1.0)):
        vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=mn, max_df=mx,
                              sublinear_tf=True, dtype=np.float32)
        try:
            return vec, vec.fit_transform(texts).T.tocsr()
        except ValueError:
            continue
    raise ValueError("no tokens in pool texts")


def topk_block(q: pl.DataFrame, p: pl.DataFrame, col: str, k: int, max_df: float,
               chunk: int, n_threads: int) -> pl.DataFrame:
    """Top-k cosine pool matches of every query on column `col`: s1_id, cand_id, score, rank."""
    if q.height == 0 or p.height == 0:
        return pl.DataFrame(schema={"s1_id": pl.String, "cand_id": pl.String,
                                    "score": pl.Float32, "rank": pl.UInt32})
    vec, PT = fit_tfidf(p[col].to_list(), max_df)
    Q = vec.transform(q[col].to_list())
    qi, pi, sc = [], [], []
    for start in range(0, Q.shape[0], chunk):
        R = sp_matmul_topn(Q[start:start + chunk], PT, top_n=k, threshold=0.0, n_threads=n_threads).tocoo()
        qi.append(R.row.astype(np.int64) + start)
        pi.append(R.col.astype(np.int64))
        sc.append(R.data.astype(np.float32))
    qi, pi, sc = np.concatenate(qi), np.concatenate(pi), np.concatenate(sc)
    return (
        pl.DataFrame({"s1_id": q["entity_id"].gather(qi), "cand_id": p["entity_id"].gather(pi), "score": sc})
        .with_columns(pl.col("score").rank("ordinal", descending=True).over("s1_id").alias("rank"))
    )


def run_country(split: str, country: str, tmap: dict, s1_filter, chunk: int, n_threads: int):
    """Stage-1 pairs and prepared records (S1 + pool) for one country.

    s1_filter: optional function s1_id -> bool selecting the S1 records to query.
    """
    q = prepare(split, 1, country, tmap)
    if s1_filter is not None:
        q = q.filter(pl.Series([s1_filter(x) for x in q["entity_id"].to_list()], dtype=pl.Boolean))
    p = pl.concat([prepare(split, s, country, tmap) for s in (2, 3)])
    pairs = None
    for name, (col, flt, k, max_df) in BLOCKS.items():
        pp = p if flt is None else p.filter(pl.col(flt))
        b = topk_block(q, pp, col, k, max_df, chunk, n_threads).rename({"score": f"s_{name}", "rank": f"r_{name}"})
        pairs = b if pairs is None else pairs.join(b, on=["s1_id", "cand_id"], how="full", coalesce=True)
    return pairs, q, p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--countries", nargs="*", help="restrict to these countries (default: all in S1)")
    ap.add_argument("--val-only", action="store_true", help="only query val S1 (train split, dev runs)")
    ap.add_argument("--dev-sample", action="store_true",
                    help="val S1 plus a 10%% non-val sample (crc32 %% 10 == 1) to train the pruner quickly")
    ap.add_argument("--crc-mod", type=int, nargs="*",
                    help="only S1 with crc32(id) %% 10 in these values (e.g. 1 = a 10%% non-val sample)")
    ap.add_argument("--chunk", type=int, default=20000, help="S1 queries scored per batch")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _dev")
    args = ap.parse_args()

    t0 = time.time()
    tmap = load_script_map()
    countries = args.countries or list_countries(args.split)
    s1_filter = None
    if args.val_only:
        s1_filter = is_val
    elif args.dev_sample:
        s1_filter = lambda x: zlib.crc32(x.encode()) % 10 in (0, 1)
    elif args.crc_mod:
        mods = set(args.crc_mod)
        s1_filter = lambda x: zlib.crc32(x.encode()) % 10 in mods
    pairs, preps = [], []
    for country in countries:
        t1 = time.time()
        pr, q, p = run_country(args.split, country, tmap, s1_filter, args.chunk, args.threads)
        pairs.append(pr)
        preps.append(pl.concat([q, p]).drop("text", "skel", "namenum", "concat3", "namehouse"))
        print(f"[{country}] S1 {q.height:,} x pool {p.height:,} -> {pr.height:,} pairs "
              f"({pr.height / max(q.height, 1):.1f}/S1) in {time.time() - t1:.0f}s, peak RAM {peak_ram_gb():.1f} GB",
              flush=True)
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    out = CAND_DIR / f"stage1_{args.split}{args.tag}.parquet"
    pl.concat(pairs).write_parquet(out)
    pl.concat(preps).write_parquet(CAND_DIR / f"prep_{args.split}{args.tag}.parquet")
    print(f"wrote {out} in {time.time() - t0:.0f}s, peak RAM {peak_ram_gb():.1f} GB")


if __name__ == "__main__":
    main()
