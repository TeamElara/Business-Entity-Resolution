"""Text preparation for blocking: one cleaned, Latin-script token string per record.

Everything here is country-agnostic and learned only from the input files and non-validation
training labels (no external data):

- `clean_expr`: lowercase, accents folded on Latin letters only (Indic vowel signs are kept),
  digit look-alikes inside words fixed ("5ystems" -> "systems", "kingst0n" -> "kingston"),
  web prefixes/suffixes dropped ("www.x.com" -> "x"), ordinal suffixes and leading zeros
  removed from numbers ("00708" -> "708", "65rd" -> "65").
- Indic scripts: a token map learned from aligned training name pairs (Latin S1 name vs
  non-Latin S2/S3 name of the same business, validation S1 excluded), with a rule-based
  romanizer as fallback. All Indic Unicode blocks share the Devanagari layout, so one
  romanizer covers Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada
  and Malayalam.
- `skeleton`: a consonant key per token ("motors", "motars" -> "mtrs") that absorbs vowel
  typos and romanization differences.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

from src.common.io import REPO_ROOT, load_ground_truth, scan_source, truth_pairs
from src.common.split import add_is_val

# v2: learned after zero-width joiners are removed in clean_expr (older cached maps are stale)
MAP_PATH = REPO_ROOT / "data" / "cand" / "script_token_map_v2.json"
NON_LATIN = r"[\p{L}&&\P{Latin}]"

# ------------------------------------------------------------------ basic cleaning

_LEET = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"}


def clean_expr(col: pl.Expr) -> pl.Expr:
    """Vectorised cleaning shared by names and addresses (see module docstring)."""
    s = (
        col.fill_null("")
        # zero-width joiners inside Indic words ("ఎస్\u200cఎస్") must not split the word
        .str.replace_all(r"[\u200b-\u200d\u2060\ufeff\u00ad]", "")
        .str.to_lowercase()
        .str.replace_all(r"https?://|www\.", " ")
        .str.replace_all(r"\.(com|net|org|in|co|biz|info|us|fr)\b", " ")
        .str.normalize("NFKD")
        .str.replace_all(r"([a-z])\p{Mn}+", "$1")  # fold accents on Latin letters only
        .str.normalize("NFC")
        .str.replace_all(r"(\d)(st|nd|rd|th)\b", "$1")
    )
    # digit look-alikes between letters, or at the start of a word followed by letters
    for d, ch in _LEET.items():
        esc = "\\" + d if d in "$" else d
        s = s.str.replace_all(rf"([a-z]){esc}([a-z])", f"${{1}}{ch}${{2}}")
        s = s.str.replace_all(rf"(^|[^\p{{L}}\p{{N}}]){esc}([a-z]{{2,}})", f"${{1}}{ch}${{2}}")
    return (
        s.str.replace_all(r"[^\p{L}\p{N}\p{M}]+", " ")
        .str.replace_all(r"\b0+(\d)", "$1")
        .str.strip_chars()
    )


# ------------------------------------------------------------------ Indic scripts

_INDIC_BASES = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)
_VOWELS = {0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ee", 0x09: "u", 0x0A: "oo", 0x0B: "ri",
           0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au"}
_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "ch", 0x1B: "chh",
         0x1C: "j", 0x1D: "jh", 0x1E: "ny", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
         0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
         0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
         0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh",
         0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "gh", 0x5B: "z", 0x5C: "r",
         0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_MATRAS = {0x3E: "aa", 0x3F: "i", 0x40: "ee", 0x41: "u", 0x42: "oo", 0x43: "ri", 0x44: "ri",
           0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
           0x4C: "au", 0x57: "au"}
_SIGNS = {0x01: "n", 0x02: "n", 0x03: "h"}
_VIRAMA, _NUKTA = 0x4D, 0x3C


def _indic_offset(ch: str):
    """Offset of `ch` inside its Indic block (Devanagari layout), or None."""
    o = ord(ch)
    for base in _INDIC_BASES:
        if base <= o < base + 0x80:
            return o - base
    return None


def romanize(token: str) -> str:
    """Rule-based romanization of one Indic token (any of the 9 Brahmic blocks).

    Inherent 'a' is written inside a word and dropped at the end. Characters outside the
    Indic blocks are kept as they are.
    """
    offs = [_indic_offset(c) for c in token]
    out = []
    i, n = 0, len(token)
    while i < n:
        o = offs[i]
        if o is None:
            out.append(token[i])
            i += 1
            continue
        j = i + 1
        while j < n and offs[j] == _NUKTA:
            j += 1
        if o in _CONS:
            out.append(_CONS[o])
            if j < n and offs[j] == _VIRAMA:
                j += 1
            elif j < n and offs[j] in _MATRAS:
                out.append(_MATRAS[offs[j]])
                j += 1
            elif j < n and offs[j] is not None and offs[j] not in _SIGNS:
                out.append("a")
        elif o in _VOWELS:
            out.append(_VOWELS[o])
        elif o in _SIGNS:
            out.append(_SIGNS[o])
        elif 0x66 <= o <= 0x6F:
            out.append(str(o - 0x66))
        i = j
    return "".join(out)


def learn_script_map(min_votes: int = 2, min_share: float = 0.6) -> dict[str, str]:
    """Non-Latin -> Latin token map from non-validation training name pairs.

    Only pairs with the same token count vote, token by token in order; a token is mapped to
    its most frequent Latin partner when that partner has >= min_share of the votes.
    """
    gt = add_is_val(load_ground_truth())
    tp = truth_pairs(gt.filter(~pl.col("is_val")))
    s1 = scan_source("train", 1, columns=["entity_id", "business_name"]).select(
        pl.col("entity_id").alias("s1_id"), clean_expr(pl.col("business_name")).alias("n1")
    ).collect()
    pool = pl.concat([
        scan_source("train", s, columns=["entity_id", "business_name"])
        .filter(pl.col("business_name").str.contains(NON_LATIN))
        .select(pl.col("entity_id").alias("cand_id"), clean_expr(pl.col("business_name")).alias("n2"))
        .collect()
        for s in (2, 3)
    ])
    pairs = (
        tp.join(pool, on="cand_id").join(s1, on="s1_id")
        .filter(~pl.col("n1").str.contains(NON_LATIN))
        .select("n1", "n2")
    )
    votes = defaultdict(Counter)
    for a, b in pairs.iter_rows():
        ta, tb = a.split(), b.split()
        if len(ta) == len(tb):
            for x, y in zip(tb, ta):
                if x != y:
                    votes[x][y] += 1
    tmap = {}
    for x, c in votes.items():
        y, v = c.most_common(1)[0]
        if v >= min_votes and v / sum(c.values()) >= min_share:
            tmap[x] = y
    return tmap


def load_script_map(path: Path = MAP_PATH) -> dict[str, str]:
    """Load the learned token map, learning and caching it on first use."""
    if path.exists():
        return json.loads(path.read_text())
    tmap = learn_script_map()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tmap, ensure_ascii=False))
    return tmap


def map_tokens(texts: pl.Series, fn, only: str | None = None) -> pl.Series:
    """Apply `fn` to every distinct token of `texts` (space separated) and rebuild the strings.

    Python runs once per distinct token (optionally only on tokens matching regex `only`);
    the rebuild is vectorised through a lookup table.
    """
    toks = texts.str.split(" ").explode().drop_nulls().unique()
    if only is not None:
        toks = toks.filter(toks.str.contains(only))
    toks = toks.to_list()
    if not toks:
        return texts
    lut = pl.DataFrame({"tok": toks, "new": [fn(t) for t in toks]})
    return (
        pl.DataFrame({"t": texts}).with_row_index("row")
        .with_columns(pl.col("t").str.split(" "))
        .explode("t")
        .join(lut, left_on="t", right_on="tok", how="left", maintain_order="left")
        .with_columns(pl.coalesce("new", "t").alias("t"))
        .group_by("row", maintain_order=True)
        .agg(pl.col("t").str.join(" "))["t"]
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def latinize(texts: pl.Series, tmap: dict[str, str]) -> pl.Series:
    """Replace non-Latin tokens by their learned Latin form, else their romanization."""
    return map_tokens(texts, lambda t: tmap.get(t) or romanize(t), only=NON_LATIN)


# ------------------------------------------------------------------ skeleton key

_SKEL_SUBS = (("ph", "f"), ("ck", "k"), ("c", "k"), ("q", "k"), ("v", "w"), ("z", "s"), ("x", "ks"))


def skeleton(token: str) -> str:
    """Consonant key of one token: first letter kept, later vowels and 'h' after a consonant
    dropped, c/q->k, v->w, z->s, repeats collapsed ("motors", "motars" -> "mtrs").
    Tokens with digits are returned unchanged."""
    if not token or any(ch.isdigit() for ch in token):
        return token
    for a, b in _SKEL_SUBS:
        token = token.replace(a, b)
    out = [token[0]]
    for i, ch in enumerate(token[1:], 1):
        if ch in "aeiouy":
            continue
        if ch == "h" and token[i - 1] not in "aeiouy":
            continue
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


def skeletonize(texts: pl.Series) -> pl.Series:
    return map_tokens(texts, skeleton)
