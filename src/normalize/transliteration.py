"""Training-pair Hindi token map with a small built-in Unicode fallback.

The mapping is learned only from non-validation true matches. It is a local
generated artifact, not checked into Git. No external transliteration library
or license is needed for the deterministic fallback below.
"""

from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import unicodedata

import polars as pl

from src.common.io import REPO_ROOT, load_ground_truth, scan_source, truth_pairs
from src.common.split import add_is_val
from .abbreviations import normalize_name_expr
from .legal import LEGAL_FORMS


DEFAULT_MAP_PATH = REPO_ROOT / "data" / "norm" / "hi_latin_map.json"

VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o",
}
CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l", "क़": "q",
    "ख़": "kh", "ग़": "gh", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
}
MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo",
    "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o",
    "ॅ": "e", "ॆ": "e", "ॊ": "o", "ॄ": "ri",
}
SIGNS = {"ं": "n", "ँ": "n", "ः": "h", "ऽ": ""}
VIRAMA = "्"
NUKTA = "़"


def has_devanagari(value: str) -> bool:
    return any("\u0900" <= ch <= "\u097f" for ch in value)


def strip_accents(value: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", value)
        if unicodedata.category(ch) != "Mn"
    )


def romanize_token(token: str) -> str:
    """Best-effort deterministic fallback for a Devanagari token.

    The learned token map is preferred. This fallback is intentionally small:
    it guarantees Latin output for unseen Hindi tokens but does not claim
    linguistic accuracy for all names or languages.
    """
    out = []
    i = 0
    while i < len(token):
        ch = token[i]
        next_i = i + 1
        if next_i < len(token) and token[next_i] == NUKTA:
            next_i += 1
        if ch in CONSONANTS:
            base = CONSONANTS[ch]
            if next_i < len(token) and token[next_i] == VIRAMA:
                out.append(base)
                i = next_i + 1
            elif next_i < len(token) and token[next_i] in MATRAS:
                out.append(base + MATRAS[token[next_i]])
                i = next_i + 1
            else:
                # Drop a final inherent schwa; preserve it inside a word.
                out.append(base + ("a" if next_i < len(token) else ""))
                i = next_i
        elif ch in VOWELS:
            out.append(VOWELS[ch])
            i += 1
        elif ch in SIGNS:
            out.append(SIGNS[ch])
            i += 1
        elif "०" <= ch <= "९":
            out.append(str(ord(ch) - ord("०")))
            i += 1
        elif ch in (NUKTA, VIRAMA):
            i += 1
        elif has_devanagari(ch):
            # Keep the fallback Latin-only for rare Unicode code points not
            # covered by the explicit alphabet above.
            out.append("x")
            i += 1
        else:
            out.append(strip_accents(ch))
            i += 1
    return "".join(out).lower()


def _strip_latin_legal_suffix(value: str) -> str:
    for phrase, _ in LEGAL_FORMS:
        suffix = " " + phrase
        if value.endswith(suffix) and len(value) > len(suffix):
            return value[:-len(suffix)]
    return value


def transliterate_name(name_core: str | None, token_map: dict[str, str]) -> str:
    """Romanize only Devanagari tokens; retain non-Devanagari name tokens."""
    if not name_core:
        return ""
    words = [
        (token_map[token] if token in token_map else romanize_token(token))
        if has_devanagari(token)
        else strip_accents(token)
        for token in name_core.split()
    ]
    return _strip_latin_legal_suffix(" ".join(words))


@lru_cache(maxsize=4)
def _read_token_map(path: str, modified_ns: int) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["tokens"]


def load_token_map(path: Path = DEFAULT_MAP_PATH) -> dict[str, str]:
    """Cache by file modification time, so a newly trained map is picked up."""
    if not path.is_file():
        return {}
    return _read_token_map(str(path.resolve()), path.stat().st_mtime_ns)


def matched_hindi_training_names() -> pl.DataFrame:
    """Non-validation true pairs from raw train TSVs, independent of Parquet."""
    truth = add_is_val(load_ground_truth()).filter(~pl.col("is_val"))
    pairs = truth_pairs(truth)
    dev = pl.concat([
        scan_source("train", source, columns=["entity_id", "business_name"])
        .filter(pl.col("business_name").str.contains(r"\p{Devanagari}"))
        for source in (2, 3)
    ]).select(
        pl.col("entity_id").alias("cand_id"),
        normalize_name_expr(pl.col("business_name")).alias("dev_name"),
    )
    latin = scan_source("train", 1, columns=["entity_id", "business_name"]).select(
        pl.col("entity_id").alias("s1_id"),
        normalize_name_expr(pl.col("business_name")).alias("latin_name"),
    )
    return pairs.lazy().join(dev, on="cand_id").join(latin, on="s1_id").select(
        "latin_name", "dev_name"
    ).collect(engine="streaming")


def build_token_map(names: pl.DataFrame, min_confidence: float = 0.60) -> tuple[dict[str, str], dict]:
    """Align equal-length true-pair token sequences and learn majority forms."""
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    usable_pairs = 0
    for latin_name, dev_name in names.select("latin_name", "dev_name").iter_rows():
        latin = (latin_name or "").split()
        dev = (dev_name or "").split()
        if len(latin) != len(dev) or not latin:
            continue
        usable_pairs += 1
        for hindi_token, latin_token in zip(dev, latin, strict=True):
            if has_devanagari(hindi_token) and latin_token and not has_devanagari(latin_token):
                votes[hindi_token][strip_accents(latin_token)] += 1
    mapping = {}
    ambiguous = 0
    for hindi_token, choices in votes.items():
        best, count = sorted(choices.items(), key=lambda item: (-item[1], item[0]))[0]
        if count / sum(choices.values()) >= min_confidence:
            mapping[hindi_token] = best
        else:
            ambiguous += 1
    stats = {
        "matched_nonval_devanagari_pairs": names.height,
        "equal_token_count_pairs": usable_pairs,
        "distinct_hindi_tokens": len(votes),
        "accepted_token_mappings": len(mapping),
        "ambiguous_tokens_left_to_fallback": ambiguous,
        "min_confidence": min_confidence,
        "validation_rule": "crc32(s1_id) % 10 != 0 for training",
    }
    return mapping, stats


def train_and_write_map(path: Path, min_confidence: float = 0.60) -> dict:
    """Build the local artifact from raw non-validation training matches."""
    names = matched_hindi_training_names()
    mapping, stats = build_token_map(names, min_confidence)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"stats": stats, "tokens": dict(sorted(mapping.items()))},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return stats
