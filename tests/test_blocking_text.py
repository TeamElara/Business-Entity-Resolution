import polars as pl
import pytest

from src.blocking.stage1 import char_trigrams
from src.blocking.text import map_tokens, skeleton


def ref_map_tokens(texts, fn, only=None):
    """Plain Python reference: map every token (or those matching `only`), then squeeze spaces."""
    import re
    out = []
    for t in texts:
        toks = (t or "").split(" ")
        new = [fn(x) if (only is None or re.search(only, x)) else x for x in toks]
        out.append(re.sub(r"\s+", " ", " ".join(new)).strip())
    return out


TEXTS = ["", "  a  b ", "x", "motors motars", "नमस्ते दुनिया", "a b", "ab 12 cd", "zzz"]


@pytest.mark.parametrize("only", [None, r"[\p{L}&&\P{Latin}]"])
def test_map_tokens_matches_reference(only):
    fn = skeleton if only is None else (lambda t: "LAT")
    got = map_tokens(pl.Series(TEXTS), fn, only).to_list()
    exp = ref_map_tokens(TEXTS, fn, r"[^\x00-ɏ\s]" if only else None)
    assert got == exp


def test_map_tokens_null_becomes_empty():
    assert map_tokens(pl.Series(["ab", None], dtype=pl.String), skeleton).to_list() == ["ab", ""]


def test_map_tokens_no_matching_token_returns_input():
    s = pl.Series(["abc def", None], dtype=pl.String)
    assert map_tokens(s, lambda t: "X", only=r"\d").to_list() == ["abc def", None]


@pytest.mark.parametrize("t", ["", "a", "ab", "abc", "abcd", "नमस्ते", "ﬁx", "nikolettamoorerhomes"])
def test_char_trigrams_matches_python(t):
    exp = " ".join(t[i:i + 3] for i in range(max(len(t) - 2, 1)))
    assert char_trigrams(pl.Series([t, "abcde"])).to_list() == [exp, "abc bcd cde"]
