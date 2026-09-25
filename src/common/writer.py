"""Writers for the two submission TSVs, so both are always written the same way.

One row per S1 (in the order of `s1_ids`), IDs comma-joined, empty when none,
tab-separated, no quoting.
"""
from pathlib import Path

import polars as pl

MATCH_COL = "matched_entity_ids"
CAND_COL = "candidate_entity_ids"


def write_id_lists(pairs: pl.DataFrame, s1_ids, path, list_col: str,
                   order_by=None, descending: bool = False) -> pl.DataFrame:
    """Write (s1_id, cand_id) pairs as `source1_entity_id<TAB>list_col`.

    s1_ids: every required S1 id (e.g. the test S1 entity_id column); S1s with no
    pairs get an empty list. Pairs for S1 ids not in s1_ids raise.
    """
    base = pl.DataFrame({"s1_id": pl.Series(s1_ids, dtype=pl.String)})
    if base["s1_id"].is_duplicated().any():
        raise ValueError("s1_ids contains duplicates")

    p = pairs.drop_nulls("cand_id")
    bad = p.filter(~pl.col("cand_id").str.contains(r"^S[23]-[^\t,]+$"))
    if bad.height:
        raise ValueError(f"{bad.height} ids are not clean S2-/S3- ids, e.g. {bad['cand_id'].head(5).to_list()}")
    extra = p.join(base, on="s1_id", how="anti")
    if extra.height:
        raise ValueError(f"{extra['s1_id'].n_unique()} S1 ids not in s1_ids, e.g. {extra['s1_id'].head(5).to_list()}")
    if order_by is not None:
        p = p.sort(["s1_id", order_by], descending=[False, descending])

    lists = p.group_by("s1_id", maintain_order=True).agg(
        pl.col("cand_id").unique(maintain_order=True).str.join(",").alias(list_col)
    )
    out = (
        base.join(lists, on="s1_id", how="left", maintain_order="left")
        .with_columns(pl.col(list_col).fill_null(""))
        .rename({"s1_id": "source1_entity_id"})
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never", line_terminator="\n")
    return out


def write_matching_results(matches: pl.DataFrame, s1_ids, path="output/matching_results.tsv"):
    """matches: (s1_id, cand_id[, prob]); ids ordered by prob (desc) when present."""
    order = "prob" if "prob" in matches.columns else None
    return write_id_lists(matches, s1_ids, path, MATCH_COL, order_by=order, descending=True)


def write_candidate_pairs(cands: pl.DataFrame, s1_ids, path="output/candidate_pairs.tsv"):
    """cands: the §8 candidate frame; ids ordered by block_rank when present."""
    order = "block_rank" if "block_rank" in cands.columns else None
    return write_id_lists(cands, s1_ids, path, CAND_COL, order_by=order)


def write_outputs(matches: pl.DataFrame, cands: pl.DataFrame, s1_ids, out_dir="output"):
    """Write both TSVs after checking that every match is one of the candidates."""
    missing = matches.drop_nulls("cand_id").join(cands, on=["s1_id", "cand_id"], how="anti")
    if missing.height:
        raise ValueError(f"{missing.height} matched pairs are not in the candidate set "
                         f"(matches must be a subset of candidates)")
    out_dir = Path(out_dir)
    write_matching_results(matches, s1_ids, out_dir / "matching_results.tsv")
    write_candidate_pairs(cands, s1_ids, out_dir / "candidate_pairs.tsv")
