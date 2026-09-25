"""Toy tests for src/common (no real data needed): python -m src.common.selftest"""
import sys
import tempfile
from pathlib import Path

import polars as pl

from .io import read_tsv
from .metrics import blocking_report, f05_entity, macro_f05, macro_f05_fast, to_set_dict, tradeoff_table
from .split import add_is_val, is_val
from .writer import write_outputs

REPO_ROOT = Path(__file__).resolve().parents[2]


def close(a, b, eps=1e-9):
    return abs(a - b) < eps


def main():
    # f05_entity: README worked example + edge cases
    assert close(f05_entity({"S2-47", "S2-193", "S3-812"}, {"S2-47", "S3-812"}), 1.25 * (2 / 3) / (0.25 * 2 / 3 + 1))
    assert f05_entity(set(), set()) == 1.0 and f05_entity({"a"}, set()) == 0.0
    assert f05_entity(set(), {"a"}) == 0.0 and f05_entity({"b"}, {"a"}) == 0.0
    print("f05_entity README example:", round(f05_entity({"S2-47", "S2-193", "S3-812"}, {"S2-47", "S3-812"}), 3))

    # is_val is deterministic crc32, not hash()
    assert is_val("S1-965667") == is_val("S1-965667")
    assert add_is_val(pl.DataFrame({"s1_id": ["S1-1", "S1-2"]}))["is_val"].dtype == pl.Boolean

    # read_tsv: quotes kept literally, "NA" stays a string, "" -> null
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.tsv"
        p.write_text('entity_id\tbusiness_name\tbusiness_address\tcountry\n'
                     'S1-1\t"Joe\'s" "Diner\tNA\tUS\n'
                     'S1-2\t\t\tIndia\n')
        df = read_tsv(p)
        assert df["business_name"].to_list() == ['"Joe\'s" "Diner', None]
        assert df["business_address"].to_list() == ["NA", None]
        assert df.schema["country"] == pl.String
    print("read_tsv: quotes/NA/empty OK")

    # toy truth: A has 2 matches, B has 1, C is a singleton, D has 3
    truth = pl.DataFrame({
        "s1_id": ["S1-A", "S1-B", "S1-C", "S1-D"],
        "matched_ids": [["S2-1", "S3-1"], ["S2-2"], [], ["S2-3", "S3-3", "S3-4"]],
    })
    assert close(macro_f05(truth, truth), 1.0)
    assert close(macro_f05({}, truth), 0.25)  # only the singleton scores
    assert close(macro_f05({"S1-A": {"S2-1", "S2-9"}}, truth), (f05_entity({"S2-1", "S2-9"}, {"S2-1", "S3-1"}) + 1) / 4)
    preds = pl.DataFrame({"s1_id": ["S1-A", "S1-A", "S1-B", "S1-C", "S1-D", "S1-Z"],
                          "cand_id": ["S2-1", "S2-9", "S2-2", "S3-9", "S3-3", "S2-5"]})
    assert close(macro_f05_fast(preds, truth), macro_f05(preds, truth))
    assert close(macro_f05_fast(preds.head(0), truth), 0.25)

    rows = [  # s1, cand, score, rank
        ("S1-A", "S2-1", 0.95, 1), ("S1-A", "S2-8", 0.90, 2), ("S1-A", "S3-1", 0.60, 3),
        ("S1-B", "S2-7", 0.80, 1), ("S1-B", "S2-2", 0.75, 2),
        ("S1-C", "S3-9", 0.40, 1),
        ("S1-D", "S3-3", 0.99, 1),
        ("S1-Z", "S2-5", 0.50, 1),  # S1 outside truth -> ignored
    ]
    cand = pl.DataFrame(rows, schema=["s1_id", "cand_id", "block_score", "block_rank"], orient="row") \
        .with_columns(pl.lit("toy").alias("sources"), pl.col("block_rank").cast(pl.Int64))

    # oracle computed with f05_entity directly
    T, C = to_set_dict(truth), to_set_dict(cand)
    oracle_loop = sum(f05_entity(T[s] & C.get(s, set()), T[s]) for s in T) / len(T)
    country = pl.DataFrame({"s1_id": ["S1-A", "S1-B", "S1-C", "S1-D"], "country": ["US", "US", "India", "France"]})
    rep = blocking_report(cand, truth, s1_country=country, pool_size=100, name="toy")
    assert close(rep["oracle_ceiling"], oracle_loop), (rep["oracle_ceiling"], oracle_loop)
    assert rep["total_pairs"] == 7 and rep["ignored_pairs"] == 1 and rep["max_cands"] == 3
    assert close(rep["pair_recall"], 4 / 6)
    assert close(rep["reduction_ratio"], 1 - 7 / (4 * 100))
    bn = {r["n_true_grp"]: r for r in rep["by_n_true"].iter_rows(named=True)}
    assert sorted(bn) == [0, 1, 2, 3] and close(bn[3]["pair_recall"], 1 / 3) and close(bn[2]["oracle"], 1.0)
    print(f"oracle vectorised == f05 loop: {oracle_loop:.4f}")

    tab = tradeoff_table(cand, truth, cutoffs=[("all", None), ("topk", 1), ("topk", 2), ("gap", 0.1), ("topk_gap", (2, 0.1))])
    top1 = tab.filter(pl.col("cutoff") == "topk=1").row(0, named=True)
    # top-1 keeps S2-1 (A), S2-7 (B), S3-9 (C), S3-3 (D): recall 2/6
    assert close(top1["avg_cands"], 1.0) and close(top1["pair_recall"], 2 / 6)
    gap = tab.filter(pl.col("cutoff") == "gap=0.1").row(0, named=True)
    # gap 0.1 keeps A:{S2-1,S2-8} B:{S2-7,S2-2} C:{S3-9} D:{S3-3} -> 6 pairs
    assert gap["total_pairs"] == 6
    assert close(tab.row(0, named=True)["oracle"], oracle_loop)

    # writer: round trip through the official validator
    sys.path.insert(0, str(REPO_ROOT / "utils"))
    import validate_submission as vs
    matches = pl.DataFrame({"s1_id": ["S1-A", "S1-A", "S1-B"], "cand_id": ["S3-1", "S2-1", "S2-2"],
                            "prob": [0.7, 0.9, 0.8]})
    s1_ids = ["S1-A", "S1-B", "S1-C", "S1-D"]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "test_source1.tsv").write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                                            + "".join(f"{s}\tx\ty\tUS\n" for s in s1_ids))
        write_outputs(matches, cand.filter(pl.col("s1_id") != "S1-Z"), s1_ids, d)
        mr = (d / "matching_results.tsv").read_text()
        cp = (d / "candidate_pairs.tsv").read_text()
        assert mr == ("source1_entity_id\tmatched_entity_ids\n"
                      "S1-A\tS2-1,S3-1\nS1-B\tS2-2\nS1-C\t\nS1-D\t\n"), repr(mr)
        assert cp.splitlines()[1] == "S1-A\tS2-1,S2-8,S3-1", repr(cp)
        errors, warnings = vs.validate(str(d / "matching_results.tsv"), str(d / "candidate_pairs.tsv"), str(d))
        assert not errors, errors
        assert not any("not present in candidate_pairs" in w for w in warnings)
        try:  # a match outside the candidates must be refused
            write_outputs(pl.DataFrame({"s1_id": ["S1-C"], "cand_id": ["S2-1"]}), cand, s1_ids, d)
            raise AssertionError("subset check did not fire")
        except ValueError:
            pass
    print("writer: output passes validate_submission, subset check fires")
    print("ALL TOY TESTS PASSED")


if __name__ == "__main__":
    main()
