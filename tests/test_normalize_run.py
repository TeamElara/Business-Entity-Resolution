"""Smoke test the actual TSV-to-Parquet runner on a tiny source."""

import sys

import polars as pl
import pyarrow.parquet as pq

from src.normalize import run


def test_normalize_source_contract_and_stats(tmp_path, monkeypatch):
    source = tmp_path / "source.tsv"
    source.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "a\tACME Ltd\t12 Main St, Boston, MA 02108\tUS\n"
        "b\tभारत ट्रेडर्स\tH NO 23, PUNE\tIndia\n"
        "c\tÉcole SARL\t195 Rue de Paris, 59000 LILLE\tFrance\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(run, "source_path", lambda *_: source)
    result = run.normalize_source("train", 1, tmp_path / "norm", batch_size=2)
    dest = tmp_path / "norm" / "train_s1.parquet"
    assert result["rows"] == 3
    assert result["by_country"]["US"]["postcode"] == 1
    assert result["by_country"]["India"]["postcode"] == 0
    assert pq.ParquetFile(dest).metadata.num_rows == 3
    frame = pl.read_parquet(dest)
    assert frame.columns == list(run.CONTRACT)
    assert all(frame.schema[name] == pl.String for name in run.CONTRACT)
    assert frame["entity_id"].to_list() == ["a", "b", "c"]


def test_run_builds_missing_hindi_map_before_sources(tmp_path, monkeypatch):
    output = tmp_path / "norm"
    events = []

    def train(path):
        assert path == output / "hi_latin_map.json"
        events.append("train")
        path.parent.mkdir(parents=True)
        path.write_text('{"tokens": {"स्काई": "sky"}}', encoding="utf-8")
        return {"accepted_token_mappings": 1}

    def source(split, number, output_dir, batch_size):
        assert output_dir == output
        assert batch_size == 100_000
        events.append(f"{split}_s{number}")
        return {"split": split, "source": number, "rows": 0}

    monkeypatch.setattr(run, "train_and_write_map", train)
    monkeypatch.setattr(run, "normalize_source", source)
    monkeypatch.setattr(sys, "argv", ["normalize.run", "--split", "test", "--output-dir", str(output)])
    run.main()
    assert events == ["train", "test_s1", "test_s2", "test_s3"]
