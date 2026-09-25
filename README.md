# Business Entity Resolution

Team solution for the Amazon ML Challenge 2026.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The challenge dataset is intentionally excluded from Git. For this checkout, `data/raw`
is a local directory junction pointing to the extracted `student_resource/dataset`
folder. The expected inputs are:

```text
data/raw/train/train_source1.tsv
data/raw/train/train_source2.tsv
data/raw/train/train_source3.tsv
data/raw/train/train_ground_truth.tsv
data/raw/test/test_source1.tsv
data/raw/test/test_source2.tsv
data/raw/test/test_source3.tsv
```

All TSV columns must be read as strings with a tab separator and quoting disabled.

## Normalization EDA

Run the reproducible sampling and source-quality report from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.normalize.eda
```

This inspects 300 rows per split/source/country group and 200 true matched pairs. Detailed findings are
maintained in `docs/eda_notes.md`; the raw local inspection files are written under the gitignored
`data/eda/` directory.

## Phase 3 and 4 normalization

```python
from src.normalize import normalize_df
cleaned = normalize_df(df)
```

Adds `name_raw`, `address_raw`, `name_norm`, and `addr_norm`, preserving input
columns and row order. Raw fields retain original nulls; cleaned missing values
become empty strings. Cleaning applies NFKC, lowercase, ampersand expansion,
punctuation/symbol separation, and whitespace collapse. Unicode combining marks
are retained for Indic scripts. Legal forms, address parsing,
transliteration and the final Parquet CLI are subsequent phases.

Run checks with `.\.venv\Scripts\python.exe -m pytest -q`.

Phase 4 expands complete cleaned tokens only. Names: `pvt`, `ltd`, `corp`,
`inc`. Addresses: `rd`, `st`, `ave`, `blvd`, `ln`, `hwy`, `bldg`, `flr`, `apt`.
The maps are field-specific and apply to all countries, including unknown labels.
Ambiguous forms such as `co`, `in`, `sa`, and `no` remain unchanged. `st` uses
the planned street convention, which can misinterpret Saint; raw text is retained.
French-specific rules and legal-suffix removal are not implemented yet.

## Team ownership

- `src/normalize/`: Mahatva
- `src/blocking/`: Ojaswi
- `src/matching/` and `src/common/`: Arihant
