# Business Entity Resolution

Team solution for the Amazon ML Challenge 2026.

## Local setup

Python 3.11+ (pins in `requirements.txt` install on 3.11 and newer).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
brew install libomp          # macOS only, needed by LightGBM
ln -s /path/to/student_resource/dataset data/raw
```

The challenge dataset is intentionally excluded from Git. For this checkout, `data/raw`
is a local directory junction (Windows) or symlink (macOS/Linux) pointing to the extracted
`student_resource/dataset` folder (or set `DATA_RAW=/path/to/dataset`). The expected inputs are:

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

## Common code (`src/common/`)

`io.py` (readers), `split.py` (`is_val`), `metrics.py` (F0.5, blocking report), `writer.py` (output TSVs).
Run from the repository root.

```python
from src.common import (load_source, load_ground_truth, scan_source, is_val, add_is_val,
                        macro_f05, blocking_report, tradeoff_table, write_outputs)

s1 = load_source("train", 1, country="India")          # all str, "" == null
truth = add_is_val(load_ground_truth())                # s1_id, matched_ids (list), is_val
val = truth.filter("is_val")
blocking_report(cand_df, val)                          # cand_df: s1_id, cand_id, sources, block_score, block_rank
tradeoff_table(cand_df, val)                           # avg cands/S1 vs oracle ceiling vs recall
# one country (8 GB machines): India val S1 vs the full India S2/S3 pool; the reduction ratio
# then uses the India pool automatically
s1_in = load_source("train", 1, country="India", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
blocking_report(cand_df, val.join(s1_in, on="s1_id", how="semi"), s1_country=s1_in)
macro_f05(pred, val)                                   # pred: dict or (s1_id, cand_id) frame
write_outputs(matches, cands, test_s1_ids, "output")   # both TSVs, checks matches ⊆ candidates
```

Checks:

```bash
python -m src.common.selftest     # toy tests, no data needed
python -m src.common.data_check   # row counts, val split, scorer sanity on real data
```

Validate outputs before uploading:

```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/raw/test
```

Organizer updates are in `docs/organizer_updates.md`; uploads are logged in `docs/submission_log.md`.

## Normalization EDA

Run the reproducible sampling and source-quality report from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.normalize.eda
```

This inspects 300 rows per split/source/country group and 200 true matched pairs. Detailed findings are
maintained in `docs/eda_notes.md`; the raw local inspection files are written under the gitignored
`data/eda/` directory.

## Phases 3-5 normalization

```python
from src.normalize import normalize_df
cleaned = normalize_df(df)
```

Adds `name_raw`, `address_raw`, `name_norm`, `name_core`, `legal_suffix`, and `addr_norm`, preserving input
columns and row order. Raw fields retain original nulls; cleaned missing values
become empty strings. Cleaning applies NFKC, lowercase, ampersand expansion,
punctuation/symbol separation, and whitespace collapse. Unicode combining marks
are retained for Indic scripts. Address parsing, transliteration and the final
Parquet CLI are subsequent phases.

Run checks with `.\.venv\Scripts\python.exe -m pytest -q`.

Phase 4 expands complete cleaned tokens only. Names: `pvt`, `ltd`, `corp`,
`inc`. Addresses: `rd`, `st`, `ave`, `blvd`, `ln`, `hwy`, `bldg`, `flr`, `apt`.
The maps are field-specific and apply to all countries, including unknown labels.
Ambiguous forms such as `co`, `in`, `sa`, and `no` remain unchanged as token
abbreviations. Dotted legal acronyms such as `L.L.C.` and `S.A.R.L.` become
`llc` and `sarl`. `st` uses the planned street convention, which can
misinterpret Saint; raw text is retained.

Phase 5 removes recognized *trailing* legal forms into `name_core` and records
their canonical value in `legal_suffix`. This covers India, US and French forms
with a generic fallback for unknown country labels. A name with no legal form
gets `legal_suffix = null`; a name made solely of a suffix retains its full
core. The full `name_norm` remains available because stripping suffixes can
make distinct businesses share one core name. French accent and address rules
remain in Phase 11.

## Team ownership

- `src/normalize/`: Mahatva
- `src/blocking/`: Ojaswi
- `src/matching/` and `src/common/`: Arihant
