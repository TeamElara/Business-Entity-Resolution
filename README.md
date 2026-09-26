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

## Phases 3-8 normalization

```python
from src.normalize import normalize_df
cleaned = normalize_df(df)
```

Adds the contract columns `name_raw`, `address_raw`, `name_norm`, `name_core`,
`legal_suffix`, `name_latin`, `addr_norm`, `postcode`, `city`, `house_no`, and
`script`, preserving input columns and row order. Raw fields retain original nulls; cleaned missing values
become empty strings. Cleaning applies NFKC, lowercase, ampersand expansion,
punctuation/symbol separation, and whitespace collapse. Unicode combining marks
are retained for Indic scripts.

Run checks with `.\.venv\Scripts\python.exe -m pytest -q`.

Phase 4 expands complete cleaned tokens only. Names: `pvt`, `ltd`, `corp`,
`inc`. Addresses: `rd`, `st`, `ave`, `blvd`, `ln`, `hwy`, `bldg`, `flr`, `apt`.
The maps are field-specific and apply to all countries, including unknown labels.
Ambiguous forms such as `co`, `in`, `sa`, and `no` remain unchanged as token
abbreviations. Dotted legal acronyms such as `L.L.C.` and `S.A.R.L.` become
`llc` and `sarl`. `st` uses the planned street convention, which can
misinterpret Saint; raw text is retained.

Phase 6 uses conservative country-aware address parsing. Five-digit street or
floor numbers are *not* blindly treated as ZIP/postal codes. The source data
rarely includes explicit postcodes, so null is preferable to an unreliable
blocking key. `city` and `house_no` are best-effort fields; inspect
`docs/normalization_report.md` for fill rates and known limitations.

Phase 7 labels names `latin`, `devanagari`, or `other` from characters, not
country labels. Phase 10 fills `name_latin`: accent-stripped `name_core` for
Latin names, and learned Hindi/Devanagari-to-Latin output for Devanagari names.
Other non-Latin scripts remain empty; an empty value is not comparable Latin.

Phase 8 produces all six files in one bounded-memory command. If the local
Hindi map is missing, the command first learns it from **non-validation**
training matches in the raw TSVs; no generated Parquet files are needed for
that step:

```powershell
.\.venv\Scripts\python.exe -m src.normalize.run --split all
```

Use `--split train` or `--split test` for only one split, and `--batch-size N`
to adjust memory use (default 100,000). The runner writes each source atomically
to `data/norm/{split}_s{1,2,3}.parquet`, plus a JSON run/fill-rate report in the
same directory. Every Parquet column is a nullable string. Generated files are
excluded from Git; teammates run the command against their own local dataset.

To retrain the gitignored map explicitly (for example after raw data changes),
run `.\.venv\Scripts\python.exe -m src.normalize.train_transliteration`,
then rerun normalization. The map aligns equal-length Hindi/Latin matched
training name tokens, accepts the majority Latin form at confidence ≥0.60,
and uses a small built-in Unicode romanization fallback for unseen/ambiguous
Devanagari tokens. No external transliteration package or license is used.

Phase 5 removes recognized *trailing* legal forms into `name_core` and records
their canonical value in `legal_suffix`. This covers India, US and French forms
with a generic fallback for unknown country labels. A name with no legal form
gets `legal_suffix = null`; a name made solely of a suffix retains its full
core. The full `name_norm` remains available because stripping suffixes can
make distinct businesses share one core name. French accent and address rules
remain in Phase 11.

## Phase 9 held-out normalization evaluation

After generating the train Parquet files, run:

```powershell
.\.venv\Scripts\python.exe -m src.normalize.evaluate
```

This evaluates **all** held-out validation S1 records against the full
same-country S2/S3 pool. It reports true-pair equality/Jaccard and retrieves
raw name+address TF-IDF top-50 candidates, then re-ranks the same candidates
by raw versus normalized name similarity. The local JSON output is
`data/eda/phase09_metrics.json`; the checked summary and limitations are in
`docs/phase09_validation.md`. Use `--max-queries-per-country N` only for a
deterministic diagnostic sample, not for headline metrics. This is not the
final challenge score or a test-set result.

## Phase 10 Hindi-to-Latin validation

After generating the train Parquet files, run:

```powershell
.\.venv\Scripts\python.exe -m src.normalize.evaluate_transliteration --threads 4
```

This reports held-out Hindi/Latin true-pair agreement and re-ranks the same
full-India raw-text TF-IDF top-50 candidate pool with/without `name_latin` for
Devanagari candidates. The machine-local JSON is
`data/eda/phase10_metrics.json`; measured results, denominator, and caveats
are in `docs/phase10_transliteration.md`. This is a candidate-ranking
diagnostic, not the final matching model or challenge score.

## Phase 11 France rules

France-specific normalization is based on unlabeled France rows in the three
test sources. Run `.\.venv\Scripts\python.exe -m src.normalize.run --split test`
to refresh the affected Parquet files. French addresses expand `R.`/`R` to
`rue` and `CH.`/`CH` to `chemin` only in guarded street positions; `BD`, `AV`,
`PL`, and `IMP` expand as whole tokens. French `St`/`Ste` becomes
`saint`/`sainte` rather than the generic English `street`. French names,
addresses, and city values fold accents and common ligatures; raw values are
preserved. French legal forms are separated when they appear at either the
start or end of a name. A compact `59046Lille Cedex`-style postal segment is
recognized, while five-digit house/CS/BP numbers are not treated as postcodes
without sufficient context. Leading zeros in house numbers remain strings.

The full-data format audit, counts, country-regression check, and limits are
in `docs/phase11_france.md`. France has no supplied match labels, so these
checks do not establish a France match score; postcode coverage remains very
low and cannot be a required blocking key.

## Phase 12 US-to-India generalization diagnostic

`python -m src.normalize.evaluate_generalization` fits Arihant's baseline-v0
matcher on 40,000 US **non-validation** S1 records, then measures US and India
held-out validation with each full same-country S2/S3 pool. It also tests
whether using our Devanagari `name_latin` in name comparisons helps the
otherwise unchanged US model. `--fallback-only --skip-india-baseline
--report-name metrics_fallback.json` separately measures the built-in
romanizer without an India-learned map. Use a new `--out-dir` for different
sample or pool settings because the US feature/model files are cached.

With thresholds chosen only on US validation, macro F0.5 was 0.9177 on US
and 0.7907 on India. India rose to 0.8086 with the built-in fallback and
0.8376 with the Phase 10 Hindi map. **The map was learned from India
non-validation matches**; only the built-in fallback is a no-India-training
comparison. No India validation labels were used to fit or tune the matcher.
See `docs/phase12_generalization.md` for exact denominators, retrieval
recalls/oracle ceilings, limitations, and the handoff to matching/blocking
owners. This diagnostic has not replaced the team's submitted model.

## Confirmed upload #2 and final package

The latest **confirmed uploaded** solution in `docs/submission_log.md` is
matching v1 (validation macro F0.5 0.9525, public leaderboard 0.919).
With pinned dependencies installed and `data/raw` set up as above, its
documented raw-data-to-output commands are:

```powershell
python -m src.normalize.run --split all
python -m src.matching.v1 features
python -m src.matching.v1 prune
cp data/models/v1_pruner_0.txt data/models/v1_pruner_pm_0.txt
cp data/models/v1_pruner_1.txt data/models/v1_pruner_pm_1.txt
python -m src.matching.v1 train --tag pm --p-min 0.003
python -m src.matching.v1 test --tag pm
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/raw/test --check-ids
```

The v1 cascade is wide retrieval → LightGBM pruner → final candidate list
(`candidate_pairs.tsv`) → separate final LightGBM matcher, which scores
exactly that list. `docs/matching_v1.md` has the measured features, cutoff,
validation, error analysis, and transfer checks. The methodology in
`Documentation_template.md` describes upload #2 and distinguishes Ojaswi's
not-yet-uploaded multi-block alternative. If a later model/blocker wins,
replace these commands and verify the **exact final** outputs, candidate
statistics, and document before creating the submission zip.

## Team ownership

- `src/normalize/`: Mahatva
- `src/blocking/`: Ojaswi
- `src/matching/` and `src/common/`: Arihant
