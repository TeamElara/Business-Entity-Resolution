# Reproduce the selected CP2 run and assemble the submission

This is the raw-data-to-output guide for the **final `cp2d` run** as of 27
September 2026: v3 + orphan/reverse features + Ojaswi's S1 no-match score
(`p_zero`) + legal-form agreement. The frozen source commit is `6eb608f`;
the final TSV MD5s are in `Documentation_template.md`. The rebuilt
model tag is `cp2d`. Its validation macro F0.5 was 0.9765 at the
validation-best threshold (0.9759 at upload `t=0.85`; India 0.9739,
US 0.9782). Arihant reports
public LB 0.943 at `t=0.85`, 0.944 at `t=0.90`, and 0.946 with US `t=0.95`
and India/France `t=0.90`. The US 0.965 probe tied at 0.946, so the final
keeps 0.95. The private score is unknown.
The selected test decision uses those country thresholds, with each S2/S3 record
assigned to at most one S1. Do not package until the exact final test files
pass the validator and match the selected leaderboard upload.

## 1. Inputs and environment

Use Python 3.11+ from the repository root. The challenge's raw TSVs are **not**
in Git or the final zip. Set `DATA_RAW` to the supplied `dataset` directory,
which contains `train/` and `test/`. The run writes generated artifacts under
`data/` and the two final TSVs under `output/`.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:DATA_RAW = "C:\path\to\student_resource\dataset"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.common.selftest
.\.venv\Scripts\python.exe -m src.normalize.run --split all
.\.venv\Scripts\python.exe -m src.blocking.reverse --split train
.\.venv\Scripts\python.exe -m src.blocking.reverse --split test
.\.venv\Scripts\python.exe -m src.blocking.orphan
.\.venv\Scripts\python.exe -m src.blocking.orphan --group-by-s1
.\.venv\Scripts\python.exe -m src.blocking.s1_zero
.\.venv\Scripts\python.exe -m src.matching.v3 features
.\.venv\Scripts\python.exe -m src.matching.v3 prune
.\.venv\Scripts\python.exe -m src.matching.v3 train --final-k 6 --extra --cp2 --rounds 12000 --tag cp2d
.\.venv\Scripts\python.exe -m src.matching.v3 test --tag cp2d --t 0.90 --t-country US=0.95
.\.venv\Scripts\python.exe utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir "$env:DATA_RAW\test" --check-ids
```

If `python` is not on PATH on Windows, use the full path to an installed
Python 3.11+ executable for the first `-m venv` command; subsequent commands
use the newly created `.venv\Scripts\python.exe` directly.

On macOS/Linux, use `python3.11 -m venv .venv`, `.venv/bin/python`, and
`export DATA_RAW=/path/to/student_resource/dataset`. LightGBM may require
`libomp` on macOS. Reverse search, feature construction and the full test pass
are long-running; allow several hours and check RAM/disk before starting.

`--extra` reads `data/cand/rev_{train,test}.parquet` and
`orphan_{train,test}.parquet`. `s1_zero.py` additionally needs
`orphan_train_grouped.parquet`, produced by the grouped orphan command, and
the reverse files. It writes `s1_zero_{train,test}.parquet` with `p_zero`.
`--cp2` then reads those two S1 files and derives legal-form agreement from
the normalized records. Check that all these artifacts exist before training;
missing extra features may be filled with nulls rather than causing a hard
failure. The `test` command builds `data/cache/v3_test` from stage 1 and the
pruner, then scores the `cp2d` model and writes both TSVs. If that cache
already exists, `python -m src.matching.v3 rescore --tag cp2d --t 0.90
--t-country US=0.95` repeats only the last step. The flag is on frozen
`main` at `6eb608f` (introduced in `4b8b035`); use the final file hashes
when assembling the zip.
The final matcher scores only the pruned top-six,
probability ≥ 0.003 candidate set, then enforces one record per S1 assignment.

### Blocking artifacts (reverse search, orphan model, S1 no-match model)

The five `src.blocking` commands above build the record- and S1-level inputs of `--extra` / `--cp2`.
Run them in this order; each writes to `data/cand/` (8 GB Mac, 8 threads, peak RAM ~3 GB):

| command | time | output |
|---|---|---|
| `reverse --split train` | 50–95 min | `rev_train.parquet` (103,176,312 rows: top 10 S1 per train S2/S3 record) |
| `reverse --split test` | 25–40 min | `rev_test.parquet` (99,660,618 rows) |
| `orphan` | 11 min | `orphan_{train,test}.parquet` (`rec_id, orphan_prob`; train out-of-fold), `orphan_auc.json` |
| `orphan --group-by-s1` | 8–12 min | `orphan_train_grouped.parquet` (input of `s1_zero`), `orphan_auc_grouped.json` |
| `s1_zero` | 7–10 min | `s1_zero_{train,test}.parquet` (`s1_id, p_zero`), `s1_zero_auc.json` |

The first command also learns `data/cand/script_token_map_v2.json` (Indic token map) from the
non-validation training pairs; a fresh clone re-learns a byte-identical map. Expected OOF AUCs:
orphan 0.9736 and rebuilt s1_zero 0.98397 (see `docs/orphan_model.md`).

Reproducibility: the final files were built with the rank fix of PR #10 (exact score ties broken by
`s1_id`, countries in sorted order); with it the reverse-search files are byte-identical across
machines (`rev_train` e7a85ab348a9…, `rev_test` 6c3efc043cbb…). The orphan and s1_zero LightGBM
models use all cores (`num_threads=0`), so on a machine with another core count their files can
differ slightly (same AUC). The source-only zip does **not** contain generated `data/cand` artifacts
or trained models; an independent raw-data rerun can therefore be methodologically reproducible
without proving byte-identical TSVs. PR #8 (not in the final) pins the auxiliary LightGBM threads
and row order for a more deterministic rebuild. Ojaswi's backup reproduction is pending.

Windows fresh-clone smoke checks on the preceding `main` at `25da840`
(Python 3.12.14) passed the pinned install, **87 tests**, the self-test,
and `--help` for normalization, reverse search, orphan, all five v3
commands, and the validator. A full normalization run wrote train
S1/S2/S3 = 2,206,821 / 5,034,616 / 5,285,603 rows and test S1/S2/S3 =
1,732,544 / 4,887,273 / 5,082,316 rows. Frozen `main` adds the
`--t-country` flag and deterministic ID tie-breaks; these are verified in
source but the full raw-to-output run has not yet been repeated on this
Windows clone. `src.blocking.s1_zero` has no help mode, so importing it
was checked without launching the full rebuild.

## 2. Validate the final files

The selected run's reported test statistics are 5.37 candidates/S1 (France
5.66, India 5.37, US 5.28), 3.20 matches/S1, and 5.5% S1s without a match.
No-candidate S1 counts are France 177, India 1,521, US 1,126. The downloaded
selected pair independently passed the validator with `--check-ids`: both files
have 1,732,544 S1 rows, 2,824 candidate rows are empty, and 95,819 match rows
are empty.

`output/matching_results.tsv` and `output/candidate_pairs.tsv` must each have
one row for every test Source 1 ID, including empty rows and France. Every
matched S2/S3 ID must exist in that S1's candidate list. The validator must
report PASS with `--check-ids`; its result, test candidate statistics, and
public leaderboard upload should be recorded from the same final run.
Frozen `main` at `6eb608f` includes equal-score reverse-search ties broken
by S1 ID and writer ties broken by candidate ID. The final matcher uses a
fixed LightGBM seed and eight threads (`src/matching/v1.py`); the auxiliary
orphan and `s1_zero` models still use all cores. Ojaswi is independently
reproducing the final TSVs from a fresh clone and comparing MD5s; her result
is pending.

## 3. Required zip layout

The organizer's *Business Entity Resolution Challenge* problem statement
specifies this structure (the README inside `code/` is mandatory). `RUN.md`
at the zip root is an additional convenience guide, not a substitute:

```text
TeamElara_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── utils/validate_submission.py
│       ├── README.md
│       ├── RUN.md
│       └── requirements.txt
├── Documentation_template.md
└── RUN.md
```

Copy all code needed for the **final** run, including `src/blocking/`,
`src/matching/`, `src/normalize/`, and `src/common/`. Exclude `.venv/`, raw
challenge data, local training outputs, caches, and any private checklist.
Confirm the package can regenerate both TSVs from the supplied train/test
inputs using only its `code/business_entity_resolution/` directory. The
methodology file must describe the same commit, candidate cutoff, models,
and output files that were actually submitted.

Once the exact selected TSVs are in `output/`, build the zip from the repo
root with one command:

```bash
bash scripts/make_submission.sh output/
```

On Windows Git Bash, set `PYTHON` to the installed Python executable if it
is not on `PATH`, for example
`PYTHON=.venv/Scripts/python.exe bash scripts/make_submission.sh output/`.
The script checks the two TSV headers, includes only tracked code/docs plus
those TSVs, tests zip integrity, and prints the layout and size. It refuses
to overwrite an existing `output/TeamElara_submission.zip`; move a previous
dry-run archive before rebuilding. It does not replace the full validator.
