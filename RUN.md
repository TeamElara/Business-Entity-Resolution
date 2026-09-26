# Reproduce the v3 candidate run and assemble the submission (draft)

This is the raw-data-to-output guide for the **v3 + orphan/reverse features**
candidate as of 26 September 2026. `extra` is a reproducible local model tag;
replace it with the exact selected tag at the final freeze. Public leaderboard
score and final pick are pending. Do not package until the exact final test
files pass the validator and match the leaderboard upload.

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
.\.venv\Scripts\python.exe -m src.matching.v3 features
.\.venv\Scripts\python.exe -m src.matching.v3 prune
.\.venv\Scripts\python.exe -m src.matching.v3 train --tag extra --final-k 6 --p-min 0.003 --extra
.\.venv\Scripts\python.exe -m src.matching.v3 test --tag extra --t 0.75 --t1 0.55
.\.venv\Scripts\python.exe utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir "$env:DATA_RAW\test" --check-ids
```

On macOS/Linux, use `python3.11 -m venv .venv`, `.venv/bin/python`, and
`export DATA_RAW=/path/to/student_resource/dataset`. LightGBM may require
`libomp` on macOS. Reverse search, feature construction and the full test pass
are long-running; allow several hours and check RAM/disk before starting.

The `--extra` training option reads `data/cand/rev_train.parquet` and
`data/cand/orphan_train.parquet`. Test scoring reads their `test` counterparts.
If either is absent, v3 fills that feature with nulls, so a run can complete
but will **not** reproduce the reported 0.9739 validation candidate. Check
that all four files exist before training. The final matcher scores only the
pruned top-six, probability ≥ 0.003 candidate set; test decisions then enforce
one record assigned to at most one S1.

Earlier fresh-clone dry run on Windows / Python 3.13 (`main` at `dc87f81`, 26 Sep):
the pinned dependency install succeeded; `src.common.selftest` passed; all
six normalization Parquets were generated from the supplied raw data. The
three test-file row counts, per-country statistics and 150-token Hindi map
match Mahatva's existing `normalization_report.json`. `pytest` initially had
**one failure** because a test required the gitignored Hindi map before it
was generated; after normalization, 86 tests passed. PR #3 makes that test
self-contained. On the current PR branch, 87 tests and `src.common.selftest`
pass on Windows, and the v3, reverse-search and orphan `--help` checks all
return successfully. `python -m src.matching.v2 --help` on clean `main` fails with
`ModuleNotFoundError: resource` on Windows; PR #3 also fixes that import and
passes its smoke test. The full v3 commands above still require a **new
clean-clone run on final main**; merge/retest these fixes before packaging.

The full raw-to-output v3 training/test sequence is **not yet verified** on
this fresh clone. Model training/test is owned by Arihant; copy only the
**exact selected run's** outputs into the final package. For a short smoke
test on a clean machine, run `python -m src.matching.v3 --help`,
`python -m src.blocking.reverse --help`, and
`python -m src.blocking.orphan --help` after installation and self-test.

## 2. Validate the final files

`output/matching_results.tsv` and `output/candidate_pairs.tsv` must each have
one row for every test Source 1 ID, including empty rows and France. Every
matched S2/S3 ID must exist in that S1's candidate list. The validator must
report PASS with `--check-ids`; its result, test candidate statistics, and
public leaderboard upload should be recorded from the same final run.

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
