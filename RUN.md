# Reproduce the CP2 candidate run and assemble the submission (draft)

This is the raw-data-to-output guide for the **CP2 candidate** as of 27
September 2026: v3 + orphan/reverse features + Ojaswi's S1 no-match score
(`p_zero`) + legal-form agreement. `cp2` is a provisional model tag; replace
it with the exact selected tag at the 27 September 4 PM freeze. CP2's reported
validation macro F0.5 is 0.9766 (India 0.9740, US 0.9782). Arihant reports
public LB 0.943 at `t=0.85`, 0.944 at `t=0.90`, and 0.946 with US `t=0.95`
and India/France `t=0.90`. The final pick and private score are pending.
The provisional test decision uses those country thresholds, with each S2/S3 record
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
.\.venv\Scripts\python.exe -m src.matching.v3 train --final-k 6 --extra --cp2 --rounds 12000 --tag cp2
.\.venv\Scripts\python.exe -m src.matching.v3 test --tag cp2 --t 0.85 --t1 0.5
.\.venv\Scripts\python.exe -m src.matching.v3 rescore --tag cp2d --t 0.90 --t-country US=0.95
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
failure. The `test` command first builds the full test candidate cache; the
explicit `rescore` command is Arihant's provisional choice for a `cp2d`
cache, which the earlier `cp2` test step does not produce. The `--t-country`
flag was still being added when this draft was updated. After the freeze,
replace the train/test commands with the exact steps that produced `cp2d`.
The final matcher scores only the pruned top-six,
probability ≥ 0.003 candidate set, then enforces one record per S1 assignment.

Fresh-clone dry runs on Windows (26 Sep): the pinned dependency install
succeeded. On the earlier `main` at `00c5bc7` with Python 3.12.14, the no-data
toy self-test passed. Before normalization, `pytest` reported 85 passed and
one Hindi-map test failure because that test expects a gitignored trained map.
The raw-data `src.normalize.run --split all` command then succeeded and wrote
all six files: train S1/S2/S3 = 2,206,821 / 5,034,616 / 5,285,603 rows;
test S1/S2/S3 = 1,732,544 / 4,887,273 / 5,082,316 rows, with 150 learned
Hindi tokens. The three test row counts match the earlier independent dry
run. On this unpatched `main`, the v3, reverse-search and orphan `--help`
commands fail on Windows with `ModuleNotFoundError: resource`.

Merged PR #3 (`5c36ad5`) makes the Hindi-map test self-contained and guards
the Unix-only `resource` import. Its code, tested against the same pinned dependency set,
passed 87 tests and `src.common.selftest`; the v3, reverse-search and orphan
`--help` commands all succeeded. The full v3 training/test sequence above
still requires a **new clean-clone run on final main** after the model
freeze; do not claim the current smoke check reproduces the final TSVs.

The full raw-to-output v3 training/test sequence is **not yet verified** on
this fresh clone. Model training/test is owned by Arihant; copy only the
**exact selected run's** outputs into the final package. For a short smoke
test on a clean machine, run `python -m src.matching.v3 --help`,
`python -m src.blocking.reverse --help`, and
`python -m src.blocking.orphan --help` after installation and self-test.
`src.blocking.s1_zero` has no help mode; running it starts the full build.

## 2. Validate the final files

`output/matching_results.tsv` and `output/candidate_pairs.tsv` must each have
one row for every test Source 1 ID, including empty rows and France. Every
matched S2/S3 ID must exist in that S1's candidate list. The validator must
report PASS with `--check-ids`; its result, test candidate statistics, and
public leaderboard upload should be recorded from the same final run.
Also confirm that the frozen reverse-search code breaks equal-score ties by
S1 ID, as reported for the team's final retrain, before claiming deterministic
reverse ranks in the methodology.

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
