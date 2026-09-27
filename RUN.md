# Reproduce the CP2 candidate run and assemble the submission (draft)

This is the raw-data-to-output guide for the **CP2 candidate** as of 26
September 2026: v3 + orphan/reverse features + Ojaswi's S1 no-match score
(`p_zero`) + legal-form agreement. `cp2` is a provisional model tag; replace
it with the exact selected tag at the 27 September 4 PM freeze. CP2's reported
validation macro F0.5 is 0.9766 (India 0.9740, US 0.9782). Its first public
upload is planned for 9 AM; no public CP2 score or final pick is claimed here.
The provisional test decision is `t=0.85`, `t1=0.5`, with each S2/S3 record
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
.\.venv\Scripts\python.exe -m src.matching.v3 rescore --tag cp2 --t 0.85 --t1 0.5
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
explicit `rescore` command uses that cache to produce the provisional
`t=0.85`, `t1=0.5` files. The final matcher scores only the pruned top-six,
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
orphan 0.9736, s1_zero 0.984–0.985 (see `docs/orphan_model.md`).

Reproducibility: the final files were built with the rank fix of PR #10 (exact score ties broken by
`s1_id`, countries in sorted order); with it the reverse-search files are byte-identical across
machines (`rev_train` e7a85ab348a9…, `rev_test` 6c3efc043cbb…). The orphan and s1_zero LightGBM
models use all cores (`num_threads=0`), so on a machine with another core count their files can
differ slightly (same AUC). To reproduce the submitted file exactly, use the shipped `data/cand`
files (`orphan_test`, `s1_zero_test`, `rev_test`) together with the final models. PR #8 (not in the
final) pins the LightGBM threads and the row order for fully deterministic rebuilds.

Fresh-clone dry runs on Windows (26 Sep): the pinned dependency install
succeeded. On current `main` at `00c5bc7` with Python 3.12.14, the no-data
toy self-test passed. Before normalization, `pytest` reported 85 passed and
one Hindi-map test failure because that test expects a gitignored trained map.
The raw-data `src.normalize.run --split all` command then succeeded and wrote
all six files: train S1/S2/S3 = 2,206,821 / 5,034,616 / 5,285,603 rows;
test S1/S2/S3 = 1,732,544 / 4,887,273 / 5,082,316 rows, with 150 learned
Hindi tokens. The three test row counts match the earlier independent dry
run. On this unpatched `main`, the v3, reverse-search and orphan `--help`
commands fail on Windows with `ModuleNotFoundError: resource`.

PR #3 makes the Hindi-map test self-contained and guards the Unix-only
`resource` import. Its code, tested against the same pinned dependency set,
passed 87 tests and `src.common.selftest`; the v3, reverse-search and orphan
`--help` commands all succeeded. The full v3 training/test sequence above
still requires a **new clean-clone run on final main** after merge and model
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
