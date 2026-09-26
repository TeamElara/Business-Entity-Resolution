# Reproduce the selected run and assemble the submission (draft)

This is a packaging dry-run guide for the selected **v2 / `k6`** pipeline as of
26 September 2026. Replace the model tag, commands, and reported metrics if the
team selects v3/v4. Do not package until the exact final test files pass the
validator and match the leaderboard upload.

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
.\.venv\Scripts\python.exe -m src.matching.v2 features
.\.venv\Scripts\python.exe -m src.matching.v2 prune
Copy-Item data/models/v2_pruner_0.txt data/models/v2_pruner_k6_0.txt
Copy-Item data/models/v2_pruner_1.txt data/models/v2_pruner_k6_1.txt
.\.venv\Scripts\python.exe -m src.matching.v2 train --tag k6 --final-k 6 --p-min 0.003
.\.venv\Scripts\python.exe -m src.matching.v2 test --tag k6
.\.venv\Scripts\python.exe utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir "$env:DATA_RAW\test" --check-ids
```

On macOS/Linux, use `python3.11 -m venv .venv`, `.venv/bin/python`,
`export DATA_RAW=/path/to/student_resource/dataset`, and `cp` for the two model
copies. LightGBM may require `libomp` on macOS. The v2 feature step takes about
an hour on the team's measured machine; check available RAM/disk before running.

The raw-to-output command sequence is not yet verified end-to-end on this
fresh clone. The packaging dry run verifies dependencies, tests, self-test,
and full normalization first. Model training/test is owned by Arihant; copy
only the **exact selected run's** outputs into the final package.

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
