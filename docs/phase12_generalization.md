# Phase 12 — US-to-India generalization and handoff v2

## Question and evaluation boundary

How well does the team's first end-to-end matcher transfer when fitted on US
records but evaluated on India? This is a **diagnostic**, not the submitted
mixed-country model, public leaderboard score, or a guarantee for France.
The full same-country S2/S3 train pools were used for candidate retrieval:
6,186,873 US and 4,133,346 India records. France has no supplied labels.

The experiment reuses Arihant's `src/matching/baseline_v0.py` index,
22 features, LightGBM settings, 30 name and 30 address retrieval neighbors,
top-8 pruning, and the shared per-S1 macro F0.5 scorer. A deterministic
40,000-S1 sample of **US non-validation** training rows supplies model labels
(2,300,047 retrieved training pairs; 132,207 positives). No India validation
labels train the matcher or select its threshold. Early stopping uses 20,000
US validation S1s; threshold selection uses all US validation S1s, so the US
number is a tuned reference, **not** an untouched test estimate. The fixed
v0 thresholds (0.65/0.55) are reported too. The US-selected thresholds
(0.70/0.55) are then frozen for every India comparison. Split membership is
`crc32(s1_id) % 10 == 0`. Candidate pools contain all same-country train
S2/S3 rows, as in the baseline; the fit/query split only applies to S1.

## Results on the complete held-out S1 sets

| Evaluation | S1 | True pairs | Stage-1 mean candidates/S1 | Stage-1 pair recall | Top-8 pair recall | Top-8 oracle F0.5 | Macro F0.5, US-selected thresholds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| US, original baseline | 132,520 | 457,848 | 57.52 | 95.25% | 95.08% | 0.9825 | **0.9177** |
| India, original baseline | 88,387 | 306,177 | 57.99 | 93.42% | 90.08% | 0.9644 | **0.7907** |
| India, built-in Devanagari romanization only | 88,387 | 306,177 | 57.99 | 93.41% | 90.97% | 0.9672 | **0.8086** |
| India, learned Hindi map + built-in fallback | 88,387 | 306,177 | 57.94 | 93.62% | 91.98% | 0.9710 | **0.8376** |

The original US-to-India drop is **12.70 F0.5 percentage points**. Applying
the built-in, code-native Devanagari-to-Latin fallback to candidate names
raises India by **1.79 points** without learning from India matches. Applying
the Phase 10 token map raises it by **4.69 points** over the original and
reduces the US-to-India gap to **8.01 points**. That map was learned from
**India non-validation true matches** (281,289 cross-script pairs in Phase
10). This is useful for the real team pipeline, but **not a pure zero-shot
country-transfer result**. Neither variant used India validation labels.

At Arihant's fixed 0.65/0.55 thresholds, the corresponding US, India
original, India fallback, and India learned-map F0.5 scores were 0.9177,
0.7892, 0.8076, and 0.8354. These are distinct from the mixed-country v0
model's 0.9043 overall validation / 0.886 public leaderboard in
`docs/submission_log.md`; different training sets prevent a direct score
comparison.

For the **31,436 India validation true pairs whose candidate name is
Devanagari** (across 15,542 S1s), stage-1 pair recall is 83.44% original,
83.52% built-in fallback, and 88.10% with the learned map. Top-8 pair
recall is **67.83% → 76.59% → 87.99%**. The large top-8 change identifies a
script-comparison/pruning issue, not just missing initial candidates.
Across all India pairs, stage-1 recall is already 93.42% and the top-8
oracle is 0.9644 even with the original preprocessing. The oracle gap to US
is much smaller than the actual F0.5 gap; downstream score/decision
transfer also needs investigation. We have **not** isolated which model
features or thresholds cause the remaining gap.

## Implementation and recommended team integration

`baseline_v0.prepare` calls `normalize_df` but then rebuilds `name_core`
from `name_norm`, ignoring the existing `name_latin` that Phase 10 produces.
The controlled variant changes only Devanagari `name_core`/`name_key` used
for query and candidate name comparison; all other prepared fields, the US-fit
model, and the decision threshold stay fixed. The logic keys on detected
script, **not country**, so it can apply to a future country's Devanagari
records. Raw names remain available. The two variants distinguish generic
script handling from the benefit of India-supervised non-validation data.

Recommendation for Arihant and Ojaswi: integrate `name_latin` as an
additional script-aware retrieval/feature signal and measure the full
mixed-country validation and candidate-list trade-off before making it the
submitted default. Keep original/core name signals as well: replacing them
wholesale can hurt same-script matching. Recheck US, India, and France
unlabeled formats, and do not use postcodes as a required blocker. The
current evidence supports this matcher-side change; it does **not** support
an arbitrary new country-specific cleaning rule. Model retraining or
calibration on available non-validation countries is a separate matching
experiment. No change was made here to Arihant's owned `src/matching/` code
or the active submission outputs.

## Reproduce

With the raw train TSVs installed locally and project dependencies (including
LightGBM) installed, run from the repo root:

```powershell
.\.venv\Scripts\python.exe -m src.normalize.evaluate_generalization
.\.venv\Scripts\python.exe -m src.normalize.evaluate_generalization --fallback-only --skip-india-baseline --report-name metrics_fallback.json
```

The first command writes `data/eda/phase12/metrics.json`; the second reuses
its US feature/model caches and writes `metrics_fallback.json`. These large
local artifacts, including the fitted US model, are gitignored. Use a fresh
`--out-dir` whenever changing `--fit-s1`, `--val-s1`, or `--pool-limit` so
cached US features/model cannot be mistaken for a new configuration. A
reduced smoke command is `--fit-s1 200 --val-s1 100 --pool-limit 100000
--out-dir data/eda/phase12_smoke`; smoke metrics are **not** the table above.
The full two-pass experiment took about 45 minutes on the local machine.

The unit checks are `tests/test_evaluate_generalization.py`: unchanged
non-Devanagari fields, distinct fallback behavior, and agreement of the
threshold-search scorer with the shared macro F0.5 reference.
