# Matching baseline v0 (upload #1): facts for the methodology document

Everything below is measured, not estimated. Code: `src/matching/baseline_v0.py`
(commit 8d7d4c4). Logs: `data/logs/v0_*.log` (local, gitignored).

## Commands (raw data -> output)

```bash
python -m src.matching.baseline_v0 features   # stage 1 + pair features for the train sample and val -> data/feat/
python -m src.matching.baseline_v0 train      # LightGBM + threshold grid on val -> data/models/
python -m src.matching.baseline_v0 test       # all test S1 -> output/matching_results.tsv, output/candidate_pairs.tsv
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir data/raw/test --check-ids
```

Runtime on an Apple M5 (10 cores, 16 GB): features 16 min, train 3 min, test 66 min.

## Pipeline

Per country, for whatever country labels the S1 file contains (no hard-coded list):

1. **Cleaning**: `src.normalize.normalize_df` (Mhtv, early version) plus a crude legal-form strip.
2. **Stage 1 (wide, internal)**: union of
   - name: TF-IDF char 4-grams of the space-free name core, `max_df=0.02`, top 30 by cosine;
   - address: TF-IDF word 1-2-grams, `max_df=0.005`, top 30 by cosine;
   over that country's S2+S3 records. About 58 pairs per S1.
3. **Model**: LightGBM binary classifier (600 trees, 63 leaves, lr 0.05) on 22 pair features:
   TF-IDF cosines and ranks, rapidfuzz name/address similarities, digit-token Jaccard,
   house-number and postcode agreement, script and missing-address flags, S3 flag, lengths,
   candidate count. No country feature.
   Training data: 80,000 non-val train S1 (40,000 per country), 4.6M pairs, 262k positives.
4. **Candidate set**: top 8 per S1 by model probability -> `candidate_pairs.tsv`.
5. **Decision**: keep candidates with prob >= 0.65; if none pass, keep the top-1 if prob >= 0.55.
   Thresholds grid-searched on val.

**Known compliance issue, fixed in the final pipeline**: in v0 the same model scores all ~58
stage-1 pairs and then decides among the top 8, so the model "sees" more pairs than
`candidate_pairs.tsv` contains. The final pipeline separates the two: a cheap pruner selects the
final set (written to `candidate_pairs.tsv`), and the final model scores exactly that set.

## Validation (220,907 val S1, crc32 split, searched in the full train pool)

| | value |
|---|---|
| macro F0.5 (all) | **0.9043** |
| macro F0.5 US / India | 0.9148 / 0.8886 |
| stage 1: pair recall / oracle ceiling / cands per S1 | 0.9452 / 0.9798 / 57.7 |
| final top-8: pair recall / oracle ceiling / cands per S1 | 0.9421 / 0.9792 / 8.0 |
| top-10 / top-5 oracle ceiling | 0.9795 / 0.9720 |
| S1 predicted empty (true singletons 5.6%) | 6.7% |

## Test output (upload #1)

- 1,732,544 rows (every test S1, France included), 8.00 candidates per S1, 3.05 matches per S1,
  6.2% empty. Per country matches/S1: India 3.02, US 3.10, France 3.01.
- `validate_submission.py --check-ids`: **PASS** (9,969,589 valid S2/S3 ids; 0 matches outside the
  candidate file).
- Public leaderboard: **0.886** (25 Sep 22:41 IST).

## Error analysis on val

| | pairs | macro F0.5 if fixed |
|---|---|---|
| predicted pairs / correct | 659,366 / 636,041 | |
| false positives | 23,325 | 0.9315 |
| false negatives rejected by the model (true match in top 8) | 83,731 | 0.9516 |
| false negatives lost in blocking (true match not in top 8) | 44,253 | |

1,961 of 12,324 val singletons get a wrong match.

**False negatives (model)**: mostly the same name with a slightly different house number, e.g.
`Atlantic Bumrungrad, LLC | 100 Whitestone Drive` vs `10 Whitestone Dr`;
`Novora Therapeutics | 229 Main Street` vs `228 Main Street`;
`Harbor Charities Inc | 1511 Ewing Avenue` vs `001511 EWING AVE`. The v0 house-number feature
counts any difference as a conflict.

**False positives**: mostly a different business at the same address, e.g.
`Baldwin Healthy of Alexander City | 5719 22, Alexander City` vs `Harless Healthy Of Alexander City Co | 5719-D 22`;
`Alliance Heart Network | 676 Seneca Avenue` vs `Alliance Nexquo Network LP | 676 SENECA AVENUE`;
and cross-script names the model cannot read, e.g. `Innovative Ventures Private Limited` vs
`इनोवेटिव ट्रेडिंग प्राइवेट लिमिटेड` ("Innovative Trading").

**False negatives (blocking)**: non-Latin names with partial addresses (e.g.
`Fortune Projects` vs `फॉर्च्यून प्रोजेक्ट्स`, `One Hotel Logistics` vs its Kannada name), and
heavily shortened addresses.

## Unseen-country check (Mhtv, phase 12)

Model trained on 40,000 US non-val S1 only: US val 0.9177, India val 0.7892 (India is 0.8886 when
India is in training). This is the closest proxy we have for France (no training labels); the
public LB gap (0.886 vs val 0.904) is consistent with France scoring around 0.80.
