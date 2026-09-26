# Matching v2: facts for the methodology document

Measured on the shared validation split (`crc32(s1_id) % 10 == 0`, 220,907 S1, 764,025 true pairs,
searched against the full same-country train S2/S3 pool). Code: `src/matching/v2.py` (reuses
`src/matching/v1.py`, see `docs/matching_v1.md`). Logs: `data/logs/v2_*.log` (local).

## Commands

```bash
python -m src.normalize.run --split all      # Mhtv: data/norm/
python -m src.matching.v2 features           # stage 1 + pair features -> data/feat/v2/{fitA,fitB,val}/ (63 min)
python -m src.matching.v2 prune              # two pruner models + candidate-size trade-off on val (4 min)
cp data/models/v2_pruner_0.txt data/models/v2_pruner_k6_0.txt   # same pruner, tagged for the top-6 run
cp data/models/v2_pruner_1.txt data/models/v2_pruner_k6_1.txt
python -m src.matching.v2 train --tag k6 --final-k 6 --p-min 0.003   # final model + thresholds (10 min)
python -m src.matching.v2 test --tag k6      # all test S1 -> output/*.tsv
```

## What changed from v1

1. **Stage 1 = union of five blocks per country** (all internal; nothing here is submitted):
   - v1: name char-4gram TF-IDF top 30, address word 1-2-gram TF-IDF top 30 (on Mhtv's norm files).
   - Ojaswi (`src/blocking/stage1.py`, her code and settings, commit 3c58950): word TF-IDF on
     "name address" top 20, consonant-skeleton TF-IDF top 20, name skeleton against pool records
     without an address top 10.
2. **New pair features (52 in total)**: Ojaswi's block scores and ranks, number of blocks that found
   the pair, number of pool records sharing the S1's / candidate's match name, rarest and commonest
   name-token IDF.
3. **New per-S1 features (22 in total, inside the candidate set only)**: does the candidate's house
   number agree with the other candidates' (pruner-weighted share, and with the most likely other
   candidate); how many other candidates share its name.
4. **More training data**: pruner on fit sample A (80k non-val S1 per country, two folds, out-of-fold
   for A); final model on A + fit sample B (120k more non-val S1 per country, scored by the pruner
   exactly as val/test are): 398,201 S1, 1.99M candidate pairs.

## Validation

| | v1 (upload #2) | v2, top 8 | **v2, top 6 (submitted)** |
|---|---|---|---|
| stage 1: pairs per S1 / pair recall / oracle | 57.6 / 0.9541 / 0.9833 | 84.3 / 0.9845 / 0.9946 | 84.3 / 0.9845 / 0.9946 |
| final candidate set: **cands per S1** / pair recall / oracle | **4.73** / 0.9527 / 0.9830 | **5.66** / 0.9807 / 0.9938 | **4.99** / 0.9643 / 0.9917 |
| **macro F0.5 (all)** | 0.9525 | 0.9703 | **0.9689** |
| macro F0.5 US / India | 0.9559 / 0.9473 | 0.9736 / 0.9653 | 0.9722 / 0.9640 |
| thresholds t / t1 | 0.75 / 0.60 | 0.70 / 0.50 | 0.70 / 0.55 |
| matches per S1 / S1 predicted empty | 3.16 / 6.4% | 3.28 / 5.9% | 3.24 / 6.0% |

Stage-1 pair recall by country: India 0.9410 (v1) → **0.9720** (v2), US 0.9619 → **0.9929**. Missed
true pairs: 35,065 (v1) → 11,844 (v2).

Final candidate set (top 6, pruner prob >= 0.003) by country: US 5.00 cands/S1, pair recall 0.9720,
oracle 0.9948; India 4.98, pair recall 0.9528, oracle 0.9871. 893 val S1 (0.4%) have no candidate.

**Top 6 vs top 8**: top 8 scores 0.0014 higher on val but keeps 13% more candidates (5.66 vs 4.99).
The organizer ranks smaller candidate sets higher, and we fixed the rule before seeing the result:
use the smaller set when it costs less than 0.002 F0.5. Top 6 it is.

Pruner trade-off on val (v2 pruner, avg cands/S1 → oracle): all 84.3 → 0.9946; top 10 → 0.9944;
top 8 → 0.9940; top 6 → 0.9918; top 8 & prob >= 0.003: 5.66 → 0.9938; top 6 & prob >= 0.003:
4.99 → 0.9917; top 6 & prob >= 0.01: 4.58 → 0.9915. For comparison, Ojaswi's own pruner on her
three blocks reaches 0.9866 at 4.9 cands/S1 (`docs/blocking.md`), so the team uses this pruner.

Most important final-model features (gain): pruner probability, gap to the best candidate, sibling
support (name × address × pruner prob of the other candidates), **sibling house agreement (new)**,
house number found among the other side's numbers, numbers on one side only.

## Public leaderboard so far

| upload | val F0.5 | public LB |
|---|---|---|
| #1 v0 | 0.9043 | 0.886 |
| #2 v1 | 0.9525 | 0.919 |
| probe: v1 India/US + v0 France | – | 0.912 → v1 is also better on France (≈ +0.05 on the France part) |
| #3 v2 | 0.9689 | pending |

The val–LB gap (0.034 for v1) is mostly test India/US being harder than val, not France (see the probe).

## Pending (added when measured)

- Unseen-country check for v2 (train US → India, train India → US).
- Test output: candidates and matches per S1 by country, validator result, public LB.
