# Reverse search and orphan model (record-level signals for the matcher)

Test has ~41% orphan S2/S3 records vs 26% in train (see `docs/test_forensics.md`), so the matcher
needs signals that say "this record probably belongs to no S1" and "this S1 is not this record's
best S1". Both are computed per record, from the inputs of the split only (no labels at test time).

## Reverse search (`src/blocking/reverse.py`)

For every S2/S3 record, its top 10 S1 records of the same country: TF-IDF fitted on the S1 side,
queried with every pool record, on the two strongest stage-1 texts (`word` = name + address,
`namenum` = name + address numbers). The two lists are unioned and ranked by the best of the two
cosines.

Output `data/cand/rev_{split}.parquet`: `rec_id, s1_id, score, rank, s_word, s_namenum, country`.
`rank` is 1..10 (1 = best), unique per record; `score = max(s_word, s_namenum)`; `s_word` /
`s_namenum` are null when the S1 was not in that block's top 10.

| split | rows | records | missing records (no token shared with any S1) | time (8 threads) |
|---|---|---|---|---|
| train | 103,176,312 | 10,318,758 | India 778 (0.02%), US 683 (0.01%) | 96 min |
| test | 99,660,618 | 9,967,741 | France 664 (0.05%), India 740 (0.02%), US 444 (0.01%) | 40 min |

Where the true S1 lands in a train record's list (7,638,365 true pairs):

| country | rank 1 | top 3 | top 10 |
|---|---|---|---|
| India | 86.2% | 90.4% | 93.4% |
| US | 85.4% | 90.0% | 93.4% |
| all | 85.7% | 90.2% | 93.4% |

## Orphan model (`src/blocking/orphan.py`)

`orphan_prob` = probability that an S2/S3 record matches no S1. LightGBM, one model for all
countries, no country feature.

- Label (train only): the record appears in no ground-truth match list (26.0% of train records).
- Features (32): record-only (name/address lengths and token counts, domain-like, phone, ID, DBA,
  junk prefix, legal form, non-Latin script, blank address, compound house number, S3), label-free
  relations to the split's own inputs (share of name tokens seen in the country's S1 names, rarest
  token frequency, exact cleaned name present in S1, pool twins), and reverse-search summaries (best
  score, second best, gap, number of S1 above 0.5 / 0.7 / 0.9, best word / namenum score).
- Train scores are out-of-fold: 5 folds by `crc32(rec_id) % 5`; each record is scored by the model
  that did not see it. Test records are scored by one model fitted on train records. The parquet
  holds the raw model output on both splits.

Output `data/cand/orphan_{split}.parquet` (`rec_id, orphan_prob`, one row per S2/S3 record, all
records present) and `data/cand/orphan_auc.json` (AUC, distributions, feature gains).

| | value |
|---|---|
| OOF AUC, all | 0.9736 (5 folds 0.9736–0.9737) |
| OOF AUC, US / India | 0.9817 / 0.9569 |
| mean OOF prob vs true orphan rate | India 0.261 vs 0.260, US 0.259 vs 0.260 (calibrated) |
| OOF prob, true orphans | median 0.879; 86.4% > 0.5; 64.1% > 0.8 |
| OOF prob, matched records | median 0.005; 5.5% > 0.5; 1.7% > 0.8 |

Top features by gain: best namenum score 37%, best reverse score 18%, address digit count 9%,
name token count 8%, best word score 4%, rarest name token frequency in S1 3%, pool twins (name +
numbers) 3%.

### Test

| country | mean orphan_prob | share > 0.5 | share > 0.8 | orphan share re-estimated by EM |
|---|---|---|---|---|
| France | 0.277 | 28.9% | 14.2% | 28.9% |
| India | 0.359 | 37.6% | 22.2% | 42.1% |
| US | 0.372 | 39.7% | 31.0% | 40.8% |

The model is calibrated to the train orphan rate (26%), so its raw mean on test is pulled towards
26%. Re-estimating the class prior by EM (Saerens et al., 2002) on the test probabilities gives
42.1% (India) and 40.8% (US), which matches the ~41% from the independent exact-name argument in
`docs/test_forensics.md`. This EM share is a diagnostic only (in `orphan_auc.json`); the parquet
keeps the raw model output so train and test features are on the same scale.

France has no training labels (train has only India and US), so its scores come from a model that
never saw France: treat the France numbers with care.

### Leakage check: folds grouped by owner S1

With random folds, an S2 and an S3 record of the same business can sit in different folds, so the
model could recognise a twin. Re-run with folds grouped by owner S1 (all records of one S1 in the
same fold, `crc32(s1_id) % 5`; orphans by `crc32(rec_id) % 5`), `python -m src.blocking.orphan
--group-by-s1`:

| OOF AUC | random folds | grouped by S1 |
|---|---|---|
| all | 0.9736 | 0.9736 |
| US | 0.9817 | 0.9817 |
| India | 0.9569 | 0.9568 |

The two OOF files agree closely (correlation 0.998, mean |diff| 0.011, 1% of records differ by
more than 0.1): no twin leakage (leaves hold at least 500 records). `orphan_train.parquet` is kept;
`orphan_train_grouped.parquet` is the grouped version.

## S1 with no match (`src/blocking/s1_zero.py`)

`p_zero` = probability that an S1 has no match at all (5.58% of train S1, same in India and US).
Same style: LightGBM, no country feature, train S1 scored out-of-fold with folds by
`crc32(s1_id) % 5`, test S1 by one model fitted on train S1. Features: the S1 record (lengths,
digits, legal form, compound house number, S1 name twins, pool records with the same name), the
reverse search seen from the S1 (how many records have it in their top 10 / as their best S1, best
and second-best score, counts above 0.7 / 0.9, best namenum score), and the orphan scores of the
records that pick it as their best S1 (sum of 1 - orphan_prob, lowest orphan_prob, how many below
0.5; train uses the S1-grouped OOF orphan scores).

| | value |
|---|---|
| OOF AUC, all / India / US | 0.9849 / 0.9781 / 0.9885 (5 folds 0.9846–0.9852) |
| p_zero > 0.5 | 122,511 train S1 flagged, precision 0.730, recall 0.726 |
| p_zero > 0.8 | 40,315 flagged, precision 0.877, recall 0.287 |
| p_zero > 0.9 | 11,965 flagged, precision 0.943, recall 0.092 |
| top features | lowest orphan_prob of its rank-1 records 73%, expected matched records 11% |

Test: mean p_zero France 0.054, India 0.051, US 0.057; EM-corrected share of S1 without a match
France 5.2%, India 4.7%, US 5.7%, i.e. about the train rate (5.6%). The extra test orphans do not
come with more S1 without matches, consistent with the forensics (orphans are lone businesses).

Output `data/cand/s1_zero_{split}.parquet` (`s1_id, p_zero`) and `data/cand/s1_zero_auc.json`.

## Run

```
python -m src.blocking.reverse --split train     # ~1.6 h on 8 threads, peak RAM 2.8 GB
python -m src.blocking.reverse --split test      # ~40 min
python -m src.blocking.orphan                    # ~11 min, needs both reverse files
python -m src.blocking.orphan --group-by-s1      # ~8 min, train only (leakage check)
python -m src.blocking.s1_zero                   # ~9 min, needs the grouped orphan file
```
