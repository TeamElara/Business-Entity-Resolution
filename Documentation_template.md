# Amazon ML Challenge 2026 — Business Entity Resolution

**Team:** Team Elara (Mahatva Goel, Arihant, Ojaswi)

**Status (26 September 2026):** selected v2 pipeline, **validation measured; v2 test outputs/public leaderboard pending**. The latest confirmed public upload is still #2 (v1). Reconcile this document to the exact final files before packaging.

## 1. Executive summary

For each deduplicated Source 1 (S1) business, we find zero or more same-entity Source 2/3 (S2/S3) records. Our selected v2 pipeline uses multilingual normalization, five complementary *internal* retrieval blocks, a LightGBM pruner, and a separate LightGBM matcher that scores **only the final candidate set**. Across 220,907 held-out US/India S1s, its chosen top-six setting scores macro F0.5 **0.9689** with 4.99 validation candidates/S1. This is **not yet a public or private leaderboard score**; upload #2 (v1) currently has the latest confirmed public score, **0.919**.

## 2. Methodology

### 2.1 Problem analysis

Training has 2,206,821 S1 and 10,320,219 S2/S3 records from India and the US. Test additionally contains 259,452 France S1s without labels. Names vary by abbreviation, legal form, typo, domain-style text, accent, and Indic script; addresses vary by order, spelling, house number, and missing components. About 2.3–3.7% of noisy-source addresses are blank. TSVs are read as strings with tabs and quoting disabled; empty-match S1s remain in the output. The documented pipeline uses only supplied data—no business lookup service, registry, API, or geocoder.

On 764,025 held-out true pairs, casefolded raw names agree exactly 10.75% of the time versus 35.88% for the later `name_core`. But ranking on core alone slightly reduced US top-10 positive-query hit (98.31% → 97.98%) on a fixed top-50 pool. We therefore retain multiple name and address signals. No true US/India pair has a reliable extracted postcode on **both** sides, so postcode equality is not mandatory for retrieval.

### 2.2 Solution strategy

Retrieval is partitioned by the input country string, which is treated as **open-set**; no country one-hot feature is learned. Shared validation is `crc32(s1_id) % 10 == 0`; every held-out S1 searches the **full** same-country training S2/S3 pool. Evaluation is per-S1 macro F0.5, including true singletons (an empty prediction scores 1.0 for them).

## 3. Candidate generation and blocking

**Selected v2 cascade:** normalization/text preparation → wide stage 1 → 52 pair features → LightGBM pruner → **final candidates** (`candidate_pairs.tsv`) → separate LightGBM matcher on exactly those candidates. Wide stage 1 unions five blocks per country: name char-four-gram TF-IDF top 30; address word-one/two-gram TF-IDF top 30; and Ojaswi's word TF-IDF on name+address top 20, consonant-skeleton TF-IDF top 20, and name skeleton against no-address candidates top 10. The two matching blocks use Mahatva's normalized files; the three Ojaswi blocks use her text preparation and non-validation-trained Indic token map. All five are **internal**; the pruned list, not the union, is submitted.

Two pruners trained on separate halves of the fit sample produce averaged probabilities. The chosen setting keeps at most **six** candidates per S1 with pruner probability **≥0.003**. Validation results against 764,025 true pairs:

| v2 stage / cutoff | Avg candidates/S1 | True-pair recall | Oracle macro F0.5 | Model macro F0.5 |
| --- | ---: | ---: | ---: | ---: |
| Wide stage 1, unpruned | 84.3 | 0.9845 | 0.9946 | — |
| Final top 8 + p ≥ 0.003 | 5.66 | 0.9807 | 0.9938 | **0.9703** |
| **Final top 6 + p ≥ 0.003 (chosen)** | **4.99** | **0.9643** | **0.9917** | **0.9689** |
| Final top 6 + p ≥ 0.01 | 4.58 | not reported | 0.9915 | not reported |

Top six costs 0.0014 validation F0.5 versus top eight but retains **13% fewer** candidates. The team stated this choice rule before seeing the comparison: favor the smaller list if the score loss is below 0.002, because candidate-set size also matters in the final review. At top six, the US has 5.00 candidates/S1, pair recall 0.9720 and oracle 0.9948; India has 4.98, pair recall 0.9528 and oracle 0.9871. There are 893 zero-candidate validation S1s (0.4%). The **test** total, mean/median/maximum, zero-candidate count, country coverage and reduction ratio must still be measured from the exact selected output file.

Ojaswi independently measured all **five of her own blocks** (adding `namenum` and `concat` to the three integrated above): on full validation, 46.4 stage-1 candidates/S1, pair recall 0.983, oracle 0.9945. These newer blocks are **not in the measured matching v2 pipeline**; they may be tested in a later version. Her separate pruner reached oracle 0.9866 at 4.9 candidates/S1 on an earlier three-block stage. The team selected Arihant's matching pruner for v2; these alternative figures must not be substituted for v2 or final-file metrics.

## 4. Matching model

Mahatva's normalizer preserves raw text, cleaned `name_core`, addresses, script, and Devanagari `name_latin` learned from **non-validation** Hindi/Latin pairs with a rule fallback. Ojaswi's text path folds Latin accents, normalizes domains/digit look-alikes, romanizes additional Indic scripts using non-validation token alignments plus a built-in fallback, and builds consonant skeletons. Neither token map uses validation labels.

V2 pruner training uses fit sample A (80,000 non-validation S1s per country, two folds). The final matcher uses out-of-fold pruner candidates from A plus sample B (120,000 more non-validation S1s per country): **398,201 S1s and 1.99 million candidate pairs**. The pruner has **52 pair features**: name/address similarities and ranks, tolerant house/number comparison, block scores/ranks, how many blocks found a pair, name rarity and token IDF. The final matcher adds **22 per-S1 relative features**, computed only within the pruned list: probability/rank gaps, sibling name/address agreement, and new sibling house-number support. Thresholds selected on validation are **0.70** (all qualifying pairs) and **0.55** (best pair fallback). There is no country one-hot feature.

The earlier v1 model improved US-only → India transfer from v0's ~0.789 to **0.8982**, but the selected v2 cross-country retraining check has **not yet been reported**. The Indic maps use India non-validation matches, so a US-only model with those maps would not be a pure zero-shot preprocessing experiment. France has no supplied match labels.

## 5. Results and error analysis

| Run | Validation macro F0.5 | US / India validation | Public LB | Test candidates/S1 |
| --- | ---: | ---: | ---: | ---: |
| Upload #1, v0 | 0.9043 | 0.9148 / 0.8886 | 0.886 | 8.00 |
| Upload #2, v1 | 0.9525 | 0.9559 / 0.9473 | **0.919** | 5.45 |
| **Selected v2, top six** | **0.9689** | **0.9722 / 0.9640** | **pending** | **pending** |

Upload #2 passed `validate_submission.py --check-ids` with no match outside the candidate list. A diagnostic hybrid using v1 India/US and v0 France scored **0.912**, below v1's 0.919; this suggests v1 also improved France, but is **not** a labeled France F0.5 estimate. V2 has not yet had that public confirmation.

Real v1 validation error analysis (not v2): **8,618 false positives**, **37,745** true matches retrieved but rejected by the final model, **1,071** cut by the pruner, and **35,065** absent from stage 1. Frequent false positives are a similar name at a different address or another business at the same address; rejected true matches often have a blank address or a slightly different house number. V2's wider stage reduces missing true pairs from 35,065 to **11,844** before pruning, but full v2 false-positive/false-negative analysis is still needed. No private score or France label score is claimed.

## 6. Conclusion

The selected v2 pipeline increases validation pair recall at wide stage 1 to 0.9845, then reduces 84.3 comparisons/S1 to 4.99 final candidates/S1 while retaining a 0.9917 oracle ceiling. It improves validation F0.5 over the confirmed v1 upload, but the **actual v2 test file, validator result and public leaderboard score are pending**. Freeze the selected code and outputs, measure their candidate statistics, then reconcile this document and the final zip to that same run.

## Appendix A. Reproduction and final package

With pinned dependencies and the supplied raw TSVs under `data/raw`, the selected top-six v2 commands are:

```text
python -m src.normalize.run --split all
python -m src.matching.v2 features
python -m src.matching.v2 prune
cp data/models/v2_pruner_0.txt data/models/v2_pruner_k6_0.txt
cp data/models/v2_pruner_1.txt data/models/v2_pruner_k6_1.txt
python -m src.matching.v2 train --tag k6 --final-k 6 --p-min 0.003
python -m src.matching.v2 test --tag k6
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/raw/test --check-ids
```

The final zip requires both validated output TSVs, self-contained runnable code and pinned dependencies under `code/business_entity_resolution/`, and this methodology. Every test S1—including France and empty-match entities—must have one output row, with valid, non-duplicated S2/S3 IDs and matches contained in candidates. `docs/matching_v2.md`, `docs/blocking.md`, and `docs/submission_log.md` are the measurement records. If a later pipeline wins, **replace all affected commands and figures before packaging**.
