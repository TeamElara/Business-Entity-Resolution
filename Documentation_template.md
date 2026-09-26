# Amazon ML Challenge 2026 — Business Entity Resolution

**Team:** Team Elara (Mahatva Goel, Arihant, Ojaswi)

**Status (26 September 2026):** v3 with orphan and reverse-search features is the current validation candidate. Upload #5 scored 0.933 on the public leaderboard at `t=0.85`, but the final pick, final code tag, and exact final output files are pending. Reconcile this draft to the exact selected files before packaging.

## 1. Executive summary

For each deduplicated Source 1 (S1) business, we find zero or more same-entity Source 2/3 (S2/S3) records. The v3 candidate uses multilingual normalization, a wide eight-block internal retrieval union, a LightGBM pruner, and a separate LightGBM matcher that scores **only the final candidate set**. The reported top-six validation macro F0.5 is **0.9739** with 5.05 candidates/S1 and a 0.9938 oracle ceiling. These are **validation**, not public/private leaderboard results. Upload #5 scored **0.933** publicly for v3 at `t=0.85`; the final pick remains pending.

## 2. Methodology

### 2.1 Problem analysis

Training has 2,206,821 S1 and 10,320,219 S2/S3 records from India and the US. Test additionally contains 259,452 France S1s without labels. Names vary by abbreviation, legal form, typo, domain-style text, accent, and Indic script; addresses vary by order, spelling, house number, and missing components. About 2.3–3.7% of noisy-source addresses are blank. TSVs are read as strings with tabs and quoting disabled; empty-match S1s remain in the output. The documented pipeline uses only supplied data—no business lookup service, registry, API, or geocoder.

On 764,025 held-out true pairs, casefolded raw names agree exactly 10.75% of the time versus 35.88% for the later `name_core`. But ranking on core alone slightly reduced US top-10 positive-query hit (98.31% → 97.98%) on a fixed top-50 pool. We therefore retain multiple name and address signals. No true US/India pair has a reliable extracted postcode on **both** sides, so postcode equality is not mandatory for retrieval.

### 2.2 Solution strategy

Retrieval is partitioned by the input country string, which is treated as **open-set**; no country one-hot feature is learned. Shared validation is `crc32(s1_id) % 10 == 0`; every held-out S1 searches the **full** same-country training S2/S3 pool. Evaluation is per-S1 macro F0.5, including true singletons (an empty prediction scores 1.0 for them).

## 3. Candidate generation and blocking

**V3 cascade:** normalization/text preparation → wide stage 1 → pair features → LightGBM pruner → **final candidates** (`candidate_pairs.tsv`) → separate LightGBM matcher on exactly those candidates. Stage 1 unions Arihant's name and address TF-IDF blocks with all six of Ojaswi's blocks: word, consonant skeleton, no-address skeleton, name+address-number, concatenated-name, and name+house-number. The first two use Mahatva's normalized files; Ojaswi's blocks use her text preparation and non-validation-trained Indic token map. All eight are **internal**; the pruned list, not the union, is submitted.

Two pruners trained on separate halves of the fit sample produce averaged probabilities. The candidate setting keeps at most **six** candidates per S1 with pruner probability **≥0.003**. The reported v3 final set averages **5.05 candidates/S1**, with **0.9938 oracle F0.5**. Exact v3 pair recall, zero-candidate count, and test-file statistics are pending. The following v2 measurements are historical comparisons, **not** v3 final-set metrics:

| Historical v2 stage / cutoff | Avg candidates/S1 | True-pair recall | Oracle macro F0.5 | Model macro F0.5 |
| --- | ---: | ---: | ---: | ---: |
| Wide stage 1, unpruned | 84.3 | 0.9845 | 0.9946 | — |
| Final top 8 + p ≥ 0.003 | 5.66 | 0.9807 | 0.9938 | **0.9703** |
| **Final top 6 + p ≥ 0.003 (chosen)** | **4.99** | **0.9643** | **0.9917** | **0.9689** |
| Final top 6 + p ≥ 0.01 | 4.58 | not reported | 0.9915 | not reported |

V2 top six cost 0.0014 validation F0.5 versus top eight but retained 13% fewer candidates. Those v2 country-wise and zero-candidate statistics must not be carried over to v3. The **test** total, mean/median/maximum, zero-candidate count, country coverage and reduction ratio must be measured from the exact selected v3 output file.

Ojaswi independently measured all **six of her own blocks** with the zero-width-joiner fix used in v3: 48.3 stage-1 candidates/S1 (p95 59, max 70), pair recall 0.9849, oracle 0.9953 on 220,907 full-validation S1s, with no S1 left without candidates. US: 46.5 candidates/S1, recall 0.990, oracle 0.997; India: 51.0, recall 0.9773, oracle 0.9929. S2 recall was 0.987 and S3 recall 0.984. This is a standalone retrieval measurement, not the eight-block union or v3 final candidate file. The team uses Arihant's pruner; Ojaswi's alternative pruner is not in v3.

## 4. Matching model

Mahatva's normalizer preserves raw text, cleaned `name_core`, addresses, script, and Devanagari `name_latin` learned from **non-validation** Hindi/Latin pairs with a rule fallback. Ojaswi's text path folds Latin accents, normalizes domains/digit look-alikes, romanizes additional Indic scripts using non-validation token alignments plus a built-in fallback, and builds consonant skeletons. Neither token map uses validation labels.

V3 retains the v2 training framework: non-validation fit sample A for two pruner folds, out-of-fold pruner candidates from A plus fit sample B for the final matcher. V2 used 398,201 S1s and 1.99 million candidate pairs; v3 training-set counts should be taken from the exact v3 run. The pair features cover name/address similarities and ranks, tolerant house/number comparison, block scores and ranks, block overlap, name rarity and token IDF. The final matcher adds per-S1 relative features and v3's `orphan_prob` plus reverse-search rank/gap signals. `orphan_prob` estimates whether a candidate S2/S3 record may be unmatched; reverse search ranks S1s from a candidate's perspective. The test decision uses **0.75** for qualifying pairs and **0.55** for the best-pair fallback, then assigns each S2/S3 record to at most one S1. There is no country one-hot feature.

The earlier v1 model improved US-only → India transfer from v0's ~0.789 to **0.8982**, but a comparable v3 cross-country retraining check has **not yet been reported**. The Indic maps use India non-validation matches, so a US-only model with those maps would not be a pure zero-shot preprocessing experiment. France has no supplied match labels; manual review is qualitative, not a France F0.5 estimate.

## 5. Results and error analysis

| Run | Validation macro F0.5 | US / India validation | Public LB | Test candidates/S1 |
| --- | ---: | ---: | ---: | ---: |
| Upload #1, v0 | 0.9043 | 0.9148 / 0.8886 | 0.886 | 8.00 |
| Upload #2, v1 | 0.9525 | 0.9559 / 0.9473 | **0.919** | 5.45 |
| V2, top six | 0.9689 | 0.9722 / 0.9640 | **0.924 reported** (`t=0.85`) | pending |
| V3 base, top six | 0.9709 | pending | pending | pending |
| V3 + orphan, top six | 0.9731 | pending | pending | pending |
| **V3 + orphan/reverse, top six** | **0.9739 reported at t=0.75** | **0.9759 / 0.9709 reported** | **0.933 at t=0.85 (upload #5)** | **5.37 test** |

Upload #5 passed `validate_submission.py --check-ids`; that check must be repeated on the **exact final files** after the final pick. A diagnostic hybrid using v1 India/US and v0 France scored **0.912**, below v1's 0.919; this suggests v1 also improved France, but is **not** a labeled France F0.5 estimate. The v3 validation and public scores are provisional team results; final-run artifacts still need reconciliation.

Real v1 validation error analysis (not v3): **8,618 false positives**, **37,745** true matches retrieved but rejected by the final model, **1,071** cut by the pruner, and **35,065** absent from stage 1. Frequent false positives are a similar name at a different address or another business at the same address; rejected true matches often have a blank address or a slightly different house number. V2's wider stage reduced missing true pairs to **11,844** before pruning. In a preliminary manual review of v1 France predictions for S1 #101–200, Mahatva marked 242 predicted pairs `ok` and 40 `wrong`, with 18 predicted pairs unresolved and one possible missed pair left unconfirmed. Of the 40 wrong predictions, 23 had a different distinctive name/location and 16 had a different house number despite overlapping names/streets. Arihant's separate review of #1–100 found 255 `ok`, 28 `wrong`, and 3 missed. These are manual diagnostics, not a labeled France F0.5 estimate; v3 error analysis remains pending. No private score is claimed.

## 6. Conclusion

The v3 candidate raises reported validation macro F0.5 to **0.9739**, compared with 0.9689 for v2, at **5.05 final candidates/S1** and a **0.9938 oracle ceiling**. This is promising but does not guarantee a better test or private-leaderboard result. Freeze the selected code and outputs, measure the exact test candidate statistics, verify `--check-ids`, and reconcile this document and final zip to that same run.

## Appendix A. Reproduction and final package

With pinned dependencies and the supplied raw TSVs located via `DATA_RAW`, the current v3 candidate commands are (full environment instructions in `RUN.md`):

```text
python -m src.normalize.run --split all
python -m src.blocking.reverse --split train
python -m src.blocking.reverse --split test
python -m src.blocking.orphan
python -m src.matching.v3 features
python -m src.matching.v3 prune
python -m src.matching.v3 train --tag extra --final-k 6 --p-min 0.003 --extra
python -m src.matching.v3 test --tag extra --t 0.75 --t1 0.55
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir "$DATA_RAW/test" --check-ids
```

`extra` is a local tag, **not yet the frozen final tag**. Arihant confirmed that the reported 0.9739 validation score uses the same decision rules as test: `t=0.75`, `t1=0.55`, with one S2/S3 record assigned to at most one S1. The final zip requires both validated output TSVs, self-contained runnable code and pinned dependencies under `code/business_entity_resolution/`, and this methodology. Every test S1—including France and empty-match entities—must have one output row, with valid, non-duplicated S2/S3 IDs and matches contained in candidates. `docs/matching_v2.md`, `docs/blocking.md`, and `docs/submission_log.md` preserve prior measurements. If a later pipeline wins, **replace all affected commands and figures before packaging**.
