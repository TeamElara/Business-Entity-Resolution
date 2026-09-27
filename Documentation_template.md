# Amazon ML Challenge 2026 — Business Entity Resolution

**Team:** Team Elara (Mahatva Goel, Arihant, Ojaswi)

**Final selection (27 September 2026):** `cp2d` with US threshold **0.95**, India/France **0.90**, and best-pair fallback **0.5** scored **0.946** on the public leaderboard. The US 0.965 probe also scored 0.946; the team retained 0.95 as the smaller change. Frozen source commit: **`6eb608f`**. Final `matching_results.tsv` MD5: **`8fc18e0a14032ce4223557f7459ee6cc`**. Final `candidate_pairs.tsv` MD5: **`197d4573a6f27523c27b0f6f738882ae`**. The private leaderboard score is unknown.

## 1. Executive summary

For each deduplicated Source 1 (S1) business, we find zero or more same-entity Source 2/3 (S2/S3) records. The selected `cp2d` run uses multilingual normalization, a wide eight-block internal retrieval union, a LightGBM pruner, and a separate LightGBM matcher that scores **only the final candidate set**. Its reported validation macro F0.5 was **0.9765** at validation-best `t=0.75` and **0.9759** at upload threshold `t=0.85` (India 0.9739, US 0.9782); the pre-rebuild CP2 comparison measured 0.9766. The validation candidate-set mean is 5.05/S1, with a 0.9938 oracle ceiling. These are **validation**, not public/private leaderboard results. The preceding v3 upload #5 scored **0.933** publicly; the selected country-threshold upload scored **0.946** publicly. The private score is unknown.

## 2. Methodology

### 2.1 Problem analysis

Training has 2,206,821 S1 and 10,320,219 S2/S3 records from India and the US. Test additionally contains 259,452 France S1s without labels. Names vary by abbreviation, legal form, typo, domain-style text, accent, and Indic script; addresses vary by order, spelling, house number, and missing components. About 2.3–3.7% of noisy-source addresses are blank. TSVs are read as strings with tabs and quoting disabled; empty-match S1s remain in the output. The documented pipeline uses only supplied data—no business lookup service, registry, API, or geocoder.

On 764,025 held-out true pairs, casefolded raw names agree exactly 10.75% of the time versus 35.88% for the later `name_core`. But ranking on core alone slightly reduced US top-10 positive-query hit (98.31% → 97.98%) on a fixed top-50 pool. We therefore retain multiple name and address signals. No true US/India pair has a reliable extracted postcode on **both** sides, so postcode equality is not mandatory for retrieval.

### 2.2 Solution strategy

Retrieval is partitioned by the input country string, which is treated as **open-set**; no country one-hot feature is learned. Shared validation is `crc32(s1_id) % 10 == 0`; every held-out S1 searches the **full** same-country training S2/S3 pool. Evaluation is per-S1 macro F0.5, including true singletons (an empty prediction scores 1.0 for them).

```text
Raw S1 / S2 / S3 TSVs
    ↓ normalize names, addresses, scripts and legal forms
Eight stage-1 retrieval blocks (Ojaswi 6 + Arihant 2)
    ↓ union, deduplicate and build cheap pair features
LightGBM pruner → top 6 per S1, probability ≥ 0.003
    ↓ candidate_pairs.tsv (the exact set scored by the matcher)
Final LightGBM: pair + per-S1 + orphan_prob + reverse rank/gap
               + p_zero + legal-form agreement
    ↓ one S2/S3 record assigned to at most one S1
US threshold 0.95; India/France 0.90; best-pair fallback t1=0.5
    ↓ matching_results.tsv
```

## 3. Candidate generation and blocking

**V3 cascade:** normalization/text preparation → wide stage 1 → pair features → LightGBM pruner → **final candidates** (`candidate_pairs.tsv`) → separate LightGBM matcher on exactly those candidates. Stage 1 unions Arihant's name and address TF-IDF blocks with all six of Ojaswi's blocks: word, consonant skeleton, no-address skeleton, name+address-number, concatenated-name, and name+house-number. The first two use Mahatva's normalized files; Ojaswi's blocks use her text preparation and non-validation-trained Indic token map. All eight are **internal**; the pruned list, not the union, is submitted.

Two pruners trained on separate halves of the fit sample produce averaged probabilities. The candidate setting keeps at most **six** candidates per S1 with pruner probability **≥0.003**. The reported validation set averages **5.05 candidates/S1**, with **0.9938 oracle F0.5**. The reported test candidate set averages **5.37/S1** (France 5.66, India 5.37, US 5.28). No-candidate S1 counts are France 177, India 1,521, and US 1,126 (2,824 total). The following v2 measurements are historical comparisons, **not** CP2 final-set metrics:

| Historical v2 stage / cutoff | Avg candidates/S1 | True-pair recall | Oracle macro F0.5 | Model macro F0.5 |
| --- | ---: | ---: | ---: | ---: |
| Wide stage 1, unpruned | 84.3 | 0.9845 | 0.9946 | — |
| Final top 8 + p ≥ 0.003 | 5.66 | 0.9807 | 0.9938 | **0.9703** |
| **Final top 6 + p ≥ 0.003 (chosen)** | **4.99** | **0.9643** | **0.9917** | **0.9689** |
| Final top 6 + p ≥ 0.01 | 4.58 | not reported | 0.9915 | not reported |

V2 top six cost 0.0014 validation F0.5 versus top eight but retained 13% fewer candidates. Those v2 country-wise statistics are historical; the CP2 test counts above come from Arihant's reported selected run. A top-eight CP2 candidate experiment gained about 0.0005 validation F0.5 but made the submitted candidate set larger, so top six was retained.

Ojaswi independently measured all **six of her own blocks** with the zero-width-joiner fix used in v3: 48.3 stage-1 candidates/S1 (p95 59, max 70), pair recall 0.9849, oracle 0.9953 on 220,907 full-validation S1s, with no S1 left without candidates. US: 46.5 candidates/S1, recall 0.990, oracle 0.997; India: 51.0, recall 0.9773, oracle 0.9929. S2 recall was 0.987 and S3 recall 0.984. This is a standalone retrieval measurement, not the eight-block union or v3 final candidate file. The team uses Arihant's pruner; Ojaswi's alternative pruner is not in v3.

An optional faster `prepare()` and cache in PR #5 matched all 108 compared text columns across the nine test source/country files, but saves less than 3% of the full test run. It is **not** in the frozen candidate code or the submitted-file generation path; retaining the exact measured stage-1 code takes priority.

## 4. Matching model

Mahatva's normalizer preserves raw text, cleaned `name_core`, addresses, script, and Devanagari `name_latin` learned from **non-validation** Hindi/Latin pairs with a rule fallback. Ojaswi's text path folds Latin accents, normalizes domains/digit look-alikes, romanizes additional Indic scripts using non-validation token alignments plus a built-in fallback, and builds consonant skeletons. Neither token map uses validation labels.

CP2 retains the v2 training framework: non-validation fit sample A for two pruner folds, out-of-fold pruner candidates from A plus fit sample B for the final matcher. V2 used 398,201 S1s and 1.99 million candidate pairs; we do not substitute those counts for CP2. The pair features cover name/address similarities and ranks, tolerant house/number comparison, block scores and ranks, block overlap, name rarity and token IDF. The final matcher adds per-S1 relative features, `orphan_prob` and reverse-search rank/gap signals. `orphan_prob` estimates whether a candidate S2/S3 record may be unmatched; reverse search ranks S1s from a candidate's perspective. CP2 adds Ojaswi's S1 no-match probability (`p_zero`) and legal-form agreement. The original `p_zero` model had S1-grouped out-of-fold AUC **0.98487** (India 0.97809, US 0.98849); the deterministic reverse-rank rebuild used for `cp2d` measured **0.98397** (India 0.97501, US 0.98847). These are auxiliary-model AUCs, not entity-resolution F0.5. Legal-form agreement restores a distinction hidden when SARL, SAS and other legal forms are stripped from `name_core`; the original form is still retained separately. Adding `p_zero` and legal form raised reported validation macro F0.5 from **0.9739 to 0.9766** in the earlier CP2 comparison. The current test decision uses **0.90** for India/France and **0.95** for the US, with best-pair fallback `t1=0.5`, then assigns each S2/S3 record to at most one S1. There is no country one-hot feature.

The validation-best primary threshold was `t=0.75`, while CP2 test uploads probed `t=0.85`, `t=0.90`, then US `t=0.95` with India/France `t=0.90`. Ojaswi's label-free test check (`docs/test_label_free_check.md`) found US predicted matches already reached the full estimated true-match count (3.42 vs 3.41/S1) and a high share of orphan-like matches (5.6% with `orphan_prob > 0.8` vs 1.9% on validation truth). This motivated a stricter US threshold; the per-country probe improved public LB by **0.002**, from 0.944 to 0.946. The selected test output has **3.20 predicted matches/S1** and **5.5% of S1s without a match**. Her earlier orphan estimate put the true test count near **3.47/S1**; its train-data check was within about 0.3%, although transfer to test is uncertain. These public results support a more conservative test threshold, but do not reveal the private score.

The earlier v1 model improved US-only → India transfer from v0's ~0.789 to **0.8982**, but a comparable v3 cross-country retraining check has **not yet been reported**. The Indic maps use India non-validation matches, so a US-only model with those maps would not be a pure zero-shot preprocessing experiment. France has no supplied match labels; manual review is qualitative, not a France F0.5 estimate.

## 5. Results and error analysis

| Upload / decision | Validation macro F0.5 | Public LB |
| --- | ---: | ---: |
| v0 baseline | 0.9043 | 0.886 |
| v1 normalization + two-stage model | 0.9525 | 0.919 |
| v2 top six, `t=0.85` | 0.9692 (0.9689 at prior setting) | 0.924 |
| v3 orphan + reverse, `t=0.85` | 0.9731 (0.9739 at val-best `t=0.75`) | 0.933 |
| `cp2d` + `p_zero` + legal form, `t=0.85` | 0.9759 (0.9765 at validation-best `t=0.75`) | 0.943 |
| Same `cp2d` model, `t=0.90` | 0.9750 | 0.944 |
| **Same `cp2d` model, US 0.95 / India-France 0.90** | No separately reported validation score | **0.946** |
| US 0.965 / India-France 0.90 probe | No separately reported validation score | 0.946; not selected |

The selected run's **test** candidate set averages **5.37 candidates/S1** (France 5.66, India 5.37, US 5.28); **2,824** S1s have no candidates (France 177, India 1,521, US 1,126). The selected matching output averages **3.20 matches/S1**, with **5.5%** of S1s unmatched. The exact Drive pair independently passed `validate_submission.py --check-ids` on all 1,732,544 test S1s: 2,824 empty candidate rows and 95,819 empty match rows. The package must contain those same files. A diagnostic hybrid using v1 India/US and v0 France scored **0.912**, below v1's 0.919; this suggests v1 also improved France, but is **not** a labeled France F0.5 estimate.

Real v1 validation error analysis (not v3): **8,618 false positives**, **37,745** true matches retrieved but rejected by the final model, **1,071** cut by the pruner, and **35,065** absent from stage 1. Frequent false positives are a similar name at a different address or another business at the same address; rejected true matches often have a blank address or a slightly different house number. V2's wider stage reduced missing true pairs to **11,844** before pruning. About **200 France S1s and 1,250 candidate pairs** were manually reviewed by two team members; one half was pre-annotated and then verified, and ambiguous pairs were excluded from the reported precision. The team's reviewed-pair precision on predicted matches rose from about **0.90 for v1 to 0.958 for v3**. This is a sampled manual diagnostic, not a France F0.5 estimate or test ground truth. Mahatva's earlier #101–200 row tally had 242 predicted pairs marked `ok`, 40 `wrong`, 18 unresolved and one possible miss unconfirmed; those preliminary counts should not be substituted for the later combined review denominator. No private score is claimed.

Tested alternatives were not used in the selected 0.946 run. The opt-in `house_no2` / `addr_norm2` feature in PR #6 improved raw house agreement on validation positives, but reduced the earlier CP2 validation macro F0.5 to **0.9763** from **0.9766**. The isolated French `Rte.` → `route` and leading/trailing `EI` rules in closed PR #4 passed label-free checks but needed a full rebuild that did not fit before the freeze. A three-seed ensemble gave no gain. Top-eight candidates added about 0.0005 validation F0.5 but enlarged the submitted candidate set. Additional orphan and `p_zero` decision rules lowered validation F0.5. PR #5's stage-1 speed-up produced identical checked outputs but saves less than 3% of full-run time; it was not needed for the selected run.

## 6. Conclusion

The selected `cp2d` model reached validation macro F0.5 **0.9765** at its validation-best threshold, compared with 0.9739 for v3 + orphan/reverse and 0.9689 for the original v2 setting, at **5.05 validation candidates/S1** and a **0.9938 oracle ceiling**. Country-specific thresholds gave the best reported public score, **0.946**. Public and validation scores do not establish the private-leaderboard result. The submitted zip must contain the exact TSV pair that passed `--check-ids` and produced that public score.

### Reproducibility at freeze

Frozen `main` at `6eb608f` includes `--t-country` (commit `4b8b035`), deterministic reverse-search ranks with exact-score ties broken by S1 ID (commit `b2b4a05`), and writer ordering with score ties broken by candidate ID. The final matcher fixes its LightGBM seed and eight training threads (`src/matching/v1.py`). A fresh clone of `6eb608f`, rescoring from the saved test cache, reproduced **both final TSVs byte-for-byte**. An independent fresh-clone run on a second machine rebuilt the test cache from raw data and reproduced **5,551,112 of 5,551,113 matched pairs**. Its one difference occurred at the top-six candidate cutoff due to floating-point differences between machines, so the complete raw-data rerun was not byte-identical. The auxiliary orphan and `s1_zero` LightGBM models still use all available cores and can also vary slightly across machines; PR #8's follow-up to pin those threads was **not** part of the final run. The source-only zip excludes the saved cache and trained models, so the byte-for-byte rescore check requires those artifacts separately. Mahatva's earlier Windows clean-clone smoke test at `25da840` passed pinned installation, 87 tests, self-test, and applicable CLI help checks, but did not reproduce the full final TSVs. The final-file MD5s above identify the selected output, not the independent raw-data rebuild.

## Appendix A. Reproduction and final package

With pinned dependencies and the supplied raw TSVs located via `DATA_RAW`, the current v3 candidate commands are (full environment instructions in `RUN.md`):

```text
python -m src.normalize.run --split all
python -m src.blocking.reverse --split train
python -m src.blocking.reverse --split test
python -m src.blocking.orphan
python -m src.blocking.orphan --group-by-s1
python -m src.blocking.s1_zero
python -m src.matching.v3 features
python -m src.matching.v3 prune
python -m src.matching.v3 train --final-k 6 --extra --cp2 --rounds 12000 --tag cp2d
python -m src.matching.v3 test --tag cp2d --t 0.90 --t-country US=0.95
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir "$DATA_RAW/test" --check-ids
```

The `test` command builds `data/cache/v3_test` (stage 1 and pruner), then scores `cp2d` and writes both TSVs. With that cache already present, `rescore --tag cp2d --t 0.90 --t-country US=0.95` repeats only the final step. The `--t-country` flag is on frozen `main` at `6eb608f`; the US 0.965 probe tied at 0.946, so the team kept 0.95. The final zip requires both validated output TSVs, self-contained runnable code and pinned dependencies under `code/business_entity_resolution/`, and this methodology. Every test S1—including France and empty-match entities—must have one output row, with valid, non-duplicated S2/S3 IDs and matches contained in candidates. `docs/matching_v2.md`, `docs/blocking.md`, and `docs/submission_log.md` preserve prior measurements.
