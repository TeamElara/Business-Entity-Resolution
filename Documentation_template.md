# Amazon ML Challenge 2026 — Business Entity Resolution

**Team:** Team Elara (Mahatva Goel, Arihant, Ojaswi)

**Status (26 September 2026):** methodology for **confirmed upload #2**. It is not yet the final-package version: the team is comparing pruners and has not frozen the final output files or candidate statistics.

## 1. Executive summary

For each deduplicated Source 1 (S1) business, our system finds zero or more same-entity records from Sources 2 and 3 (S2/S3). The current uploaded solution uses multilingual text normalization, a wide name/address search, a LightGBM candidate pruner, and a separate LightGBM matching model that scores **only the final candidate set**. On 220,907 held-out US/India S1s it reaches macro F0.5 **0.9525**; upload #2 scored **0.919** on the public leaderboard. The private score and France-specific F0.5 are unknown.

## 2. Methodology

### 2.1 Problem analysis

Training contains 2,206,821 S1 records and 10,320,219 S2/S3 records from India and the US. Test adds France (259,452 S1s), absent from training. Names have legal-form and abbreviation changes, typos, domains, accents, and Indic scripts. Addresses have reordered or missing parts; about 2.3–3.7% of noisy-source addresses are blank. We read TSV fields as strings with tabs and quoting disabled and preserve empty-match S1s. The documented pipeline uses the supplied data, not external business databases, APIs, or geocoders.

On 764,025 held-out true pairs, casefolded raw-name equality is 10.75% versus 35.88% for the later normalized `name_core`. That field is **not** a universal replacement: fixed-pool top-10 positive-query hit decreases on US from 98.31% to 97.98% when ranking on core alone. The pipeline therefore keeps multiple name/address signals. No true US/India pair has a reliable extracted postcode on **both** sides, so postcode equality cannot be required for retrieval. These are normalization diagnostics, not leaderboard scores.

### 2.2 Solution strategy

We partition retrieval by the input country string while accepting unseen labels; country is not a one-hot feature. The shared validation rule is `crc32(s1_id) % 10 == 0`. Every validation S1 searches the **full same-country** training S2/S3 pool. The objective is per-S1 macro F0.5, including true singletons, for which predicting an empty list earns 1.0.

## 3. Candidate generation and blocking

**Uploaded v1 cascade:** normalized text → wide stage 1 → pair features → LightGBM pruner → **final candidates** (`candidate_pairs.tsv`) → separate LightGBM matcher on exactly those candidates. The wide stage is the union of name character-four-gram TF-IDF top 30 and address word-one/two-gram TF-IDF top 30 in each country's pool (57.6 pairs/S1 on validation). Names use Mahatva's `name_core`/Devanagari `name_latin` and Ojaswi's additional Indic romanization, text cleaning, and consonant skeletons. Ojaswi's separate multi-block stage 1 is **not** in upload #2.

Two LightGBM pruners trained on separate halves of the fit sample produce averaged probabilities. The uploaded cutoff retains the highest **eight** per S1 whose pruner probability is at least **0.003**. It gives 4.73 final candidates/S1 on validation. The pruner is distinct from the final matcher, resolving upload #1's problem of using its matching model to score the wider pool before writing only top eight.

| Uploaded v1 validation stage/cutoff | Avg candidates/S1 | True-pair recall | Oracle macro F0.5 |
| --- | ---: | ---: | ---: |
| Wide stage 1, unpruned | 57.6 | 0.9541 | 0.9833 |
| Top 10 by pruner | up to 10 | not logged | 0.9832 |
| Top 8 by pruner | up to 8 | not logged | 0.9832 |
| Top 6 by pruner | up to 6 | not logged | 0.9815 |
| Top 8 and probability ≥ 0.003, **uploaded** | **4.73** | **0.9527** | **0.9830** |

The uploaded cutoff loses only 0.0003 oracle F0.5 versus the wide stage while shrinking the list about 12-fold. On test, upload #2 reported **5.45 candidates/S1** overall (India 5.25, US 5.29, France 6.48). Its final test median, maximum, zero-candidate S1 count, total pairs, and country-weighted reduction ratio still need to be measured from the **exact final candidate file**, not inferred from validation or another run.

**Parallel, not-yet-uploaded blocking experiment (Ojaswi):** a word + consonant-skeleton + missing-address three-block union, followed by a separate pruner. Ojaswi reported this full-validation trade-off over all 220,907 S1s; the chosen point is provisional until the team compares it with Arihant's uploaded pruner. The four-block `namenum` addition is in code but has no final full-validation table yet.

| Ojaswi three-block alternative | Avg candidates/S1 | Pair recall | Oracle F0.5 |
| --- | ---: | ---: | ---: |
| Wide stage 1 | 36.7 | 0.968 | 0.9886 |
| Top 1 + p ≥ 0.05, max 8 | 4.2 | 0.955 | 0.9847 |
| Top 1 + p ≥ 0.02, max 8 | 4.9 | 0.961 | 0.9866 |
| Top 1 + p ≥ 0.01, max 10 | 5.6 | 0.964 | 0.9875 |
| **Top 1 + p ≥ 0.005, max 12 (provisional)** | **6.3** | **0.966** | **0.9880** |

That provisional point preserves all but 0.0006 of the wide-stage oracle while reducing candidates about six-fold. These are Ojaswi's reported validation figures, **not** the metrics of upload #2 or the frozen final package.

## 4. Matching model

Upload #2 uses 160,000 **non-validation** training S1s (80,000 per country), yielding 9.2 million wide-stage pairs and 528,798 positives. Its pruner has 40 pair features: TF-IDF similarities/ranks, RapidFuzz name/address scores, shared numbers, tolerant house-number comparisons, name-token differences, script/missing-address/source flags, and related text similarities. The final LightGBM matcher adds **19 per-S1 relative features** computed only within the pruned candidate set, such as pruner-rank gaps and candidate-to-candidate similarity. It is trained on out-of-fold pruner candidates to match validation/test conditions. There is no country one-hot feature. The final decision uses probability **≥0.75**, or the best candidate alone when **≥0.60** and none pass the first threshold; both were selected on validation.

India/US transfer improved but is not solved: with one-country training, v1 scores **0.8982** when trained on US and tested on India, versus v0's ~0.789; India-to-US is **0.9164**. The Indic token maps use India **non-validation** matches, so the US-to-India result is not a pure zero-shot preprocessing test. France has no supplied match labels.

## 5. Results and error analysis

| Confirmed upload | Validation macro F0.5 | US / India validation | Public LB | Test candidates/S1 |
| --- | ---: | ---: | ---: | ---: |
| #1, baseline v0 | 0.9043 | 0.9148 / 0.8886 | 0.886 | 8.00 |
| **#2, matching v1** | **0.9525** | **0.9559 / 0.9473** | **0.919** | **5.45** |

Upload #2 reports 3.23 matches/S1 and 6.0% empty predictions on test. The submission validator **passed with `--check-ids`**, including no match outside the candidate file. A diagnostic upload combining v1 India/US with v0 France scored **0.912**, below v1's 0.919; this suggests v1 is better on France too, but is **not** a labeled France F0.5 estimate or a final-system variant.

On v1 validation, 698,762 predicted pairs include 690,144 correct and **8,618 false positives**. Of true pairs, **37,745** were retrieved but rejected by the final model, **1,071** were pruned, and **35,065** were never found by wide stage 1. Most false positives are extra links next to correct matches: same name/different address, same address/different name, or house-number near misses. Rejected true matches often have one blank address (39%) or a house number one edit away or farther (39%). Remaining wide-stage misses include other Indic-script spellings and typos with shortened addresses. These are measured v1 validation categories, not errors inferred from the public leaderboard.

## 6. Conclusion

Separating wide retrieval, pruning, and final matching reduced validation candidates from 57.6 to 4.73/S1 while retaining an oracle F0.5 of 0.9830; the matched output improved substantially over v0. Before the final zip, the team must choose and freeze the winning stage-1/pruner combination, measure candidate statistics on its actual test file, and verify that code, both TSVs, and this document describe **the same run**. No private-leaderboard rank or France label score is claimed.

## Appendix A. Reproduction and package

For confirmed upload #2, with pinned `requirements.txt` and the supplied raw TSVs available under `data/raw`, run from the repository root:

```text
python -m src.normalize.run --split all
python -m src.matching.v1 features
python -m src.matching.v1 prune
python -m src.matching.v1 train --tag pm --p-min 0.003
python -m src.matching.v1 test --tag pm
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/raw/test --check-ids
```

The final package must include the validated `output/matching_results.tsv` and `output/candidate_pairs.tsv`, runnable code and pinned dependencies under `code/business_entity_resolution/`, and this methodology. Every test S1, including France and true singletons, needs exactly one output row; matches must be valid S2/S3 IDs contained in that S1's final candidate list. The latest logs and methods are in `docs/matching_v1.md`, `docs/submission_log.md`, and Mahatva's phase reports. If a later upload or Ojaswi's alternative blocker wins, **replace the commands and every affected number above before packaging**.
