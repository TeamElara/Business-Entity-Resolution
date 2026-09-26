# Amazon ML Challenge 2026 — Business Entity Resolution

**Team:** Team Elara (Mahatva Goel, Arihant, Ojaswi)

**Document status:** 26 September 2026, verified account of **live upload #1**; not yet reconciled with a final package.

**Confirmed upload:** 25 September 2026, commit `8d7d4c4` (see `docs/submission_log.md`).

## 1. Executive summary

For every Source 1 (S1) business, we search Source 2 and Source 3 (S2/S3)
for zero or more records of the same entity. The confirmed upload uses
same-country TF-IDF name/address retrieval, a LightGBM pair scorer, top-eight
candidate pruning, and precision-oriented thresholds. Its held-out macro
F0.5 is **0.9043** and its public leaderboard score is **0.886**. These are
different populations; the private score is not known. Since upload #1,
normalization and a new wide-blocking stage have been developed, but neither
is claimed here as part of that uploaded result.

## 2. Methodology

### 2.1 Problem analysis

The provided train set contains 2,206,821 S1 records and 10,320,219 S2/S3
records from India and the US. The test set also includes France (259,452
S1 records), which has no training labels. Names vary by legal suffix,
abbreviation, word order, typo, domain-style text, accent, and Indic script;
addresses vary by abbreviation, component order, and missing information.
About 2.3–3.7% of noisy-source addresses are blank. We read TSVs as strings
with tabs and quoting disabled, preserving IDs and empty-list singletons.
No external business registry, API, geocoder, or test-label lookup was used.

On 764,025 held-out true pairs, casefolded raw names agree exactly in 10.75%
of cases; the later `name_core` normalization agrees in 35.88%. But on a
fixed top-50 candidate pool, name-core re-ranking slightly *reduces* US
top-10 positive-query hit (98.31% to 97.98%). It is therefore not safe to
replace all raw-name evidence with cleaned names. No true US/India pair has
a reliable extracted postcode on **both** sides, so postcode equality cannot
be a mandatory blocker. These are normalization diagnostics, **not** upload
#1 model scores.

### 2.2 Solution strategy

**Approach:** per-country retrieval → pair features and ML scoring → short
candidate list → thresholded matches. Country selects a search partition,
not a one-hot model feature; unseen labels, including France, must remain
processable. The shared validation split is `crc32(s1_id) % 10 == 0`, with
the full same-country training S2/S3 pool searchable for validation S1s.
The matching objective is macro per-S1 F0.5, including S1s with no match.

## 3. Candidate generation and blocking

Upload #1 fits a character four-gram TF-IDF index over cleaned candidate
names and a word one/two-gram TF-IDF index over candidate addresses **within
each country**. Each retrieves up to 30 neighbors per S1; their union is the
wide, internal stage (about 58 pairs/S1 in a later like-for-like diagnostic).
LightGBM scores those pairs, then the highest-probability **eight** per S1
are retained as the reported final candidate list. The upload log records
**8.00 candidates/S1** on test. The code comment records a small validation
oracle change between top eight and top ten (0.9792 versus 0.9795), the
reason for preferring the smaller list; the exact trade-off run and its
denominator were not preserved in the submission log.

The following table is a **separate Phase 12 US-only-model diagnostic**, not
upload #1 or the new Ojaswi blocker. It uses the same retrieval/top-eight
recipe on every held-out US/India S1 against the full same-country pools.
Pair recall is the fraction of all true pairs retained; oracle F0.5 assumes
perfect decisions within those candidates.

| Country / stage | Mean candidates/S1 | True-pair recall | Oracle macro F0.5 |
| --- | ---: | ---: | ---: |
| US, wide stage 1 | 57.52 | 95.25% | 0.9828 |
| US, final top 8 | 8.00 | 95.08% | 0.9825 |
| India, wide stage 1 | 57.99 | 93.42% | 0.9752 |
| India, final top 8 | 8.00 | 90.08% | 0.9644 |

The currently uploaded code caps test candidates at eight, so its maximum is
at most eight. The **measured** test median, exact maximum, reduction ratio,
and a full cutoff trade-off table for the final package have not yet been
provided; they must be checked on the actual final candidate file. On main,
Ojaswi has also added experimental word, consonant-skeleton, and missing-
address blocks (`src/blocking/stage1.py`). The team tracker records a
**20,000-validation-S1 sample per country** for their union: 94.5% India
true-pair recall at 38.5 candidates/S1 and 98.4% US recall at 35.4
candidates/S1. These are stage-1 sample figures, not full-validation oracle,
final-pruner, or uploaded-model measurements. No integration into upload #1
is claimed.

**Submission-audit check:** upload #1 applies LightGBM to the wide pool to
select the top eight, while the submitted `candidate_pairs.tsv` contains
only those eight. The organizer's updated wording asks for the exact set
the final model scores. The team has been asked to confirm or adjust this
interpretation before the final package; we do not claim it is resolved.

## 4. Matching model

Upload #1 uses LightGBM binary classification on **22** pair features:
name/address TF-IDF cosine and retrieval ranks; RapidFuzz name token-set,
full, and partial ratios; address token-set/full ratios; shared address
number Jaccard, number counts, house-number/PIN equality; candidate script,
missing-address and S3 indicators; name lengths; and candidate count. The
model is fitted on 40,000 **non-validation** S1s per training country;
validation labels select early stopping and the final thresholds. A pair
is returned when probability is at least **0.65**, or the rank-one pair
alone when its probability is at least **0.55**. Thus empty predictions are
possible and important for true singletons.

Mahatva's later normalization preserves raw and cleaned forms, legal suffix,
address fields, script, and a Devanagari-to-Latin `name_latin` field. A
separate US-only-model test gives India macro F0.5 **0.7907** originally,
**0.8086** using only a built-in Devanagari fallback, and **0.8376** with
the Hindi map learned from India **non-validation** matches. None of these
is the uploaded mixed-country model score or a zero-shot result for the
learned map. The upload #1 matcher does not yet consume `name_latin`.

## 5. Results and error analysis

| Verified result | Value | Scope |
| --- | ---: | --- |
| Macro F0.5 | **0.9043** | Uploaded v0, shared US/India validation |
| US / India macro F0.5 | **0.915 / 0.889** | Same uploaded-v0 validation run |
| Public leaderboard | **0.886** | Upload #1; private score unknown |
| Test candidates/S1 | **8.00** | Upload #1 log; median/reduction unmeasured |
| Test matches/S1; empty-match S1 | **3.05; 6.2%** | Upload #1 log |

Known failure mechanisms from **separate diagnostics**, not counted upload
#1 false-positive/false-negative rates: removing a shared legal suffix can
promote a wrong similarly named business (a precision risk); raw-name
re-ranking can miss Indian same-entity names in Devanagari (a recall risk);
and a true match absent from the wide stage cannot be recovered by any
downstream matcher. The Phase 12 India top-eight pair recall on Devanagari
true pairs is only **67.83%** before script-aware name comparison. A labeled
sample of actual final-model false positives and negatives remains needed.
No France F0.5 is reported because France has no provided labels.

## 6. Conclusion

Upload #1 demonstrates a compact candidate-plus-matcher baseline, but the
public score is lower than validation and the private score is not known.
The next validated improvement should retain both original and normalized
evidence, add script-aware matching, and verify the exact final candidate
set against the organizer's updated rule. Do not infer a France score or
final leaderboard rank from the available evidence.

## Appendix A. Code and reproducibility

For the **confirmed historical upload**, check out commit `8d7d4c4`, install
the pinned `requirements.txt`, point `data/raw` (or `DATA_RAW`) at the
provided dataset, then run these commands from the repository root:

```text
python -m src.matching.baseline_v0 features
python -m src.matching.baseline_v0 train
python -m src.matching.baseline_v0 test
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/raw/test --check-ids
```

The final package must contain `output/matching_results.tsv`,
`output/candidate_pairs.tsv`, runnable `code/business_entity_resolution/`
(including `src/`, README, and pinned dependencies), and this filled-in
methodology. All test S1 IDs, including France and singletons, need one row;
matches must be a subset of valid S2/S3 candidates with no duplicates.
The upload log states that its validator passed with ID checking, but the
historical output files are not in this checkout. Reproduction at the
current branch head is **not** asserted equivalent to commit `8d7d4c4`,
because normalization has since changed. Re-run and validate the actual
final pipeline before packaging.

## Appendix B. Evidence and final-package reconciliation

`docs/eda_notes.md`, `docs/phase09_validation.md`,
`docs/phase10_transliteration.md`, `docs/phase11_france.md`, and
`docs/phase12_generalization.md` contain the diagnostic denominators and
methods. Before calling this the **final** submission document, replace
upload #1 results if a later model is chosen; verify the final output files,
chosen cutoff and full blocker trade-off, candidate median/maximum/reduction,
actual model error sample, exact runnable commands, and candidate-file rule.
These are evidence gates, not assumed results.
