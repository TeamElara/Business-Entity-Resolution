# Phase 09 — held-out normalization value and top-k rank test

Run on 25 Sep 2026 with `python -m src.normalize.evaluate` after generating
the three train Parquet files. The machine-local, gitignored full JSON output is
`data/eda/phase09_metrics.json`. The evaluation code is
`src/normalize/evaluate.py`; no validation labels are used to fit retrieval or
to alter normalization rules.

## Population and definitions

- Team validation split: `crc32(s1_id) % 10 == 0`.
- 220,907 validation S1 records; 208,583 have at least one true match.
- 764,025 true S1-to-S2/S3 pairs, all joined to normalized source IDs
  without missing pair records.
- Exact equality excludes empty names. Raw casefold equality ignores case but
  not punctuation/legal forms. Raw token Jaccard applies the same basic Unicode
  and punctuation tokenization to both names; core Jaccard uses `name_core`.
  Jaccard counts **unique** tokens. This makes the token comparison about
  abbreviation/legal cleaning, not punctuation tokenization alone.
- No true training pair has a reliable postcode on **both** sides, so
  postcode-equality rate is **N/A (denominator zero)**, not 0%.

| True-pair group | Pairs | Raw exact | Raw casefold | `name_core` exact | Raw token Jaccard | Core Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 764,025 | 4.63% | 10.75% | 35.88% | 0.618 | 0.632 |
| India | 306,177 | 2.70% | 6.58% | 28.20% | 0.533 | 0.526 |
| US | 457,848 | 5.92% | 13.53% | 41.02% | 0.676 | 0.702 |
| Cross-script | 54,983 | 0.00% | 0.00% | 0.00% | 0.026 | 0.020 |

Exact equality improves strongly, even against the case-insensitive baseline.
Token overlap is mixed: mean Jaccard decreases slightly in India and for
cross-script pairs. Phase 10 transliteration is needed for the latter; do not
claim the current `name_latin` solves them.

## Fixed-candidate rank experiment

For **every** validation S1, retrieve up to 50 candidates from the full
same-country S2/S3 pool using word TF-IDF on **raw name + raw address**.
The vectorizer uses `min_df=2`, `max_df=0.01`, sublinear TF, and float32, as in
the team's initial TF-IDF blocker. It is fitted on unlabeled candidate text.
The pool has 4,133,346 India and 6,186,873 US records.

Re-rank the **same** retrieved 50 candidates twice with RapidFuzz `WRatio`:
first on casefolded raw names, then on `name_core`. Ties use the fixed TF-IDF
score and candidate ID. This isolates name normalization's ranking effect;
it does **not** measure a separately normalized retrieval index. Top-k query
hit rate is the share of positive S1s with at least one true match in top-k.
Pair recall is the share of all true pairs in top-k.

| Group | Positive S1 | Top-50 retrieval query hit | Top-50 pair recall | Top-5 query hit raw → core | Top-10 query hit raw → core | Top-10 pair recall raw → core |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 208,583 | 99.44% | 95.46% | 93.62% → 93.78% | 95.84% → 96.03% | 78.04% → 79.00% |
| India | 83,503 | 98.86% | 92.00% | 89.87% → 90.43% | 92.15% → 93.11% | 69.35% → 72.90% |
| US | 125,080 | 99.83% | 97.78% | 96.12% → 96.02% | 98.31% → 97.98% | 83.85% → 83.08% |

Overall top-10 query hit improves **+0.19 percentage points** (1,869 S1
wins versus 1,481 losses), and top-10 pair recall improves **+0.96 points**.
India improves **+0.96 points** in top-10 query hit and **+3.55 points** in
top-10 pair recall. US *regresses* **−0.33 points** and **−0.77 points**,
respectively. Thus `name_core` is valuable, but should **not** replace raw-name
evidence unconditionally in the final US ranker. Retaining both raw and core
features is a next modeling hypothesis, not a result of this experiment.

The 50-candidate retrieval ceiling is high but not perfect: 4.54% of true
pairs are absent before name re-ranking and cannot be recovered by it. This is
why the blocker and its candidate count still need their own validation.

### Error inspection (illustrative, not the headline run)

A separate deterministic diagnostic sample of 2,000 validation S1s per
country used the same full S2/S3 pools. It found 26 core top-10 query wins
versus 14 losses in India, and 11 wins versus 17 losses in the US. Example
India win: `Sky Infra Private Limited` to `Sri #skyinfra` moves from raw rank
15 to core rank 1 when the unshared legal words are removed. Example US loss:
`Cure Hair Studio LLC` has true variants at raw ranks 1/6 but core ranks
14/16; stripping the shared `LLC` appears to remove useful separation amid
similar nearby names. This is a plausible mechanism, **not** proof that every
US regression has the same cause. The local diagnostic details are in
`data/eda/phase09_diagnostic_sample.json` and are excluded from Git.

## Handoff and limits

- Ojaswi (blocking): Postcode equality cannot be mandatory for US/India;
  use the measured TF-IDF/address fallback. The fixed-pool top-50 ceiling
  above is **not** the same as final recall at ~8–15 candidates/S1.
- Arihant (matching): Preserve both raw and normalized name signals;
  normalized-only re-ranking helps India but slightly hurts US. Include
  country/script and address evidence without making country-specific text
  rules a hard dependency.
- Phase 10: 54,983 cross-script true pairs remain at 0% exact name equality;
  measure transliteration on this group without training on validation pairs.
- This is a retrieval/ranking diagnostic, **not** the challenge's final F0.5,
  candidate-pair score, or France test performance. France has no labeled
  training pairs here.

The full run took 543.9 seconds on this machine. Denominators, per-country
totals, and top-5 ≤ top-10 ≤ top-50 containment were checked after the run.
