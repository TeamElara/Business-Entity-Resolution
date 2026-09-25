# Phase 10 — Hindi/Devanagari to Latin

Run on 25 Sep 2026. The generated token map, normalized Parquet files, and
full JSON metrics are local and gitignored under `data/`. Reproduce with
`python -m src.normalize.run --split all` followed by
`python -m src.normalize.evaluate_transliteration --threads 4`.

## Method and leakage boundary

The map is learned from 281,289 **non-validation** true pairs joining a Latin
S1 name to a Devanagari S2/S3 name. Both names receive the shared basic name
cleaning; 281,281 pairs have equal token counts and can vote on aligned
tokens. A Hindi token receives its most common Latin partner if that partner
has at least 60% of its votes. This accepts 150 of 153 distinct Hindi tokens;
the other 3 and unseen tokens use a small code-native Devanagari romanizer.
No external transliteration library or license is needed. The rule for
validation is `crc32(s1_id) % 10 == 0`, and those pairs are excluded from map
training. Validation labels are used only to measure the result.

`name_raw`, `name_norm`, `name_core`, and `script` retain their established
meaning. Only `name_latin` changes for Devanagari-labeled records; other
non-Latin scripts remain empty. The Latin legal suffix is removed from the
transliterated value when recognized, to compare with S1 `name_core`.

## Held-out pair results

Among **31,436** true validation pairs with a Devanagari candidate
(15,542 positive S1 queries):

| Measure | `name_core` | `name_latin` |
|---|---:|---:|
| Exact name equality | 0.00% | 95.52% |
| Mean unique-token Jaccard | 0.0209 | 0.9719 |
| Mean RapidFuzz WRatio | 13.29 | 99.53 |

The learned map directly covers 113,201 of 115,349 Devanagari token
occurrences in these held-out names (98.14%); the rest use the fallback.
None of the resulting `name_latin` values contains Devanagari. Across all six
regenerated Parquet files, all **917,598** Devanagari-labeled rows have a
nonempty Latin name and none retains a non-Latin letter in `name_latin`.
Unrelated `other` scripts remain empty.

This is the Devanagari subset, **not** all 54,983 cross-script true pairs from
Phase 9; other scripts are outside this phase. Exact agreement is especially
high because the provided Hindi vocabulary is small and synthetic. It does
not establish accuracy for arbitrary Indian names or unseen languages.

## Fixed-pool top-10 rank test

For each of the 15,542 validation S1s with a true Devanagari match, retrieve
up to 50 candidates from the **full 4,133,346-row India S2/S3 pool** using the
same raw-name + raw-address TF-IDF settings as Phase 9. Re-rank those **same**
candidates by `name_core` WRatio, then again changing only Devanagari
candidate scoring to `name_latin` WRatio. Ties use TF-IDF score and ID.
Thus the experiment isolates the transliteration field; it does not measure
a new retrieval index or the final matching model.

| Measure | Baseline core | Hindi `name_latin` |
|---|---:|---:|
| Queries with a true Devanagari match in top 5 | 766 / 15,542 = 4.93% | 13,082 / 15,542 = 84.17% |
| Queries with a true Devanagari match in top 10 | 1,215 / 15,542 = 7.82% | 13,284 / 15,542 = 85.47% |
| True Devanagari pairs in top 10 | 1,269 / 31,436 = 4.04% | 24,101 / 31,436 = 76.67% |
| All true pairs in top 10, these queries | 24,160 / 58,660 = 41.19% | 46,144 / 58,660 = 78.66% |

The fixed top-50 retrieval stage contains 24,243 / 31,436 true Devanagari
pairs (77.12%) and reaches 13,346 / 15,542 Hindi-positive queries (85.87%).
The new top-10 query hit rate is near that ceiling; transliteration cannot
recover a true match not retrieved. There are 12,070 query-level top-10 wins
and 1 loss. An illustrative win is `star infotech` versus
`स्टार इंफोटेक प्राइवेट लिमिटेड`, where true candidates move from ranks
49–50 to 3–4. The observed loss is `laxmi systems` versus mixed-script
`लक्ष्मी systems`: `lakshmee systems` moves a true candidate from rank 10
to 11. This spelling variant deserves a model feature or alias treatment,
not an untested global rewrite.

## Handoff and limits

- Ojaswi: `name_latin` is now a strong Hindi-aware signal for blocking/ranking,
  but a 50-candidate diagnostic is not final recall at a short 8–15-candidate
  cutoff. The retrieval ceiling leaves 14.13% of these queries without a true
  Devanagari candidate.
- Arihant: keep raw/core and Latin name similarities as separate model
  features. Use `script` to distinguish absent transliteration from a genuine
  no-match; do not overwrite original-script evidence.
- The fallback is best-effort for rare words and mixed-script spellings; it is
  not a general-purpose multilingual transliterator. France rules and the
  unseen-country test remain Phases 11 and 12.
- These are held-out training-label diagnostics, not public/private
  leaderboard results, final F0.5, or test-set outcomes.

The full Phase 10 evaluation took 76.8 seconds on this machine. A fresh raw-TSV
map rebuild was byte-equivalent at the parsed JSON level to the map used for
the full run. The suite passed with 58 tests at this checkpoint.
