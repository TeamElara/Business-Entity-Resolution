# Normalization EDA Notes

Generated on 25 Sep 2026 from the provided challenge inputs only. No external data or lookup service was
used.

## Method

- Read every TSV column as a string with tab separation and quoting disabled.
- Inspected a deterministic pseudo-random sample of 300 rows for every available
  `(split, source, country)` group: 4,500 records across 15 groups.
- Inspected 200 deterministic true S1-to-S2/S3 matched pairs from the training ground truth.
- Computed the quality counts below over the complete source files, not only the samples.
- Reproduce the local inspection files with `python -m src.normalize.eda`. The generated row samples are
  stored under the gitignored `data/eda/` directory.

## Dataset checks

| Split | Source | Country | Rows | Blank address | Devanagari name | Domain-like name | Junk-prefixed name |
|---|---:|---|---:|---:|---:|---:|---:|
| train | S1 | India | 883,188 | 0.00% | 0.00% | 0.00% | 0.00% |
| train | S1 | US | 1,323,633 | 0.00% | 0.00% | 0.00% | 0.00% |
| train | S2 | India | 2,017,799 | 2.87% | 13.35% | 3.30% | 0.46% |
| train | S2 | US | 3,016,817 | 3.68% | 0.00% | 4.28% | 0.67% |
| train | S3 | India | 2,115,547 | 3.07% | 7.47% | 3.54% | 0.49% |
| train | S3 | US | 3,170,056 | 3.50% | 0.00% | 4.10% | 0.63% |
| test | S1 | France | 259,452 | 0.00% | 0.00% | 0.00% | 0.01% |
| test | S1 | India | 809,986 | 0.00% | 0.00% | 0.00% | 0.00% |
| test | S1 | US | 663,106 | 0.00% | 0.00% | 0.00% | 0.00% |
| test | S2 | France | 703,378 | 3.06% | 0.00% | 3.49% | 0.01% |
| test | S2 | India | 2,312,565 | 2.28% | 13.37% | 2.68% | 0.43% |
| test | S2 | US | 1,871,330 | 2.94% | 0.00% | 3.53% | 0.62% |
| test | S3 | France | 731,615 | 2.94% | 0.00% | 3.43% | 0.01% |
| test | S3 | India | 2,405,000 | 2.46% | 7.53% | 2.87% | 0.46% |
| test | S3 | US | 1,945,701 | 2.84% | 0.00% | 3.39% | 0.62% |

Business names are present in every source row. S1 is materially cleaner than S2/S3: it has no blank
addresses, domain-style names, or script variation in the training data. Missing address handling is
therefore mainly needed for S2/S3.

## True-pair baseline (200 sampled pairs)

| Diagnostic | Result |
|---|---:|
| Country agreement | 100.0% |
| Raw case-insensitive name equality | 8.0% |
| Equality after NFKC/lowercase/punctuation-only cleaning | 18.5% |
| Address equality after the same simple cleaning | 6.5% |
| Mean name token Jaccard | 0.582 |
| Mean name fuzzy ratio | 0.774 |
| Name token Jaccard below 0.25 | 18.5% |
| Name token Jaccard at least 0.75 | 34.5% |

India is harder than the US in this sample: mean cleaned-name token Jaccard is 0.464 for India versus
0.678 for the US. This is consistent with cross-script names and larger address-format variation.

## Observed noise patterns

### Names

- Legal forms vary heavily: `Pvt`/`Private`, `Ltd`/`Limited`, `Inc`, `Corp`, `LLC`, and French forms such
  as `SARL`, `SAS`, `SASU`, `SA`, and `EURL`.
- Some S2/S3 names are domains or handles derived from the core business name. Examples include collapsed
  words followed by `.com`, `www.`, or leading marker characters. Domain-like names occur in roughly
  2.7%-4.3% of noisy-source rows.
- Ampersand versus `and`, punctuation, word concatenation, capitalization, and word reordering are common.
- Indian candidates contain Devanagari and also other Indic scripts such as Kannada, Tamil, and Telugu.
  The contract's `script` value should label Devanagari explicitly and place other non-Latin scripts in
  `other`; later transliteration work must not accidentally delete Indic combining marks.
- A true match can have a completely different trade/DBA name, so addresses must remain an independent
  matching signal rather than merely confirming a name match.

### Addresses

- Components frequently change order while describing the same place.
- `Road`/`Rd`, `Street`/`St`, state names/codes, capitalization, punctuation, building markers, and unit
  notation vary across sources.
- Candidate addresses can be partial: city/state and the house/building number may survive while landmarks
  or locality components disappear. Around 2.3%-3.7% of S2/S3 addresses are fully blank.
- India includes landmark-style text and mixed scripts. House numbers can contain letters, hyphens, and
  slashes, so extraction must not assume a plain integer.
- France uses accented text, French street terms (`rue`, `avenue`, `boulevard`, etc.), French legal forms,
  and different component order. France has no training labels, so its rules must be deterministic and
  generic rather than learned from test outcomes.

## Normalization implications

### Phase 3 implementation checkpoint

The importable `src.normalize.normalize_df` basic cleaner passed 15 tests.
On the saved 4,500-row EDA sample it changed 4,256 names and 4,393 addresses.
On the 200 sampled true pairs, exact raw name equality was 5/200 (2.5%),
case-insensitive equality was 16/200 (8.0%), and Phase 3 normalized equality
was 37/200 (18.5%). Normalized address equality was 13/200 (6.5%). These are
descriptive EDA diagnostics, not held-out validation or candidate-ranking scores.
Raw strings and nulls are preserved; normalized nulls become empty strings.
Combining marks are retained to preserve Indic letters and vowel signs.

1. Preserve raw text alongside every normalized value so that later feature work can compare both.
2. Apply Unicode NFKC, lowercase, safe boundary-aware abbreviation expansion, punctuation-to-space, and
   whitespace collapse. Never perform substring abbreviation replacement.
3. Produce `name_core` without trailing legal forms, but retain the canonical removed form separately.
4. Preserve useful alphanumeric content from domain-like and junk-prefixed names; remove wrappers and
   domain suffixes without treating the entire value as missing.
5. Extract postcode and house number as high-value blocking fields, while keeping a normalized full address
   for partial/word-order-robust similarity.
6. Keep country labels open-set. Country may safely prevent cross-country comparisons in the supplied data,
   but normalization logic must still have generic fallbacks for unseen labels.
7. Evaluate normalization not only by equality/Jaccard but by true-match top-5 and top-10 rank, because the
   final blocker is expected to retain only about 8-15 candidates per S1.
