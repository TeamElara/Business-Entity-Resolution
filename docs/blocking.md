# Blocking / candidate generation (`src/blocking/`)

All numbers are on the shared validation split (`crc32(s1_id) % 10 == 0`, 220,907 train S1),
each S1 searched against the **full** same-country train S2+S3 pool, measured with
`src/common/metrics.py::blocking_report`. "Oracle ceiling" is the macro F0.5 a perfect matcher
would reach with the candidate set (precision 1, recall = candidates ∩ truth / truth).

## Where blocking sits in the final pipeline

```
                   per country (labels read from S1, never a fixed list)
 S1 ──┐
      ├─ text preparation (text.py) ──┬─ word      : TF-IDF on "name address"        top 20 ─┐
S2+S3 ┘                               ├─ skel      : TF-IDF on skeleton text         top 20  │
                                      ├─ noaddr    : name skeleton vs pool w/o addr  top 10  ├─ stage 1 (wide, internal)
                                      ├─ namenum   : name + address numbers          top 10  │
                                      ├─ concat    : joined name (char 3-grams) vs
                                      │              domain-like pool names           top 5   │
                                      ├─ namehouse : name + whole house numbers       top 5   │
                                      └─ (matching: name char-4gram, address 1-2gram)       ─┘
                                                           │
                              pruner (src/matching) → final candidate set = candidate_pairs.tsv
                                                           │
                              final model scores exactly that set → matching_results.tsv
```

The blocks of this package form the wide stage 1 together with the two TF-IDF blocks of
`src/matching`. They are exposed as `stage1.BLOCKS` (text column, pool filter, top-k, max_df),
`stage1.prepare()` (all text columns for one source file and country) and `stage1.topk_block()`.
The cut to the final candidate set is done by the matching pruner.

## Text preparation (`text.py`), all country-agnostic

- Lowercase; accents folded on Latin letters only (Indic vowel signs kept); digit look-alikes inside
  words fixed (`5ystems` → `systems`, `Kingst0n` → `kingston`); `www.` / `.com` dropped; ordinal
  suffixes and leading zeros removed from numbers (`00708` → `708`, `65rd` → `65`).
- Non-Latin names: a token map (1,285 tokens) learned from aligned **non-validation** training pairs
  (Latin S1 name vs Hindi / Telugu / Tamil / Bengali / Gujarati / Kannada / Malayalam S2/S3 name of
  the same business, equal token count, majority vote ≥ 60%). Unseen tokens use a rule-based
  romanizer written for the Devanagari layout, which all nine Indic Unicode blocks share. No external
  transliteration data or package. The map covers 92% of non-Latin test tokens.
- Consonant skeleton per token (`motors`, `मोटर्स` → `motars` → `mtrs`) for typo-tolerant matching.

## Why each block exists (miss analysis)

Name-only character 3-gram TF-IDF reached only **0.54** pair recall at top 20 on India: many
businesses share generic names. True matches keep the address (street, house number, city) far more
stable than the name, so the main block is word TF-IDF on name + address (**0.937** recall at top 20).
Of the 47.8k true val pairs this block still missed (6.3 recall points):

| cause | recall pts | block / fix |
|---|---|---|
| candidate name in an Indic script (S1 is always Latin) | 2.1 | token map + romanizer (text.py) |
| candidate has no address | 1.3 | `noaddr`: name skeleton against the no-address pool only |
| domain as the name (`xyz.com`) | 0.8 | web suffix removal, `skel` |
| typos, leet digits, leading zeros, abbreviations | 2.2 | cleaning + `skel` |

Second round (India, after three blocks): 5.6% of true pairs still missing, many with the same name
and house number but a long S1 address that dilutes the cosine. `namenum` (name + only the numbers
of the address) recovers 2.2 points of India recall. Postcodes are rarely present in the raw
addresses, so a postcode block was dropped.

Third round (both countries, after four blocks): 16.9k true pairs (2.2%) still missing. 35% of them
have a candidate name that is one concatenated token (`nikolettamoorerhomes.com` for "Nikoletta
Moorer Homes", `truexchurch.com` for "Truex and Church LLC"), which no word block can match.
`concat` drops legal and filler words (English, Indian and French forms: pvt, ltd, llc, sarl, sas,
and, et, de, …) and spaces from the name and compares character 3-grams against only the pool records
whose name is a single token of 8+ characters (4% of India's pool, 5% of the US pool), so it costs
seconds. Initials as names ("AM" for "Arjun Mechanical") account for only 38 misses and were left out.

Fourth round (after five blocks, 13.1k misses): mostly generic names shared by several businesses in
one city, where the true record ranks below other same-name records. Small address numbers ("52",
"4") are too frequent to survive `max_df`, but the whole house number ("J-52/4", "70/1/1",
"2-1-241/62") is rare. `namehouse` = name + whole house-number tokens taken from the raw address
(`j_52_4`), top 5: India +0.4 points; 45% of India S1 have such numbers, 1% of US S1. Larger top-k
on the existing blocks was also measured (word 30 / namenum 15 / noaddr 20: India +0.5, US +0.2 points
for ~20 more candidates per S1) and not adopted.

Zero-width joiners (U+200C/U+200D) inside Telugu / Kannada / Malayalam / Hindi words used to split one
word into two ("ఎస్‌ఎస్" = SS became "es es"). `clean_expr` now removes them before tokenizing; the
re-learned map has 1,320 tokens and covers 94.7% of non-Latin test tokens (was 91.7%).

## Results (full val)

| stage 1 variant | cands / S1 | pair recall | oracle ceiling | US recall | India recall |
|---|---|---|---|---|---|
| word block only, top 20 (v0) | 20.0 | 0.937 | 0.977 | 0.966 | 0.895 |
| word + skel + noaddr | 36.7 | 0.968 | 0.989 | 0.984 | 0.944 |
| word + skel + noaddr + namenum | 41.5 | 0.978 | 0.993 | 0.986 | 0.966 |
| all five blocks (+ concat) | 46.4 | 0.983 | 0.9945 | 0.990 | 0.972 |
| **all six blocks (+ namehouse)** | **48.3** (p95 59, max 70) | **0.984** | **0.9951** | **0.990** | **0.976** |

By country (6 blocks): US 46.5 cands/S1, recall 0.990, oracle 0.997; India 51.0 cands/S1, recall
0.976, oracle 0.992. By source: S2 recall 0.986, S3 0.983. No val S1 is left without candidates.
Recall is flat across the number of true matches (0.983 to 0.986). (Measured before the zero-width
joiner fix, which only adds recall.)

What each block contributes (6-block union, full val):

| block | pairs / S1 | true pairs found only by this block |
|---|---|---|
| word | 20 | 0.85% |
| namenum | 10 | 0.52% |
| concat | 5 | 0.47% |
| noaddr | 10 | 0.36% |
| skel | 20 | 0.24% |
| namehouse | 5 | 0.16% |

Runtime (8 GB laptop, 8 threads): stage 1 (6 blocks) on all 221k val S1 against the full pools in 21 min,
peak RAM 3.1 GB (the TF-IDF fit is per country; S1 queries are scored in chunks).

## Pruner experiment (not in the final pipeline)

`features.py` / `pruner.py` / `run.py` implement an alternative stage 2: LightGBM on country-agnostic
pair features and an adaptive cutoff per S1 (top 1, then prob ≥ min_prob, at most k_max). On the
three-block stage 1 (val):

| cutoff | cands / S1 | pair recall | oracle ceiling |
|---|---|---|---|
| no pruning | 36.7 | 0.968 | 0.9886 |
| prob ≥ 0.005, max 12 | 6.3 | 0.966 | 0.9880 |
| prob ≥ 0.01, max 10 | 5.6 | 0.964 | 0.9875 |
| prob ≥ 0.02, max 8 | 4.9 | 0.961 | 0.9866 |
| prob ≥ 0.05, max 8 | 4.2 | 0.955 | 0.9847 |

The team kept one pruner (the matching pruner) so that the final model is trained on the same kind
of candidate sets it scores.

## France (unseen country, no labels)

The blocks contain no country-specific rule; France goes through the same code. As a label-free
check, 5,000 random S1 per country were run through all five blocks and we counted "strong"
candidates (name token-set similarity ≥ 90 and address similarity ≥ 70):

| | cands / S1 | S1 with ≥ 1 strong candidate | strong candidates / S1 |
|---|---|---|---|
| US (val) | 44.4 | 96.9% | 3.2 |
| India (val) | 49.1 | 95.6% | 3.0 |
| France (test) | 49.1 | 97.4% | 3.8 |

France pools have a similar share of domain-like names (4.9%) and records without an address (3.0%)
as US/India, so `concat` and `noaddr` apply there too. French legal forms (SARL, SAS, EURL) and the
`rue`/`r` abbreviation are frequent tokens that TF-IDF down-weights anyway; `concat` drops French
legal and filler words (sarl, sas, sa, eurl, sci, snc, et, de, la, le, les, des, du).

To also score the S1 of a split without labels: `python -m src.blocking.stage1 --split test`.

## Raw text vs. the normalized files

The blocks read the raw TSVs and apply `text.py`. We also ran the same five blocks on Mahatva's
normalized columns (`name_latin`, else `name_norm`; `addr_norm`), with the same `text.py` steps on
top, for 20,000 val S1 per country:

| input | India recall | US recall | cands / S1 |
|---|---|---|---|
| **raw + text.py (used)** | **0.9732** | **0.9898** | 49.1 / 44.5 |
| normalized + text.py | 0.9683 | 0.9861 | 49.2 / 44.5 |
| union of both | 0.9744 | 0.9905 | 56.0 / 52.2 |

The largest drop is in `concat` (India alone 0.044 → 0.008): the normalized names change legal words
and spacing, so joined names no longer equal the domain-like candidate names. The union adds only
0.1 point for 7–8 more candidates per S1, so the blocks stay on raw text. The normalized columns
are used by the matching features instead.

## Speed of `prepare()` and the optional cache

`char_trigrams` is vectorized (polars, no Python loop) and `map_tokens` (used by `latinize` and
`skeletonize`) rebuilds the strings from list offsets instead of a join + group-by. Outputs are
unchanged: `python -m src.blocking.prep_bench` runs the old code (c71a977) and the new code on the
same files and compares every column (row count, dtype, order-sensitive hash, exact equality).
All 12 columns are identical on every test file (France, India, US; S1, S2, S3):

| test | old | new | cached read |
|---|---|---|---|
| India (S1+S2+S3, 5.5M records) | 136 s | 85 s | 1 s |
| US (4.5M records) | 58 s | 45 s | 1 s |
| France (1.7M records) | 62 s | 59 s | 1 s |

Set `BLOCKING_PREP_CACHE=data/cand/prep_cache` to cache `prepare()` per (split, source, country);
later runs read the parquet back (identical frame). The cache key covers `stage1.py`, `text.py`, the
token map and the raw file, so any change prepares again. `prepare()` is a few minutes of a ~2 h
matching run, so this matters mostly for repeated runs.
