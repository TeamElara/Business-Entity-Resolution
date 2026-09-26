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

## Results (full val)

| stage 1 variant | cands / S1 | pair recall | oracle ceiling | US recall | India recall |
|---|---|---|---|---|---|
| word block only, top 20 (v0) | 20.0 | 0.937 | 0.977 | 0.966 | 0.895 |
| word + skel + noaddr | 36.7 | 0.968 | 0.989 | 0.984 | 0.944 |
| word + skel + noaddr + namenum | 41.5 | 0.978 | 0.993 | 0.986 | 0.966 |
| **all five blocks (+ concat)** | **46.4** (p95 56, max 65) | **0.983** | **0.9945** | **0.990** | **0.972** |

By country (5 blocks): US 44.5 cands/S1, recall 0.990, oracle 0.997; India 49.1 cands/S1, recall
0.972, oracle 0.991. By source: S2 recall 0.985, S3 0.981. No val S1 is left without candidates.
Recall is flat across the number of true matches (0.981 to 0.985).

What each block contributes (5-block union, full val):

| block | pairs / S1 | recall alone | true pairs found only by this block |
|---|---|---|---|
| word | 20 | 0.954 | 0.98% |
| namenum | 10 | 0.824 | 0.97% |
| concat | 5 | 0.044 | 0.49% |
| noaddr | 10 | 0.039 | 0.38% |
| skel | 20 | 0.929 | 0.26% |

Runtime (8 GB laptop, 8 threads): stage 1 (5 blocks) on all 221k val S1 against the full pools in 19 min,
peak RAM 2.4 GB (the TF-IDF fit is per country; S1 queries are scored in chunks).

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
