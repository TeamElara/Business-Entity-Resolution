# Matching v1: facts for the methodology document

Everything below is measured on the shared validation split (`crc32(s1_id) % 10 == 0`,
220,907 S1, searched against the full same-country train S2/S3 pool). Code:
`src/matching/v1.py`. Logs: `data/logs/v1_*.log` (local, gitignored).

## Commands (norm files -> output)

```bash
python -m src.normalize.run --split all            # Mhtv: data/norm/{train,test}_s{1,2,3}.parquet
python -m src.matching.v1 features                 # stage 1 + pair features, fit sample + val -> data/feat/
python -m src.matching.v1 prune                    # pruner models + candidate-size trade-off on val
python -m src.matching.v1 train --tag pm --p-min 0.003   # final model + thresholds (pruner files copied to *_pm_*)
python -m src.matching.v1 test --tag pm            # all test S1 -> output/*.tsv
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir data/raw/test --check-ids
```

Runtime on an Apple M5 (10 cores, 16 GB): normalization 2 min, features 24 min, pruner 2.5 min,
final model 4 min.

## Pipeline (organizer rule: the final model scores exactly `candidate_pairs.tsv`)

Per country, for whatever country labels the S1 file contains:

1. **Inputs**: Mhtv's normalized files. Match name = `name_core` (Latin script), `name_latin`
   (Devanagari, learned Hindi token map + rule fallback) or Ojaswi's `latinize` (other Indic scripts:
   learned token map + rule-based romanizer). Ojaswi's `clean_expr` on names and addresses (leading
   zeros, digit look-alikes, web prefixes); crude legal-word strip; consonant skeletons (`skeletonize`).
2. **Stage 1 (wide, internal)**: name char-4gram TF-IDF top 30 ∪ address word 1-2-gram TF-IDF top 30
   within the country's S2+S3 pool. 57.6 pairs per S1.
3. **Pruner**: LightGBM (400 trees, 63 leaves) on 40 pair features of every stage-1 pair; two models
   trained on the two S1 halves of the fit sample, averaged. Keep the top 8 per S1 with pruner
   probability >= 0.003. **This is the final candidate set, written to `candidate_pairs.tsv`.**
4. **Final model**: LightGBM on the 40 pair features + 19 per-S1 relative features computed only
   inside that candidate set. Trained on out-of-fold pruner output of the fit sample, so its
   training candidates look like val/test candidates. It scores every candidate and nothing else.
5. **Decision**: prob >= 0.75; if none pass, the best candidate alone if prob >= 0.60 (grid on val).

Training data: 160,000 non-val train S1 (80,000 per country), 9.2M stage-1 pairs, 528,798 positives.
No country feature; France is processed by the same code.

### Features

- v0's 22 (TF-IDF cosines/ranks, RapidFuzz name/address similarities, number Jaccard, house number,
  6-digit code, script / missing-address / S3 flags, lengths, stage-1 count), now on the new text.
- **Tolerant house numbers**: leading zeros removed; `house_tol` = 1 equal, 0.5 one digit
  inserted/deleted/changed (`100` vs `10`, `229` vs `228`), -1 different; Levenshtein distance;
  full house-string equality (`g-3/571`); house number found anywhere among the other side's numbers;
  numbers present on one side only.
- **Name differences**: skeleton tokens present on one side only (count and max IDF within the
  country pool) and the similarity of those leftover tokens (typo vs genuinely different word);
  token-sort ratio, skeleton ratio, first-token Jaro-Winkler / equality; city similarity.
- **Per-S1 relative (final model only, inside the candidate set)**: pruner probability, rank and gap
  to the best candidate; set size; gaps to the S1's best name/address/number/house scores; name and
  address score ranks; number of candidates with a high name score / equal house number; similarity of
  each candidate to the other candidates of the same S1 (max and pruner-weighted mean for name and
  address, max of name × address × sibling pruner probability).

## Validation

| | v0 (upload #1) | v1 |
|---|---|---|
| stage 1: pair recall / oracle / cands per S1 | 0.9452 / 0.9798 / 57.7 | 0.9541 / 0.9833 / 57.6 |
| final candidate set: pair recall / oracle / **cands per S1** | 0.9421 / 0.9792 / **8.00** | 0.9527 / 0.9830 / **4.73** |
| **macro F0.5 (all)** | 0.9043 | **0.9525** |
| macro F0.5 US / India | 0.9148 / 0.8886 | 0.9559 / 0.9473 |
| S1 predicted empty (true singletons 5.6%) | 6.7% | 6.4% |

Final candidate set per country: US 4.73 cands/S1, pair recall 0.9616, oracle 0.9867; India 4.73
cands/S1, pair recall 0.9395, oracle 0.9776 (median 5 in both). 1,841 val S1 (0.8%) have an empty
candidate set; true singletons get 1.92 candidates on average.

Pruner candidate-size trade-off on val (oracle ceiling at avg cands/S1): all 57.6 → 0.9833;
top 10 → 0.9832; top 8 → 0.9832; top 6 → 0.9815; top 5 → 0.9773; top 8 and prob >= 0.003 → 0.9830
at 4.73. With the full top 8 the final model scores 0.9527, so the smaller set costs 0.0002.

**Where the gain comes from** (same val candidates): deciding with the pruner probability alone
gives 0.9417 (new text + pair features + 2x training data: +0.037 over v0); the final model with
per-S1 relative features gives 0.9527 (+0.011).

## Unseen-country check (train on one country's fit S1 only; thresholds tuned on that country's val)

| | v0: own country | v0: other country | v1: own country | v1: other country |
|---|---|---|---|---|
| train US → India | 0.9176 | 0.7890 | 0.9538 | **0.8982** |
| train India → US | 0.8924 | 0.8801 | 0.9471 | **0.9164** |

v0 numbers from the saved v0 features (US → India reproduces Mhtv's phase 12: 0.7892). The v1
pruner and final model are both retrained on the single country. Hindi/Indic token maps are learned
from non-val India pairs, so US → India is not a pure zero-shot result for the script mapping.

## Error analysis on val (`python -m src.matching.error_analysis --tag pm`)

| | pairs | macro F0.5 if fixed |
|---|---|---|
| predicted pairs / correct | 698,762 / 690,144 | (0.9525 now) |
| false positives | 8,618 | 0.9635 |
| true matches rejected by the final model | 37,745 | 0.9720 |
| true matches cut by the pruner | 1,071 | +0.0003 on top |
| true matches never retrieved by stage 1 | 35,065 | (ceiling 0.9830 → 1.0) |

Versus v0: false positives 23,325 → 8,618, model-rejected true matches 83,731 → 37,745.
Loss by S1: 26.7% of S1 are partly right (67% of the loss); 1.2% have true matches but get no
prediction (25% of the loss); 0.3% are singletons given a match (7%). The best thresholds per
country equal the global ones (US and India both t=0.75), i.e. the model is calibrated alike in
both countries.

**Rejected true matches**: 39% have no address on one side (e.g. `Classic Telecom Enterprises |
2607 Bellefonte Drive` vs `Classic Telecom | —`); 39% have a house number 1 edit away (`10695` vs
`10693`, `224` vs `24`) or further; 10% have a meaningless replacement name at the same address
(`XX Allied Arch` vs `Evotavo`); 2.5% cross-script. Many are genuinely ambiguous in the data: among
candidates without an address whose name matches (token-set >= 0.9), only 54% are true matches
even when the name is unique in the pool, and 3% when 20+ records share it.

**False positives**: 89% are an extra candidate next to correct matches of the same S1. Types:
same name with a different/missing address (23%), same address with a different or meaningless
name (23%), same name with a house number 1 edit or more away (24%), near-identical records (8%).

**Stage-1 misses**: other-script names whose romanization differs from the English spelling
(`Raj Agro` vs `రాజ్ ఆగ్రో`, `Best Energy` vs `பெஸ்ட் எனர்ஜி`) and typos combined with a missing
or shortened address. These are the cases Ojaswi's skeleton and no-address blocks target.
