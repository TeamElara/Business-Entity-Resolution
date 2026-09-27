# Label-free check of a test submission (0.933 upload)

`python -m src.blocking.v3_check --matches output/.../matching_results.tsv` computes, per country and
without labels, how a test prediction compares with what a perfect prediction looks like on
validation (`--val` on the val truth gives the reference column). Checked file: the 26 Sep upload #5
(v3 + orphan / reverse-search features, t = 0.85), public LB 0.933.

| | India val truth | India test pred | US val truth | US test pred | France test pred |
|---|---|---|---|---|---|
| matches / S1 | 3.46 | 3.26 | 3.46 | **3.42** | 3.16 |
| label-free target (pool × (1 − EM orphan share) / S1) | | 3.37 | | 3.41 | 3.93 (unreliable) |
| S1 with no match | 5.5% | 5.5% | 5.6% | 5.5% | 5.5% |
| pairs with orphan_prob > 0.8 | 1.5% | 2.6% | 1.9% | **5.6%** | 1.8% |
| pairs with orphan_prob > 0.5 | 6.8% | 9.2% | 4.6% | **9.7%** | 6.2% |
| S1 is the record's rank-1 S1 (reverse search) | 86.1% | 90.6% | 85.4% | 89.1% | 77.5% |
| S1 in the record's reverse top 10 | 93.4% | 96.3% | 93.3% | 96.0% | 89.6% |
| strong pairs (name token-set ≥ 90, address ≥ 70) | 74.4% | 78.5% | 73.9% | 77.3% | 82.3% |

Candidates (candidate_pairs.tsv): 5.27–5.66 per S1 (max 6); S1 without candidates: France 177,
India 1,521, US 1,126; every match is in its S1's candidate list.

Reading:
- **US looks over-predicted**: 3.42 matches/S1 equals the full label-free estimate of the true count
  (3.41), while on validation the best F0.5 came from predicting ~6% fewer matches than the truth;
  and 5.6% of US predicted pairs have orphan_prob > 0.8, three times the share among true pairs
  (1.9%).
- **India looks right**: 3% under its target, orphan_prob > 0.8 share 2.6% vs 1.5%.
- **France may be under-predicted**: 3.16 matches/S1 against 3.8 strong candidates per S1 in the
  blocking check (US/India have ~10% fewer strong candidates than true matches). The France EM target
  is unreliable (no France labels), and its low rank-1 share likely comes from many exactly tied
  duplicate S1.
- "Records claimed by 2+ S1" is 0 in the TSV because the one-record-one-S1 rule is applied.
