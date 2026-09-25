# Submission log

Upload only when val macro F0.5 improves. Max 5 uploads/day (Arihant uploads).

| upload # | date | git commit | val F0.5 | public LB | avg cands/S1 | what changed |
|---|---|---|---|---|---|---|
| 1 | 25 Sep 22:41 | 8d7d4c4 | 0.9043 | 0.886 | 8.00 | Baseline v0 (`python -m src.matching.baseline_v0 features/train/test`): name char-4gram + address word 1-2gram TF-IDF top 30 each per country, LightGBM on 22 country-agnostic pair features (80k train S1), top-8 candidates, t=0.65 / t1=0.55. Val by country: US 0.915, India 0.889. Test: 3.05 matches/S1, 6.2% empty; validator PASS with --check-ids. LB gap vs val likely France (unseen, 15% of test): India/US reweighted val ≈ 0.90 ⇒ France ≈ 0.80 |
