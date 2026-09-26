# Submission log

Upload only when val macro F0.5 improves. Max 5 uploads/day (Arihant uploads).

| upload # | date | git commit | val F0.5 | public LB | avg cands/S1 | what changed |
|---|---|---|---|---|---|---|
| 1 | 25 Sep 22:41 | 8d7d4c4 | 0.9043 | 0.886 | 8.00 | Baseline v0 (`python -m src.matching.baseline_v0 features/train/test`): name char-4gram + address word 1-2gram TF-IDF top 30 each per country, LightGBM on 22 country-agnostic pair features (80k train S1), top-8 candidates, t=0.65 / t1=0.55. Val by country: US 0.915, India 0.889. Test: 3.05 matches/S1, 6.2% empty; validator PASS with --check-ids. LB gap vs val likely France (unseen, 15% of test): India/US reweighted val ≈ 0.90 ⇒ France ≈ 0.80 |
| 2 | 26 Sep 10:45 | 4fbdd84 | 0.9525 | 0.919 | 5.45 | Matching v1 (`python -m src.matching.v1 test --tag pm`): Mhtv's norm files + name_latin, Ojaswi's latinize/clean_expr/skeletons; stage 1 (v0 TF-IDF, 57.6/S1) → pruner LightGBM (40 pair features incl. tolerant house numbers, name differences) → top 8 & prob ≥ 0.003 → final LightGBM (+19 per-S1 features inside the candidate set), t=0.75 / t1=0.60. Val US 0.9559, India 0.9473; unseen-country US→India 0.8982, India→US 0.9164. Test cands/S1 India 5.25, US 5.29, France 6.48; 3.23 matches/S1, 6.0% empty; validator PASS with --check-ids. Val–LB gap grew (0.018 → 0.034): if test India/US ≈ val, France ≈ 0.73 (v0 ≈ 0.80) → hybrid probe (v1 India/US + v0 France) to check |
| probe | 26 Sep 11:48 | 4fbdd84 | – | 0.912 | – | Diagnostic only, not a candidate for the final: v1 matches for India/US + v0 matches for France (`data/pred/probe_hybrid_fr_v0`). 0.912 < 0.919 ⇒ v1 is also better on France (≈ +0.05 on the France part, since France is 15% of test); the larger val–LB gap comes from test India/US being harder than val |
