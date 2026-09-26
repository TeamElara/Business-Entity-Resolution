# France test S2/S3 normalization gap audit (26 Sep 2026)

Label-free scan of **1,434,993** France records (703,378 S2; 731,615 S3)
using the raw and normalized columns in `data/norm/test_s{2,3}.parquet`.
These are pattern/extraction counts, **not** false-positive/false-negative or
France accuracy measurements. The local detailed JSON and audit script are
under ignored `data/eda/phase13/`; do not commit raw challenge files.

| Pattern | France S2+S3 rows | Potential normalization gap | Representative supplied-data example / interpretation |
| --- | ---: | ---: | --- |
| Postcode + CEDEX | 87 | 0 missing postcode | `33072BORDEAUX CEDEX` → `33072`; the tested compact form works. |
| House number `bis`/`ter` | 53,988 | 53,988 lose suffix in `house_no` | `17 BIS RUE MAURICE TERRIEN` → `house_no=17`, though `addr_norm` retains `bis`. |
| `Chem.` | 0 | 0 | Not present in this test S2/S3 scan. |
| `Rte.` / `RTE` | 11,593 | 11,587 lack `route` token | `4 RTE. DE BORDEAUX` → `4 rte de bordeaux`; likely safe France-only expansion. |
| `Imp.` / `IMP` | 12,006 | 0 lack `impasse` token | `2 IMP. LAMARTINE` → `2 impasse lamartine`; already covered. |
| `ZI` / `ZA` / `ZAC` | 9 / 35 / 18 | Manual review | `ZI ALFRED DANEY`, `ZA LES GENTELLERIES`, `ZAC DE LA CHAUSSEE` remain abbreviated; rare. |
| `BP` / `B.P.` box | 702 | Manual review | `BP 94189` is retained; the normalizer correctly does **not** assume it is a postcode. |
| `l'` / `d'` in address | 72,294 | Manual review | `FABRE D'EGLANTINE` → `fabre d eglantine`; punctuation is separated, not removed. |
| Hyphenated address text | 669,746 | Manual review | `Hauts-de-France` → `hauts de france`; broad exposure count, not necessarily a defect. |
| `SCI` | 61,226 | 4,589 not extracted as legal form | `SOCIALE SCI DÉVELOPPEMENT` has `SCI` mid-name; the current rule intentionally handles leading/trailing forms, so this is not automatically a miss. |
| `SELARL` | 0 | 0 | No occurrence in the scanned names. |
| `EI` | 14,305 | 14,221 not extracted | `Scene Union EI` retains `ei` in `name_core`; candidate for France-only legal-form handling. |
| `SA à directoire` | 0 | 0 | No occurrence in the scanned names. |
| `EURL` | 94,133 | 7,822 not extracted | `Solidarite EURL Cie` has a **mid-name** token; not automatically a legal-form miss. |
| `SASU` | 69,966 | 5,479 not extracted | `Pessac SASU Services` is also mid-name; not automatically a legal-form miss. |

The audit found zero `l'`/`d'` names in these files (their addresses are common).
Rows can match more than one pattern, and potential-gap counts are heuristic;
the same business appearing in several source rows is counted several times.

## Candidate actions for v4, not changes to the selected v2 run

1. Test a **France-only `rte` → `route`** address expansion. It is an exact
   street-type abbreviation and leaves US/India normalization unchanged.
2. Test **France-only leading/trailing `EI`** as a legal form. Preserve the
   full `name_norm`; only the comparison core should drop it, with safeguards
   for names that would become empty.
3. Discuss `bis`/`ter` house-number semantics with the matching owner before
   changing `house_no`. The normalized full address already retains the suffix;
   changing the extracted field can alter model features and needs validation.
4. Use Arihant's `france_review_v1.csv` to prioritize likely v1 match errors.
   The first 100 of its 200 sampled S1 blocks were reviewed separately:
   251 predicted pairs marked `ok`, 31 marked `wrong`, one rejected pair
   marked `missed`, and seven rows given an ambiguity note instead of a
   forced verdict. These are **manual inferences, not France ground-truth
   labels**. The marked CSV is kept outside Git for team handoff.

Any accepted fix requires targeted tests, full normalization regression
checks (including unchanged US/India outputs), and a new model/test run before
the final submission. Do not silently apply it to the already selected v2
outputs.

The isolated [v4 candidate PR #4](https://github.com/TeamElara/Business-Entity-Resolution/pull/4)
implements only the France-specific `Rte.` and `EI` changes. On regenerated
test normalization, all US/India source/country row counts and full-row
hash checksums match the prior outputs exactly. France `addr_norm` changes
for 398 S1 / 5,781 S2 / 5,812 S3; `name_core`/`legal_suffix` change for
4,183 S1 / 7,339 S2 / 7,435 S3. No model/leaderboard improvement is claimed.
