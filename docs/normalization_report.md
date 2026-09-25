# Normalization checkpoint — Phases 6–8

Run on 25 Sep 2026 with `python -m src.normalize.run --split all` against the
provided dataset. Six Parquet files contain **24,229,173 rows** in total. The
sum of per-source processing times was **120.9 seconds** on this machine. A
spot check during the run showed a Python process peak working set of about
**1.42 GB**; this is not a full-run peak measurement, but is comfortably below
the 16 GB RAM target. The generated JSON counts are at
`data/norm/normalization_report.json` (gitignored, local only).

## Address-field coverage

Coverage is the proportion with a non-null extraction, **not** a verified
accuracy or match-rate metric. S1/S2/S3 are the challenge's three sources.

| Split | Source | Country | Rows | Postcode | City | House no. |
|---|---:|---|---:|---:|---:|---:|
| train | S1 | India | 883,188 | 0.00% | 97.70% | 75.85% |
| train | S1 | US | 1,323,633 | 0.00% | 99.09% | 99.76% |
| train | S2 | India | 2,017,799 | 0.00% | 94.63% | 71.40% |
| train | S2 | US | 3,016,817 | 0.00% | 95.89% | 81.23% |
| train | S3 | India | 2,115,547 | 0.00% | 96.32% | 69.88% |
| train | S3 | US | 3,170,056 | 0.00% | 96.01% | 82.51% |
| test | S1 | France | 259,452 | 0.29% | 96.56% | 99.32% |
| test | S1 | India | 809,986 | 0.00% | 97.70% | 75.82% |
| test | S1 | US | 663,106 | 0.00% | 99.10% | 99.76% |
| test | S2 | France | 703,378 | 0.19% | 94.77% | 90.81% |
| test | S2 | India | 2,312,565 | 0.00% | 95.15% | 73.14% |
| test | S2 | US | 1,871,330 | 0.00% | 96.63% | 83.06% |
| test | S3 | France | 731,615 | 0.21% | 94.87% | 91.12% |
| test | S3 | India | 2,405,000 | 0.00% | 96.95% | 71.71% |
| test | S3 | US | 1,945,701 | 0.00% | 96.68% | 84.17% |

### Interpretation for blocking

The raw US and India addresses virtually never provide unambiguous postal
codes. A naive five-digit US match mostly extracts **house/floor numbers**; a
naive six-digit India match can extract **survey numbers**. We therefore leave
these fields null instead of making false blocking keys. France has a small
number of genuine five-digit codes (examples: `59000 LILLE`, `62100, CALAIS`).
**Do not require postcode equality** in candidate generation; use names,
address text, city, and house number with a fallback route. We found and fixed
false positives caused by words like `Pinnacle`, US `Fl 13887`, and `Sd 46092`.

The city extractor chooses plausible comma-delimited segments, skipping known
state/region/department labels. The house-number extractor handles simple and
marked alphanumeric numbers. Both are best-effort and can still be wrong on
reordered or unstructured addresses. No country-specific rules are learned
from test outcomes; France patterns are deterministic text-format rules.

## Script and contract

`script` is `latin`, `devanagari`, or `other`. For example, train S2 India has
269,424 Devanagari names, 1,543,454 Latin names, and 204,921 names in other
scripts. `name_latin` currently strips accents from Latin `name_core`; it is
empty for non-Latin names until Phase 10 transliteration. Empty means
"unavailable", not "same name".

Each file has all 13 contract columns in order and as nullable strings:
`entity_id,country,name_raw,address_raw,name_norm,name_core,legal_suffix,name_latin,addr_norm,postcode,city,house_no,script`.
Parquet metadata row counts match the six raw source counts, and country totals
match each file's row count. No partial output files remained. The unit and
TSV-to-Parquet integration suite passed (49 tests). Phase 9 rank/val-pair
measurement is **not** included in this checkpoint.
