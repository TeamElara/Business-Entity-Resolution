# Phase 11 — France-specific normalization

Completed 25 Sep 2026 from **unlabeled France test-input formats only**.
There are no France training labels, so the checks below are format and
regression audits, **not** a France matching score or leaderboard estimate.
The raw test TSVs and generated Parquets remain local and gitignored.

## Observed formats and rules

The three France test sources contain 259,452 S1, 703,378 S2, and 731,615 S3
rows (1,694,445 total). S2 and S3 each have about 178,000 addresses with a
standalone `R`/`R.` token; other common shortened street labels include
`BD`, `AV`, `PL`, `CH`, and `IMP`. French legal forms frequently occur at the
**start** of S2/S3 names as well as at the end. Examples from unlabeled rows
include `SCI Ptit Amicale`, `S.A.R.L. Rural Maison`, and `50 R. DE DJON`.

- Expand `BD→boulevard`, `AV→avenue`, `PL→place`, `IMP→impasse` as complete
  French address tokens. Expand `R/R.→rue` and `CH/CH.→chemin` only in guarded
  street positions. This avoids turning `R. I.` initials or `CH Dron` hospital
  text into streets. Some less structured true street abbreviations remain
  unexpanded deliberately. `St/Ste` becomes `saint/sainte` in France instead
  of the generic English `street` convention.
- Fold accents and the French `œ/æ` ligatures in France normalized names,
  addresses, and cities. Preserve `name_raw` and `address_raw` verbatim.
  Existing dotted legal forms and all seven requested legal forms—SARL, SAS,
  SASU, SA, EURL, SCI, SNC—are recognized. For France only, a leading form is
  stored in `legal_suffix` and removed from `name_core` when no trailing legal
  form was found. A trailing form takes priority if both ends have one.
- Continue conservative five-digit `code postal` extraction. Add compact
  `59046Lille Cedex`-style segments and normalize their city. Preserve
  leading-zero house numbers such as `042` and `017` as strings. Do not infer
  a code from isolated five-digit house/CS/BP numbers or from concatenated
  delivery numbers such as `CS9211933071` without a safe boundary.

## Full-file audit after regeneration

`python -m src.normalize.run --split test` regenerated the three test Parquets.
The training Parquets are unaffected because the supplied training countries
are India and US. Coverage means a non-null extraction, **not** field accuracy.

| France source | Rows | Legal form found | Postcode | City | House no. | Leading-zero house no. |
|---|---:|---:|---:|---:|---:|---:|
| Test S1 | 259,452 | 174,579 | 766 (0.30%) | 250,515 | 257,688 | 106 |
| Test S2 | 703,378 | 387,694 | 1,345 (0.19%) | 666,498 | 638,758 | 23,539 |
| Test S3 | 731,615 | 396,410 | 1,542 (0.21%) | 693,932 | 666,678 | 23,453 |

Before the leading-form rule, legal forms were detected on 174,575 S1,
341,597 S2, and 350,044 S3 France rows. The new S2/S3 coverage therefore
captures many otherwise missed leading forms; it does not prove every form
is semantically legal rather than part of a trade name. No France normalized
name or address retains a combining accent or `œ/æ` ligature. A direct audit
of all six Parquets confirmed the 13 ordered string columns and 24,229,173
total rows, with no partial files left.

For the US and India rows in all three regenerated test files, full-data
hashes of `name_norm`, `name_core`, `legal_suffix`, `name_latin`, `addr_norm`,
`postcode`, `city`, `house_no`, and `script` matched their pre-Phase-11 values
**exactly** (six source/country groups). The generic unknown-country path is
unchanged and has dedicated tests. The suite passed **83 tests**.

## Handoff and limitations

The original French text remains available for the blocker and matcher; use
the normalized fields as additional signals. French postcodes appear in only
about 0.2–0.3% of rows, so **do not require postcode equality** to retrieve
or retain candidates. The guarded `R/CH` rules intentionally trade some
abbreviation recall for fewer false expansions. France has no known matches
in the supplied labels, so Phase 12/generalization work must not treat this
format audit as measured French recall, F0.5, or test performance.
