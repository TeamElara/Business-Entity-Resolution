# Organizer updates

## 25 Sep 2026: candidate_pairs.tsv counts toward the final ranking

- `candidate_pairs.tsv` counts toward the **final ranking**. Judges review it and the code that produces it.
- A **smaller candidate set per S1 ranks higher**.
- It must be **exactly the set the final model runs inference on** (the last filtering stage), not an
  earlier, wider blocking pass.
- Every matched ID in `matching_results.tsv` must appear in `candidate_pairs.tsv` (matches ⊆ candidates).

What this means for us:
- The submitted `candidate_pairs.tsv` is Arihant's final pruner output after the
  wide stage-1 union (including Ojaswi's blocks). Ojaswi's separate pruner was
  evaluated but is not used in the selected v3/CP2 pipeline. The provisional
  cutoff is top 6 and pruner probability ≥ 0.003; confirm the exact cutoff at
  the final model freeze.
- The model scores exactly those pairs and nothing extra; `candidate_pairs.tsv` is written from the same
  dataframe the model predicted on (`src/common/writer.py::write_outputs` enforces matches ⊆ candidates).
- Always report the oracle ceiling together with avg candidates per S1 (`src/common/metrics.py`).
