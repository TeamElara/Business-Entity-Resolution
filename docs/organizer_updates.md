# Organizer updates

## 25 Sep 2026: candidate_pairs.tsv counts toward the final ranking

- `candidate_pairs.tsv` counts toward the **final ranking**. Judges review it and the code that produces it.
- A **smaller candidate set per S1 ranks higher**.
- It must be **exactly the set the final model runs inference on** (the last filtering stage), not an
  earlier, wider blocking pass.
- Every matched ID in `matching_results.tsv` must appear in `candidate_pairs.tsv` (matches ⊆ candidates).

What this means for us:
- Ojaswi's `data/cand/{split}.parquet` is the final, pruned set (wide internal stage → pruner → final set).
  Target **~8–15 candidates per S1 on average** at a high oracle ceiling.
- The model scores exactly those pairs and nothing extra; `candidate_pairs.tsv` is written from the same
  dataframe the model predicted on (`src/common/writer.py::write_outputs` enforces matches ⊆ candidates).
- Always report the oracle ceiling together with avg candidates per S1 (`src/common/metrics.py`).
