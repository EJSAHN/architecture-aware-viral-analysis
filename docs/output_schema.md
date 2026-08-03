# Output schema

The analysis pipeline writes workbook and CSV outputs summarizing:

- panel registry and analysis inputs,
- geometric embeddings,
- statistical validation,
- novelty-oriented signals,
- component architecture summaries.

Private figure-generation scripts are intentionally excluded from this public repository.

## Optional robustness outputs

When the optional workflows are run, the following additional workbooks are produced:

- `06_robustness_validation.xlsx` — k-mer and sliding-window sensitivity summaries;
- `07_alignment_similarity_benchmark.xlsx` — alignment-based local similarity profiles and concordance with compositional mosaicity.
