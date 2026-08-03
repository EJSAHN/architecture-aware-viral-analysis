# Robustness and alignment-based similarity benchmark

The primary analysis remains unchanged. Two optional workflows are provided for parameter sensitivity and conventional alignment-based cross-checking.

## K-mer and sliding-window sensitivity

The robustness command recomputes the principal validation metrics with canonical k-mer sizes 4 and 5 and evaluates mosaicity across 300-, 400-, and 500-nt windows. For fair comparison across k, the total pseudocount mass is held constant while the per-feature pseudocount is scaled to the size of the canonical k-mer vocabulary.

```bash
python -m viral_genome_analysis robustness \
  --curation-results-dir <CURATION_OUTPUT_DIR> \
  --output-dir <ROBUSTNESS_OUTPUT_DIR> \
  --config examples/strict_config.yaml \
  --kmer-sizes 4 5 \
  --window-sizes 300 400 500 \
  --top-n 20
```

The main output is `tables/06_robustness_validation.xlsx`. It contains panel-level validation under each k-mer size, rank concordance for custom metrics, sliding-window rank stability, and consensus candidate rankings.

## Alignment-based similarity benchmark

This workflow is intended as a conventional cross-check rather than a replacement or superiority test. It exports the complete monopartite panel, identifies high-mosaicity candidates and low-mosaicity controls, and scores local similarity to group consensus sequences after multiple alignment.

### 1. Prepare the benchmark panel

```bash
python -m viral_genome_analysis prepare-similarity-benchmark \
  --curation-results-dir <CURATION_OUTPUT_DIR> \
  --analysis-results-dir <PRIMARY_ANALYSIS_OUTPUT_DIR> \
  --output-dir <BENCHMARK_DIR> \
  --top-n 10 \
  --controls-per-group 5
```

### 2. Align the exported sequences

MAFFT can be installed with Conda:

```bash
conda install -c bioconda mafft
```

Then run:

```bash
mafft --auto --adjustdirectionaccurately \
  <BENCHMARK_DIR>/monopartite_similarity_benchmark.fasta \
  > <BENCHMARK_DIR>/monopartite_similarity_benchmark_aligned.fasta
```

### 3. Score the alignment-based profiles

```bash
python -m viral_genome_analysis score-similarity-benchmark \
  --aligned-fasta <BENCHMARK_DIR>/monopartite_similarity_benchmark_aligned.fasta \
  --manifest-workbook <BENCHMARK_DIR>/similarity_benchmark_manifest.xlsx \
  --output-dir <BENCHMARK_SCORE_DIR> \
  --window-size 400 \
  --step-size 100
```

The resulting workbook, `tables/07_alignment_similarity_benchmark.xlsx`, reports local group-similarity profiles and their concordance with compositional mosaicity. The aligned FASTA can also be loaded into SimPlot++ for an independent visual or formal recombination assessment.

## Completeness-aware isolate analysis

Singleton component records are excluded from isolate-level concordance, neighbor discordance is rescaled by its component-count-specific maximum, and full expected component sets are distinguished from informative partial sets. Full-set comparisons are treated as primary for cross-architecture inference, while all isolates represented by at least two components are reported as sensitivity analyses.
