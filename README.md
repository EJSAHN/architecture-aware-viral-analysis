# architecture-aware-viral-analysis

Alignment-free architecture-aware analysis of curated plant DNA virus panels, including geometric embedding, validation, mosaicity, component concordance, and configuration tension.

This repository provides the **second stage** of the workflow. It expects the standardized outputs produced by `architecture-aware-viral-curation` and generates analysis workbooks suitable for manuscript development and downstream figure building.

## What the package does

Given a curation-results directory, the pipeline will:

1. load curated metadata and curated FASTA,
2. compute reverse-complement-collapsed k-mer feature spaces,
3. build clr/Aitchison geometric embeddings,
4. run permutation-based multivariate separation tests,
5. evaluate dispersion so separation is not confounded by spread,
6. run repeated cross-validated label-recovery benchmarks,
7. score architecture-aware signals including mosaicity, orthogonal heterogeneity, neighbor concordance, and configuration tension,
8. write Excel workbooks and CSV bundles for downstream interpretation.

## Installation

```bash
pip install -e .
```

Or with Conda:

```bash
conda env create -f environment.yml
conda activate viral-genome-analysis
pip install -e .
```

## Basic usage

```bash
python -m viral_genome_analysis run --curation-results-dir <CURATION_OUTPUT_DIR> --output-dir <ANALYSIS_OUTPUT_DIR>
```

To run with an explicit configuration file:

```bash
python -m viral_genome_analysis write-config --output <CONFIG_PATH>
python -m viral_genome_analysis run --curation-results-dir <CURATION_OUTPUT_DIR> --output-dir <ANALYSIS_OUTPUT_DIR> --config <CONFIG_PATH>
```

Example configuration files are provided in `examples/`.

## Expected input layout

```text
curation_results/
├── tables/
│   └── 02_curated_metadata.xlsx
├── curated_fasta/
│   └── all_curated_sequences.fasta
├── run_manifest.json
└── resolved_config.json
```

Within `02_curated_metadata.xlsx`, the default expected sheets are:

- `curated_records`
- `component_completeness`
- `component_matrix`

These defaults can be adjusted in the YAML configuration.

## Output structure

```text
analysis_results/
├── tables/
│   ├── 01_analysis_registry.xlsx
│   ├── 02_geometry_embeddings.xlsx
│   ├── 03_statistical_validation.xlsx
│   ├── 04_novelty_signals.xlsx
│   └── 05_component_architecture.xlsx
├── csv/
│   ├── analysis_registry/
│   ├── geometry_embeddings/
│   ├── statistical_validation/
│   ├── novelty_signals/
│   └── component_architecture/
├── analysis.log
├── analysis_manifest.json
└── resolved_config.json
```

## Validation philosophy

The analysis layer is designed to make reviewer-facing claims difficult to dismiss on statistical grounds. It includes:

- permutation-based multivariate group separation,
- dispersion checks,
- repeated stratified cross-validation,
- label-permutation nulls for classification,
- bootstrap confidence intervals,
- FDR correction,
- sensitivity checks across complementary distance summaries.

## Documentation

- `docs/input_schema.md` — required curation-stage inputs
- `docs/output_schema.md` — summary of generated analysis outputs

## Repository scope

This repository covers **analysis only**. Raw GenBank downloads should first be processed with the companion repository:

- `architecture-aware-viral-curation`

## License

MIT
