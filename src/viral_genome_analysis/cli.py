from __future__ import annotations

import argparse

from .config import write_default_config
from .pipeline import run_pipeline
from .robustness import run_robustness_analysis
from .similarity import prepare_similarity_benchmark, score_similarity_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="viral-genome-analysis",
        description="Run architecture-aware statistical and geometric analysis on curated viral genome datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the primary analysis pipeline")
    run_parser.add_argument("--curation-results-dir", required=True, help="Folder produced by the curation pipeline")
    run_parser.add_argument("--output-dir", required=True, help="Folder where analysis outputs will be written")
    run_parser.add_argument("--config", default=None, help="Optional YAML config override")
    run_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    robustness_parser = subparsers.add_parser(
        "robustness",
        help="Run k-mer-size and sliding-window sensitivity analyses",
    )
    robustness_parser.add_argument("--curation-results-dir", required=True)
    robustness_parser.add_argument("--output-dir", required=True)
    robustness_parser.add_argument("--config", default=None)
    robustness_parser.add_argument("--kmer-sizes", nargs="+", type=int, default=[4, 5])
    robustness_parser.add_argument("--window-sizes", nargs="+", type=int, default=[300, 400, 500])
    robustness_parser.add_argument("--top-n", type=int, default=20)

    prepare_parser = subparsers.add_parser(
        "prepare-similarity-benchmark",
        help="Export a monopartite benchmark FASTA and candidate/control manifest",
    )
    prepare_parser.add_argument("--curation-results-dir", required=True)
    prepare_parser.add_argument("--analysis-results-dir", required=True)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--config", default=None)
    prepare_parser.add_argument("--top-n", type=int, default=10)
    prepare_parser.add_argument("--controls-per-group", type=int, default=5)

    score_parser = subparsers.add_parser(
        "score-similarity-benchmark",
        help="Score an aligned FASTA with a SimPlot-style group similarity profile",
    )
    score_parser.add_argument("--aligned-fasta", required=True)
    score_parser.add_argument("--manifest-workbook", required=True)
    score_parser.add_argument("--output-dir", required=True)
    score_parser.add_argument("--window-size", type=int, default=400)
    score_parser.add_argument("--step-size", type=int, default=100)

    config_parser = subparsers.add_parser("write-config", help="Write a default YAML config template")
    config_parser.add_argument("--output", required=True, help="Path to write the default config YAML")

    return parser


def _print_results(results: dict[str, object]) -> None:
    for key, value in results.items():
        print(f"{key}: {value}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "write-config":
        output = write_default_config(args.output)
        print(f"Wrote default config to: {output}")
        return

    if args.command == "run":
        _print_results(
            run_pipeline(
                curation_results_dir=args.curation_results_dir,
                output_dir=args.output_dir,
                config_path=args.config,
                verbose=args.verbose,
            )
        )
        return

    if args.command == "robustness":
        _print_results(
            run_robustness_analysis(
                curation_results_dir=args.curation_results_dir,
                output_dir=args.output_dir,
                config_path=args.config,
                kmer_sizes=args.kmer_sizes,
                window_sizes=args.window_sizes,
                top_n=args.top_n,
            )
        )
        return

    if args.command == "prepare-similarity-benchmark":
        _print_results(
            prepare_similarity_benchmark(
                curation_results_dir=args.curation_results_dir,
                analysis_results_dir=args.analysis_results_dir,
                output_dir=args.output_dir,
                config_path=args.config,
                top_n=args.top_n,
                controls_per_group=args.controls_per_group,
            )
        )
        return

    if args.command == "score-similarity-benchmark":
        _print_results(
            score_similarity_benchmark(
                aligned_fasta=args.aligned_fasta,
                manifest_workbook=args.manifest_workbook,
                output_dir=args.output_dir,
                window_size=args.window_size,
                step_size=args.step_size,
            )
        )
        return

    parser.error("Unknown command")
