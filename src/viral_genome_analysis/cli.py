from __future__ import annotations

import argparse

from .config import write_default_config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="viral-genome-analysis",
        description="Run statistical and geometric analysis on curated viral genome datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the analysis pipeline")
    run_parser.add_argument("--curation-results-dir", required=True, help="Folder produced by the curation pipeline")
    run_parser.add_argument("--output-dir", required=True, help="Folder where analysis outputs will be written")
    run_parser.add_argument("--config", default=None, help="Optional YAML config override")
    run_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    config_parser = subparsers.add_parser("write-config", help="Write a default YAML config template")
    config_parser.add_argument("--output", required=True, help="Path to write the default config YAML")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "write-config":
        output = write_default_config(args.output)
        print(f"Wrote default config to: {output}")
        return

    if args.command == "run":
        results = run_pipeline(
            curation_results_dir=args.curation_results_dir,
            output_dir=args.output_dir,
            config_path=args.config,
            verbose=args.verbose,
        )
        for key, value in results.items():
            print(f"{key}: {value}")
        return

    parser.error("Unknown command")
