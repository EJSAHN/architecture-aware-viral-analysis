from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "input": {
        "metadata_workbook_relative": "tables/02_curated_metadata.xlsx",
        "metadata_sheet": "curated_records",
        "component_completeness_sheet": "component_completeness",
        "component_matrix_sheet": "component_matrix",
        "sequence_fasta_relative": "curated_fasta/all_curated_sequences.fasta",
    },
    "analysis": {
        "kmer_size": 4,
        "reverse_complement_collapse": True,
        "pseudocount": 0.5,
        "pca_components": 8,
        "min_group_size_monopartite": 8,
        "min_component_records": 10,
        "min_windows_per_sequence": 3,
        "sliding_window_size": 400,
        "sliding_window_step": 100,
        "top_n_neighbors": 3,
        "top_n_mosaic_candidates": 100,
        "tension_embedding_dimensions": 3,
        "min_isolate_components_for_tension": 2,
        "orthogonal_axis_geometry_components": 3,
        "random_seed": 7,
    },
    "statistics": {
        "global_permutations": 49,
        "pairwise_permutations": 49,
        "classification_permutations": 9,
        "bootstrap_iterations": 200,
        "cv_folds": 5,
        "cv_repeats": 3,
    },
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return config
    path = Path(config_path)
    override = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(override, dict):
        raise ValueError("Config file must define a YAML dictionary at the top level.")
    return deep_update(config, override)


def write_default_config(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False), encoding="utf-8")
    return path
