from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from Bio import SeqIO


REQUIRED_COLUMNS = [
    "accession_version",
    "source_group",
    "schema_type",
    "sequence_length",
]


def _read_fasta_map(fasta_path: Path) -> dict[str, str]:
    sequence_map: dict[str, str] = {}
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        accession = str(record.id).split()[0]
        sequence_map[accession] = str(record.seq).upper()
    return sequence_map


def load_curation_results(results_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(results_dir)
    metadata_path = root / config["input"]["metadata_workbook_relative"]
    fasta_path = root / config["input"]["sequence_fasta_relative"]
    manifest_path = root / "run_manifest.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata workbook not found: {metadata_path}")
    if not fasta_path.exists():
        raise FileNotFoundError(f"Curated FASTA not found: {fasta_path}")

    curated_df = pd.read_excel(metadata_path, sheet_name=config["input"]["metadata_sheet"])
    component_completeness = pd.read_excel(metadata_path, sheet_name=config["input"]["component_completeness_sheet"])
    component_matrix = pd.read_excel(metadata_path, sheet_name=config["input"]["component_matrix_sheet"])

    missing = [col for col in REQUIRED_COLUMNS if col not in curated_df.columns]
    if missing:
        raise ValueError(f"Curated metadata sheet is missing required columns: {missing}")

    sequence_map = _read_fasta_map(fasta_path)
    curated_df = curated_df.copy()
    curated_df["sequence"] = curated_df["accession_version"].astype(str).map(sequence_map)
    if curated_df["sequence"].isna().any():
        missing_accessions = curated_df.loc[curated_df["sequence"].isna(), "accession_version"].astype(str).tolist()[:10]
        raise ValueError(
            "Some curated records were not found in the FASTA file. "
            f"Examples: {missing_accessions}"
        )

    curated_df["analysis_label_all"] = curated_df.apply(
        lambda row: (
            f"{row['source_group']}|{row['normalized_component']}"
            if pd.notna(row.get("normalized_component")) and str(row.get("schema_type")) != "monopartite_genome"
            else str(row["source_group"])
        ),
        axis=1,
    )

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return {
        "curated_records": curated_df,
        "component_completeness": component_completeness,
        "component_matrix": component_matrix,
        "metadata_workbook": metadata_path,
        "sequence_fasta": fasta_path,
        "run_manifest": manifest,
    }
