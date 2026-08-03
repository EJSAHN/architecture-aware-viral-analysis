from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from scipy.stats import spearmanr

from .config import load_config
from .excel import write_csv_bundle, write_workbook
from .io import load_curation_results
from .panels import build_panel_registry
from .utils import ensure_dir


def _read_mosaicity_scores(analysis_results_dir: str | Path) -> pd.DataFrame:
    root = Path(analysis_results_dir)
    csv_path = root / "csv" / "novelty_signals" / "mosaicity_scores.csv"
    workbook_path = root / "tables" / "04_novelty_signals.xlsx"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if workbook_path.exists():
        return pd.read_excel(workbook_path, sheet_name="mosaicity_scores")
    raise FileNotFoundError(
        "Mosaicity scores were not found. Expected either "
        f"{csv_path} or {workbook_path}."
    )


def prepare_similarity_benchmark(
    curation_results_dir: str | Path,
    analysis_results_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    top_n: int = 10,
    controls_per_group: int = 5,
) -> dict[str, Path]:
    """Prepare a reproducible alignment benchmark panel.

    The exported FASTA contains all monopartite genomes so that a conventional
    multiple alignment can be generated once. The manifest identifies high-ranked
    mosaicity candidates and low-ranked controls for downstream comparison.
    """

    output_root = ensure_dir(Path(output_dir))
    config = load_config(config_path)
    loaded = load_curation_results(curation_results_dir, config)
    registry, _, _ = build_panel_registry(loaded["curated_records"].copy(), config)
    mono = registry.loc[registry["panel_monopartite"]].copy().reset_index(drop=True)
    scores = _read_mosaicity_scores(analysis_results_dir).copy()
    scores["mosaicity_rank"] = pd.to_numeric(scores["mosaicity_rank"], errors="coerce")

    top = scores.nsmallest(int(top_n), "mosaicity_rank").copy()
    control_rows: list[pd.DataFrame] = []
    for source_group, group_scores in scores.groupby("source_group"):
        control_rows.append(group_scores.nlargest(int(controls_per_group), "mosaicity_rank"))
    controls = pd.concat(control_rows, ignore_index=True) if control_rows else pd.DataFrame()

    roles = pd.DataFrame({"accession_version": mono["accession_version"].astype(str)})
    roles["benchmark_role"] = "reference"
    roles.loc[roles["accession_version"].isin(top["accession_version"].astype(str)), "benchmark_role"] = "high_mosaicity_candidate"
    roles.loc[roles["accession_version"].isin(controls["accession_version"].astype(str)), "benchmark_role"] = "low_mosaicity_control"

    manifest = mono[
        [
            "accession_version",
            "source_group",
            "organism",
            "isolate",
            "sequence_length",
        ]
    ].copy()
    manifest["accession_version"] = manifest["accession_version"].astype(str)
    manifest = manifest.merge(roles, on="accession_version", how="left")
    manifest = manifest.merge(
        scores[
            [
                "accession_version",
                "compositional_mosaicity_index",
                "mosaicity_rank",
                "global_nearest_group",
                "dominant_window_group",
            ]
        ],
        on="accession_version",
        how="left",
    )

    fasta_path = output_root / "monopartite_similarity_benchmark.fasta"
    records: list[SeqRecord] = []
    role_map = manifest.set_index("accession_version")["benchmark_role"].to_dict()
    group_map = manifest.set_index("accession_version")["source_group"].to_dict()
    for _, row in mono.iterrows():
        accession = str(row["accession_version"])
        description = f"source_group={group_map.get(accession, '')};role={role_map.get(accession, 'reference')}"
        records.append(SeqRecord(Seq(str(row["sequence"])), id=accession, description=description))
    SeqIO.write(records, fasta_path, "fasta")

    manifest_path = output_root / "similarity_benchmark_manifest.xlsx"
    with pd.ExcelWriter(manifest_path, engine="openpyxl") as writer:
        manifest.sort_values(["benchmark_role", "source_group", "mosaicity_rank"], na_position="last").to_excel(
            writer,
            sheet_name="benchmark_manifest",
            index=False,
        )
        top.to_excel(writer, sheet_name="high_candidates", index=False)
        controls.to_excel(writer, sheet_name="low_controls", index=False)

    readme_path = output_root / "SIMILARITY_BENCHMARK_README.txt"
    readme_path.write_text(
        "Generate a multiple sequence alignment with MAFFT, for example:\n\n"
        "mafft --auto --adjustdirectionaccurately monopartite_similarity_benchmark.fasta "
        "> monopartite_similarity_benchmark_aligned.fasta\n\n"
        "Then run the score-similarity-benchmark command on the aligned FASTA. "
        "The same alignment can also be inspected in SimPlot++ as an external conventional benchmark.\n",
        encoding="utf-8",
    )
    return {
        "benchmark_fasta": fasta_path,
        "benchmark_manifest": manifest_path,
        "readme": readme_path,
    }


def _alignment_map(path: str | Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for record in SeqIO.parse(str(path), "fasta"):
        record_id = str(record.id).split()[0]
        # MAFFT --adjustdirectionaccurately prefixes reverse-complemented
        # sequence identifiers with _R_. Normalize that prefix so aligned
        # records continue to match accession identifiers in the manifest.
        if record_id.startswith("_R_"):
            record_id = record_id[3:]
        if record_id in records:
            raise ValueError(f"Duplicate aligned sequence identifier after normalization: {record_id}")
        records[record_id] = str(record.seq).upper()
    if not records:
        raise ValueError(f"No sequences were found in aligned FASTA: {path}")
    lengths = {len(seq) for seq in records.values()}
    if len(lengths) != 1:
        raise ValueError("Aligned FASTA sequences must all have the same alignment length.")
    return records


def _consensus(sequences: list[str]) -> str:
    if not sequences:
        return ""
    alignment_length = len(sequences[0])
    chars: list[str] = []
    for idx in range(alignment_length):
        counts = Counter(seq[idx] for seq in sequences if seq[idx] in {"A", "C", "G", "T"})
        chars.append(counts.most_common(1)[0][0] if counts else "N")
    return "".join(chars)


def _candidate_windows(aligned_candidate: str, window_size: int, step_size: int) -> list[tuple[int, int, list[int]]]:
    ungapped_columns = [idx for idx, base in enumerate(aligned_candidate) if base not in {"-", "."}]
    if len(ungapped_columns) < window_size:
        return [(1, len(ungapped_columns), ungapped_columns)] if ungapped_columns else []
    windows: list[tuple[int, int, list[int]]] = []
    for start in range(0, len(ungapped_columns) - window_size + 1, step_size):
        columns = ungapped_columns[start : start + window_size]
        windows.append((start + 1, start + window_size, columns))
    if windows and windows[-1][1] < len(ungapped_columns):
        start = len(ungapped_columns) - window_size
        windows.append((start + 1, len(ungapped_columns), ungapped_columns[start:]))
    return windows


def _identity(candidate: str, consensus: str, columns: list[int]) -> float:
    comparable = 0
    matches = 0
    for col in columns:
        a = candidate[col]
        b = consensus[col]
        if a not in {"A", "C", "G", "T"} or b not in {"A", "C", "G", "T"}:
            continue
        comparable += 1
        matches += int(a == b)
    return float(matches / comparable) if comparable else float("nan")


def score_similarity_benchmark(
    aligned_fasta: str | Path,
    manifest_workbook: str | Path,
    output_dir: str | Path,
    window_size: int = 400,
    step_size: int = 100,
) -> dict[str, Path]:
    """Score an alignment-based, SimPlot-style group similarity profile."""

    output_root = ensure_dir(Path(output_dir))
    aligned = _alignment_map(aligned_fasta)
    manifest = pd.read_excel(manifest_workbook, sheet_name="benchmark_manifest")
    manifest["accession_version"] = manifest["accession_version"].astype(str)
    manifest = manifest[manifest["accession_version"].isin(aligned)].copy()
    groups = sorted(manifest["source_group"].dropna().astype(str).unique())
    if len(groups) < 2:
        raise ValueError("At least two source groups are required for similarity-profile scoring.")

    score_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    target_manifest = manifest[manifest["benchmark_role"].isin(["high_mosaicity_candidate", "low_mosaicity_control"])].copy()
    for _, row in target_manifest.iterrows():
        accession = str(row["accession_version"])
        candidate = aligned[accession]
        consensus_by_group: dict[str, str] = {}
        for group in groups:
            member_ids = manifest.loc[
                (manifest["source_group"].astype(str) == group) & (manifest["accession_version"] != accession),
                "accession_version",
            ].astype(str)
            consensus_by_group[group] = _consensus([aligned[idx] for idx in member_ids if idx in aligned])

        assignments: list[str] = []
        margins: list[float] = []
        identities_by_window: list[dict[str, float]] = []
        for window_index, (start, end, columns) in enumerate(_candidate_windows(candidate, int(window_size), int(step_size)), start=1):
            identities = {group: _identity(candidate, consensus, columns) for group, consensus in consensus_by_group.items()}
            finite = [(group, value) for group, value in identities.items() if np.isfinite(value)]
            if not finite:
                continue
            ordered = sorted(finite, key=lambda item: item[1], reverse=True)
            best_group, best_identity = ordered[0]
            second_identity = ordered[1][1] if len(ordered) > 1 else best_identity
            assignments.append(best_group)
            margins.append(float(best_identity - second_identity))
            identities_by_window.append(identities)
            output_row: dict[str, Any] = {
                "accession_version": accession,
                "source_group": row["source_group"],
                "benchmark_role": row["benchmark_role"],
                "window_index": window_index,
                "candidate_start": start,
                "candidate_end": end,
                "best_group": best_group,
                "best_identity": best_identity,
                "identity_margin_best_minus_second": float(best_identity - second_identity),
            }
            for group, value in identities.items():
                output_row[f"identity_to_{group}"] = value
            window_rows.append(output_row)

        if not assignments:
            continue
        counts = Counter(assignments)
        dominant_group, dominant_count = counts.most_common(1)[0]
        entropy = -sum((count / len(assignments)) * math.log(count / len(assignments), 2) for count in counts.values())
        max_entropy = math.log(len(groups), 2) if len(groups) > 1 else 1.0
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
        switches = sum(a != b for a, b in zip(assignments[:-1], assignments[1:]))
        switch_rate = switches / max(1, len(assignments) - 1)
        source_group = str(row["source_group"])
        discordance = sum(a != source_group for a in assignments) / len(assignments)
        ambiguity = 1.0 - float(np.mean(margins))
        profile_score = float(np.mean([entropy_norm, switch_rate, discordance, ambiguity]))
        score_rows.append(
            {
                "accession_version": accession,
                "source_group": source_group,
                "benchmark_role": row["benchmark_role"],
                "n_windows": len(assignments),
                "dominant_similarity_group": dominant_group,
                "dominant_similarity_fraction": dominant_count / len(assignments),
                "alignment_profile_entropy_norm": entropy_norm,
                "alignment_profile_switch_rate": switch_rate,
                "alignment_profile_discordance_fraction": discordance,
                "alignment_profile_ambiguity_score": ambiguity,
                "alignment_similarity_profile_score": profile_score,
                "compositional_mosaicity_index": row.get("compositional_mosaicity_index", np.nan),
                "mosaicity_rank": row.get("mosaicity_rank", np.nan),
            }
        )

    scores = pd.DataFrame(score_rows)
    windows = pd.DataFrame(window_rows)
    concordance_rows: list[dict[str, Any]] = []
    if not scores.empty:
        pair = scores[["compositional_mosaicity_index", "alignment_similarity_profile_score"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) >= 3 and pair.nunique().min() >= 2:
            result = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
            concordance_rows.append(
                {
                    "comparison": "compositional_mosaicity_vs_alignment_profile",
                    "n_sequences": len(pair),
                    "spearman_rho": float(result.statistic),
                    "spearman_p_value": float(result.pvalue),
                }
            )
    role_summary = (
        scores.groupby("benchmark_role", as_index=False)
        .agg(
            n_sequences=("accession_version", "count"),
            alignment_profile_mean=("alignment_similarity_profile_score", "mean"),
            alignment_profile_median=("alignment_similarity_profile_score", "median"),
            compositional_mosaicity_mean=("compositional_mosaicity_index", "mean"),
            compositional_mosaicity_median=("compositional_mosaicity_index", "median"),
        )
        if not scores.empty
        else pd.DataFrame()
    )

    sheets = {
        "benchmark_settings": pd.DataFrame(
            {
                "parameter": ["aligned_fasta", "manifest_workbook", "window_size", "step_size"],
                "value": [str(Path(aligned_fasta).resolve()), str(Path(manifest_workbook).resolve()), int(window_size), int(step_size)],
            }
        ),
        "alignment_profile_scores": scores,
        "alignment_profile_windows": windows,
        "mosaicity_concordance": pd.DataFrame(concordance_rows),
        "candidate_control_summary": role_summary,
    }
    tables_dir = ensure_dir(output_root / "tables")
    csv_dir = ensure_dir(output_root / "csv" / "similarity_benchmark")
    workbook = write_workbook(sheets, tables_dir / "07_alignment_similarity_benchmark.xlsx")
    write_csv_bundle(sheets, csv_dir)
    manifest_path = output_root / "similarity_benchmark_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "aligned_fasta": str(Path(aligned_fasta).resolve()),
                "manifest_workbook": str(Path(manifest_workbook).resolve()),
                "window_size": int(window_size),
                "step_size": int(step_size),
                "workbook": str(workbook),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"similarity_benchmark_workbook": workbook, "manifest": manifest_path}
