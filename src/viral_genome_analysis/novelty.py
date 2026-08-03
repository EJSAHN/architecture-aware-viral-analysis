from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .kmers import build_kmer_vocabulary, compute_kmer_features, sequence_kmer_counts


def _window_iter(sequence: str, window_size: int, step_size: int) -> list[tuple[int, int, str]]:
    if len(sequence) <= window_size:
        return [(1, len(sequence), sequence)]
    windows: list[tuple[int, int, str]] = []
    for start in range(0, len(sequence) - window_size + 1, step_size):
        window_seq = sequence[start : start + window_size]
        windows.append((start + 1, start + window_size, window_seq))
    if windows and windows[-1][1] < len(sequence):
        start = len(sequence) - window_size
        windows.append((start + 1, len(sequence), sequence[start:]))
    return windows


def _window_clr(sequence: str, k: int, mapping: dict[str, int], vocab_size: int, pseudocount: float) -> np.ndarray:
    counts = sequence_kmer_counts(sequence, k=k, mapping=mapping, vocab_size=vocab_size)
    freq = counts + pseudocount
    freq = freq / freq.sum()
    clr = np.log(freq) - np.log(freq).mean()
    return clr


def compositional_mosaicity(
    df: pd.DataFrame,
    group_col: str,
    config: dict[str, Any],
    clr_matrix: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_df = df.copy().reset_index(drop=True)
    eligible_counts = panel_df[group_col].value_counts()
    eligible_labels = eligible_counts[eligible_counts >= int(config["analysis"]["min_group_size_monopartite"])].index.tolist()
    panel_df = panel_df[panel_df[group_col].isin(eligible_labels)].copy().reset_index(drop=True)
    if panel_df.empty or panel_df[group_col].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()

    k = int(config["analysis"]["kmer_size"])
    pseudocount = float(config["analysis"]["pseudocount"])
    vocabulary, mapping = build_kmer_vocabulary(k, bool(config["analysis"]["reverse_complement_collapse"]))
    if clr_matrix is None:
        _, clr_matrix, _ = compute_kmer_features(
            panel_df["sequence"].tolist(),
            k=k,
            collapse_reverse_complements=bool(config["analysis"]["reverse_complement_collapse"]),
            pseudocount=pseudocount,
        )

    group_labels = pd.Index(pd.unique(panel_df[group_col]))
    centroids = np.vstack([clr_matrix[panel_df[group_col].to_numpy() == label].mean(axis=0) for label in group_labels])

    score_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    window_size = int(config["analysis"]["sliding_window_size"])
    step_size = int(config["analysis"]["sliding_window_step"])
    min_windows = int(config["analysis"]["min_windows_per_sequence"])

    for row_index, row in panel_df.iterrows():
        windows = _window_iter(str(row["sequence"]), window_size=window_size, step_size=step_size)
        if len(windows) < min_windows:
            continue

        assignments: list[str] = []
        margins: list[float] = []
        full_distances = cdist(clr_matrix[row_index : row_index + 1], centroids, metric="euclidean")[0]
        full_label = str(group_labels[int(np.argmin(full_distances))])

        for window_index, (start, end, subseq) in enumerate(windows, start=1):
            clr = _window_clr(subseq, k=k, mapping=mapping, vocab_size=len(vocabulary), pseudocount=pseudocount)
            distances = cdist(clr.reshape(1, -1), centroids, metric="euclidean")[0]
            order = np.argsort(distances)
            best_label = str(group_labels[int(order[0])])
            second_distance = float(distances[int(order[1])]) if len(order) > 1 else float(distances[int(order[0])])
            margin = second_distance - float(distances[int(order[0])])
            assignments.append(best_label)
            margins.append(margin)
            window_rows.append(
                {
                    "accession_version": row["accession_version"],
                    "source_group": row["source_group"],
                    "global_nearest_group": full_label,
                    "window_index": window_index,
                    "window_start": start,
                    "window_end": end,
                    "window_nearest_group": best_label,
                    "distance_to_best_group": float(distances[int(order[0])]),
                    "distance_margin_second_minus_first": margin,
                }
            )

        counts = Counter(assignments)
        dominant_group, dominant_count = counts.most_common(1)[0]
        entropy = -sum((count / len(assignments)) * math.log(count / len(assignments), 2) for count in counts.values())
        max_entropy = math.log(len(group_labels), 2) if len(group_labels) > 1 else 1.0
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
        switches = sum(a != b for a, b in zip(assignments[:-1], assignments[1:]))
        switch_rate = switches / max(1, len(assignments) - 1)
        discordance_fraction = sum(a != full_label for a in assignments) / len(assignments)
        ambiguity_score = 1.0 / (1.0 + float(np.mean(margins)))
        compositional_mosaicity_index = float(
            np.mean([entropy_norm, switch_rate, discordance_fraction, ambiguity_score])
        )
        score_rows.append(
            {
                "accession_version": row["accession_version"],
                "source_group": row["source_group"],
                "organism": row.get("organism", ""),
                "isolate": row.get("isolate", ""),
                "sequence_length": row["sequence_length"],
                "n_windows": len(assignments),
                "global_nearest_group": full_label,
                "dominant_window_group": dominant_group,
                "dominant_window_fraction": dominant_count / len(assignments),
                "window_label_entropy_norm": entropy_norm,
                "switch_rate": switch_rate,
                "discordance_fraction": discordance_fraction,
                "ambiguity_score": ambiguity_score,
                "compositional_mosaicity_index": compositional_mosaicity_index,
            }
        )

    score_df = pd.DataFrame(score_rows).sort_values(
        ["compositional_mosaicity_index", "discordance_fraction", "switch_rate"],
        ascending=[False, False, False],
        ignore_index=True,
    )
    if not score_df.empty:
        score_df["mosaicity_rank"] = np.arange(1, len(score_df) + 1)
        top_ids = set(score_df.head(int(config["analysis"]["top_n_mosaic_candidates"]))["accession_version"].tolist())
        window_df = pd.DataFrame(window_rows)
        window_df = window_df[window_df["accession_version"].isin(top_ids)].copy()
    else:
        window_df = pd.DataFrame(window_rows)
    return score_df, window_df


def _component_completeness_lookup(component_completeness_df: pd.DataFrame | None) -> dict[tuple[str, str], dict[str, Any]]:
    if component_completeness_df is None or component_completeness_df.empty:
        return {}
    required = {"source_group", "isolate_key"}
    if not required.issubset(component_completeness_df.columns):
        return {}
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in component_completeness_df.iterrows():
        source_group = str(row.get("source_group", ""))
        isolate_key = str(row.get("isolate_key", ""))
        if not source_group or not isolate_key or isolate_key.lower() == "nan":
            continue
        expected = pd.to_numeric(pd.Series([row.get("expected_component_count")]), errors="coerce").iloc[0]
        full_set = row.get("all_expected_components_present", False)
        lookup[(source_group, isolate_key)] = {
            "expected_component_count": int(expected) if pd.notna(expected) else None,
            "full_expected_component_set": bool(full_set),
        }
    return lookup


def component_neighbor_concordance(
    df: pd.DataFrame,
    config: dict[str, Any],
    full_clr_matrix: np.ndarray | None = None,
    component_completeness_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Quantify component-wise nearest-neighbour agreement for informative isolates.

    Isolates represented by only one component are excluded because concordance is
    undefined for a singleton. The raw score is retained, and a component-count-
    normalized score rescales the raw discordance by its maximum possible value
    (1 - 1/m) for an isolate with m represented components.
    """

    component_df = df.copy()
    component_df = component_df[component_df["panel_componentized"]].copy()
    if component_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    k = int(config["analysis"]["kmer_size"])
    pseudocount = float(config["analysis"]["pseudocount"])
    completeness_lookup = _component_completeness_lookup(component_completeness_df)
    discordance_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for source_group, group_df in component_df.groupby("source_group"):
        group_df = group_df[group_df["isolate"].notna() & group_df["normalized_component"].notna()].copy()
        if group_df.empty:
            continue

        # One canonical representative per isolate/component combination for concordance scoring.
        group_df = (
            group_df.sort_values(
                ["metadata_score", "sequence_length", "accession_version"],
                ascending=[False, False, True],
            )
            .drop_duplicates(subset=["isolate", "normalized_component"], keep="first")
            .reset_index(drop=True)
        )

        component_counts = group_df["normalized_component"].value_counts()
        eligible_components = component_counts[component_counts >= 2].index.tolist()
        group_df = group_df[group_df["normalized_component"].isin(eligible_components)].copy().reset_index(drop=True)
        if group_df.empty or group_df["normalized_component"].nunique() < 2:
            continue

        expected_from_group = int(group_df["normalized_component"].nunique())
        group_edge_rows: list[dict[str, Any]] = []
        for component_label, comp_df in group_df.groupby("normalized_component"):
            if comp_df["isolate"].nunique() < 2:
                continue
            if full_clr_matrix is not None and "feature_row" in comp_df.columns:
                clr_matrix = full_clr_matrix[comp_df["feature_row"].to_numpy()]
            else:
                _, clr_matrix, _ = compute_kmer_features(
                    comp_df["sequence"].tolist(),
                    k=k,
                    collapse_reverse_complements=bool(config["analysis"]["reverse_complement_collapse"]),
                    pseudocount=pseudocount,
                )
            distances = cdist(clr_matrix, clr_matrix, metric="euclidean")
            np.fill_diagonal(distances, np.inf)
            isolates = comp_df["isolate"].astype(str).tolist()
            accessions = comp_df["accession_version"].astype(str).tolist()
            for i in range(len(comp_df)):
                ordered = np.argsort(distances[i])
                target_index = None
                for candidate in ordered:
                    if isolates[int(candidate)] != isolates[i]:
                        target_index = int(candidate)
                        break
                if target_index is None:
                    continue
                edge = {
                    "source_group": source_group,
                    "component_label": component_label,
                    "accession_version": accessions[i],
                    "isolate": isolates[i],
                    "target_accession_version": accessions[target_index],
                    "target_isolate": isolates[target_index],
                    "distance_to_target": float(distances[i, target_index]),
                }
                edge_rows.append(edge)
                group_edge_rows.append(edge)

        edge_df = pd.DataFrame(group_edge_rows)
        if edge_df.empty:
            continue

        for isolate, isolate_edges in edge_df.groupby("isolate"):
            components_present = int(isolate_edges["component_label"].nunique())
            if components_present < 2:
                # A single component cannot provide an isolate-level concordance measure.
                continue
            target_counts = isolate_edges["target_isolate"].value_counts()
            top_target = str(target_counts.index[0])
            top_count = int(target_counts.iloc[0])
            agreement_fraction = top_count / components_present
            raw_discordance = 1.0 - agreement_fraction
            max_discordance = 1.0 - (1.0 / components_present)
            normalized_discordance = raw_discordance / max_discordance if max_discordance > 0 else float("nan")

            completeness = completeness_lookup.get((str(source_group), str(isolate)), {})
            expected_component_count = completeness.get("expected_component_count")
            if expected_component_count is None:
                expected_component_count = expected_from_group
            full_set = completeness.get("full_expected_component_set")
            if full_set is None:
                full_set = components_present == int(expected_component_count)

            discordance_rows.append(
                {
                    "source_group": source_group,
                    "isolate": isolate,
                    "components_present": components_present,
                    "expected_component_count": int(expected_component_count),
                    "full_expected_component_set": bool(full_set),
                    "unique_target_isolates": int(isolate_edges["target_isolate"].nunique()),
                    "dominant_target_isolate": top_target,
                    "target_agreement_fraction": agreement_fraction,
                    "neighbor_discordance_score": raw_discordance,
                    "maximum_possible_neighbor_discordance": max_discordance,
                    "normalized_neighbor_discordance_score": normalized_discordance,
                }
            )

        source_scores = pd.DataFrame([row for row in discordance_rows if row["source_group"] == source_group])
        full_scores = source_scores[source_scores["full_expected_component_set"]] if not source_scores.empty else pd.DataFrame()
        summary_rows.append(
            {
                "source_group": source_group,
                "n_component_records_used": int(len(group_df)),
                "n_informative_isolates_scored": int(len(source_scores)),
                "n_full_set_isolates_scored": int(len(full_scores)),
                "n_components_considered": int(edge_df["component_label"].nunique()),
                "median_neighbor_discordance_all_informative": (
                    float(source_scores["neighbor_discordance_score"].median()) if not source_scores.empty else float("nan")
                ),
                "median_normalized_discordance_all_informative": (
                    float(source_scores["normalized_neighbor_discordance_score"].median()) if not source_scores.empty else float("nan")
                ),
                "median_neighbor_discordance_full_sets": (
                    float(full_scores["neighbor_discordance_score"].median()) if not full_scores.empty else float("nan")
                ),
                "median_normalized_discordance_full_sets": (
                    float(full_scores["normalized_neighbor_discordance_score"].median()) if not full_scores.empty else float("nan")
                ),
            }
        )

    concordance_df = pd.DataFrame(discordance_rows)
    if not concordance_df.empty:
        concordance_df = concordance_df.sort_values(
            ["source_group", "normalized_neighbor_discordance_score", "components_present"],
            ascending=[True, False, False],
            ignore_index=True,
        )
        concordance_df["discordance_rank_within_group"] = (
            concordance_df.groupby("source_group")["normalized_neighbor_discordance_score"]
            .rank(method="dense", ascending=False)
        )
    edge_df = pd.DataFrame(edge_rows)
    summary_df = pd.DataFrame(summary_rows)
    return concordance_df, edge_df, summary_df


def paired_component_summary(component_completeness_df: pd.DataFrame) -> pd.DataFrame:
    if component_completeness_df.empty:
        return pd.DataFrame()
    summary = (
        component_completeness_df.groupby(["source_group", "schema_type"], as_index=False)
        .agg(
            n_isolates=("isolate_key", "count"),
            expected_component_count=("expected_component_count", "median"),
            full_set_isolates=("all_expected_components_present", "sum"),
            median_component_count=("component_count", "median"),
            median_completeness_fraction=("completeness_fraction", "median"),
        )
    )
    summary["full_set_fraction"] = summary["full_set_isolates"] / summary["n_isolates"]
    return summary.sort_values(["source_group", "schema_type"], ignore_index=True)
