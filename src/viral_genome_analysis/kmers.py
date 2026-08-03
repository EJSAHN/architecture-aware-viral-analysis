from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

from .utils import reverse_complement


def build_kmer_vocabulary(k: int = 4, collapse_reverse_complements: bool = True) -> tuple[list[str], dict[str, int]]:
    kmers = ["".join(chars) for chars in product("ACGT", repeat=k)]
    canonical_labels: list[str] = []
    index: dict[str, int] = {}
    seen: set[str] = set()

    for kmer in kmers:
        label = min(kmer, reverse_complement(kmer)) if collapse_reverse_complements else kmer
        if label not in seen:
            seen.add(label)
            canonical_labels.append(label)
        index[kmer] = canonical_labels.index(label)
    return canonical_labels, index


def sequence_kmer_counts(sequence: str, k: int, mapping: dict[str, int], vocab_size: int) -> np.ndarray:
    counts = np.zeros(vocab_size, dtype=float)
    sequence = sequence.upper()
    if len(sequence) < k:
        return counts
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i : i + k]
        if set(kmer) <= set("ACGT"):
            counts[mapping[kmer]] += 1.0
    return counts


def compute_kmer_features(
    sequences: Iterable[str],
    k: int = 4,
    collapse_reverse_complements: bool = True,
    pseudocount: float = 0.5,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    vocabulary, mapping = build_kmer_vocabulary(k=k, collapse_reverse_complements=collapse_reverse_complements)
    n_features = len(vocabulary)
    count_rows: list[np.ndarray] = []

    for sequence in sequences:
        count_rows.append(sequence_kmer_counts(str(sequence), k, mapping, n_features))

    counts = np.vstack(count_rows)
    frequencies = counts + float(pseudocount)
    frequencies = frequencies / frequencies.sum(axis=1, keepdims=True)
    clr = np.log(frequencies) - np.log(frequencies).mean(axis=1, keepdims=True)
    freq_df = pd.DataFrame(frequencies, columns=vocabulary)
    return freq_df, clr, vocabulary


def pca_coordinates(
    df: pd.DataFrame,
    clr_matrix: np.ndarray,
    n_components: int = 8,
    panel_name: str = "all_curated",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_components = max(2, min(n_components, clr_matrix.shape[0], clr_matrix.shape[1]))
    pca = PCA(n_components=n_components, random_state=7)
    coords = pca.fit_transform(clr_matrix)

    embed = df.copy().reset_index(drop=True)
    embed["panel_name"] = panel_name
    for idx in range(coords.shape[1]):
        embed[f"pc{idx + 1}"] = coords[:, idx]
        embed[f"pc{idx + 1}_variance_explained"] = float(pca.explained_variance_ratio_[idx])

    variance_table = pd.DataFrame(
        {
            "panel_name": panel_name,
            "component_index": [f"PC{i + 1}" for i in range(coords.shape[1])],
            "variance_explained": pca.explained_variance_ratio_,
            "cumulative_variance_explained": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    return embed, variance_table


def nearest_neighbor_table(
    df: pd.DataFrame,
    clr_matrix: np.ndarray,
    label_col: str,
    panel_name: str,
) -> pd.DataFrame:
    if len(df) < 2:
        return pd.DataFrame(columns=[
            "panel_name",
            "accession_version",
            "label",
            "nearest_any_accession",
            "nearest_any_label",
            "nearest_any_distance",
            "nearest_same_label_accession",
            "nearest_same_label_distance",
            "nearest_other_label_accession",
            "nearest_other_label",
            "nearest_other_label_distance",
        ])

    distances = squareform(pdist(clr_matrix, metric="euclidean"))
    np.fill_diagonal(distances, np.inf)
    labels = df[label_col].astype(str).tolist()
    accessions = df["accession_version"].astype(str).tolist()

    rows: list[dict[str, object]] = []
    for i in range(len(df)):
        nearest_any_idx = int(np.argmin(distances[i]))
        same_candidates = [j for j in range(len(df)) if labels[j] == labels[i] and j != i]
        other_candidates = [j for j in range(len(df)) if labels[j] != labels[i]]

        nearest_same_idx = min(same_candidates, key=lambda j: distances[i, j]) if same_candidates else None
        nearest_other_idx = min(other_candidates, key=lambda j: distances[i, j]) if other_candidates else None

        rows.append(
            {
                "panel_name": panel_name,
                "accession_version": accessions[i],
                "label": labels[i],
                "nearest_any_accession": accessions[nearest_any_idx],
                "nearest_any_label": labels[nearest_any_idx],
                "nearest_any_distance": float(distances[i, nearest_any_idx]),
                "nearest_same_label_accession": accessions[nearest_same_idx] if nearest_same_idx is not None else "",
                "nearest_same_label_distance": float(distances[i, nearest_same_idx]) if nearest_same_idx is not None else np.nan,
                "nearest_other_label_accession": accessions[nearest_other_idx] if nearest_other_idx is not None else "",
                "nearest_other_label": labels[nearest_other_idx] if nearest_other_idx is not None else "",
                "nearest_other_label_distance": float(distances[i, nearest_other_idx]) if nearest_other_idx is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def distance_metric_sensitivity(
    panel_name: str,
    df: pd.DataFrame,
    frequency_df: pd.DataFrame,
    clr_matrix: np.ndarray,
) -> pd.DataFrame:
    if len(df) < 3:
        return pd.DataFrame(
            [{"panel_name": panel_name, "n_records": len(df), "pearson_r": np.nan, "spearman_r": np.nan}]
        )
    aitchison = pdist(clr_matrix, metric="euclidean")
    jensen_shannon = pdist(frequency_df.to_numpy(), metric="jensenshannon")
    pearson_r, pearson_p = pearsonr(aitchison, jensen_shannon)
    spearman_r, spearman_p = spearmanr(aitchison, jensen_shannon)
    return pd.DataFrame(
        [
            {
                "panel_name": panel_name,
                "n_records": len(df),
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
            }
        ]
    )
