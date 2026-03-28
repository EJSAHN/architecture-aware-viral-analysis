from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA

from .kmers import compute_kmer_features
from .statistics import correlation_test_summary, group_value_summary, pairwise_value_group_tests


def _standardize(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values)
    if mask.sum() == 0:
        return np.full_like(values, np.nan, dtype=float)
    mean = float(np.nanmean(values))
    sd = float(np.nanstd(values, ddof=0))
    if not np.isfinite(sd) or sd == 0:
        out = np.zeros_like(values, dtype=float)
        out[~mask] = np.nan
        return out
    return (values - mean) / sd


def orthogonal_heterogeneity_axis(
    monopartite_df: pd.DataFrame,
    monopartite_embedding: pd.DataFrame,
    mosaicity_scores: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if monopartite_df.empty or monopartite_embedding.empty or mosaicity_scores.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    merge_cols = ["accession_version", "source_group", "organism", "isolate", "sequence_length", "gc_fraction", "n_fraction", "base_entropy"]
    merge_cols = [c for c in merge_cols if c in monopartite_df.columns]
    base = monopartite_df[merge_cols].drop_duplicates(subset=["accession_version"]).copy()

    pc_cols = [col for col in monopartite_embedding.columns if col.startswith("pc") and col[2:].isdigit()]
    pc_cols = pc_cols[: max(1, int(config["analysis"].get("orthogonal_axis_geometry_components", 3)))]
    embed_subset = monopartite_embedding[["accession_version"] + pc_cols].drop_duplicates(subset=["accession_version"]).copy()

    score_cols = [
        "accession_version",
        "source_group",
        "n_windows",
        "global_nearest_group",
        "dominant_window_group",
        "dominant_window_fraction",
        "window_label_entropy_norm",
        "switch_rate",
        "discordance_fraction",
        "ambiguity_score",
        "compositional_mosaicity_index",
        "mosaicity_rank",
    ]
    score_cols = [c for c in score_cols if c in mosaicity_scores.columns]
    mosaic_subset = mosaicity_scores[score_cols].drop_duplicates(subset=["accession_version"]).copy()

    merged = base.merge(embed_subset, on="accession_version", how="inner").merge(mosaic_subset, on=["accession_version", "source_group"], how="inner")
    if merged.shape[0] < 4:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    merged["length_log10"] = np.log10(pd.to_numeric(merged["sequence_length"], errors="coerce"))
    merged["dominant_window_instability"] = 1.0 - pd.to_numeric(merged.get("dominant_window_fraction", np.nan), errors="coerce")

    feature_priority = [
        "compositional_mosaicity_index",
        "window_label_entropy_norm",
        "switch_rate",
        "discordance_fraction",
        "ambiguity_score",
        "dominant_window_instability",
        "gc_fraction",
        "base_entropy",
        "n_fraction",
        "length_log10",
    ]
    feature_cols = [c for c in feature_priority if c in merged.columns]
    if len(feature_cols) < 2 or len(pc_cols) < 1:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    work = merged.copy()
    for col in feature_cols + pc_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=feature_cols + pc_cols).reset_index(drop=True)
    if work.shape[0] < 4:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    standardized_features = np.column_stack([_standardize(work[col]) for col in feature_cols])
    geometry_matrix = np.column_stack([_standardize(work[col]) for col in pc_cols])
    design = np.column_stack([np.ones(len(work)), geometry_matrix])

    residual_columns: list[np.ndarray] = []
    for idx, _feature_name in enumerate(feature_cols):
        y = standardized_features[:, idx]
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual_columns.append(y - design @ beta)

    residual_matrix = np.column_stack(residual_columns)
    n_components = min(3, residual_matrix.shape[0], residual_matrix.shape[1])
    if n_components < 1:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    pca = PCA(n_components=n_components, random_state=int(config["analysis"]["random_seed"]))
    axis_scores = pca.fit_transform(residual_matrix)

    if "compositional_mosaicity_index" in work.columns and axis_scores.shape[1] >= 1:
        corr = np.corrcoef(axis_scores[:, 0], work["compositional_mosaicity_index"].to_numpy(dtype=float))[0, 1]
        if np.isfinite(corr) and corr < 0:
            axis_scores[:, 0] *= -1
            pca.components_[0, :] *= -1

    score_df = work.copy()
    for component_index in range(axis_scores.shape[1]):
        score_df[f"orthogonal_axis_{component_index + 1}"] = axis_scores[:, component_index]
    score_df["orthogonal_axis_abs_1"] = score_df["orthogonal_axis_1"].abs()
    score_df["orthogonal_axis_rank"] = score_df["orthogonal_axis_1"].rank(method="dense", ascending=False)
    score_df["orthogonal_axis_abs_rank"] = score_df["orthogonal_axis_abs_1"].rank(method="dense", ascending=False)
    score_df = score_df.sort_values(["orthogonal_axis_1", "compositional_mosaicity_index"], ascending=[False, False], ignore_index=True)

    loadings_df = pd.DataFrame({"feature_name": feature_cols})
    for component_index in range(axis_scores.shape[1]):
        loadings_df[f"orthogonal_axis_{component_index + 1}_loading"] = pca.components_[component_index, :]
        loadings_df[f"orthogonal_axis_{component_index + 1}_variance_explained"] = float(pca.explained_variance_ratio_[component_index])

    validation_rows = []
    for pc in pc_cols:
        validation_rows.append(
            correlation_test_summary(
                score_df["orthogonal_axis_1"],
                score_df[pc],
                panel_name="monopartite",
                label_name="orthogonal_axis_validation",
                x_name="orthogonal_axis_1",
                y_name=pc,
                n_bootstrap=int(config["statistics"]["bootstrap_iterations"]),
                random_seed=int(config["analysis"]["random_seed"]),
                method="spearman",
            )
        )
    for metric_name in ["compositional_mosaicity_index", "window_label_entropy_norm", "switch_rate", "discordance_fraction", "ambiguity_score"]:
        if metric_name in score_df.columns:
            validation_rows.append(
                correlation_test_summary(
                    score_df["orthogonal_axis_1"],
                    score_df[metric_name],
                    panel_name="monopartite",
                    label_name="orthogonal_axis_validation",
                    x_name="orthogonal_axis_1",
                    y_name=metric_name,
                    n_bootstrap=int(config["statistics"]["bootstrap_iterations"]),
                    random_seed=int(config["analysis"]["random_seed"]),
                    method="spearman",
                )
            )

    group_summary = group_value_summary(
        score_df,
        value_col="orthogonal_axis_1",
        group_col="source_group",
        panel_name="monopartite",
        label_name="orthogonal_axis_1",
        n_bootstrap=int(config["statistics"]["bootstrap_iterations"]),
        random_seed=int(config["analysis"]["random_seed"]),
    )
    group_tests = pairwise_value_group_tests(
        score_df,
        value_col="orthogonal_axis_1",
        group_col="source_group",
        panel_name="monopartite",
        label_name="orthogonal_axis_1",
        n_permutations=int(config["statistics"]["pairwise_permutations"]),
        n_bootstrap=int(config["statistics"]["bootstrap_iterations"]),
        random_seed=int(config["analysis"]["random_seed"]),
    )
    validation_rows.extend([group_summary, group_tests])

    validation_df = pd.concat([df for df in validation_rows if not df.empty], ignore_index=True) if validation_rows else pd.DataFrame()
    return score_df, loadings_df, validation_df, group_summary


def component_configuration_tension(
    registry_df: pd.DataFrame,
    config: dict[str, Any],
    full_clr_matrix: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    component_df = registry_df.loc[registry_df["panel_componentized"]].copy()
    if component_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    random_seed = int(config["analysis"]["random_seed"])
    min_components = int(config["analysis"].get("min_isolate_components_for_tension", 2))
    n_embed_dims = int(config["analysis"].get("tension_embedding_dimensions", 3))
    k = int(config["analysis"]["kmer_size"])
    pseudocount = float(config["analysis"]["pseudocount"])

    vector_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for source_group, group_df in component_df.groupby("source_group"):
        group_df = group_df[group_df["isolate"].notna() & group_df["normalized_component"].notna()].copy()
        if group_df.empty:
            continue

        group_df = (
            group_df.sort_values(["metadata_score", "sequence_length", "accession_version"], ascending=[False, False, True])
            .drop_duplicates(subset=["isolate", "normalized_component"], keep="first")
            .reset_index(drop=True)
        )
        component_counts = group_df["normalized_component"].value_counts()
        eligible_components = component_counts[component_counts >= 2].index.tolist()
        group_df = group_df[group_df["normalized_component"].isin(eligible_components)].copy().reset_index(drop=True)
        if group_df.empty or group_df["normalized_component"].nunique() < 2:
            continue

        if full_clr_matrix is not None and "feature_row" in group_df.columns:
            clr_matrix = full_clr_matrix[group_df["feature_row"].to_numpy()]
        else:
            _, clr_matrix, _ = compute_kmer_features(
                group_df["sequence"].tolist(),
                k=k,
                collapse_reverse_complements=bool(config["analysis"]["reverse_complement_collapse"]),
                pseudocount=pseudocount,
            )

        n_components = min(max(2, n_embed_dims), clr_matrix.shape[0], clr_matrix.shape[1])
        if n_components < 2:
            continue
        pca = PCA(n_components=n_components, random_state=random_seed)
        coords = pca.fit_transform(clr_matrix)

        coord_cols = [f"coord_{i + 1}" for i in range(coords.shape[1])]
        group_df = group_df.copy()
        for idx, col in enumerate(coord_cols):
            group_df[col] = coords[:, idx]

        group_vector_rows = []
        for component_label, comp_df in group_df.groupby("normalized_component"):
            comp_coords = comp_df[coord_cols].to_numpy(dtype=float)
            if comp_df["isolate"].nunique() < 2:
                continue
            distances = cdist(comp_coords, comp_coords, metric="euclidean")
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

                vector = comp_coords[target_index] - comp_coords[i]
                vector_length = float(np.linalg.norm(vector))
                unit_vector = vector / vector_length if vector_length > 0 else np.zeros_like(vector)
                row = {
                    "source_group": source_group,
                    "component_label": component_label,
                    "accession_version": accessions[i],
                    "isolate": isolates[i],
                    "target_accession_version": accessions[target_index],
                    "target_isolate": isolates[target_index],
                    "vector_length": vector_length,
                }
                for axis_index, col in enumerate(coord_cols):
                    row[col] = float(comp_coords[i, axis_index])
                    row[f"vector_{axis_index + 1}"] = float(vector[axis_index])
                    row[f"unit_vector_{axis_index + 1}"] = float(unit_vector[axis_index])
                group_vector_rows.append(row)

        if not group_vector_rows:
            continue
        group_vector_df = pd.DataFrame(group_vector_rows)
        vector_rows.extend(group_vector_rows)

        positive_lengths = group_vector_df.loc[group_vector_df["vector_length"] > 0, "vector_length"]
        group_scale = float(positive_lengths.median()) if not positive_lengths.empty else 1.0
        if not np.isfinite(group_scale) or group_scale <= 0:
            group_scale = 1.0

        for isolate, iso_df in group_vector_df.groupby("isolate"):
            components_present = int(iso_df["component_label"].nunique())
            if components_present < min_components:
                continue
            pos = iso_df[coord_cols].to_numpy(dtype=float)
            vec_cols = [f"vector_{i + 1}" for i in range(len(coord_cols))]
            vecs = iso_df[vec_cols].to_numpy(dtype=float)
            lengths = iso_df["vector_length"].to_numpy(dtype=float)
            if np.isfinite(lengths).sum() == 0:
                continue

            safe_lengths = lengths.copy()
            safe_lengths[safe_lengths <= 0] = np.nan
            units = np.divide(vecs, safe_lengths[:, None], out=np.zeros_like(vecs), where=np.isfinite(safe_lengths[:, None]))
            directional_resultant = float(np.linalg.norm(np.nanmean(units, axis=0)))
            directional_tension = 1.0 - directional_resultant

            mean_length = float(np.nanmean(lengths))
            length_cv = float(np.nanstd(lengths, ddof=1) / mean_length) if components_present > 1 and mean_length > 0 else 0.0
            length_irregularity = length_cv / (1.0 + length_cv)

            centroid = pos.mean(axis=0)
            spread = float(np.linalg.norm(pos - centroid, axis=1).mean())
            spread_norm = spread / group_scale if group_scale > 0 else spread
            spread_score = spread_norm / (1.0 + spread_norm)
            pairwise_mean_distance = float(pdist(pos, metric="euclidean").mean()) if pos.shape[0] > 1 else 0.0

            tension_index = float(np.mean([directional_tension, length_irregularity, spread_score]))
            score_rows.append(
                {
                    "source_group": source_group,
                    "isolate": isolate,
                    "components_present": components_present,
                    "mean_vector_length": mean_length,
                    "vector_length_cv": length_cv,
                    "directional_tension": directional_tension,
                    "spread_score": spread_score,
                    "mean_centroid_spread": spread,
                    "mean_pairwise_component_distance": pairwise_mean_distance,
                    "component_configuration_tension_index": tension_index,
                }
            )

        source_scores = pd.DataFrame([row for row in score_rows if row["source_group"] == source_group])
        if source_scores.empty:
            continue
        tension_values = source_scores["component_configuration_tension_index"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "source_group": source_group,
                "n_isolates_scored": int(source_scores.shape[0]),
                "mean_tension": float(np.nanmean(tension_values)),
                "median_tension": float(np.nanmedian(tension_values)),
                "sd_tension": float(np.nanstd(tension_values, ddof=1)) if source_scores.shape[0] > 1 else 0.0,
                "median_directional_tension": float(np.nanmedian(source_scores["directional_tension"])),
                "median_spread_score": float(np.nanmedian(source_scores["spread_score"])),
                "median_vector_length_cv": float(np.nanmedian(source_scores["vector_length_cv"])),
                "embedding_dimensions_used": int(len(coord_cols)),
                "group_vector_length_scale": group_scale,
            }
        )

    scores_df = pd.DataFrame(score_rows)
    vector_df = pd.DataFrame(vector_rows)
    summary_df = pd.DataFrame(summary_rows)

    if not scores_df.empty:
        scores_df = scores_df.sort_values(
            ["source_group", "component_configuration_tension_index", "directional_tension"],
            ascending=[True, False, False],
            ignore_index=True,
        )
        scores_df["tension_rank_within_group"] = (
            scores_df.groupby("source_group")["component_configuration_tension_index"].rank(method="dense", ascending=False)
        )
    return scores_df, vector_df, summary_df
