from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .excel import write_csv_bundle, write_workbook
from .extensions import component_configuration_tension, orthogonal_heterogeneity_axis
from .io import load_curation_results
from .kmers import compute_kmer_features, distance_metric_sensitivity, nearest_neighbor_table, pca_coordinates
from .novelty import compositional_mosaicity, component_neighbor_concordance, paired_component_summary
from .panels import build_panel_registry
from .statistics import (
    bootstrap_centroid_distances,
    correlation_test_summary,
    dispersion_test,
    group_value_summary,
    pairwise_dispersion_tests,
    pairwise_permutation_manova,
    pairwise_value_group_tests,
    panel_distance_summary,
    permutation_manova,
    repeated_cv_classification,
    silhouette_tables,
)
from .utils import ensure_dir, flatten_config

LOGGER = logging.getLogger(__name__)


def _prepare_logging(output_dir: Path, verbose: bool = False) -> None:
    ensure_dir(output_dir)
    log_level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.handlers = []
    root.setLevel(log_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "analysis.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    root.addHandler(file_handler)


def _core_registry_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "accession_version",
        "organism",
        "source_group",
        "schema_type",
        "normalized_component",
        "isolate",
        "geo_loc_name",
        "sequence_length",
        "gc_fraction",
        "base_entropy",
        "panel_all_curated",
        "panel_monopartite",
        "panel_componentized",
        "analysis_unit",
        "relative_file_path",
    ]
    keep = [col for col in preferred if col in df.columns]
    return df[keep].copy().sort_values(["source_group", "schema_type", "normalized_component", "accession_version"], ignore_index=True)


def _panel_centroid_table(embed_df: pd.DataFrame, label_col: str, panel_name: str) -> pd.DataFrame:
    if embed_df.empty:
        return pd.DataFrame()
    pc_cols = [c for c in embed_df.columns if c.startswith("pc") and c[2:].isdigit()]
    if not pc_cols:
        return pd.DataFrame()
    grouped = embed_df.groupby(label_col, as_index=False)[pc_cols].mean()
    grouped.insert(0, "panel_name", panel_name)
    grouped = grouped.rename(columns={label_col: "group_label"})
    return grouped


def run_pipeline(
    curation_results_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    verbose: bool = False,
) -> dict[str, Path]:
    output_root = Path(output_dir)
    _prepare_logging(output_root, verbose=verbose)
    config = load_config(config_path)

    LOGGER.info("Loading curated results directory: %s", curation_results_dir)
    loaded = load_curation_results(curation_results_dir, config)
    curated = loaded["curated_records"].copy()
    component_completeness = loaded["component_completeness"].copy()
    component_matrix = loaded["component_matrix"].copy()

    LOGGER.info("Building panel registry")
    registry_df, panel_counts, grouped_counts = build_panel_registry(curated, config)

    analysis_inputs = pd.DataFrame(
        flatten_config("config", config)
        + [
            {"key": "input.curation_results_dir", "value": str(Path(curation_results_dir).resolve())},
            {"key": "input.metadata_workbook", "value": str(loaded["metadata_workbook"])},
            {"key": "input.sequence_fasta", "value": str(loaded["sequence_fasta"])},
        ]
        + [{"key": f"curation_manifest.{k}", "value": str(v)} for k, v in loaded["run_manifest"].items()]
    )

    LOGGER.info("Computing k-mer feature space for all curated records")
    registry_df = registry_df.reset_index(drop=True)
    registry_df["feature_row"] = np.arange(len(registry_df))
    all_freq, all_clr, _ = compute_kmer_features(
        registry_df["sequence"].tolist(),
        k=int(config["analysis"]["kmer_size"]),
        collapse_reverse_complements=bool(config["analysis"]["reverse_complement_collapse"]),
        pseudocount=float(config["analysis"]["pseudocount"]),
    )
    all_embed, all_var = pca_coordinates(
        registry_df,
        all_clr,
        n_components=int(config["analysis"]["pca_components"]),
        panel_name="all_curated",
    )
    sensitivity_tables = [distance_metric_sensitivity("all_curated", registry_df, all_freq, all_clr)]
    panel_centroids = [_panel_centroid_table(all_embed, "analysis_unit", "all_curated")]

    # Monopartite analysis -------------------------------------------------
    mono_df = registry_df[registry_df["panel_monopartite"]].copy().reset_index(drop=True)
    mono_embed = pd.DataFrame()
    mono_var = pd.DataFrame()
    mono_neighbors = pd.DataFrame()
    mono_mosaic_scores = pd.DataFrame()
    mono_mosaic_windows = pd.DataFrame()
    mono_orthogonal_scores = pd.DataFrame()
    mono_orthogonal_loadings = pd.DataFrame()
    mono_orthogonal_validation = pd.DataFrame()
    mono_orthogonal_group_summary = pd.DataFrame()

    global_tests: list[pd.DataFrame] = []
    pairwise_tests: list[pd.DataFrame] = []
    dispersion_summaries: list[pd.DataFrame] = []
    pairwise_dispersion: list[pd.DataFrame] = []
    dispersion_records: list[pd.DataFrame] = []
    silhouette_summary: list[pd.DataFrame] = []
    silhouette_groups: list[pd.DataFrame] = []
    classification_summary: list[pd.DataFrame] = []
    classification_folds: list[pd.DataFrame] = []
    classification_perm: list[pd.DataFrame] = []
    centroid_boot: list[pd.DataFrame] = []
    distance_summaries: list[pd.DataFrame] = []

    if not mono_df.empty and mono_df["source_group"].nunique() >= 2:
        LOGGER.info("Running monopartite panel analysis on %s records", len(mono_df))
        mono_idx = mono_df["feature_row"].to_numpy()
        mono_freq = all_freq.iloc[mono_idx].reset_index(drop=True)
        mono_clr = all_clr[mono_idx]
        mono_embed, mono_var = pca_coordinates(
            mono_df,
            mono_clr,
            n_components=int(config["analysis"]["pca_components"]),
            panel_name="monopartite",
        )
        panel_centroids.append(_panel_centroid_table(mono_embed, "source_group", "monopartite"))
        mono_neighbors = nearest_neighbor_table(mono_df, mono_clr, label_col="source_group", panel_name="monopartite")
        sensitivity_tables.append(distance_metric_sensitivity("monopartite", mono_df, mono_freq, mono_clr))

        global_tests.append(
            permutation_manova(
                mono_clr,
                mono_df["source_group"].to_numpy(),
                panel_name="monopartite",
                label_name="source_group",
                n_permutations=int(config["statistics"]["global_permutations"]),
                random_seed=int(config["analysis"]["random_seed"]),
            )
        )
        pairwise_tests.append(
            pairwise_permutation_manova(
                mono_clr,
                mono_df["source_group"].to_numpy(),
                panel_name="monopartite",
                label_name="source_group",
                n_permutations=int(config["statistics"]["pairwise_permutations"]),
                random_seed=int(config["analysis"]["random_seed"]),
            )
        )
        disp_summary, disp_records = dispersion_test(
            mono_clr,
            mono_df["source_group"].to_numpy(),
            panel_name="monopartite",
            label_name="source_group",
            n_permutations=int(config["statistics"]["global_permutations"]),
            random_seed=int(config["analysis"]["random_seed"]),
        )
        dispersion_summaries.append(disp_summary)
        dispersion_records.append(disp_records)
        pairwise_dispersion.append(
            pairwise_dispersion_tests(
                mono_clr,
                mono_df["source_group"].to_numpy(),
                panel_name="monopartite",
                label_name="source_group",
                n_permutations=int(config["statistics"]["pairwise_permutations"]),
                random_seed=int(config["analysis"]["random_seed"]),
            )
        )
        sil_summary, sil_groups = silhouette_tables(
            mono_clr, mono_df["source_group"].to_numpy(), panel_name="monopartite", label_name="source_group"
        )
        silhouette_summary.append(sil_summary)
        silhouette_groups.append(sil_groups)
        cv_summary, cv_folds, cv_perm = repeated_cv_classification(
            mono_clr,
            mono_df["source_group"].to_numpy(),
            panel_name="monopartite",
            label_name="source_group",
            n_splits=int(config["statistics"]["cv_folds"]),
            n_repeats=int(config["statistics"]["cv_repeats"]),
            n_permutations=int(config["statistics"]["classification_permutations"]),
            random_seed=int(config["analysis"]["random_seed"]),
        )
        classification_summary.append(cv_summary)
        classification_folds.append(cv_folds)
        classification_perm.append(cv_perm)
        centroid_boot.append(
            bootstrap_centroid_distances(
                mono_clr,
                mono_df["source_group"].to_numpy(),
                panel_name="monopartite",
                label_name="source_group",
                n_bootstrap=int(config["statistics"]["bootstrap_iterations"]),
                random_seed=int(config["analysis"]["random_seed"]),
            )
        )
        distance_summaries.append(
            panel_distance_summary(
                mono_clr,
                mono_df["source_group"].to_numpy(),
                panel_name="monopartite",
                label_name="source_group",
            )
        )
        mono_mosaic_scores, mono_mosaic_windows = compositional_mosaicity(mono_df, "source_group", config, clr_matrix=mono_clr)
        mono_orthogonal_scores, mono_orthogonal_loadings, mono_orthogonal_validation, mono_orthogonal_group_summary = orthogonal_heterogeneity_axis(
            mono_df,
            mono_embed,
            mono_mosaic_scores,
            config,
        )

    # Componentized analysis within each source group ----------------------
    component_df = registry_df[registry_df["panel_componentized"]].copy().reset_index(drop=True)
    component_embeds: list[pd.DataFrame] = []
    component_vars: list[pd.DataFrame] = []
    component_neighbors: list[pd.DataFrame] = []
    component_balance_rows: list[dict[str, object]] = []

    if not component_df.empty:
        LOGGER.info("Running componentized panel analysis across %s records", len(component_df))
        min_component_records = int(config["analysis"]["min_component_records"])
        for source_group, source_df in component_df.groupby("source_group"):
            eligible_counts = source_df["normalized_component"].value_counts()
            eligible_components = eligible_counts[eligible_counts >= min_component_records].index.tolist()
            source_df = source_df[source_df["normalized_component"].isin(eligible_components)].copy().reset_index(drop=True)
            if source_df["normalized_component"].nunique() < 2:
                continue

            panel_name = f"componentized:{source_group}"
            source_idx = source_df["feature_row"].to_numpy()
            freq_df = all_freq.iloc[source_idx].reset_index(drop=True)
            clr_matrix = all_clr[source_idx]
            embed_df, var_df = pca_coordinates(
                source_df,
                clr_matrix,
                n_components=int(config["analysis"]["pca_components"]),
                panel_name=panel_name,
            )
            component_embeds.append(embed_df)
            component_vars.append(var_df)
            component_neighbors.append(
                nearest_neighbor_table(source_df, clr_matrix, label_col="normalized_component", panel_name=panel_name)
            )
            sensitivity_tables.append(distance_metric_sensitivity(panel_name, source_df, freq_df, clr_matrix))
            panel_centroids.append(_panel_centroid_table(embed_df, "normalized_component", panel_name))

            global_tests.append(
                permutation_manova(
                    clr_matrix,
                    source_df["normalized_component"].to_numpy(),
                    panel_name=panel_name,
                    label_name="normalized_component",
                    n_permutations=int(config["statistics"]["global_permutations"]),
                    random_seed=int(config["analysis"]["random_seed"]),
                )
            )
            pairwise_tests.append(
                pairwise_permutation_manova(
                    clr_matrix,
                    source_df["normalized_component"].to_numpy(),
                    panel_name=panel_name,
                    label_name="normalized_component",
                    n_permutations=int(config["statistics"]["pairwise_permutations"]),
                    random_seed=int(config["analysis"]["random_seed"]),
                )
            )
            disp_summary, disp_records = dispersion_test(
                clr_matrix,
                source_df["normalized_component"].to_numpy(),
                panel_name=panel_name,
                label_name="normalized_component",
                n_permutations=int(config["statistics"]["global_permutations"]),
                random_seed=int(config["analysis"]["random_seed"]),
            )
            dispersion_summaries.append(disp_summary)
            dispersion_records.append(disp_records)
            pairwise_dispersion.append(
                pairwise_dispersion_tests(
                    clr_matrix,
                    source_df["normalized_component"].to_numpy(),
                    panel_name=panel_name,
                    label_name="normalized_component",
                    n_permutations=int(config["statistics"]["pairwise_permutations"]),
                    random_seed=int(config["analysis"]["random_seed"]),
                )
            )
            sil_summary, sil_groups = silhouette_tables(
                clr_matrix,
                source_df["normalized_component"].to_numpy(),
                panel_name=panel_name,
                label_name="normalized_component",
            )
            silhouette_summary.append(sil_summary)
            silhouette_groups.append(sil_groups)
            cv_summary, cv_folds, cv_perm = repeated_cv_classification(
                clr_matrix,
                source_df["normalized_component"].to_numpy(),
                panel_name=panel_name,
                label_name="normalized_component",
                n_splits=int(config["statistics"]["cv_folds"]),
                n_repeats=int(config["statistics"]["cv_repeats"]),
                n_permutations=int(config["statistics"]["classification_permutations"]),
                random_seed=int(config["analysis"]["random_seed"]),
            )
            classification_summary.append(cv_summary)
            classification_folds.append(cv_folds)
            classification_perm.append(cv_perm)
            centroid_boot.append(
                bootstrap_centroid_distances(
                    clr_matrix,
                    source_df["normalized_component"].to_numpy(),
                    panel_name=panel_name,
                    label_name="normalized_component",
                    n_bootstrap=int(config["statistics"]["bootstrap_iterations"]),
                    random_seed=int(config["analysis"]["random_seed"]),
                )
            )
            distance_summaries.append(
                panel_distance_summary(
                    clr_matrix,
                    source_df["normalized_component"].to_numpy(),
                    panel_name=panel_name,
                    label_name="normalized_component",
                )
            )
            component_balance_rows.append(
                {
                    "source_group": source_group,
                    "n_records_used": len(source_df),
                    "n_components_tested": source_df["normalized_component"].nunique(),
                    "components_tested": "; ".join(sorted(source_df["normalized_component"].unique().tolist())),
                }
            )

    LOGGER.info("Scoring component-neighbor concordance")
    component_concordance, component_edges, component_group_summary = component_neighbor_concordance(registry_df, config, full_clr_matrix=all_clr)

    LOGGER.info("Scoring component-configuration tension")
    component_tension_scores, component_tension_vectors, component_tension_source_summary = component_configuration_tension(
        registry_df,
        config,
        full_clr_matrix=all_clr,
    )
    component_tension_group_summary = pd.DataFrame()
    component_tension_group_tests = pd.DataFrame()
    component_tension_correlations = pd.DataFrame()
    if not component_tension_scores.empty:
        component_tension_group_summary = group_value_summary(
            component_tension_scores,
            value_col="component_configuration_tension_index",
            group_col="source_group",
            panel_name="componentized",
            label_name="component_configuration_tension_index",
            n_bootstrap=int(config["statistics"]["bootstrap_iterations"]),
            random_seed=int(config["analysis"]["random_seed"]),
        )
        component_tension_group_tests = pairwise_value_group_tests(
            component_tension_scores,
            value_col="component_configuration_tension_index",
            group_col="source_group",
            panel_name="componentized",
            label_name="component_configuration_tension_index",
            n_permutations=int(config["statistics"]["pairwise_permutations"]),
            n_bootstrap=int(config["statistics"]["bootstrap_iterations"]),
            random_seed=int(config["analysis"]["random_seed"]),
        )

    if not component_tension_scores.empty and not component_concordance.empty:
        concordance_cols = [
            col
            for col in ["source_group", "isolate", "neighbor_discordance_score", "target_agreement_fraction", "unique_target_isolates"]
            if col in component_concordance.columns
        ]
        component_tension_scores = component_tension_scores.merge(component_concordance[concordance_cols], on=["source_group", "isolate"], how="left")
        corr_tables = []
        if "neighbor_discordance_score" in component_tension_scores.columns:
            corr_tables.append(
                correlation_test_summary(
                    component_tension_scores["component_configuration_tension_index"],
                    component_tension_scores["neighbor_discordance_score"],
                    panel_name="componentized",
                    label_name="component_tension_vs_concordance",
                    x_name="component_configuration_tension_index",
                    y_name="neighbor_discordance_score",
                    n_bootstrap=int(config["statistics"]["bootstrap_iterations"]),
                    random_seed=int(config["analysis"]["random_seed"]),
                    method="spearman",
                )
            )
            for source_group, sub in component_tension_scores.groupby("source_group"):
                corr_tables.append(
                    correlation_test_summary(
                        sub["component_configuration_tension_index"],
                        sub["neighbor_discordance_score"],
                        panel_name=f"componentized:{source_group}",
                        label_name="component_tension_vs_concordance",
                        x_name="component_configuration_tension_index",
                        y_name="neighbor_discordance_score",
                        n_bootstrap=int(config["statistics"]["bootstrap_iterations"]),
                        random_seed=int(config["analysis"]["random_seed"]),
                        method="spearman",
                    )
                )
        component_tension_correlations = pd.concat([df for df in corr_tables if not df.empty], ignore_index=True) if corr_tables else pd.DataFrame()

    paired_summary = paired_component_summary(component_completeness)

    registry_sheets = {
        "analysis_inputs": analysis_inputs,
        "analysis_registry": _core_registry_columns(registry_df),
        "panel_counts": panel_counts,
        "grouped_counts": grouped_counts,
        "distance_metric_sensitivity": pd.concat(sensitivity_tables, ignore_index=True) if sensitivity_tables else pd.DataFrame(),
    }

    geometry_sheets = {
        "all_embedding": all_embed,
        "all_variance_explained": all_var,
        "monopartite_embedding": mono_embed,
        "monopartite_variance_explained": mono_var,
        "monopartite_neighbors": mono_neighbors,
        "component_embedding": pd.concat(component_embeds, ignore_index=True) if component_embeds else pd.DataFrame(),
        "component_variance_explained": pd.concat(component_vars, ignore_index=True) if component_vars else pd.DataFrame(),
        "component_neighbors": pd.concat(component_neighbors, ignore_index=True) if component_neighbors else pd.DataFrame(),
        "panel_centroids": pd.concat([df for df in panel_centroids if not df.empty], ignore_index=True) if panel_centroids else pd.DataFrame(),
    }

    stats_sheets = {
        "global_permutation_tests": pd.concat(global_tests, ignore_index=True) if global_tests else pd.DataFrame(),
        "pairwise_permutation_tests": pd.concat(pairwise_tests, ignore_index=True) if pairwise_tests else pd.DataFrame(),
        "dispersion_tests": pd.concat(dispersion_summaries, ignore_index=True) if dispersion_summaries else pd.DataFrame(),
        "pairwise_dispersion_tests": pd.concat(pairwise_dispersion, ignore_index=True) if pairwise_dispersion else pd.DataFrame(),
        "dispersion_records": pd.concat(dispersion_records, ignore_index=True) if dispersion_records else pd.DataFrame(),
        "silhouette_summary": pd.concat(silhouette_summary, ignore_index=True) if silhouette_summary else pd.DataFrame(),
        "silhouette_by_group": pd.concat(silhouette_groups, ignore_index=True) if silhouette_groups else pd.DataFrame(),
        "classification_summary": pd.concat(classification_summary, ignore_index=True) if classification_summary else pd.DataFrame(),
        "classification_folds": pd.concat(classification_folds, ignore_index=True) if classification_folds else pd.DataFrame(),
        "classification_permutation_null": pd.concat(classification_perm, ignore_index=True) if classification_perm else pd.DataFrame(),
        "centroid_distance_bootstrap": pd.concat(centroid_boot, ignore_index=True) if centroid_boot else pd.DataFrame(),
        "distance_summaries": pd.concat(distance_summaries, ignore_index=True) if distance_summaries else pd.DataFrame(),
        "orthogonal_axis_validation": mono_orthogonal_validation,
        "orthogonal_axis_group_summary": mono_orthogonal_group_summary,
        "component_tension_group_summary": component_tension_group_summary,
        "component_tension_group_tests": component_tension_group_tests,
        "component_tension_correlations": component_tension_correlations,
    }

    novelty_sheets = {
        "mosaicity_scores": mono_mosaic_scores,
        "mosaicity_windows": mono_mosaic_windows,
        "orthogonal_axis_scores": mono_orthogonal_scores,
        "orthogonal_axis_loadings": mono_orthogonal_loadings,
        "component_concordance": component_concordance,
        "component_neighbor_targets": component_edges,
        "component_group_summary": component_group_summary,
        "component_tension_scores": component_tension_scores,
        "component_tension_vectors": component_tension_vectors,
        "component_tension_source_summary": component_tension_source_summary,
    }

    architecture_sheets = {
        "component_completeness_input": component_completeness,
        "component_matrix_input": component_matrix,
        "paired_component_summary": paired_summary,
        "component_balance_analysis": pd.DataFrame(component_balance_rows),
    }

    tables_dir = output_root / "tables"
    csv_dir = output_root / "csv"

    registry_xlsx = write_workbook(registry_sheets, tables_dir / "01_analysis_registry.xlsx")
    geometry_xlsx = write_workbook(geometry_sheets, tables_dir / "02_geometry_embeddings.xlsx")
    stats_xlsx = write_workbook(stats_sheets, tables_dir / "03_statistical_validation.xlsx")
    novelty_xlsx = write_workbook(novelty_sheets, tables_dir / "04_novelty_signals.xlsx")
    architecture_xlsx = write_workbook(architecture_sheets, tables_dir / "05_component_architecture.xlsx")

    write_csv_bundle(registry_sheets, csv_dir / "analysis_registry")
    write_csv_bundle(geometry_sheets, csv_dir / "geometry_embeddings")
    write_csv_bundle(stats_sheets, csv_dir / "statistical_validation")
    write_csv_bundle(novelty_sheets, csv_dir / "novelty_signals")
    write_csv_bundle(architecture_sheets, csv_dir / "component_architecture")

    manifest = {
        "curation_results_dir": str(Path(curation_results_dir).resolve()),
        "output_dir": str(output_root.resolve()),
        "n_curated_records_input": int(len(registry_df)),
        "n_monopartite_records_analyzed": int(len(mono_df)),
        "n_componentized_records_analyzed": int(len(component_df)),
        "analysis_registry_workbook": str(registry_xlsx),
        "geometry_workbook": str(geometry_xlsx),
        "stats_workbook": str(stats_xlsx),
        "novelty_workbook": str(novelty_xlsx),
        "component_architecture_workbook": str(architecture_xlsx),
    }
    (output_root / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_root / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    LOGGER.info("Analysis pipeline complete for %s curated records", len(registry_df))
    return {
        "analysis_registry_workbook": registry_xlsx,
        "geometry_workbook": geometry_xlsx,
        "stats_workbook": stats_xlsx,
        "novelty_workbook": novelty_xlsx,
        "component_architecture_workbook": architecture_xlsx,
        "manifest": output_root / "analysis_manifest.json",
    }
