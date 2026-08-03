from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import load_config
from .excel import write_csv_bundle, write_workbook
from .extensions import component_configuration_tension, orthogonal_heterogeneity_axis
from .io import load_curation_results
from .kmers import build_kmer_vocabulary, compute_kmer_features, pca_coordinates
from .novelty import compositional_mosaicity, component_neighbor_concordance
from .panels import build_panel_registry
from .statistics import permutation_manova, repeated_cv_classification, silhouette_tables
from .utils import ensure_dir

LOGGER = logging.getLogger(__name__)


def _parse_positive_ints(values: Iterable[int]) -> list[int]:
    parsed = sorted({int(v) for v in values})
    if not parsed or any(v <= 0 for v in parsed):
        raise ValueError("All sensitivity values must be positive integers.")
    return parsed


def _safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan"), float("nan"), int(len(pair))
    result = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    return float(result.statistic), float(result.pvalue), int(len(pair))


def _top_overlap(df_a: pd.DataFrame, df_b: pd.DataFrame, id_col: str, rank_col: str, top_n: int) -> tuple[int, float]:
    ids_a = set(df_a.nsmallest(top_n, rank_col)[id_col].astype(str))
    ids_b = set(df_b.nsmallest(top_n, rank_col)[id_col].astype(str))
    overlap = len(ids_a & ids_b)
    denom = max(1, min(top_n, len(ids_a), len(ids_b)))
    return overlap, float(overlap / denom)


def _panel_validation(
    X: np.ndarray,
    labels: np.ndarray,
    panel_name: str,
    label_name: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    manova = permutation_manova(
        X,
        labels,
        panel_name=panel_name,
        label_name=label_name,
        n_permutations=int(config["statistics"]["global_permutations"]),
        random_seed=int(config["analysis"]["random_seed"]),
    )
    silhouette, _ = silhouette_tables(X, labels, panel_name=panel_name, label_name=label_name)
    classification, _, _ = repeated_cv_classification(
        X,
        labels,
        panel_name=panel_name,
        label_name=label_name,
        n_splits=int(config["statistics"]["cv_folds"]),
        n_repeats=int(config["statistics"]["cv_repeats"]),
        n_permutations=int(config["statistics"]["classification_permutations"]),
        random_seed=int(config["analysis"]["random_seed"]),
    )
    out = manova.merge(silhouette[["panel_name", "label_name", "silhouette_mean"]], on=["panel_name", "label_name"], how="left")
    out = out.merge(
        classification[
            [
                "panel_name",
                "label_name",
                "accuracy_mean",
                "balanced_accuracy_mean",
                "macro_f1_mean",
                "macro_f1_permutation_p_value",
            ]
        ],
        on=["panel_name", "label_name"],
        how="left",
    )
    return out


def _mosaicity_group_summary(scores: pd.DataFrame, scenario: str, kmer_size: int, window_size: int) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    grouped = (
        scores.groupby("source_group", as_index=False)
        .agg(
            n_genomes=("accession_version", "count"),
            mosaicity_mean=("compositional_mosaicity_index", "mean"),
            mosaicity_median=("compositional_mosaicity_index", "median"),
            mosaicity_max=("compositional_mosaicity_index", "max"),
            median_switch_rate=("switch_rate", "median"),
            median_discordance_fraction=("discordance_fraction", "median"),
        )
    )
    grouped.insert(0, "window_size", int(window_size))
    grouped.insert(0, "kmer_size", int(kmer_size))
    grouped.insert(0, "scenario", scenario)
    return grouped


def run_robustness_analysis(
    curation_results_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    kmer_sizes: Iterable[int] = (4, 5),
    window_sizes: Iterable[int] = (300, 400, 500),
    top_n: int = 20,
) -> dict[str, Path]:
    """Run k-mer and sliding-window sensitivity analyses.

    The primary pipeline is not modified. This function recomputes the principal
    validation metrics under a small, explicit parameter grid and aggregates the
    results into one consolidated workbook.
    """

    output_root = ensure_dir(Path(output_dir))
    config = load_config(config_path)
    k_values = _parse_positive_ints(kmer_sizes)
    window_values = _parse_positive_ints(window_sizes)
    top_n = int(top_n)
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    loaded = load_curation_results(curation_results_dir, config)
    registry_df, _, _ = build_panel_registry(loaded["curated_records"].copy(), config)
    registry_df = registry_df.reset_index(drop=True)
    registry_df["feature_row"] = np.arange(len(registry_df))
    mono_df = registry_df.loc[registry_df["panel_monopartite"]].copy().reset_index(drop=True)

    panel_validation_tables: list[pd.DataFrame] = []
    mosaicity_tables: dict[int, pd.DataFrame] = {}
    orthogonal_tables: dict[int, pd.DataFrame] = {}
    concordance_tables: dict[int, pd.DataFrame] = {}
    tension_tables: dict[int, pd.DataFrame] = {}
    kmer_group_summaries: list[pd.DataFrame] = []

    all_clr_by_k: dict[int, np.ndarray] = {}
    mono_clr_by_k: dict[int, np.ndarray] = {}

    baseline_k_config = int(config["analysis"]["kmer_size"])
    baseline_vocab, _ = build_kmer_vocabulary(
        baseline_k_config,
        bool(config["analysis"]["reverse_complement_collapse"]),
    )
    total_pseudocount_mass = float(config["analysis"]["pseudocount"]) * len(baseline_vocab)
    pseudocount_by_k: dict[int, float] = {}

    for kmer_size in k_values:
        LOGGER.info("Running k-mer sensitivity scenario k=%s", kmer_size)
        scenario_config = deepcopy(config)
        scenario_config["analysis"]["kmer_size"] = int(kmer_size)
        scenario_vocab, _ = build_kmer_vocabulary(
            int(kmer_size),
            bool(scenario_config["analysis"]["reverse_complement_collapse"]),
        )
        scenario_pseudocount = total_pseudocount_mass / len(scenario_vocab)
        scenario_config["analysis"]["pseudocount"] = float(scenario_pseudocount)
        pseudocount_by_k[int(kmer_size)] = float(scenario_pseudocount)

        _, all_clr, _ = compute_kmer_features(
            registry_df["sequence"].tolist(),
            k=kmer_size,
            collapse_reverse_complements=bool(scenario_config["analysis"]["reverse_complement_collapse"]),
            pseudocount=float(scenario_pseudocount),
        )
        all_clr_by_k[kmer_size] = all_clr

        if not mono_df.empty and mono_df["source_group"].nunique() >= 2:
            mono_idx = mono_df["feature_row"].to_numpy()
            mono_clr = all_clr[mono_idx]
            mono_clr_by_k[kmer_size] = mono_clr
            validation = _panel_validation(
                mono_clr,
                mono_df["source_group"].astype(str).to_numpy(),
                panel_name="monopartite",
                label_name="source_group",
                config=scenario_config,
            )
            validation.insert(0, "pseudocount_per_feature", float(scenario_pseudocount))
            validation.insert(0, "kmer_size", kmer_size)
            panel_validation_tables.append(validation)

            mosaicity, _ = compositional_mosaicity(
                mono_df,
                group_col="source_group",
                config=scenario_config,
                clr_matrix=mono_clr,
            )
            if not mosaicity.empty:
                mosaicity = mosaicity.copy()
                mosaicity.insert(0, "pseudocount_per_feature", float(scenario_pseudocount))
                mosaicity.insert(0, "kmer_size", kmer_size)
                mosaicity_tables[kmer_size] = mosaicity
                kmer_group_summaries.append(
                    _mosaicity_group_summary(
                        mosaicity,
                        scenario=f"k{kmer_size}",
                        kmer_size=kmer_size,
                        window_size=int(scenario_config["analysis"]["sliding_window_size"]),
                    )
                )

                mono_embed, _ = pca_coordinates(
                    mono_df,
                    mono_clr,
                    n_components=int(scenario_config["analysis"]["pca_components"]),
                    panel_name="monopartite",
                )
                orthogonal_scores, _, _, _ = orthogonal_heterogeneity_axis(
                    mono_df,
                    mono_embed,
                    mosaicity,
                    scenario_config,
                )
                if not orthogonal_scores.empty:
                    orthogonal_scores = orthogonal_scores.copy()
                    orthogonal_scores.insert(0, "pseudocount_per_feature", float(scenario_pseudocount))
                    orthogonal_scores.insert(0, "kmer_size", kmer_size)
                    orthogonal_tables[kmer_size] = orthogonal_scores

        component_df = registry_df.loc[registry_df["panel_componentized"]].copy().reset_index(drop=True)
        for source_group, group_df in component_df.groupby("source_group"):
            group_idx = group_df["feature_row"].to_numpy()
            group_clr = all_clr[group_idx]
            labels = group_df["normalized_component"].astype(str).to_numpy()
            if len(np.unique(labels)) < 2:
                continue
            validation = _panel_validation(
                group_clr,
                labels,
                panel_name=f"componentized:{source_group}",
                label_name="normalized_component",
                config=scenario_config,
            )
            validation.insert(0, "pseudocount_per_feature", float(scenario_pseudocount))
            validation.insert(0, "kmer_size", kmer_size)
            panel_validation_tables.append(validation)

        concordance, _, _ = component_neighbor_concordance(
            registry_df,
            scenario_config,
            full_clr_matrix=all_clr,
            component_completeness_df=loaded["component_completeness"],
        )
        if not concordance.empty:
            concordance = concordance.copy()
            concordance.insert(0, "pseudocount_per_feature", float(scenario_pseudocount))
            concordance.insert(0, "kmer_size", kmer_size)
            concordance_tables[kmer_size] = concordance

        tension, _, _ = component_configuration_tension(
            registry_df,
            scenario_config,
            full_clr_matrix=all_clr,
            component_completeness_df=loaded["component_completeness"],
        )
        if not tension.empty:
            tension = tension.copy()
            tension.insert(0, "pseudocount_per_feature", float(scenario_pseudocount))
            tension.insert(0, "kmer_size", kmer_size)
            tension_tables[kmer_size] = tension

    baseline_k = int(config["analysis"]["kmer_size"])
    if baseline_k not in k_values:
        baseline_k = k_values[0]

    kmer_stability_rows: list[dict[str, Any]] = []
    baseline_mosaic = mosaicity_tables.get(baseline_k, pd.DataFrame())
    baseline_orthogonal = orthogonal_tables.get(baseline_k, pd.DataFrame())
    baseline_concordance = concordance_tables.get(baseline_k, pd.DataFrame())
    baseline_tension = tension_tables.get(baseline_k, pd.DataFrame())

    for kmer_size in k_values:
        if kmer_size == baseline_k:
            continue
        current = mosaicity_tables.get(kmer_size, pd.DataFrame())
        if not baseline_mosaic.empty and not current.empty:
            merged = baseline_mosaic[["accession_version", "compositional_mosaicity_index", "mosaicity_rank"]].merge(
                current[["accession_version", "compositional_mosaicity_index", "mosaicity_rank"]],
                on="accession_version",
                suffixes=("_baseline", "_alternative"),
            )
            rho, p_value, n = _safe_spearman(
                merged["compositional_mosaicity_index_baseline"],
                merged["compositional_mosaicity_index_alternative"],
            )
            overlap, overlap_fraction = _top_overlap(baseline_mosaic, current, "accession_version", "mosaicity_rank", top_n)
            kmer_stability_rows.append(
                {
                    "metric": "compositional_mosaicity_index",
                    "baseline_k": baseline_k,
                    "alternative_k": kmer_size,
                    "n_matched": n,
                    "spearman_rho": rho,
                    "spearman_p_value": p_value,
                    "top_n": top_n,
                    "top_n_overlap": overlap,
                    "top_n_overlap_fraction": overlap_fraction,
                }
            )

        current = orthogonal_tables.get(kmer_size, pd.DataFrame())
        if not baseline_orthogonal.empty and not current.empty:
            merged = baseline_orthogonal[["accession_version", "orthogonal_axis_1"]].merge(
                current[["accession_version", "orthogonal_axis_1"]],
                on="accession_version",
                suffixes=("_baseline", "_alternative"),
            )
            rho, p_value, n = _safe_spearman(merged["orthogonal_axis_1_baseline"], merged["orthogonal_axis_1_alternative"])
            kmer_stability_rows.append(
                {
                    "metric": "orthogonal_axis_1",
                    "baseline_k": baseline_k,
                    "alternative_k": kmer_size,
                    "n_matched": n,
                    "spearman_rho": rho,
                    "spearman_p_value": p_value,
                    "top_n": np.nan,
                    "top_n_overlap": np.nan,
                    "top_n_overlap_fraction": np.nan,
                }
            )

        current = concordance_tables.get(kmer_size, pd.DataFrame())
        if not baseline_concordance.empty and not current.empty:
            base_subset = baseline_concordance.loc[baseline_concordance["full_expected_component_set"].astype(bool)]
            current_subset = current.loc[current["full_expected_component_set"].astype(bool)]
            merged = base_subset[["source_group", "isolate", "normalized_neighbor_discordance_score"]].merge(
                current_subset[["source_group", "isolate", "normalized_neighbor_discordance_score"]],
                on=["source_group", "isolate"],
                suffixes=("_baseline", "_alternative"),
            )
            rho, p_value, n = _safe_spearman(
                merged["normalized_neighbor_discordance_score_baseline"],
                merged["normalized_neighbor_discordance_score_alternative"],
            )
            kmer_stability_rows.append(
                {
                    "metric": "normalized_neighbor_discordance_score_full_sets",
                    "baseline_k": baseline_k,
                    "alternative_k": kmer_size,
                    "n_matched": n,
                    "spearman_rho": rho,
                    "spearman_p_value": p_value,
                    "top_n": np.nan,
                    "top_n_overlap": np.nan,
                    "top_n_overlap_fraction": np.nan,
                }
            )

        current = tension_tables.get(kmer_size, pd.DataFrame())
        if not baseline_tension.empty and not current.empty:
            base_subset = baseline_tension.loc[baseline_tension["full_expected_component_set"].astype(bool)]
            current_subset = current.loc[current["full_expected_component_set"].astype(bool)]
            merged = base_subset[["source_group", "isolate", "component_configuration_tension_index"]].merge(
                current_subset[["source_group", "isolate", "component_configuration_tension_index"]],
                on=["source_group", "isolate"],
                suffixes=("_baseline", "_alternative"),
            )
            rho, p_value, n = _safe_spearman(
                merged["component_configuration_tension_index_baseline"],
                merged["component_configuration_tension_index_alternative"],
            )
            kmer_stability_rows.append(
                {
                    "metric": "component_configuration_tension_index_full_sets",
                    "baseline_k": baseline_k,
                    "alternative_k": kmer_size,
                    "n_matched": n,
                    "spearman_rho": rho,
                    "spearman_p_value": p_value,
                    "top_n": np.nan,
                    "top_n_overlap": np.nan,
                    "top_n_overlap_fraction": np.nan,
                }
            )

    # Sliding-window sensitivity at the primary k-mer size.
    window_score_tables: dict[int, pd.DataFrame] = {}
    window_group_summaries: list[pd.DataFrame] = []
    primary_mono_clr = mono_clr_by_k.get(baseline_k)
    if primary_mono_clr is None and not mono_df.empty:
        _, primary_mono_clr, _ = compute_kmer_features(
            mono_df["sequence"].tolist(),
            k=baseline_k,
            collapse_reverse_complements=bool(config["analysis"]["reverse_complement_collapse"]),
            pseudocount=float(config["analysis"]["pseudocount"]),
        )

    if primary_mono_clr is not None:
        for window_size in window_values:
            LOGGER.info("Running window sensitivity scenario window=%s", window_size)
            scenario_config = deepcopy(config)
            scenario_config["analysis"]["kmer_size"] = baseline_k
            scenario_config["analysis"]["sliding_window_size"] = int(window_size)
            scores, _ = compositional_mosaicity(
                mono_df,
                group_col="source_group",
                config=scenario_config,
                clr_matrix=primary_mono_clr,
            )
            if scores.empty:
                continue
            scores = scores.copy()
            scores.insert(0, "window_size", window_size)
            window_score_tables[window_size] = scores
            window_group_summaries.append(
                _mosaicity_group_summary(
                    scores,
                    scenario=f"window_{window_size}",
                    kmer_size=baseline_k,
                    window_size=window_size,
                )
            )

    baseline_window = int(config["analysis"]["sliding_window_size"])
    if baseline_window not in window_values:
        baseline_window = window_values[0]
    baseline_window_scores = window_score_tables.get(baseline_window, pd.DataFrame())
    window_stability_rows: list[dict[str, Any]] = []
    candidate_tables: list[pd.DataFrame] = []

    for window_size, scores in window_score_tables.items():
        candidate_subset = scores[
            [
                "accession_version",
                "source_group",
                "compositional_mosaicity_index",
                "mosaicity_rank",
                "window_label_entropy_norm",
                "switch_rate",
                "discordance_fraction",
                "ambiguity_score",
            ]
        ].copy()
        candidate_subset = candidate_subset.rename(
            columns={
                col: f"{col}_w{window_size}"
                for col in candidate_subset.columns
                if col not in {"accession_version", "source_group"}
            }
        )
        candidate_tables.append(candidate_subset)

        if window_size == baseline_window or baseline_window_scores.empty:
            continue
        merged = baseline_window_scores[["accession_version", "compositional_mosaicity_index"]].merge(
            scores[["accession_version", "compositional_mosaicity_index"]],
            on="accession_version",
            suffixes=("_baseline", "_alternative"),
        )
        rho, p_value, n = _safe_spearman(
            merged["compositional_mosaicity_index_baseline"],
            merged["compositional_mosaicity_index_alternative"],
        )
        overlap, overlap_fraction = _top_overlap(
            baseline_window_scores,
            scores,
            "accession_version",
            "mosaicity_rank",
            top_n,
        )
        window_stability_rows.append(
            {
                "baseline_window_size": baseline_window,
                "alternative_window_size": window_size,
                "kmer_size": baseline_k,
                "n_matched": n,
                "spearman_rho": rho,
                "spearman_p_value": p_value,
                "top_n": top_n,
                "top_n_overlap": overlap,
                "top_n_overlap_fraction": overlap_fraction,
            }
        )

    candidate_consensus = pd.DataFrame()
    if candidate_tables:
        candidate_consensus = candidate_tables[0]
        for table in candidate_tables[1:]:
            candidate_consensus = candidate_consensus.merge(table, on=["accession_version", "source_group"], how="outer")
        rank_cols = [c for c in candidate_consensus.columns if c.startswith("mosaicity_rank_w")]
        score_cols = [c for c in candidate_consensus.columns if c.startswith("compositional_mosaicity_index_w")]
        if rank_cols:
            candidate_consensus["mean_mosaicity_rank"] = candidate_consensus[rank_cols].mean(axis=1)
            candidate_consensus["sd_mosaicity_rank"] = candidate_consensus[rank_cols].std(axis=1, ddof=1)
            candidate_consensus["top_n_scenarios"] = (candidate_consensus[rank_cols] <= top_n).sum(axis=1)
        if score_cols:
            candidate_consensus["mean_mosaicity_score"] = candidate_consensus[score_cols].mean(axis=1)
            candidate_consensus["sd_mosaicity_score"] = candidate_consensus[score_cols].std(axis=1, ddof=1)
        candidate_consensus = candidate_consensus.sort_values(
            ["top_n_scenarios", "mean_mosaicity_rank"],
            ascending=[False, True],
            ignore_index=True,
        )

    kmer_scores_long = pd.concat(mosaicity_tables.values(), ignore_index=True) if mosaicity_tables else pd.DataFrame()
    orthogonal_long = pd.concat(orthogonal_tables.values(), ignore_index=True) if orthogonal_tables else pd.DataFrame()
    concordance_long = pd.concat(concordance_tables.values(), ignore_index=True) if concordance_tables else pd.DataFrame()
    tension_long = pd.concat(tension_tables.values(), ignore_index=True) if tension_tables else pd.DataFrame()
    window_scores_long = pd.concat(window_score_tables.values(), ignore_index=True) if window_score_tables else pd.DataFrame()

    sheets = {
        "parameter_grid": pd.DataFrame(
            {
                "parameter": ["baseline_kmer_size", "kmer_sizes", "pseudocount_scaling", "pseudocount_by_k", "baseline_window_size", "window_sizes", "top_n"],
                "value": [baseline_k, ";".join(map(str, k_values)), "constant total pseudocount mass across k", ";".join(f"k{k}:{pseudocount_by_k[k]:.12g}" for k in k_values), baseline_window, ";".join(map(str, window_values)), top_n],
            }
        ),
        "kmer_panel_validation": pd.concat(panel_validation_tables, ignore_index=True) if panel_validation_tables else pd.DataFrame(),
        "kmer_metric_stability": pd.DataFrame(kmer_stability_rows),
        "kmer_mosaicity_group_summary": pd.concat(kmer_group_summaries, ignore_index=True) if kmer_group_summaries else pd.DataFrame(),
        "kmer_mosaicity_scores": kmer_scores_long,
        "kmer_orthogonal_scores": orthogonal_long,
        "kmer_component_concordance": concordance_long,
        "kmer_component_tension": tension_long,
        "window_group_summary": pd.concat(window_group_summaries, ignore_index=True) if window_group_summaries else pd.DataFrame(),
        "window_rank_stability": pd.DataFrame(window_stability_rows),
        "window_candidate_consensus": candidate_consensus,
        "window_mosaicity_scores": window_scores_long,
    }

    tables_dir = ensure_dir(output_root / "tables")
    csv_dir = ensure_dir(output_root / "csv" / "robustness_validation")
    workbook = write_workbook(sheets, tables_dir / "06_robustness_validation.xlsx")
    write_csv_bundle(sheets, csv_dir)

    manifest = {
        "curation_results_dir": str(Path(curation_results_dir).resolve()),
        "output_dir": str(output_root.resolve()),
        "config_path": str(Path(config_path).resolve()) if config_path else None,
        "kmer_sizes": k_values,
        "window_sizes": window_values,
        "baseline_kmer_size": baseline_k,
        "baseline_window_size": baseline_window,
        "top_n": top_n,
        "pseudocount_scaling": "constant total pseudocount mass across k",
        "pseudocount_by_k": pseudocount_by_k,
        "workbook": str(workbook),
    }
    manifest_path = output_root / "robustness_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"robustness_workbook": workbook, "manifest": manifest_path}
