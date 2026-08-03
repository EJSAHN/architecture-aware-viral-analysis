import numpy as np
import pandas as pd

from viral_genome_analysis.extensions import component_configuration_tension, orthogonal_heterogeneity_axis


def test_orthogonal_heterogeneity_axis_returns_expected_columns():
    monopartite_df = pd.DataFrame(
        {
            "accession_version": ["A1", "A2", "B1", "B2"],
            "source_group": ["A", "A", "B", "B"],
            "organism": ["x", "x", "y", "y"],
            "isolate": ["i1", "i2", "i3", "i4"],
            "sequence_length": [1000, 1005, 980, 990],
            "gc_fraction": [0.45, 0.46, 0.50, 0.49],
            "n_fraction": [0.0, 0.0, 0.0, 0.0],
            "base_entropy": [1.95, 1.90, 1.80, 1.82],
        }
    )
    embedding = monopartite_df[["accession_version", "source_group"]].copy()
    embedding["pc1"] = [-3.0, -2.8, 2.9, 3.2]
    embedding["pc2"] = [0.1, -0.1, 0.0, 0.1]
    mosaicity = pd.DataFrame(
        {
            "accession_version": ["A1", "A2", "B1", "B2"],
            "source_group": ["A", "A", "B", "B"],
            "n_windows": [5, 5, 5, 5],
            "global_nearest_group": ["A", "A", "B", "B"],
            "dominant_window_group": ["A", "A", "B", "B"],
            "dominant_window_fraction": [0.8, 0.9, 0.7, 0.85],
            "window_label_entropy_norm": [0.5, 0.2, 0.6, 0.3],
            "switch_rate": [0.4, 0.1, 0.5, 0.2],
            "discordance_fraction": [0.3, 0.0, 0.4, 0.1],
            "ambiguity_score": [0.4, 0.2, 0.5, 0.3],
            "compositional_mosaicity_index": [0.4, 0.1, 0.5, 0.2],
            "mosaicity_rank": [2, 4, 1, 3],
        }
    )
    config = {
        "analysis": {"orthogonal_axis_geometry_components": 2, "random_seed": 7},
        "statistics": {"bootstrap_iterations": 20, "pairwise_permutations": 9},
    }
    scores, loadings, validation, summary = orthogonal_heterogeneity_axis(monopartite_df, embedding, mosaicity, config)
    assert "orthogonal_axis_1" in scores.columns
    assert "orthogonal_axis_1_loading" in loadings.columns
    assert not validation.empty
    assert not summary.empty


def test_component_configuration_tension_returns_scores():
    registry_df = pd.DataFrame(
        {
            "accession_version": ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"],
            "source_group": ["G", "G", "G", "G", "G", "G", "G", "G"],
            "panel_componentized": [True] * 8,
            "normalized_component": ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
            "isolate": ["I1", "I2", "I1", "I2", "I3", "I4", "I3", "I4"],
            "metadata_score": [10] * 8,
            "sequence_length": [100] * 8,
            "sequence": ["A" * 100] * 8,
            "feature_row": np.arange(8),
        }
    )
    full_clr = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [1.0, 1.1, 0.0],
            [1.2, 1.1, 0.0],
        ]
    )
    config = {
        "analysis": {
            "random_seed": 7,
            "min_isolate_components_for_tension": 2,
            "tension_embedding_dimensions": 2,
            "kmer_size": 4,
            "pseudocount": 0.5,
            "reverse_complement_collapse": True,
        }
    }
    scores, vectors, summary = component_configuration_tension(registry_df, config, full_clr_matrix=full_clr)
    assert not scores.empty
    assert "component_configuration_tension_index" in scores.columns
    assert not vectors.empty
    assert not summary.empty
