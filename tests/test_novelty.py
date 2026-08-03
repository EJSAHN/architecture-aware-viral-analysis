import pandas as pd

from viral_genome_analysis.novelty import compositional_mosaicity


def test_mosaicity_outputs_expected_columns():
    df = pd.DataFrame(
        {
            "accession_version": ["A1", "A2", "B1", "B2"],
            "source_group": ["A", "A", "B", "B"],
            "sequence": [
                "ACGT" * 200,
                "ACGA" * 200,
                "TTTT" * 200,
                "TTTA" * 200,
            ],
            "sequence_length": [800, 800, 800, 800],
            "organism": ["x", "x", "y", "y"],
            "isolate": ["i1", "i2", "i3", "i4"],
        }
    )
    config = {
        "analysis": {
            "kmer_size": 4,
            "reverse_complement_collapse": True,
            "pseudocount": 0.5,
            "min_group_size_monopartite": 2,
            "sliding_window_size": 200,
            "sliding_window_step": 100,
            "min_windows_per_sequence": 2,
            "top_n_mosaic_candidates": 10,
        }
    }
    scores, windows = compositional_mosaicity(df, "source_group", config)
    assert "compositional_mosaicity_index" in scores.columns
    assert "window_nearest_group" in windows.columns


def test_component_concordance_excludes_singletons_and_normalizes_by_component_count():
    import numpy as np
    from viral_genome_analysis.novelty import component_neighbor_concordance

    registry = pd.DataFrame(
        {
            "accession_version": ["A1", "A2", "B1", "B2", "C1"],
            "source_group": ["G"] * 5,
            "panel_componentized": [True] * 5,
            "normalized_component": ["X", "X", "Y", "Y", "X"],
            "isolate": ["I1", "I2", "I1", "I2", "I3"],
            "metadata_score": [10] * 5,
            "sequence_length": [100] * 5,
            "sequence": ["A" * 100] * 5,
            "feature_row": np.arange(5),
        }
    )
    full_clr = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
        ]
    )
    completeness = pd.DataFrame(
        {
            "source_group": ["G", "G", "G"],
            "isolate_key": ["I1", "I2", "I3"],
            "expected_component_count": [2, 2, 2],
            "all_expected_components_present": [True, True, False],
        }
    )
    config = {"analysis": {"kmer_size": 4, "pseudocount": 0.5, "reverse_complement_collapse": True}}
    scores, _, _ = component_neighbor_concordance(
        registry,
        config,
        full_clr_matrix=full_clr,
        component_completeness_df=completeness,
    )
    assert set(scores["isolate"]) == {"I1", "I2"}
    assert scores["components_present"].min() == 2
    assert scores["normalized_neighbor_discordance_score"].between(0, 1).all()
    assert scores["full_expected_component_set"].all()
