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
