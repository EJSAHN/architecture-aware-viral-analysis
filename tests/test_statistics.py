import numpy as np

from viral_genome_analysis.statistics import permutation_manova, repeated_cv_classification


def test_permutation_manova_runs_on_toy_data():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [4.0, 4.0],
        [4.1, 4.0],
    ])
    labels = np.array(["A", "A", "B", "B"])
    result = permutation_manova(X, labels, panel_name="toy", label_name="group", n_permutations=19, random_seed=7)
    assert result.shape[0] == 1
    assert result.loc[0, "pseudo_f"] > 0


def test_repeated_cv_classification_runs_on_toy_data():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.2, 0.0],
        [4.0, 4.0],
        [4.1, 4.0],
        [4.2, 4.0],
    ])
    labels = np.array(["A", "A", "A", "B", "B", "B"])
    summary, folds, null_df = repeated_cv_classification(
        X,
        labels,
        panel_name="toy",
        label_name="group",
        n_splits=3,
        n_repeats=1,
        n_permutations=5,
        random_seed=7,
    )
    assert summary.shape[0] == 1
    assert not folds.empty
    assert not null_df.empty
