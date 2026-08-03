import pandas as pd

from viral_genome_analysis.robustness import _safe_spearman, _top_overlap


def test_safe_spearman_perfect_rank_agreement():
    rho, p_value, n = _safe_spearman(pd.Series([1, 2, 3, 4]), pd.Series([10, 20, 30, 40]))
    assert n == 4
    assert rho == 1.0
    assert p_value <= 0.05


def test_top_overlap():
    a = pd.DataFrame({"id": ["a", "b", "c"], "rank": [1, 2, 3]})
    b = pd.DataFrame({"id": ["b", "a", "d"], "rank": [1, 2, 3]})
    overlap, fraction = _top_overlap(a, b, "id", "rank", 2)
    assert overlap == 2
    assert fraction == 1.0
