import numpy as np
import pandas as pd

from viral_genome_analysis.kmers import build_kmer_vocabulary, compute_kmer_features


def test_reverse_complement_collapsed_vocabulary_size():
    vocab, mapping = build_kmer_vocabulary(k=4, collapse_reverse_complements=True)
    assert len(vocab) == 136
    assert mapping["AAAA"] == mapping["TTTT"]


def test_feature_matrix_shape_and_normalization():
    sequences = ["ACGTACGTACGT", "AAAACCCCGGGGTTTT"]
    freq_df, clr, vocab = compute_kmer_features(sequences, k=3, collapse_reverse_complements=True, pseudocount=0.5)
    assert freq_df.shape[0] == 2
    assert np.allclose(freq_df.sum(axis=1).to_numpy(), 1.0)
    assert clr.shape == freq_df.shape
