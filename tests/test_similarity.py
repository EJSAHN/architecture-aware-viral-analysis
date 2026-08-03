from viral_genome_analysis.similarity import _candidate_windows, _consensus, _identity


def test_consensus_ignores_gaps_and_ambiguous_bases():
    assert _consensus(["ACGT", "AC-T", "ACNT"]) == "ACGT"


def test_candidate_windows_uses_ungapped_coordinates():
    windows = _candidate_windows("AA--CCGG", window_size=4, step_size=2)
    assert windows[0][0:2] == (1, 4)
    assert len(windows[0][2]) == 4


def test_identity_counts_only_comparable_bases():
    assert _identity("ACGT", "AC-T", [0, 1, 2, 3]) == 1.0


def test_alignment_map_normalizes_mafft_reverse_prefix(tmp_path):
    from viral_genome_analysis.similarity import _alignment_map

    aligned = tmp_path / "aligned.fasta"
    aligned.write_text(">_R_ACC1\nACGT--\n>ACC2\nACGT--\n", encoding="utf-8")
    result = _alignment_map(aligned)
    assert set(result) == {"ACC1", "ACC2"}
    assert result["ACC1"] == "ACGT--"
