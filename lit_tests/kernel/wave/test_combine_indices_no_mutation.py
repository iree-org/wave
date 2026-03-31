# RUN: python %s
# Test that combine_indices does not mutate its first argument (thread_independent_index).
# Without deepcopy in combine_indices, the same IndexSequence instances can be shared
# across nodes during propagation, and in-place mutation would change other nodes' indices.

from wave_lang.kernel._support.indexing import IndexSequence, index_symbol
from wave_lang.kernel.wave.analysis.index_sequence_analysis import combine_indices


def test_combine_indices_does_not_mutate_input():
    # Use a symbol as dimension key (same type as in real traces).
    dim = index_symbol("M")
    thread_independent = {dim: IndexSequence(0, 1, 1)}
    thread_dependent = {dim: IndexSequence(10, 5, 2)}

    result = combine_indices(thread_independent, thread_dependent)

    # Result should have combined values.
    assert result[dim].start == 10
    assert result[dim].size == 5
    assert result[dim].stride == 2

    # Input must be unchanged (deepcopy in combine_indices prevents in-place mutation).
    assert thread_independent[dim].start == 0
    assert thread_independent[dim].size == 1
    assert thread_independent[dim].stride == 1


def test_combine_indices_shared_input_unchanged_after_second_call():
    # Simulate two nodes sharing the same index dict (e.g. same reference passed
    # via worklist). Second combine_indices must not mutate the first result.
    dim = index_symbol("K")
    shared_base = {dim: IndexSequence(0, 1, 1)}

    result1 = combine_indices(shared_base, {dim: IndexSequence(2, 3, 1)})
    result2 = combine_indices(shared_base, {dim: IndexSequence(20, 10, 2)})

    # Each result is independent.
    assert (
        result1[dim].start == 2 and result1[dim].size == 3 and result1[dim].stride == 1
    )
    assert (
        result2[dim].start == 20
        and result2[dim].size == 10
        and result2[dim].stride == 2
    )

    # Shared input is still unchanged.
    assert shared_base[dim].start == 0
    assert shared_base[dim].size == 1
    assert shared_base[dim].stride == 1


if __name__ == "__main__":
    test_combine_indices_does_not_mutate_input()
    test_combine_indices_shared_input_unchanged_after_second_call()
    print("All tests passed.")
