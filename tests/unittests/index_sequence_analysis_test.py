# Copyright 2026 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for index_sequence_analysis.py, specifically testing that
`populate_read_write_source_indices` only sets elements_per_thread
for the first contiguous dimension."""

import pytest

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as wave
from wave_lang.kernel.ops.wave_ops import Read, Write, get_custom
from wave_lang.kernel.wave.analysis.index_sequence_analysis import (
    populate_read_write_source_indices,
)
from wave_lang.kernel.wave.constraints import (
    WorkgroupConstraint,
)


def test_copy_kernel_single_contiguous_dim():
    """
    Test that when multiple dimensions are contiguous (because trailing dims
    have unit vector shapes), only the first contiguous dimension gets
    elements_per_thread set in the index, not all of them.

    This is the core regression test for the fix in commit 51302195.
    """
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Vector shapes: M=4 (elements_per_thread), N=1, K=1
    # This makes all dimensions "contiguous" according to is_contiguous_dim,
    # but only the first one (M) should get the elements_per_thread size.
    BLOCK_M = 64
    BLOCK_N = 1
    BLOCK_K = 1

    constraints = [
        wave.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 4, N: 1, K: 1},
        ),
        wave.WorkgroupConstraint(M, BLOCK_M, 0),
        wave.WorkgroupConstraint(N, BLOCK_N, 1),
        wave.WorkgroupConstraint(K, BLOCK_K, 2),
        wave.WaveConstraint(M, BLOCK_M),
        wave.WaveConstraint(N, BLOCK_N),
        wave.WaveConstraint(K, BLOCK_K),
    ]

    @wave.wave_trace_only(constraints)
    def copy_kernel(
        src: tkl.Memory[M, N, K, ADDRESS_SPACE, tkl.f16],
        dst: tkl.Memory[M, N, K, ADDRESS_SPACE, tkl.f16],
    ):
        val = wave.read(src, elements_per_thread=4)
        wave.write(val, dst, elements_per_thread=4)

    trace = copy_kernel()

    # Find the Read node to test with populate_read_write_source_indices directly
    read_node = None
    write_node = None
    for node in trace.get_root_graph().nodes:
        custom = get_custom(node)
        if isinstance(custom, Read):
            read_node = custom
        elif isinstance(custom, Write):
            write_node = custom

    assert read_node is not None, "Read node not found in trace"
    assert write_node is not None, "Write node not found in trace"

    # Get the hardware and workgroup constraints
    hardware_constraint = constraints[0]
    workgroup_constraints = [
        c for c in constraints if isinstance(c, WorkgroupConstraint)
    ]

    # Test populate_read_write_source_indices on the Read node
    result = populate_read_write_source_indices(
        read_node, hardware_constraint, workgroup_constraints
    )

    assert len(result) == 1
    _, index, _ = result[0]

    # The key assertion: only M should have size=4, not N or K
    # Before the fix, all contiguous dims would incorrectly get size=4
    if M in index:
        assert index[M].size == 4, f"M should have size=4, got {index[M].size}"
    if N in index:
        assert index[N].size == 1, f"N should have size=1 (not 4), got {index[N].size}"
    if K in index:
        assert index[K].size == 1, f"K should have size=1 (not 4), got {index[K].size}"

    # Same check for write node
    result = populate_read_write_source_indices(
        write_node, hardware_constraint, workgroup_constraints
    )

    assert len(result) == 1
    _, index, _ = result[0]

    if M in index:
        assert index[M].size == 4, f"M should have size=4, got {index[M].size}"
    if N in index:
        assert index[N].size == 1, f"N should have size=1 (not 4), got {index[N].size}"
    if K in index:
        assert index[K].size == 1, f"K should have size=1 (not 4), got {index[K].size}"


def test_copy_kernel_innermost_contiguous_only():
    """
    Test that when only the innermost dimension is contiguous,
    only it gets elements_per_thread in the index.
    """
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Both dimensions have non-unit vector shapes, only N (innermost) is contiguous
    BLOCK_M = 16
    BLOCK_N = 64

    constraints = [
        wave.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 16, N: 4},
        ),
        wave.WorkgroupConstraint(M, BLOCK_M, 1),
        wave.WorkgroupConstraint(N, BLOCK_N, 0),
        wave.WaveConstraint(M, BLOCK_M),
        wave.WaveConstraint(N, BLOCK_N),
    ]

    @wave.wave_trace_only(constraints)
    def copy_kernel(
        src: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        dst: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        val = wave.read(src, elements_per_thread=4)
        wave.write(val, dst, elements_per_thread=4)

    trace = copy_kernel()

    # Find the Read node
    read_node = None
    for node in trace.get_root_graph().nodes:
        custom = get_custom(node)
        if isinstance(custom, Read):
            read_node = custom
            break

    assert read_node is not None, "Read node not found in trace"

    # Get the hardware and workgroup constraints
    hardware_constraint = constraints[0]
    workgroup_constraints = [
        c for c in constraints if isinstance(c, WorkgroupConstraint)
    ]

    # Test populate_read_write_source_indices
    result = populate_read_write_source_indices(
        read_node, hardware_constraint, workgroup_constraints
    )

    assert len(result) == 1
    _, index, _ = result[0]

    # M is not contiguous (N has non-unit vector shape), should have size=1
    if M in index:
        assert index[M].size == 1, f"M should have size=1, got {index[M].size}"
    # N is contiguous (innermost) and should have elements_per_thread=4
    if N in index:
        assert index[N].size == 4, f"N should have size=4, got {index[N].size}"
