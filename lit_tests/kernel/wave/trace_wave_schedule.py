#!/usr/bin/env python3
# RUN: python %s | FileCheck %s

from wave_lang.kernel.wave.utils.general_utils import run_test
from wave_lang.kernel.wave import wave_schedule
import wave_lang.kernel.lang as wave_lang
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel._support.indexing import IndexingContext
from wave_lang.kernel.wave.utils.print_utils import print_trace
import wave_lang.kernel.wave as wave

M = wave_lang.sym.M
N = wave_lang.sym.N
K = wave_lang.sym.K
BLOCK_M = wave_lang.sym.BLOCK_M
BLOCK_N = wave_lang.sym.BLOCK_N
BLOCK_K = wave_lang.sym.BLOCK_K


@run_test
def test_trace_schedule_get_node_by_tag():
    """Test tracing schedule operations with get_node_by_tag."""

    # Create a simple kernel with tagged nodes
    @wave.wave_trace_only()
    def kernel_with_tags():
        c_reg = wave_lang.Register[M, N, wave_lang.f32](0.0)

        @wave.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(acc):
            return acc

    kernel_trace = kernel_with_tags()

    @wave_schedule.wave_schedule()
    def schedule_get_node():
        k_loop = wave.get_node_by_tag("k_loop")

    constraints = [
        wave.WorkgroupConstraint(M, BLOCK_M, 0),
        wave.WorkgroupConstraint(N, BLOCK_N, 1),
        wave.TilingConstraint(K, BLOCK_K),
    ]

    with IndexingContext() as idxc:
        idxc.subs = {
            M: 128,
            N: 128,
            K: 128,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 32,
        }
        schedule_trace = schedule_get_node._trace(kernel_trace, constraints)
        print_trace(schedule_trace)

    # CHECK-LABEL: test_trace_schedule_get_node_by_tag
    # CHECK: %get_node_by_tag : [num_users=0] = [target=wave_lang.kernel.ops.wave_schedule_ops.get_node_by_tag]
    # CHECK: return None
    # CHECK: unknown: get_node_by_tag(args=(), kwargs={}) type(None)
    # CHECK: output(return_vals=(None,)) type(None)


@run_test
def test_trace_schedule_complex_graph():
    """Test tracing schedule operations that create a more complex FX graph."""

    @wave.wave_trace_only()
    def kernel_complex():
        c_reg = wave_lang.Register[M, N, wave_lang.f32](0.0)

        @wave.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(acc):
            inner_reg = wave_lang.Register[M, N, wave_lang.f32](0.0)

            @wave.iterate(N, init_args=[inner_reg], tag="inner_loop")
            def inner_loop(inner_acc):
                return inner_acc

            return acc

    kernel_trace = kernel_complex()

    @wave_schedule.wave_schedule()
    def complex_schedule():
        # Test multiple schedule operations
        k_loop = wave.get_node_by_tag("k_loop")
        inner_loop = wave.get_node_by_tag("inner_loop")

    constraints = [
        wave.WorkgroupConstraint(M, BLOCK_M, 0),
        wave.WorkgroupConstraint(N, BLOCK_N, 1),
        wave.TilingConstraint(K, BLOCK_K),
    ]

    with IndexingContext() as idxc:
        idxc.subs = {
            M: 128,
            N: 128,
            K: 128,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 32,
        }
        schedule_trace = complex_schedule._trace(kernel_trace, constraints)
        print_trace(schedule_trace)

    # CHECK-LABEL: test_trace_schedule_complex_graph
    # CHECK: %get_node_by_tag : [num_users=0] = [target=wave_lang.kernel.ops.wave_schedule_ops.get_node_by_tag]
    # CHECK: %get_node_by_tag_1 : [num_users=0] = [target=wave_lang.kernel.ops.wave_schedule_ops.get_node_by_tag]
    # CHECK: return None
    # CHECK: unknown: get_node_by_tag(args=(), kwargs={}) type(None)
    # CHECK: unknown: get_node_by_tag_1(args=(), kwargs={}) type(None)
    # CHECK: output(return_vals=(None,)) type(None)
