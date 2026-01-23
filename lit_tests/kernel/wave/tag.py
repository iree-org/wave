# RUN: python %s | FileCheck %s

"""
Tests for tkw.tag() function.

This module tests that tkw.tag() correctly assigns tags to Python arithmetic
operators in the FX graph. Tags enable custom wave schedules to identify and
manipulate specific operations.
"""

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.wave.utils.general_utils import run_test
from wave_lang.kernel.wave.utils.print_utils import print_trace

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE


def print_tags(trace):
    """Print tags associated with FX nodes in the trace."""
    print("Tags in FX graph:")
    graph = trace.get_root_graph()
    for node in graph.nodes:
        tag = getattr(node, "tag", None)
        if tag is not None:
            print(f"  {node.name}: tag={tag}")


@run_test
def test_tag_multiply():
    """Test tagging a multiplication operation."""

    @tkw.wave_trace_only()
    def test(A: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        a = tkw.read(A)
        b = tkw.read(A)
        # Tag the multiplication
        result = tkw.tag(a * b, "multiply_ab")
        tkw.write(result, A, elements_per_thread=4)

    trace = test()
    print_trace(trace)
    print_tags(trace)

    # CHECK: %a
    # CHECK-NEXT: %read
    # CHECK-NEXT: %read_1
    # CHECK-NEXT: %mul
    # CHECK-NEXT: %write
    # CHECK-NEXT: return None

    # CHECK: Custom format:
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: mul(lhs=read, rhs=read_1)
    # CHECK-NEXT: write(register_=mul
    # CHECK-NEXT: output

    # Verify tag is present
    # CHECK: Tags in FX graph:
    # CHECK-NEXT: mul: tag=multiply_ab


@run_test
def test_tag_add():
    """Test tagging an addition operation."""

    @tkw.wave_trace_only()
    def test(A: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        a = tkw.read(A)
        b = tkw.read(A)
        # Tag the addition
        result = tkw.tag(a + b, "add_ab")
        tkw.write(result, A, elements_per_thread=4)

    trace = test()
    print_trace(trace)
    print_tags(trace)

    # CHECK: %a
    # CHECK-NEXT: %read
    # CHECK-NEXT: %read_1
    # CHECK-NEXT: %add
    # CHECK-NEXT: %write
    # CHECK-NEXT: return None

    # CHECK: Custom format:
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: add(lhs=read, rhs=read_1)
    # CHECK-NEXT: write(register_=add
    # CHECK-NEXT: output

    # Verify tag is present
    # CHECK: Tags in FX graph:
    # CHECK-NEXT: add: tag=add_ab


@run_test
def test_tag_subtract():
    """Test tagging a subtraction operation."""

    @tkw.wave_trace_only()
    def test(A: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        a = tkw.read(A)
        b = tkw.read(A)
        # Tag the subtraction
        result = tkw.tag(a - b, "sub_ab")
        tkw.write(result, A, elements_per_thread=4)

    trace = test()
    print_trace(trace)
    print_tags(trace)

    # CHECK: %a
    # CHECK-NEXT: %read
    # CHECK-NEXT: %read_1
    # CHECK-NEXT: %sub
    # CHECK-NEXT: %write
    # CHECK-NEXT: return None

    # CHECK: Custom format:
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: sub(lhs=read, rhs=read_1)
    # CHECK-NEXT: write(register_=sub
    # CHECK-NEXT: output

    # Verify tag is present
    # CHECK: Tags in FX graph:
    # CHECK-NEXT: sub: tag=sub_ab


@run_test
def test_tag_divide():
    """Test tagging a division operation."""

    @tkw.wave_trace_only()
    def test(A: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        a = tkw.read(A)
        b = tkw.read(A)
        # Tag the division
        result = tkw.tag(a / b, "div_ab")
        tkw.write(result, A, elements_per_thread=4)

    trace = test()
    print_trace(trace)
    print_tags(trace)

    # CHECK: %a
    # CHECK-NEXT: %read
    # CHECK-NEXT: %read_1
    # CHECK-NEXT: %truediv
    # CHECK-NEXT: %write
    # CHECK-NEXT: return None

    # CHECK: Custom format:
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: truediv(lhs=read, rhs=read_1)
    # CHECK-NEXT: write(register_=truediv
    # CHECK-NEXT: output

    # Verify tag is present
    # CHECK: Tags in FX graph:
    # CHECK-NEXT: truediv: tag=div_ab


@run_test
def test_tag_multiple_operations():
    """Test tagging multiple different operations."""

    @tkw.wave_trace_only()
    def test(A: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        a = tkw.read(A)
        b = tkw.read(A)
        # Tag multiple operations
        sum_result = tkw.tag(a + b, "sum_op")
        diff_result = tkw.tag(a - b, "diff_op")
        prod_result = tkw.tag(sum_result * diff_result, "prod_op")
        tkw.write(prod_result, A, elements_per_thread=4)

    trace = test()
    print_trace(trace)
    print_tags(trace)

    # CHECK: %a
    # CHECK-NEXT: %read
    # CHECK-NEXT: %read_1
    # CHECK-NEXT: %add
    # CHECK-NEXT: %sub
    # CHECK-NEXT: %mul
    # CHECK-NEXT: %write
    # CHECK-NEXT: return None

    # CHECK: Custom format:
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: add(lhs=read, rhs=read_1)
    # CHECK-NEXT: sub(lhs=read, rhs=read_1)
    # CHECK-NEXT: mul(lhs=add, rhs=sub)
    # CHECK-NEXT: write(register_=mul
    # CHECK-NEXT: output

    # Verify all tags are present
    # CHECK: Tags in FX graph:
    # CHECK-DAG: add: tag=sum_op
    # CHECK-DAG: sub: tag=diff_op
    # CHECK-DAG: mul: tag=prod_op


@run_test
def test_tag_with_scalar():
    """Test tagging operations with scalar values."""

    @tkw.wave_trace_only()
    def test(A: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        a = tkw.read(A)
        scale = tkl.Register[M, N, tkl.f16](0.5)
        # Tag multiplication with scalar
        result = tkw.tag(a * scale, "scaled")
        tkw.write(result, A, elements_per_thread=4)

    trace = test()
    print_trace(trace)
    print_tags(trace)

    # CHECK: %a
    # CHECK-NEXT: %read
    # CHECK-NEXT: %register
    # CHECK-NEXT: %mul
    # CHECK-NEXT: %write
    # CHECK-NEXT: return None

    # CHECK: Custom format:
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: register
    # CHECK-NEXT: mul(lhs=read, rhs=register)
    # CHECK-NEXT: write(register_=mul
    # CHECK-NEXT: output

    # Verify tag is present
    # CHECK: Tags in FX graph:
    # CHECK-NEXT: mul: tag=scaled


@run_test
def test_tag_chained_operations():
    """Test tagging in a chain of operations, only tagging final result."""

    @tkw.wave_trace_only()
    def test(A: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        a = tkw.read(A)
        b = tkw.read(A)
        # Chain operations but only tag the final result
        intermediate = a + b
        result = tkw.tag(intermediate * a, "final_result")
        tkw.write(result, A, elements_per_thread=4)

    trace = test()
    print_trace(trace)
    print_tags(trace)

    # CHECK: %a
    # CHECK-NEXT: %read
    # CHECK-NEXT: %read_1
    # CHECK-NEXT: %add
    # CHECK-NEXT: %mul
    # CHECK-NEXT: %write
    # CHECK-NEXT: return None

    # CHECK: Custom format:
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: add(lhs=read, rhs=read_1)
    # CHECK-NEXT: mul(lhs=add, rhs=read)
    # CHECK-NEXT: write(register_=mul
    # CHECK-NEXT: output

    # Verify only final operation has tag
    # CHECK: Tags in FX graph:
    # CHECK-NEXT: mul: tag=final_result


@run_test
def test_tag_returns_same_proxy():
    """Test that tkw.tag returns the same proxy for chaining."""

    @tkw.wave_trace_only()
    def test(A: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        a = tkw.read(A)
        b = tkw.read(A)
        # Verify the returned value can be used in further operations
        tagged = tkw.tag(a * b, "product")
        # Use the tagged result in another operation
        result = tagged + a
        tkw.write(result, A, elements_per_thread=4)

    trace = test()
    print_trace(trace)
    print_tags(trace)

    # CHECK: %a
    # CHECK-NEXT: %read
    # CHECK-NEXT: %read_1
    # CHECK-NEXT: %mul
    # CHECK-NEXT: %add
    # CHECK-NEXT: %write
    # CHECK-NEXT: return None

    # CHECK: Custom format:
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: mul(lhs=read, rhs=read_1)
    # CHECK-NEXT: add(lhs=mul, rhs=read)
    # CHECK-NEXT: write(register_=add
    # CHECK-NEXT: output

    # Verify tag is on mul, and add uses mul as input
    # CHECK: Tags in FX graph:
    # CHECK-NEXT: mul: tag=product


@run_test
def test_tag_with_tkw_operations():
    """Test that tkw.tag works alongside native tkw operations with tags."""

    @tkw.wave_trace_only()
    def test(A: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        # Native tkw operation with tag
        a = tkw.read(A, tag="read_a")
        b = tkw.read(A, tag="read_b")
        # Python operation with tkw.tag
        result = tkw.tag(a * b, "multiply")
        tkw.write(result, A, elements_per_thread=4, tag="write_out")

    trace = test()
    print_trace(trace)
    print_tags(trace)

    # CHECK: %a
    # CHECK-NEXT: %read
    # CHECK-NEXT: %read_1
    # CHECK-NEXT: %mul
    # CHECK-NEXT: %write
    # CHECK-NEXT: return None

    # CHECK: Custom format:
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: mul(lhs=read, rhs=read_1)
    # CHECK-NEXT: write(register_=mul
    # CHECK-NEXT: output

    # Verify all tags are present
    # CHECK: Tags in FX graph:
    # CHECK-DAG: read: tag=read_a
    # CHECK-DAG: read_1: tag=read_b
    # CHECK-DAG: mul: tag=multiply
    # CHECK-DAG: write: tag=write_out
