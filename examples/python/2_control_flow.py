"""
Control Flow Examples

Demonstrates unstructured loops with dynamic conditions and iteration patterns.
Shows how to implement custom loop control with runtime-determined exit conditions.
"""

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
import torch

from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from utils import parse_args, list_tests, run_test


M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
B = tkl.sym.B
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
BLOCK_B = tkl.sym.BLOCK_B
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


def test_iteration_with_condition():
    """Unstructured loop with runtime-determined exit condition, showing per-thread iteration control."""
    LIMIT_VAL = tkl.sym.LIMIT_VAL

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={B: 0, M: 64, LIMIT_VAL: 0},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    # B is iterated over and so we define a tiling constraint on it.
    # However, there is no notion of tile size for the iteration as
    # it is an unstructured loop.
    constraints += [tkw.TilingConstraint(B)]

    i = tkw.IndexMapping.iterator(0)
    d0 = tkw.IndexMapping.dynamic_val(0)

    limit_val_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={M: d0},
        outputs={M: i},
        dynamic_val_mappings={M: i},
    )

    @tkw.wave(constraints)
    def iterated_gemm(
        a: tkl.Memory[M, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, ADDRESS_SPACE, tkl.i32],
        c: tkl.Memory[M, ADDRESS_SPACE_0, tkl.i32],
        init_value: tkl.i32,  # type: ignore
    ):

        tid = tkw.scalar(tkw.THREAD_0, tkl.i32)
        limit_val = tkw.read(
            a, mapping=limit_val_map, mapping_dynamic_vals=(tid,), elements_per_thread=1
        )
        tkw.set_symbol(LIMIT_VAL, limit_val)
        condition = B < LIMIT_VAL

        init_val = tkw.read(
            b, mapping=limit_val_map, mapping_dynamic_vals=(tid,), elements_per_thread=1
        )
        ones_b = tkw.Register[B, tkl.i32](1)

        # init_val = tkw.scalar(0, tkl.i32)
        @tkw.iterate(B, start=init_val, condition=condition, init_args=[])
        def body():
            c_reg = tkw.read(c)
            b_reg = tkw.read(b)

            # c_reg = c_reg + b_reg
            c_reg = tkw.Register[M, tkl.i32](tkw.THREAD_0)
            tkw.write(c_reg, c)

            # Set the next value for the iteration.
            # In this case, we are using a simple increment operation,
            # but this can be replaced with any other operation.
            index_b = tkw.self_index(B, tkl.i32)
            next_value = tkw.apply_expr(index_b, lambda x: x + 1)
            # next_value = index_b + ones_b
            tkw.set_symbol(B, next_value)

    options = WaveCompileOptions(
        subs={
            M: 64,
            B: 10,
            BLOCK_M: 64,
            BLOCK_B: 1,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        print_ir_after="all",
        print_ir_before="all",
    )
    iterated_gemm = wave_compile(options, iterated_gemm)
    print(iterated_gemm.asm)

    # generate random input tensors between -1 and 1
    a = torch.randint(0, 4, (64,), dtype=torch.int32).cuda()
    b = torch.randint(1, 2, (64,), dtype=torch.int32).cuda()
    c = torch.zeros((64,), dtype=torch.int32).cuda()

    iterated_gemm(a, b, c, 0)
    print(a)
    print(b)
    print(c)


if __name__ == "__main__":
    args = parse_args()
    if args.list_tests:
        list_tests(globals())
        exit(0)

    if not args.test:
        print("Error: --test argument is required")
        print("Use --list_tests to see available tests")
        exit(1)

    success = run_test(args.test, globals(), args.debug, args.repeat)
    exit(0 if success else 1)
