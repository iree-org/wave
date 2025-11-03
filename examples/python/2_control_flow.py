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
I = tkl.sym.I
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
BLOCK_B = tkl.sym.BLOCK_B
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


def test_iteration_with_condition(is_debug=False):
    """Unstructured loop with runtime-determined exit condition, showing per-thread iteration control."""
    LIMIT_VAL = tkl.sym.LIMIT_VAL

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={I: 0, M: 64},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    # I is iterated over and so we define a tiling constraint on it.
    # However, there is no notion of tile size for the iteration as
    # it is an unstructured loop.
    constraints += [tkw.TilingConstraint(I)]

    @tkw.wave(constraints)
    def wave_kernel(
        init_value: tkl.i32,  # type: ignore
        c: tkl.Memory[M, ADDRESS_SPACE_0, tkl.i32],
        limit: tkl.i32,  # type: ignore
    ):

        tkw.set_symbol(LIMIT_VAL, limit)
        condition = I < LIMIT_VAL

        @tkw.iterate(I, start=init_value, condition=condition, init_args=[])
        def body():
            c_reg = tkw.read(c)

            c_reg = c_reg + tkw.Register[M, tkl.i32](1)
            tkw.write(c_reg, c)

            # Set the next value for the iteration.
            # In this case, we are using a simple increment operation,
            # but this can be replaced with any other operation.
            index_b = tkw.self_index(I, tkl.i32)
            next_value = tkw.apply_expr(index_b, lambda x: x + 1)
            tkw.set_symbol(I, next_value)

    options = WaveCompileOptions(
        subs={
            M: 64,
            BLOCK_M: 64,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        print_ir_after="all" if is_debug else [],
    )

    wave_kernel = wave_compile(options, wave_kernel)
    if is_debug:
        print(wave_kernel.asm)

    # Setup test parameters
    init_value = 1  # Starting value for iteration
    c = torch.zeros((64,), dtype=torch.int32).cuda()
    limit = 4  # Maximum number of iterations

    wave_kernel(init_value, c, limit)
    print("Init value:", init_value)
    print("Limit:", limit)
    print("Result c (should be incremented limit-init_value times):", c)
    print("Expected: each element should be", limit - init_value)


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
