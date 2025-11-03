"""
Atomic Operations Examples

Demonstrates atomic memory operations including atomic add with return values
and reading back scalar values after atomic operations.
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


def test_atomic_add_return_value(is_debug=False):
    """Atomic add operation that returns the old value before the addition."""
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

    simple_read_mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={M: i},
        outputs={M: i},
    )

    @tkw.wave(constraints)
    def wave_kernel(
        a: tkl.Memory[M, ADDRESS_SPACE, tkl.i32],
        c: tkl.Memory[M, ADDRESS_SPACE_0, tkl.i32],
    ):

        one_reg = tkw.Register[M, tkl.i32](1)
        res = tkw.atomic_add(one_reg, a, mapping=simple_read_mapping)
        tkw.write(res, c)

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
        print_ir_after="all" if is_debug else [],
    )
    wave_kernel = wave_compile(options, wave_kernel)
    if is_debug:
        print(wave_kernel.asm)

    # generate random input tensors between -1 and 1
    a = torch.randint(1, 2, (64,), dtype=torch.int32).cuda()
    c = torch.zeros((64,), dtype=torch.int32).cuda()

    wave_kernel(a, c)
    print(a)
    print(c)


def test_read_back_scalar(is_debug=False):
    """Perform atomic add to shared memory then read back a scalar value using dynamic mapping."""
    ONE = tkl.sym.ONE
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={B: 0, M: 64, ONE: 1},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(ONE, ONE, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(ONE, ONE)]

    i = tkw.IndexMapping.iterator(0)
    d0 = tkw.IndexMapping.dynamic_val(0)

    simple_read_mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={M: i},
        outputs={M: i},
    )

    dynamic_read_mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={M: d0},
        outputs={M: i},
        dynamic_val_mappings={M: i},
    )

    @tkw.wave(constraints)
    def wave_kernel(
        a: tkl.Memory[M, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, ADDRESS_SPACE, tkl.i32],
        c: tkl.Memory[ONE, ADDRESS_SPACE, tkl.i32],
    ):

        tid = tkw.scalar(THREAD_0, tkl.i32)
        one_reg = tkw.Register[M, tkl.i32](1)
        res = tkw.atomic_add(
            one_reg, a, mapping=dynamic_read_mapping, mapping_dynamic_vals=(tid,)
        )
        val = tkw.read(
            res,
            mapping=dynamic_read_mapping,
            mapping_dynamic_vals=(tid,),
            elements_per_thread=1,
        )
        tkw.write(val, c)

    options = WaveCompileOptions(
        subs={
            M: 64,
            ONE: 1,
            BLOCK_M: 64,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        print_ir_after="all" if is_debug else [],
        minimize_shared_allocs=False,
    )
    wave_kernel = wave_compile(options, wave_kernel)
    if is_debug:
        print(wave_kernel.asm)

    # generate random input tensors between -1 and 1
    a = torch.randint(1, 2, (64,), dtype=torch.int32).cuda()
    b = torch.zeros((64,), dtype=torch.int32).cuda()
    c = torch.zeros((1,), dtype=torch.int32).cuda()

    wave_kernel(a, b, c)
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
