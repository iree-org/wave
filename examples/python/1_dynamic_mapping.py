"""
Dynamic Mapping Examples

Demonstrates reading and writing with dynamic index mappings and offset-based access patterns.
Includes broadcast operations where indices are computed at runtime.
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


def get_wave_compile_options(
    m: int = 16,
    n: int = 16,
    k: int = 16,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 16,
    address_space: tkl.AddressSpace = tkl.AddressSpace.SHARED_MEMORY.value,
    canonicalize: bool = False,
    dynamic_symbols=[],
    additional_symbols={},
    is_debug: bool = False,
):
    bindings = {
        M: m,
        N: n,
        K: k,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        ADDRESS_SPACE: address_space,
    }
    bindings.update(additional_symbols)

    # Remove dynamic symbols from the bindings.
    for sym in dynamic_symbols:
        if sym in bindings:
            del bindings[sym]

    return WaveCompileOptions(
        subs=bindings,
        canonicalize=canonicalize,
        dynamic_symbols=dynamic_symbols,
        print_ir_after="all" if is_debug else [],
    )


def test_dynamic_offset_0(is_debug=False):
    """1D tensor read with a single dynamic offset value, showcasing simpler offset-based indexing."""
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={N: 16},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    j = tkw.IndexMapping.iterator(0)
    k = tkw.IndexMapping.dynamic_val(0)
    mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={N: (k + j) % 16},
        outputs={N: j},
        dynamic_val_mappings={N: j},
    )

    @tkw.wave(constraints)
    def kernel(
        a: tkl.Memory[N, ADDRESS_SPACE, tkl.i32],
        off: tkl.i32,
        b: tkl.Memory[N, ADDRESS_SPACE, tkl.i32],
    ):

        res = tkw.read(
            a,
            mapping=mapping,
            mapping_dynamic_vals=(off,),
        )
        tkw.write(res, b)

    compiled_kernel = wave_compile(
        get_wave_compile_options(canonicalize=True, is_debug=is_debug),
        kernel,
    )
    if is_debug:
        print(compiled_kernel.asm)

    # create input tensors
    a = (
        torch.arange(0, 16, dtype=torch.int32)
        .reshape(
            16,
        )
        .cuda()
    )
    off = 1
    b = torch.zeros((16,), dtype=torch.int32).cuda()

    compiled_kernel(a, off, b)
    print(a)
    print(off)
    print(b)


def test_dynamic_offset_1(is_debug=False):
    """1D tensor read with a single dynamic offset value, showcasing simpler offset-based indexing."""
    ONE = tkl.sym.ONE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={N: 16, ONE: 1},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    j = tkw.IndexMapping.iterator(0)
    k = tkw.IndexMapping.dynamic_val(0)
    mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={N: k + j % 16},
        outputs={N: j},
        dynamic_val_mappings={ONE: j // 16},
    )

    @tkw.wave(constraints)
    def kernel(
        a: tkl.Memory[N, ADDRESS_SPACE, tkl.i32],
        off: tkl.Memory[ONE, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[N, ADDRESS_SPACE, tkl.i32],
    ):
        offset = tkw.read(off)
        res = tkw.read(
            a,
            mapping=mapping,
            mapping_dynamic_vals=(offset,),
        )
        tkw.write(res, b)

    compiled_kernel = wave_compile(
        get_wave_compile_options(
            canonicalize=True, additional_symbols={ONE: 1}, is_debug=is_debug
        ),
        kernel,
    )
    if is_debug:
        print(compiled_kernel.asm)

    # create input tensors
    a = (
        torch.arange(0, 16, dtype=torch.int32)
        .reshape(
            16,
        )
        .cuda()
    )
    off = torch.ones((1,), dtype=torch.int32).cuda()
    b = torch.zeros((16,), dtype=torch.int32).cuda()

    compiled_kernel(a, off, b)
    print(a)
    print(off)
    print(b)


def test_fixed_offset(is_debug=False):
    """Simple read with constant offset mapping, demonstrating basic index transformation without dynamic values."""
    ONE = tkl.sym.ONE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={N: 16, ONE: 1},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    j = tkw.IndexMapping.iterator(0)
    k = tkw.IndexMapping.dynamic_val(0)

    seq_len_mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={N: j},
        outputs={N: j},
    )

    seq_len_mapping_w = tkw.IndexMapping(
        num_iterators=1,
        inputs={N: j},
        outputs={N: j + ONE},
    )

    @tkw.wave(constraints)
    def kernel(
        a: tkl.Memory[N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[N, ADDRESS_SPACE, tkl.i32],
    ):
        temp = tkw.Register[N, tkl.i32](0)
        temp = tkw.read(
            a,
            mapping=seq_len_mapping,
        )
        tkw.write(temp, b, mapping=seq_len_mapping_w)

    compiled_kernel = wave_compile(
        get_wave_compile_options(
            canonicalize=True, additional_symbols={ONE: 1}, is_debug=is_debug
        ),
        kernel,
    )
    if is_debug:
        print(compiled_kernel.asm)

    # create input tensors
    a = (
        torch.arange(0, 16, dtype=torch.int32)
        .reshape(
            16,
        )
        .cuda()
    )
    b = torch.zeros((16,), dtype=torch.int32).cuda()

    compiled_kernel(a, b)
    print(a)
    print(b)


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
