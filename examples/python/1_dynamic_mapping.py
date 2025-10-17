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
    )


def test_read_write_dynamic_mapping_broadcast():
    """Read from 2D tensor using dynamic offsets per row, demonstrating runtime-computed indices with broadcast."""
    ONE = tkl.sym.ONE
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: 16, N: 16, ONE: 1},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.dynamic_val(0)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: k + j % 16},
        outputs={M: i, N: j},
        dynamic_val_mappings={M: i, ONE: j // 16},
    )

    @tkw.wave(constraints)
    def read_write_dynamic_mapping_broadcast(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        off: tkl.Memory[M, ONE, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
    ):
        offset = tkw.read(off)
        res = tkw.read(
            a,
            mapping=mapping,
            mapping_dynamic_vals=(offset,),
        )
        tkw.write(res, b)

    read_write_dynamic_mapping_broadcast = wave_compile(
        get_wave_compile_options(
            m=16,
            n=16,
            k=16,
            block_m=16,
            block_n=16,
            block_k=16,
            address_space=tkl.AddressSpace.SHARED_MEMORY.value,
            canonicalize=True,
            additional_symbols={ONE: 1, BLOCK_N: 16},
        ),
        read_write_dynamic_mapping_broadcast,
    )
    print(read_write_dynamic_mapping_broadcast.asm)

    # Need at least offset_max + 16 columns
    a = torch.arange(0, 16 * 16, dtype=torch.int32).reshape(16, 16).cuda()
    off = torch.zeros((16, 1), dtype=torch.int32).cuda()
    b = torch.zeros((16, 16), dtype=torch.int32).cuda()

    read_write_dynamic_mapping_broadcast(a, off, b)
    print(a)
    print(off)
    print(b)


def test_one_read_write_dynamic_mapping_broadcast():
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
    def read_write_dynamic_mapping_broadcast(
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

    read_write_dynamic_mapping_broadcast = wave_compile(
        get_wave_compile_options(canonicalize=True, additional_symbols={ONE: 1}),
        read_write_dynamic_mapping_broadcast,
    )
    print(read_write_dynamic_mapping_broadcast.asm)

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

    read_write_dynamic_mapping_broadcast(a, off, b)
    print(a)
    print(off)
    print(b)


def test_one_nooffset_dynamic_mapping_broadcast():
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
        outputs={N: j + 1},
    )

    @tkw.wave(constraints)
    def read_write_dynamic_mapping_broadcast(
        a: tkl.Memory[N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[N, ADDRESS_SPACE, tkl.i32],
    ):
        temp = tkw.Register[N, tkl.i32](0)
        temp = tkw.read(
            a,
            mapping=seq_len_mapping,
        )
        tkw.write(temp, b, mapping=seq_len_mapping_w)

    read_write_dynamic_mapping_broadcast = wave_compile(
        get_wave_compile_options(canonicalize=True, additional_symbols={ONE: 1}),
        read_write_dynamic_mapping_broadcast,
    )
    print(read_write_dynamic_mapping_broadcast.asm)

    # create input tensors
    a = (
        torch.arange(0, 16, dtype=torch.int32)
        .reshape(
            16,
        )
        .cuda()
    )
    b = torch.zeros((16,), dtype=torch.int32).cuda()

    read_write_dynamic_mapping_broadcast(a, b)
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
