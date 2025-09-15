import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
import torch

from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile


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
    canonicalize: bool = False, dynamic_symbols=[], additional_symbols={}
):
    bindings = {
        M: 16,
        N: 16,
        K: 16,
        BLOCK_M: 16,
        BLOCK_N: 16,
        BLOCK_K: 16,
        ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
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
        get_wave_compile_options(canonicalize=True, additional_symbols={ONE: 1}),
        read_write_dynamic_mapping_broadcast,
    )
    print(read_write_dynamic_mapping_broadcast.asm)

    # create input tensors
    a = torch.arange(0, 256, dtype=torch.int32).reshape(16, 16).cuda()
    off = torch.arange(0, 16, dtype=torch.int32).reshape(16, 1).cuda()
    b = torch.zeros((16, 16), dtype=torch.int32).cuda()

    read_write_dynamic_mapping_broadcast(a, off, b)
    print(a)
    print(off)
    print(b)


def test_one_read_write_dynamic_mapping_broadcast():
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
        offset = tkw.Register[ONE, tkl.i32](1)
        # offset = tkw.scalar(1, tkl.i32)
        # res = tkw.read(
        #     a,
        #     mapping=mapping,
        #     mapping_dynamic_vals=(offset,),
        # )
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


def test_iteration_with_condition():
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


def test_atomic_add_return_value():
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
    def iterated_gemm(
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
        print_ir_after="all",
        print_ir_before="all",
    )
    iterated_gemm = wave_compile(options, iterated_gemm)
    print(iterated_gemm.asm)

    # generate random input tensors between -1 and 1
    a = torch.randint(1, 2, (64,), dtype=torch.int32).cuda()
    c = torch.zeros((64,), dtype=torch.int32).cuda()

    iterated_gemm(a, c)
    print(a)
    print(c)


if __name__ == "__main__":
    import sys

    globals()[sys.argv[1]]()
