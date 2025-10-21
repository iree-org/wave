# RUN: python %s | FileCheck %s


import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import (
    run_test,
)
from wave_lang.support.location_config import (
    LocationCaptureConfig,
    LocationCaptureLevel,
)

M = tkl.sym.M
N = tkl.sym.N
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE


def get_wave_compile_options(
    canonicalize: bool = False,
    dynamic_symbols=[],
    additional_symbols={},
    location_capture_config=LocationCaptureConfig(
        level=LocationCaptureLevel.FILE_LINE_COL
    ),
    drop_debug_info_before_mlir=True,
):
    bindings = {
        M: 227,
        N: 227,
        BLOCK_M: 16,
        BLOCK_N: 16,
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
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
        compile_to_mlir=True,
        location_capture_config=location_capture_config,
        drop_debug_info_before_mlir=drop_debug_info_before_mlir,
    )


@run_test
def test_bounds_check():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(threads_per_wave=64, vector_shapes={M: 16, N: 16})
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def read_write(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        tkw.write(res, b)

    options = get_wave_compile_options(canonicalize=True)
    options.use_bound_check = True
    read_write = wave_compile(options, read_write)
    print(read_write.asm)

    # CHECK-LABEL:    test_bounds_check
    # CHECK:          func.func @read_write
    # CHECK:            %[[C227:.*]] = arith.constant 227 : index
    # CHECK:            %[[D0:.*]] = affine.apply #map()[%block_id_y]

    #                   Mask check
    # CHECK:            %[[D1:.*]] = arith.cmpi slt, %[[D0]], %[[C227]] : index

    # CHECK:            %[[D2:.*]] = affine.apply #map1()[%thread_id_x, %block_id_x]

    #                   Mask check
    # CHECK:            %[[D3:.*]] = arith.cmpi slt, %[[D2]], %[[C227]] : index

    #                   Bounds check
    # CHECK:            %[[D4:.*]] = arith.cmpi sge, %[[D0]], %[[C227]] : index
    # CHECK:            %[[D5:.*]] = arith.cmpi sge, %[[D2]], %[[C227]] : index
    # CHECK:            %[[D6:.*]] = arith.ori %[[D4]], %[[D5]] : i1

    #                   Combine with mask
    # CHECK:            %[[D7:.*]] = arith.andi %[[D3]], %[[D6]] : i1
    # CHECK:            %[[D8:.*]] = arith.andi %[[D7]], %[[D1]] : i1
    # CHECK:            scf.if %[[D8]]
    # CHECK:              gpu.printf
    # CHECK:              llvm.intr.trap


@run_test
def test_bounds_check_mapping():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(threads_per_wave=64, vector_shapes={M: 16, N: 16})
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2, inputs={M: i + 1024, N: j}, outputs={M: i, N: j}
    )

    @tkw.wave(constraints)
    def read_write(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a, mapping=mapping)
        tkw.write(res, b)

    options = get_wave_compile_options(canonicalize=True)
    options.use_bound_check = True
    read_write = wave_compile(options, read_write)
    print(read_write.asm)

    # CHECK-LABEL:    test_bounds_check_mapping
    # CHECK:          func.func @read_write
    # CHECK:            scf.if
    # CHECK:              gpu.printf
    # CHECK:              llvm.intr.trap


@run_test
def test_bounds_check_mapping_dynamic_vals():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(threads_per_wave=64, vector_shapes={M: 16, N: 16})
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
        inputs={M: i + k, N: j},
        outputs={M: i, N: j},
        dynamic_val_mappings={M: i, N: j},
    )

    @tkw.wave(constraints)
    def read_write(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        off: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        off = tkw.read(off)
        res = tkw.read(a, mapping=mapping, mapping_dynamic_vals=(off,))
        tkw.write(res, b)

    options = get_wave_compile_options(canonicalize=True)
    options.use_bound_check = True
    read_write = wave_compile(options, read_write)
    print(read_write.asm)

    # CHECK-LABEL:    test_bounds_check_mapping_dynamic_vals
    # CHECK:          func.func @read_write
    # CHECK:            scf.if
    # CHECK:              gpu.printf
    # CHECK:              llvm.intr.trap
