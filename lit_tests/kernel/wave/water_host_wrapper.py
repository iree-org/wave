# REQUIRES: water
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
    canonicalize: bool = False,
    dynamic_symbols=[],
    additional_symbols={},
    location_capture_config=LocationCaptureConfig(
        level=LocationCaptureLevel.FILE_LINE_COL
    ),
    drop_debug_info_before_mlir=True,
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
        compile_to_mlir=True,
        location_capture_config=location_capture_config,
        drop_debug_info_before_mlir=drop_debug_info_before_mlir,
        use_water_pipeline=True,
    )


def get_constraints():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(threads_per_wave=64, vector_shapes={M: 16, N: 16})
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]
    return constraints


@run_test
def test_read_write():
    constraints = get_constraints()

    @tkw.wave(constraints)
    def read_write(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        tkw.write(res, b)

    read_write = wave_compile(get_wave_compile_options(canonicalize=True), read_write)
    print(read_write.asm)

    # CHECK-LABEL:    test_read_write
    # CHECK-DAG:        #[[MAP0:.*]] = affine_map<()[s0] -> (s0 - (s0 floordiv 64) * 48)>
    # CHECK:          gpu.module @gpu_module
    # CHECK:          gpu.func @read_write
    # CHECK-SAME:       (%[[D0:.*]]: memref<f16>, %[[D1:.*]]: memref<f16>)
    # CHECK-DAG:        %[[C0:.*]] = arith.constant 0 : index
    # CHECK:            %[[thread_id_x:.*]] = gpu.thread_id  x
    # CHECK:            %[[S0:.*]] = memref.reinterpret_cast %[[D0]] to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<f16> to memref<16x16xf16, strided<[16, 1]>>
    # CHECK:            %[[S1:.*]] = memref.reinterpret_cast %[[D1]] to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<f16> to memref<16x16xf16, strided<[16, 1]>>
    # CHECK:            %[[I0:.*]] = affine.apply #[[MAP0]]()[%[[thread_id_x]]]
    # CHECK:            %[[V:.*]] = vector.load %[[S0]][%[[I0]], %[[C0]]] : memref<16x16xf16, strided<[16, 1]>>, vector<16xf16>
    # CHECK:            vector.store %[[V]], %[[S1]][%[[I0]], %[[C0]]] : memref<16x16xf16, strided<[16, 1]>>, vector<16xf16>
    # CHECK:            return

    # CHECK-LABEL:    func.func @isolated_benchmark
    # CHECK-SAME:       (%[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr, %[[ARG2:.*]]: !llvm.ptr)
    # CHECK-DAG:        %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG:        %[[C1:.*]] = arith.constant 1 : index
    # CHECK-DAG:        %[[C64:.*]] = arith.constant 64 : index
    # CHECK:            %[[BUF1:.*]] = call @wave_get_buffer(%[[ARG1]]) : (!llvm.ptr) -> memref<?xi8>
    # CHECK:            %[[VIEW1:.*]] = memref.view %[[BUF1]][%[[C0]]][] : memref<?xi8> to memref<f16>
    # CHECK:            %[[BUF2:.*]] = call @wave_get_buffer(%[[ARG2]]) : (!llvm.ptr) -> memref<?xi8>
    # CHECK:            %[[VIEW2:.*]] = memref.view %[[BUF2]][%[[C0]]][] : memref<?xi8> to memref<f16>
    # CHECK:            gpu.launch_func @gpu_module::@read_write blocks in (%[[C1]], %[[C1]], %[[C1]]) threads in (%[[C64]], %[[C1]], %[[C1]]) args(%[[VIEW1]] : memref<f16>, %[[VIEW2]] : memref<f16>)
    # CHECK:            return


@run_test
def test_scalars():
    constraints = get_constraints()

    @tkw.wave(constraints)
    def scalars(
        b: tkl.f32,  # TODO: cannot pass tkl.f16 yet as we need to link compiler-rt into execution engine
        c: tkl.i32,
        d: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        b = tkw.broadcast(b, target_shape=[M, N])
        c = tkw.broadcast(c, target_shape=[M, N])
        res = tkw.cast(b, tkl.f16) + tkw.cast(c, tkl.f16)
        tkw.write(res, d)

    scalars = wave_compile(get_wave_compile_options(canonicalize=True), scalars)
    print(scalars.asm)

    # CHECK-LABEL:    test_scalars
    # CHECK-LABEL:    func.func @isolated_benchmark
    # CHECK-SAME:       (%[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr, %[[ARG2:.*]]: !llvm.ptr, %[[ARG3:.*]]: !llvm.ptr)
    # CHECK-DAG:        %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG:        %[[C1:.*]] = arith.constant 1 : index
    # CHECK-DAG:        %[[C64:.*]] = arith.constant 64 : index
    # CHECK:            %[[BUF:.*]] = call @wave_get_buffer(%[[ARG3]]) : (!llvm.ptr) -> memref<?xi8>
    # CHECK:            %[[VIEW:.*]] = memref.view %[[BUF]][%[[C0]]][] : memref<?xi8> to memref<f16>
    # CHECK:            %[[F64:.*]] = call @wave_get_float64(%[[ARG1]]) : (!llvm.ptr) -> f64
    # CHECK:            %[[F32:.*]] = arith.truncf %[[F64]] : f64 to f32
    # CHECK:            %[[I64:.*]] = call @wave_get_int64(%[[ARG2]]) : (!llvm.ptr) -> i64
    # CHECK:            %[[I32:.*]] = arith.trunci %[[I64]] : i64 to i32
    # CHECK:            gpu.launch_func @gpu_module::@scalars blocks in (%[[C1]], %[[C1]], %[[C1]]) threads in (%[[C64]], %[[C1]], %[[C1]]) args(%[[VIEW]] : memref<f16>, %[[F32]] : f32, %[[I32]] : i32)
    # CHECK:            return


@run_test
def test_dynamic_symbols():
    constraints = get_constraints()

    @tkw.wave(constraints)
    def read_write(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        tkw.write(res, b)

    read_write = wave_compile(
        get_wave_compile_options(canonicalize=True, dynamic_symbols=[M, N]), read_write
    )
    print(read_write.asm)

    # CHECK-LABEL:    test_dynamic_symbols

    # CHECK-LABEL:    func.func @isolated_benchmark
    # CHECK-SAME:       (%[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr, %[[ARG2:.*]]: !llvm.ptr)
    # CHECK-DAG:        %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG:        %[[C1:.*]] = arith.constant 1 : index
    # CHECK-DAG:        %[[C64:.*]] = arith.constant 64 : index
    # CHECK-DAG:        %[[C0_I32:.*]] = arith.constant 0 : i32
    # CHECK-DAG:        %[[C1_I32:.*]] = arith.constant 1 : i32
    # CHECK:            %[[DIM0_I64:.*]] = call @wave_get_dim(%[[ARG1]], %[[C0_I32]]) : (!llvm.ptr, i32) -> i64
    # CHECK:            %[[DIM0:.*]] = arith.index_cast %[[DIM0_I64]] : i64 to index
    # CHECK:            %[[DIM1_I64:.*]] = call @wave_get_dim(%[[ARG1]], %[[C1_I32]]) : (!llvm.ptr, i32) -> i64
    # CHECK:            %[[DIM1:.*]] = arith.index_cast %[[DIM1_I64]] : i64 to index
    # CHECK:            %[[BLOCK_M:.*]] = affine.apply #map2()[%[[DIM0]]]
    # CHECK:            %[[BLOCK_N:.*]] = affine.apply #map2()[%[[DIM1]]]
    # CHECK:            %[[BUF1:.*]] = call @wave_get_buffer(%[[ARG1]]) : (!llvm.ptr) -> memref<?xi8>
    # CHECK:            %[[VIEW1:.*]] = memref.view %[[BUF1]][%[[C0]]][] : memref<?xi8> to memref<f16>
    # CHECK:            %[[BUF2:.*]] = call @wave_get_buffer(%[[ARG2]]) : (!llvm.ptr) -> memref<?xi8>
    # CHECK:            %[[VIEW2:.*]] = memref.view %[[BUF2]][%[[C0]]][] : memref<?xi8> to memref<f16>
    # CHECK:            gpu.launch_func @gpu_module::@read_write blocks in (%[[BLOCK_M]], %[[BLOCK_N]], %[[C1]]) threads in (%[[C64]], %[[C1]], %[[C1]]) args(%[[VIEW1]] : memref<f16>, %[[VIEW2]] : memref<f16>, %[[DIM0]] : index, %[[DIM1]] : index)
    # CHECK:            return
