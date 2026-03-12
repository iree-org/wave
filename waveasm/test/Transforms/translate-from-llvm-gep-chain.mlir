// RUN: waveasm-translate %s --waveasm-translate-from-llvm | FileCheck %s
// Verify chained GEPs (GEP on GEP) are handled by adding offsets.

// CHECK: waveasm.program @test__waveasm
// CHECK: waveasm.v_add_u32
// CHECK: waveasm.buffer_load_dword
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module {
  llvm.func @test(%arg0: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %0 = llvm.mlir.constant(0 : i16) : i16
    %1 = llvm.mlir.constant(2147483645 : i64) : i64
    %2 = llvm.mlir.constant(822243328 : i32) : i32
    %3 = rocdl.workitem.id.x range <i32, 0, 64> : i32
    %4 = llvm.sext %3 : i32 to i64
    %5 = rocdl.make.buffer.rsrc %arg0, %0, %1, %2 : !llvm.ptr to <7>
    // First GEP from buffer resource.
    %6 = llvm.getelementptr nusw %5[%4] : (!llvm.ptr<7>, i64) -> !llvm.ptr<7>, i8
    // Chained GEP on previous GEP result.
    %7 = llvm.mlir.constant(256 : i64) : i64
    %8 = llvm.getelementptr nusw %6[%7] : (!llvm.ptr<7>, i64) -> !llvm.ptr<7>, i8
    %9 = llvm.load %8 : !llvm.ptr<7> -> i32
    llvm.return
  }
}
