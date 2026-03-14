// RUN: waveasm-translate %s --waveasm-translate-from-llvm | FileCheck %s
// Verify wide (dwordx2/x4) buffer loads and stores.

// CHECK: waveasm.program @test__waveasm
// CHECK: waveasm.buffer_load_dwordx2
// CHECK: waveasm.buffer_load_dwordx4
// CHECK: waveasm.buffer_store_dwordx2
// CHECK: waveasm.buffer_store_dwordx4
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module {
  llvm.func @test(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %0 = llvm.mlir.constant(0 : i16) : i16
    %1 = llvm.mlir.constant(2147483645 : i64) : i64
    %2 = llvm.mlir.constant(822243328 : i32) : i32
    %3 = rocdl.workitem.id.x range <i32, 0, 64> : i32
    %4 = llvm.sext %3 : i32 to i64
    %5 = rocdl.make.buffer.rsrc %arg0, %0, %1, %2 : !llvm.ptr to <7>
    %6 = llvm.getelementptr nusw %5[%4] : (!llvm.ptr<7>, i64) -> !llvm.ptr<7>, i8
    // 8-byte load (dwordx2).
    %7 = llvm.load %6 : !llvm.ptr<7> -> vector<4xf16>
    // 16-byte load (dwordx4).
    %8 = llvm.load %6 : !llvm.ptr<7> -> vector<4xf32>
    %9 = rocdl.make.buffer.rsrc %arg1, %0, %1, %2 : !llvm.ptr to <7>
    %10 = llvm.getelementptr nusw %9[%4] : (!llvm.ptr<7>, i64) -> !llvm.ptr<7>, i8
    // 8-byte store (dwordx2).
    llvm.store %7, %10 : vector<4xf16>, !llvm.ptr<7>
    // 16-byte store (dwordx4).
    llvm.store %8, %10 : vector<4xf32>, !llvm.ptr<7>
    llvm.return
  }
}
