// RUN: not waveasm-translate %s --waveasm-translate-from-llvm 2>&1 | FileCheck %s
// Verify that multi-index GEPs produce a diagnostic instead of crashing.

// CHECK: GEP with multiple indices not yet supported

gpu.module @gpu_module {
  llvm.func @test(%arg0: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %0 = llvm.mlir.constant(0 : i16) : i16
    %1 = llvm.mlir.constant(2147483645 : i64) : i64
    %2 = llvm.mlir.constant(822243328 : i32) : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = rocdl.make.buffer.rsrc %arg0, %0, %1, %2 : !llvm.ptr to <7>
    %6 = llvm.getelementptr %5[%3, %4] : (!llvm.ptr<7>, i32, i32) -> !llvm.ptr<7>, !llvm.array<4 x i32>
    llvm.return
  }
}
