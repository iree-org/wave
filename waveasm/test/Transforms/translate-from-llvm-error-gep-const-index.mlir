// RUN: not waveasm-translate %s --waveasm-translate-from-llvm 2>&1 | FileCheck %s
// Verify that buffer GEPs with constant attr indices (not dynamic Values)
// produce a diagnostic.

// CHECK: buffer GEP with constant index not yet supported

gpu.module @gpu_module {
  llvm.func @test(%arg0: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %0 = llvm.mlir.constant(0 : i16) : i16
    %1 = llvm.mlir.constant(2147483645 : i64) : i64
    %2 = llvm.mlir.constant(822243328 : i32) : i32
    %3 = rocdl.make.buffer.rsrc %arg0, %0, %1, %2 : !llvm.ptr to <7>
    %4 = llvm.getelementptr %3[42] : (!llvm.ptr<7>) -> !llvm.ptr<7>, i8
    llvm.return
  }
}
