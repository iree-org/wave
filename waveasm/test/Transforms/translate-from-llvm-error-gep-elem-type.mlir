// RUN: not waveasm-translate %s --waveasm-translate-from-llvm 2>&1 | FileCheck %s
// Verify that single-index GEPs with unsupported element types fail instead of
// silently treating the index as a byte offset.

// CHECK: unsupported GEP element type for byte offset computation

gpu.module @gpu_module {
  llvm.func @test(%arg0: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %idx = llvm.mlir.constant(1 : i64) : i64
    %gep = llvm.getelementptr %arg0[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, i32)>
    llvm.return
  }
}
