// RUN: not waveasm-translate %s --waveasm-translate-from-llvm 2>&1 | FileCheck %s
// Verify that llvm.mlir.addressof is currently limited to LDS globals.

// CHECK: llvm.mlir.addressof currently supports only LDS globals in addrspace(3)

gpu.module @gpu_module {
  llvm.mlir.global private @global() : i32
  llvm.func @test() attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %0 = llvm.mlir.addressof @global : !llvm.ptr
    llvm.return
  }
}
