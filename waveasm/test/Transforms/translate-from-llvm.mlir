// RUN: not waveasm-translate %s --waveasm-translate-from-llvm 2>&1 | FileCheck %s
// Verify the LLVM→WaveASM translation pass rejects unhandled ops.

gpu.module @test_kernel {
  llvm.func @copy_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>} {
    // CHECK: error: 'llvm.mlir.constant' op unhandled op in LLVM->WaveASM translation
    %c0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return
  }
}
