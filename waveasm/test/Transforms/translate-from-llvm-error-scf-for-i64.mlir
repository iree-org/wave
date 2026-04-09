// RUN: not waveasm-translate %s --waveasm-translate-from-llvm 2>&1 | FileCheck %s
// Verify that scf.for loop control stays explicitly limited to i32 lowering.

// CHECK: scf.for lower bound must lower to i32

gpu.module @gpu_module {
  llvm.func @test() attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %lb = llvm.mlir.constant(0 : i64) : i64
    %ub = llvm.mlir.constant(64 : i64) : i64
    %step = llvm.mlir.constant(16 : i64) : i64
    %init = llvm.mlir.constant(0 : i32) : i32
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (i32) : i64 {
      scf.yield %acc : i32
    }
    llvm.return
  }
}
