// RUN: waveasm-translate %s --waveasm-translate-from-llvm | FileCheck %s
// Verify scf.for translation to waveasm.loop with condition terminator.
// The loop body contains an add; iter_args carry the accumulator.

// CHECK: waveasm.program @test__waveasm
// Loop with one iter_arg (the accumulator).
// CHECK: waveasm.loop
// Loop body: arith.add for the accumulation.
// CHECK: waveasm.arith.add
// IV increment.
// CHECK: waveasm.arith.add
// Condition: s_cmp_lt_u32 for the loop back-edge.
// CHECK: waveasm.s_cmp_lt_u32
// CHECK: waveasm.condition
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module {
  llvm.func @test() attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %lb = llvm.mlir.constant(0 : i32) : i32
    %ub = llvm.mlir.constant(64 : i32) : i32
    %step = llvm.mlir.constant(16 : i32) : i32
    %init = llvm.mlir.constant(0 : i32) : i32
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (i32) : i32 {
      %sum = llvm.add %acc, %iv : i32
      scf.yield %sum : i32
    }
    llvm.return
  }
}
