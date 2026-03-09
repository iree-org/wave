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

gpu.module @gpu_module attributes {llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"} {
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
