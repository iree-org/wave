// RUN: waveasm-translate %s --waveasm-translate-from-llvm | FileCheck %s
// Verify sdiv and srem with positive power-of-2 constants.
// Negative dividends require bias/correction so we should see compare/select
// scaffolding in addition to the final shift/mask-shaped arithmetic.

// CHECK: waveasm.program @test__waveasm
// sdiv lowers through signed-bias correction before the arithmetic shift.
// CHECK: waveasm.arith.cmp slt
// CHECK: waveasm.arith.select
// CHECK: waveasm.v_ashrrev_i32
// srem keeps LLVM sign semantics via mask + conditional correction.
// CHECK: waveasm.arith.and
// CHECK: waveasm.arith.add
// CHECK: waveasm.arith.select
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module {
  llvm.func @test() attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %tid = rocdl.workitem.id.x range <i32, 0, 64> : i32
    %16 = llvm.mlir.constant(16 : i32) : i32
    %neg5 = llvm.mlir.constant(-5 : i32) : i32
    %4 = llvm.mlir.constant(4 : i32) : i32
    %div = llvm.sdiv %tid, %16 : i32
    %rem = llvm.srem %tid, %16 : i32
    %negdiv = llvm.sdiv %neg5, %4 : i32
    %negrem = llvm.srem %neg5, %4 : i32
    llvm.return
  }
}
