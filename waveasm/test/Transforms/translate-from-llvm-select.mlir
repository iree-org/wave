// RUN: waveasm-translate %s --waveasm-translate-from-llvm | FileCheck %s
// Verify llvm.select is translated to waveasm.arith.select pseudo-op
// with correct operand wiring from icmp condition.

// CHECK: waveasm.program @test__waveasm
// CHECK: [[C10:%.*]] = waveasm.constant 10
// CHECK: [[A:%.*]] = waveasm.v_mov_b32 [[C10]]
// CHECK: [[C20:%.*]] = waveasm.constant 20
// CHECK: [[B:%.*]] = waveasm.v_mov_b32 [[C20]]
// CHECK: [[C1:%.*]] = waveasm.constant 1
// CHECK: [[X:%.*]] = waveasm.v_mov_b32 [[C1]]
// CHECK: [[C2:%.*]] = waveasm.constant 2
// CHECK: [[Y:%.*]] = waveasm.v_mov_b32 [[C2]]
// CHECK: [[CMP:%.*]] = waveasm.arith.cmp slt, [[A]], [[B]]
// CHECK: waveasm.arith.select [[CMP]], [[X]], [[Y]]
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module {
  llvm.func @test() attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %a = llvm.mlir.constant(10 : i32) : i32
    %b = llvm.mlir.constant(20 : i32) : i32
    %x = llvm.mlir.constant(1 : i32) : i32
    %y = llvm.mlir.constant(2 : i32) : i32
    %cmp = llvm.icmp "slt" %a, %b : i32
    %sel = llvm.select %cmp, %x, %y : i1, i32
    llvm.return
  }
}
