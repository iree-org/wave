// RUN: waveasm-translate %s --waveasm-translate-from-llvm --waveasm-arith-legalization | FileCheck %s
// Verify that i64 constants are split into lo/hi halves and that i64 select
// legalizes correctly through the full translation + legalization pipeline.

// CHECK-LABEL: waveasm.program @test_i64_select__waveasm

// i64 constant 100 split into v_mov_b32(100) + v_mov_b32(0) + pack.
// CHECK-DAG: %[[C100_LO:.*]] = waveasm.v_mov_b32 %{{.*}} : !waveasm.imm<100> -> !waveasm.vreg
// CHECK-DAG: %[[C100_HI:.*]] = waveasm.v_mov_b32 %{{.*}} : !waveasm.imm<0> -> !waveasm.vreg
// CHECK: waveasm.pack %[[C100_LO]], %[[C100_HI]]
// CHECK-SAME: -> !waveasm.vreg<2, 2>

// sext i32 -> i64: v_ashrrev_i32 for sign extension + pack.
// CHECK: %[[TID:.*]] = waveasm.precolored.vreg 0 : !waveasm.vreg
// CHECK: %[[SIGN:.*]] = waveasm.v_ashrrev_i32 %{{.*}}, %[[TID]]
// CHECK: waveasm.pack %[[TID]], %[[SIGN]]
// CHECK-SAME: -> !waveasm.vreg<2, 2>

// i32 compare sets VCC.
// CHECK: waveasm.v_cmp_lt_i32

// i64 select: splitI64 looks through pack, uses original lo/hi directly.
// CHECK: waveasm.v_cndmask_b32 %[[C100_LO]], %[[TID]], %{{.*}}
// CHECK: waveasm.v_cndmask_b32 %[[C100_HI]], %[[SIGN]], %{{.*}}
// CHECK: waveasm.pack %{{.*}}, %{{.*}} : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2, 2>

// No arith pseudo-ops should remain.
// CHECK-NOT: waveasm.arith.
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module {
  llvm.func @test_i64_select() attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %c100 = llvm.mlir.constant(100 : i64) : i64
    %c27 = llvm.mlir.constant(27 : i32) : i32
    %tid = rocdl.workitem.id.x range <i32, 0, 64> : i32
    %ext = llvm.sext %tid : i32 to i64
    %cmp = llvm.icmp "slt" %tid, %c27 : i32
    %sel = llvm.select %cmp, %ext, %c100 : i1, i64
    %trunc = llvm.trunc %sel : i64 to i32
    llvm.return
  }
}
