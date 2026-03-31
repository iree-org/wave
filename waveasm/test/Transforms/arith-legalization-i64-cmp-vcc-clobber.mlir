// RUN: waveasm-translate %s --waveasm-translate-from-llvm --waveasm-arith-legalization | FileCheck %s
// Verify that i64 compare results survive VCC clobbering by intervening
// i64 add operations (v_add_co_u32).  The compare must materialize its
// boolean result to a VGPR, and the select must re-establish VCC from
// that VGPR right before v_cndmask_b32.

// CHECK-LABEL: waveasm.program @test_i64_cmp_vcc_clobber__waveasm

// The i64 ordered compare decomposes into hi/lo comparisons.  The final
// step selects between hi and lo results based on hi-equality.  This
// v_cmp_eq + v_cndmask_b32 produces the materialized boolean.
// CHECK:      waveasm.v_cmp_eq_i32
// CHECK-NEXT: %[[VCC_PH:.*]] = waveasm.constant
// CHECK-NEXT: %[[BOOL:.*]] = waveasm.v_cndmask_b32 %{{.*}}, %{{.*}}, %[[VCC_PH]]

// Intervening i64 add legalizes to v_add_co_u32 which clobbers VCC.
// CHECK: waveasm.v_add_co_u32
// CHECK: waveasm.v_addc_co_u32

// The select re-establishes VCC from the materialized boolean
// right before v_cndmask_b32.
// CHECK:      waveasm.v_cmp_ne_i32 %[[BOOL]], %{{.*}}
// CHECK:      waveasm.v_cndmask_b32
// CHECK:      waveasm.v_cndmask_b32

// No arith pseudo-ops should remain.
// CHECK-NOT: waveasm.arith.
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module {
  llvm.func @test_i64_cmp_vcc_clobber(%arg0: !llvm.ptr, %arg1: i64) attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %c100 = llvm.mlir.constant(100 : i64) : i64
    %c2 = llvm.mlir.constant(2 : i64) : i64
    %tid = rocdl.workitem.id.x range <i32, 0, 64> : i32
    %tid64 = llvm.sext %tid : i32 to i64

    // i64 compare — result must survive across the add below.
    %cmp = llvm.icmp "slt" %tid64, %arg1 : i64

    // i64 add — legalizes to v_add_co_u32 + v_addc_co_u32, clobbering VCC.
    %sum = llvm.add %tid64, %c2 : i64

    // i64 select — must see the compare result, not the stale carry from add.
    %sel = llvm.select %cmp, %tid64, %c100 : i1, i64
    %trunc = llvm.trunc %sel : i64 to i32

    // Use %sum to prevent DCE.
    %sum32 = llvm.trunc %sum : i64 to i32
    llvm.return
  }
}
