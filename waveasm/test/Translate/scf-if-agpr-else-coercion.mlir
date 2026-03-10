// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: scf.if where the then branch yields a computed VGPR value and the
// else branch yields a constant (which translates to an immediate).
//
// In split-K kernels the prologue is wrapped in an scf.if that guards
// against out-of-bounds splits.  The then branch executes computation
// (producing registers) while the else branch yields zero-initialized
// values (immediates).  The backend must coerce the else-yield immediates
// into register types so that both branches yield type-compatible values.

module {
  gpu.module @test_if_else_coercion {

    // CHECK-LABEL: waveasm.program @if_else_coercion
    gpu.func @if_else_coercion() kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cond_i1 = arith.cmpi slt, %c0, %c1 : index

      %zero_i32 = arith.constant 0 : i32
      %one_i32 = arith.constant 1 : i32

      // Then branch: compute a VGPR value (addi -> v_add_u32)
      // Else branch: yield a constant zero (-> immediate)
      // Backend must coerce the else-yield immediate to VGPR.
      //
      // CHECK:      waveasm.if
      // CHECK:        waveasm.v_add_u32
      // CHECK:        waveasm.yield {{.*}} : !waveasm.vreg
      // CHECK:      } else {
      // CHECK:        waveasm.v_mov_b32 {{.*}} -> !waveasm.vreg
      // CHECK:        waveasm.yield {{.*}} : !waveasm.vreg
      %result = scf.if %cond_i1 -> i32 {
        %val = arith.addi %zero_i32, %one_i32 : i32
        scf.yield %val : i32
      } else {
        scf.yield %zero_i32 : i32
      }

      // CHECK: waveasm.s_endpgm
      gpu.return
    }
  }
}
