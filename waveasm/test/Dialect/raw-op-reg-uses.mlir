// RUN: waveasm-translate %s | FileCheck %s
// Verify RawOp register-use attributes parse and roundtrip correctly.

// CHECK-LABEL: waveasm.program @raw_op_reg_uses
waveasm.program @raw_op_reg_uses target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Plain raw op (no annotations).
  // CHECK: waveasm.raw "s_nop 0"{{$}}
  waveasm.raw "s_nop 0"

  // Single sreg_uses.
  // CHECK: waveasm.raw "s_and_b32 s5, s5, 0xFFFF" sreg_uses = [5]
  waveasm.raw "s_and_b32 s5, s5, 0xFFFF" sreg_uses = [5]

  // Multiple sreg_uses.
  // CHECK: waveasm.raw "s_mov_b64 s[4:5], s[0:1]" sreg_uses = [0, 1]
  waveasm.raw "s_mov_b64 s[4:5], s[0:1]" sreg_uses = [0, 1]

  // vreg_uses only.
  // CHECK: waveasm.raw "v_nop" vreg_uses = [3, 4]
  waveasm.raw "v_nop" vreg_uses = [3, 4]

  // areg_uses only.
  // CHECK: waveasm.raw "s_nop 1" areg_uses = [0]
  waveasm.raw "s_nop 1" areg_uses = [0]

  // All three register-use kinds.
  // CHECK: waveasm.raw "s_nop 2" sreg_uses = [10] vreg_uses = [0] areg_uses = [0, 1]
  waveasm.raw "s_nop 2" sreg_uses = [10] vreg_uses = [0] areg_uses = [0, 1]

  waveasm.s_endpgm
}
