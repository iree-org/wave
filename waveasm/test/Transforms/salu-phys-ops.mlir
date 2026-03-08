// RUN: waveasm-translate --waveasm-scoped-cse %s 2>&1 | FileCheck %s --check-prefix=CSE
// RUN: waveasm-translate --disable-pass-verifier --waveasm-linear-scan --emit-assembly %s | FileCheck %s --check-prefix=ASM
//
// Test SALUPhys ops: non-Pure physical-register-write variants of SALU
// instructions.  They must survive CSE (SpecialRegOp trait) and emit
// the correct assembly mnemonic (without the _phys suffix).

//===----------------------------------------------------------------------===//
// Test 1: s_mov_b32_phys / s_mov_b64_phys survive CSE
//===----------------------------------------------------------------------===//

// CSE-LABEL: waveasm.program @phys_mov_survives_cse
// ASM-LABEL: phys_mov_survives_cse:
waveasm.program @phys_mov_survives_cse target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %dst0 = waveasm.precolored.sreg 8 : !waveasm.psreg<8>
  %dst1 = waveasm.precolored.sreg 9 : !waveasm.psreg<9>
  %dst_pair = waveasm.precolored.sreg 10, 2 : !waveasm.psreg<10, 2>
  %src_pair = waveasm.precolored.sreg 2, 2 : !waveasm.psreg<2, 2>
  %imm_size = waveasm.constant 4096 : !waveasm.imm<4096>
  %imm_stride = waveasm.constant 131072 : !waveasm.imm<131072>

  // Two s_mov_b32_phys to different destinations -- both must survive DCE
  // (zero results + SpecialRegOp trait prevents trivial dead elimination).
  // CSE: waveasm.s_mov_b32_phys
  // CSE: waveasm.s_mov_b32_phys
  waveasm.s_mov_b32_phys %dst0, %imm_size : !waveasm.psreg<8>, !waveasm.imm<4096>
  waveasm.s_mov_b32_phys %dst1, %imm_stride : !waveasm.psreg<9>, !waveasm.imm<131072>

  // s_mov_b64_phys must also survive.
  // CSE: waveasm.s_mov_b64_phys
  waveasm.s_mov_b64_phys %dst_pair, %src_pair : !waveasm.psreg<10, 2>, !waveasm.psreg<2, 2>

  // ASM: s_mov_b32 s8, 4096
  // ASM: s_mov_b32 s9, 131072
  // ASM: s_mov_b64 s[10:11], s[2:3]

  // CSE: waveasm.s_endpgm
  // ASM: s_endpgm
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 2: AND-OR pattern uses Pure SSA ops + final MOV_PHYS write.
// The SSA data dependency AND->OR->MOV enforces ordering structurally.
//===----------------------------------------------------------------------===//

// CSE-LABEL: waveasm.program @ssa_and_or_with_phys_write
// ASM-LABEL: ssa_and_or_with_phys_write:
waveasm.program @ssa_and_or_with_phys_write target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %dst = waveasm.precolored.sreg 5 : !waveasm.psreg<5>
  %src = waveasm.precolored.sreg 3 : !waveasm.psreg<3>
  %mask = waveasm.constant 65535 : !waveasm.imm<65535>
  %flags = waveasm.constant 1077936128 : !waveasm.imm<1077936128>

  // Pure AND and OR compute in SSA; final MOV_PHYS writes to physical reg.
  // CSE: waveasm.s_and_b32
  // CSE: waveasm.s_or_b32
  // CSE: waveasm.s_mov_b32_phys
  %and = waveasm.s_and_b32 %src, %mask : !waveasm.psreg<3>, !waveasm.imm<65535> -> !waveasm.sreg
  %or = waveasm.s_or_b32 %and, %flags : !waveasm.sreg, !waveasm.imm<1077936128> -> !waveasm.sreg
  waveasm.s_mov_b32_phys %dst, %or : !waveasm.psreg<5>, !waveasm.sreg

  // ASM: s_and_b32
  // ASM: s_or_b32
  // ASM: s_mov_b32 s5,

  // CSE: waveasm.s_endpgm
  // ASM: s_endpgm
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 3: Pure s_mov_b32 IS still CSE'd (contrast with _phys variant)
//===----------------------------------------------------------------------===//

// CSE-LABEL: waveasm.program @pure_mov_is_csed
waveasm.program @pure_mov_is_csed target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %imm = waveasm.constant 42 : !waveasm.imm<42>

  // Two identical Pure s_mov_b32 -- second should be CSE'd away.
  // CSE: waveasm.s_mov_b32
  %r0 = waveasm.s_mov_b32 %imm : !waveasm.imm<42> -> !waveasm.sreg
  // CSE-NOT: waveasm.s_mov_b32 %{{.*}} : !waveasm.imm<42>
  %r1 = waveasm.s_mov_b32 %imm : !waveasm.imm<42> -> !waveasm.sreg

  waveasm.s_endpgm
}
