// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test: When a PackOp input comes from an ExtractOp on a different packed
// value, the liveness pass must insert an explicit s_mov_b32/v_mov_b32 copy.
//
// ExtractOp is non-emitting (it aliases source[offset] without generating an
// instruction). The PackOp post-pass assigns input[i] = result_base + i,
// which may differ from the extract's physical register. Without a copy the
// SRD word silently gets the wrong value.

// -----------------------------------------------------------------------
// Test 1: Extract one word from a source SRD and repack into a new SRD.
// The extract result must be copied into the new pack's register slot.
// -----------------------------------------------------------------------

// CHECK-LABEL: waveasm.program @extract_word_into_new_pack
waveasm.program @extract_word_into_new_pack target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %imm0 = waveasm.constant 0 : !waveasm.imm<0>
  %imm42 = waveasm.constant 42 : !waveasm.imm<42>

  %w0 = waveasm.s_mov_b32 %imm0 : !waveasm.imm<0> -> !waveasm.sreg
  %w1 = waveasm.s_mov_b32 %imm0 : !waveasm.imm<0> -> !waveasm.sreg
  %w2 = waveasm.s_mov_b32 %imm42 : !waveasm.imm<42> -> !waveasm.sreg
  %w3 = waveasm.s_mov_b32 %imm0 : !waveasm.imm<0> -> !waveasm.sreg
  %src = waveasm.pack %w0, %w1, %w2, %w3
       : (!waveasm.sreg, !waveasm.sreg, !waveasm.sreg, !waveasm.sreg)
       -> !waveasm.sreg<4>

  // Keep source alive past the new pack.
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %ld = waveasm.buffer_load_dword %src, %voff, %imm0
      : !waveasm.sreg<4>, !waveasm.pvreg<0>, !waveasm.imm<0> -> !waveasm.vreg

  // Extract word 2 from the source and build a new 4-wide SRD.
  // CHECK: waveasm.extract {{.*}}[2]
  %ext = waveasm.extract %src[2] : !waveasm.sreg<4> -> !waveasm.sreg

  %n0 = waveasm.s_mov_b32 %imm0 : !waveasm.imm<0> -> !waveasm.sreg
  %n1 = waveasm.s_mov_b32 %imm0 : !waveasm.imm<0> -> !waveasm.sreg
  %n3 = waveasm.s_mov_b32 %imm0 : !waveasm.imm<0> -> !waveasm.sreg

  // A copy of the extract result must appear before the pack.
  // CHECK: waveasm.s_mov_b32 %{{.*}} : !waveasm.psreg<{{[0-9]+}}> -> !waveasm.psreg<{{[0-9]+}}>
  // CHECK-NEXT: waveasm.pack
  %dst = waveasm.pack %n0, %n1, %ext, %n3
       : (!waveasm.sreg, !waveasm.sreg, !waveasm.sreg, !waveasm.sreg)
       -> !waveasm.sreg<4>

  waveasm.buffer_store_dword %ld, %dst, %voff
      : !waveasm.vreg, !waveasm.sreg<4>, !waveasm.pvreg<0>
  waveasm.s_endpgm
}

// -----------------------------------------------------------------------
// Test 2: Two words extracted from the same source and repacked.
// Both extracts must get their own copy.
// -----------------------------------------------------------------------

// CHECK-LABEL: waveasm.program @two_extracts_into_new_pack
waveasm.program @two_extracts_into_new_pack target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %imm0 = waveasm.constant 0 : !waveasm.imm<0>
  %imm7 = waveasm.constant 7 : !waveasm.imm<7>
  %imm9 = waveasm.constant 9 : !waveasm.imm<9>

  %w0 = waveasm.s_mov_b32 %imm0 : !waveasm.imm<0> -> !waveasm.sreg
  %w1 = waveasm.s_mov_b32 %imm0 : !waveasm.imm<0> -> !waveasm.sreg
  %w2 = waveasm.s_mov_b32 %imm7 : !waveasm.imm<7> -> !waveasm.sreg
  %w3 = waveasm.s_mov_b32 %imm9 : !waveasm.imm<9> -> !waveasm.sreg
  %src = waveasm.pack %w0, %w1, %w2, %w3
       : (!waveasm.sreg, !waveasm.sreg, !waveasm.sreg, !waveasm.sreg)
       -> !waveasm.sreg<4>

  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %ld = waveasm.buffer_load_dword %src, %voff, %imm0
      : !waveasm.sreg<4>, !waveasm.pvreg<0>, !waveasm.imm<0> -> !waveasm.vreg

  // Extract words 2 and 3, provide new words 0 and 1.
  // CHECK: waveasm.extract {{.*}}[2]
  // CHECK: waveasm.extract {{.*}}[3]
  %ext2 = waveasm.extract %src[2] : !waveasm.sreg<4> -> !waveasm.sreg
  %ext3 = waveasm.extract %src[3] : !waveasm.sreg<4> -> !waveasm.sreg

  %n0 = waveasm.s_mov_b32 %imm0 : !waveasm.imm<0> -> !waveasm.sreg
  %n1 = waveasm.s_mov_b32 %imm0 : !waveasm.imm<0> -> !waveasm.sreg

  // Both extracts get copies before the pack.
  // CHECK: waveasm.s_mov_b32 %{{.*}} : !waveasm.psreg<{{[0-9]+}}> -> !waveasm.psreg<{{[0-9]+}}>
  // CHECK: waveasm.s_mov_b32 %{{.*}} : !waveasm.psreg<{{[0-9]+}}> -> !waveasm.psreg<{{[0-9]+}}>
  // CHECK: waveasm.pack
  %dst = waveasm.pack %n0, %n1, %ext2, %ext3
       : (!waveasm.sreg, !waveasm.sreg, !waveasm.sreg, !waveasm.sreg)
       -> !waveasm.sreg<4>

  waveasm.buffer_store_dword %ld, %dst, %voff
      : !waveasm.vreg, !waveasm.sreg<4>, !waveasm.pvreg<0>
  waveasm.s_endpgm
}

// -----------------------------------------------------------------------
// Test 3: VGPR extract into a VGPR pack (same principle, different type).
// -----------------------------------------------------------------------

// CHECK-LABEL: waveasm.program @vgpr_extract_into_pack
waveasm.program @vgpr_extract_into_pack target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %imm0 = waveasm.constant 0 : !waveasm.imm<0>
  %imm5 = waveasm.constant 5 : !waveasm.imm<5>

  %v0 = waveasm.v_mov_b32 %imm0 : !waveasm.imm<0> -> !waveasm.vreg
  %v1 = waveasm.v_mov_b32 %imm5 : !waveasm.imm<5> -> !waveasm.vreg
  %src = waveasm.pack %v0, %v1 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2>

  // Keep source alive.
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  waveasm.buffer_store_dwordx2 %src, %srd, %voff
      : !waveasm.vreg<2>, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>

  // Extract element 1 and repack with a new element.
  // CHECK: waveasm.extract {{.*}}[1]
  %ext = waveasm.extract %src[1] : !waveasm.vreg<2> -> !waveasm.vreg

  %v2 = waveasm.v_mov_b32 %imm0 : !waveasm.imm<0> -> !waveasm.vreg

  // A v_mov_b32 copy must appear for the extract input.
  // CHECK: waveasm.v_mov_b32 %{{.*}} : !waveasm.pvreg<{{[0-9]+}}> -> !waveasm.pvreg<{{[0-9]+}}>
  // CHECK-NEXT: waveasm.pack
  %dst = waveasm.pack %v2, %ext : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2>

  waveasm.buffer_store_dwordx2 %dst, %srd, %voff
      : !waveasm.vreg<2>, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>
  waveasm.s_endpgm
}
