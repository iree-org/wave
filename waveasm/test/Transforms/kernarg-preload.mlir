// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test kernel argument preloading attribute (MI350X / gfx95* optimization)

// Test 1: gfx950 with preloading enabled (3 pointers = 6 SGPRs)
// CHECK-LABEL: waveasm.program @gemm_preload
waveasm.program @gemm_preload
    target = #waveasm.target<#waveasm.gfx950, 5>
    abi = #waveasm.abi<>
    attributes {kernarg_preload_length = 6 : i64} {

  // With preloading enabled, kernel args are pre-loaded into SGPRs
  // s[2:3] = ptr A, s[4:5] = ptr B, s[6:7] = ptr C
  %ptrA = waveasm.precolored.sreg 2, 2 : !waveasm.psreg<2, 2>
  %ptrB = waveasm.precolored.sreg 4, 2 : !waveasm.psreg<4, 2>
  %ptrC = waveasm.precolored.sreg 6, 2 : !waveasm.psreg<6, 2>

  // Use the preloaded pointers
  %off = waveasm.constant 0 : !waveasm.imm<0>
  // CHECK: waveasm.s_load_dwordx4
  %srd = waveasm.s_load_dwordx4 %ptrA, %off : !waveasm.psreg<2, 2>, !waveasm.imm<0> -> !waveasm.sreg<4, 4>

  waveasm.s_endpgm
}

// Test 2: gfx942 without preloading (older target doesn't support it)
// CHECK-LABEL: waveasm.program @gemm_no_preload
waveasm.program @gemm_no_preload
    target = #waveasm.target<#waveasm.gfx942, 5>
    abi = #waveasm.abi<> {

  // Without preloading, kernel args must be loaded via s_load
  %kernarg_ptr = waveasm.precolored.sreg 0, 2 : !waveasm.psreg<0, 2>
  %off = waveasm.constant 0 : !waveasm.imm<0>

  // CHECK: waveasm.s_load_dwordx2
  %ptrA = waveasm.s_load_dwordx2 %kernarg_ptr, %off : !waveasm.psreg<0, 2>, !waveasm.imm<0> -> !waveasm.sreg<2>

  waveasm.s_endpgm
}

// Test 3: Maximum preload (8 pointers = 16 SGPRs, hardware limit)
// CHECK-LABEL: waveasm.program @max_preload
waveasm.program @max_preload
    target = #waveasm.target<#waveasm.gfx950, 5>
    abi = #waveasm.abi<>
    attributes {kernarg_preload_length = 16 : i64} {

  // 8 pointers preloaded: s[2:3] through s[16:17]
  %ptr0 = waveasm.precolored.sreg 2, 2 : !waveasm.psreg<2, 2>
  %ptr1 = waveasm.precolored.sreg 4, 2 : !waveasm.psreg<4, 2>
  %ptr2 = waveasm.precolored.sreg 6, 2 : !waveasm.psreg<6, 2>
  %ptr3 = waveasm.precolored.sreg 8, 2 : !waveasm.psreg<8, 2>
  %ptr4 = waveasm.precolored.sreg 10, 2 : !waveasm.psreg<10, 2>
  %ptr5 = waveasm.precolored.sreg 12, 2 : !waveasm.psreg<12, 2>
  %ptr6 = waveasm.precolored.sreg 14, 2 : !waveasm.psreg<14, 2>
  %ptr7 = waveasm.precolored.sreg 16, 2 : !waveasm.psreg<16, 2>

  waveasm.s_endpgm
}

// Test 4: Code object version check (preload only works with COv5+)
// CHECK-LABEL: waveasm.program @cov4_no_preload
waveasm.program @cov4_no_preload
    target = #waveasm.target<#waveasm.gfx950, 4>
    abi = #waveasm.abi<>
    attributes {kernarg_preload_length = 6 : i64} {

  // COv4 doesn't support preloading, so even if attribute is set,
  // the assembly emitter won't emit preload directives
  waveasm.s_endpgm
}
