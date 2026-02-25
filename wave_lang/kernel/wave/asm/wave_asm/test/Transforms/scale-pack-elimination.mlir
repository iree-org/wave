// RUN: waveasm-translate --waveasm-scale-pack-elimination %s 2>&1 | FileCheck %s
//
// Tests for the ScalePackElimination pass.
//
// The pass finds BFE(dword,{0,8,16,24},8) -> iter_arg -> LSHL_OR repack
// chains and replaces them with a single dword iter_arg, eliminating
// 3 LSHL_OR + 4 BFE = 7 VALU per chain per iteration.

//===----------------------------------------------------------------------===//
// Basic: one pack chain, loop carries 4 bytes -> 1 dword.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @basic_scale_pack
waveasm.program @basic_scale_pack
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %eight = waveasm.constant 8 : !waveasm.imm<8>
  %sixteen = waveasm.constant 16 : !waveasm.imm<16>
  %twentyfour = waveasm.constant 24 : !waveasm.imm<24>
  %limit = waveasm.constant 4 : !waveasm.imm<4>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %soff = waveasm.precolored.sreg 4 : !waveasm.psreg<4>
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

  // Pre-loop: load a dword, extract 4 bytes.
  %init_dword = waveasm.buffer_load_dword %srd, %tid, %soff
      : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.psreg<4> -> !waveasm.vreg
  %b0 = waveasm.v_bfe_u32 %init_dword, %zero, %eight : !waveasm.vreg, !waveasm.imm<0>, !waveasm.imm<8> -> !waveasm.vreg
  %b1 = waveasm.v_bfe_u32 %init_dword, %eight, %eight : !waveasm.vreg, !waveasm.imm<8>, !waveasm.imm<8> -> !waveasm.vreg
  %b2 = waveasm.v_bfe_u32 %init_dword, %sixteen, %eight : !waveasm.vreg, !waveasm.imm<16>, !waveasm.imm<8> -> !waveasm.vreg
  %b3 = waveasm.v_bfe_u32 %init_dword, %twentyfour, %eight : !waveasm.vreg, !waveasm.imm<24>, !waveasm.imm<8> -> !waveasm.vreg

  // After elimination: loop should carry (iv, dword) instead of (iv, b0, b1, b2, b3).
  // CHECK: waveasm.loop
  // CHECK-SAME: -> (!waveasm.sreg, !waveasm.vreg) {
  %riv, %rb0, %rb1, %rb2, %rb3 = waveasm.loop(
      %iv = %init_iv, %arg_b0 = %b0, %arg_b1 = %b1, %arg_b2 = %b2, %arg_b3 = %b3)
      : (!waveasm.sreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg)
     -> (!waveasm.sreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg) {

    // Repack: lshl_or chain. Should be eliminated.
    // CHECK-NOT: waveasm.v_lshl_or_b32
    %t1 = waveasm.v_lshl_or_b32 %arg_b1, %eight, %arg_b0
        : !waveasm.vreg, !waveasm.imm<8>, !waveasm.vreg -> !waveasm.vreg
    %t2 = waveasm.v_lshl_or_b32 %arg_b2, %sixteen, %t1
        : !waveasm.vreg, !waveasm.imm<16>, !waveasm.vreg -> !waveasm.vreg
    %packed = waveasm.v_lshl_or_b32 %arg_b3, %twentyfour, %t2
        : !waveasm.vreg, !waveasm.imm<24>, !waveasm.vreg -> !waveasm.vreg

    // Use the packed value (stands in for v_mfma_scale).
    %use = waveasm.v_add_u32 %packed, %tid : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

    // Bottom: load next dword, extract bytes for yield.
    // After elimination: the BFE ops should be removed (single-use for yield).
    // CHECK-NOT: waveasm.v_bfe_u32
    %next_dword = waveasm.buffer_load_dword %srd, %tid, %soff
        : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.psreg<4> -> !waveasm.vreg
    %nb0 = waveasm.v_bfe_u32 %next_dword, %zero, %eight : !waveasm.vreg, !waveasm.imm<0>, !waveasm.imm<8> -> !waveasm.vreg
    %nb1 = waveasm.v_bfe_u32 %next_dword, %eight, %eight : !waveasm.vreg, !waveasm.imm<8>, !waveasm.imm<8> -> !waveasm.vreg
    %nb2 = waveasm.v_bfe_u32 %next_dword, %sixteen, %eight : !waveasm.vreg, !waveasm.imm<16>, !waveasm.imm<8> -> !waveasm.vreg
    %nb3 = waveasm.v_bfe_u32 %next_dword, %twentyfour, %eight : !waveasm.vreg, !waveasm.imm<24>, !waveasm.imm<8> -> !waveasm.vreg

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<4> -> !waveasm.sreg

    // Yield should carry (next_iv, next_dword) instead of (next_iv, nb0..nb3).
    // CHECK: waveasm.condition
    // CHECK-SAME: iter_args(%{{.*}}, %{{.*}}) : !waveasm.sreg, !waveasm.vreg
    waveasm.condition %cond : !waveasm.sreg
        iter_args(%next_iv, %nb0, %nb1, %nb2, %nb3)
        : !waveasm.sreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Negative: byte args used outside the pack chain -> no transformation.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @no_transform_extra_use
waveasm.program @no_transform_extra_use
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %eight = waveasm.constant 8 : !waveasm.imm<8>
  %sixteen = waveasm.constant 16 : !waveasm.imm<16>
  %twentyfour = waveasm.constant 24 : !waveasm.imm<24>
  %limit = waveasm.constant 4 : !waveasm.imm<4>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %soff = waveasm.precolored.sreg 4 : !waveasm.psreg<4>
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

  %init_dword = waveasm.buffer_load_dword %srd, %tid, %soff
      : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.psreg<4> -> !waveasm.vreg
  %b0 = waveasm.v_bfe_u32 %init_dword, %zero, %eight : !waveasm.vreg, !waveasm.imm<0>, !waveasm.imm<8> -> !waveasm.vreg
  %b1 = waveasm.v_bfe_u32 %init_dword, %eight, %eight : !waveasm.vreg, !waveasm.imm<8>, !waveasm.imm<8> -> !waveasm.vreg
  %b2 = waveasm.v_bfe_u32 %init_dword, %sixteen, %eight : !waveasm.vreg, !waveasm.imm<16>, !waveasm.imm<8> -> !waveasm.vreg
  %b3 = waveasm.v_bfe_u32 %init_dword, %twentyfour, %eight : !waveasm.vreg, !waveasm.imm<24>, !waveasm.imm<8> -> !waveasm.vreg

  // Loop should remain unchanged because %arg_b0 has an extra use.
  // CHECK: waveasm.loop
  // CHECK-SAME: -> (!waveasm.sreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg) {
  %riv, %rb0, %rb1, %rb2, %rb3 = waveasm.loop(
      %iv = %init_iv, %arg_b0 = %b0, %arg_b1 = %b1, %arg_b2 = %b2, %arg_b3 = %b3)
      : (!waveasm.sreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg)
     -> (!waveasm.sreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg) {

    %t1 = waveasm.v_lshl_or_b32 %arg_b1, %eight, %arg_b0
        : !waveasm.vreg, !waveasm.imm<8>, !waveasm.vreg -> !waveasm.vreg
    %t2 = waveasm.v_lshl_or_b32 %arg_b2, %sixteen, %t1
        : !waveasm.vreg, !waveasm.imm<16>, !waveasm.vreg -> !waveasm.vreg
    %packed = waveasm.v_lshl_or_b32 %arg_b3, %twentyfour, %t2
        : !waveasm.vreg, !waveasm.imm<24>, !waveasm.vreg -> !waveasm.vreg

    // Extra use of %arg_b0 outside the pack chain -> blocks transformation.
    // CHECK: waveasm.v_lshl_or_b32
    %extra = waveasm.v_add_u32 %arg_b0, %tid : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

    %next_dword = waveasm.buffer_load_dword %srd, %tid, %soff
        : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.psreg<4> -> !waveasm.vreg
    %nb0 = waveasm.v_bfe_u32 %next_dword, %zero, %eight : !waveasm.vreg, !waveasm.imm<0>, !waveasm.imm<8> -> !waveasm.vreg
    %nb1 = waveasm.v_bfe_u32 %next_dword, %eight, %eight : !waveasm.vreg, !waveasm.imm<8>, !waveasm.imm<8> -> !waveasm.vreg
    %nb2 = waveasm.v_bfe_u32 %next_dword, %sixteen, %eight : !waveasm.vreg, !waveasm.imm<16>, !waveasm.imm<8> -> !waveasm.vreg
    %nb3 = waveasm.v_bfe_u32 %next_dword, %twentyfour, %eight : !waveasm.vreg, !waveasm.imm<24>, !waveasm.imm<8> -> !waveasm.vreg

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<4> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg
        iter_args(%next_iv, %nb0, %nb1, %nb2, %nb3)
        : !waveasm.sreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Negative: yield BFE sources differ -> no transformation.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @no_transform_different_yield_source
waveasm.program @no_transform_different_yield_source
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %eight = waveasm.constant 8 : !waveasm.imm<8>
  %sixteen = waveasm.constant 16 : !waveasm.imm<16>
  %twentyfour = waveasm.constant 24 : !waveasm.imm<24>
  %limit = waveasm.constant 4 : !waveasm.imm<4>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %soff = waveasm.precolored.sreg 4 : !waveasm.psreg<4>
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

  %init_dword = waveasm.buffer_load_dword %srd, %tid, %soff
      : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.psreg<4> -> !waveasm.vreg
  %b0 = waveasm.v_bfe_u32 %init_dword, %zero, %eight : !waveasm.vreg, !waveasm.imm<0>, !waveasm.imm<8> -> !waveasm.vreg
  %b1 = waveasm.v_bfe_u32 %init_dword, %eight, %eight : !waveasm.vreg, !waveasm.imm<8>, !waveasm.imm<8> -> !waveasm.vreg
  %b2 = waveasm.v_bfe_u32 %init_dword, %sixteen, %eight : !waveasm.vreg, !waveasm.imm<16>, !waveasm.imm<8> -> !waveasm.vreg
  %b3 = waveasm.v_bfe_u32 %init_dword, %twentyfour, %eight : !waveasm.vreg, !waveasm.imm<24>, !waveasm.imm<8> -> !waveasm.vreg

  // CHECK: waveasm.loop
  // CHECK-SAME: -> (!waveasm.sreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg) {
  %riv, %rb0, %rb1, %rb2, %rb3 = waveasm.loop(
      %iv = %init_iv, %arg_b0 = %b0, %arg_b1 = %b1, %arg_b2 = %b2, %arg_b3 = %b3)
      : (!waveasm.sreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg)
     -> (!waveasm.sreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg) {

    %t1 = waveasm.v_lshl_or_b32 %arg_b1, %eight, %arg_b0
        : !waveasm.vreg, !waveasm.imm<8>, !waveasm.vreg -> !waveasm.vreg
    %t2 = waveasm.v_lshl_or_b32 %arg_b2, %sixteen, %t1
        : !waveasm.vreg, !waveasm.imm<16>, !waveasm.vreg -> !waveasm.vreg
    %packed = waveasm.v_lshl_or_b32 %arg_b3, %twentyfour, %t2
        : !waveasm.vreg, !waveasm.imm<24>, !waveasm.vreg -> !waveasm.vreg

    // Yield BFE bytes from TWO different dwords -> can't merge.
    %dword_a = waveasm.buffer_load_dword %srd, %tid, %soff
        : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.psreg<4> -> !waveasm.vreg
    %dword_b = waveasm.buffer_load_dword %srd, %tid, %soff
        : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.psreg<4> -> !waveasm.vreg
    // byte0,1 from dword_a; byte2,3 from dword_b.
    %nb0 = waveasm.v_bfe_u32 %dword_a, %zero, %eight : !waveasm.vreg, !waveasm.imm<0>, !waveasm.imm<8> -> !waveasm.vreg
    %nb1 = waveasm.v_bfe_u32 %dword_a, %eight, %eight : !waveasm.vreg, !waveasm.imm<8>, !waveasm.imm<8> -> !waveasm.vreg
    %nb2 = waveasm.v_bfe_u32 %dword_b, %sixteen, %eight : !waveasm.vreg, !waveasm.imm<16>, !waveasm.imm<8> -> !waveasm.vreg
    %nb3 = waveasm.v_bfe_u32 %dword_b, %twentyfour, %eight : !waveasm.vreg, !waveasm.imm<24>, !waveasm.imm<8> -> !waveasm.vreg

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<4> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg
        iter_args(%next_iv, %nb0, %nb1, %nb2, %nb3)
        : !waveasm.sreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Two independent pack chains in the same loop.
// Both should be eliminated, collapsing 8 byte iter_args into 2 dwords.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @two_chains
waveasm.program @two_chains
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %eight = waveasm.constant 8 : !waveasm.imm<8>
  %sixteen = waveasm.constant 16 : !waveasm.imm<16>
  %twentyfour = waveasm.constant 24 : !waveasm.imm<24>
  %limit = waveasm.constant 4 : !waveasm.imm<4>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %soff = waveasm.precolored.sreg 4 : !waveasm.psreg<4>
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

  // Chain A: first scale dword.
  %dword_a = waveasm.buffer_load_dword %srd, %tid, %soff
      : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.psreg<4> -> !waveasm.vreg
  %a0 = waveasm.v_bfe_u32 %dword_a, %zero, %eight : !waveasm.vreg, !waveasm.imm<0>, !waveasm.imm<8> -> !waveasm.vreg
  %a1 = waveasm.v_bfe_u32 %dword_a, %eight, %eight : !waveasm.vreg, !waveasm.imm<8>, !waveasm.imm<8> -> !waveasm.vreg
  %a2 = waveasm.v_bfe_u32 %dword_a, %sixteen, %eight : !waveasm.vreg, !waveasm.imm<16>, !waveasm.imm<8> -> !waveasm.vreg
  %a3 = waveasm.v_bfe_u32 %dword_a, %twentyfour, %eight : !waveasm.vreg, !waveasm.imm<24>, !waveasm.imm<8> -> !waveasm.vreg

  // Chain B: second scale dword.
  %dword_b = waveasm.buffer_load_dword %srd, %tid, %soff
      : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.psreg<4> -> !waveasm.vreg
  %b0 = waveasm.v_bfe_u32 %dword_b, %zero, %eight : !waveasm.vreg, !waveasm.imm<0>, !waveasm.imm<8> -> !waveasm.vreg
  %b1 = waveasm.v_bfe_u32 %dword_b, %eight, %eight : !waveasm.vreg, !waveasm.imm<8>, !waveasm.imm<8> -> !waveasm.vreg
  %b2 = waveasm.v_bfe_u32 %dword_b, %sixteen, %eight : !waveasm.vreg, !waveasm.imm<16>, !waveasm.imm<8> -> !waveasm.vreg
  %b3 = waveasm.v_bfe_u32 %dword_b, %twentyfour, %eight : !waveasm.vreg, !waveasm.imm<24>, !waveasm.imm<8> -> !waveasm.vreg

  // After elimination: (iv, dword_a, dword_b) = 3 iter_args instead of 9.
  // CHECK: waveasm.loop
  // CHECK-SAME: -> (!waveasm.sreg, !waveasm.vreg, !waveasm.vreg) {
  %riv, %ra0, %ra1, %ra2, %ra3, %rb0, %rb1, %rb2, %rb3 = waveasm.loop(
      %iv = %init_iv,
      %arg_a0 = %a0, %arg_a1 = %a1, %arg_a2 = %a2, %arg_a3 = %a3,
      %arg_b0 = %b0, %arg_b1 = %b1, %arg_b2 = %b2, %arg_b3 = %b3)
      : (!waveasm.sreg,
         !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg,
         !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg)
     -> (!waveasm.sreg,
         !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg,
         !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg) {

    // Pack chain A.
    // CHECK-NOT: waveasm.v_lshl_or_b32
    %at1 = waveasm.v_lshl_or_b32 %arg_a1, %eight, %arg_a0
        : !waveasm.vreg, !waveasm.imm<8>, !waveasm.vreg -> !waveasm.vreg
    %at2 = waveasm.v_lshl_or_b32 %arg_a2, %sixteen, %at1
        : !waveasm.vreg, !waveasm.imm<16>, !waveasm.vreg -> !waveasm.vreg
    %packed_a = waveasm.v_lshl_or_b32 %arg_a3, %twentyfour, %at2
        : !waveasm.vreg, !waveasm.imm<24>, !waveasm.vreg -> !waveasm.vreg

    // Pack chain B.
    %bt1 = waveasm.v_lshl_or_b32 %arg_b1, %eight, %arg_b0
        : !waveasm.vreg, !waveasm.imm<8>, !waveasm.vreg -> !waveasm.vreg
    %bt2 = waveasm.v_lshl_or_b32 %arg_b2, %sixteen, %bt1
        : !waveasm.vreg, !waveasm.imm<16>, !waveasm.vreg -> !waveasm.vreg
    %packed_b = waveasm.v_lshl_or_b32 %arg_b3, %twentyfour, %bt2
        : !waveasm.vreg, !waveasm.imm<24>, !waveasm.vreg -> !waveasm.vreg

    // Use both packed values.
    %use_a = waveasm.v_add_u32 %packed_a, %tid : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg
    %use_b = waveasm.v_add_u32 %packed_b, %tid : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

    // Bottom: reload both dwords, extract bytes for yield.
    // CHECK-NOT: waveasm.v_bfe_u32
    %next_dword_a = waveasm.buffer_load_dword %srd, %tid, %soff
        : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.psreg<4> -> !waveasm.vreg
    %na0 = waveasm.v_bfe_u32 %next_dword_a, %zero, %eight : !waveasm.vreg, !waveasm.imm<0>, !waveasm.imm<8> -> !waveasm.vreg
    %na1 = waveasm.v_bfe_u32 %next_dword_a, %eight, %eight : !waveasm.vreg, !waveasm.imm<8>, !waveasm.imm<8> -> !waveasm.vreg
    %na2 = waveasm.v_bfe_u32 %next_dword_a, %sixteen, %eight : !waveasm.vreg, !waveasm.imm<16>, !waveasm.imm<8> -> !waveasm.vreg
    %na3 = waveasm.v_bfe_u32 %next_dword_a, %twentyfour, %eight : !waveasm.vreg, !waveasm.imm<24>, !waveasm.imm<8> -> !waveasm.vreg

    %next_dword_b = waveasm.buffer_load_dword %srd, %tid, %soff
        : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.psreg<4> -> !waveasm.vreg
    %nb0 = waveasm.v_bfe_u32 %next_dword_b, %zero, %eight : !waveasm.vreg, !waveasm.imm<0>, !waveasm.imm<8> -> !waveasm.vreg
    %nb1 = waveasm.v_bfe_u32 %next_dword_b, %eight, %eight : !waveasm.vreg, !waveasm.imm<8>, !waveasm.imm<8> -> !waveasm.vreg
    %nb2 = waveasm.v_bfe_u32 %next_dword_b, %sixteen, %eight : !waveasm.vreg, !waveasm.imm<16>, !waveasm.imm<8> -> !waveasm.vreg
    %nb3 = waveasm.v_bfe_u32 %next_dword_b, %twentyfour, %eight : !waveasm.vreg, !waveasm.imm<24>, !waveasm.imm<8> -> !waveasm.vreg

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<4> -> !waveasm.sreg

    // Yield should carry (next_iv, next_dword_a, next_dword_b).
    // CHECK: waveasm.condition
    // CHECK-SAME: iter_args(%{{.*}}, %{{.*}}, %{{.*}}) : !waveasm.sreg, !waveasm.vreg, !waveasm.vreg
    waveasm.condition %cond : !waveasm.sreg
        iter_args(%next_iv, %na0, %na1, %na2, %na3, %nb0, %nb1, %nb2, %nb3)
        : !waveasm.sreg,
          !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg,
          !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Post-loop: multiple byte uses from the same chain (bytes 0, 1, 3).
// Each should get its own v_bfe_u32 extraction after the loop.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @post_loop_multi_byte_use
waveasm.program @post_loop_multi_byte_use
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %eight = waveasm.constant 8 : !waveasm.imm<8>
  %sixteen = waveasm.constant 16 : !waveasm.imm<16>
  %twentyfour = waveasm.constant 24 : !waveasm.imm<24>
  %limit = waveasm.constant 4 : !waveasm.imm<4>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %soff = waveasm.precolored.sreg 4 : !waveasm.psreg<4>
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

  %init_dword = waveasm.buffer_load_dword %srd, %tid, %soff
      : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.psreg<4> -> !waveasm.vreg
  %b0 = waveasm.v_bfe_u32 %init_dword, %zero, %eight : !waveasm.vreg, !waveasm.imm<0>, !waveasm.imm<8> -> !waveasm.vreg
  %b1 = waveasm.v_bfe_u32 %init_dword, %eight, %eight : !waveasm.vreg, !waveasm.imm<8>, !waveasm.imm<8> -> !waveasm.vreg
  %b2 = waveasm.v_bfe_u32 %init_dword, %sixteen, %eight : !waveasm.vreg, !waveasm.imm<16>, !waveasm.imm<8> -> !waveasm.vreg
  %b3 = waveasm.v_bfe_u32 %init_dword, %twentyfour, %eight : !waveasm.vreg, !waveasm.imm<24>, !waveasm.imm<8> -> !waveasm.vreg

  // Loop still gets transformed.
  // CHECK: waveasm.loop
  // CHECK-SAME: -> (!waveasm.sreg, !waveasm.vreg) {
  %riv, %rb0, %rb1, %rb2, %rb3 = waveasm.loop(
      %iv = %init_iv, %arg_b0 = %b0, %arg_b1 = %b1, %arg_b2 = %b2, %arg_b3 = %b3)
      : (!waveasm.sreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg)
     -> (!waveasm.sreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg) {

    %t1 = waveasm.v_lshl_or_b32 %arg_b1, %eight, %arg_b0
        : !waveasm.vreg, !waveasm.imm<8>, !waveasm.vreg -> !waveasm.vreg
    %t2 = waveasm.v_lshl_or_b32 %arg_b2, %sixteen, %t1
        : !waveasm.vreg, !waveasm.imm<16>, !waveasm.vreg -> !waveasm.vreg
    %packed = waveasm.v_lshl_or_b32 %arg_b3, %twentyfour, %t2
        : !waveasm.vreg, !waveasm.imm<24>, !waveasm.vreg -> !waveasm.vreg

    %use = waveasm.v_add_u32 %packed, %tid : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

    %next_dword = waveasm.buffer_load_dword %srd, %tid, %soff
        : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.psreg<4> -> !waveasm.vreg
    %nb0 = waveasm.v_bfe_u32 %next_dword, %zero, %eight : !waveasm.vreg, !waveasm.imm<0>, !waveasm.imm<8> -> !waveasm.vreg
    %nb1 = waveasm.v_bfe_u32 %next_dword, %eight, %eight : !waveasm.vreg, !waveasm.imm<8>, !waveasm.imm<8> -> !waveasm.vreg
    %nb2 = waveasm.v_bfe_u32 %next_dword, %sixteen, %eight : !waveasm.vreg, !waveasm.imm<16>, !waveasm.imm<8> -> !waveasm.vreg
    %nb3 = waveasm.v_bfe_u32 %next_dword, %twentyfour, %eight : !waveasm.vreg, !waveasm.imm<24>, !waveasm.imm<8> -> !waveasm.vreg

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<4> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg
        iter_args(%next_iv, %nb0, %nb1, %nb2, %nb3)
        : !waveasm.sreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg
  }

  // Post-loop uses of bytes 0, 1, and 3. Each gets a v_bfe_u32 extraction.
  // Byte 0: v_bfe_u32(dword, 0, 8).
  // CHECK: waveasm.v_bfe_u32 %{{.*}}, %{{.*}}, %{{.*}} : !waveasm.vreg, !waveasm.imm<0>, !waveasm.imm<8>
  // Byte 1: v_bfe_u32(dword, 8, 8).
  // CHECK: waveasm.v_bfe_u32 %{{.*}}, %{{.*}}, %{{.*}} : !waveasm.vreg, !waveasm.imm<8>, !waveasm.imm<8>
  // Byte 3: v_bfe_u32(dword, 24, 8).
  // CHECK: waveasm.v_bfe_u32 %{{.*}}, %{{.*}}, %{{.*}} : !waveasm.vreg, !waveasm.imm<24>, !waveasm.imm<8>
  %post0 = waveasm.v_add_u32 %rb0, %tid : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg
  %post1 = waveasm.v_add_u32 %rb1, %tid : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg
  %post3 = waveasm.v_add_u32 %rb3, %tid : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

  waveasm.s_endpgm
}
