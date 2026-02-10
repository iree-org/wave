// RUN: waveasm-translate --waveasm-memory-offset-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: DS read with constant add in address -> fold into offset:N
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_ds_read_const_add
waveasm.program @test_ds_read_const_add target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %base = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Address = base + 8192 via constant add
  %c8192 = waveasm.constant 8192 : !waveasm.imm<8192>
  %addr = waveasm.v_add_u32 %base, %c8192 : !waveasm.pvreg<0>, !waveasm.imm<8192> -> !waveasm.vreg

  // The constant 8192 should be folded into the ds_read offset field
  // CHECK: waveasm.ds_read_b128
  // CHECK-SAME: offset = 8192
  // CHECK-NOT: waveasm.v_add_u32
  %result = waveasm.ds_read_b128 %addr : !waveasm.vreg -> !waveasm.vreg<4, 4>

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: DS read with V_MOV_B32(constant) in address -> fold into offset:N
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_ds_read_mov_const
waveasm.program @test_ds_read_mov_const target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %base = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Address = base + V_MOV_B32(64)
  %c64 = waveasm.constant 64 : !waveasm.imm<64>
  %mov64 = waveasm.v_mov_b32 %c64 : !waveasm.imm<64> -> !waveasm.vreg
  %addr = waveasm.v_add_u32 %base, %mov64 : !waveasm.pvreg<0>, !waveasm.vreg -> !waveasm.vreg

  // The constant 64 should be folded into the ds_read offset field
  // CHECK: waveasm.ds_read_b128
  // CHECK-SAME: offset = 64
  %result = waveasm.ds_read_b128 %addr : !waveasm.vreg -> !waveasm.vreg<4, 4>

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: Buffer store with constant add in address -> fold into instOffset
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_buffer_store_const_add
waveasm.program @test_buffer_store_const_add target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %data = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %base = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // Address = base + 1024
  %c1024 = waveasm.constant 1024 : !waveasm.imm<1024>
  %addr = waveasm.v_add_u32 %base, %c1024 : !waveasm.pvreg<1>, !waveasm.imm<1024> -> !waveasm.vreg

  // The constant 1024 should be folded into the buffer_store instOffset
  // CHECK: waveasm.buffer_store_dword
  // CHECK-SAME: instOffset = 1024
  waveasm.buffer_store_dword %data, %srd, %addr : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.vreg

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: Shift distribution - V_LSHLREV_B32(N, V_ADD_U32(base, K))
// Should become V_LSHLREV_B32(N, base) with offset K << N
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_shift_distribution
waveasm.program @test_shift_distribution target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %data = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %base_row = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // base_row + 1
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %row_plus_1 = waveasm.v_add_u32 %base_row, %c1 : !waveasm.pvreg<1>, !waveasm.imm<1> -> !waveasm.vreg

  // (base_row + 1) << 10  = (base_row << 10) + (1 << 10) = (base_row << 10) + 1024
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %shifted = waveasm.v_lshlrev_b32 %c10, %row_plus_1 : !waveasm.imm<10>, !waveasm.vreg -> !waveasm.vreg

  // The shift should distribute: addr should use base_row shifted, with offset 1024
  // CHECK: waveasm.v_lshlrev_b32
  // CHECK: waveasm.buffer_store_dword
  // CHECK-SAME: instOffset = 1024
  waveasm.buffer_store_dword %data, %srd, %shifted : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.vreg

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: Buffer store offset exceeds limit -> NO fold
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_offset_limit
waveasm.program @test_offset_limit target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %data = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %base = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // Address = base + 8192 (exceeds buffer max of 4095)
  %c8192 = waveasm.constant 8192 : !waveasm.imm<8192>
  %addr = waveasm.v_add_u32 %base, %c8192 : !waveasm.pvreg<1>, !waveasm.imm<8192> -> !waveasm.vreg

  // Should NOT be folded because 8192 > 4095 (buffer max offset)
  // CHECK: waveasm.v_add_u32
  // CHECK: waveasm.buffer_store_dword
  // CHECK-NOT: instOffset = 8192
  waveasm.buffer_store_dword %data, %srd, %addr : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.vreg

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: DS read with large offset IS allowed (DS max = 65535)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_ds_large_offset
waveasm.program @test_ds_large_offset target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %base = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Address = base + 16384 (within DS max of 65535)
  %c16384 = waveasm.constant 16384 : !waveasm.imm<16384>
  %addr = waveasm.v_add_u32 %base, %c16384 : !waveasm.pvreg<0>, !waveasm.imm<16384> -> !waveasm.vreg

  // Should be folded - 16384 <= 65535
  // CHECK: waveasm.ds_read_b128
  // CHECK-SAME: offset = 16384
  %result = waveasm.ds_read_b128 %addr : !waveasm.vreg -> !waveasm.vreg<4, 4>

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: Shift + add combined pattern
// V_ADD_U32(V_LSHLREV_B32(10, V_ADD_U32(base, 1)), col_offset)
// -> V_ADD_U32(V_LSHLREV_B32(10, base), col_offset) with offset 1<<10=1024
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_shift_add_combined
waveasm.program @test_shift_add_combined target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %data = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %base_row = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %col_offset = waveasm.precolored.vreg 2 : !waveasm.pvreg<2>

  // row = base_row + 2
  %c2 = waveasm.constant 2 : !waveasm.imm<2>
  %row_plus_2 = waveasm.v_add_u32 %base_row, %c2 : !waveasm.pvreg<1>, !waveasm.imm<2> -> !waveasm.vreg

  // dimOffset0 = row << 10 = (base_row + 2) << 10
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %shifted = waveasm.v_lshlrev_b32 %c10, %row_plus_2 : !waveasm.imm<10>, !waveasm.vreg -> !waveasm.vreg

  // voffset = dimOffset0 + col_offset
  %voffset = waveasm.v_add_u32 %shifted, %col_offset : !waveasm.vreg, !waveasm.pvreg<2> -> !waveasm.vreg

  // Should fold: offset = 2 << 10 = 2048, addr = (base_row << 10) + col_offset
  // CHECK: waveasm.v_lshlrev_b32
  // CHECK: waveasm.v_add_u32
  // CHECK: waveasm.buffer_store_dword
  // CHECK-SAME: instOffset = 2048
  waveasm.buffer_store_dword %data, %srd, %voffset : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.vreg

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: Existing offset is accumulated (not replaced)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_existing_offset
waveasm.program @test_existing_offset target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %base = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Address = base + 100
  %c100 = waveasm.constant 100 : !waveasm.imm<100>
  %addr = waveasm.v_add_u32 %base, %c100 : !waveasm.pvreg<0>, !waveasm.imm<100> -> !waveasm.vreg

  // ds_read with existing offset of 200 -> total should be 200 + 100 = 300
  // CHECK: waveasm.ds_read_b32
  // CHECK-SAME: offset = 300
  %result = waveasm.ds_read_b32 %addr {offset = 200 : i64} : !waveasm.vreg -> !waveasm.vreg

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: V_OR_B32 with constant treated like V_ADD_U32 for shift distribution
// The affine handler uses OR when bit ranges don't overlap.
// V_LSHLREV_B32(10, V_OR_B32(base, 1)) -> V_LSHLREV_B32(10, base) + offset:1024
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_or_shift_distribution
waveasm.program @test_or_shift_distribution target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %data = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %base_row = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %col_offset = waveasm.precolored.vreg 2 : !waveasm.pvreg<2>

  // row_or_1 = base_row | 1 (non-overlapping bit ranges)
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %row_or_1 = waveasm.v_or_b32 %base_row, %c1 : !waveasm.pvreg<1>, !waveasm.imm<1> -> !waveasm.vreg

  // shifted = (base_row | 1) << 10
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %shifted = waveasm.v_lshlrev_b32 %c10, %row_or_1 : !waveasm.imm<10>, !waveasm.vreg -> !waveasm.vreg

  // voffset = shifted + col_offset
  %voffset = waveasm.v_add_u32 %shifted, %col_offset : !waveasm.vreg, !waveasm.pvreg<2> -> !waveasm.vreg

  // Should fold: offset = 1 << 10 = 1024
  // CHECK: waveasm.v_lshlrev_b32
  // CHECK: waveasm.v_add_u32
  // CHECK: waveasm.buffer_store_dword
  // CHECK-SAME: instOffset = 1024
  waveasm.buffer_store_dword %data, %srd, %voffset : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.vreg

  waveasm.s_endpgm
}
