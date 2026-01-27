// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test MFMA instruction variants - F16, BF16, I8, F32, F64

// Test 1: MFMA F16 variants
// CHECK-LABEL: waveasm.program @mfma_f16_variants
waveasm.program @mfma_f16_variants target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0, 2 : !waveasm.pvreg<0, 2>
  %b = waveasm.precolored.vreg 2, 2 : !waveasm.pvreg<2, 2>
  %c4 = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %c16 = waveasm.precolored.vreg 8, 16 : !waveasm.pvreg<8, 16>

  // CHECK: waveasm.v_mfma_f32_16x16x16_f16
  %v0 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %c4 : !waveasm.pvreg<0, 2>, !waveasm.pvreg<2, 2>, !waveasm.pvreg<4, 4> -> !waveasm.vreg<4, 4>

  // CHECK: waveasm.v_mfma_f32_32x32x8_f16
  %v1 = waveasm.v_mfma_f32_32x32x8_f16 %a, %b, %c16 : !waveasm.pvreg<0, 2>, !waveasm.pvreg<2, 2>, !waveasm.pvreg<8, 16> -> !waveasm.vreg<16, 4>

  // CHECK: waveasm.v_mfma_f32_4x4x4_f16
  %v2 = waveasm.v_mfma_f32_4x4x4_f16 %a, %b, %c4 : !waveasm.pvreg<0, 2>, !waveasm.pvreg<2, 2>, !waveasm.pvreg<4, 4> -> !waveasm.vreg<4, 4>

  waveasm.s_endpgm
}

// Test 2: MFMA BF16 variants
// CHECK-LABEL: waveasm.program @mfma_bf16_variants
waveasm.program @mfma_bf16_variants target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0, 2 : !waveasm.pvreg<0, 2>
  %b = waveasm.precolored.vreg 2, 2 : !waveasm.pvreg<2, 2>
  %c4 = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %c16 = waveasm.precolored.vreg 8, 16 : !waveasm.pvreg<8, 16>

  // CHECK: waveasm.v_mfma_f32_16x16x16_bf16
  %v0 = waveasm.v_mfma_f32_16x16x16_bf16 %a, %b, %c4 : !waveasm.pvreg<0, 2>, !waveasm.pvreg<2, 2>, !waveasm.pvreg<4, 4> -> !waveasm.vreg<4, 4>

  // CHECK: waveasm.v_mfma_f32_32x32x8_bf16
  %v1 = waveasm.v_mfma_f32_32x32x8_bf16 %a, %b, %c16 : !waveasm.pvreg<0, 2>, !waveasm.pvreg<2, 2>, !waveasm.pvreg<8, 16> -> !waveasm.vreg<16, 4>

  // CHECK: waveasm.v_mfma_f32_4x4x4_bf16
  %v2 = waveasm.v_mfma_f32_4x4x4_bf16 %a, %b, %c4 : !waveasm.pvreg<0, 2>, !waveasm.pvreg<2, 2>, !waveasm.pvreg<4, 4> -> !waveasm.vreg<4, 4>

  waveasm.s_endpgm
}

// Test 3: MFMA I8 variants
// CHECK-LABEL: waveasm.program @mfma_i8_variants
waveasm.program @mfma_i8_variants target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %b = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %c4 = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %c16 = waveasm.precolored.vreg 8, 16 : !waveasm.pvreg<8, 16>

  // CHECK: waveasm.v_mfma_i32_16x16x16_i8
  %v0 = waveasm.v_mfma_i32_16x16x16_i8 %a, %b, %c4 : !waveasm.pvreg<0>, !waveasm.pvreg<1>, !waveasm.pvreg<4, 4> -> !waveasm.vreg<4, 4>

  // CHECK: waveasm.v_mfma_i32_32x32x8_i8
  %v1 = waveasm.v_mfma_i32_32x32x8_i8 %a, %b, %c16 : !waveasm.pvreg<0>, !waveasm.pvreg<1>, !waveasm.pvreg<8, 16> -> !waveasm.vreg<16, 4>

  // CHECK: waveasm.v_mfma_i32_4x4x4_i8
  %v2 = waveasm.v_mfma_i32_4x4x4_i8 %a, %b, %c4 : !waveasm.pvreg<0>, !waveasm.pvreg<1>, !waveasm.pvreg<4, 4> -> !waveasm.vreg<4, 4>

  waveasm.s_endpgm
}

// Test 4: MFMA F32 variants
// CHECK-LABEL: waveasm.program @mfma_f32_variants
waveasm.program @mfma_f32_variants target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %b = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %c4 = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %c16 = waveasm.precolored.vreg 8, 16 : !waveasm.pvreg<8, 16>

  // CHECK: waveasm.v_mfma_f32_16x16x4_f32
  %v0 = waveasm.v_mfma_f32_16x16x4_f32 %a, %b, %c4 : !waveasm.pvreg<0>, !waveasm.pvreg<1>, !waveasm.pvreg<4, 4> -> !waveasm.vreg<4, 4>

  // CHECK: waveasm.v_mfma_f32_32x32x2_f32
  %v1 = waveasm.v_mfma_f32_32x32x2_f32 %a, %b, %c16 : !waveasm.pvreg<0>, !waveasm.pvreg<1>, !waveasm.pvreg<8, 16> -> !waveasm.vreg<16, 4>

  // CHECK: waveasm.v_mfma_f32_4x4x1_f32
  %v2 = waveasm.v_mfma_f32_4x4x1_f32 %a, %b, %c4 : !waveasm.pvreg<0>, !waveasm.pvreg<1>, !waveasm.pvreg<4, 4> -> !waveasm.vreg<4, 4>

  waveasm.s_endpgm
}

// Test 5: MFMA F64 variants
// CHECK-LABEL: waveasm.program @mfma_f64_variants
waveasm.program @mfma_f64_variants target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0, 2 : !waveasm.pvreg<0, 2>
  %b = waveasm.precolored.vreg 2, 2 : !waveasm.pvreg<2, 2>
  %c8 = waveasm.precolored.vreg 4, 8 : !waveasm.pvreg<4, 8>
  %c2 = waveasm.precolored.vreg 12, 2 : !waveasm.pvreg<12, 2>

  // CHECK: waveasm.v_mfma_f64_16x16x4_f64
  %v0 = waveasm.v_mfma_f64_16x16x4_f64 %a, %b, %c8 : !waveasm.pvreg<0, 2>, !waveasm.pvreg<2, 2>, !waveasm.pvreg<4, 8> -> !waveasm.vreg<8, 4>

  // CHECK: waveasm.v_mfma_f64_4x4x4_f64
  %v1 = waveasm.v_mfma_f64_4x4x4_f64 %a, %b, %c2 : !waveasm.pvreg<0, 2>, !waveasm.pvreg<2, 2>, !waveasm.pvreg<12, 2> -> !waveasm.vreg<2, 2>

  waveasm.s_endpgm
}

// Test 6: MFMA FP8 variants (gfx950+)
// CHECK-LABEL: waveasm.program @mfma_fp8_variants
waveasm.program @mfma_fp8_variants target = #waveasm.target<"gfx950", 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0, 2 : !waveasm.pvreg<0, 2>
  %b = waveasm.precolored.vreg 2, 2 : !waveasm.pvreg<2, 2>
  %c4 = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %c16 = waveasm.precolored.vreg 8, 16 : !waveasm.pvreg<8, 16>

  // CHECK: waveasm.v_mfma_f32_16x16x32_fp8_fp8
  %v0 = waveasm.v_mfma_f32_16x16x32_fp8_fp8 %a, %b, %c4 : !waveasm.pvreg<0, 2>, !waveasm.pvreg<2, 2>, !waveasm.pvreg<4, 4> -> !waveasm.vreg<4, 4>

  // CHECK: waveasm.v_mfma_f32_32x32x16_fp8_fp8
  %v1 = waveasm.v_mfma_f32_32x32x16_fp8_fp8 %a, %b, %c16 : !waveasm.pvreg<0, 2>, !waveasm.pvreg<2, 2>, !waveasm.pvreg<8, 16> -> !waveasm.vreg<16, 4>

  waveasm.s_endpgm
}
