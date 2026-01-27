// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test floating-point arithmetic operations

// Test 1: Basic FP arithmetic operations
// CHECK-LABEL: waveasm.program @fp_arith_basic
waveasm.program @fp_arith_basic target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %b = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // CHECK: waveasm.v_add_f32
  %v0 = waveasm.v_add_f32 %a, %b : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // CHECK: waveasm.v_sub_f32
  %v1 = waveasm.v_sub_f32 %a, %b : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // CHECK: waveasm.v_mul_f32
  %v2 = waveasm.v_mul_f32 %a, %b : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 2: FMA operations
// CHECK-LABEL: waveasm.program @fp_fma
waveasm.program @fp_fma target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %b = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %c = waveasm.precolored.vreg 2 : !waveasm.pvreg<2>

  // CHECK: waveasm.v_fma_f32
  %v0 = waveasm.v_fma_f32 %a, %b, %c : !waveasm.pvreg<0>, !waveasm.pvreg<1>, !waveasm.pvreg<2> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 3: FP comparison operations
// CHECK-LABEL: waveasm.program @fp_cmp
waveasm.program @fp_cmp target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %b = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // CHECK: waveasm.v_cmp_eq_f32
  waveasm.v_cmp_eq_f32 %a, %b : !waveasm.pvreg<0>, !waveasm.pvreg<1>

  // CHECK: waveasm.v_cmp_lt_f32
  waveasm.v_cmp_lt_f32 %a, %b : !waveasm.pvreg<0>, !waveasm.pvreg<1>

  // CHECK: waveasm.v_cmp_gt_f32
  waveasm.v_cmp_gt_f32 %a, %b : !waveasm.pvreg<0>, !waveasm.pvreg<1>

  waveasm.s_endpgm
}

// Test 4: FP conversions
// CHECK-LABEL: waveasm.program @fp_cvt
waveasm.program @fp_cvt target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // CHECK: waveasm.v_cvt_f32_f16
  %v0 = waveasm.v_cvt_f32_f16 %a : !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: waveasm.v_cvt_f16_f32
  %v1 = waveasm.v_cvt_f16_f32 %a : !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: waveasm.v_cvt_f32_i32
  %v2 = waveasm.v_cvt_f32_i32 %a : !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: waveasm.v_cvt_i32_f32
  %v3 = waveasm.v_cvt_i32_f32 %a : !waveasm.pvreg<0> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 5: FP special functions
// CHECK-LABEL: waveasm.program @fp_special
waveasm.program @fp_special target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // CHECK: waveasm.v_rcp_f32
  %v0 = waveasm.v_rcp_f32 %a : !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: waveasm.v_rsq_f32
  %v1 = waveasm.v_rsq_f32 %a : !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: waveasm.v_sqrt_f32
  %v2 = waveasm.v_sqrt_f32 %a : !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: waveasm.v_floor_f32
  %v3 = waveasm.v_floor_f32 %a : !waveasm.pvreg<0> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 6: MAC/MAD operations
// CHECK-LABEL: waveasm.program @fp_mac
waveasm.program @fp_mac target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %b = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %c = waveasm.precolored.vreg 2 : !waveasm.pvreg<2>

  // CHECK: waveasm.v_mad_f32
  %v0 = waveasm.v_mad_f32 %a, %b, %c : !waveasm.pvreg<0>, !waveasm.pvreg<1>, !waveasm.pvreg<2> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 7: FP min/max operations
// CHECK-LABEL: waveasm.program @fp_minmax
waveasm.program @fp_minmax target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %b = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // CHECK: waveasm.v_min_f32
  %v0 = waveasm.v_min_f32 %a, %b : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // CHECK: waveasm.v_max_f32
  %v1 = waveasm.v_max_f32 %a, %b : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  waveasm.s_endpgm
}
