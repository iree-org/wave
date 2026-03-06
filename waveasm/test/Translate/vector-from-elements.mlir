// RUN: waveasm-translate --target=gfx942 %s 2>&1 | FileCheck %s
//
// Test: vector.from_elements handler packs scalar elements into VGPR dwords.

module {
  gpu.module @test_from_elements {
    // CHECK-LABEL: waveasm.program @from_elements_i32
    gpu.func @from_elements_i32(%a: i32, %b: i32) kernel {
      // Two i32 elements -> 2-dword pack
      // CHECK: waveasm.pack
      %v = vector.from_elements %a, %b : vector<2xi32>
      gpu.return
    }
  }
}

module {
  gpu.module @test_from_elements_f16 {
    // CHECK-LABEL: waveasm.program @from_elements_f16_pair
    gpu.func @from_elements_f16_pair(%a: f16, %b: f16) kernel {
      // Two f16 elements -> 1-dword pack via shift+or
      // CHECK: waveasm.v_lshlrev_b32
      // CHECK: waveasm.v_or_b32
      %v = vector.from_elements %a, %b : vector<2xf16>
      gpu.return
    }
  }
}
