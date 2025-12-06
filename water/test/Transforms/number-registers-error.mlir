// RUN: water-opt %s --pass-pipeline='builtin.module(func.func(water-number-registers))' --verify-diagnostics

func.func @test_dynamic_size_error(%n: index) {
  // expected-error @+1 {{Cannot allocate dynamic-sized memref in register space}}
  %reg = memref.alloca(%n) : memref<?xf32, 128 : i32>
  return
}
