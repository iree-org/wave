// RUN: waveasm-translate --target=gfx942 %s 2>&1 | FileCheck %s
//
// Test waitcnt insertion after memory operations

module {
  gpu.module @test_waitcnt {
    // CHECK: waveasm.program @waitcnt_kernel
    gpu.func @waitcnt_kernel(%arg0: memref<16x16xf32, #gpu.address_space<workgroup>>) kernel {
      %c0 = arith.constant 0 : index

      // Load from LDS - should generate ds_read
      // CHECK: waveasm.ds_read_b32
      %a = vector.load %arg0[%c0, %c0] : memref<16x16xf32, #gpu.address_space<workgroup>>, vector<1xf32>

      // Use the loaded value - should have waitcnt before use
      // Note: The waitcnt will be inserted by the waveasm-insert-waitcnt pass
      // For now, just verify the translation works
      // CHECK: waveasm.s_endpgm
      gpu.return
    }
  }
}
