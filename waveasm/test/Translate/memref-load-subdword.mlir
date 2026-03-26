// RUN: waveasm-translate --target=gfx942 %s | FileCheck %s
//
// Test: memref.load handler emits subdword buffer loads (UBYTE, USHORT)
// for sub-32-bit element types.

module {
  gpu.module @test_subdword_load {
    // CHECK-LABEL: waveasm.program @load_i8
    gpu.func @load_i8(%buf: memref<64xi8>) kernel {
      %c0 = arith.constant 0 : index
      // 8-bit load -> buffer_load_ubyte
      // CHECK: waveasm.buffer_load_ubyte
      %val = memref.load %buf[%c0] : memref<64xi8>
      gpu.return
    }
  }
}

module {
  gpu.module @test_short_load {
    // CHECK-LABEL: waveasm.program @load_i16
    gpu.func @load_i16(%buf: memref<64xi16>) kernel {
      %c0 = arith.constant 0 : index
      // 16-bit load -> buffer_load_ushort
      // CHECK: waveasm.buffer_load_ushort
      %val = memref.load %buf[%c0] : memref<64xi16>
      gpu.return
    }
  }
}

module {
  gpu.module @test_dword_load {
    // CHECK-LABEL: waveasm.program @load_i32
    gpu.func @load_i32(%buf: memref<64xi32>) kernel {
      %c0 = arith.constant 0 : index
      // 32-bit load -> buffer_load_dword
      // CHECK: waveasm.buffer_load_dword
      %val = memref.load %buf[%c0] : memref<64xi32>
      gpu.return
    }
  }
}
