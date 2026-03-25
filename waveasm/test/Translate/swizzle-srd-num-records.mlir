// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: fat_raw_buffer_cast with cacheSwizzleStride but no pending SRD base
// adjustment (no reinterpret_cast). This exercises the swizzle SRD construction
// path in handleFatRawBufferCast, which copies num_records from the source SRD
// rather than hardcoding a value. This ensures OOB sentinel addresses
// (0x7FFFFFFF) remain out-of-bounds, causing the hardware to return zero
// instead of faulting.

// CHECK-LABEL: waveasm.program @swizzle_srd_copies_num_records

module {
  gpu.module @test_swizzle_srd {
    gpu.func @swizzle_srd_copies_num_records(
        %src: memref<?xf16>,
        %tidx: index {llvm.mlir.workitem_id_x}
    ) kernel {
      %c0 = arith.constant 0 : index
      %valid = arith.constant 2147483645 : i64
      %stride = arith.constant 128 : i14

      %lds = memref.alloc() : memref<2048xi8, 3>
      %view = memref.view %lds[%c0][]
          : memref<2048xi8, 3> to memref<64x8xf16, 3>

      // fat_raw_buffer_cast with cacheSwizzleStride, no reinterpret_cast.
      // The swizzle SRD must copy num_records from the source SRD (s_mov_b32
      // sN, sM) rather than using 0xFFFFFFFF which would make sentinel offsets
      // in-bounds.
      //
      // Swizzle SRD construction via typed ops + PackOp:
      //   extract source SRD words
      //   s_mov_b32 (copy base lo)
      //   s_and_b32 + s_or_b32 (set swizzle bits in base hi)
      //   s_mov_b32 (copy num_records from source -- NOT a constant)
      //   s_mov_b32 (flags constant)
      //   pack into 4-wide SGPR
      //
      // CHECK: waveasm.extract
      // CHECK: waveasm.s_mov_b32
      // CHECK: waveasm.s_and_b32
      // CHECK: waveasm.s_or_b32
      // CHECK: waveasm.s_mov_b32 %{{.*}} : !waveasm.sreg -> !waveasm.sreg
      // CHECK: waveasm.s_mov_b32 %{{.*}} : !waveasm.imm
      // CHECK: waveasm.pack
      %buf = amdgpu.fat_raw_buffer_cast %src
          validBytes(%valid) cacheSwizzleStride(%stride) resetOffset
          : memref<?xf16>
            to memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>

      // Use the swizzled buffer in a gather_to_lds.
      // CHECK: waveasm.buffer_load_dword_lds
      "amdgpu.gather_to_lds"(%buf, %tidx, %view, %c0, %c0)
          <{operandSegmentSizes = array<i32: 1, 1, 1, 2>,
            transferType = vector<2xf16>}>
          : (memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>,
             index, memref<64x8xf16, 3>, index, index) -> ()

      gpu.return
    }
  }
}
