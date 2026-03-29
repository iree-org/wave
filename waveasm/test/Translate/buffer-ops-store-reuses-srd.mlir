// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: vector.store reuses the PackOp-produced SRD from a prior vector.load
// on the same adjusted memref. Before the fix, handleVectorStore had an inline
// SRD lookup that checked getSRDIndex but not getSRDValue, so the store fell
// back to the unadjusted prologue SRD. The fix replaces the inline lookup with
// lookupSRD() which checks getSRDValue first.

// CHECK-LABEL: waveasm.program @store_reuses_load_srd

module {
func.func @store_reuses_load_srd(%arg0: memref<f16>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %valid = arith.constant 2147483645 : i64
  %stride = arith.constant 128 : i14

  %block_id = gpu.block_id x upper_bound 4
  %thread_id = gpu.thread_id x upper_bound 64

  %wg_row = arith.muli %block_id, %c128 : index
  %th_offset = arith.muli %thread_id, %c128 : index

  // reinterpret_cast -> cast -> fat_raw_buffer_cast creates a pending SRD
  // base adjustment. The first consumer (vector.load) triggers
  // emitSRDBaseAdjustment and stores the result via setSRDValue.
  %rc = memref.reinterpret_cast %arg0 to
      offset: [%wg_row], sizes: [1073741822], strides: [1]
      : memref<f16> to memref<1073741822xf16, strided<[1], offset: ?>>
  %cast = memref.cast %rc
      : memref<1073741822xf16, strided<[1], offset: ?>>
        to memref<?xf16, strided<[1], offset: ?>>
  %buf = amdgpu.fat_raw_buffer_cast %cast
      validBytes(%valid) cacheSwizzleStride(%stride) resetOffset
      : memref<?xf16, strided<[1], offset: ?>>
        to memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>

  // Load triggers the SRD adjustment: extract + typed ops + pack.
  // CHECK: waveasm.extract
  // CHECK: waveasm.s_add_u32
  // CHECK: waveasm.s_addc_u32
  // CHECK: [[SRD:%.*]] = waveasm.pack {{.*}} -> !waveasm.sreg<4, 4>
  // CHECK: waveasm.buffer_load_dwordx2 [[SRD]], %{{.*}}, %{{.*}}
  %loaded = vector.load %buf[%th_offset]
      : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>

  // Store to the SAME buffer must reuse the PackOp SRD from above.
  // There should be NO second extract/s_add_u32/pack sequence.
  // The buffer_store must reference the same [[SRD]] value.
  // CHECK-NOT: waveasm.s_add_u32
  // CHECK: waveasm.buffer_store_dwordx2 %{{.*}}, [[SRD]], %{{.*}}
  vector.store %loaded, %buf[%th_offset]
      : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>

  return
}
}
