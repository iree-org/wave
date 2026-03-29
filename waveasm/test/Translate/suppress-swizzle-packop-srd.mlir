// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: suppress-swizzle path with a PackOp-built source SRD.
//
// The first fat_raw_buffer_cast (with gather_to_lds consumer) goes through
// the full SRD rebuild, producing a PackOp result. The second cast uses the
// first cast's result as its source, has only vector.load consumers
// (suppress=true), and specifies non-max validBytes.
//
// Before the fix, the suppress path tried to extract a physical register
// index from srcMapped. PackOp results have SRegType (no physical index),
// so the in-place patch was silently skipped, leaving num_records stale.
//
// After the fix, PackOp SRDs fall through to the full rebuild path which
// builds a new SRD with correct num_records via buildSrdWord2.

// CHECK-LABEL: waveasm.program @suppress_swizzle_packop_srd

module {
  gpu.module @test_suppress_swizzle {
    gpu.func @suppress_swizzle_packop_srd(
        %src: memref<?xf16>,
        %tidx: index {llvm.mlir.workitem_id_x}
    ) kernel {
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %valid_max = arith.constant 2147483645 : i64
      %valid_tight = arith.constant 8192 : i64
      %stride = arith.constant 128 : i14

      %lds = memref.alloc() : memref<2048xi8, 3>
      %view = memref.view %lds[%c0][]
          : memref<2048xi8, 3> to memref<64x8xf16, 3>

      // First cast: has gather_to_lds consumer -> full SRD rebuild via PackOp.
      // CHECK: waveasm.pack {{.*}} -> !waveasm.sreg<4, 4>
      %buf1 = amdgpu.fat_raw_buffer_cast %src
          validBytes(%valid_max) cacheSwizzleStride(%stride) resetOffset
          : memref<?xf16>
            to memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>

      "amdgpu.gather_to_lds"(%buf1, %tidx, %view, %c0, %c0)
          <{operandSegmentSizes = array<i32: 1, 1, 1, 2>,
            transferType = vector<2xf16>}>
          : (memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>,
             index, memref<64x8xf16, 3>, index, index) -> ()

      // Second cast: source is the RESULT of the first cast (so srcMapped
      // is the PackOp SRD). Only vector.load consumers (suppress=true),
      // non-max validBytes (8192) -> enters suppress-swizzle path.
      //
      // With the fix, this falls through to the full rebuild and produces
      // a new PackOp with tight num_records (8192).
      // CHECK: waveasm.s_mov_b32 %{{.*}} : !waveasm.imm<8192>
      // CHECK: waveasm.pack {{.*}} -> !waveasm.sreg<4, 4>
      %buf2 = amdgpu.fat_raw_buffer_cast %buf1
          validBytes(%valid_tight) cacheSwizzleStride(%stride) resetOffset
          : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>
            to memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>

      // CHECK: waveasm.buffer_load_dwordx2
      %loaded = vector.load %buf2[%c64]
          : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>

      gpu.return
    }
  }
}
