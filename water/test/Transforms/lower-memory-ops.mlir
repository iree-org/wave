// RUN: water-opt %s --pass-pipeline='builtin.module(func.func(water-lower-memory-ops{chipset=gfx950}))' | FileCheck %s --check-prefixes=CHECK,GFX9
// RUN: water-opt %s --pass-pipeline='builtin.module(func.func(water-lower-memory-ops{chipset=gfx1200}))' | FileCheck %s --check-prefixes=CHECK,GFX12

// Test lowering of vector memory operations to AMDGPU global_load/store inline assembly

// CHECK-LABEL: func.func @simple_function
func.func @simple_function(%arg0: f32) -> f32 {
  // CHECK: return %arg0
  return %arg0 : f32
}

// CHECK-LABEL: func.func @vector_load
func.func @vector_load(%memref: memref<1024xf32>, %offset: index) -> vector<4xf32> {
  // CHECK: memref.extract_aligned_pointer_as_index
  // CHECK: arith.index_cast
  // CHECK: llvm.inttoptr
  // GFX9: llvm.inline_asm has_side_effects "global_load_dwordx4 $0, $1, off", "=v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_load_b128 $0, $1, off", "=v,v"
  %result = vector.load %memref[%offset] : memref<1024xf32>, vector<4xf32>
  // CHECK: return
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @vector_store
func.func @vector_store(%memref: memref<1024xf32>, %offset: index, %data: vector<4xf32>) {
  // CHECK: memref.extract_aligned_pointer_as_index
  // CHECK: arith.index_cast
  // CHECK: llvm.inttoptr
  // GFX9: llvm.inline_asm has_side_effects "global_store_dwordx4 $0, $1, off", "v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_store_b128 $0, $1, off", "v,v"
  vector.store %data, %memref[%offset] : memref<1024xf32>, vector<4xf32>
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @vector_load_b32
func.func @vector_load_b32(%memref: memref<1024xf32>, %offset: index) -> vector<1xf32> {
  // GFX9: llvm.inline_asm has_side_effects "global_load_dword $0, $1, off", "=v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_load_b32 $0, $1, off", "=v,v"
  %result = vector.load %memref[%offset] : memref<1024xf32>, vector<1xf32>
  return %result : vector<1xf32>
}

// CHECK-LABEL: func.func @vector_load_b64
func.func @vector_load_b64(%memref: memref<1024xf32>, %offset: index) -> vector<2xf32> {
  // GFX9: llvm.inline_asm has_side_effects "global_load_dwordx2 $0, $1, off", "=v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_load_b64 $0, $1, off", "=v,v"
  %result = vector.load %memref[%offset] : memref<1024xf32>, vector<2xf32>
  return %result : vector<2xf32>
}

// CHECK-LABEL: func.func @vector_load_b96
func.func @vector_load_b96(%memref: memref<1024xf32>, %offset: index) -> vector<3xf32> {
  // GFX9: llvm.inline_asm has_side_effects "global_load_dwordx3 $0, $1, off", "=v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_load_b96 $0, $1, off", "=v,v"
  %result = vector.load %memref[%offset] : memref<1024xf32>, vector<3xf32>
  return %result : vector<3xf32>
}

// CHECK-LABEL: func.func @vector_load_b128
func.func @vector_load_b128(%memref: memref<1024xf32>, %offset: index) -> vector<4xf32> {
  // GFX9: llvm.inline_asm has_side_effects "global_load_dwordx4 $0, $1, off", "=v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_load_b128 $0, $1, off", "=v,v"
  %result = vector.load %memref[%offset] : memref<1024xf32>, vector<4xf32>
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @vector_store_b32
func.func @vector_store_b32(%memref: memref<1024xf32>, %offset: index, %data: vector<1xf32>) {
  // GFX9: llvm.inline_asm has_side_effects "global_store_dword $0, $1, off", "v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_store_b32 $0, $1, off", "v,v"
  vector.store %data, %memref[%offset] : memref<1024xf32>, vector<1xf32>
  return
}

// CHECK-LABEL: func.func @vector_store_b64
func.func @vector_store_b64(%memref: memref<1024xf32>, %offset: index, %data: vector<2xf32>) {
  // GFX9: llvm.inline_asm has_side_effects "global_store_dwordx2 $0, $1, off", "v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_store_b64 $0, $1, off", "v,v"
  vector.store %data, %memref[%offset] : memref<1024xf32>, vector<2xf32>
  return
}

// CHECK-LABEL: func.func @vector_store_b96
func.func @vector_store_b96(%memref: memref<1024xf32>, %offset: index, %data: vector<3xf32>) {
  // GFX9: llvm.inline_asm has_side_effects "global_store_dwordx3 $0, $1, off", "v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_store_b96 $0, $1, off", "v,v"
  vector.store %data, %memref[%offset] : memref<1024xf32>, vector<3xf32>
  return
}

// CHECK-LABEL: func.func @vector_store_b128
func.func @vector_store_b128(%memref: memref<1024xf32>, %offset: index, %data: vector<4xf32>) {
  // GFX9: llvm.inline_asm has_side_effects "global_store_dwordx4 $0, $1, off", "v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_store_b128 $0, $1, off", "v,v"
  vector.store %data, %memref[%offset] : memref<1024xf32>, vector<4xf32>
  return
}

// CHECK-LABEL: func.func @load_store_sequence
func.func @load_store_sequence(%src: memref<1024xf32>, %dst: memref<1024xf32>, %offset: index) {
  // Test lowering of load/store sequence

  // GFX9: llvm.inline_asm has_side_effects "global_load_dwordx4 $0, $1, off", "=v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_load_b128 $0, $1, off", "=v,v"
  %data = vector.load %src[%offset] : memref<1024xf32>, vector<4xf32>

  // GFX9: llvm.inline_asm has_side_effects "global_store_dwordx4 $0, $1, off", "v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_store_b128 $0, $1, off", "v,v"
  vector.store %data, %dst[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK: return
  return
}

// -----
// Buffer operations tests

// CHECK-LABEL: func.func @buffer_load_b32
func.func @buffer_load_b32(%memref: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index) -> vector<1xf32> {
  // GFX9: llvm.inline_asm has_side_effects "buffer_load_dword $0, $1, $2, 0 offen", "=v,v,s"
  // GFX12: llvm.inline_asm has_side_effects "buffer_load_b32 $0, $1, $2, 0 offen", "=v,v,s"
  %result = vector.load %memref[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  return %result : vector<1xf32>
}

// CHECK-LABEL: func.func @buffer_load_b64
func.func @buffer_load_b64(%memref: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index) -> vector<2xf32> {
  // GFX9: llvm.inline_asm has_side_effects "buffer_load_dwordx2 $0, $1, $2, 0 offen", "=v,v,s"
  // GFX12: llvm.inline_asm has_side_effects "buffer_load_b64 $0, $1, $2, 0 offen", "=v,v,s"
  %result = vector.load %memref[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<2xf32>
  return %result : vector<2xf32>
}

// CHECK-LABEL: func.func @buffer_load_b96
func.func @buffer_load_b96(%memref: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index) -> vector<3xf32> {
  // GFX9: llvm.inline_asm has_side_effects "buffer_load_dwordx3 $0, $1, $2, 0 offen", "=v,v,s"
  // GFX12: llvm.inline_asm has_side_effects "buffer_load_b96 $0, $1, $2, 0 offen", "=v,v,s"
  %result = vector.load %memref[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<3xf32>
  return %result : vector<3xf32>
}

// CHECK-LABEL: func.func @buffer_load_b128
func.func @buffer_load_b128(%memref: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index) -> vector<4xf32> {
  // GFX9: llvm.inline_asm has_side_effects "buffer_load_dwordx4 $0, $1, $2, 0 offen", "=v,v,s"
  // GFX12: llvm.inline_asm has_side_effects "buffer_load_b128 $0, $1, $2, 0 offen", "=v,v,s"
  %result = vector.load %memref[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @buffer_store_b32
func.func @buffer_store_b32(%memref: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index, %data: vector<1xf32>) {
  // GFX9: llvm.inline_asm has_side_effects "buffer_store_dword $0, $1, $2, 0 offen", "v,v,s"
  // GFX12: llvm.inline_asm has_side_effects "buffer_store_b32 $0, $1, $2, 0 offen", "v,v,s"
  vector.store %data, %memref[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  return
}

// CHECK-LABEL: func.func @buffer_store_b64
func.func @buffer_store_b64(%memref: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index, %data: vector<2xf32>) {
  // GFX9: llvm.inline_asm has_side_effects "buffer_store_dwordx2 $0, $1, $2, 0 offen", "v,v,s"
  // GFX12: llvm.inline_asm has_side_effects "buffer_store_b64 $0, $1, $2, 0 offen", "v,v,s"
  vector.store %data, %memref[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<2xf32>
  return
}

// CHECK-LABEL: func.func @buffer_store_b96
func.func @buffer_store_b96(%memref: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index, %data: vector<3xf32>) {
  // GFX9: llvm.inline_asm has_side_effects "buffer_store_dwordx3 $0, $1, $2, 0 offen", "v,v,s"
  // GFX12: llvm.inline_asm has_side_effects "buffer_store_b96 $0, $1, $2, 0 offen", "v,v,s"
  vector.store %data, %memref[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<3xf32>
  return
}

// CHECK-LABEL: func.func @buffer_store_b128
func.func @buffer_store_b128(%memref: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index, %data: vector<4xf32>) {
  // GFX9: llvm.inline_asm has_side_effects "buffer_store_dwordx4 $0, $1, $2, 0 offen", "v,v,s"
  // GFX12: llvm.inline_asm has_side_effects "buffer_store_b128 $0, $1, $2, 0 offen", "v,v,s"
  vector.store %data, %memref[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  return
}

// CHECK-LABEL: func.func @mixed_global_and_buffer
func.func @mixed_global_and_buffer(%global: memref<1024xf32>, %buffer: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index) {
  // Load from global memory (should use global_load)
  // GFX9: llvm.inline_asm has_side_effects "global_load_dwordx4 $0, $1, off", "=v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_load_b128 $0, $1, off", "=v,v"
  %global_data = vector.load %global[%offset] : memref<1024xf32>, vector<4xf32>

  // Store to buffer memory (should use buffer_store)
  // GFX9: llvm.inline_asm has_side_effects "buffer_store_dwordx4 $0, $1, $2, 0 offen", "v,v,s"
  // GFX12: llvm.inline_asm has_side_effects "buffer_store_b128 $0, $1, $2, 0 offen", "v,v,s"
  vector.store %global_data, %buffer[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>

  // Load from buffer memory (should use buffer_load)
  // GFX9: llvm.inline_asm has_side_effects "buffer_load_dwordx4 $0, $1, $2, 0 offen", "=v,v,s"
  // GFX12: llvm.inline_asm has_side_effects "buffer_load_b128 $0, $1, $2, 0 offen", "=v,v,s"
  %buffer_data = vector.load %buffer[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>

  // Store to global memory (should use global_store)
  // GFX9: llvm.inline_asm has_side_effects "global_store_dwordx4 $0, $1, off", "v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_store_b128 $0, $1, off", "v,v"
  vector.store %buffer_data, %global[%offset] : memref<1024xf32>, vector<4xf32>

  return
}
// -----
// DS operations tests

// CHECK-LABEL: func.func @ds_load_b32
func.func @ds_load_b32(%lds: memref<1024xf32, #gpu.address_space<workgroup>>, %offset: index) -> vector<1xf32> {
  // CHECK: llvm.inline_asm has_side_effects "ds_read_b32 $0, $1", "=v,v"
  %result = vector.load %lds[%offset] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<1xf32>
  return %result : vector<1xf32>
}

// CHECK-LABEL: func.func @ds_load_b64
func.func @ds_load_b64(%lds: memref<1024xf32, #gpu.address_space<workgroup>>, %offset: index) -> vector<2xf32> {
  // CHECK: llvm.inline_asm has_side_effects "ds_read_b64 $0, $1", "=v,v"
  %result = vector.load %lds[%offset] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<2xf32>
  return %result : vector<2xf32>
}

// CHECK-LABEL: func.func @ds_load_b96
func.func @ds_load_b96(%lds: memref<1024xf32, #gpu.address_space<workgroup>>, %offset: index) -> vector<3xf32> {
  // CHECK: llvm.inline_asm has_side_effects "ds_read_b96 $0, $1", "=v,v"
  %result = vector.load %lds[%offset] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<3xf32>
  return %result : vector<3xf32>
}

// CHECK-LABEL: func.func @ds_load_b128
func.func @ds_load_b128(%lds: memref<1024xf32, #gpu.address_space<workgroup>>, %offset: index) -> vector<4xf32> {
  // CHECK: llvm.inline_asm has_side_effects "ds_read_b128 $0, $1", "=v,v"
  %result = vector.load %lds[%offset] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @ds_store_b32
func.func @ds_store_b32(%lds: memref<1024xf32, #gpu.address_space<workgroup>>, %offset: index, %data: vector<1xf32>) {
  // CHECK: llvm.inline_asm has_side_effects "ds_write_b32 $0, $1", "v,v"
  vector.store %data, %lds[%offset] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<1xf32>
  return
}

// CHECK-LABEL: func.func @ds_store_b64
func.func @ds_store_b64(%lds: memref<1024xf32, #gpu.address_space<workgroup>>, %offset: index, %data: vector<2xf32>) {
  // CHECK: llvm.inline_asm has_side_effects "ds_write_b64 $0, $1", "v,v"
  vector.store %data, %lds[%offset] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<2xf32>
  return
}

// CHECK-LABEL: func.func @ds_store_b96
func.func @ds_store_b96(%lds: memref<1024xf32, #gpu.address_space<workgroup>>, %offset: index, %data: vector<3xf32>) {
  // CHECK: llvm.inline_asm has_side_effects "ds_write_b96 $0, $1", "v,v"
  vector.store %data, %lds[%offset] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<3xf32>
  return
}

// CHECK-LABEL: func.func @ds_store_b128
func.func @ds_store_b128(%lds: memref<1024xf32, #gpu.address_space<workgroup>>, %offset: index, %data: vector<4xf32>) {
  // CHECK: llvm.inline_asm has_side_effects "ds_write_b128 $0, $1", "v,v"
  vector.store %data, %lds[%offset] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>
  return
}

// CHECK-LABEL: func.func @mixed_global_buffer_and_ds
func.func @mixed_global_buffer_and_ds(%global: memref<1024xf32>, %buffer: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %lds: memref<1024xf32, #gpu.address_space<workgroup>>, %offset: index) {
  // Load from global (should use global_load)
  // GFX9: llvm.inline_asm has_side_effects "global_load_dwordx4 $0, $1, off", "=v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_load_b128 $0, $1, off", "=v,v"
  %global_data = vector.load %global[%offset] : memref<1024xf32>, vector<4xf32>

  // Store to LDS (should use ds_write)
  // CHECK: llvm.inline_asm has_side_effects "ds_write_b128 $0, $1", "v,v"
  vector.store %global_data, %lds[%offset] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>

  // Load from LDS (should use ds_read)
  // CHECK: llvm.inline_asm has_side_effects "ds_read_b128 $0, $1", "=v,v"
  %lds_data = vector.load %lds[%offset] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>

  // Store to buffer (should use buffer_store)
  // GFX9: llvm.inline_asm has_side_effects "buffer_store_dwordx4 $0, $1, $2, 0 offen", "v,v,s"
  // GFX12: llvm.inline_asm has_side_effects "buffer_store_b128 $0, $1, $2, 0 offen", "v,v,s"
  vector.store %lds_data, %buffer[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>

  return
}

// -----
// Scalar (memref) operations tests

// CHECK-LABEL: func.func @scalar_load_global_f32
func.func @scalar_load_global_f32(%memref: memref<1024xf32>, %offset: index) -> f32 {
  // GFX9: llvm.inline_asm has_side_effects "global_load_dword $0, $1, off", "=v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_load_b32 $0, $1, off", "=v,v"
  %result = memref.load %memref[%offset] : memref<1024xf32>
  return %result : f32
}

// CHECK-LABEL: func.func @scalar_load_global_f64
func.func @scalar_load_global_f64(%memref: memref<1024xf64>, %offset: index) -> f64 {
  // GFX9: llvm.inline_asm has_side_effects "global_load_dwordx2 $0, $1, off", "=v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_load_b64 $0, $1, off", "=v,v"
  %result = memref.load %memref[%offset] : memref<1024xf64>
  return %result : f64
}

// CHECK-LABEL: func.func @scalar_store_global_f32
func.func @scalar_store_global_f32(%memref: memref<1024xf32>, %offset: index, %data: f32) {
  // GFX9: llvm.inline_asm has_side_effects "global_store_dword $0, $1, off", "v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_store_b32 $0, $1, off", "v,v"
  memref.store %data, %memref[%offset] : memref<1024xf32>
  return
}

// CHECK-LABEL: func.func @scalar_store_global_f64
func.func @scalar_store_global_f64(%memref: memref<1024xf64>, %offset: index, %data: f64) {
  // GFX9: llvm.inline_asm has_side_effects "global_store_dwordx2 $0, $1, off", "v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_store_b64 $0, $1, off", "v,v"
  memref.store %data, %memref[%offset] : memref<1024xf64>
  return
}

// CHECK-LABEL: func.func @scalar_load_buffer_f32
func.func @scalar_load_buffer_f32(%buffer: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index) -> f32 {
  // GFX9: llvm.inline_asm has_side_effects "buffer_load_dword $0, $1, $2, 0 offen", "=v,v,s"
  // GFX12: llvm.inline_asm has_side_effects "buffer_load_b32 $0, $1, $2, 0 offen", "=v,v,s"
  %result = memref.load %buffer[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>
  return %result : f32
}

// CHECK-LABEL: func.func @scalar_store_buffer_f32
func.func @scalar_store_buffer_f32(%buffer: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index, %data: f32) {
  // GFX9: llvm.inline_asm has_side_effects "buffer_store_dword $0, $1, $2, 0 offen", "v,v,s"
  // GFX12: llvm.inline_asm has_side_effects "buffer_store_b32 $0, $1, $2, 0 offen", "v,v,s"
  memref.store %data, %buffer[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>
  return
}

// CHECK-LABEL: func.func @scalar_load_ds_f32
func.func @scalar_load_ds_f32(%lds: memref<1024xf32, #gpu.address_space<workgroup>>, %offset: index) -> f32 {
  // CHECK: llvm.inline_asm has_side_effects "ds_read_b32 $0, $1", "=v,v"
  %result = memref.load %lds[%offset] : memref<1024xf32, #gpu.address_space<workgroup>>
  return %result : f32
}

// CHECK-LABEL: func.func @scalar_store_ds_f32
func.func @scalar_store_ds_f32(%lds: memref<1024xf32, #gpu.address_space<workgroup>>, %offset: index, %data: f32) {
  // CHECK: llvm.inline_asm has_side_effects "ds_write_b32 $0, $1", "v,v"
  memref.store %data, %lds[%offset] : memref<1024xf32, #gpu.address_space<workgroup>>
  return
}

// CHECK-LABEL: func.func @mixed_scalar_and_vector
func.func @mixed_scalar_and_vector(%memref: memref<1024xf32>, %offset: index) {
  // Scalar load
  // GFX9: llvm.inline_asm has_side_effects "global_load_dword $0, $1, off", "=v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_load_b32 $0, $1, off", "=v,v"
  %scalar = memref.load %memref[%offset] : memref<1024xf32>

  // Vector load
  // GFX9: llvm.inline_asm has_side_effects "global_load_dwordx4 $0, $1, off", "=v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_load_b128 $0, $1, off", "=v,v"
  %vector = vector.load %memref[%offset] : memref<1024xf32>, vector<4xf32>

  // Scalar store
  // GFX9: llvm.inline_asm has_side_effects "global_store_dword $0, $1, off", "v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_store_b32 $0, $1, off", "v,v"
  memref.store %scalar, %memref[%offset] : memref<1024xf32>

  // Vector store
  // GFX9: llvm.inline_asm has_side_effects "global_store_dwordx4 $0, $1, off", "v,v"
  // GFX12: llvm.inline_asm has_side_effects "global_store_b128 $0, $1, off", "v,v"
  vector.store %vector, %memref[%offset] : memref<1024xf32>, vector<4xf32>

  return
}

// Test copy to register space with pre-numbered allocas

// CHECK-LABEL: func.func @copy_global_to_reg_scalar
// GFX9-SAME{LITERAL}: passthrough = [["amdgpu-num-vgpr", "255"]]
// GFX12-SAME{LITERAL}: passthrough = [["amdgpu-num-vgpr", "255"]]
func.func @copy_global_to_reg_scalar(%arg0: memref<100xf32>) -> f32 attributes {water.total_vgprs = 1 : i32} {
  %c0 = arith.constant 0 : index
  %reg = memref.alloca() {water.vgpr_number = 0 : i32, water.vgpr_count = 1 : i32} : memref<1xf32, 128 : i32>
  %subview = memref.subview %arg0[%c0] [1] [1] : memref<100xf32> to memref<1xf32, strided<[1], offset: ?>>
  // GFX9: llvm.inline_asm has_side_effects "global_load_dword $0, $1, off", "={v255},v"
  // GFX12: llvm.inline_asm has_side_effects "global_load_b32 $0, $1, off", "={v255},v"
  memref.copy %subview, %reg : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, 128 : i32>
  // GFX9: llvm.inline_asm "; reg_load v255", "={v255}"
  // GFX12: llvm.inline_asm "; reg_load v255", "={v255}"
  %val = memref.load %reg[%c0] : memref<1xf32, 128 : i32>
  // CHECK-NOT: memref.alloca
  return %val : f32
}

// CHECK-LABEL: func.func @copy_global_to_reg_vector
// GFX9-SAME{LITERAL}: passthrough = [["amdgpu-num-vgpr", "252"]]
// GFX12-SAME{LITERAL}: passthrough = [["amdgpu-num-vgpr", "252"]]
func.func @copy_global_to_reg_vector(%arg0: memref<100xf32>) -> vector<4xf32> attributes {water.total_vgprs = 4 : i32} {
  %c0 = arith.constant 0 : index
  %reg = memref.alloca() {water.vgpr_number = 0 : i32, water.vgpr_count = 4 : i32} : memref<4xf32, 128 : i32>
  %subview = memref.subview %arg0[%c0] [4] [1] : memref<100xf32> to memref<4xf32, strided<[1], offset: ?>>
  // GFX9: llvm.inline_asm has_side_effects "global_load_dwordx4 $0, $1, off", "={v[252:255]},v"
  // GFX12: llvm.inline_asm has_side_effects "global_load_b128 $0, $1, off", "={v[252:255]},v"
  memref.copy %subview, %reg : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32, 128 : i32>
  // GFX9: llvm.inline_asm "; reg_load v[252:255]", "={v[252:255]}"
  // GFX12: llvm.inline_asm "; reg_load v[252:255]", "={v[252:255]}"
  %val = vector.load %reg[%c0] : memref<4xf32, 128 : i32>, vector<4xf32>
  // CHECK-NOT: memref.alloca
  return %val : vector<4xf32>
}

// CHECK-LABEL: func.func @copy_buffer_to_reg
// GFX9-SAME{LITERAL}: passthrough = [["amdgpu-num-vgpr", "252"]]
// GFX12-SAME{LITERAL}: passthrough = [["amdgpu-num-vgpr", "252"]]
func.func @copy_buffer_to_reg(%arg0: memref<100xf32, #amdgpu.address_space<fat_raw_buffer>>) -> vector<4xf32> attributes {water.total_vgprs = 4 : i32} {
  %c0 = arith.constant 0 : index
  %reg = memref.alloca() {water.vgpr_number = 0 : i32, water.vgpr_count = 4 : i32} : memref<4xf32, 128 : i32>
  %subview = memref.subview %arg0[%c0] [4] [1] : memref<100xf32, #amdgpu.address_space<fat_raw_buffer>> to memref<4xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
  // GFX9: llvm.inline_asm has_side_effects "buffer_load_dwordx4 $0, $1, $2, 0 offen", "={v[252:255]},v,s"
  // GFX12: llvm.inline_asm has_side_effects "buffer_load_b128 $0, $1, $2, 0 offen", "={v[252:255]},v,s"
  memref.copy %subview, %reg : memref<4xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>> to memref<4xf32, 128 : i32>
  // GFX9: llvm.inline_asm "; reg_load v[252:255]", "={v[252:255]}"
  // GFX12: llvm.inline_asm "; reg_load v[252:255]", "={v[252:255]}"
  %val = vector.load %reg[%c0] : memref<4xf32, 128 : i32>, vector<4xf32>
  // CHECK-NOT: memref.alloca
  return %val : vector<4xf32>
}

// CHECK-LABEL: func.func @copy_workgroup_to_reg
// GFX9-SAME{LITERAL}: passthrough = [["amdgpu-num-vgpr", "252"]]
// GFX12-SAME{LITERAL}: passthrough = [["amdgpu-num-vgpr", "252"]]
func.func @copy_workgroup_to_reg(%arg0: memref<100xf32, #gpu.address_space<workgroup>>) -> vector<4xf32> attributes {water.total_vgprs = 4 : i32} {
  %c0 = arith.constant 0 : index
  %reg = memref.alloca() {water.vgpr_number = 0 : i32, water.vgpr_count = 4 : i32} : memref<4xf32, 128 : i32>
  %subview = memref.subview %arg0[%c0] [4] [1] : memref<100xf32, #gpu.address_space<workgroup>> to memref<4xf32, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
  // GFX9: llvm.inline_asm has_side_effects "ds_read_b128 $0, $1", "={v[252:255]},v"
  // GFX12: llvm.inline_asm has_side_effects "ds_read_b128 $0, $1", "={v[252:255]},v"
  memref.copy %subview, %reg : memref<4xf32, strided<[1], offset: ?>, #gpu.address_space<workgroup>> to memref<4xf32, 128 : i32>
  // GFX9: llvm.inline_asm "; reg_load v[252:255]", "={v[252:255]}"
  // GFX12: llvm.inline_asm "; reg_load v[252:255]", "={v[252:255]}"
  %val = vector.load %reg[%c0] : memref<4xf32, 128 : i32>, vector<4xf32>
  // CHECK-NOT: memref.alloca
  return %val : vector<4xf32>
}

// CHECK-LABEL: func.func @store_to_reg
// GFX9-SAME{LITERAL}: passthrough = [["amdgpu-num-vgpr", "255"]]
// GFX12-SAME{LITERAL}: passthrough = [["amdgpu-num-vgpr", "255"]]
func.func @store_to_reg(%val: f32) -> f32 attributes {water.total_vgprs = 1 : i32} {
  %c0 = arith.constant 0 : index
  %reg = memref.alloca() {water.vgpr_number = 0 : i32, water.vgpr_count = 1 : i32} : memref<1xf32, 128 : i32>
  // GFX9: llvm.inline_asm has_side_effects "; reg_store v255", "={v255},0"
  // GFX12: llvm.inline_asm has_side_effects "; reg_store v255", "={v255},0"
  memref.store %val, %reg[%c0] : memref<1xf32, 128 : i32>
  // GFX9: llvm.inline_asm "; reg_load v255", "={v255}"
  // GFX12: llvm.inline_asm "; reg_load v255", "={v255}"
  %result = memref.load %reg[%c0] : memref<1xf32, 128 : i32>
  // CHECK-NOT: memref.alloca
  return %result : f32
}

// CHECK-LABEL: func.func @multiple_reg_allocas
// GFX9-SAME{LITERAL}: passthrough = [["amdgpu-num-vgpr", "247"]]
// GFX12-SAME{LITERAL}: passthrough = [["amdgpu-num-vgpr", "247"]]
func.func @multiple_reg_allocas(%arg0: memref<100xf32>, %arg1: memref<100xf32, #gpu.address_space<workgroup>>) -> (f32, vector<4xf32>, vector<4xf32>) attributes {water.total_vgprs = 9 : i32} {
  %c0 = arith.constant 0 : index
  %reg0 = memref.alloca() {water.vgpr_number = 0 : i32, water.vgpr_count = 1 : i32} : memref<1xf32, 128 : i32>
  %reg1 = memref.alloca() {water.vgpr_number = 1 : i32, water.vgpr_count = 4 : i32} : memref<4xf32, 128 : i32>
  %reg2 = memref.alloca() {water.vgpr_number = 5 : i32, water.vgpr_count = 4 : i32} : memref<4xf32, 128 : i32>
  // GFX9: llvm.inline_asm has_side_effects "global_load_dword $0, $1, off", "={v247},v"
  // GFX12: llvm.inline_asm has_side_effects "global_load_b32 $0, $1, off", "={v247},v"
  %sv0 = memref.subview %arg0[%c0] [1] [1] : memref<100xf32> to memref<1xf32, strided<[1], offset: ?>>
  memref.copy %sv0, %reg0 : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, 128 : i32>
  // GFX9: llvm.inline_asm has_side_effects "global_load_dwordx4 $0, $1, off", "={v[248:251]},v"
  // GFX12: llvm.inline_asm has_side_effects "global_load_b128 $0, $1, off", "={v[248:251]},v"
  %sv1 = memref.subview %arg0[%c0] [4] [1] : memref<100xf32> to memref<4xf32, strided<[1], offset: ?>>
  memref.copy %sv1, %reg1 : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32, 128 : i32>
  // GFX9: llvm.inline_asm has_side_effects "ds_read_b128 $0, $1", "={v[252:255]},v"
  // GFX12: llvm.inline_asm has_side_effects "ds_read_b128 $0, $1", "={v[252:255]},v"
  %sv2 = memref.subview %arg1[%c0] [4] [1] : memref<100xf32, #gpu.address_space<workgroup>> to memref<4xf32, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
  memref.copy %sv2, %reg2 : memref<4xf32, strided<[1], offset: ?>, #gpu.address_space<workgroup>> to memref<4xf32, 128 : i32>
  // GFX9: llvm.inline_asm "; reg_load v247", "={v247}"
  // GFX12: llvm.inline_asm "; reg_load v247", "={v247}"
  %val0 = memref.load %reg0[%c0] : memref<1xf32, 128 : i32>
  // GFX9: llvm.inline_asm "; reg_load v[248:251]", "={v[248:251]}"
  // GFX12: llvm.inline_asm "; reg_load v[248:251]", "={v[248:251]}"
  %val1 = vector.load %reg1[%c0] : memref<4xf32, 128 : i32>, vector<4xf32>
  // GFX9: llvm.inline_asm "; reg_load v[252:255]", "={v[252:255]}"
  // GFX12: llvm.inline_asm "; reg_load v[252:255]", "={v[252:255]}"
  %val2 = vector.load %reg2[%c0] : memref<4xf32, 128 : i32>, vector<4xf32>
  // CHECK-NOT: memref.alloca
  return %val0, %val1, %val2 : f32, vector<4xf32>, vector<4xf32>
}
