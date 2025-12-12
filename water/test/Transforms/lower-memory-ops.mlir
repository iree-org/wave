// RUN: water-opt %s --water-lower-memory-ops | FileCheck %s

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
  // CHECK: llvm.inline_asm has_side_effects "global_load_b128 $0, $1, off", "=v,v"
  %result = vector.load %memref[%offset] : memref<1024xf32>, vector<4xf32>
  // CHECK: return
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @vector_store
func.func @vector_store(%memref: memref<1024xf32>, %offset: index, %data: vector<4xf32>) {
  // CHECK: memref.extract_aligned_pointer_as_index
  // CHECK: arith.index_cast
  // CHECK: llvm.inttoptr
  // CHECK: llvm.inline_asm has_side_effects "global_store_b128 $0, $1, off", "v,v"
  vector.store %data, %memref[%offset] : memref<1024xf32>, vector<4xf32>
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @vector_load_b32
func.func @vector_load_b32(%memref: memref<1024xf32>, %offset: index) -> vector<1xf32> {
  // CHECK: llvm.inline_asm has_side_effects "global_load_b32 $0, $1, off", "=v,v"
  %result = vector.load %memref[%offset] : memref<1024xf32>, vector<1xf32>
  return %result : vector<1xf32>
}

// CHECK-LABEL: func.func @vector_load_b64
func.func @vector_load_b64(%memref: memref<1024xf32>, %offset: index) -> vector<2xf32> {
  // CHECK: llvm.inline_asm has_side_effects "global_load_b64 $0, $1, off", "=v,v"
  %result = vector.load %memref[%offset] : memref<1024xf32>, vector<2xf32>
  return %result : vector<2xf32>
}

// CHECK-LABEL: func.func @vector_load_b96
func.func @vector_load_b96(%memref: memref<1024xf32>, %offset: index) -> vector<3xf32> {
  // CHECK: llvm.inline_asm has_side_effects "global_load_b96 $0, $1, off", "=v,v"
  %result = vector.load %memref[%offset] : memref<1024xf32>, vector<3xf32>
  return %result : vector<3xf32>
}

// CHECK-LABEL: func.func @vector_load_b128
func.func @vector_load_b128(%memref: memref<1024xf32>, %offset: index) -> vector<4xf32> {
  // CHECK: llvm.inline_asm has_side_effects "global_load_b128 $0, $1, off", "=v,v"
  %result = vector.load %memref[%offset] : memref<1024xf32>, vector<4xf32>
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @vector_store_b32
func.func @vector_store_b32(%memref: memref<1024xf32>, %offset: index, %data: vector<1xf32>) {
  // CHECK: llvm.inline_asm has_side_effects "global_store_b32 $0, $1, off", "v,v"
  vector.store %data, %memref[%offset] : memref<1024xf32>, vector<1xf32>
  return
}

// CHECK-LABEL: func.func @vector_store_b64
func.func @vector_store_b64(%memref: memref<1024xf32>, %offset: index, %data: vector<2xf32>) {
  // CHECK: llvm.inline_asm has_side_effects "global_store_b64 $0, $1, off", "v,v"
  vector.store %data, %memref[%offset] : memref<1024xf32>, vector<2xf32>
  return
}

// CHECK-LABEL: func.func @vector_store_b96
func.func @vector_store_b96(%memref: memref<1024xf32>, %offset: index, %data: vector<3xf32>) {
  // CHECK: llvm.inline_asm has_side_effects "global_store_b96 $0, $1, off", "v,v"
  vector.store %data, %memref[%offset] : memref<1024xf32>, vector<3xf32>
  return
}

// CHECK-LABEL: func.func @vector_store_b128
func.func @vector_store_b128(%memref: memref<1024xf32>, %offset: index, %data: vector<4xf32>) {
  // CHECK: llvm.inline_asm has_side_effects "global_store_b128 $0, $1, off", "v,v"
  vector.store %data, %memref[%offset] : memref<1024xf32>, vector<4xf32>
  return
}

// CHECK-LABEL: func.func @load_store_sequence
func.func @load_store_sequence(%src: memref<1024xf32>, %dst: memref<1024xf32>, %offset: index) {
  // Test lowering of load/store sequence

  // CHECK: llvm.inline_asm has_side_effects "global_load_b128 $0, $1, off", "=v,v"
  %data = vector.load %src[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK: llvm.inline_asm has_side_effects "global_store_b128 $0, $1, off", "v,v"
  vector.store %data, %dst[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK: return
  return
}

// -----
// Buffer operations tests

// CHECK-LABEL: func.func @buffer_load_b32
func.func @buffer_load_b32(%memref: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index) -> vector<1xf32> {
  // CHECK: llvm.inline_asm has_side_effects "buffer_load_dword $0, $1, $2, 0 offen", "=v,v,s"
  %result = vector.load %memref[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  return %result : vector<1xf32>
}

// CHECK-LABEL: func.func @buffer_load_b64
func.func @buffer_load_b64(%memref: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index) -> vector<2xf32> {
  // CHECK: llvm.inline_asm has_side_effects "buffer_load_dwordx2 $0, $1, $2, 0 offen", "=v,v,s"
  %result = vector.load %memref[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<2xf32>
  return %result : vector<2xf32>
}

// CHECK-LABEL: func.func @buffer_load_b96
func.func @buffer_load_b96(%memref: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index) -> vector<3xf32> {
  // CHECK: llvm.inline_asm has_side_effects "buffer_load_dwordx3 $0, $1, $2, 0 offen", "=v,v,s"
  %result = vector.load %memref[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<3xf32>
  return %result : vector<3xf32>
}

// CHECK-LABEL: func.func @buffer_load_b128
func.func @buffer_load_b128(%memref: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index) -> vector<4xf32> {
  // CHECK: llvm.inline_asm has_side_effects "buffer_load_dwordx4 $0, $1, $2, 0 offen", "=v,v,s"
  %result = vector.load %memref[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @buffer_store_b32
func.func @buffer_store_b32(%memref: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index, %data: vector<1xf32>) {
  // CHECK: llvm.inline_asm has_side_effects "buffer_store_dword $0, $1, $2, 0 offen", "v,v,s"
  vector.store %data, %memref[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  return
}

// CHECK-LABEL: func.func @buffer_store_b64
func.func @buffer_store_b64(%memref: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index, %data: vector<2xf32>) {
  // CHECK: llvm.inline_asm has_side_effects "buffer_store_dwordx2 $0, $1, $2, 0 offen", "v,v,s"
  vector.store %data, %memref[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<2xf32>
  return
}

// CHECK-LABEL: func.func @buffer_store_b96
func.func @buffer_store_b96(%memref: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index, %data: vector<3xf32>) {
  // CHECK: llvm.inline_asm has_side_effects "buffer_store_dwordx3 $0, $1, $2, 0 offen", "v,v,s"
  vector.store %data, %memref[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<3xf32>
  return
}

// CHECK-LABEL: func.func @buffer_store_b128
func.func @buffer_store_b128(%memref: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index, %data: vector<4xf32>) {
  // CHECK: llvm.inline_asm has_side_effects "buffer_store_dwordx4 $0, $1, $2, 0 offen", "v,v,s"
  vector.store %data, %memref[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  return
}

// CHECK-LABEL: func.func @mixed_global_and_buffer
func.func @mixed_global_and_buffer(%global: memref<1024xf32>, %buffer: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index) {
  // Load from global memory (should use global_load)
  // CHECK: llvm.inline_asm has_side_effects "global_load_b128 $0, $1, off", "=v,v"
  %global_data = vector.load %global[%offset] : memref<1024xf32>, vector<4xf32>

  // Store to buffer memory (should use buffer_store)
  // CHECK: llvm.inline_asm has_side_effects "buffer_store_dwordx4 $0, $1, $2, 0 offen", "v,v,s"
  vector.store %global_data, %buffer[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>

  // Load from buffer memory (should use buffer_load)
  // CHECK: llvm.inline_asm has_side_effects "buffer_load_dwordx4 $0, $1, $2, 0 offen", "=v,v,s"
  %buffer_data = vector.load %buffer[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>

  // Store to global memory (should use global_store)
  // CHECK: llvm.inline_asm has_side_effects "global_store_b128 $0, $1, off", "v,v"
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
  // CHECK: llvm.inline_asm has_side_effects "global_load_b128 $0, $1, off", "=v,v"
  %global_data = vector.load %global[%offset] : memref<1024xf32>, vector<4xf32>

  // Store to LDS (should use ds_write)
  // CHECK: llvm.inline_asm has_side_effects "ds_write_b128 $0, $1", "v,v"
  vector.store %global_data, %lds[%offset] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>

  // Load from LDS (should use ds_read)
  // CHECK: llvm.inline_asm has_side_effects "ds_read_b128 $0, $1", "=v,v"
  %lds_data = vector.load %lds[%offset] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>

  // Store to buffer (should use buffer_store)
  // CHECK: llvm.inline_asm has_side_effects "buffer_store_dwordx4 $0, $1, $2, 0 offen", "v,v,s"
  vector.store %lds_data, %buffer[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>

  return
}

// -----
// Scalar (memref) operations tests

// CHECK-LABEL: func.func @scalar_load_global_f32
func.func @scalar_load_global_f32(%memref: memref<1024xf32>, %offset: index) -> f32 {
  // CHECK: llvm.inline_asm has_side_effects "global_load_b32 $0, $1, off", "=v,v"
  %result = memref.load %memref[%offset] : memref<1024xf32>
  return %result : f32
}

// CHECK-LABEL: func.func @scalar_load_global_f64
func.func @scalar_load_global_f64(%memref: memref<1024xf64>, %offset: index) -> f64 {
  // CHECK: llvm.inline_asm has_side_effects "global_load_b64 $0, $1, off", "=v,v"
  %result = memref.load %memref[%offset] : memref<1024xf64>
  return %result : f64
}

// CHECK-LABEL: func.func @scalar_store_global_f32
func.func @scalar_store_global_f32(%memref: memref<1024xf32>, %offset: index, %data: f32) {
  // CHECK: llvm.inline_asm has_side_effects "global_store_b32 $0, $1, off", "v,v"
  memref.store %data, %memref[%offset] : memref<1024xf32>
  return
}

// CHECK-LABEL: func.func @scalar_store_global_f64
func.func @scalar_store_global_f64(%memref: memref<1024xf64>, %offset: index, %data: f64) {
  // CHECK: llvm.inline_asm has_side_effects "global_store_b64 $0, $1, off", "v,v"
  memref.store %data, %memref[%offset] : memref<1024xf64>
  return
}

// CHECK-LABEL: func.func @scalar_load_buffer_f32
func.func @scalar_load_buffer_f32(%buffer: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index) -> f32 {
  // CHECK: llvm.inline_asm has_side_effects "buffer_load_dword $0, $1, $2, 0 offen", "=v,v,s"
  %result = memref.load %buffer[%offset] : memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>
  return %result : f32
}

// CHECK-LABEL: func.func @scalar_store_buffer_f32
func.func @scalar_store_buffer_f32(%buffer: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>, %offset: index, %data: f32) {
  // CHECK: llvm.inline_asm has_side_effects "buffer_store_dword $0, $1, $2, 0 offen", "v,v,s"
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
  // CHECK: llvm.inline_asm has_side_effects "global_load_b32 $0, $1, off", "=v,v"
  %scalar = memref.load %memref[%offset] : memref<1024xf32>

  // Vector load
  // CHECK: llvm.inline_asm has_side_effects "global_load_b128 $0, $1, off", "=v,v"
  %vector = vector.load %memref[%offset] : memref<1024xf32>, vector<4xf32>

  // Scalar store
  // CHECK: llvm.inline_asm has_side_effects "global_store_b32 $0, $1, off", "v,v"
  memref.store %scalar, %memref[%offset] : memref<1024xf32>

  // Vector store
  // CHECK: llvm.inline_asm has_side_effects "global_store_b128 $0, $1, off", "v,v"
  vector.store %vector, %memref[%offset] : memref<1024xf32>, vector<4xf32>

  return
}
