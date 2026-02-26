// RUN: water-opt %s --water-fuse-scale-loads="wave-size=32" | FileCheck %s

// CHECK-LABEL: func.func @fuse_scale_loads
// CHECK-SAME: (%[[PTRA:.*]]: !llvm.ptr, %[[PTRB:.*]]: !llvm.ptr, %[[MATA:.*]]: vector<64xf8E4M3FN>, %[[MATB:.*]]: vector<64xf8E4M3FN>, %[[ACC:.*]]: vector<8xf32>)
func.func @fuse_scale_loads(%ptrA: !llvm.ptr, %ptrB: !llvm.ptr,
                            %matA: vector<64xf8E4M3FN>, %matB: vector<64xf8E4M3FN>,
                            %acc: vector<8xf32>) -> vector<8xf32> {
  // CHECK-DAG: %[[HALF:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[LANE:.*]] = gpu.lane_id upper_bound 32
  // CHECK: %[[CMP:.*]] = arith.cmpi ult, %[[LANE]], %[[HALF]]
  // CHECK: %[[SEL:.*]] = arith.select %[[CMP]], %[[PTRA]], %[[PTRB]]
  // CHECK: %[[LOAD:.*]] = llvm.load %[[SEL]]
  // CHECK: amdgpu.scaled_wmma 16x16x128 (%[[LOAD]] * %[[MATA]]) * (%[[LOAD]] * %[[MATB]])
  // CHECK-SAME: b_first_scale_lane = 16 : i32
  %scaleA = llvm.load %ptrA : !llvm.ptr -> vector<4xf8E8M0FNU>
  %scaleB = llvm.load %ptrB : !llvm.ptr -> vector<4xf8E8M0FNU>
  %result = amdgpu.scaled_wmma 16x16x128 (%scaleA * %matA) * (%scaleB * %matB) + %acc
    {a_first_scale_lane = 0 : i32, b_first_scale_lane = 0 : i32}
    : vector<4xf8E8M0FNU>, vector<64xf8E4M3FN>, vector<4xf8E8M0FNU>, vector<64xf8E4M3FN>, vector<8xf32>
  return %result : vector<8xf32>
}

// -----

// Test with elementwise ops between load and wmma - loads must have same result type.
// CHECK-LABEL: func.func @fuse_scale_loads_with_elementwise
func.func @fuse_scale_loads_with_elementwise(%ptrA: !llvm.ptr, %ptrB: !llvm.ptr,
                                             %matA: vector<64xf8E4M3FN>, %matB: vector<64xf8E4M3FN>,
                                             %acc: vector<8xf32>) -> vector<8xf32> {
  // CHECK: %[[LOAD:.*]] = llvm.load
  // CHECK: %[[BCAST1:.*]] = llvm.bitcast %[[LOAD]]
  // CHECK: %[[BCAST2:.*]] = llvm.bitcast %[[LOAD]]
  // CHECK: amdgpu.scaled_wmma 16x16x128 (%[[BCAST1]] * {{.*}}) * (%[[BCAST2]] * {{.*}})
  // CHECK-SAME: b_first_scale_lane = 16 : i32
  %loadA = llvm.load %ptrA : !llvm.ptr -> vector<4xf8E4M3FN>
  %loadB = llvm.load %ptrB : !llvm.ptr -> vector<4xf8E4M3FN>
  %scaleA = llvm.bitcast %loadA : vector<4xf8E4M3FN> to vector<4xf8E8M0FNU>
  %scaleB = llvm.bitcast %loadB : vector<4xf8E4M3FN> to vector<4xf8E8M0FNU>
  %result = amdgpu.scaled_wmma 16x16x128 (%scaleA * %matA) * (%scaleB * %matB) + %acc
    {a_first_scale_lane = 0 : i32, b_first_scale_lane = 0 : i32}
    : vector<4xf8E8M0FNU>, vector<64xf8E4M3FN>, vector<4xf8E8M0FNU>, vector<64xf8E4M3FN>, vector<8xf32>
  return %result : vector<8xf32>
}

// -----

// Test that we don't fuse when already fused (b_first_scale_lane != 0).
// CHECK-LABEL: func.func @no_fuse_already_offset
func.func @no_fuse_already_offset(%ptrA: !llvm.ptr, %ptrB: !llvm.ptr,
                                  %matA: vector<64xf8E4M3FN>, %matB: vector<64xf8E4M3FN>,
                                  %acc: vector<8xf32>) -> vector<8xf32> {
  // CHECK: llvm.load %{{.*}} : !llvm.ptr -> vector<4xf8E8M0FNU>
  // CHECK: llvm.load %{{.*}} : !llvm.ptr -> vector<4xf8E8M0FNU>
  // CHECK: amdgpu.scaled_wmma
  // CHECK-SAME: b_first_scale_lane = 16 : i32
  %scaleA = llvm.load %ptrA : !llvm.ptr -> vector<4xf8E8M0FNU>
  %scaleB = llvm.load %ptrB : !llvm.ptr -> vector<4xf8E8M0FNU>
  %result = amdgpu.scaled_wmma 16x16x128 (%scaleA * %matA) * (%scaleB * %matB) + %acc
    {a_first_scale_lane = 0 : i32, b_first_scale_lane = 16 : i32}
    : vector<4xf8E8M0FNU>, vector<64xf8E4M3FN>, vector<4xf8E8M0FNU>, vector<64xf8E4M3FN>, vector<8xf32>
  return %result : vector<8xf32>
}

// -----

// Test that we don't fuse when loads have the same address.
// CHECK-LABEL: func.func @no_fuse_same_addr
func.func @no_fuse_same_addr(%ptr: !llvm.ptr,
                             %matA: vector<64xf8E4M3FN>, %matB: vector<64xf8E4M3FN>,
                             %acc: vector<8xf32>) -> vector<8xf32> {
  // CHECK: llvm.load %{{.*}} : !llvm.ptr -> vector<4xf8E8M0FNU>
  // CHECK: llvm.load %{{.*}} : !llvm.ptr -> vector<4xf8E8M0FNU>
  // CHECK: amdgpu.scaled_wmma
  // CHECK-SAME: b_first_scale_lane = 0 : i32
  %scaleA = llvm.load %ptr : !llvm.ptr -> vector<4xf8E8M0FNU>
  %scaleB = llvm.load %ptr : !llvm.ptr -> vector<4xf8E8M0FNU>
  %result = amdgpu.scaled_wmma 16x16x128 (%scaleA * %matA) * (%scaleB * %matB) + %acc
    {a_first_scale_lane = 0 : i32, b_first_scale_lane = 0 : i32}
    : vector<4xf8E8M0FNU>, vector<64xf8E4M3FN>, vector<4xf8E8M0FNU>, vector<64xf8E4M3FN>, vector<8xf32>
  return %result : vector<8xf32>
}
