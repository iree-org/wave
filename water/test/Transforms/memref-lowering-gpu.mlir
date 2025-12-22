// RUN: water-opt %s --water-memref-lowering | FileCheck %s

module attributes {gpu.container_module} {
gpu.module @kernels {
  // CHECK-LABEL: gpu.func @gpu_kernel_1d
  // CHECK-SAME: (%{{.*}}: !llvm.ptr)
  gpu.func @gpu_kernel_1d(%arg0: memref<100xf32>) kernel {
    gpu.return
  }

  // CHECK-LABEL: gpu.func @gpu_kernel_2d
  // CHECK-SAME: (%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr)
  gpu.func @gpu_kernel_2d(%arg0: memref<10x20xf32>, %arg1: memref<10x20xf32>) kernel {
    gpu.return
  }

  // CHECK-LABEL: gpu.func @gpu_kernel_mixed
  // CHECK-SAME: (%{{.*}}: !llvm.ptr, %{{.*}}: i32)
  gpu.func @gpu_kernel_mixed(%arg0: memref<100xf32>, %arg1: i32) kernel {
    gpu.return
  }
}

// CHECK-LABEL: func @test_gpu_launch_1d
// CHECK-SAME: (%[[ARG:.*]]: !llvm.ptr)
func.func @test_gpu_launch_1d(%arg0: memref<100xf32>) {
  %c1 = arith.constant 1 : index
  // CHECK: gpu.launch_func @kernels::@gpu_kernel_1d
  // CHECK-SAME: args(%[[ARG]] : !llvm.ptr)
  gpu.launch_func @kernels::@gpu_kernel_1d
    blocks in (%c1, %c1, %c1)
    threads in (%c1, %c1, %c1)
    args(%arg0 : memref<100xf32>)
  return
}

// CHECK-LABEL: func @test_gpu_launch_2d
// CHECK-SAME: (%[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr)
func.func @test_gpu_launch_2d(%arg0: memref<10x20xf32>, %arg1: memref<10x20xf32>) {
  %c1 = arith.constant 1 : index
  // CHECK: gpu.launch_func @kernels::@gpu_kernel_2d
  // CHECK-SAME: args(%[[ARG0]] : !llvm.ptr, %[[ARG1]] : !llvm.ptr)
  gpu.launch_func @kernels::@gpu_kernel_2d
    blocks in (%c1, %c1, %c1)
    threads in (%c1, %c1, %c1)
    args(%arg0 : memref<10x20xf32>, %arg1 : memref<10x20xf32>)
  return
}

// CHECK-LABEL: func @test_gpu_launch_mixed
// CHECK-SAME: (%[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: i32)
func.func @test_gpu_launch_mixed(%arg0: memref<100xf32>, %arg1: i32) {
  %c1 = arith.constant 1 : index
  // CHECK: gpu.launch_func @kernels::@gpu_kernel_mixed
  // CHECK-SAME: args(%[[ARG0]] : !llvm.ptr, %[[ARG1]] : i32)
  gpu.launch_func @kernels::@gpu_kernel_mixed
    blocks in (%c1, %c1, %c1)
    threads in (%c1, %c1, %c1)
    args(%arg0 : memref<100xf32>, %arg1 : i32)
  return
}
}
