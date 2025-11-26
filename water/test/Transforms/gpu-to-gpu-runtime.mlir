// RUN: water-opt %s --water-gpu-to-gpu-runtime | FileCheck %s

module attributes {gpu.container_module} {
  // CHECK-DAG: llvm.mlir.global internal constant @[[KERNEL_DATA:kernel_data[_0-9]*]]
  // CHECK-DAG: llvm.mlir.global internal @[[KERNEL_HANDLE:kernel_handle[_0-9]*]]
  // CHECK-DAG: llvm.mlir.global internal constant @[[KERNEL_NAME:my_kernel[_0-9]*]]

  gpu.binary @kernel_binary [
    #gpu.object<#rocdl.target, "\00\01\02\03">
  ]

  // CHECK-LABEL: llvm.func @test_launch
  // CHECK-SAME: (%[[STREAM:.*]]: !llvm.ptr, %[[ARG0:.*]]: f32, %[[ARG1:.*]]: i64)
  llvm.func @test_launch(%stream: !llvm.ptr, %arg0: f32, %arg1: i64) {
    %c128 = arith.constant 128 : i64
    %c256 = arith.constant 256 : i64
    %c1 = arith.constant 1 : i64

    // CHECK-DAG: %[[HANDLE_ADDR:.*]] = llvm.mlir.addressof @[[KERNEL_HANDLE]]
    // CHECK-DAG: %[[DATA_ADDR:.*]] = llvm.mlir.addressof @[[KERNEL_DATA]]
    // CHECK-DAG: %[[NAME_ADDR:.*]] = llvm.mlir.addressof @[[KERNEL_NAME]]

    // CHECK-DAG: %[[DATA_ADDR_GEP:.*]] = llvm.getelementptr %[[DATA_ADDR]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
    // CHECK-DAG: %[[NAME_ADDR_GEP:.*]] = llvm.getelementptr %[[NAME_ADDR]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i8>

    // CHECK: %[[DATA_SIZE:.*]] = llvm.mlir.constant(4 : i64) : i64

    // CHECK: %[[FUNC:.*]] = llvm.call @wave_load_kernel(%[[STREAM]], %[[HANDLE_ADDR]], %[[DATA_ADDR_GEP]], %[[DATA_SIZE]], %[[NAME_ADDR_GEP]])
    // CHECK-SAME: : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> !llvm.ptr

    // CHECK-DAG: %[[SHARED_MEM:.*]] = llvm.mlir.constant(0 : i32) : i32

    // CHECK-DAG: %[[ARG0_ALLOCA:.*]] = llvm.alloca %{{.*}} x f32
    // CHECK-DAG: llvm.store %[[ARG0]], %[[ARG0_ALLOCA]]

    // CHECK-DAG: %[[ARG1_ALLOCA:.*]] = llvm.alloca %{{.*}} x i64
    // CHECK-DAG: llvm.store %[[ARG1]], %[[ARG1_ALLOCA]]

    // CHECK-DAG: %[[ARGS_ARRAY:.*]] = llvm.mlir.poison : !llvm.array<2 x ptr>
    // CHECK-DAG: %[[ARGS_ARRAY_1:.*]] = llvm.insertvalue %[[ARG0_ALLOCA]], %[[ARGS_ARRAY]][0]
    // CHECK-DAG: %[[ARGS_ARRAY_2:.*]] = llvm.insertvalue %[[ARG1_ALLOCA]], %[[ARGS_ARRAY_1]][1]

    // CHECK: %[[ARGS_PTR:.*]] = llvm.alloca %{{.*}} x !llvm.ptr
    // CHECK: llvm.store %[[ARGS_ARRAY_2]], %[[ARGS_PTR]]

    // CHECK: %[[ARGS_COUNT:.*]] = llvm.mlir.constant(2 : i32) : i32

    // CHECK: llvm.call @wave_launch_kernel(%[[STREAM]], %[[FUNC]], %[[SHARED_MEM]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[ARGS_PTR]], %[[ARGS_COUNT]])
    // CHECK-SAME: : (!llvm.ptr, !llvm.ptr, i32, i64, i64, i64, i64, i64, i64, !llvm.ptr, i32) -> ()

    // CHECK-NOT: gpu.launch_func
    gpu.launch_func @kernel_binary::@my_kernel
      blocks in (%c128, %c1, %c1)
      threads in (%c256, %c1, %c1) : i64
      args(%arg0: f32, %arg1: i64)

    llvm.return
  }

  // CHECK-NOT: gpu.binary
}
