// RUN: water-opt %s --water-gpu-to-gpu-runtime --split-input-file --verify-diagnostics | FileCheck %s

module attributes {gpu.container_module} {
  // CHECK: llvm.mlir.global internal constant @[[KERNEL_DATA:my_kernel_data[_0-9]*]]
  // CHECK: llvm.mlir.global internal constant @[[KERNEL_NAME:my_kernel[_0-9]*]]
  // CHECK: llvm.mlir.global internal @[[KERNEL_HANDLE:my_kernel_handle[_0-9]*]]

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

    // CHECK: %[[ARGS_PTR:.*]] = llvm.alloca %{{.*}} x !llvm.array<2 x ptr>
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

// -----

module attributes {gpu.container_module} {
  // CHECK: llvm.mlir.global internal constant @[[KERNEL3_DATA:kernel_a_data[_0-9]*]]
  // CHECK: llvm.mlir.global internal constant @[[KERNEL3_NAME:kernel_a[_0-9]*]]
  // CHECK: llvm.mlir.global internal @[[KERNEL3_HANDLE:kernel_a_handle[_0-9]*]]

  // CHECK: llvm.mlir.global internal constant @[[KERNEL2_DATA:kernel_b_data[_0-9]*]]
  // CHECK: llvm.mlir.global internal constant @[[KERNEL2_NAME:kernel_b[_0-9]*]]
  // CHECK: llvm.mlir.global internal @[[KERNEL2_HANDLE:kernel_b_handle[_0-9]*]]

  // CHECK: llvm.mlir.global internal constant @[[KERNEL1_DATA:kernel_a_data[_0-9]*]]
  // CHECK: llvm.mlir.global internal constant @[[KERNEL1_NAME:kernel_a[_0-9]*]]
  // CHECK: llvm.mlir.global internal @[[KERNEL1_HANDLE:kernel_a_handle[_0-9]*]]

  gpu.binary @kernel_binary_a [
    #gpu.object<#rocdl.target, "\00\01\02\03\04\05">
  ]

  gpu.binary @kernel_binary_b [
    #gpu.object<#rocdl.target, "\10\11\12\13">
  ]

  // CHECK-LABEL: llvm.func @test_multiple_launches
  // CHECK-SAME: (%[[STREAM:.*]]: !llvm.ptr, %[[ARG0:.*]]: f32, %[[ARG1:.*]]: i64, %[[ARG2:.*]]: i32)
  llvm.func @test_multiple_launches(%stream: !llvm.ptr, %arg0: f32, %arg1: i64, %arg2: i32) {
    %c64 = arith.constant 64 : i64
    %c128 = arith.constant 128 : i64
    %c256 = arith.constant 256 : i64
    %c512 = arith.constant 512 : i64
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64

    // First launch
    // CHECK-DAG: %[[HANDLE1_ADDR:.*]] = llvm.mlir.addressof @[[KERNEL1_HANDLE]]
    // CHECK-DAG: %[[NAME1_ADDR:.*]] = llvm.mlir.addressof @[[KERNEL1_NAME]]
    // CHECK-DAG: %[[DATA1_ADDR:.*]] = llvm.mlir.addressof @[[KERNEL1_DATA]]

    // CHECK-DAG: %[[DATA1_ADDR_GEP:.*]] = llvm.getelementptr %[[DATA1_ADDR]][0, 0]
    // CHECK-DAG: %[[NAME1_ADDR_GEP:.*]] = llvm.getelementptr %[[NAME1_ADDR]][0, 0]

    // CHECK: %[[DATA1_SIZE:.*]] = llvm.mlir.constant(6 : i64) : i64

    // CHECK: %[[FUNC1:.*]] = llvm.call @wave_load_kernel(%[[STREAM]], %[[HANDLE1_ADDR]], %[[DATA1_ADDR_GEP]], %[[DATA1_SIZE]], %[[NAME1_ADDR_GEP]])
    // CHECK-SAME: : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> !llvm.ptr

    // CHECK: %[[SHARED_MEM1:.*]] = llvm.mlir.constant(0 : i32) : i32

    // CHECK: llvm.call @wave_launch_kernel(%[[STREAM]], %[[FUNC1]], %[[SHARED_MEM1]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})

    // CHECK-NOT: gpu.launch_func @kernel_binary_a
    gpu.launch_func @kernel_binary_a::@kernel_a
      blocks in (%c128, %c1, %c1)
      threads in (%c256, %c1, %c1) : i64
      args(%arg0: f32, %arg1: i64)

    // Second launch
    // CHECK-DAG: %[[HANDLE2_ADDR:.*]] = llvm.mlir.addressof @[[KERNEL2_HANDLE]]
    // CHECK-DAG: %[[DATA2_ADDR:.*]] = llvm.mlir.addressof @[[KERNEL2_DATA]]
    // CHECK-DAG: %[[NAME2_ADDR:.*]] = llvm.mlir.addressof @[[KERNEL2_NAME]]

    // CHECK-DAG: %[[DATA2_ADDR_GEP:.*]] = llvm.getelementptr %[[DATA2_ADDR]][0, 0]
    // CHECK-DAG: %[[NAME2_ADDR_GEP:.*]] = llvm.getelementptr %[[NAME2_ADDR]][0, 0]

    // CHECK: %[[DATA2_SIZE:.*]] = llvm.mlir.constant(4 : i64) : i64

    // CHECK: %[[FUNC2:.*]] = llvm.call @wave_load_kernel(%[[STREAM]], %[[HANDLE2_ADDR]], %[[DATA2_ADDR_GEP]], %[[DATA2_SIZE]], %[[NAME2_ADDR_GEP]])
    // CHECK-SAME: : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> !llvm.ptr

    // CHECK: %[[SHARED_MEM2:.*]] = llvm.mlir.constant(0 : i32) : i32

    // CHECK: llvm.call @wave_launch_kernel(%[[STREAM]], %[[FUNC2]], %[[SHARED_MEM2]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})

    // CHECK-NOT: gpu.launch_func @kernel_binary_b
    gpu.launch_func @kernel_binary_b::@kernel_b
      blocks in (%c64, %c2, %c1)
      threads in (%c512, %c1, %c1) : i64
      args(%arg2: i32)

    // Third launch
    // CHECK-DAG: %[[HANDLE3_ADDR:.*]] = llvm.mlir.addressof @[[KERNEL3_HANDLE]]
    // CHECK-DAG: %[[DATA3_ADDR:.*]] = llvm.mlir.addressof @[[KERNEL3_DATA]]
    // CHECK-DAG: %[[NAME3_ADDR:.*]] = llvm.mlir.addressof @[[KERNEL3_NAME]]

    // CHECK-DAG: %[[DATA3_ADDR_GEP:.*]] = llvm.getelementptr %[[DATA3_ADDR]][0, 0]
    // CHECK-DAG: %[[NAME3_ADDR_GEP:.*]] = llvm.getelementptr %[[NAME3_ADDR]][0, 0]

    // CHECK: %[[DATA3_SIZE:.*]] = llvm.mlir.constant(6 : i64) : i64

    // CHECK: %[[FUNC3:.*]] = llvm.call @wave_load_kernel(%[[STREAM]], %[[HANDLE3_ADDR]], %[[DATA3_ADDR_GEP]], %[[DATA3_SIZE]], %[[NAME3_ADDR_GEP]])
    // CHECK-SAME: : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> !llvm.ptr

    // CHECK: %[[SHARED_MEM3:.*]] = llvm.mlir.constant(0 : i32) : i32

    // CHECK: llvm.call @wave_launch_kernel(%[[STREAM]], %[[FUNC3]], %[[SHARED_MEM3]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})

    // CHECK-NOT: gpu.launch_func @kernel_binary_a
    gpu.launch_func @kernel_binary_a::@kernel_a
      blocks in (%c256, %c1, %c1)
      threads in (%c128, %c1, %c1) : i64
      args(%arg1: i64, %arg2: i32)

    llvm.return
  }

  // CHECK-NOT: gpu.binary
}

// -----

module attributes {gpu.container_module} {
  gpu.binary @kernel_binary [
    #gpu.object<#rocdl.target, "\00\01\02\03">
  ]

  llvm.func @test_no_arguments() {
    %c1 = arith.constant 1 : i64

    // expected-error @+1 {{func op must have at least one argument}}
    gpu.launch_func @kernel_binary::@my_kernel
      blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) : i64

    llvm.return
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.binary @kernel_binary [
    #gpu.object<#rocdl.target, "\00\01\02\03">
  ]

  llvm.func @test_non_pointer_stream(%stream: i64) {
    %c1 = arith.constant 1 : i64

    // expected-error @+1 {{stream argument must be a pointer}}
    gpu.launch_func @kernel_binary::@my_kernel
      blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) : i64

    llvm.return
  }
}
