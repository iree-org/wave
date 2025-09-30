#map = affine_map<()[s0] -> (s0 * 32)>
#map1 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 32)>
#map2 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 32 + 16)>
#map3 = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 16) * 16)>
#map4 = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 16) * 16 + 16)>
#map5 = affine_map<()[s0, s1] -> (s0 * 32 + ((s1 mod 64) floordiv 16) * 4)>
#map6 = affine_map<()[s0] -> (s0 * 32 + 32)>
#map7 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4)>
#map8 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 1)>
#map9 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 2)>
#map10 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 3)>
#map11 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 16)>
#map12 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17)>
#map13 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 18)>
#map14 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19)>
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @gemm {
    stream.executable.export public @gemm workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      stream.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding, %arg5: i32) attributes {translation_info = #translation} {
        %cst = arith.constant dense<0.000000e+00> : vector<4xf16>
        %cst_0 = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
        %c4 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        %c32 = arith.constant 32 : index
        %c0 = arith.constant 0 : index
        %cst_1 = arith.constant dense<0.000000e+00> : vector<4xf32>
        %thread_id_x = gpu.thread_id  x upper_bound 128
        %thread_id_y = gpu.thread_id  y upper_bound 2
        %0 = arith.index_cast %arg5 : i32 to index
        %1 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<64x128xf16, strided<[128, 1], offset: ?>>
        %2 = arith.cmpi slt, %thread_id_x, %c32 : index
        scf.if %2 {
          %36 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<32xi32, strided<[1], offset: ?>>
          %37 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<64x128xf16, strided<[128, 1], offset: ?>>
          %38 = vector.load %36[%thread_id_x] : memref<32xi32, strided<[1], offset: ?>>, vector<1xi32>
          %39 = vector.extract %38[0] : i32 from vector<1xi32>
          %40 = arith.index_cast %39 : i32 to index
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %41 = affine.apply #map()[%arg6]
            %42 = vector.load %37[%40, %41] : memref<64x128xf16, strided<[128, 1], offset: ?>>, vector<1xf16>
            vector.store %42, %1[%thread_id_x, %41] : memref<64x128xf16, strided<[128, 1], offset: ?>>, vector<1xf16>
            vector.store %42, %1[%thread_id_x, %41] : memref<64x128xf16, strided<[128, 1], offset: ?>>, vector<1xf16>
          }
        }
        %3 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x64x128xf16, strided<[8192, 128, 1], offset: ?>>
        %4 = affine.apply #map1()[%thread_id_x]
        %5 = affine.apply #map2()[%thread_id_x]
        %6 = affine.apply #map3()[%thread_id_x, %thread_id_y]
        %7 = affine.apply #map4()[%thread_id_x, %thread_id_y]
        %8:4 = scf.for %arg6 = %c0 to %c4 step %c1 iter_args(%arg7 = %cst_1, %arg8 = %cst_1, %arg9 = %cst_1, %arg10 = %cst_1) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          %36 = affine.apply #map5()[%arg6, %thread_id_x]
          %37 = vector.broadcast %36 : index to vector<4xindex>
          %38 = arith.addi %37, %cst_0 overflow<nsw, nuw> : vector<4xindex>
          %39 = affine.apply #map6()[%arg6]
          %40 = vector.broadcast %39 : index to vector<4xindex>
          %41 = arith.cmpi slt, %38, %40 : vector<4xindex>
          %42 = vector.maskedload %1[%4, %36], %41, %cst : memref<64x128xf16, strided<[128, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %43 = vector.maskedload %1[%5, %36], %41, %cst : memref<64x128xf16, strided<[128, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %44 = vector.maskedload %3[%0, %6, %36], %41, %cst : memref<8x64x128xf16, strided<[8192, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %45 = vector.maskedload %3[%0, %7, %36], %41, %cst : memref<8x64x128xf16, strided<[8192, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %46 = amdgpu.mfma %42 * %44 + %arg7 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %47 = amdgpu.mfma %42 * %45 + %arg8 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %48 = amdgpu.mfma %43 * %44 + %arg9 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %49 = amdgpu.mfma %43 * %45 + %arg10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          scf.yield %46, %47, %48, %49 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        %9 = vector.extract_strided_slice %8#0 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %10 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<64x64xf32, strided<[64, 1], offset: ?>>
        %11 = affine.apply #map7()[%thread_id_x]
        %12 = affine.apply #map3()[%thread_id_x, %thread_id_y]
        vector.store %9, %10[%11, %12] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
        %13 = vector.extract_strided_slice %8#0 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %14 = affine.apply #map8()[%thread_id_x]
        vector.store %13, %10[%14, %12] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
        %15 = vector.extract_strided_slice %8#0 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %16 = affine.apply #map9()[%thread_id_x]
        vector.store %15, %10[%16, %12] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
        %17 = vector.extract_strided_slice %8#0 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %18 = affine.apply #map10()[%thread_id_x]
        vector.store %17, %10[%18, %12] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
        %19 = vector.extract_strided_slice %8#1 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %20 = affine.apply #map4()[%thread_id_x, %thread_id_y]
        vector.store %19, %10[%11, %20] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
        %21 = vector.extract_strided_slice %8#1 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %21, %10[%14, %20] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
        %22 = vector.extract_strided_slice %8#1 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %22, %10[%16, %20] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
        %23 = vector.extract_strided_slice %8#1 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %23, %10[%18, %20] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
        %24 = vector.extract_strided_slice %8#2 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %25 = affine.apply #map11()[%thread_id_x]
        vector.store %24, %10[%25, %12] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
        %26 = vector.extract_strided_slice %8#2 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %27 = affine.apply #map12()[%thread_id_x]
        vector.store %26, %10[%27, %12] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
        %28 = vector.extract_strided_slice %8#2 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %29 = affine.apply #map13()[%thread_id_x]
        vector.store %28, %10[%29, %12] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
        %30 = vector.extract_strided_slice %8#2 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %31 = affine.apply #map14()[%thread_id_x]
        vector.store %30, %10[%31, %12] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
        %32 = vector.extract_strided_slice %8#3 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %32, %10[%25, %20] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
        %33 = vector.extract_strided_slice %8#3 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %33, %10[%27, %20] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
        %34 = vector.extract_strided_slice %8#3 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %34, %10[%29, %20] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
        %35 = vector.extract_strided_slice %8#3 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %35, %10[%31, %20] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: i32, %arg6: !hal.fence, %arg7: !hal.fence) -> (!hal.buffer_view, !hal.buffer_view) {
    %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<64x128xf16>
    %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x64x128xf16>
    %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<32xi32>
    %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<64x128xf16>
    %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<64x64xf32>
    %5:2 = flow.dispatch @gemm::@gemm(%0, %1, %2, %3, %4, %arg5) : (tensor<64x128xf16>, tensor<8x64x128xf16>, tensor<32xi32>, tensor<64x128xf16>, tensor<64x64xf32>, i32) -> (%3, %4)
    %6:2 = hal.tensor.barrier join(%5#0, %5#1 : tensor<64x128xf16>, tensor<64x64xf32>) => %arg7 : !hal.fence
    %7 = hal.tensor.export %6#0 : tensor<64x128xf16> -> !hal.buffer_view
    %8 = hal.tensor.export %6#1 : tensor<64x64xf32> -> !hal.buffer_view
    return %7, %8 : !hal.buffer_view, !hal.buffer_view
  }
}
