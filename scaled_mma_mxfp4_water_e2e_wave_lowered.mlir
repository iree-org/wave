#map = affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 32 + s0 floordiv 8 - ((s1 * 16 + s0 floordiv 8) floordiv 32) * 32)>
#map1 = affine_map<()[s0, s1] -> ((s1 * 16 + s0 floordiv 8) mod 32)>
#map2 = affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 8) * 128)>
#map3 = affine_map<()[s0] -> (s0 mod 8)>
#map4 = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 16) * 16)>
#map5 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
#map6 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16)>
#map7 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16 + 64)>
#map8 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 16)>
#map9 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 - (s1 floordiv 8) * 128)>
#map10 = affine_map<()[s0, s1] -> (s0 + s1 * 8 - (s0 floordiv 8) * 8)>
#map11 = affine_map<()[s0] -> (s0 * 32)>
#map12 = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4)>
#map13 = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 1)>
#map14 = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 2)>
#map15 = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 3)>
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @scaled_gemm {
    stream.executable.export public @scaled_gemm workgroups() -> (index, index, index) {
      %c320 = arith.constant 320 : index
      %c1 = arith.constant 1 : index
      stream.return %c320, %c320, %c1 : index, index, index
    }
    builtin.module {
      func.func @scaled_gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
        %c10240 = arith.constant 10240 : index
        %c1 = arith.constant 1 : index
        %c40 = arith.constant 40 : index
        %c4352 = arith.constant 4352 : index
        %c9216 = arith.constant 9216 : index
        %c8704 = arith.constant 8704 : index
        %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
        %c0 = arith.constant 0 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<i8>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<i8>
        %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<i8>
        %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<i8>
        %4 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<f32>
        %block_id_x = gpu.block_id x upper_bound 320
        %block_id_y = gpu.block_id y upper_bound 320
        %thread_id_x = gpu.thread_id x upper_bound 128
        %thread_id_y = gpu.thread_id y upper_bound 2
        %reinterpret_cast = memref.reinterpret_cast %0 to offset: [0], sizes: [10240, 5120], strides: [5120, 1] : memref<i8> to memref<10240x5120xi8, strided<[5120, 1]>>
        %reinterpret_cast_0 = memref.reinterpret_cast %1 to offset: [0], sizes: [10240, 320], strides: [320, 1] : memref<i8> to memref<10240x320xi8, strided<[320, 1]>>
        %reinterpret_cast_1 = memref.reinterpret_cast %2 to offset: [0], sizes: [10240, 5120], strides: [5120, 1] : memref<i8> to memref<10240x5120xi8, strided<[5120, 1]>>
        %reinterpret_cast_2 = memref.reinterpret_cast %3 to offset: [0], sizes: [10240, 320], strides: [320, 1] : memref<i8> to memref<10240x320xi8, strided<[320, 1]>>
        %alloc = memref.alloc() : memref<9728xi8, #gpu.address_space<workgroup>>
        %view = memref.view %alloc[%c8704][] : memref<9728xi8, #gpu.address_space<workgroup>> to memref<32x16xi8, #gpu.address_space<workgroup>>
        %view_3 = memref.view %alloc[%c0][] : memref<9728xi8, #gpu.address_space<workgroup>> to memref<32x136xi8, #gpu.address_space<workgroup>>
        %view_4 = memref.view %alloc[%c9216][] : memref<9728xi8, #gpu.address_space<workgroup>> to memref<32x16xi8, #gpu.address_space<workgroup>>
        %view_5 = memref.view %alloc[%c4352][] : memref<9728xi8, #gpu.address_space<workgroup>> to memref<32x136xi8, #gpu.address_space<workgroup>>
        %5 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_x]
        %6 = affine.apply #map1()[%thread_id_x, %thread_id_y]
        %7 = affine.apply #map2()[%thread_id_x]
        %8 = affine.apply #map3()[%thread_id_x]
        %9 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_y]
        %10 = affine.apply #map4()[%thread_id_x, %thread_id_y]
        %11 = affine.apply #map5()[%thread_id_x]
        %12 = affine.apply #map6()[%thread_id_x]
        %13 = affine.apply #map7()[%thread_id_x]
        %14 = affine.apply #map8()[%thread_id_x]
        %15 = scf.for %arg5 = %c0 to %c40 step %c1 iter_args(%arg6 = %cst) -> (vector<4xf32>) {
          %36 = affine.apply #map9()[%arg5, %thread_id_x]
          %37 = vector.load %reinterpret_cast[%5, %36] : memref<10240x5120xi8, strided<[5120, 1]>>, vector<16xi8>
          amdgpu.lds_barrier
          vector.store %37, %view_5[%6, %7] : memref<32x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %38 = affine.apply #map10()[%thread_id_x, %arg5]
          %39 = vector.load %reinterpret_cast_0[%5, %38] : memref<10240x320xi8, strided<[320, 1]>>, vector<1xi8>
          vector.store %39, %view_4[%6, %8] : memref<32x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %40 = vector.load %reinterpret_cast_1[%9, %36] : memref<10240x5120xi8, strided<[5120, 1]>>, vector<16xi8>
          vector.store %40, %view_3[%6, %7] : memref<32x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %41 = vector.load %reinterpret_cast_2[%9, %38] : memref<10240x320xi8, strided<[320, 1]>>, vector<1xi8>
          vector.store %41, %view[%6, %8] : memref<32x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          amdgpu.lds_barrier
          %42 = vector.load %view[%10, %11] : memref<32x16xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %43 = vector.extract_strided_slice %42 {offsets = [0], sizes = [1], strides = [1]} : vector<8xi8> to vector<1xi8>
          %44 = vector.extract_strided_slice %42 {offsets = [4], sizes = [1], strides = [1]} : vector<8xi8> to vector<1xi8>
          %45 = vector.load %view_3[%10, %12] : memref<32x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %46 = vector.load %view_3[%10, %13] : memref<32x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %47 = vector.load %view_4[%14, %11] : memref<32x16xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %48 = vector.extract_strided_slice %47 {offsets = [0], sizes = [1], strides = [1]} : vector<8xi8> to vector<1xi8>
          %49 = vector.extract_strided_slice %47 {offsets = [4], sizes = [1], strides = [1]} : vector<8xi8> to vector<1xi8>
          %50 = vector.load %view_5[%14, %12] : memref<32x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %51 = vector.load %view_5[%14, %13] : memref<32x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %52 = vector.bitcast %50 : vector<16xi8> to vector<32xf4E2M1FN>
          %53 = vector.bitcast %51 : vector<16xi8> to vector<32xf4E2M1FN>
          %54 = vector.bitcast %48 : vector<1xi8> to vector<1xf8E8M0FNU>
          %55 = vector.bitcast %49 : vector<1xi8> to vector<1xf8E8M0FNU>
          %56 = vector.bitcast %45 : vector<16xi8> to vector<32xf4E2M1FN>
          %57 = vector.bitcast %46 : vector<16xi8> to vector<32xf4E2M1FN>
          %58 = vector.bitcast %43 : vector<1xi8> to vector<1xf8E8M0FNU>
          %59 = vector.bitcast %44 : vector<1xi8> to vector<1xf8E8M0FNU>
          %60 = vector.extract %54[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %61 = vector.extract %58[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %62 = amdgpu.scaled_mfma 16x16x128 (%60[0] * %52) * (%61[0] * %56) + %arg6 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %63 = vector.extract %55[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %64 = vector.extract %59[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %65 = amdgpu.scaled_mfma 16x16x128 (%63[0] * %53) * (%64[0] * %57) + %62 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          scf.yield %65 : vector<4xf32>
        }
        %16 = vector.extract_strided_slice %15 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %17 = affine.apply #map11()[%block_id_x]
        %18 = affine.apply #map11()[%block_id_y]
        %19 = affine.apply #map12()[%thread_id_x]
        %20 = arith.muli %17, %c10240 overflow<nsw> : index
        %21 = arith.muli %19, %c10240 overflow<nsw> : index
        %22 = arith.addi %20, %18 overflow<nsw> : index
        %23 = arith.addi %21, %10 overflow<nsw> : index
        %reinterpret_cast_6 = memref.reinterpret_cast %4 to offset: [%22], sizes: [536870910], strides: [1] : memref<f32> to memref<536870910xf32, strided<[1], offset: ?>>
        vector.store %16, %reinterpret_cast_6[%23] : memref<536870910xf32, strided<[1], offset: ?>>, vector<1xf32>
        %24 = vector.extract_strided_slice %15 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %25 = affine.apply #map13()[%thread_id_x]
        %26 = arith.muli %25, %c10240 overflow<nsw> : index
        %27 = arith.addi %26, %10 overflow<nsw> : index
        vector.store %24, %reinterpret_cast_6[%27] : memref<536870910xf32, strided<[1], offset: ?>>, vector<1xf32>
        %28 = vector.extract_strided_slice %15 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %29 = affine.apply #map14()[%thread_id_x]
        %30 = arith.muli %29, %c10240 overflow<nsw> : index
        %31 = arith.addi %30, %10 overflow<nsw> : index
        vector.store %28, %reinterpret_cast_6[%31] : memref<536870910xf32, strided<[1], offset: ?>>, vector<1xf32>
        %32 = vector.extract_strided_slice %15 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %33 = affine.apply #map15()[%thread_id_x]
        %34 = arith.muli %33, %c10240 overflow<nsw> : index
        %35 = arith.addi %34, %10 overflow<nsw> : index
        vector.store %32, %reinterpret_cast_6[%35] : memref<536870910xf32, strided<[1], offset: ?>>, vector<1xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.fence, %arg6: !hal.fence) -> !hal.buffer_view {
    %0 = hal.tensor.import wait(%arg5) => %arg0 : !hal.buffer_view -> tensor<10240x5120xi8>
    %1 = hal.tensor.import wait(%arg5) => %arg1 : !hal.buffer_view -> tensor<10240x320xi8>
    %2 = hal.tensor.import wait(%arg5) => %arg2 : !hal.buffer_view -> tensor<10240x5120xi8>
    %3 = hal.tensor.import wait(%arg5) => %arg3 : !hal.buffer_view -> tensor<10240x320xi8>
    %4 = hal.tensor.import wait(%arg5) => %arg4 : !hal.buffer_view -> tensor<10240x10240xf32>
    %5 = flow.dispatch @scaled_gemm::@scaled_gemm(%0, %1, %2, %3, %4) : (tensor<10240x5120xi8>, tensor<10240x320xi8>, tensor<10240x5120xi8>, tensor<10240x320xi8>, tensor<10240x10240xf32>) -> %4
    %6 = hal.tensor.barrier join(%5 : tensor<10240x10240xf32>) => %arg6 : !hal.fence
    %7 = hal.tensor.export %6 : tensor<10240x10240xf32> -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
}
