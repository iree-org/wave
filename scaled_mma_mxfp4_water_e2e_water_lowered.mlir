#map = affine_map<()[s0, s1, s2] -> (s0 * 32 + s2 * 16 + s1 floordiv 8 - ((s2 * 16 + s1 floordiv 8) floordiv 32) * 32)>
#map1 = affine_map<()[s0, s1] -> (s0 * 16 + s1 * 128 - (s0 floordiv 8) * 128)>
#map2 = affine_map<()[s0, s1] -> ((s1 * 16 + s0 floordiv 8) mod 32)>
#map3 = affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 8) * 128)>
#map4 = affine_map<()[s0, s1] -> (s0 + s1 * 8 - (s0 floordiv 8) * 8)>
#map5 = affine_map<()[s0] -> (s0 mod 8)>
#map6 = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 16) * 16)>
#map7 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
#map8 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16)>
#map9 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16 + 64)>
#map10 = affine_map<()[s0] -> (s0 + (s0 floordiv 64) * 16 - (s0 floordiv 16) * 16)>
#map11 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4)>
#map12 = affine_map<()[s0, s1, s2] -> (s0 * 32 + s1 + s2 * 16 - (s1 floordiv 16) * 16)>
#map13 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 1)>
#map14 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 2)>
#map15 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 3)>
module {
  func.func @kernel(%arg0: memref<10240x5120xi8, #gpu.address_space<global>>, %arg1: memref<10240x320xi8, #gpu.address_space<global>>, %arg2: memref<10240x5120xi8, #gpu.address_space<global>>, %arg3: memref<10240x320xi8, #gpu.address_space<global>>, %arg4: memref<10240x10240xf32, #gpu.address_space<global>>) {
    %c1 = arith.constant 1 : index
    %c40 = arith.constant 40 : index
    %c4352 = arith.constant 4352 : index
    %c9216 = arith.constant 9216 : index
    %c0 = arith.constant 0 : index
    %c8704 = arith.constant 8704 : index
    %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
    %alloc = memref.alloc() : memref<9728xi8, #gpu.address_space<workgroup>>
    %view = memref.view %alloc[%c8704][] : memref<9728xi8, #gpu.address_space<workgroup>> to memref<32x16xi8, #gpu.address_space<workgroup>>
    %view_0 = memref.view %alloc[%c0][] : memref<9728xi8, #gpu.address_space<workgroup>> to memref<32x136xi8, #gpu.address_space<workgroup>>
    %view_1 = memref.view %alloc[%c9216][] : memref<9728xi8, #gpu.address_space<workgroup>> to memref<32x16xi8, #gpu.address_space<workgroup>>
    %view_2 = memref.view %alloc[%c4352][] : memref<9728xi8, #gpu.address_space<workgroup>> to memref<32x136xi8, #gpu.address_space<workgroup>>
    %0 = scf.for %arg5 = %c0 to %c40 step %c1 iter_args(%arg6 = %cst) -> (vector<4xf32>) {
      %block_id_x_3 = gpu.block_id x
      %thread_id_x_4 = gpu.thread_id x
      %thread_id_y_5 = gpu.thread_id y
      %10 = affine.apply #map()[%block_id_x_3, %thread_id_x_4, %thread_id_y_5]
      %11 = affine.apply #map1()[%thread_id_x_4, %arg5]
      %12 = vector.load %arg0[%10, %11] : memref<10240x5120xi8, #gpu.address_space<global>>, vector<16xi8>
      amdgpu.lds_barrier
      %13 = affine.apply #map2()[%thread_id_x_4, %thread_id_y_5]
      %14 = affine.apply #map3()[%thread_id_x_4]
      vector.store %12, %view_2[%13, %14] : memref<32x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
      %15 = affine.apply #map4()[%thread_id_x_4, %arg5]
      %16 = vector.load %arg1[%10, %15] : memref<10240x320xi8, #gpu.address_space<global>>, vector<1xi8>
      %17 = affine.apply #map5()[%thread_id_x_4]
      vector.store %16, %view_1[%13, %17] : memref<32x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
      %block_id_y_6 = gpu.block_id y
      %18 = affine.apply #map()[%block_id_y_6, %thread_id_x_4, %thread_id_y_5]
      %19 = vector.load %arg2[%18, %11] : memref<10240x5120xi8, #gpu.address_space<global>>, vector<16xi8>
      vector.store %19, %view_0[%13, %14] : memref<32x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
      %20 = vector.load %arg3[%18, %15] : memref<10240x320xi8, #gpu.address_space<global>>, vector<1xi8>
      vector.store %20, %view[%13, %17] : memref<32x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
      amdgpu.lds_barrier
      %21 = affine.apply #map6()[%thread_id_x_4, %thread_id_y_5]
      %22 = affine.apply #map7()[%thread_id_x_4]
      %23 = vector.load %view[%21, %22] : memref<32x16xi8, #gpu.address_space<workgroup>>, vector<8xi8>
      %24 = vector.extract_strided_slice %23 {offsets = [0], sizes = [1], strides = [1]} : vector<8xi8> to vector<1xi8>
      %25 = vector.extract_strided_slice %23 {offsets = [4], sizes = [1], strides = [1]} : vector<8xi8> to vector<1xi8>
      %26 = affine.apply #map8()[%thread_id_x_4]
      %27 = vector.load %view_0[%21, %26] : memref<32x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
      %28 = affine.apply #map9()[%thread_id_x_4]
      %29 = vector.load %view_0[%21, %28] : memref<32x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
      %30 = affine.apply #map10()[%thread_id_x_4]
      %31 = vector.load %view_1[%30, %22] : memref<32x16xi8, #gpu.address_space<workgroup>>, vector<8xi8>
      %32 = vector.extract_strided_slice %31 {offsets = [0], sizes = [1], strides = [1]} : vector<8xi8> to vector<1xi8>
      %33 = vector.extract_strided_slice %31 {offsets = [4], sizes = [1], strides = [1]} : vector<8xi8> to vector<1xi8>
      %34 = vector.load %view_2[%30, %26] : memref<32x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
      %35 = vector.load %view_2[%30, %28] : memref<32x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
      %36 = vector.bitcast %34 : vector<16xi8> to vector<32xf4E2M1FN>
      %37 = vector.bitcast %35 : vector<16xi8> to vector<32xf4E2M1FN>
      %38 = vector.bitcast %32 : vector<1xi8> to vector<1xf8E8M0FNU>
      %39 = vector.bitcast %33 : vector<1xi8> to vector<1xf8E8M0FNU>
      %40 = vector.bitcast %27 : vector<16xi8> to vector<32xf4E2M1FN>
      %41 = vector.bitcast %29 : vector<16xi8> to vector<32xf4E2M1FN>
      %42 = vector.bitcast %24 : vector<1xi8> to vector<1xf8E8M0FNU>
      %43 = vector.bitcast %25 : vector<1xi8> to vector<1xf8E8M0FNU>
      %44 = vector.extract %38[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
      %45 = vector.extract %42[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
      %46 = amdgpu.scaled_mfma 16x16x128 (%44[0] * %36) * (%45[0] * %40) + %arg6 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
      %47 = vector.extract %39[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
      %48 = vector.extract %43[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
      %49 = amdgpu.scaled_mfma 16x16x128 (%47[0] * %37) * (%48[0] * %41) + %46 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
      scf.yield %49 : vector<4xf32>
    }
    %1 = vector.extract_strided_slice %0 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %block_id_x = gpu.block_id x
    %thread_id_x = gpu.thread_id x
    %2 = affine.apply #map11()[%block_id_x, %thread_id_x]
    %block_id_y = gpu.block_id y
    %thread_id_y = gpu.thread_id y
    %3 = affine.apply #map12()[%block_id_y, %thread_id_x, %thread_id_y]
    vector.store %1, %arg4[%2, %3] : memref<10240x10240xf32, #gpu.address_space<global>>, vector<1xf32>
    %4 = vector.extract_strided_slice %0 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %5 = affine.apply #map13()[%block_id_x, %thread_id_x]
    vector.store %4, %arg4[%5, %3] : memref<10240x10240xf32, #gpu.address_space<global>>, vector<1xf32>
    %6 = vector.extract_strided_slice %0 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %7 = affine.apply #map14()[%block_id_x, %thread_id_x]
    vector.store %6, %arg4[%7, %3] : memref<10240x10240xf32, #gpu.address_space<global>>, vector<1xf32>
    %8 = vector.extract_strided_slice %0 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %9 = affine.apply #map15()[%block_id_x, %thread_id_x]
    vector.store %8, %arg4[%9, %3] : memref<10240x10240xf32, #gpu.address_space<global>>, vector<1xf32>
    return
  }
}

