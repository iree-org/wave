// RUN: water-opt %s --allow-unregistered-dialect

module attributes {transform.with_named_sequence} {
  "stream.executable"() <{sym_name = "gemm", sym_visibility = "private"}> ({
    "stream.executable.export"() <{function_ref = @gemm, sym_name = "gemm", sym_visibility = "public"}> ({
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      "stream.return"(%c1, %c2, %c1) : (index, index, index) -> ()
    }) : () -> ()
    builtin.module {
      func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign"}}>} {
        %cst = arith.constant dense<0.000000e+00> : vector<8xbf16>
        %cst_0 = arith.constant dense<511> : vector<8xindex>
        %cst_1 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xindex>
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c16 = arith.constant 16 : index
        %c4608 = arith.constant 4608 : index
        %c0 = arith.constant 0 : index
        %c2_2 = arith.constant 2 : index
        %cst_3 = arith.constant dense<0.000000e+00> : vector<16xf32>
        %block_id_y = gpu.block_id  y upper_bound 2
        %thread_id_x = gpu.thread_id  x upper_bound 128
        %thread_id_y = gpu.thread_id  y upper_bound 2
        %alloc = memref.alloc() : memref<9216xi8, #gpu.address_space<workgroup>>
        %view = memref.view %alloc[%c0][] : memref<9216xi8, #gpu.address_space<workgroup>> to memref<64x36xbf16, #gpu.address_space<workgroup>>
        %view_4 = memref.view %alloc[%c4608][] : memref<9216xi8, #gpu.address_space<workgroup>> to memref<64x36xbf16, #gpu.address_space<workgroup>>
        %0 = "stream.binding.subspan"(%arg1, %c0) : (!stream.binding, index) -> memref<128x511xbf16, strided<[511, 1], offset: ?>>
        %1 = "stream.binding.subspan"(%arg0, %c0) : (!stream.binding, index) -> memref<64x511xbf16, strided<[511, 1], offset: ?>>
        %2 = affine.apply affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 4) mod 64)>()[%thread_id_x, %thread_id_y]
        %3 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x]
        %4 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 64 + s0 floordiv 4 - ((s1 * 32 + s0 floordiv 4) floordiv 64) * 64)>()[%thread_id_x, %thread_id_y, %block_id_y]
        %5 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 32) * 32)>()[%thread_id_x, %thread_id_y]
        %6 = affine.apply affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8)>()[%thread_id_x]
        %7 = affine.apply affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8 + 16)>()[%thread_id_x]
        %8 = affine.apply affine_map<()[s0] -> (s0 mod 32 + (s0 floordiv 64) * 32)>()[%thread_id_x]
        %9 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x]
        %10 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32 + 32)>()[%thread_id_x]
        %11 = vector.load %1[%2, %10] : memref<64x511xbf16, strided<[511, 1], offset: ?>>, vector<8xbf16>
        %12 = vector.load %0[%4, %10] : memref<128x511xbf16, strided<[511, 1], offset: ?>>, vector<8xbf16>
        %13 = vector.load %1[%2, %9] : memref<64x511xbf16, strided<[511, 1], offset: ?>>, vector<8xbf16>
        %14 = vector.load %0[%4, %9] : memref<128x511xbf16, strided<[511, 1], offset: ?>>, vector<8xbf16>
        %15:5 = scf.for %arg3 = %c0 to %c16 step %c1 iter_args(%arg4 = %cst_3, %arg5 = %11, %arg6 = %12, %arg7 = %13, %arg8 = %14) -> (vector<16xf32>, vector<8xbf16>, vector<8xbf16>, vector<8xbf16>, vector<8xbf16>) {
          %50 = arith.addi %arg3, %c2_2 : index
          %51 = affine.apply affine_map<()[s0, s1] -> (s0 * 32 + s1 * 8 - (s1 floordiv 4) * 32)>()[%50, %thread_id_x]
          %52 = vector.broadcast %51 : index to vector<8xindex>
          %53 = arith.addi %52, %cst_1 overflow<nsw, nuw> : vector<8xindex>
          %54 = arith.cmpi slt, %53, %cst_0 : vector<8xindex>
          %55 = vector.maskedload %1[%2, %51], %54, %cst : memref<64x511xbf16, strided<[511, 1], offset: ?>>, vector<8xi1>, vector<8xbf16> into vector<8xbf16>
          %56 = vector.maskedload %0[%4, %51], %54, %cst : memref<128x511xbf16, strided<[511, 1], offset: ?>>, vector<8xi1>, vector<8xbf16> into vector<8xbf16>
          vector.store %arg5, %view_4[%2, %3] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
          vector.store %arg6, %view[%2, %3] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
          amdgpu.lds_barrier
          %57 = vector.load %view[%5, %6] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
          %58 = vector.load %view[%5, %7] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
          %59 = vector.load %view_4[%8, %6] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
          %60 = vector.load %view_4[%8, %7] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
          %61 = amdgpu.mfma %59 * %57 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
          %62 = amdgpu.mfma %60 * %58 + %61 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
          scf.yield %62, %arg7, %arg8, %55, %56 : vector<16xf32>, vector<8xbf16>, vector<8xbf16>, vector<8xbf16>, vector<8xbf16>
        }
        %16 = vector.extract_strided_slice %15#0 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %17 = "stream.binding.subspan"(%arg2, %c0) : (!stream.binding, index) -> memref<64x128xf32, strided<[128, 1], offset: ?>>
        %18 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4)>()[%thread_id_x]
        %19 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * 64 + s2 * 32 - (s0 floordiv 32) * 32)>()[%thread_id_x, %block_id_y, %thread_id_y]
        vector.store %16, %17[%18, %19] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %20 = vector.extract_strided_slice %15#0 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %21 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 1)>()[%thread_id_x]
        vector.store %20, %17[%21, %19] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %22 = vector.extract_strided_slice %15#0 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %23 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 2)>()[%thread_id_x]
        vector.store %22, %17[%23, %19] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %24 = vector.extract_strided_slice %15#0 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %25 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 3)>()[%thread_id_x]
        vector.store %24, %17[%25, %19] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %26 = vector.extract_strided_slice %15#0 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %27 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 8)>()[%thread_id_x]
        vector.store %26, %17[%27, %19] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %28 = vector.extract_strided_slice %15#0 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %29 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 9)>()[%thread_id_x]
        vector.store %28, %17[%29, %19] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %30 = vector.extract_strided_slice %15#0 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %31 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 10)>()[%thread_id_x]
        vector.store %30, %17[%31, %19] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %32 = vector.extract_strided_slice %15#0 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %33 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 11)>()[%thread_id_x]
        vector.store %32, %17[%33, %19] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %34 = vector.extract_strided_slice %15#0 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %35 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 16)>()[%thread_id_x]
        vector.store %34, %17[%35, %19] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %36 = vector.extract_strided_slice %15#0 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %37 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 17)>()[%thread_id_x]
        vector.store %36, %17[%37, %19] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %38 = vector.extract_strided_slice %15#0 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %39 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 18)>()[%thread_id_x]
        vector.store %38, %17[%39, %19] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %40 = vector.extract_strided_slice %15#0 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %41 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 19)>()[%thread_id_x]
        vector.store %40, %17[%41, %19] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %42 = vector.extract_strided_slice %15#0 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %43 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 24)>()[%thread_id_x]
        vector.store %42, %17[%43, %19] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %44 = vector.extract_strided_slice %15#0 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %45 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 25)>()[%thread_id_x]
        vector.store %44, %17[%45, %19] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %46 = vector.extract_strided_slice %15#0 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %47 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 26)>()[%thread_id_x]
        vector.store %46, %17[%47, %19] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %48 = vector.extract_strided_slice %15#0 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %49 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 27)>()[%thread_id_x]
        vector.store %48, %17[%49, %19] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        return
      }
    }
    "stream.executable.end"() : () -> ()
  }) : () -> ()
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
    %0 = "hal.tensor.import"(%arg0, %arg3) <{operandSegmentSizes = array<i32: 1, 0, 1>, target_encoding = tensor<64x511xbf16>}> : (!hal.buffer_view, !hal.fence) -> tensor<64x511xbf16>
    %1 = "hal.tensor.import"(%arg1, %arg3) <{operandSegmentSizes = array<i32: 1, 0, 1>, target_encoding = tensor<128x511xbf16>}> : (!hal.buffer_view, !hal.fence) -> tensor<128x511xbf16>
    %2 = "hal.tensor.import"(%arg2, %arg3) <{operandSegmentSizes = array<i32: 1, 0, 1>, target_encoding = tensor<64x128xf32>}> : (!hal.buffer_view, !hal.fence) -> tensor<64x128xf32>
    %3 = "flow.dispatch"(%0, %1, %2) <{entry_points = [@gemm::@gemm], operandSegmentSizes = array<i32: 0, 3, 0, 0>, tied_operands = [2 : index]}> : (tensor<64x511xbf16>, tensor<128x511xbf16>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %4 = "hal.tensor.barrier"(%3, %arg4) : (tensor<64x128xf32>, !hal.fence) -> tensor<64x128xf32>
    %5 = "hal.tensor.export"(%4) <{source_encoding = tensor<64x128xf32>}> : (tensor<64x128xf32>) -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
