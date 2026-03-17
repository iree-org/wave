#map = affine_map<()[s0] -> (s0 - (s0 floordiv 64) * 48)>
module attributes {gpu.container_module, transform.with_named_sequence} {
  gpu.module @gpu_module {
    gpu.func @read_write(%arg0: memref<f16> {llvm.inreg}, %arg1: memref<f16> {llvm.inreg}) kernel attributes {known_block_size = array<i32: 64, 1, 1>} {
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %thread_id_x = gpu.thread_id  x upper_bound 64
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<f16> to memref<16x16xf16, strided<[16, 1]>>
      %0 = affine.apply #map()[%thread_id_x]
      %1 = vector.load %reinterpret_cast[%0, %c0] : memref<16x16xf16, strided<[16, 1]>>, vector<16xf16>
      %2 = arith.muli %0, %c16 overflow<nsw> : index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1073741822], strides: [1] : memref<f16> to memref<1073741822xf16, strided<[1]>>
      vector.store %1, %reinterpret_cast_0[%2] : memref<1073741822xf16, strided<[1]>>, vector<16xf16>
      gpu.return
    }
  }
  func.func private @wave_get_buffer(!llvm.ptr) -> memref<?xi8> attributes {llvm.emit_c_interface}
  func.func private @wave_get_dim(!llvm.ptr, i32) -> i64 attributes {llvm.emit_c_interface}
  func.func private @wave_get_int64(!llvm.ptr) -> i64 attributes {llvm.emit_c_interface}
  func.func private @wave_get_float64(!llvm.ptr) -> f64 attributes {llvm.emit_c_interface}
  func.func @isolated_benchmark(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %0 = call @wave_get_buffer(%arg1) : (!llvm.ptr) -> memref<?xi8>
    %view = memref.view %0[%c0][] : memref<?xi8> to memref<f16>
    %1 = call @wave_get_buffer(%arg2) : (!llvm.ptr) -> memref<?xi8>
    %view_0 = memref.view %1[%c0][] : memref<?xi8> to memref<f16>
    gpu.launch_func  @gpu_module::@read_write blocks in (%c1, %c1, %c1) threads in (%c64, %c1, %c1)  args(%view : memref<f16>, %view_0 : memref<f16>)
    return
  }
}
