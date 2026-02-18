#map = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 256) * 256)>
#map1 = affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>
#map2 = affine_map<()[s0] -> (s0 mod 8)>
#map3 = affine_map<()[s0] -> (s0 * 16)>
#map4 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 32) * 256)>
#map5 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
#map6 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 32) * 256 + 64)>
#map7 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
#map8 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 16) floordiv 32) * 256 + 128)>
#map9 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
#map10 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 24) floordiv 32) * 256 + 192)>
#map11 = affine_map<()[s0, s1, s2] -> (s1 * 128 + s2 * 256 + s0 floordiv 2 - ((s1 * 128 + s0 floordiv 2) floordiv 256) * 256)>
#map12 = affine_map<()[s0] -> ((s0 floordiv 2) mod 2)>
#map13 = affine_map<()[s0] -> (s0 mod 2)>
#map14 = affine_map<()[s0] -> (s0 * 4)>
#map15 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 32 - ((s1 * 4 + s0 floordiv 64) floordiv 8) * 256)>
#map16 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
#map17 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64)>
#map18 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
#map19 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 16)>
#map20 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 32)>
#map21 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 48)>
#map22 = affine_map<()[s0] -> (s0 * 4 + (s0 mod 64) floordiv 16 - (s0 floordiv 2) * 8)>
#map23 = affine_map<()[s0] -> (s0 * 256)>
#map24 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 256 + s2 * 128 - (s0 floordiv 16) * 16)>
#map25 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 256 + s2 * 128 - (s0 floordiv 16) * 16 + 16)>
#map26 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 256 + s2 * 128 - (s0 floordiv 16) * 16 + 32)>
#map27 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 256 + s2 * 128 - (s0 floordiv 16) * 16 + 48)>
#map28 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 256 + s2 * 128 - (s0 floordiv 16) * 16 + 64)>
#map29 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 256 + s2 * 128 - (s0 floordiv 16) * 16 + 80)>
#map30 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 256 + s2 * 128 - (s0 floordiv 16) * 16 + 96)>
#map31 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 256 + s2 * 128 - (s0 floordiv 16) * 16 + 112)>
#map32 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
#map33 = affine_map<()[s0, s1] -> (s1 * 4 + (s0 mod 64) floordiv 16)>
#map34 = affine_map<()[s0, s1, s2] -> (s2 * 128 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256)>
#map35 = affine_map<()[s0, s1] -> (s0 * 16 + s1 * 2048 - (s0 floordiv 16) * 256 + ((s0 mod 64) floordiv 16) * 256 - ((s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256) * 4096)>
#map36 = affine_map<()[s0, s1, s2] -> (s2 * 128 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 16)>
#map37 = affine_map<()[s0, s1, s2] -> (s2 * 128 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 32)>
#map38 = affine_map<()[s0, s1, s2] -> (s2 * 128 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 48)>
#map39 = affine_map<()[s0, s1, s2] -> (s2 * 128 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 64)>
#map40 = affine_map<()[s0, s1, s2] -> (s2 * 128 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 80)>
#map41 = affine_map<()[s0, s1, s2] -> (s2 * 128 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 96)>
#map42 = affine_map<()[s0, s1, s2] -> (s2 * 128 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 112)>
#map43 = affine_map<()[s0, s1] -> (s1 * 8 + (s0 mod 64) floordiv 16)>
#map44 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + 128)>
#map45 = affine_map<()[s0, s1] -> (s0 * 8 + s1 * 4 + 8)>
#map46 = affine_map<()[s0, s1, s2] -> (s2 * 128 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256)>
#map47 = affine_map<()[s0, s1] -> (s0 * 16 + s1 * 2048 - (s0 floordiv 16) * 256 + ((s0 mod 64) floordiv 16) * 256 - ((s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256) * 4096 + 1024)>
#map48 = affine_map<()[s0, s1, s2] -> (s2 * 128 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256 + 16)>
#map49 = affine_map<()[s0, s1, s2] -> (s2 * 128 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256 + 32)>
#map50 = affine_map<()[s0, s1, s2] -> (s2 * 128 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256 + 48)>
#map51 = affine_map<()[s0, s1, s2] -> (s2 * 128 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256 + 64)>
#map52 = affine_map<()[s0, s1, s2] -> (s2 * 128 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256 + 80)>
#map53 = affine_map<()[s0, s1, s2] -> (s2 * 128 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256 + 96)>
#map54 = affine_map<()[s0, s1, s2] -> (s2 * 128 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256 + 112)>
#map55 = affine_map<()[s0, s1] -> (s1 * 8 + (s0 mod 64) floordiv 16 + 4)>
#map56 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 3968) floordiv 256)>
#map57 = affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 16) * 256 + ((s0 mod 64) floordiv 16) * 256 - ((s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 128) floordiv 256) * 4096 + 2048)>
#map58 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 4032) floordiv 256)>
#map59 = affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 16) * 256 + ((s0 mod 64) floordiv 16) * 256 - ((s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 192) floordiv 256) * 4096 + 3072)>
#map60 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 3968) floordiv 256 + 16)>
#map61 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 4032) floordiv 256 + 16)>
#map62 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 3968) floordiv 256 + 32)>
#map63 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 4032) floordiv 256 + 32)>
#map64 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 3968) floordiv 256 + 48)>
#map65 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 4032) floordiv 256 + 48)>
#map66 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 3968) floordiv 256 + 64)>
#map67 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 4032) floordiv 256 + 64)>
#map68 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 3968) floordiv 256 + 80)>
#map69 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 4032) floordiv 256 + 80)>
#map70 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 3968) floordiv 256 + 96)>
#map71 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 4032) floordiv 256 + 96)>
#map72 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 3968) floordiv 256 + 112)>
#map73 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 4032) floordiv 256 + 112)>
#map74 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 248)>
#map75 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 252)>
#map76 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4)>
#map77 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16)>
#map78 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 1)>
#map79 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 2)>
#map80 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 3)>
#map81 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 16)>
#map82 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 32)>
#map83 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 48)>
#map84 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 64)>
#map85 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 80)>
#map86 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 96)>
#map87 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 112)>
#map88 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 16)>
#map89 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 17)>
#map90 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 18)>
#map91 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 19)>
#map92 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 32)>
#map93 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 33)>
#map94 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 34)>
#map95 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 35)>
#map96 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 48)>
#map97 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 49)>
#map98 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 50)>
#map99 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 51)>
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @gemm {
    stream.executable.export public @gemm workgroups() -> (index, index, index) {
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      stream.return %c4, %c4, %c1 : index, index, index
    }
    builtin.module {
      func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
        %c1024_i14 = arith.constant 1024 : i14
        %c4_i32 = arith.constant 4 : i32
        %c256_i14 = arith.constant 256 : i14
        %c4096_i14 = arith.constant 4096 : i14
        %c2147483643_i64 = arith.constant 2147483643 : i64
        %c1024 = arith.constant 1024 : index
        %c31 = arith.constant 31 : index
        %c2147483646_i64 = arith.constant 2147483646 : i64
        %c4096 = arith.constant 4096 : index
        %c1 = arith.constant 1 : index
        %c256 = arith.constant 256 : index
        %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
        %c0 = arith.constant 0 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<i8>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<i8>
        %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<i8>
        %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<i8>
        %4 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<f32>
        %block_id_x = gpu.block_id  x upper_bound 4
        %block_id_y = gpu.block_id  y upper_bound 4
        %thread_id_x = gpu.thread_id  x upper_bound 256
        %thread_id_y = gpu.thread_id  y upper_bound 2
        %alloc = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
        %alloc_0 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
        %alloc_1 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
        %alloc_2 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
        %5 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_x]
        %6 = affine.apply #map1()[%thread_id_x]
        %7 = affine.apply #map2()[%thread_id_x]
        %8 = arith.xori %7, %6 : index
        %9 = affine.apply #map3()[%8]
        %10 = affine.apply #map4()[%thread_id_x, %thread_id_y]
        %11 = gpu.subgroup_broadcast %10,  first_active_lane : index
        %12 = gpu.subgroup_broadcast %c0,  first_active_lane : index
        %13 = arith.muli %5, %c4096 overflow<nsw> : index
        %14 = arith.addi %13, %9 overflow<nsw> : index
        %reinterpret_cast = memref.reinterpret_cast %0 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast = memref.cast %reinterpret_cast : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %15 = amdgpu.fat_raw_buffer_cast %cast validBytes(%c2147483646_i64) cacheSwizzleStride(%c4096_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        amdgpu.gather_to_lds %15[%14], %alloc_2[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %16 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_x]
        %17 = affine.apply #map6()[%thread_id_x, %thread_id_y]
        %18 = gpu.subgroup_broadcast %17,  first_active_lane : index
        %19 = arith.muli %16, %c4096 overflow<nsw> : index
        %20 = arith.addi %19, %9 overflow<nsw> : index
        amdgpu.gather_to_lds %15[%20], %alloc_2[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %21 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_x]
        %22 = affine.apply #map8()[%thread_id_x, %thread_id_y]
        %23 = gpu.subgroup_broadcast %22,  first_active_lane : index
        %24 = arith.muli %21, %c4096 overflow<nsw> : index
        %25 = arith.addi %24, %9 overflow<nsw> : index
        amdgpu.gather_to_lds %15[%25], %alloc_2[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %26 = affine.apply #map9()[%thread_id_x, %thread_id_y, %block_id_x]
        %27 = affine.apply #map10()[%thread_id_x, %thread_id_y]
        %28 = gpu.subgroup_broadcast %27,  first_active_lane : index
        %29 = arith.muli %26, %c4096 overflow<nsw> : index
        %30 = arith.addi %29, %9 overflow<nsw> : index
        amdgpu.gather_to_lds %15[%30], %alloc_2[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %31 = affine.apply #map11()[%thread_id_x, %thread_id_y, %block_id_x]
        %32 = affine.apply #map12()[%thread_id_x]
        %33 = affine.apply #map13()[%thread_id_x]
        %34 = arith.xori %33, %32 : index
        %35 = affine.apply #map14()[%34]
        %36 = affine.apply #map15()[%thread_id_x, %thread_id_y]
        %37 = gpu.subgroup_broadcast %36,  first_active_lane : index
        %38 = arith.muli %31, %c256 overflow<nsw> : index
        %39 = arith.addi %38, %35 overflow<nsw> : index
        %reinterpret_cast_3 = memref.reinterpret_cast %1 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast_4 = memref.cast %reinterpret_cast_3 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %40 = amdgpu.fat_raw_buffer_cast %cast_4 validBytes(%c2147483646_i64) cacheSwizzleStride(%c256_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        amdgpu.gather_to_lds %40[%39], %alloc_0[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
        amdgpu.lds_barrier
        %41 = affine.apply #map16()[%thread_id_x, %thread_id_y]
        %42 = arith.index_cast %41 : index to i32
        %43 = arith.cmpi sge, %42, %c4_i32 : i32
        %44 = arith.cmpi slt, %42, %c4_i32 : i32
        scf.if %43 {
          rocdl.s.barrier
        }
        %45 = affine.apply #map17()[%thread_id_x]
        %46 = affine.apply #map18()[%thread_id_x]
        %47 = arith.xori %46, %7 : index
        %48 = affine.apply #map3()[%47]
        %49 = affine.apply #map19()[%thread_id_x]
        %50 = affine.apply #map20()[%thread_id_x]
        %51 = affine.apply #map21()[%thread_id_x]
        %52 = affine.apply #map22()[%thread_id_x]
        %53 = affine.apply #map23()[%block_id_y]
        %54 = arith.muli %53, %c4096 overflow<nsw> : index
        %reinterpret_cast_5 = memref.reinterpret_cast %2 to offset: [%54], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1], offset: ?>>
        %cast_6 = memref.cast %reinterpret_cast_5 : memref<2147483646xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
        %55 = amdgpu.fat_raw_buffer_cast %cast_6 validBytes(%c2147483646_i64) cacheSwizzleStride(%c4096_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        %56 = affine.apply #map24()[%thread_id_x, %block_id_y, %thread_id_y]
        %57 = arith.muli %56, %c256 overflow<nsw> : index
        %reinterpret_cast_7 = memref.reinterpret_cast %3 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast_8 = memref.cast %reinterpret_cast_7 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %58 = amdgpu.fat_raw_buffer_cast %cast_8 validBytes(%c2147483646_i64) cacheSwizzleStride(%c256_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        %59 = affine.apply #map25()[%thread_id_x, %block_id_y, %thread_id_y]
        %60 = arith.muli %59, %c256 overflow<nsw> : index
        %61 = affine.apply #map26()[%thread_id_x, %block_id_y, %thread_id_y]
        %62 = arith.muli %61, %c256 overflow<nsw> : index
        %63 = affine.apply #map27()[%thread_id_x, %block_id_y, %thread_id_y]
        %64 = arith.muli %63, %c256 overflow<nsw> : index
        %65 = affine.apply #map28()[%thread_id_x, %block_id_y, %thread_id_y]
        %66 = arith.muli %65, %c256 overflow<nsw> : index
        %67 = affine.apply #map29()[%thread_id_x, %block_id_y, %thread_id_y]
        %68 = arith.muli %67, %c256 overflow<nsw> : index
        %69 = affine.apply #map30()[%thread_id_x, %block_id_y, %thread_id_y]
        %70 = arith.muli %69, %c256 overflow<nsw> : index
        %71 = affine.apply #map31()[%thread_id_x, %block_id_y, %thread_id_y]
        %72 = arith.muli %71, %c256 overflow<nsw> : index
        %73 = affine.apply #map32()[%thread_id_x]
        %74 = arith.xori %73, %7 : index
        %75 = affine.apply #map3()[%74]
        %76 = arith.xori %33, %c1 : index
        %77 = affine.apply #map33()[%thread_id_x, %76]
        %78:36 = scf.for %arg5 = %c0 to %c31 step %c1 iter_args(%arg6 = %cst, %arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst, %arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %cst, %arg21 = %cst, %arg22 = %cst, %arg23 = %cst, %arg24 = %cst, %arg25 = %cst, %arg26 = %cst, %arg27 = %cst, %arg28 = %cst, %arg29 = %cst, %arg30 = %cst, %arg31 = %cst, %arg32 = %cst, %arg33 = %cst, %arg34 = %cst, %arg35 = %cst, %arg36 = %cst, %arg37 = %cst, %arg38 = %alloc_2, %arg39 = %alloc_1, %arg40 = %alloc_0, %arg41 = %alloc) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>) {
          %664 = vector.load %arg38[%45, %48] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %665 = vector.load %arg38[%49, %48] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %666 = vector.load %arg38[%50, %48] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %667 = vector.load %arg38[%51, %48] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %668 = vector.load %arg40[%45, %52] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %669 = vector.load %arg40[%49, %52] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %670 = vector.load %arg40[%50, %52] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %671 = vector.load %arg40[%51, %52] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %672 = affine.apply #map34()[%thread_id_x, %arg5, %thread_id_y]
          %673 = affine.apply #map35()[%thread_id_x, %arg5]
          %674 = arith.muli %672, %c4096 overflow<nsw> : index
          %675 = arith.addi %674, %673 overflow<nsw> : index
          %676 = vector.load %55[%675] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %677 = affine.apply #map36()[%thread_id_x, %arg5, %thread_id_y]
          %678 = arith.muli %677, %c4096 overflow<nsw> : index
          %679 = arith.addi %678, %673 overflow<nsw> : index
          %680 = vector.load %55[%679] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %681 = affine.apply #map37()[%thread_id_x, %arg5, %thread_id_y]
          %682 = arith.muli %681, %c4096 overflow<nsw> : index
          %683 = arith.addi %682, %673 overflow<nsw> : index
          %684 = vector.load %55[%683] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %685 = affine.apply #map38()[%thread_id_x, %arg5, %thread_id_y]
          %686 = arith.muli %685, %c4096 overflow<nsw> : index
          %687 = arith.addi %686, %673 overflow<nsw> : index
          %688 = vector.load %55[%687] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %689 = affine.apply #map39()[%thread_id_x, %arg5, %thread_id_y]
          %690 = arith.muli %689, %c4096 overflow<nsw> : index
          %691 = arith.addi %690, %673 overflow<nsw> : index
          %692 = vector.load %55[%691] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %693 = affine.apply #map40()[%thread_id_x, %arg5, %thread_id_y]
          %694 = arith.muli %693, %c4096 overflow<nsw> : index
          %695 = arith.addi %694, %673 overflow<nsw> : index
          %696 = vector.load %55[%695] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %697 = affine.apply #map41()[%thread_id_x, %arg5, %thread_id_y]
          %698 = arith.muli %697, %c4096 overflow<nsw> : index
          %699 = arith.addi %698, %673 overflow<nsw> : index
          %700 = vector.load %55[%699] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %701 = affine.apply #map42()[%thread_id_x, %arg5, %thread_id_y]
          %702 = arith.muli %701, %c4096 overflow<nsw> : index
          %703 = arith.addi %702, %673 overflow<nsw> : index
          %704 = vector.load %55[%703] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %705 = affine.apply #map43()[%thread_id_x, %arg5]
          %706 = arith.addi %57, %705 overflow<nsw> : index
          %707 = vector.load %58[%706] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
          %708 = arith.addi %60, %705 overflow<nsw> : index
          %709 = vector.load %58[%708] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
          %710 = arith.addi %62, %705 overflow<nsw> : index
          %711 = vector.load %58[%710] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
          %712 = arith.addi %64, %705 overflow<nsw> : index
          %713 = vector.load %58[%712] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
          %714 = arith.addi %66, %705 overflow<nsw> : index
          %715 = vector.load %58[%714] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
          %716 = arith.addi %68, %705 overflow<nsw> : index
          %717 = vector.load %58[%716] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
          %718 = arith.addi %70, %705 overflow<nsw> : index
          %719 = vector.load %58[%718] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
          %720 = arith.addi %72, %705 overflow<nsw> : index
          %721 = vector.load %58[%720] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
          %722 = vector.bitcast %664 : vector<16xi8> to vector<32xf4E2M1FN>
          %723 = vector.bitcast %665 : vector<16xi8> to vector<32xf4E2M1FN>
          %724 = vector.bitcast %666 : vector<16xi8> to vector<32xf4E2M1FN>
          %725 = vector.bitcast %667 : vector<16xi8> to vector<32xf4E2M1FN>
          %726 = vector.bitcast %668 : vector<1xi8> to vector<1xf8E8M0FNU>
          %727 = vector.bitcast %669 : vector<1xi8> to vector<1xf8E8M0FNU>
          %728 = vector.bitcast %670 : vector<1xi8> to vector<1xf8E8M0FNU>
          %729 = vector.bitcast %671 : vector<1xi8> to vector<1xf8E8M0FNU>
          %730 = vector.bitcast %676 : vector<16xi8> to vector<32xf4E2M1FN>
          %731 = vector.bitcast %680 : vector<16xi8> to vector<32xf4E2M1FN>
          %732 = vector.bitcast %684 : vector<16xi8> to vector<32xf4E2M1FN>
          %733 = vector.bitcast %688 : vector<16xi8> to vector<32xf4E2M1FN>
          %734 = vector.bitcast %692 : vector<16xi8> to vector<32xf4E2M1FN>
          %735 = vector.bitcast %696 : vector<16xi8> to vector<32xf4E2M1FN>
          %736 = vector.bitcast %700 : vector<16xi8> to vector<32xf4E2M1FN>
          %737 = vector.bitcast %704 : vector<16xi8> to vector<32xf4E2M1FN>
          %738 = vector.bitcast %707 : vector<1xi8> to vector<1xf8E8M0FNU>
          %739 = vector.bitcast %709 : vector<1xi8> to vector<1xf8E8M0FNU>
          %740 = vector.bitcast %711 : vector<1xi8> to vector<1xf8E8M0FNU>
          %741 = vector.bitcast %713 : vector<1xi8> to vector<1xf8E8M0FNU>
          %742 = vector.bitcast %715 : vector<1xi8> to vector<1xf8E8M0FNU>
          %743 = vector.bitcast %717 : vector<1xi8> to vector<1xf8E8M0FNU>
          %744 = vector.bitcast %719 : vector<1xi8> to vector<1xf8E8M0FNU>
          %745 = vector.bitcast %721 : vector<1xi8> to vector<1xf8E8M0FNU>
          rocdl.sched.barrier 0
          %746 = affine.apply #map44()[%arg5, %8]
          %747 = arith.addi %13, %746 overflow<nsw> : index
          amdgpu.gather_to_lds %15[%747], %arg39[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %748 = arith.addi %19, %746 overflow<nsw> : index
          amdgpu.gather_to_lds %15[%748], %arg39[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %749 = arith.addi %24, %746 overflow<nsw> : index
          amdgpu.gather_to_lds %15[%749], %arg39[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %750 = arith.addi %29, %746 overflow<nsw> : index
          amdgpu.gather_to_lds %15[%750], %arg39[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %751 = affine.apply #map45()[%arg5, %34]
          %752 = arith.addi %38, %751 overflow<nsw> : index
          amdgpu.gather_to_lds %40[%752], %arg41[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
          rocdl.sched.barrier 0
          rocdl.s.barrier
          rocdl.sched.barrier 0
          rocdl.s.setprio 1
          %753 = vector.extract %726[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %754 = vector.extract %738[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %755 = amdgpu.scaled_mfma 16x16x128 (%753[0] * %722) * (%754[0] * %730) + %arg6 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %756 = vector.extract %739[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %757 = amdgpu.scaled_mfma 16x16x128 (%753[0] * %722) * (%756[0] * %731) + %arg7 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %758 = vector.extract %740[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %759 = amdgpu.scaled_mfma 16x16x128 (%753[0] * %722) * (%758[0] * %732) + %arg8 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %760 = vector.extract %741[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %761 = amdgpu.scaled_mfma 16x16x128 (%753[0] * %722) * (%760[0] * %733) + %arg9 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %762 = vector.extract %742[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %763 = amdgpu.scaled_mfma 16x16x128 (%753[0] * %722) * (%762[0] * %734) + %arg10 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %764 = vector.extract %743[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %765 = amdgpu.scaled_mfma 16x16x128 (%753[0] * %722) * (%764[0] * %735) + %arg11 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %766 = vector.extract %744[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %767 = amdgpu.scaled_mfma 16x16x128 (%753[0] * %722) * (%766[0] * %736) + %arg12 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %768 = vector.extract %745[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %769 = amdgpu.scaled_mfma 16x16x128 (%753[0] * %722) * (%768[0] * %737) + %arg13 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %770 = vector.extract %727[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %771 = amdgpu.scaled_mfma 16x16x128 (%770[0] * %723) * (%754[0] * %730) + %arg14 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %772 = amdgpu.scaled_mfma 16x16x128 (%770[0] * %723) * (%756[0] * %731) + %arg15 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %773 = amdgpu.scaled_mfma 16x16x128 (%770[0] * %723) * (%758[0] * %732) + %arg16 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %774 = amdgpu.scaled_mfma 16x16x128 (%770[0] * %723) * (%760[0] * %733) + %arg17 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %775 = amdgpu.scaled_mfma 16x16x128 (%770[0] * %723) * (%762[0] * %734) + %arg18 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %776 = amdgpu.scaled_mfma 16x16x128 (%770[0] * %723) * (%764[0] * %735) + %arg19 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %777 = amdgpu.scaled_mfma 16x16x128 (%770[0] * %723) * (%766[0] * %736) + %arg20 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %778 = amdgpu.scaled_mfma 16x16x128 (%770[0] * %723) * (%768[0] * %737) + %arg21 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %779 = vector.extract %728[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %780 = amdgpu.scaled_mfma 16x16x128 (%779[0] * %724) * (%754[0] * %730) + %arg22 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %781 = amdgpu.scaled_mfma 16x16x128 (%779[0] * %724) * (%756[0] * %731) + %arg23 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %782 = amdgpu.scaled_mfma 16x16x128 (%779[0] * %724) * (%758[0] * %732) + %arg24 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %783 = amdgpu.scaled_mfma 16x16x128 (%779[0] * %724) * (%760[0] * %733) + %arg25 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %784 = amdgpu.scaled_mfma 16x16x128 (%779[0] * %724) * (%762[0] * %734) + %arg26 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %785 = amdgpu.scaled_mfma 16x16x128 (%779[0] * %724) * (%764[0] * %735) + %arg27 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %786 = amdgpu.scaled_mfma 16x16x128 (%779[0] * %724) * (%766[0] * %736) + %arg28 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %787 = amdgpu.scaled_mfma 16x16x128 (%779[0] * %724) * (%768[0] * %737) + %arg29 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %788 = vector.extract %729[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %789 = amdgpu.scaled_mfma 16x16x128 (%788[0] * %725) * (%754[0] * %730) + %arg30 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %790 = amdgpu.scaled_mfma 16x16x128 (%788[0] * %725) * (%756[0] * %731) + %arg31 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %791 = amdgpu.scaled_mfma 16x16x128 (%788[0] * %725) * (%758[0] * %732) + %arg32 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %792 = amdgpu.scaled_mfma 16x16x128 (%788[0] * %725) * (%760[0] * %733) + %arg33 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %793 = amdgpu.scaled_mfma 16x16x128 (%788[0] * %725) * (%762[0] * %734) + %arg34 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %794 = amdgpu.scaled_mfma 16x16x128 (%788[0] * %725) * (%764[0] * %735) + %arg35 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %795 = amdgpu.scaled_mfma 16x16x128 (%788[0] * %725) * (%766[0] * %736) + %arg36 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %796 = amdgpu.scaled_mfma 16x16x128 (%788[0] * %725) * (%768[0] * %737) + %arg37 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          rocdl.s.setprio 0
          rocdl.sched.barrier 0
          amdgpu.memory_counter_wait load(5)
          rocdl.s.barrier
          rocdl.sched.barrier 0
          %797 = vector.load %arg38[%45, %75] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %798 = vector.load %arg38[%49, %75] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %799 = vector.load %arg38[%50, %75] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %800 = vector.load %arg38[%51, %75] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %801 = vector.load %arg40[%45, %77] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %802 = vector.load %arg40[%49, %77] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %803 = vector.load %arg40[%50, %77] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %804 = vector.load %arg40[%51, %77] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %805 = affine.apply #map46()[%thread_id_x, %arg5, %thread_id_y]
          %806 = affine.apply #map47()[%thread_id_x, %arg5]
          %807 = arith.muli %805, %c4096 overflow<nsw> : index
          %808 = arith.addi %807, %806 overflow<nsw> : index
          %809 = vector.load %55[%808] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %810 = affine.apply #map48()[%thread_id_x, %arg5, %thread_id_y]
          %811 = arith.muli %810, %c4096 overflow<nsw> : index
          %812 = arith.addi %811, %806 overflow<nsw> : index
          %813 = vector.load %55[%812] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %814 = affine.apply #map49()[%thread_id_x, %arg5, %thread_id_y]
          %815 = arith.muli %814, %c4096 overflow<nsw> : index
          %816 = arith.addi %815, %806 overflow<nsw> : index
          %817 = vector.load %55[%816] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %818 = affine.apply #map50()[%thread_id_x, %arg5, %thread_id_y]
          %819 = arith.muli %818, %c4096 overflow<nsw> : index
          %820 = arith.addi %819, %806 overflow<nsw> : index
          %821 = vector.load %55[%820] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %822 = affine.apply #map51()[%thread_id_x, %arg5, %thread_id_y]
          %823 = arith.muli %822, %c4096 overflow<nsw> : index
          %824 = arith.addi %823, %806 overflow<nsw> : index
          %825 = vector.load %55[%824] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %826 = affine.apply #map52()[%thread_id_x, %arg5, %thread_id_y]
          %827 = arith.muli %826, %c4096 overflow<nsw> : index
          %828 = arith.addi %827, %806 overflow<nsw> : index
          %829 = vector.load %55[%828] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %830 = affine.apply #map53()[%thread_id_x, %arg5, %thread_id_y]
          %831 = arith.muli %830, %c4096 overflow<nsw> : index
          %832 = arith.addi %831, %806 overflow<nsw> : index
          %833 = vector.load %55[%832] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %834 = affine.apply #map54()[%thread_id_x, %arg5, %thread_id_y]
          %835 = arith.muli %834, %c4096 overflow<nsw> : index
          %836 = arith.addi %835, %806 overflow<nsw> : index
          %837 = vector.load %55[%836] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %838 = affine.apply #map55()[%thread_id_x, %arg5]
          %839 = arith.addi %57, %838 overflow<nsw> : index
          %840 = vector.load %58[%839] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
          %841 = arith.addi %60, %838 overflow<nsw> : index
          %842 = vector.load %58[%841] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
          %843 = arith.addi %62, %838 overflow<nsw> : index
          %844 = vector.load %58[%843] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
          %845 = arith.addi %64, %838 overflow<nsw> : index
          %846 = vector.load %58[%845] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
          %847 = arith.addi %66, %838 overflow<nsw> : index
          %848 = vector.load %58[%847] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
          %849 = arith.addi %68, %838 overflow<nsw> : index
          %850 = vector.load %58[%849] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
          %851 = arith.addi %70, %838 overflow<nsw> : index
          %852 = vector.load %58[%851] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
          %853 = arith.addi %72, %838 overflow<nsw> : index
          %854 = vector.load %58[%853] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
          %855 = vector.bitcast %797 : vector<16xi8> to vector<32xf4E2M1FN>
          %856 = vector.bitcast %798 : vector<16xi8> to vector<32xf4E2M1FN>
          %857 = vector.bitcast %799 : vector<16xi8> to vector<32xf4E2M1FN>
          %858 = vector.bitcast %800 : vector<16xi8> to vector<32xf4E2M1FN>
          %859 = vector.bitcast %801 : vector<1xi8> to vector<1xf8E8M0FNU>
          %860 = vector.bitcast %802 : vector<1xi8> to vector<1xf8E8M0FNU>
          %861 = vector.bitcast %803 : vector<1xi8> to vector<1xf8E8M0FNU>
          %862 = vector.bitcast %804 : vector<1xi8> to vector<1xf8E8M0FNU>
          %863 = vector.bitcast %809 : vector<16xi8> to vector<32xf4E2M1FN>
          %864 = vector.bitcast %813 : vector<16xi8> to vector<32xf4E2M1FN>
          %865 = vector.bitcast %817 : vector<16xi8> to vector<32xf4E2M1FN>
          %866 = vector.bitcast %821 : vector<16xi8> to vector<32xf4E2M1FN>
          %867 = vector.bitcast %825 : vector<16xi8> to vector<32xf4E2M1FN>
          %868 = vector.bitcast %829 : vector<16xi8> to vector<32xf4E2M1FN>
          %869 = vector.bitcast %833 : vector<16xi8> to vector<32xf4E2M1FN>
          %870 = vector.bitcast %837 : vector<16xi8> to vector<32xf4E2M1FN>
          %871 = vector.bitcast %840 : vector<1xi8> to vector<1xf8E8M0FNU>
          %872 = vector.bitcast %842 : vector<1xi8> to vector<1xf8E8M0FNU>
          %873 = vector.bitcast %844 : vector<1xi8> to vector<1xf8E8M0FNU>
          %874 = vector.bitcast %846 : vector<1xi8> to vector<1xf8E8M0FNU>
          %875 = vector.bitcast %848 : vector<1xi8> to vector<1xf8E8M0FNU>
          %876 = vector.bitcast %850 : vector<1xi8> to vector<1xf8E8M0FNU>
          %877 = vector.bitcast %852 : vector<1xi8> to vector<1xf8E8M0FNU>
          %878 = vector.bitcast %854 : vector<1xi8> to vector<1xf8E8M0FNU>
          rocdl.sched.barrier 0
          amdgpu.memory_counter_wait load(0)
          rocdl.s.barrier
          rocdl.sched.barrier 0
          rocdl.s.setprio 1
          %879 = vector.extract %859[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %880 = vector.extract %871[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %881 = amdgpu.scaled_mfma 16x16x128 (%879[0] * %855) * (%880[0] * %863) + %755 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %882 = vector.extract %872[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %883 = amdgpu.scaled_mfma 16x16x128 (%879[0] * %855) * (%882[0] * %864) + %757 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %884 = vector.extract %873[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %885 = amdgpu.scaled_mfma 16x16x128 (%879[0] * %855) * (%884[0] * %865) + %759 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %886 = vector.extract %874[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %887 = amdgpu.scaled_mfma 16x16x128 (%879[0] * %855) * (%886[0] * %866) + %761 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %888 = vector.extract %875[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %889 = amdgpu.scaled_mfma 16x16x128 (%879[0] * %855) * (%888[0] * %867) + %763 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %890 = vector.extract %876[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %891 = amdgpu.scaled_mfma 16x16x128 (%879[0] * %855) * (%890[0] * %868) + %765 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %892 = vector.extract %877[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %893 = amdgpu.scaled_mfma 16x16x128 (%879[0] * %855) * (%892[0] * %869) + %767 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %894 = vector.extract %878[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %895 = amdgpu.scaled_mfma 16x16x128 (%879[0] * %855) * (%894[0] * %870) + %769 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %896 = vector.extract %860[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %897 = amdgpu.scaled_mfma 16x16x128 (%896[0] * %856) * (%880[0] * %863) + %771 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %898 = amdgpu.scaled_mfma 16x16x128 (%896[0] * %856) * (%882[0] * %864) + %772 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %899 = amdgpu.scaled_mfma 16x16x128 (%896[0] * %856) * (%884[0] * %865) + %773 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %900 = amdgpu.scaled_mfma 16x16x128 (%896[0] * %856) * (%886[0] * %866) + %774 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %901 = amdgpu.scaled_mfma 16x16x128 (%896[0] * %856) * (%888[0] * %867) + %775 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %902 = amdgpu.scaled_mfma 16x16x128 (%896[0] * %856) * (%890[0] * %868) + %776 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %903 = amdgpu.scaled_mfma 16x16x128 (%896[0] * %856) * (%892[0] * %869) + %777 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %904 = amdgpu.scaled_mfma 16x16x128 (%896[0] * %856) * (%894[0] * %870) + %778 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %905 = vector.extract %861[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %906 = amdgpu.scaled_mfma 16x16x128 (%905[0] * %857) * (%880[0] * %863) + %780 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %907 = amdgpu.scaled_mfma 16x16x128 (%905[0] * %857) * (%882[0] * %864) + %781 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %908 = amdgpu.scaled_mfma 16x16x128 (%905[0] * %857) * (%884[0] * %865) + %782 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %909 = amdgpu.scaled_mfma 16x16x128 (%905[0] * %857) * (%886[0] * %866) + %783 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %910 = amdgpu.scaled_mfma 16x16x128 (%905[0] * %857) * (%888[0] * %867) + %784 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %911 = amdgpu.scaled_mfma 16x16x128 (%905[0] * %857) * (%890[0] * %868) + %785 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %912 = amdgpu.scaled_mfma 16x16x128 (%905[0] * %857) * (%892[0] * %869) + %786 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %913 = amdgpu.scaled_mfma 16x16x128 (%905[0] * %857) * (%894[0] * %870) + %787 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %914 = vector.extract %862[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
          %915 = amdgpu.scaled_mfma 16x16x128 (%914[0] * %858) * (%880[0] * %863) + %789 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %916 = amdgpu.scaled_mfma 16x16x128 (%914[0] * %858) * (%882[0] * %864) + %790 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %917 = amdgpu.scaled_mfma 16x16x128 (%914[0] * %858) * (%884[0] * %865) + %791 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %918 = amdgpu.scaled_mfma 16x16x128 (%914[0] * %858) * (%886[0] * %866) + %792 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %919 = amdgpu.scaled_mfma 16x16x128 (%914[0] * %858) * (%888[0] * %867) + %793 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %920 = amdgpu.scaled_mfma 16x16x128 (%914[0] * %858) * (%890[0] * %868) + %794 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %921 = amdgpu.scaled_mfma 16x16x128 (%914[0] * %858) * (%892[0] * %869) + %795 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %922 = amdgpu.scaled_mfma 16x16x128 (%914[0] * %858) * (%894[0] * %870) + %796 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          rocdl.s.setprio 0
          rocdl.sched.barrier 0
          amdgpu.lds_barrier
          scf.yield %881, %883, %885, %887, %889, %891, %893, %895, %897, %898, %899, %900, %901, %902, %903, %904, %906, %907, %908, %909, %910, %911, %912, %913, %915, %916, %917, %918, %919, %920, %921, %922, %arg39, %arg38, %arg41, %arg40 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>
        }
        scf.if %44 {
          rocdl.s.barrier
        }
        %79 = affine.apply #map17()[%thread_id_x]
        %80 = affine.apply #map22()[%thread_id_x]
        %81 = vector.load %78#34[%79, %80] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %82 = arith.xori %33, %c1 : index
        %83 = affine.apply #map33()[%thread_id_x, %82]
        %84 = vector.load %78#34[%79, %83] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %85 = affine.apply #map19()[%thread_id_x]
        %86 = vector.load %78#34[%85, %80] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %87 = vector.load %78#34[%85, %83] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %88 = affine.apply #map20()[%thread_id_x]
        %89 = vector.load %78#34[%88, %80] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %90 = vector.load %78#34[%88, %83] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %91 = affine.apply #map21()[%thread_id_x]
        %92 = vector.load %78#34[%91, %80] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %93 = vector.load %78#34[%91, %83] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %94 = affine.apply #map18()[%thread_id_x]
        %95 = arith.xori %94, %7 : index
        %96 = affine.apply #map3()[%95]
        %97 = vector.load %78#32[%79, %96] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %98 = affine.apply #map32()[%thread_id_x]
        %99 = arith.xori %98, %7 : index
        %100 = affine.apply #map3()[%99]
        %101 = vector.load %78#32[%79, %100] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %102 = vector.load %78#32[%85, %96] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %103 = vector.load %78#32[%85, %100] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %104 = vector.load %78#32[%88, %96] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %105 = vector.load %78#32[%88, %100] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %106 = vector.load %78#32[%91, %96] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %107 = vector.load %78#32[%91, %100] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %108 = affine.apply #map23()[%block_id_y]
        %109 = affine.apply #map56()[%thread_id_x, %thread_id_y]
        %110 = affine.apply #map57()[%thread_id_x]
        %111 = arith.muli %108, %c4096 overflow<nsw> : index
        %112 = arith.muli %109, %c4096 overflow<nsw> : index
        %113 = arith.addi %112, %110 overflow<nsw> : index
        %reinterpret_cast_9 = memref.reinterpret_cast %2 to offset: [%111], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1], offset: ?>>
        %cast_10 = memref.cast %reinterpret_cast_9 : memref<2147483646xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
        %114 = amdgpu.fat_raw_buffer_cast %cast_10 validBytes(%c2147483646_i64) cacheSwizzleStride(%c4096_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        %115 = vector.load %114[%113] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %116 = affine.apply #map58()[%thread_id_x, %thread_id_y]
        %117 = affine.apply #map59()[%thread_id_x]
        %118 = arith.muli %116, %c4096 overflow<nsw> : index
        %119 = arith.addi %118, %117 overflow<nsw> : index
        %120 = vector.load %114[%119] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %121 = affine.apply #map60()[%thread_id_x, %thread_id_y]
        %122 = arith.muli %121, %c4096 overflow<nsw> : index
        %123 = arith.addi %122, %110 overflow<nsw> : index
        %124 = vector.load %114[%123] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %125 = affine.apply #map61()[%thread_id_x, %thread_id_y]
        %126 = arith.muli %125, %c4096 overflow<nsw> : index
        %127 = arith.addi %126, %117 overflow<nsw> : index
        %128 = vector.load %114[%127] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %129 = affine.apply #map62()[%thread_id_x, %thread_id_y]
        %130 = arith.muli %129, %c4096 overflow<nsw> : index
        %131 = arith.addi %130, %110 overflow<nsw> : index
        %132 = vector.load %114[%131] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %133 = affine.apply #map63()[%thread_id_x, %thread_id_y]
        %134 = arith.muli %133, %c4096 overflow<nsw> : index
        %135 = arith.addi %134, %117 overflow<nsw> : index
        %136 = vector.load %114[%135] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %137 = affine.apply #map64()[%thread_id_x, %thread_id_y]
        %138 = arith.muli %137, %c4096 overflow<nsw> : index
        %139 = arith.addi %138, %110 overflow<nsw> : index
        %140 = vector.load %114[%139] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %141 = affine.apply #map65()[%thread_id_x, %thread_id_y]
        %142 = arith.muli %141, %c4096 overflow<nsw> : index
        %143 = arith.addi %142, %117 overflow<nsw> : index
        %144 = vector.load %114[%143] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %145 = affine.apply #map66()[%thread_id_x, %thread_id_y]
        %146 = arith.muli %145, %c4096 overflow<nsw> : index
        %147 = arith.addi %146, %110 overflow<nsw> : index
        %148 = vector.load %114[%147] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %149 = affine.apply #map67()[%thread_id_x, %thread_id_y]
        %150 = arith.muli %149, %c4096 overflow<nsw> : index
        %151 = arith.addi %150, %117 overflow<nsw> : index
        %152 = vector.load %114[%151] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %153 = affine.apply #map68()[%thread_id_x, %thread_id_y]
        %154 = arith.muli %153, %c4096 overflow<nsw> : index
        %155 = arith.addi %154, %110 overflow<nsw> : index
        %156 = vector.load %114[%155] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %157 = affine.apply #map69()[%thread_id_x, %thread_id_y]
        %158 = arith.muli %157, %c4096 overflow<nsw> : index
        %159 = arith.addi %158, %117 overflow<nsw> : index
        %160 = vector.load %114[%159] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %161 = affine.apply #map70()[%thread_id_x, %thread_id_y]
        %162 = arith.muli %161, %c4096 overflow<nsw> : index
        %163 = arith.addi %162, %110 overflow<nsw> : index
        %164 = vector.load %114[%163] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %165 = affine.apply #map71()[%thread_id_x, %thread_id_y]
        %166 = arith.muli %165, %c4096 overflow<nsw> : index
        %167 = arith.addi %166, %117 overflow<nsw> : index
        %168 = vector.load %114[%167] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %169 = affine.apply #map72()[%thread_id_x, %thread_id_y]
        %170 = arith.muli %169, %c4096 overflow<nsw> : index
        %171 = arith.addi %170, %110 overflow<nsw> : index
        %172 = vector.load %114[%171] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %173 = affine.apply #map73()[%thread_id_x, %thread_id_y]
        %174 = arith.muli %173, %c4096 overflow<nsw> : index
        %175 = arith.addi %174, %117 overflow<nsw> : index
        %176 = vector.load %114[%175] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %177 = affine.apply #map24()[%thread_id_x, %block_id_y, %thread_id_y]
        %178 = affine.apply #map74()[%thread_id_x]
        %179 = arith.muli %177, %c256 overflow<nsw> : index
        %180 = arith.addi %179, %178 overflow<nsw> : index
        %reinterpret_cast_11 = memref.reinterpret_cast %3 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast_12 = memref.cast %reinterpret_cast_11 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %181 = amdgpu.fat_raw_buffer_cast %cast_12 validBytes(%c2147483646_i64) cacheSwizzleStride(%c256_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        %182 = vector.load %181[%180] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
        %183 = affine.apply #map75()[%thread_id_x]
        %184 = arith.addi %179, %183 overflow<nsw> : index
        %185 = vector.load %181[%184] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
        %186 = affine.apply #map25()[%thread_id_x, %block_id_y, %thread_id_y]
        %187 = arith.muli %186, %c256 overflow<nsw> : index
        %188 = arith.addi %187, %178 overflow<nsw> : index
        %189 = vector.load %181[%188] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
        %190 = arith.addi %187, %183 overflow<nsw> : index
        %191 = vector.load %181[%190] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
        %192 = affine.apply #map26()[%thread_id_x, %block_id_y, %thread_id_y]
        %193 = arith.muli %192, %c256 overflow<nsw> : index
        %194 = arith.addi %193, %178 overflow<nsw> : index
        %195 = vector.load %181[%194] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
        %196 = arith.addi %193, %183 overflow<nsw> : index
        %197 = vector.load %181[%196] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
        %198 = affine.apply #map27()[%thread_id_x, %block_id_y, %thread_id_y]
        %199 = arith.muli %198, %c256 overflow<nsw> : index
        %200 = arith.addi %199, %178 overflow<nsw> : index
        %201 = vector.load %181[%200] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
        %202 = arith.addi %199, %183 overflow<nsw> : index
        %203 = vector.load %181[%202] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
        %204 = affine.apply #map28()[%thread_id_x, %block_id_y, %thread_id_y]
        %205 = arith.muli %204, %c256 overflow<nsw> : index
        %206 = arith.addi %205, %178 overflow<nsw> : index
        %207 = vector.load %181[%206] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
        %208 = arith.addi %205, %183 overflow<nsw> : index
        %209 = vector.load %181[%208] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
        %210 = affine.apply #map29()[%thread_id_x, %block_id_y, %thread_id_y]
        %211 = arith.muli %210, %c256 overflow<nsw> : index
        %212 = arith.addi %211, %178 overflow<nsw> : index
        %213 = vector.load %181[%212] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
        %214 = arith.addi %211, %183 overflow<nsw> : index
        %215 = vector.load %181[%214] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
        %216 = affine.apply #map30()[%thread_id_x, %block_id_y, %thread_id_y]
        %217 = arith.muli %216, %c256 overflow<nsw> : index
        %218 = arith.addi %217, %178 overflow<nsw> : index
        %219 = vector.load %181[%218] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
        %220 = arith.addi %217, %183 overflow<nsw> : index
        %221 = vector.load %181[%220] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
        %222 = affine.apply #map31()[%thread_id_x, %block_id_y, %thread_id_y]
        %223 = arith.muli %222, %c256 overflow<nsw> : index
        %224 = arith.addi %223, %178 overflow<nsw> : index
        %225 = vector.load %181[%224] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
        %226 = arith.addi %223, %183 overflow<nsw> : index
        %227 = vector.load %181[%226] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi8>
        %228 = vector.bitcast %97 : vector<16xi8> to vector<32xf4E2M1FN>
        %229 = vector.bitcast %101 : vector<16xi8> to vector<32xf4E2M1FN>
        %230 = vector.bitcast %102 : vector<16xi8> to vector<32xf4E2M1FN>
        %231 = vector.bitcast %103 : vector<16xi8> to vector<32xf4E2M1FN>
        %232 = vector.bitcast %104 : vector<16xi8> to vector<32xf4E2M1FN>
        %233 = vector.bitcast %105 : vector<16xi8> to vector<32xf4E2M1FN>
        %234 = vector.bitcast %106 : vector<16xi8> to vector<32xf4E2M1FN>
        %235 = vector.bitcast %107 : vector<16xi8> to vector<32xf4E2M1FN>
        %236 = vector.bitcast %81 : vector<1xi8> to vector<1xf8E8M0FNU>
        %237 = vector.bitcast %84 : vector<1xi8> to vector<1xf8E8M0FNU>
        %238 = vector.bitcast %86 : vector<1xi8> to vector<1xf8E8M0FNU>
        %239 = vector.bitcast %87 : vector<1xi8> to vector<1xf8E8M0FNU>
        %240 = vector.bitcast %89 : vector<1xi8> to vector<1xf8E8M0FNU>
        %241 = vector.bitcast %90 : vector<1xi8> to vector<1xf8E8M0FNU>
        %242 = vector.bitcast %92 : vector<1xi8> to vector<1xf8E8M0FNU>
        %243 = vector.bitcast %93 : vector<1xi8> to vector<1xf8E8M0FNU>
        %244 = vector.bitcast %115 : vector<16xi8> to vector<32xf4E2M1FN>
        %245 = vector.bitcast %120 : vector<16xi8> to vector<32xf4E2M1FN>
        %246 = vector.bitcast %124 : vector<16xi8> to vector<32xf4E2M1FN>
        %247 = vector.bitcast %128 : vector<16xi8> to vector<32xf4E2M1FN>
        %248 = vector.bitcast %132 : vector<16xi8> to vector<32xf4E2M1FN>
        %249 = vector.bitcast %136 : vector<16xi8> to vector<32xf4E2M1FN>
        %250 = vector.bitcast %140 : vector<16xi8> to vector<32xf4E2M1FN>
        %251 = vector.bitcast %144 : vector<16xi8> to vector<32xf4E2M1FN>
        %252 = vector.bitcast %148 : vector<16xi8> to vector<32xf4E2M1FN>
        %253 = vector.bitcast %152 : vector<16xi8> to vector<32xf4E2M1FN>
        %254 = vector.bitcast %156 : vector<16xi8> to vector<32xf4E2M1FN>
        %255 = vector.bitcast %160 : vector<16xi8> to vector<32xf4E2M1FN>
        %256 = vector.bitcast %164 : vector<16xi8> to vector<32xf4E2M1FN>
        %257 = vector.bitcast %168 : vector<16xi8> to vector<32xf4E2M1FN>
        %258 = vector.bitcast %172 : vector<16xi8> to vector<32xf4E2M1FN>
        %259 = vector.bitcast %176 : vector<16xi8> to vector<32xf4E2M1FN>
        %260 = vector.bitcast %182 : vector<1xi8> to vector<1xf8E8M0FNU>
        %261 = vector.bitcast %185 : vector<1xi8> to vector<1xf8E8M0FNU>
        %262 = vector.bitcast %189 : vector<1xi8> to vector<1xf8E8M0FNU>
        %263 = vector.bitcast %191 : vector<1xi8> to vector<1xf8E8M0FNU>
        %264 = vector.bitcast %195 : vector<1xi8> to vector<1xf8E8M0FNU>
        %265 = vector.bitcast %197 : vector<1xi8> to vector<1xf8E8M0FNU>
        %266 = vector.bitcast %201 : vector<1xi8> to vector<1xf8E8M0FNU>
        %267 = vector.bitcast %203 : vector<1xi8> to vector<1xf8E8M0FNU>
        %268 = vector.bitcast %207 : vector<1xi8> to vector<1xf8E8M0FNU>
        %269 = vector.bitcast %209 : vector<1xi8> to vector<1xf8E8M0FNU>
        %270 = vector.bitcast %213 : vector<1xi8> to vector<1xf8E8M0FNU>
        %271 = vector.bitcast %215 : vector<1xi8> to vector<1xf8E8M0FNU>
        %272 = vector.bitcast %219 : vector<1xi8> to vector<1xf8E8M0FNU>
        %273 = vector.bitcast %221 : vector<1xi8> to vector<1xf8E8M0FNU>
        %274 = vector.bitcast %225 : vector<1xi8> to vector<1xf8E8M0FNU>
        %275 = vector.bitcast %227 : vector<1xi8> to vector<1xf8E8M0FNU>
        %276 = vector.extract %236[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %277 = vector.extract %260[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %278 = amdgpu.scaled_mfma 16x16x128 (%276[0] * %228) * (%277[0] * %244) + %78#0 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %279 = vector.extract %237[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %280 = vector.extract %261[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %281 = amdgpu.scaled_mfma 16x16x128 (%279[0] * %229) * (%280[0] * %245) + %278 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %282 = vector.extract %262[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %283 = amdgpu.scaled_mfma 16x16x128 (%276[0] * %228) * (%282[0] * %246) + %78#1 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %284 = vector.extract %263[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %285 = amdgpu.scaled_mfma 16x16x128 (%279[0] * %229) * (%284[0] * %247) + %283 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %286 = vector.extract %264[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %287 = amdgpu.scaled_mfma 16x16x128 (%276[0] * %228) * (%286[0] * %248) + %78#2 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %288 = vector.extract %265[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %289 = amdgpu.scaled_mfma 16x16x128 (%279[0] * %229) * (%288[0] * %249) + %287 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %290 = vector.extract %266[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %291 = amdgpu.scaled_mfma 16x16x128 (%276[0] * %228) * (%290[0] * %250) + %78#3 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %292 = vector.extract %267[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %293 = amdgpu.scaled_mfma 16x16x128 (%279[0] * %229) * (%292[0] * %251) + %291 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %294 = vector.extract %268[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %295 = amdgpu.scaled_mfma 16x16x128 (%276[0] * %228) * (%294[0] * %252) + %78#4 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %296 = vector.extract %269[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %297 = amdgpu.scaled_mfma 16x16x128 (%279[0] * %229) * (%296[0] * %253) + %295 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %298 = vector.extract %270[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %299 = amdgpu.scaled_mfma 16x16x128 (%276[0] * %228) * (%298[0] * %254) + %78#5 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %300 = vector.extract %271[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %301 = amdgpu.scaled_mfma 16x16x128 (%279[0] * %229) * (%300[0] * %255) + %299 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %302 = vector.extract %272[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %303 = amdgpu.scaled_mfma 16x16x128 (%276[0] * %228) * (%302[0] * %256) + %78#6 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %304 = vector.extract %273[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %305 = amdgpu.scaled_mfma 16x16x128 (%279[0] * %229) * (%304[0] * %257) + %303 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %306 = vector.extract %274[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %307 = amdgpu.scaled_mfma 16x16x128 (%276[0] * %228) * (%306[0] * %258) + %78#7 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %308 = vector.extract %275[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %309 = amdgpu.scaled_mfma 16x16x128 (%279[0] * %229) * (%308[0] * %259) + %307 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %310 = vector.extract %238[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %311 = amdgpu.scaled_mfma 16x16x128 (%310[0] * %230) * (%277[0] * %244) + %78#8 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %312 = vector.extract %239[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %313 = amdgpu.scaled_mfma 16x16x128 (%312[0] * %231) * (%280[0] * %245) + %311 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %314 = amdgpu.scaled_mfma 16x16x128 (%310[0] * %230) * (%282[0] * %246) + %78#9 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %315 = amdgpu.scaled_mfma 16x16x128 (%312[0] * %231) * (%284[0] * %247) + %314 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %316 = amdgpu.scaled_mfma 16x16x128 (%310[0] * %230) * (%286[0] * %248) + %78#10 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %317 = amdgpu.scaled_mfma 16x16x128 (%312[0] * %231) * (%288[0] * %249) + %316 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %318 = amdgpu.scaled_mfma 16x16x128 (%310[0] * %230) * (%290[0] * %250) + %78#11 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %319 = amdgpu.scaled_mfma 16x16x128 (%312[0] * %231) * (%292[0] * %251) + %318 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %320 = amdgpu.scaled_mfma 16x16x128 (%310[0] * %230) * (%294[0] * %252) + %78#12 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %321 = amdgpu.scaled_mfma 16x16x128 (%312[0] * %231) * (%296[0] * %253) + %320 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %322 = amdgpu.scaled_mfma 16x16x128 (%310[0] * %230) * (%298[0] * %254) + %78#13 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %323 = amdgpu.scaled_mfma 16x16x128 (%312[0] * %231) * (%300[0] * %255) + %322 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %324 = amdgpu.scaled_mfma 16x16x128 (%310[0] * %230) * (%302[0] * %256) + %78#14 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %325 = amdgpu.scaled_mfma 16x16x128 (%312[0] * %231) * (%304[0] * %257) + %324 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %326 = amdgpu.scaled_mfma 16x16x128 (%310[0] * %230) * (%306[0] * %258) + %78#15 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %327 = amdgpu.scaled_mfma 16x16x128 (%312[0] * %231) * (%308[0] * %259) + %326 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %328 = vector.extract %240[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %329 = amdgpu.scaled_mfma 16x16x128 (%328[0] * %232) * (%277[0] * %244) + %78#16 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %330 = vector.extract %241[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %331 = amdgpu.scaled_mfma 16x16x128 (%330[0] * %233) * (%280[0] * %245) + %329 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %332 = amdgpu.scaled_mfma 16x16x128 (%328[0] * %232) * (%282[0] * %246) + %78#17 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %333 = amdgpu.scaled_mfma 16x16x128 (%330[0] * %233) * (%284[0] * %247) + %332 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %334 = amdgpu.scaled_mfma 16x16x128 (%328[0] * %232) * (%286[0] * %248) + %78#18 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %335 = amdgpu.scaled_mfma 16x16x128 (%330[0] * %233) * (%288[0] * %249) + %334 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %336 = amdgpu.scaled_mfma 16x16x128 (%328[0] * %232) * (%290[0] * %250) + %78#19 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %337 = amdgpu.scaled_mfma 16x16x128 (%330[0] * %233) * (%292[0] * %251) + %336 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %338 = amdgpu.scaled_mfma 16x16x128 (%328[0] * %232) * (%294[0] * %252) + %78#20 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %339 = amdgpu.scaled_mfma 16x16x128 (%330[0] * %233) * (%296[0] * %253) + %338 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %340 = amdgpu.scaled_mfma 16x16x128 (%328[0] * %232) * (%298[0] * %254) + %78#21 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %341 = amdgpu.scaled_mfma 16x16x128 (%330[0] * %233) * (%300[0] * %255) + %340 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %342 = amdgpu.scaled_mfma 16x16x128 (%328[0] * %232) * (%302[0] * %256) + %78#22 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %343 = amdgpu.scaled_mfma 16x16x128 (%330[0] * %233) * (%304[0] * %257) + %342 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %344 = amdgpu.scaled_mfma 16x16x128 (%328[0] * %232) * (%306[0] * %258) + %78#23 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %345 = amdgpu.scaled_mfma 16x16x128 (%330[0] * %233) * (%308[0] * %259) + %344 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %346 = vector.extract %242[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %347 = amdgpu.scaled_mfma 16x16x128 (%346[0] * %234) * (%277[0] * %244) + %78#24 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %348 = vector.extract %243[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
        %349 = amdgpu.scaled_mfma 16x16x128 (%348[0] * %235) * (%280[0] * %245) + %347 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %350 = amdgpu.scaled_mfma 16x16x128 (%346[0] * %234) * (%282[0] * %246) + %78#25 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %351 = amdgpu.scaled_mfma 16x16x128 (%348[0] * %235) * (%284[0] * %247) + %350 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %352 = amdgpu.scaled_mfma 16x16x128 (%346[0] * %234) * (%286[0] * %248) + %78#26 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %353 = amdgpu.scaled_mfma 16x16x128 (%348[0] * %235) * (%288[0] * %249) + %352 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %354 = amdgpu.scaled_mfma 16x16x128 (%346[0] * %234) * (%290[0] * %250) + %78#27 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %355 = amdgpu.scaled_mfma 16x16x128 (%348[0] * %235) * (%292[0] * %251) + %354 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %356 = amdgpu.scaled_mfma 16x16x128 (%346[0] * %234) * (%294[0] * %252) + %78#28 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %357 = amdgpu.scaled_mfma 16x16x128 (%348[0] * %235) * (%296[0] * %253) + %356 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %358 = amdgpu.scaled_mfma 16x16x128 (%346[0] * %234) * (%298[0] * %254) + %78#29 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %359 = amdgpu.scaled_mfma 16x16x128 (%348[0] * %235) * (%300[0] * %255) + %358 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %360 = amdgpu.scaled_mfma 16x16x128 (%346[0] * %234) * (%302[0] * %256) + %78#30 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %361 = amdgpu.scaled_mfma 16x16x128 (%348[0] * %235) * (%304[0] * %257) + %360 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %362 = amdgpu.scaled_mfma 16x16x128 (%346[0] * %234) * (%306[0] * %258) + %78#31 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %363 = amdgpu.scaled_mfma 16x16x128 (%348[0] * %235) * (%308[0] * %259) + %362 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %364 = vector.extract_strided_slice %281 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %365 = affine.apply #map23()[%block_id_x]
        %366 = affine.apply #map76()[%thread_id_x]
        %367 = affine.apply #map77()[%thread_id_x, %thread_id_y]
        %368 = arith.muli %365, %c1024 overflow<nsw> : index
        %369 = arith.muli %366, %c1024 overflow<nsw> : index
        %370 = arith.addi %368, %108 overflow<nsw> : index
        %371 = arith.addi %369, %367 overflow<nsw> : index
        %reinterpret_cast_13 = memref.reinterpret_cast %4 to offset: [%370], sizes: [536870910], strides: [1] : memref<f32> to memref<536870910xf32, strided<[1], offset: ?>>
        %cast_14 = memref.cast %reinterpret_cast_13 : memref<536870910xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        %372 = amdgpu.fat_raw_buffer_cast %cast_14 validBytes(%c2147483643_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>
        vector.store %364, %372[%371] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %373 = vector.extract_strided_slice %281 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %374 = affine.apply #map78()[%thread_id_x]
        %375 = arith.muli %374, %c1024 overflow<nsw> : index
        %376 = arith.addi %375, %367 overflow<nsw> : index
        vector.store %373, %372[%376] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %377 = vector.extract_strided_slice %281 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %378 = affine.apply #map79()[%thread_id_x]
        %379 = arith.muli %378, %c1024 overflow<nsw> : index
        %380 = arith.addi %379, %367 overflow<nsw> : index
        vector.store %377, %372[%380] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %381 = vector.extract_strided_slice %281 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %382 = affine.apply #map80()[%thread_id_x]
        %383 = arith.muli %382, %c1024 overflow<nsw> : index
        %384 = arith.addi %383, %367 overflow<nsw> : index
        vector.store %381, %372[%384] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %385 = vector.extract_strided_slice %285 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %386 = affine.apply #map81()[%thread_id_x, %thread_id_y]
        %387 = arith.addi %369, %386 overflow<nsw> : index
        vector.store %385, %372[%387] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %388 = vector.extract_strided_slice %285 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %389 = arith.addi %375, %386 overflow<nsw> : index
        vector.store %388, %372[%389] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %390 = vector.extract_strided_slice %285 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %391 = arith.addi %379, %386 overflow<nsw> : index
        vector.store %390, %372[%391] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %392 = vector.extract_strided_slice %285 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %393 = arith.addi %383, %386 overflow<nsw> : index
        vector.store %392, %372[%393] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %394 = vector.extract_strided_slice %289 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %395 = affine.apply #map82()[%thread_id_x, %thread_id_y]
        %396 = arith.addi %369, %395 overflow<nsw> : index
        vector.store %394, %372[%396] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %397 = vector.extract_strided_slice %289 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %398 = arith.addi %375, %395 overflow<nsw> : index
        vector.store %397, %372[%398] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %399 = vector.extract_strided_slice %289 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %400 = arith.addi %379, %395 overflow<nsw> : index
        vector.store %399, %372[%400] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %401 = vector.extract_strided_slice %289 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %402 = arith.addi %383, %395 overflow<nsw> : index
        vector.store %401, %372[%402] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %403 = vector.extract_strided_slice %293 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %404 = affine.apply #map83()[%thread_id_x, %thread_id_y]
        %405 = arith.addi %369, %404 overflow<nsw> : index
        vector.store %403, %372[%405] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %406 = vector.extract_strided_slice %293 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %407 = arith.addi %375, %404 overflow<nsw> : index
        vector.store %406, %372[%407] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %408 = vector.extract_strided_slice %293 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %409 = arith.addi %379, %404 overflow<nsw> : index
        vector.store %408, %372[%409] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %410 = vector.extract_strided_slice %293 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %411 = arith.addi %383, %404 overflow<nsw> : index
        vector.store %410, %372[%411] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %412 = vector.extract_strided_slice %297 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %413 = affine.apply #map84()[%thread_id_x, %thread_id_y]
        %414 = arith.addi %369, %413 overflow<nsw> : index
        vector.store %412, %372[%414] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %415 = vector.extract_strided_slice %297 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %416 = arith.addi %375, %413 overflow<nsw> : index
        vector.store %415, %372[%416] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %417 = vector.extract_strided_slice %297 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %418 = arith.addi %379, %413 overflow<nsw> : index
        vector.store %417, %372[%418] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %419 = vector.extract_strided_slice %297 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %420 = arith.addi %383, %413 overflow<nsw> : index
        vector.store %419, %372[%420] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %421 = vector.extract_strided_slice %301 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %422 = affine.apply #map85()[%thread_id_x, %thread_id_y]
        %423 = arith.addi %369, %422 overflow<nsw> : index
        vector.store %421, %372[%423] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %424 = vector.extract_strided_slice %301 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %425 = arith.addi %375, %422 overflow<nsw> : index
        vector.store %424, %372[%425] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %426 = vector.extract_strided_slice %301 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %427 = arith.addi %379, %422 overflow<nsw> : index
        vector.store %426, %372[%427] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %428 = vector.extract_strided_slice %301 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %429 = arith.addi %383, %422 overflow<nsw> : index
        vector.store %428, %372[%429] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %430 = vector.extract_strided_slice %305 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %431 = affine.apply #map86()[%thread_id_x, %thread_id_y]
        %432 = arith.addi %369, %431 overflow<nsw> : index
        vector.store %430, %372[%432] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %433 = vector.extract_strided_slice %305 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %434 = arith.addi %375, %431 overflow<nsw> : index
        vector.store %433, %372[%434] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %435 = vector.extract_strided_slice %305 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %436 = arith.addi %379, %431 overflow<nsw> : index
        vector.store %435, %372[%436] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %437 = vector.extract_strided_slice %305 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %438 = arith.addi %383, %431 overflow<nsw> : index
        vector.store %437, %372[%438] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %439 = vector.extract_strided_slice %309 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %440 = affine.apply #map87()[%thread_id_x, %thread_id_y]
        %441 = arith.addi %369, %440 overflow<nsw> : index
        vector.store %439, %372[%441] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %442 = vector.extract_strided_slice %309 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %443 = arith.addi %375, %440 overflow<nsw> : index
        vector.store %442, %372[%443] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %444 = vector.extract_strided_slice %309 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %445 = arith.addi %379, %440 overflow<nsw> : index
        vector.store %444, %372[%445] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %446 = vector.extract_strided_slice %309 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %447 = arith.addi %383, %440 overflow<nsw> : index
        vector.store %446, %372[%447] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %448 = vector.extract_strided_slice %313 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %449 = affine.apply #map88()[%thread_id_x]
        %450 = arith.muli %449, %c1024 overflow<nsw> : index
        %451 = arith.addi %450, %367 overflow<nsw> : index
        vector.store %448, %372[%451] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %452 = vector.extract_strided_slice %313 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %453 = affine.apply #map89()[%thread_id_x]
        %454 = arith.muli %453, %c1024 overflow<nsw> : index
        %455 = arith.addi %454, %367 overflow<nsw> : index
        vector.store %452, %372[%455] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %456 = vector.extract_strided_slice %313 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %457 = affine.apply #map90()[%thread_id_x]
        %458 = arith.muli %457, %c1024 overflow<nsw> : index
        %459 = arith.addi %458, %367 overflow<nsw> : index
        vector.store %456, %372[%459] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %460 = vector.extract_strided_slice %313 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %461 = affine.apply #map91()[%thread_id_x]
        %462 = arith.muli %461, %c1024 overflow<nsw> : index
        %463 = arith.addi %462, %367 overflow<nsw> : index
        vector.store %460, %372[%463] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %464 = vector.extract_strided_slice %315 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %465 = arith.addi %450, %386 overflow<nsw> : index
        vector.store %464, %372[%465] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %466 = vector.extract_strided_slice %315 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %467 = arith.addi %454, %386 overflow<nsw> : index
        vector.store %466, %372[%467] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %468 = vector.extract_strided_slice %315 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %469 = arith.addi %458, %386 overflow<nsw> : index
        vector.store %468, %372[%469] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %470 = vector.extract_strided_slice %315 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %471 = arith.addi %462, %386 overflow<nsw> : index
        vector.store %470, %372[%471] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %472 = vector.extract_strided_slice %317 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %473 = arith.addi %450, %395 overflow<nsw> : index
        vector.store %472, %372[%473] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %474 = vector.extract_strided_slice %317 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %475 = arith.addi %454, %395 overflow<nsw> : index
        vector.store %474, %372[%475] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %476 = vector.extract_strided_slice %317 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %477 = arith.addi %458, %395 overflow<nsw> : index
        vector.store %476, %372[%477] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %478 = vector.extract_strided_slice %317 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %479 = arith.addi %462, %395 overflow<nsw> : index
        vector.store %478, %372[%479] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %480 = vector.extract_strided_slice %319 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %481 = arith.addi %450, %404 overflow<nsw> : index
        vector.store %480, %372[%481] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %482 = vector.extract_strided_slice %319 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %483 = arith.addi %454, %404 overflow<nsw> : index
        vector.store %482, %372[%483] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %484 = vector.extract_strided_slice %319 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %485 = arith.addi %458, %404 overflow<nsw> : index
        vector.store %484, %372[%485] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %486 = vector.extract_strided_slice %319 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %487 = arith.addi %462, %404 overflow<nsw> : index
        vector.store %486, %372[%487] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %488 = vector.extract_strided_slice %321 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %489 = arith.addi %450, %413 overflow<nsw> : index
        vector.store %488, %372[%489] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %490 = vector.extract_strided_slice %321 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %491 = arith.addi %454, %413 overflow<nsw> : index
        vector.store %490, %372[%491] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %492 = vector.extract_strided_slice %321 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %493 = arith.addi %458, %413 overflow<nsw> : index
        vector.store %492, %372[%493] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %494 = vector.extract_strided_slice %321 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %495 = arith.addi %462, %413 overflow<nsw> : index
        vector.store %494, %372[%495] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %496 = vector.extract_strided_slice %323 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %497 = arith.addi %450, %422 overflow<nsw> : index
        vector.store %496, %372[%497] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %498 = vector.extract_strided_slice %323 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %499 = arith.addi %454, %422 overflow<nsw> : index
        vector.store %498, %372[%499] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %500 = vector.extract_strided_slice %323 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %501 = arith.addi %458, %422 overflow<nsw> : index
        vector.store %500, %372[%501] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %502 = vector.extract_strided_slice %323 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %503 = arith.addi %462, %422 overflow<nsw> : index
        vector.store %502, %372[%503] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %504 = vector.extract_strided_slice %325 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %505 = arith.addi %450, %431 overflow<nsw> : index
        vector.store %504, %372[%505] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %506 = vector.extract_strided_slice %325 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %507 = arith.addi %454, %431 overflow<nsw> : index
        vector.store %506, %372[%507] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %508 = vector.extract_strided_slice %325 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %509 = arith.addi %458, %431 overflow<nsw> : index
        vector.store %508, %372[%509] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %510 = vector.extract_strided_slice %325 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %511 = arith.addi %462, %431 overflow<nsw> : index
        vector.store %510, %372[%511] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %512 = vector.extract_strided_slice %327 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %513 = arith.addi %450, %440 overflow<nsw> : index
        vector.store %512, %372[%513] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %514 = vector.extract_strided_slice %327 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %515 = arith.addi %454, %440 overflow<nsw> : index
        vector.store %514, %372[%515] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %516 = vector.extract_strided_slice %327 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %517 = arith.addi %458, %440 overflow<nsw> : index
        vector.store %516, %372[%517] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %518 = vector.extract_strided_slice %327 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %519 = arith.addi %462, %440 overflow<nsw> : index
        vector.store %518, %372[%519] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %520 = vector.extract_strided_slice %331 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %521 = affine.apply #map92()[%thread_id_x]
        %522 = arith.muli %521, %c1024 overflow<nsw> : index
        %523 = arith.addi %522, %367 overflow<nsw> : index
        vector.store %520, %372[%523] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %524 = vector.extract_strided_slice %331 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %525 = affine.apply #map93()[%thread_id_x]
        %526 = arith.muli %525, %c1024 overflow<nsw> : index
        %527 = arith.addi %526, %367 overflow<nsw> : index
        vector.store %524, %372[%527] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %528 = vector.extract_strided_slice %331 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %529 = affine.apply #map94()[%thread_id_x]
        %530 = arith.muli %529, %c1024 overflow<nsw> : index
        %531 = arith.addi %530, %367 overflow<nsw> : index
        vector.store %528, %372[%531] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %532 = vector.extract_strided_slice %331 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %533 = affine.apply #map95()[%thread_id_x]
        %534 = arith.muli %533, %c1024 overflow<nsw> : index
        %535 = arith.addi %534, %367 overflow<nsw> : index
        vector.store %532, %372[%535] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %536 = vector.extract_strided_slice %333 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %537 = arith.addi %522, %386 overflow<nsw> : index
        vector.store %536, %372[%537] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %538 = vector.extract_strided_slice %333 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %539 = arith.addi %526, %386 overflow<nsw> : index
        vector.store %538, %372[%539] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %540 = vector.extract_strided_slice %333 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %541 = arith.addi %530, %386 overflow<nsw> : index
        vector.store %540, %372[%541] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %542 = vector.extract_strided_slice %333 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %543 = arith.addi %534, %386 overflow<nsw> : index
        vector.store %542, %372[%543] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %544 = vector.extract_strided_slice %335 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %545 = arith.addi %522, %395 overflow<nsw> : index
        vector.store %544, %372[%545] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %546 = vector.extract_strided_slice %335 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %547 = arith.addi %526, %395 overflow<nsw> : index
        vector.store %546, %372[%547] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %548 = vector.extract_strided_slice %335 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %549 = arith.addi %530, %395 overflow<nsw> : index
        vector.store %548, %372[%549] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %550 = vector.extract_strided_slice %335 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %551 = arith.addi %534, %395 overflow<nsw> : index
        vector.store %550, %372[%551] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %552 = vector.extract_strided_slice %337 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %553 = arith.addi %522, %404 overflow<nsw> : index
        vector.store %552, %372[%553] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %554 = vector.extract_strided_slice %337 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %555 = arith.addi %526, %404 overflow<nsw> : index
        vector.store %554, %372[%555] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %556 = vector.extract_strided_slice %337 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %557 = arith.addi %530, %404 overflow<nsw> : index
        vector.store %556, %372[%557] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %558 = vector.extract_strided_slice %337 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %559 = arith.addi %534, %404 overflow<nsw> : index
        vector.store %558, %372[%559] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %560 = vector.extract_strided_slice %339 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %561 = arith.addi %522, %413 overflow<nsw> : index
        vector.store %560, %372[%561] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %562 = vector.extract_strided_slice %339 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %563 = arith.addi %526, %413 overflow<nsw> : index
        vector.store %562, %372[%563] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %564 = vector.extract_strided_slice %339 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %565 = arith.addi %530, %413 overflow<nsw> : index
        vector.store %564, %372[%565] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %566 = vector.extract_strided_slice %339 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %567 = arith.addi %534, %413 overflow<nsw> : index
        vector.store %566, %372[%567] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %568 = vector.extract_strided_slice %341 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %569 = arith.addi %522, %422 overflow<nsw> : index
        vector.store %568, %372[%569] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %570 = vector.extract_strided_slice %341 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %571 = arith.addi %526, %422 overflow<nsw> : index
        vector.store %570, %372[%571] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %572 = vector.extract_strided_slice %341 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %573 = arith.addi %530, %422 overflow<nsw> : index
        vector.store %572, %372[%573] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %574 = vector.extract_strided_slice %341 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %575 = arith.addi %534, %422 overflow<nsw> : index
        vector.store %574, %372[%575] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %576 = vector.extract_strided_slice %343 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %577 = arith.addi %522, %431 overflow<nsw> : index
        vector.store %576, %372[%577] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %578 = vector.extract_strided_slice %343 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %579 = arith.addi %526, %431 overflow<nsw> : index
        vector.store %578, %372[%579] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %580 = vector.extract_strided_slice %343 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %581 = arith.addi %530, %431 overflow<nsw> : index
        vector.store %580, %372[%581] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %582 = vector.extract_strided_slice %343 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %583 = arith.addi %534, %431 overflow<nsw> : index
        vector.store %582, %372[%583] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %584 = vector.extract_strided_slice %345 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %585 = arith.addi %522, %440 overflow<nsw> : index
        vector.store %584, %372[%585] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %586 = vector.extract_strided_slice %345 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %587 = arith.addi %526, %440 overflow<nsw> : index
        vector.store %586, %372[%587] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %588 = vector.extract_strided_slice %345 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %589 = arith.addi %530, %440 overflow<nsw> : index
        vector.store %588, %372[%589] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %590 = vector.extract_strided_slice %345 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %591 = arith.addi %534, %440 overflow<nsw> : index
        vector.store %590, %372[%591] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %592 = vector.extract_strided_slice %349 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %593 = affine.apply #map96()[%thread_id_x]
        %594 = arith.muli %593, %c1024 overflow<nsw> : index
        %595 = arith.addi %594, %367 overflow<nsw> : index
        vector.store %592, %372[%595] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %596 = vector.extract_strided_slice %349 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %597 = affine.apply #map97()[%thread_id_x]
        %598 = arith.muli %597, %c1024 overflow<nsw> : index
        %599 = arith.addi %598, %367 overflow<nsw> : index
        vector.store %596, %372[%599] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %600 = vector.extract_strided_slice %349 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %601 = affine.apply #map98()[%thread_id_x]
        %602 = arith.muli %601, %c1024 overflow<nsw> : index
        %603 = arith.addi %602, %367 overflow<nsw> : index
        vector.store %600, %372[%603] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %604 = vector.extract_strided_slice %349 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %605 = affine.apply #map99()[%thread_id_x]
        %606 = arith.muli %605, %c1024 overflow<nsw> : index
        %607 = arith.addi %606, %367 overflow<nsw> : index
        vector.store %604, %372[%607] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %608 = vector.extract_strided_slice %351 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %609 = arith.addi %594, %386 overflow<nsw> : index
        vector.store %608, %372[%609] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %610 = vector.extract_strided_slice %351 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %611 = arith.addi %598, %386 overflow<nsw> : index
        vector.store %610, %372[%611] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %612 = vector.extract_strided_slice %351 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %613 = arith.addi %602, %386 overflow<nsw> : index
        vector.store %612, %372[%613] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %614 = vector.extract_strided_slice %351 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %615 = arith.addi %606, %386 overflow<nsw> : index
        vector.store %614, %372[%615] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %616 = vector.extract_strided_slice %353 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %617 = arith.addi %594, %395 overflow<nsw> : index
        vector.store %616, %372[%617] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %618 = vector.extract_strided_slice %353 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %619 = arith.addi %598, %395 overflow<nsw> : index
        vector.store %618, %372[%619] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %620 = vector.extract_strided_slice %353 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %621 = arith.addi %602, %395 overflow<nsw> : index
        vector.store %620, %372[%621] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %622 = vector.extract_strided_slice %353 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %623 = arith.addi %606, %395 overflow<nsw> : index
        vector.store %622, %372[%623] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %624 = vector.extract_strided_slice %355 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %625 = arith.addi %594, %404 overflow<nsw> : index
        vector.store %624, %372[%625] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %626 = vector.extract_strided_slice %355 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %627 = arith.addi %598, %404 overflow<nsw> : index
        vector.store %626, %372[%627] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %628 = vector.extract_strided_slice %355 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %629 = arith.addi %602, %404 overflow<nsw> : index
        vector.store %628, %372[%629] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %630 = vector.extract_strided_slice %355 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %631 = arith.addi %606, %404 overflow<nsw> : index
        vector.store %630, %372[%631] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %632 = vector.extract_strided_slice %357 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %633 = arith.addi %594, %413 overflow<nsw> : index
        vector.store %632, %372[%633] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %634 = vector.extract_strided_slice %357 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %635 = arith.addi %598, %413 overflow<nsw> : index
        vector.store %634, %372[%635] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %636 = vector.extract_strided_slice %357 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %637 = arith.addi %602, %413 overflow<nsw> : index
        vector.store %636, %372[%637] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %638 = vector.extract_strided_slice %357 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %639 = arith.addi %606, %413 overflow<nsw> : index
        vector.store %638, %372[%639] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %640 = vector.extract_strided_slice %359 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %641 = arith.addi %594, %422 overflow<nsw> : index
        vector.store %640, %372[%641] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %642 = vector.extract_strided_slice %359 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %643 = arith.addi %598, %422 overflow<nsw> : index
        vector.store %642, %372[%643] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %644 = vector.extract_strided_slice %359 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %645 = arith.addi %602, %422 overflow<nsw> : index
        vector.store %644, %372[%645] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %646 = vector.extract_strided_slice %359 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %647 = arith.addi %606, %422 overflow<nsw> : index
        vector.store %646, %372[%647] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %648 = vector.extract_strided_slice %361 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %649 = arith.addi %594, %431 overflow<nsw> : index
        vector.store %648, %372[%649] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %650 = vector.extract_strided_slice %361 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %651 = arith.addi %598, %431 overflow<nsw> : index
        vector.store %650, %372[%651] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %652 = vector.extract_strided_slice %361 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %653 = arith.addi %602, %431 overflow<nsw> : index
        vector.store %652, %372[%653] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %654 = vector.extract_strided_slice %361 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %655 = arith.addi %606, %431 overflow<nsw> : index
        vector.store %654, %372[%655] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %656 = vector.extract_strided_slice %363 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %657 = arith.addi %594, %440 overflow<nsw> : index
        vector.store %656, %372[%657] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %658 = vector.extract_strided_slice %363 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %659 = arith.addi %598, %440 overflow<nsw> : index
        vector.store %658, %372[%659] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %660 = vector.extract_strided_slice %363 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %661 = arith.addi %602, %440 overflow<nsw> : index
        vector.store %660, %372[%661] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %662 = vector.extract_strided_slice %363 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %663 = arith.addi %606, %440 overflow<nsw> : index
        vector.store %662, %372[%663] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.fence, %arg6: !hal.fence) -> !hal.buffer_view {
    %0 = hal.tensor.import wait(%arg5) => %arg0 : !hal.buffer_view -> tensor<1024x4096xi8>
    %1 = hal.tensor.import wait(%arg5) => %arg1 : !hal.buffer_view -> tensor<1024x256xi8>
    %2 = hal.tensor.import wait(%arg5) => %arg2 : !hal.buffer_view -> tensor<1024x4096xi8>
    %3 = hal.tensor.import wait(%arg5) => %arg3 : !hal.buffer_view -> tensor<1024x256xi8>
    %4 = hal.tensor.import wait(%arg5) => %arg4 : !hal.buffer_view -> tensor<1024x1024xf32>
    %5 = flow.dispatch @gemm::@gemm(%0, %1, %2, %3, %4) : (tensor<1024x4096xi8>, tensor<1024x256xi8>, tensor<1024x4096xi8>, tensor<1024x256xi8>, tensor<1024x1024xf32>) -> %4
    %6 = hal.tensor.barrier join(%5 : tensor<1024x1024xf32>) => %arg6 : !hal.fence
    %7 = hal.tensor.export %6 : tensor<1024x1024xf32> -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
}
