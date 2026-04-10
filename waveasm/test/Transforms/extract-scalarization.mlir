// RUN: waveasm-translate --waveasm-extract-scalarization %s | FileCheck %s
//
// Test: extract scalarization pre-translation pass.
// Note: without a follow-up canonicalization, dead vector ops and
// trivial addi(x, 0) remain but are harmless.

module {
  gpu.module @test_extract_scalarization {

    // ---------------------------------------------------------------
    // Positive: extract[0] from addi(broadcast(x), iota)
    // iota[0] == 0, so the scalar result is addi(x, 0).
    // ---------------------------------------------------------------
    // CHECK-LABEL: func @extract_zero
    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    // CHECK: %[[ADD:.*]] = arith.addi %arg0, %[[C0]] : i32
    // CHECK: return %[[ADD]]
    func.func @extract_zero(%x: i32) -> i32 {
      %bcast = vector.broadcast %x : i32 to vector<4xi32>
      %iota = arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
      %sum = arith.addi %bcast, %iota : vector<4xi32>
      %r = vector.extract %sum[0] : i32 from vector<4xi32>
      return %r : i32
    }

    // ---------------------------------------------------------------
    // Positive: extract[2] from addi(broadcast(x), dense<[10,11,12,13]>)
    // ---------------------------------------------------------------
    // CHECK-LABEL: func @extract_two
    // CHECK: %[[C12:.*]] = arith.constant 12 : i32
    // CHECK: %[[ADD:.*]] = arith.addi %arg0, %[[C12]] : i32
    // CHECK: return %[[ADD]]
    func.func @extract_two(%x: i32) -> i32 {
      %bcast = vector.broadcast %x : i32 to vector<4xi32>
      %offsets = arith.constant dense<[10, 11, 12, 13]> : vector<4xi32>
      %sum = arith.addi %bcast, %offsets : vector<4xi32>
      %r = vector.extract %sum[2] : i32 from vector<4xi32>
      return %r : i32
    }

    // ---------------------------------------------------------------
    // Positive: extract[0] through index_cast and select (buffer_ops pattern)
    // ---------------------------------------------------------------
    // CHECK-LABEL: func @extract_with_select
    // CHECK: arith.addi %arg0,
    // CHECK: %[[SENT:.*]] = arith.constant 2147483647 : i32
    // CHECK: arith.select %arg1,
    func.func @extract_with_select(%x: i32, %mask: i1) -> i32 {
      %bcast = vector.broadcast %x : i32 to vector<4xi32>
      %iota = arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
      %sum = arith.addi %bcast, %iota : vector<4xi32>
      %sum_idx = arith.index_cast %sum : vector<4xi32> to vector<4xindex>
      %sentinel = arith.constant dense<2147483647> : vector<4xindex>
      %sel = arith.select %mask, %sum_idx, %sentinel : vector<4xindex>
      %lane = vector.extract %sel[0] : index from vector<4xindex>
      %r = arith.index_cast %lane : index to i32
      return %r : i32
    }

    // ---------------------------------------------------------------
    // Positive: multiple extracts at different indices from same vector
    // ---------------------------------------------------------------
    // CHECK-LABEL: func @extract_multiple
    // CHECK: %[[C5:.*]] = arith.constant 5 : i32
    // CHECK: arith.addi %arg0, %[[C5]] : i32
    // CHECK: %[[C7:.*]] = arith.constant 7 : i32
    // CHECK: arith.addi %arg0, %[[C7]] : i32
    func.func @extract_multiple(%x: i32) -> i32 {
      %bcast = vector.broadcast %x : i32 to vector<4xi32>
      %offsets = arith.constant dense<[5, 5, 7, 7]> : vector<4xi32>
      %sum = arith.addi %bcast, %offsets : vector<4xi32>
      %a = vector.extract %sum[0] : i32 from vector<4xi32>
      %b = vector.extract %sum[2] : i32 from vector<4xi32>
      %r = arith.addi %a, %b : i32
      return %r : i32
    }

    // ---------------------------------------------------------------
    // Splat dense constant: now scalarized via general fallback
    // ---------------------------------------------------------------
    // CHECK-LABEL: func @no_rewrite_splat
    // CHECK: arith.constant 42 : i32
    // CHECK: arith.addi
    func.func @no_rewrite_splat(%x: i32) -> i32 {
      %bcast = vector.broadcast %x : i32 to vector<4xi32>
      %splat = arith.constant dense<42> : vector<4xi32>
      %sum = arith.addi %bcast, %splat : vector<4xi32>
      %r = vector.extract %sum[0] : i32 from vector<4xi32>
      return %r : i32
    }
  }
}
