// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_PASSES
#define WATER_PASSES

include "mlir/Pass/PassBase.td"

def WaterAssertInBoundsPass : Pass<"water-assert-in-bounds"> {
  let summary = "Insert assertions that memref accesses are in bounds.";
  let dependentDialects = [
    "arith::ArithDialect",
    "cf::ControlFlowDialect",
    "memref::MemRefDialect",
  ];
  let options = [
    Option<"warnOnUnknown", "warn-on-unknown", "bool", /*default=*/"true",
      "Warn on unknown side-effecting operations">,
    Option<"includeVectorLoadStore", "include-vector-load-store", "bool",
      /*default=*/"false",
      "Include vector.load/store operations despite them allowing "
      "out-of-bounds">,
    Option<"checkEachDim", "check-each-dim", "bool", /*default=*/"true",
      "Check each dimension individually">,
    // TODO: consider making this a separate pass as it has to operate at
    // module level.
    Option<"createSpeculativeFuncs", "create-speculative-funcs", "bool",
      /*default=*/"false",
      "Create a function that performs assertions speculatively instead of "
      "in-place checks">
  ];
}

def WaterCheckStaticAssertionsPass : Pass<"water-check-static-assertions"> {
  let summary = "Fail if the IR contains a known-false assertion.";
}

def WaterGreedySLPVectorizer : Pass<"water-greedy-slp-vectorizer"> {
  let summary = "SLP Vectorizer Pass";
  let description = [{
    This pass implements the SLP (Superword Level Parallelism) vectorizer.
    It detects consecutive operations that can be put together into vector
    operations. The pass works bi-directionaly, starting from reads or stores,
    in search of scalars to combine.

    This is greedy vectorizer, it doesn't have any cost model (yet) and it tries
    to create vector ops if we have at least 2 potential ops.
  }];
  let dependentDialects = ["mlir::vector::VectorDialect"];

  let options = [
    Option<"maxVectorBitwidth", "max-vector-bitwidth", "unsigned",
      /*default=*/"std::numeric_limits<unsigned>::max()",
      "Maximum supported vector bitwidth">,
  ];
}

#endif // WATER_PASSES
