// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "mlir/Pass/PassBase.td"

def TestInUseAnalysisPass : Pass<"water-test-in-use-analysis"> {
  let summary = "Test pass for value in-use analysis";
  let description = [{
    Runs the value in-use dataflow analysis and attaches attributes to
    operations indicating whether their results and block arguments of nested
    regions are identified as in-use, not-in-use or unknown.
  }];
}

def TestWaveDialectConstructorsPass : Pass<"water-test-wave-dialect-constructors"> {
  let summary = "Test error messages from Wave dialect constructors";
  let description = [{
    Construct attributes and types from primitive elements using `getChecked` to
    trigger error emission instead of assertions.
  }];

  let dependentDialects = ["::wave::WaveDialect"];
}

def TestWaveDialectInferIndexExprsPass : Pass<"water-test-wave-infer-index-exprs"> {
  let summary =
    "Test lattice operations in Wave dialect index expression propagation";
  let description = [{
    Similar as normal water-wave-infer-index-exprs pass, but joins lattice
    values for operation results and block arguments after initialization with
    optional values provided in the following attributes:

      - "wave.override_result_index" for results;
      - "wave_test.override_operand_index" for operands;
      - "wave.override_regionN_blockM_args" for arguments of the M-th block
        of N-th region;

    These attributes are expected to be lists either a unit attribute (ignored)
    or an attribute similar to that produced by water-wave-infer-index-exprs
    pass. The array is coindexed with the corresponding op results or block
    arguments.
  }];
}
