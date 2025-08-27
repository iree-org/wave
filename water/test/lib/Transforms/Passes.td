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
