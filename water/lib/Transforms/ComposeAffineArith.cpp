// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::water {
#define GEN_PASS_DEF_WATERCOMPOSEAFFINEARITHPASS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

class ComposeAffineArithPass
    : public water::impl::WaterComposeAffineArithPassBase<
          ComposeAffineArithPass> {
public:
  void runOnOperation() override {
    // TODO: Implement the transformation logic.
    // Walk through arith operations and compose them with affine.apply ops.
  }
};
