// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::affine;

namespace mlir::water {
#define GEN_PASS_DEF_WATERREORDERAFFINEPASS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

class ReorderAffinePass
    : public water::impl::WaterReorderAffinePassBase<ReorderAffinePass> {
public:
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    getOperation()->walk([&](AffineApplyOp applyOp) {
      reorderOperandsByHoistability(rewriter, applyOp);
    });
  }
};
