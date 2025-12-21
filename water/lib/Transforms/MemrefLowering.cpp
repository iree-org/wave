// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::water {
#define GEN_PASS_DEF_WATERMEMREFLOWERINGPASS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

namespace {

class MemrefLoweringPass
    : public water::impl::WaterMemrefLoweringPassBase<MemrefLoweringPass> {
public:
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    // TODO: Add patterns here.

    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
