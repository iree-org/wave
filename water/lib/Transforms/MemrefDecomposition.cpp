// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::water {
#define GEN_PASS_DEF_WATERMEMREFDECOMPOSITIONPASS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

namespace {

/// Pattern to decompose memref.load operations by materializing index
/// calculations.
struct DecomposeMemrefLoadOp : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    // TODO: Implement index calculation materialization.
    // For now, this is just a stub.
    return failure();
  }
};

/// Pattern to decompose memref.store operations by materializing index
/// calculations.
struct DecomposeMemrefStoreOp : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    // TODO: Implement index calculation materialization.
    // For now, this is just a stub.
    return failure();
  }
};

class MemrefDecompositionPass
    : public water::impl::WaterMemrefDecompositionPassBase<
          MemrefDecompositionPass> {
public:
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<DecomposeMemrefLoadOp, DecomposeMemrefStoreOp>(ctx);

    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
