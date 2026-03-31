// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/Transforms/Passes.h"

#define DEBUG_TYPE "wave-expand-variadic-reductions"

namespace wave {
#define GEN_PASS_DEF_WATERWAVEEXPANDVARIADICREDUCTIONSPASS
#include "water/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace wave

using namespace mlir;
using namespace wave;

namespace {

/// Expand a variadic reduction with N inputs into N chained single-input
/// reductions:
///   %r = wave.sum %a, %b, %c init(%init) <scope>
/// becomes:
///   %0 = wave.sum %a init(%init) <scope>
///   %1 = wave.sum %b init(%0) <scope>
///   %r = wave.sum %c init(%1) <scope>
template <typename ReductionOp>
struct ExpandVariadicReduction : public OpRewritePattern<ReductionOp> {
  using OpRewritePattern<ReductionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReductionOp op,
                                PatternRewriter &rewriter) const override {
    OperandRange inputs = op.getInputs();
    if (inputs.size() <= 1)
      return failure();

    Value acc = op.getInit();
    Value result;
    for (Value input : inputs) {
      auto newOp =
          ReductionOp::create(rewriter, op.getLoc(), op.getResult().getType(),
                              input, acc, op.getScopeAttr(), op.getAxisAttr(),
                              op.getIndexAttr(), op.getVectorShapeAttr());
      result = newOp.getResult();
      acc = result;
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ExpandVariadicReductions
    : public wave::impl::WaterWaveExpandVariadicReductionsPassBase<
          ExpandVariadicReductions> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ExpandVariadicReduction<SumOp>,
                 ExpandVariadicReduction<MaxElementOp>>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
