// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::affine;

namespace mlir::water {
#define GEN_PASS_DEF_WATERCOMPOSEAFFINEARITHPASS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

namespace {

template <typename ArithOp, typename CombineFn>
struct ComposeArithWithAffineApply : public OpRewritePattern<ArithOp> {
  using OpRewritePattern<ArithOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ArithOp arithOp,
                                PatternRewriter &rewriter) const override {
    // Check if the operation has nsw flag.
    if (arithOp.getOverflowFlags() != arith::IntegerOverflowFlags::nsw)
      return failure();

    Value lhs = arithOp.getLhs();
    Value rhs = arithOp.getRhs();

    // Try to find affine.apply on either side.
    auto lhsAffine = lhs.getDefiningOp<AffineApplyOp>();
    auto rhsAffine = rhs.getDefiningOp<AffineApplyOp>();

    // At least one operand must be affine.apply.
    if (!lhsAffine && !rhsAffine)
      return failure();

    AffineApplyOp affineOp;
    Value otherOperand;

    if (lhsAffine) {
      affineOp = lhsAffine;
      otherOperand = rhs;
    } else {
      affineOp = rhsAffine;
      otherOperand = lhs;
    }

    // Get the affine map and operands from the affine.apply.
    AffineMap map = affineOp.getAffineMap();
    SmallVector<Value> mapOperands(affineOp.getMapOperands());

    // Create a new symbol for the other operand.
    unsigned numDims = map.getNumDims();
    unsigned numSymbols = map.getNumSymbols();

    // Get the affine expression and combine with the new symbol.
    AffineExpr expr = map.getResult(0);
    AffineExpr newSymbol =
        getAffineSymbolExpr(numSymbols, rewriter.getContext());
    AffineExpr newExpr = CombineFn()(expr, newSymbol);

    // Create new affine map with additional symbol.
    AffineMap newMap = AffineMap::get(numDims, numSymbols + 1, newExpr);

    // Add the other operand as a new symbol operand.
    mapOperands.push_back(otherOperand);

    // Create new affine.apply op.
    rewriter.replaceOpWithNewOp<AffineApplyOp>(arithOp, newMap, mapOperands);

    return success();
  }
};

struct AffineAdd {
  AffineExpr operator()(AffineExpr lhs, AffineExpr rhs) const {
    return lhs + rhs;
  }
};

struct AffineMul {
  AffineExpr operator()(AffineExpr lhs, AffineExpr rhs) const {
    return lhs * rhs;
  }
};

} // namespace

class ComposeAffineArithPass
    : public water::impl::WaterComposeAffineArithPassBase<
          ComposeAffineArithPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ComposeArithWithAffineApply<arith::AddIOp, AffineAdd>,
                 ComposeArithWithAffineApply<arith::MulIOp, AffineMul>>(
        &getContext());

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
