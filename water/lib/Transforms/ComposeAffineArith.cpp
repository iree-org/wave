// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
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

// Collect all terms from a commutative binary operation chain.
void collectCommutativeTerms(AffineExpr expr, AffineExprKind kind,
                             SmallVectorImpl<AffineExpr> &terms) {
  if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
    if (binExpr.getKind() == kind) {
      collectCommutativeTerms(binExpr.getLHS(), kind, terms);
      collectCommutativeTerms(binExpr.getRHS(), kind, terms);
      return;
    }
  }
  terms.push_back(expr);
}

// Get positions of dims/symbols used in an affine expression.
void getUsedOperands(AffineExpr expr, SmallVectorImpl<unsigned> &dimPositions,
                     SmallVectorImpl<unsigned> &symPositions) {
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    dimPositions.push_back(dimExpr.getPosition());
  } else if (auto symExpr = dyn_cast<AffineSymbolExpr>(expr)) {
    symPositions.push_back(symExpr.getPosition());
  } else if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
    getUsedOperands(binExpr.getLHS(), dimPositions, symPositions);
    getUsedOperands(binExpr.getRHS(), dimPositions, symPositions);
  }
}

// Compute hoistability score for an affine expression term.
// Returns the minimum number of enclosing invariant loops across all operands.
int64_t getTermHoistability(AffineExpr term, AffineApplyOp applyOp) {
  SmallVector<unsigned> dimPositions, symPositions;
  getUsedOperands(term, dimPositions, symPositions);

  // If it's a constant, it's maximally hoistable.
  if (dimPositions.empty() && symPositions.empty())
    return INT64_MAX;

  int64_t minHoistability = INT64_MAX;

  // Check hoistability of dimension operands.
  for (unsigned pos : dimPositions) {
    OpOperand &operand = applyOp->getOpOperand(pos);
    int64_t h = numEnclosingInvariantLoops(operand);
    minHoistability = std::min(minHoistability, h);
  }

  // Check hoistability of symbol operands.
  unsigned numDims = applyOp.getAffineMap().getNumDims();
  for (unsigned pos : symPositions) {
    OpOperand &operand = applyOp->getOpOperand(numDims + pos);
    int64_t h = numEnclosingInvariantLoops(operand);
    minHoistability = std::min(minHoistability, h);
  }

  return minHoistability;
}

// Rebuild a commutative binary expression from sorted terms.
// Most hoistable terms first, least hoistable last.
// Build as right-associative tree to preserve order.
AffineExpr rebuildCommutativeExpr(ArrayRef<AffineExpr> terms,
                                  AffineExprKind kind) {
  assert(!terms.empty() && "Cannot rebuild from empty terms");
  if (terms.size() == 1)
    return terms[0];

  // Build right-associative: ((a + b) + c) + d.
  // This puts the least hoistable term (last in array) deepest in the tree.
  AffineExpr result = terms[0];
  for (unsigned i = 1; i < terms.size(); ++i) {
    result = getAffineBinaryOpExpr(kind, result, terms[i]);
  }
  return result;
}

// Recursively reorder commutative operations in an affine expression.
AffineExpr reorderCommutativeOps(AffineExpr expr, AffineApplyOp applyOp) {
  if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
    AffineExprKind kind = binExpr.getKind();

    // Only reorder commutative operations.
    if (kind == AffineExprKind::Add || kind == AffineExprKind::Mul) {
      // Collect all terms in the commutative chain.
      SmallVector<AffineExpr> terms;
      collectCommutativeTerms(expr, kind, terms);

      // Recursively reorder each term.
      for (auto &term : terms)
        term = reorderCommutativeOps(term, applyOp);

      // Sort terms by hoistability (most hoistable first, least hoistable
      // last).
      llvm::stable_sort(terms, [&](AffineExpr a, AffineExpr b) {
        return getTermHoistability(a, applyOp) >
               getTermHoistability(b, applyOp);
      });

      // Rebuild the expression with sorted terms.
      return rebuildCommutativeExpr(terms, kind);
    }

    // For non-commutative operations, recursively reorder children.
    return getAffineBinaryOpExpr(
        kind, reorderCommutativeOps(binExpr.getLHS(), applyOp),
        reorderCommutativeOps(binExpr.getRHS(), applyOp));
  }

  return expr;
}

struct ReorderAffineApplyOperands : public OpRewritePattern<AffineApplyOp> {
  using OpRewritePattern<AffineApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    AffineMap map = applyOp.getAffineMap();
    if (map.getNumResults() != 1)
      return failure();

    AffineExpr expr = map.getResult(0);
    AffineExpr reorderedExpr = reorderCommutativeOps(expr, applyOp);

    // Check if the expression changed.
    if (reorderedExpr == expr)
      return failure();

    // Create new affine map with reordered expression.
    AffineMap newMap =
        AffineMap::get(map.getNumDims(), map.getNumSymbols(), reorderedExpr);

    // Replace with new affine.apply.
    rewriter.replaceOpWithNewOp<AffineApplyOp>(applyOp, newMap,
                                               applyOp.getMapOperands());
    return success();
  }
};

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
    patterns.add<ReorderAffineApplyOperands,
                 ComposeArithWithAffineApply<arith::AddIOp, AffineAdd>,
                 ComposeArithWithAffineApply<arith::MulIOp, AffineMul>>(
        &getContext());

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
