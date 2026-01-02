// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/DebugLog.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::affine;

#define DEBUG_TYPE "water-reorder-affine-exprs"

namespace mlir::water {
#define GEN_PASS_DEF_WATERREORDERAFFINEEXPRSPASS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

// Context for stable hashing - maps pointers to indices.
struct HashContext {
  llvm::DenseMap<const void *, unsigned> ptrIndices;

  unsigned getIndex(const void *ptr) {
    auto [it, inserted] = ptrIndices.try_emplace(ptr, ptrIndices.size());
    return it->second;
  }

  unsigned getValueIndex(Value val) {
    return getIndex(val.getAsOpaquePointer());
  }

  unsigned getExprIndex(AffineExpr expr) {
    return getIndex(expr.getAsOpaquePointer());
  }
};

// Collect all terms from a commutative binary operation chain.
static void collectCommutativeTerms(AffineExpr expr, AffineExprKind kind,
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
static void getUsedOperands(AffineExpr expr,
                            SmallVectorImpl<unsigned> &dimPositions,
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
static int64_t getTermHoistability(AffineExpr term, AffineApplyOp applyOp) {
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
static AffineExpr rebuildCommutativeExpr(ArrayRef<AffineExpr> terms,
                                         AffineExprKind kind) {
  assert(!terms.empty() && "Cannot rebuild from empty terms");
  if (terms.size() == 1)
    return terms[0];

  // Build right-associative: ((a + b) + c) + d.
  // This puts the least hoistable term (last in array) deepest in the tree.
  AffineExpr result = terms[0];
  for (auto i : llvm::seq<unsigned>(1, terms.size()))
    result = getAffineBinaryOpExpr(kind, result, terms[i]);

  return result;
}

// Compute hash for affine expression including operand Values (read-only).
static size_t computeExprHash(AffineExpr expr, AffineApplyOp applyOp,
                              HashContext &ctx) {
  if (auto constExpr = dyn_cast<AffineConstantExpr>(expr))
    return llvm::hash_combine(AffineExprKind::Constant, constExpr.getValue());

  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    Value operand = applyOp.getMapOperands()[dimExpr.getPosition()];
    unsigned valueIdx = ctx.getValueIndex(operand);
    return llvm::hash_combine(AffineExprKind::DimId, valueIdx);
  }

  if (auto symExpr = dyn_cast<AffineSymbolExpr>(expr)) {
    unsigned numDims = applyOp.getAffineMap().getNumDims();
    Value operand = applyOp.getMapOperands()[numDims + symExpr.getPosition()];
    unsigned valueIdx = ctx.getValueIndex(operand);
    return llvm::hash_combine(AffineExprKind::SymbolId, valueIdx);
  }

  if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
    size_t lhsHash = computeExprHash(binExpr.getLHS(), applyOp, ctx);
    size_t rhsHash = computeExprHash(binExpr.getRHS(), applyOp, ctx);
    return llvm::hash_combine(binExpr.getKind(), lhsHash, rhsHash);
  }

  return 0;
}

// Compute total score for expression by summing hit counts of all
// sub-expressions.
static unsigned
computeHashScore(AffineExpr expr, AffineApplyOp applyOp, HashContext &ctx,
                 const llvm::SmallDenseMap<size_t, unsigned> &stats) {
  size_t hash = computeExprHash(expr, applyOp, ctx);
  unsigned score = stats.lookup(hash);

  if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
    score += computeHashScore(binExpr.getLHS(), applyOp, ctx, stats);
    score += computeHashScore(binExpr.getRHS(), applyOp, ctx, stats);
  }

  return score;
}

// Recursively compute hash for affine expression including operand Values.
// Populates statistics for all sub-expressions.
static size_t
hashExprWithOperands(AffineExpr expr, AffineApplyOp applyOp, HashContext &ctx,
                     llvm::SmallDenseMap<size_t, unsigned> &stats) {
  size_t hash = computeExprHash(expr, applyOp, ctx);

  // Track this sub-expression in statistics.
  stats[hash]++;

  // Recurse into sub-expressions to track them too.
  if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
    hashExprWithOperands(binExpr.getLHS(), applyOp, ctx, stats);
    hashExprWithOperands(binExpr.getRHS(), applyOp, ctx, stats);
  }

  return hash;
}

// Recursively reorder commutative operations in an affine expression.
// Tries all permutations to maximize hash hits.
static AffineExpr
reorderCommutativeOps(AffineExpr expr, AffineApplyOp applyOp, HashContext &ctx,
                      const llvm::SmallDenseMap<size_t, unsigned> &stats) {
  if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
    AffineExprKind kind = binExpr.getKind();

    // Only reorder commutative operations.
    if (kind == AffineExprKind::Add || kind == AffineExprKind::Mul) {
      // Collect all terms in the commutative chain.
      SmallVector<AffineExpr> terms;
      collectCommutativeTerms(expr, kind, terms);

      // Recursively reorder each term.
      for (auto &term : terms)
        term = reorderCommutativeOps(term, applyOp, ctx, stats);

      // Rebuild in original order (with recursively reordered subterms).
      AffineExpr originalReordered = rebuildCommutativeExpr(terms, kind);
      unsigned originalScore =
          computeHashScore(originalReordered, applyOp, ctx, stats);

      // Sort for initial canonical ordering using stable indices.
      llvm::stable_sort(terms, [&](AffineExpr a, AffineExpr b) {
        return ctx.getExprIndex(a) < ctx.getExprIndex(b);
      });

      // Try all permutations and choose the one with maximum hash hits.
      AffineExpr bestExpr = rebuildCommutativeExpr(terms, kind);
      unsigned bestScore = computeHashScore(bestExpr, applyOp, ctx, stats);

      do {
        AffineExpr candidate = rebuildCommutativeExpr(terms, kind);
        unsigned score = computeHashScore(candidate, applyOp, ctx, stats);
        if (score > bestScore) {
          bestScore = score;
          bestExpr = candidate;
        }
      } while (std::next_permutation(
          terms.begin(), terms.end(), [&](AffineExpr a, AffineExpr b) {
            return ctx.getExprIndex(a) < ctx.getExprIndex(b);
          }));

      // If best score equals original score, return the original form.
      if (bestScore == originalScore)
        return originalReordered;

      return bestExpr;
    }

    // For non-commutative operations, recursively reorder children.
    return getAffineBinaryOpExpr(
        kind, reorderCommutativeOps(binExpr.getLHS(), applyOp, ctx, stats),
        reorderCommutativeOps(binExpr.getRHS(), applyOp, ctx, stats));
  }

  return expr;
}

namespace {

class ReorderAffineExprsPass
    : public water::impl::WaterReorderAffineExprsPassBase<
          ReorderAffineExprsPass> {
public:
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    SmallVector<std::pair<AffineApplyOp, AffineExpr>> opsToRewrite;
    llvm::SmallDenseMap<size_t, unsigned> exprStats;
    HashContext hashCtx;

    // Collect all affine.apply ops that need rewriting.
    getOperation()->walk([&](AffineApplyOp applyOp) {
      AffineMap map = applyOp.getAffineMap();
      if (map.getNumResults() != 1)
        return;

      AffineExpr expr = map.getResult(0);
      AffineExpr reorderedExpr =
          reorderCommutativeOps(expr, applyOp, hashCtx, exprStats);

      // Compute hash and update statistics for reordered expression and all
      // sub-expressions.
      hashExprWithOperands(reorderedExpr, applyOp, hashCtx, exprStats);

      // Check if the expression changed.
      if (reorderedExpr != expr) {
        LDBG() << "Reordered expression: " << expr << " -> " << reorderedExpr;
        opsToRewrite.push_back({applyOp, reorderedExpr});
      }
    });

    // Rewrite collected ops.
    for (auto &[applyOp, reorderedExpr] : opsToRewrite) {
      rewriter.setInsertionPoint(applyOp);
      AffineMap map = applyOp.getAffineMap();

      // Create new affine map with reordered expression.
      AffineMap newMap =
          AffineMap::get(map.getNumDims(), map.getNumSymbols(), reorderedExpr);

      // Replace with new affine.apply.
      rewriter.replaceOpWithNewOp<AffineApplyOp>(applyOp, newMap,
                                                 applyOp.getMapOperands());
    }
  }
};
} // namespace
