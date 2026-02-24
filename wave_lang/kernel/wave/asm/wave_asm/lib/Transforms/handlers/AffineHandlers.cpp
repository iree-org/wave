// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Affine Dialect Handlers
//===----------------------------------------------------------------------===//
//
// This file implements handlers for Affine dialect operations:
//   - affine.apply
//
// The implementation includes:
//   - Thread ID upper bound simplification
//   - Bit range tracking for OR optimization
//   - Power-of-2 optimizations (shift instead of multiply/divide)
//   - v_lshl_or_b32 fusion for non-overlapping bit ranges
//
//===----------------------------------------------------------------------===//

#include "Handlers.h"

#include "waveasm/Dialect/WaveASMOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

#include <functional>

using namespace mlir;

namespace waveasm {

/// Handle affine.apply - compile affine expression to arithmetic instructions
LogicalResult handleAffineApply(Operation *op, TranslationContext &ctx) {
  auto applyOp = cast<affine::AffineApplyOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto map = applyOp.getAffineMap();

  // Get the single operand (for single-dimension maps)
  if (applyOp.getOperands().empty()) {
    return op->emitError("affine.apply with no operands");
  }

  Value baseValue;
  if (auto mapped = ctx.getMapper().getMapped(applyOp.getOperands()[0])) {
    baseValue = *mapped;
  } else {
    return op->emitError("operand not mapped");
  }

  // For single result affine maps, analyze the expression
  if (map.getNumResults() != 1) {
    return op->emitError("only single-result affine maps supported");
  }

  AffineExpr expr = map.getResult(0);

  // Get thread ID upper bound for the first operand (used for simplification)
  // If the first operand is a thread ID with known upper bound, we can
  // simplify floor divisions where divisor >= upper_bound to 0
  int64_t threadIdUpperBound = 0;
  if (applyOp.getOperands().size() > 0) {
    threadIdUpperBound = ctx.getThreadIdUpperBound(applyOp.getOperands()[0]);
  }

  // HIGH-LEVEL SIMPLIFICATION: Check if the entire expression simplifies to
  // just the input symbol when floor divisions evaluate to 0 Pattern: s0 + (s0
  // floordiv N) * C where N >= upper_bound
  //       => s0 + 0 * C = s0
  if (threadIdUpperBound > 0) {
    // Check if expression is Add(symbol, Mul(FloorDiv(symbol, N), C))
    // where N >= threadIdUpperBound
    if (auto addExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
      if (addExpr.getKind() == AffineExprKind::Add) {
        // Check if LHS is the symbol and RHS is a Mul containing FloorDiv
        if (isa<AffineSymbolExpr>(addExpr.getLHS())) {
          if (auto mulExpr = dyn_cast<AffineBinaryOpExpr>(addExpr.getRHS())) {
            if (mulExpr.getKind() == AffineExprKind::Mul) {
              // Check if LHS of Mul is FloorDiv with divisor >= upperBound
              if (auto floorExpr =
                      dyn_cast<AffineBinaryOpExpr>(mulExpr.getLHS())) {
                if (floorExpr.getKind() == AffineExprKind::FloorDiv) {
                  if (auto constDiv =
                          dyn_cast<AffineConstantExpr>(floorExpr.getRHS())) {
                    if (constDiv.getValue() >= threadIdUpperBound) {
                      // Expression simplifies to just the symbol (s0)
                      // Map result to the thread ID value
                      ctx.getMapper().mapValue(applyOp.getResult(), baseValue);
                      return success();
                    }
                  }
                }
              }
              // Also check RHS of Mul
              if (auto floorExpr =
                      dyn_cast<AffineBinaryOpExpr>(mulExpr.getRHS())) {
                if (floorExpr.getKind() == AffineExprKind::FloorDiv) {
                  if (auto constDiv =
                          dyn_cast<AffineConstantExpr>(floorExpr.getRHS())) {
                    if (constDiv.getValue() >= threadIdUpperBound) {
                      ctx.getMapper().mapValue(applyOp.getResult(), baseValue);
                      return success();
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // NOTE: We used to extract constant addends for buffer store offset:N
  // optimization but this caused bugs when the affine result was used in arith
  // operations (the constant was lost). For now, just compile the full
  // expression.
  // TODO: Re-enable constant extraction only for values used directly in memory
  // ops
  int64_t constAddend = 0;
  AffineExpr exprToCompile = expr;

  // Simple pattern matching for common affine expressions
  // Pattern: d0 mod N -> v_and_b32 (when N is power of 2)
  // Pattern: d0 floordiv N -> v_lshrrev_b32 (when N is power of 2)
  // Pattern: d0 * N -> v_lshlrev_b32 (when N is power of 2)

  // Result type that includes bit range tracking for OR optimization
  struct ExprResult {
    Value value;
    BitRange range;
    ExprResult(Value v, BitRange r) : value(v), range(r) {}
  };

  // Helper to emit the compiled expression with bit range tracking
  std::function<ExprResult(AffineExpr)> compileExpr =
      [&](AffineExpr e) -> ExprResult {
    // Dimension reference
    if (auto dimExpr = dyn_cast<AffineDimExpr>(e)) {
      if (dimExpr.getPosition() < applyOp.getOperands().size()) {
        Value operand = applyOp.getOperands()[dimExpr.getPosition()];
        if (auto mapped = ctx.getMapper().getMapped(operand)) {
          // Use tracked bit range if available
          BitRange range = ctx.getBitRange(*mapped);
          return ExprResult(*mapped, range);
        }
      }
      return ExprResult(baseValue, ctx.getBitRange(baseValue));
    }

    // Symbol reference
    if (auto symExpr = dyn_cast<AffineSymbolExpr>(e)) {
      int64_t symIdx = map.getNumDims() + symExpr.getPosition();
      if (symIdx < static_cast<int64_t>(applyOp.getOperands().size())) {
        Value operand = applyOp.getOperands()[symIdx];
        if (auto mapped = ctx.getMapper().getMapped(operand)) {
          BitRange range = ctx.getBitRange(*mapped);
          return ExprResult(*mapped, range);
        }
      }
      return ExprResult(baseValue, ctx.getBitRange(baseValue));
    }

    // Constant
    if (auto constExpr = dyn_cast<AffineConstantExpr>(e)) {
      int64_t val = constExpr.getValue();
      auto immType = ctx.createImmType(val);
      Value constVal = ConstantOp::create(builder, loc, immType, val);
      return ExprResult(constVal, BitRange::fromConstant(val));
    }

    // Binary expressions
    if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(e)) {
      ExprResult lhsResult = compileExpr(binExpr.getLHS());
      ExprResult rhsResult = compileExpr(binExpr.getRHS());
      Value lhs = lhsResult.value;
      Value rhs = rhsResult.value;
      BitRange lhsRange = lhsResult.range;
      BitRange rhsRange = rhsResult.range;

      switch (binExpr.getKind()) {
      case AffineExprKind::Add: {
        // Check if bit ranges overlap - if not, use OR instead of ADD
        if (!lhsRange.overlaps(rhsRange)) {
          // Check if either operand is a shift (Mul by power of 2)
          // If so, emit v_lshl_or_b32 directly instead of lshlrev + or
          auto tryFuseShiftOr =
              [&](AffineExpr shiftExpr, Value orend,
                  BitRange orendRange) -> std::optional<ExprResult> {
            if (auto mulExpr = dyn_cast<AffineBinaryOpExpr>(shiftExpr)) {
              if (mulExpr.getKind() == AffineExprKind::Mul) {
                // Check for power of 2 multiplier
                if (auto constRhs =
                        dyn_cast<AffineConstantExpr>(mulExpr.getRHS())) {
                  int64_t val = constRhs.getValue();
                  if (val > 0 && (val & (val - 1)) == 0) {
                    // It's a shift! Emit v_lshl_or_b32 directly
                    int64_t shiftAmount = log2(val);
                    // Get the base value being shifted (compile without the
                    // multiply)
                    ExprResult baseResult = compileExpr(mulExpr.getLHS());
                    auto shiftImm = ctx.createImmType(shiftAmount);
                    auto shiftConst =
                        ConstantOp::create(builder, loc, shiftImm, shiftAmount);
                    // v_lshl_or_b32: dst = (src << shift) | orend
                    Value fusedResult = V_LSHL_OR_B32::create(
                        builder, loc, vregType, baseResult.value, shiftConst,
                        orend);
                    BitRange shiftedRange =
                        baseResult.range.shiftLeft(shiftAmount);
                    BitRange resultRange = shiftedRange.merge(orendRange);
                    ctx.setBitRange(fusedResult, resultRange);
                    return ExprResult(fusedResult, resultRange);
                  }
                }
                // Also check LHS for constant
                if (auto constLhs =
                        dyn_cast<AffineConstantExpr>(mulExpr.getLHS())) {
                  int64_t val = constLhs.getValue();
                  if (val > 0 && (val & (val - 1)) == 0) {
                    int64_t shiftAmount = log2(val);
                    ExprResult baseResult = compileExpr(mulExpr.getRHS());
                    auto shiftImm = ctx.createImmType(shiftAmount);
                    auto shiftConst =
                        ConstantOp::create(builder, loc, shiftImm, shiftAmount);
                    Value fusedResult = V_LSHL_OR_B32::create(
                        builder, loc, vregType, baseResult.value, shiftConst,
                        orend);
                    BitRange shiftedRange =
                        baseResult.range.shiftLeft(shiftAmount);
                    BitRange resultRange = shiftedRange.merge(orendRange);
                    ctx.setBitRange(fusedResult, resultRange);
                    return ExprResult(fusedResult, resultRange);
                  }
                }
              }
            }
            return std::nullopt;
          };

          // Try to fuse: check if LHS is a shift
          if (auto result = tryFuseShiftOr(binExpr.getLHS(), rhs, rhsRange)) {
            return *result;
          }
          // Try to fuse: check if RHS is a shift
          if (auto result = tryFuseShiftOr(binExpr.getRHS(), lhs, lhsRange)) {
            return *result;
          }

          // No fusion possible, emit regular v_or_b32
          Value orResult = V_OR_B32::create(builder, loc, vregType, lhs, rhs);
          BitRange resultRange = lhsRange.merge(rhsRange);
          ctx.setBitRange(orResult, resultRange);
          return ExprResult(orResult, resultRange);
        }
        // Overlapping ranges - must use ADD
        Value addResult = V_ADD_U32::create(builder, loc, vregType, lhs, rhs);
        BitRange resultRange = lhsRange.extendForAdd(rhsRange);
        ctx.setBitRange(addResult, resultRange);
        return ExprResult(addResult, resultRange);
      }

      case AffineExprKind::Mul: {
        // PATTERN: floor(x / N) * N = x & ~(N-1)  when N is power of 2
        // Detect Mul(FloorDiv(expr, N), N) and emit AND directly.
        // This saves 1 instruction vs (x >> log2(N)) << log2(N).
        auto tryFloorMulToAnd =
            [&](AffineExpr divExpr,
                AffineExpr constExpr) -> std::optional<ExprResult> {
          auto floorDiv = dyn_cast<AffineBinaryOpExpr>(divExpr);
          if (!floorDiv || floorDiv.getKind() != AffineExprKind::FloorDiv)
            return std::nullopt;
          auto mulConst = dyn_cast<AffineConstantExpr>(constExpr);
          auto divConst = dyn_cast<AffineConstantExpr>(floorDiv.getRHS());
          if (!mulConst || !divConst)
            return std::nullopt;
          if (mulConst.getValue() != divConst.getValue())
            return std::nullopt;
          int64_t N = mulConst.getValue();
          if (N <= 0 || !isPowerOf2(N))
            return std::nullopt;
          // Compile the inner expression (the x in floor(x/N)*N)
          ExprResult innerResult = compileExpr(floorDiv.getLHS());
          int64_t mask = ~(N - 1) & 0xFFFFFFFF;
          auto maskImm = ctx.createImmType(mask);
          auto maskConst = ConstantOp::create(builder, loc, maskImm, mask);
          // NOTE: constant must be src0 (first operand) for VOP2 encoding.
          // src1 must be a VGPR on AMDGCN.
          Value andResult = V_AND_B32::create(builder, loc, vregType, maskConst,
                                              innerResult.value);
          // Result has same bit range as inner, but low bits cleared
          BitRange resultRange = innerResult.range;
          // Clear bits below log2(N) -- conservative: use inner range
          ctx.setBitRange(andResult, resultRange);
          return ExprResult(andResult, resultRange);
        };

        if (auto result = tryFloorMulToAnd(binExpr.getLHS(), binExpr.getRHS()))
          return *result;
        if (auto result = tryFloorMulToAnd(binExpr.getRHS(), binExpr.getLHS()))
          return *result;

        // Constant folding: if either operand is constant 0, result is 0
        if (auto constLhs = dyn_cast<AffineConstantExpr>(binExpr.getLHS())) {
          if (constLhs.getValue() == 0) {
            auto immZero = ctx.createImmType(0);
            return ExprResult(ConstantOp::create(builder, loc, immZero, 0),
                              BitRange(0, 0));
          }
        }
        if (auto constRhs = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
          if (constRhs.getValue() == 0) {
            auto immZero = ctx.createImmType(0);
            return ExprResult(ConstantOp::create(builder, loc, immZero, 0),
                              BitRange(0, 0));
          }
          // Check if RHS is constant power of 2 -> use shift
          int64_t val = constRhs.getValue();
          if (isPowerOf2(val)) {
            int64_t shiftAmount = log2(val);
            auto shiftAmt = ctx.createImmType(shiftAmount);
            auto shiftConst =
                ConstantOp::create(builder, loc, shiftAmt, shiftAmount);
            Value shiftResult =
                V_LSHLREV_B32::create(builder, loc, vregType, shiftConst, lhs);
            // Shift the bit range left by shiftAmount
            BitRange resultRange = lhsRange.shiftLeft(shiftAmount);
            ctx.setBitRange(shiftResult, resultRange);
            return ExprResult(shiftResult, resultRange);
          }
        }
        // Also check LHS for power of 2 multiply
        if (auto constLhs = dyn_cast<AffineConstantExpr>(binExpr.getLHS())) {
          int64_t val = constLhs.getValue();
          if (isPowerOf2(val)) {
            int64_t shiftAmount = log2(val);
            auto shiftAmt = ctx.createImmType(shiftAmount);
            auto shiftConst =
                ConstantOp::create(builder, loc, shiftAmt, shiftAmount);
            Value shiftResult =
                V_LSHLREV_B32::create(builder, loc, vregType, shiftConst, rhs);
            BitRange resultRange = rhsRange.shiftLeft(shiftAmount);
            ctx.setBitRange(shiftResult, resultRange);
            return ExprResult(shiftResult, resultRange);
          }
        }
        Value mulResult =
            V_MUL_LO_U32::create(builder, loc, vregType, lhs, rhs);
        return ExprResult(mulResult, BitRange()); // Conservative: full range
      }

      case AffineExprKind::FloorDiv: {
        // Check if RHS is constant
        if (auto constRhs = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
          int64_t divisor = constRhs.getValue();

          // SIMPLIFICATION: If the LHS is a thread ID with upper_bound <=
          // divisor, then floor(tid / divisor) = 0 for all valid thread IDs.
          // Example: tid_x in [0, 63], floor(tid_x / 64) = 0
          if (threadIdUpperBound > 0 && divisor >= threadIdUpperBound) {
            // LHS is in range [0, upper_bound-1], so floor(LHS / divisor) = 0
            auto immZero = ctx.createImmType(0);
            return ExprResult(ConstantOp::create(builder, loc, immZero, 0),
                              BitRange(0, 0));
          }

          // Check if divisor is power of 2 -> use right shift
          if (isPowerOf2(divisor)) {
            int64_t shiftAmount = log2(divisor);
            auto shiftAmt = ctx.createImmType(shiftAmount);
            auto shiftConst =
                ConstantOp::create(builder, loc, shiftAmt, shiftAmount);
            Value shiftResult =
                V_LSHRREV_B32::create(builder, loc, vregType, shiftConst, lhs);
            // Shift the bit range right by shiftAmount
            BitRange resultRange = lhsRange.shiftRight(shiftAmount);
            ctx.setBitRange(shiftResult, resultRange);
            return ExprResult(shiftResult, resultRange);
          }
        }
        // General floordiv - needs more complex handling
        return ExprResult(lhs, BitRange()); // Conservative
      }

      case AffineExprKind::CeilDiv: {
        if (auto constRhs = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
          int64_t divisor = constRhs.getValue();

          // Optimization: ceildiv(x, 2^k) = (x + 2^k - 1) >> k.
          // Only valid for non-negative x because V_LSHRREV_B32 is a logical
          // (unsigned) right shift. Negative values would produce a large
          // positive result instead of the correct negative ceiling.
          // Affine expressions in this context originate from index
          // computations which are non-negative by construction.
          if (isPowerOf2(divisor)) {
            int64_t shiftAmount = log2(divisor);
            int64_t bias = divisor - 1;
            auto biasImm = ctx.createImmType(bias);
            auto biasConst = ConstantOp::create(builder, loc, biasImm, bias);
            Value biased =
                V_ADD_U32::create(builder, loc, vregType, biasConst, lhs);
            auto shiftAmt = ctx.createImmType(shiftAmount);
            auto shiftConst =
                ConstantOp::create(builder, loc, shiftAmt, shiftAmount);
            Value shiftResult = V_LSHRREV_B32::create(builder, loc, vregType,
                                                      shiftConst, biased);
            BitRange resultRange =
                lhsRange.extendForAdd(BitRange::fromConstant(bias))
                    .shiftRight(shiftAmount);
            ctx.setBitRange(shiftResult, resultRange);
            return ExprResult(shiftResult, resultRange);
          }
        }
        return ExprResult(lhs, BitRange());
      }

      case AffineExprKind::Mod: {
        // Check if RHS is constant power of 2 -> use AND
        if (auto constRhs = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
          int64_t val = constRhs.getValue();
          if (isPowerOf2(val)) {
            auto maskVal = ctx.createImmType(val - 1);
            auto maskConst = ConstantOp::create(builder, loc, maskVal, val - 1);
            Value andResult =
                V_AND_B32::create(builder, loc, vregType, lhs, maskConst);
            // Result uses bits 0..(log2(val)-1)
            BitRange resultRange = BitRange(0, log2(val) - 1);
            ctx.setBitRange(andResult, resultRange);
            return ExprResult(andResult, resultRange);
          }
        }
        // General mod - needs more complex handling
        return ExprResult(lhs, BitRange()); // Conservative
      }

      default:
        return ExprResult(lhs,
                          BitRange()); // Unsupported, return LHS as fallback
      }
    }

    return ExprResult(baseValue, BitRange()); // Fallback
  };

  ExprResult result = compileExpr(exprToCompile);
  ctx.getMapper().mapValue(applyOp.getResult(), result.value);
  ctx.setBitRange(result.value, result.range);

  // Track the constant addend for buffer store offset:N optimization
  if (constAddend != 0) {
    ctx.setConstOffset(applyOp.getResult(), constAddend);
  }

  return success();
}

} // namespace waveasm
