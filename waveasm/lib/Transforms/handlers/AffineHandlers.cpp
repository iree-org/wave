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
#include "llvm/Support/Debug.h"

#include <cassert>
#include <functional>

#define DEBUG_TYPE "waveasm-affine-handlers"

using namespace mlir;

namespace waveasm {

// 32-bit unsigned Barrett reduction: floordiv(x, d) exact for all uint32.
// Matches LLVM's __udivsi3 lowering:
//   1. Float rcp seed  →  Newton-Raphson refinement  →  exact reciprocal
//   2. q = mulhi(x, recip)
//   3. Two remainder-based correction steps
Value emitUnsignedFloordiv(Value x, Value d, OpBuilder &builder, Location loc,
                           TranslationContext &ctx) {
  auto vregType = ctx.createVRegType();
  auto zeroImm = ctx.createImmType(0);
  auto zeroConst = ConstantOp::create(builder, loc, zeroImm, 0);

  // --- Step 1: float reciprocal seed ---
  // Scale factor 0x4f7ffffe = 2^32 * (1 - 2^-24), converts rcp to a ~32-bit
  // fixed-point reciprocal estimate.
  Value df = V_CVT_F32_U32::create(builder, loc, vregType, d);
  Value rcp = V_RCP_F32::create(builder, loc, vregType, df);
  const int32_t kScale = bitCast<int32_t>(0x4f7ffffeu);
  auto scaleImm = ctx.createImmType(kScale);
  auto scaleConst = ConstantOp::create(builder, loc, scaleImm, kScale);
  Value rcpScaled = V_MUL_F32::create(builder, loc, vregType, scaleConst, rcp);
  Value recip = V_CVT_U32_F32::create(builder, loc, vregType, rcpScaled);

  // --- Step 2: Newton-Raphson refinement (1 iteration) ---
  Value negD = V_SUB_U32::create(builder, loc, vregType, zeroConst, d);
  Value e = V_MUL_LO_U32::create(builder, loc, vregType, negD, recip);
  Value err = V_MUL_HI_U32::create(builder, loc, vregType, recip, e);
  recip = V_ADD_U32::create(builder, loc, vregType, recip, err);

  // --- Step 3: quotient ---
  Value q = V_MUL_HI_U32::create(builder, loc, vregType, x, recip);

  // --- Step 4: remainder + two correction steps ---
  Value qd = V_MUL_LO_U32::create(builder, loc, vregType, q, d);
  Value rem = V_SUB_U32::create(builder, loc, vregType, x, qd);

  auto oneImm = ctx.createImmType(1);
  auto oneConst = ConstantOp::create(builder, loc, oneImm, 1);
  Value oneVgpr = V_MOV_B32::create(builder, loc, vregType, oneConst);

  // First correction: if rem >= d then q++, rem -= d
  Value remMinusD = V_SUBREV_U32::create(builder, loc, vregType, d, rem);
  V_CMP_GE_U32::create(builder, loc, rem, d);
  Value inc1 = V_CNDMASK_B32::create(builder, loc, vregType, zeroConst, oneVgpr,
                                     zeroConst);
  q = V_ADD_U32::create(builder, loc, vregType, q, inc1);
  rem =
      V_CNDMASK_B32::create(builder, loc, vregType, rem, remMinusD, zeroConst);

  // Second correction: if rem >= d then q++
  V_CMP_GE_U32::create(builder, loc, rem, d);
  Value inc2 = V_CNDMASK_B32::create(builder, loc, vregType, zeroConst, oneVgpr,
                                     zeroConst);
  q = V_ADD_U32::create(builder, loc, vregType, q, inc2);

  return q;
}

//===----------------------------------------------------------------------===//
// Magic number division by constant (Hacker's Delight, section 10-9)
//===----------------------------------------------------------------------===//
//
// For a constant divisor d >= 2, computes magic multiplier m and post-shift s:
//   floor(x / d) = mulhi(x, m) >> s              (simple form)
//   floor(x / d) = (mulhi(x,m) + ((x-mulhi(x,m))>>1)) >> (s-1)  (add form)
// Exact for all 32-bit unsigned x. Produces 2-5 VALU instructions vs ~20 for
// the general Barrett reduction.
//===----------------------------------------------------------------------===//

struct UnsignedMagic {
  uint32_t magic;
  unsigned shift;
  bool needsAdd;
};

static UnsignedMagic computeMagicUnsigned(uint32_t d) {
  assert(d >= 2 && "divisor must be >= 2");

  uint32_t p;
  uint64_t nc, delta, q1, r1, q2, r2;
  bool needsAdd = false;

  nc = UINT64_C(0xFFFFFFFF) - (UINT64_C(0xFFFFFFFF) % d);
  p = 31;
  q1 = UINT64_C(0x80000000) / nc;
  r1 = UINT64_C(0x80000000) - q1 * nc;
  q2 = UINT64_C(0x7FFFFFFF) / d;
  r2 = UINT64_C(0x7FFFFFFF) - q2 * d;

  do {
    p++;
    if (r1 >= nc - r1) {
      q1 = 2 * q1 + 1;
      r1 = 2 * r1 - nc;
    } else {
      q1 = 2 * q1;
      r1 = 2 * r1;
    }
    if (r2 + 1 >= d - r2) {
      if (q2 >= UINT64_C(0x7FFFFFFF))
        needsAdd = true;
      q2 = 2 * q2 + 1;
      r2 = 2 * r2 + 1 - d;
    } else {
      if (q2 >= UINT64_C(0x80000000))
        needsAdd = true;
      q2 = 2 * q2;
      r2 = 2 * r2 + 1;
    }
    delta = d - 1 - r2;
  } while (p < 64 && (q1 < delta || (q1 == delta && r1 == 0)));

  return {static_cast<uint32_t>(q2 + 1), p - 32, needsAdd};
}

Value emitConstantUnsignedFloordiv(Value x, int64_t divisor, OpBuilder &builder,
                                   Location loc, TranslationContext &ctx) {
  assert(divisor >= 2 && "divisor must be >= 2");

  auto vregType = ctx.createVRegType();
  UnsignedMagic mag = computeMagicUnsigned(static_cast<uint32_t>(divisor));

  LLVM_DEBUG(llvm::dbgs() << "Magic number for divisor " << divisor
                          << ": magic=0x" << llvm::utohexstr(mag.magic)
                          << ", shift=" << mag.shift
                          << ", needsAdd=" << mag.needsAdd << "\n");

  int64_t magicVal = static_cast<int64_t>(mag.magic);
  auto magicImm = ctx.createImmType(magicVal);
  Value magicConst = ConstantOp::create(builder, loc, magicImm, magicVal);

  Value q = V_MUL_HI_U32::create(builder, loc, vregType, x, magicConst);

  // Emit a right shift by `amount` bits, or return `val` unchanged if
  // amount==0.
  auto emitShiftRight = [&](Value val, int64_t amount) -> Value {
    if (amount == 0)
      return val;
    auto shiftImm = ctx.createImmType(amount);
    auto shiftConst = ConstantOp::create(builder, loc, shiftImm, amount);
    return V_LSHRREV_B32::create(builder, loc, vregType, shiftConst, val);
  };

  if (mag.needsAdd) {
    // result = (q + ((x - q) >> 1)) >> (shift - 1)
    Value xSubQ = V_SUB_U32::create(builder, loc, vregType, x, q);
    auto oneImm = ctx.createImmType(1);
    auto oneConst = ConstantOp::create(builder, loc, oneImm, 1);
    Value halfDiff =
        V_LSHRREV_B32::create(builder, loc, vregType, oneConst, xSubQ);
    Value sum = V_ADD_U32::create(builder, loc, vregType, q, halfDiff);
    return emitShiftRight(sum, mag.shift - 1);
  }

  return emitShiftRight(q, mag.shift);
}

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
          BitRange range = ctx.getBitRange(*mapped);
          return ExprResult(*mapped, range);
        }
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "WARNING: affine dim d" << dimExpr.getPosition()
                 << " not mapped, falling back to baseValue\n");
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
      LLVM_DEBUG(llvm::dbgs()
                 << "WARNING: affine symbol s" << symExpr.getPosition()
                 << " (idx=" << symIdx << ") not mapped, falling back\n");
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

          // If the LHS max value (from BitRange) is less than the divisor,
          // floor(lhs / divisor) == 0 for all inputs. Generalizes the old
          // thread-ID upper-bound check to work with any tracked bit-range.
          if (lhsRange.highBit < 31) {
            int64_t maxVal = (1LL << (lhsRange.highBit + 1)) - 1;
            if (maxVal < divisor) {
              auto immZero = ctx.createImmType(0);
              return ExprResult(ConstantOp::create(builder, loc, immZero, 0),
                                BitRange(0, 0));
            }
          }

          // Check if divisor is power of 2 -> use right shift
          if (isPowerOf2(divisor)) {
            int64_t shiftAmount = log2(divisor);
            auto shiftAmt = ctx.createImmType(shiftAmount);
            auto shiftConst =
                ConstantOp::create(builder, loc, shiftAmt, shiftAmount);
            Value shiftResult =
                V_LSHRREV_B32::create(builder, loc, vregType, shiftConst, lhs);
            BitRange resultRange = lhsRange.shiftRight(shiftAmount);
            ctx.setBitRange(shiftResult, resultRange);
            return ExprResult(shiftResult, resultRange);
          }

          // Non-power-of-2 constant: magic number multiplication
          if (divisor >= 2) {
            Value q =
                emitConstantUnsignedFloordiv(lhs, divisor, builder, loc, ctx);
            return ExprResult(q, BitRange());
          }
        }
        // Symbolic divisor fallback: Barrett reduction
        {
          Value qFixed = emitUnsignedFloordiv(lhs, rhs, builder, loc, ctx);
          return ExprResult(qFixed, BitRange());
        }
      }

      case AffineExprKind::CeilDiv: {
        if (auto constRhs = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
          int64_t divisor = constRhs.getValue();

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

          // Non-power-of-2 constant: ceildiv(x, d) = floordiv(x + d - 1, d)
          if (divisor >= 2) {
            int64_t bias = divisor - 1;
            auto biasImm = ctx.createImmType(bias);
            auto biasConst = ConstantOp::create(builder, loc, biasImm, bias);
            Value biased =
                V_ADD_U32::create(builder, loc, vregType, biasConst, lhs);
            Value q = emitConstantUnsignedFloordiv(biased, divisor, builder,
                                                   loc, ctx);
            return ExprResult(q, BitRange());
          }
        }
        // Symbolic divisor fallback: ceildiv(x, d) = floordiv(x + d - 1, d)
        {
          auto oneImm = ctx.createImmType(1);
          auto oneConst = ConstantOp::create(builder, loc, oneImm, 1);
          Value dMinus1 =
              V_SUB_U32::create(builder, loc, vregType, rhs, oneConst);
          Value biased =
              V_ADD_U32::create(builder, loc, vregType, lhs, dMinus1);
          Value qFixed = emitUnsignedFloordiv(biased, rhs, builder, loc, ctx);
          return ExprResult(qFixed, BitRange());
        }
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
            BitRange resultRange = BitRange(0, log2(val) - 1);
            ctx.setBitRange(andResult, resultRange);
            return ExprResult(andResult, resultRange);
          }

          // Non-power-of-2 constant: x mod d = x - floordiv(x, d) * d
          if (val >= 2) {
            Value q = emitConstantUnsignedFloordiv(lhs, val, builder, loc, ctx);
            auto dImm = ctx.createImmType(val);
            auto dConst = ConstantOp::create(builder, loc, dImm, val);
            Value qd = V_MUL_LO_U32::create(builder, loc, vregType, q, dConst);
            Value rem = V_SUB_U32::create(builder, loc, vregType, lhs, qd);
            return ExprResult(rem, BitRange());
          }
        }
        // Symbolic divisor fallback: x mod d = x - floordiv(x, d) * d
        {
          Value q = emitUnsignedFloordiv(lhs, rhs, builder, loc, ctx);
          Value qd = V_MUL_LO_U32::create(builder, loc, vregType, q, rhs);
          Value rem = V_SUB_U32::create(builder, loc, vregType, lhs, qd);
          return ExprResult(rem, BitRange());
        }
      }

      default:
        return ExprResult(lhs,
                          BitRange()); // Unsupported, return LHS as fallback
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "WARNING: unhandled affine expression kind "
                            << static_cast<int>(e.getKind())
                            << ", falling back to baseValue\n");
    return ExprResult(baseValue, BitRange());
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
