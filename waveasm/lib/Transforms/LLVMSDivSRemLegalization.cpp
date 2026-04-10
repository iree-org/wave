// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// LLVM sdiv/srem legalization
//
// Rewrites signed division and remainder by positive power-of-two constants
// into equivalent LLVM dialect compare/select/add/and/ashr sequences before
// LLVM->WaveASM translation changes the abstraction level.
//===----------------------------------------------------------------------===//

#include "waveasm/Transforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMLLVMSDIVSREMLEGALIZATION
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

using namespace mlir;

namespace {

static std::optional<int64_t> getConstantI32Value(Value value) {
  LLVM::ConstantOp constOp = value.getDefiningOp<LLVM::ConstantOp>();
  if (!constOp)
    return std::nullopt;
  IntegerAttr intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
  if (!intAttr)
    return std::nullopt;
  return intAttr.getInt();
}

static std::optional<std::pair<int64_t, int64_t>>
matchPositivePowerOfTwoI32Divisor(Value rhs) {
  std::optional<int64_t> constVal = getConstantI32Value(rhs);
  if (!constVal || *constVal <= 0 || !llvm::isPowerOf2_64(*constVal))
    return std::nullopt;
  int64_t divisor = *constVal;
  int64_t shiftAmt = llvm::Log2_64(static_cast<uint64_t>(divisor));
  return std::pair<int64_t, int64_t>{divisor, shiftAmt};
}

static Value createI32Constant(PatternRewriter &rewriter, Location loc,
                               int64_t value) {
  Type i32 = rewriter.getI32Type();
  return LLVM::ConstantOp::create(rewriter, loc, i32,
                                  rewriter.getIntegerAttr(i32, value));
}

struct LegalizePowerOfTwoSDivPattern : OpRewritePattern<LLVM::SDivOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(LLVM::SDivOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().isSignlessInteger(32))
      return failure();
    std::optional<std::pair<int64_t, int64_t>> divisorAndShift =
        matchPositivePowerOfTwoI32Divisor(op.getRhs());
    if (!divisorAndShift)
      return failure();

    int64_t divisor = divisorAndShift->first;
    int64_t shiftAmt = divisorAndShift->second;
    Location loc = op.getLoc();

    Value zero = createI32Constant(rewriter, loc, 0);
    Value biasImm = createI32Constant(rewriter, loc, divisor - 1);
    Value shiftConst = createI32Constant(rewriter, loc, shiftAmt);
    Value isNegative = LLVM::ICmpOp::create(
        rewriter, loc, LLVM::ICmpPredicate::slt, op.getLhs(), zero);
    Value bias = LLVM::SelectOp::create(rewriter, loc, isNegative, biasImm,
                                        zero, LLVM::FastmathFlags::none);
    Value biased = LLVM::AddOp::create(rewriter, loc, op.getLhs(), bias,
                                       LLVM::IntegerOverflowFlags::none);
    Value result = LLVM::AShrOp::create(rewriter, loc, biased, shiftConst,
                                        /*isExact=*/false);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LegalizePowerOfTwoSRemPattern : OpRewritePattern<LLVM::SRemOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(LLVM::SRemOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().isSignlessInteger(32))
      return failure();
    std::optional<std::pair<int64_t, int64_t>> divisorAndShift =
        matchPositivePowerOfTwoI32Divisor(op.getRhs());
    if (!divisorAndShift)
      return failure();

    int64_t divisor = divisorAndShift->first;
    Location loc = op.getLoc();

    Value zero = createI32Constant(rewriter, loc, 0);
    Value maskConst = createI32Constant(rewriter, loc, divisor - 1);
    Value negDivisor = createI32Constant(rewriter, loc, -divisor);
    Value rawRem = LLVM::AndOp::create(rewriter, loc, op.getLhs(), maskConst);
    Value isNegative = LLVM::ICmpOp::create(
        rewriter, loc, LLVM::ICmpPredicate::slt, op.getLhs(), zero);
    Value isNonZero = LLVM::ICmpOp::create(
        rewriter, loc, LLVM::ICmpPredicate::ne, rawRem, zero);
    Value needsAdjust =
        LLVM::AndOp::create(rewriter, loc, isNegative, isNonZero);
    Value adjust =
        LLVM::SelectOp::create(rewriter, loc, needsAdjust, negDivisor, zero,
                               LLVM::FastmathFlags::none);
    Value result = LLVM::AddOp::create(rewriter, loc, rawRem, adjust,
                                       LLVM::IntegerOverflowFlags::none);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LLVMSDivSRemLegalizationPass
    : public waveasm::impl::WAVEASMLLVMSDivSRemLegalizationBase<
          LLVMSDivSRemLegalizationPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LegalizePowerOfTwoSDivPattern, LegalizePowerOfTwoSRemPattern>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
