// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Peephole Optimization Pass
//
// This pass implements peephole optimizations for WAVEASM IR:
// 1. Instruction fusion: lshl + add -> v_mad_u32_u24
// 2. Dead code elimination
// 3. Redundant move elimination
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace waveasm;

namespace {

//===----------------------------------------------------------------------===//
// Pattern: lshlrev + or -> v_lshl_or_b32
//
// Fuses:
//   %shifted = v_lshlrev_b32 %shift_amt, %src
//   %result = v_or_b32 %shifted, %orend
// Into:
//   %result = v_lshl_or_b32 %src, %shift_amt, %orend
//
// This is a common pattern in address calculations where bit ranges
// don't overlap, allowing OR to be used instead of ADD.
//===----------------------------------------------------------------------===//

struct LshlOrFusePattern : public OpRewritePattern<V_OR_B32> {
  using OpRewritePattern<V_OR_B32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_OR_B32 orOp,
                                PatternRewriter &rewriter) const override {
    // Check if either operand is the result of a left shift
    auto checkOperand = [&](Value shifted, Value orend) -> LogicalResult {
      auto lshlOp = shifted.getDefiningOp<V_LSHLREV_B32>();
      if (!lshlOp)
        return failure();

      // Check that the shift has only one use (this or)
      if (!lshlOp.getResult().hasOneUse())
        return failure();

      // Get the shift amount - must be a constant for efficiency
      auto constOp = lshlOp.getSrc0().getDefiningOp<ConstantOp>();
      if (!constOp)
        return failure();

      int64_t shiftAmount = constOp.getValue();

      // v_lshl_or_b32 operands: src (value to shift), shift_amt, orend
      // In v_lshlrev_b32: src0 = shift_amt, src1 = value to shift
      Value shiftSrc = lshlOp.getSrc1();  // value being shifted
      Value shiftAmt = lshlOp.getSrc0();  // shift amount

      auto loc = orOp.getLoc();
      auto resultType = orOp.getResult().getType();

      // Create v_lshl_or_b32: dst = (src << shift_amt) | orend
      auto fusedOp = V_LSHL_OR_B32::create(rewriter, loc, resultType,
                                            shiftSrc, shiftAmt, orend);

      rewriter.replaceOp(orOp, fusedOp.getResult());
      return success();
    };

    // Try both operand orderings
    if (succeeded(checkOperand(orOp.getSrc0(), orOp.getSrc1())))
      return success();
    if (succeeded(checkOperand(orOp.getSrc1(), orOp.getSrc0())))
      return success();

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: lshl + add -> v_lshl_add_u32
//
// Fuses:
//   %shifted = v_lshlrev_b32 %shift_amt, %base
//   %result = v_add_u32 %shifted, %addend
// Into:
//   %result = v_lshl_add_u32 %base, %shift_amt, %addend
//
// This matches Python's use of the fused shift-add instruction.
//===----------------------------------------------------------------------===//

struct LshlAddPattern : public OpRewritePattern<V_ADD_U32> {
  using OpRewritePattern<V_ADD_U32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_ADD_U32 addOp,
                                PatternRewriter &rewriter) const override {
    // Check if either operand is the result of a left shift
    auto checkOperand = [&](Value shifted, Value other) -> LogicalResult {
      auto lshlOp = shifted.getDefiningOp<V_LSHLREV_B32>();
      if (!lshlOp)
        return failure();

      // Check that the shift has only one use (this add)
      if (!lshlOp.getResult().hasOneUse())
        return failure();

      // Get the shift amount - must be a constant for v_lshl_add_u32
      auto constOp = lshlOp.getSrc0().getDefiningOp<ConstantOp>();
      if (!constOp)
        return failure();

      int64_t shiftAmount = constOp.getValue();

      // Shift amount must be non-negative and <= 31 for 32-bit operations
      if (shiftAmount < 0 || shiftAmount > 31)
        return failure();

      auto loc = addOp.getLoc();

      // Get the base value being shifted
      Value base = lshlOp.getSrc1();

      // Get or create the shift amount constant
      Value shiftConst = lshlOp.getSrc0();

      // Create v_lshl_add_u32: dst = (base << shift) + other
      auto resultType = addOp.getResult().getType();
      auto lshlAddOp = V_LSHL_ADD_U32::create(rewriter, loc, resultType, base,
                                              shiftConst, other);

      rewriter.replaceOp(addOp, lshlAddOp.getResult());
      return success();
    };

    // Try both operand orderings
    if (succeeded(checkOperand(addOp.getSrc0(), addOp.getSrc1())))
      return success();
    if (succeeded(checkOperand(addOp.getSrc1(), addOp.getSrc0())))
      return success();

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Redundant move elimination
//
// Removes:
//   %dst = v_mov_b32 %src
// When %src is already in a VGPR and can be used directly.
//===----------------------------------------------------------------------===//

struct RedundantMovePattern : public OpRewritePattern<V_MOV_B32> {
  using OpRewritePattern<V_MOV_B32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_MOV_B32 movOp,
                                PatternRewriter &rewriter) const override {
    Value src = movOp.getSrc();

    // Skip if source is an immediate (move is necessary)
    if (isa<ImmType>(src.getType()))
      return failure();

    // Skip if source is an SGPR (move to VGPR is necessary for broadcast)
    if (isa<SRegType, PSRegType>(src.getType()))
      return failure();

    // Source is already a VGPR, so the move is redundant
    rewriter.replaceOp(movOp, src);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Multiply by power of 2 -> shift
//
// Transforms:
//   %dst = v_mul_lo_u32 %src, %const  (where const is power of 2)
// Into:
//   %dst = v_lshlrev_b32 log2(const), %src
//===----------------------------------------------------------------------===//

struct MulPow2ToShiftPattern : public OpRewritePattern<V_MUL_LO_U32> {
  using OpRewritePattern<V_MUL_LO_U32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_MUL_LO_U32 mulOp,
                                PatternRewriter &rewriter) const override {
    auto checkOperand = [&](Value constVal, Value other) -> LogicalResult {
      auto constOp = constVal.getDefiningOp<ConstantOp>();
      if (!constOp)
        return failure();

      int64_t value = constOp.getValue();

      // Check if value is a power of 2
      if (value <= 0 || (value & (value - 1)) != 0)
        return failure();

      // Calculate log2
      int64_t shiftAmount = 0;
      while ((1LL << shiftAmount) < value)
        ++shiftAmount;

      // Create the shift constant
      auto loc = mulOp.getLoc();
      auto immType = rewriter.getType<ImmType>(shiftAmount);
      auto shiftConst = ConstantOp::create(rewriter, loc, immType, shiftAmount);

      // Create lshlrev: dst = other << shiftAmount
      auto resultType = mulOp.getResult().getType();
      auto lshlOp =
          V_LSHLREV_B32::create(rewriter, loc, resultType, shiftConst, other);

      rewriter.replaceOp(mulOp, lshlOp.getResult());
      return success();
    };

    // Try both operand orderings
    if (succeeded(checkOperand(mulOp.getSrc0(), mulOp.getSrc1())))
      return success();
    if (succeeded(checkOperand(mulOp.getSrc1(), mulOp.getSrc0())))
      return success();

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Add zero elimination
//
// Removes:
//   %dst = v_add_u32 %src, 0
// Replacing with %src directly.
//===----------------------------------------------------------------------===//

struct AddZeroPattern : public OpRewritePattern<V_ADD_U32> {
  using OpRewritePattern<V_ADD_U32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_ADD_U32 addOp,
                                PatternRewriter &rewriter) const override {
    auto checkOperand = [&](Value constVal, Value other) -> LogicalResult {
      auto constOp = constVal.getDefiningOp<ConstantOp>();
      if (!constOp)
        return failure();

      if (constOp.getValue() != 0)
        return failure();

      rewriter.replaceOp(addOp, other);
      return success();
    };

    if (succeeded(checkOperand(addOp.getSrc0(), addOp.getSrc1())))
      return success();
    if (succeeded(checkOperand(addOp.getSrc1(), addOp.getSrc0())))
      return success();

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Multiply by one elimination
//
// Removes:
//   %dst = v_mul_lo_u32 %src, 1
// Replacing with %src directly.
//===----------------------------------------------------------------------===//

struct MulOnePattern : public OpRewritePattern<V_MUL_LO_U32> {
  using OpRewritePattern<V_MUL_LO_U32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_MUL_LO_U32 mulOp,
                                PatternRewriter &rewriter) const override {
    auto checkOperand = [&](Value constVal, Value other) -> LogicalResult {
      auto constOp = constVal.getDefiningOp<ConstantOp>();
      if (!constOp)
        return failure();

      if (constOp.getValue() != 1)
        return failure();

      rewriter.replaceOp(mulOp, other);
      return success();
    };

    if (succeeded(checkOperand(mulOp.getSrc0(), mulOp.getSrc1())))
      return success();
    if (succeeded(checkOperand(mulOp.getSrc1(), mulOp.getSrc0())))
      return success();

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Multiply by zero -> constant zero
//
// Transforms:
//   %dst = v_mul_lo_u32 %src, 0
// Into:
//   %dst = constant 0
//===----------------------------------------------------------------------===//

struct MulZeroPattern : public OpRewritePattern<V_MUL_LO_U32> {
  using OpRewritePattern<V_MUL_LO_U32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_MUL_LO_U32 mulOp,
                                PatternRewriter &rewriter) const override {
    auto checkOperand = [&](Value constVal) -> bool {
      auto constOp = constVal.getDefiningOp<ConstantOp>();
      return constOp && constOp.getValue() == 0;
    };

    if (checkOperand(mulOp.getSrc0()) || checkOperand(mulOp.getSrc1())) {
      auto loc = mulOp.getLoc();
      auto immType = rewriter.getType<ImmType>(0);
      auto zeroConst = ConstantOp::create(rewriter, loc, immType, 0);
      // Move to VGPR if needed
      auto vregType = rewriter.getType<VRegType>(1, 1);
      auto movOp = V_MOV_B32::create(rewriter, loc, vregType, zeroConst);
      rewriter.replaceOp(mulOp, movOp.getResult());
      return success();
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: MFMA with zeroed accumulator -> MFMA with inline 0
//
// When the accumulator input to an MFMA is a zero constant (via v_mov_b32),
// replace it with an inline 0 constant directly. This:
// 1. Eliminates the zeroing mov instruction
// 2. Uses hardware's native inline constant support
//===----------------------------------------------------------------------===//

struct MFMAZeroAccumPattern : public RewritePattern {
  MFMAZeroAccumPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Check if this is an MFMA operation (has TiedOperandIndex attribute)
    // MFMA ops have 3 operands: a, b, acc
    if (op->getNumOperands() != 3 || op->getNumResults() != 1)
      return failure();

    // Check for MFMA ops by name pattern
    StringRef opName = op->getName().getStringRef();
    if (!opName.contains("mfma"))
      return failure();

    // Get the accumulator operand (operand index 2)
    Value accum = op->getOperand(2);

    // Check if accumulator comes from a v_mov_b32 of zero
    auto movOp = accum.getDefiningOp<V_MOV_B32>();
    if (!movOp)
      return failure();

    // Don't optimize away accumulator initializations marked with no_cse
    // These are loop-carried accumulators that need to exist as registers
    if (movOp->hasAttr("no_cse"))
      return failure();

    // Check if the mov source is a zero constant
    auto constOp = movOp.getSrc().getDefiningOp<ConstantOp>();
    if (!constOp || constOp.getValue() != 0)
      return failure();

    // Check that the mov result is only used by this MFMA
    if (!accum.hasOneUse())
      return failure();

    // Replace the accumulator with an inline zero constant
    auto loc = op->getLoc();
    auto immType = rewriter.getType<ImmType>(0);
    auto zeroConst = ConstantOp::create(rewriter, loc, immType, 0);

    // Update the MFMA's accumulator operand
    op->setOperand(2, zeroConst);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Peephole Optimization Pass
//===----------------------------------------------------------------------===//

struct PeepholePass
    : public PassWrapper<PeepholePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PeepholePass)

  PeepholePass() = default;

  StringRef getArgument() const override { return "waveasm-peephole"; }

  StringRef getDescription() const override {
    return "Peephole optimizations for WAVEASM IR";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    RewritePatternSet patterns(ctx);

    // Add all peephole patterns
    patterns.add<LshlOrFusePattern>(ctx);  // Fuse lshlrev + or -> lshl_or
    patterns.add<LshlAddPattern>(ctx);
    patterns.add<RedundantMovePattern>(ctx);
    patterns.add<MulPow2ToShiftPattern>(ctx);
    patterns.add<AddZeroPattern>(ctx);
    patterns.add<MulOnePattern>(ctx);
    patterns.add<MulZeroPattern>(ctx);
    patterns.add<MFMAZeroAccumPattern>(ctx);

    // Apply patterns greedily
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace waveasm {

std::unique_ptr<mlir::Pass> createWAVEASMPeepholePass() {
  return std::make_unique<PeepholePass>();
}

} // namespace waveasm
