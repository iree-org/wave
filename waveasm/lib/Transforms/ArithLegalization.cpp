// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Arithmetic Legalization Pass
//
// Lowers generic arithmetic pseudo-ops (arith.add, arith.mul, arith.cmp,
// arith.select, arith.trunc, arith.sext, arith.zext) to concrete SALU or
// VALU machine ops based on operand register files.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMARITHLEGALIZATION
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

using namespace mlir;
using namespace waveasm;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Return true if any operand is a VGPR (divergent context).
static bool anyVGPR(ValueRange operands) {
  return llvm::any_of(operands,
                      [](Value v) { return isVGPRType(v.getType()); });
}

/// Narrow a wide (size>1) register to its low sub-register.
/// For precolored SGPRs, creates a new precolored SGPR with size=1.
/// TODO: Proper i64 legalization with carry chains.
static Value narrowToI32(Value v, OpBuilder &builder, Location loc) {
  if (auto psreg = v.getDefiningOp<PrecoloredSRegOp>()) {
    if (psreg.getSize() > 1) {
      auto sregTy = SRegType::get(builder.getContext(), 1, 1);
      return PrecoloredSRegOp::create(builder, loc, sregTy, psreg.getIndex(),
                                      /*size=*/1);
    }
  }
  if (auto sreg = dyn_cast<SRegType>(v.getType())) {
    if (sreg.getSize() > 1) {
      auto sregTy = SRegType::get(builder.getContext(), 1, 1);
      return S_MOV_B32::create(builder, loc, sregTy, v);
    }
  }
  if (auto vreg = dyn_cast<VRegType>(v.getType())) {
    if (vreg.getSize() > 1) {
      auto vregTy = VRegType::get(builder.getContext());
      return V_MOV_B32::create(builder, loc, vregTy, v);
    }
  }
  return v;
}

/// Move an SGPR value to a VGPR via v_mov_b32.
static Value sgprToVgpr(Value v, OpBuilder &builder, Location loc) {
  if (!isSGPRType(v.getType()))
    return v;
  v = narrowToI32(v, builder, loc);
  auto vregTy = VRegType::get(builder.getContext());
  return V_MOV_B32::create(builder, loc, vregTy, v);
}

//===----------------------------------------------------------------------===//
// Legalization functions
//===----------------------------------------------------------------------===//

static void legalizeAdd(Arith_AddOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  Value lhs = narrowToI32(op.getLhs(), builder, loc);
  Value rhs = narrowToI32(op.getRhs(), builder, loc);

  Value result;
  if (anyVGPR({lhs, rhs})) {
    if (isSGPRType(lhs.getType()) && isSGPRType(rhs.getType()))
      lhs = sgprToVgpr(lhs, builder, loc);
    auto vregTy = VRegType::get(builder.getContext());
    result = V_ADD_U32::create(builder, loc, vregTy, lhs, rhs);
  } else {
    auto sregTy = SRegType::get(builder.getContext(), 1, 1);
    result =
        S_ADD_U32::create(builder, loc, sregTy, sregTy, lhs, rhs)->getResult(0);
  }
  op.replaceAllUsesWith(result);
  op.erase();
}

static void legalizeMul(Arith_MulOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  Value lhs = narrowToI32(op.getLhs(), builder, loc);
  Value rhs = narrowToI32(op.getRhs(), builder, loc);

  Value result;
  if (anyVGPR({lhs, rhs})) {
    if (isSGPRType(lhs.getType()) && isSGPRType(rhs.getType()))
      lhs = sgprToVgpr(lhs, builder, loc);
    auto vregTy = VRegType::get(builder.getContext());
    result = V_MUL_LO_U32::create(builder, loc, vregTy, lhs, rhs);
  } else {
    auto sregTy = SRegType::get(builder.getContext(), 1, 1);
    result = S_MUL_I32::create(builder, loc, sregTy, lhs, rhs);
  }
  op.replaceAllUsesWith(result);
  op.erase();
}

static void legalizeCmp(Arith_CmpOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  Value lhs = narrowToI32(op.getLhs(), builder, loc);
  Value rhs = narrowToI32(op.getRhs(), builder, loc);
  auto pred = op.getPredicate();

  if (anyVGPR({lhs, rhs})) {
    // VALU: v_cmp sets VCC, no explicit result.
    if (isSGPRType(lhs.getType()))
      lhs = sgprToVgpr(lhs, builder, loc);

    switch (pred) {
    case CmpPredicate::eq:
      V_CMP_EQ_I32::create(builder, loc, lhs, rhs);
      break;
    case CmpPredicate::ne:
      V_CMP_NE_I32::create(builder, loc, lhs, rhs);
      break;
    case CmpPredicate::slt:
      V_CMP_LT_I32::create(builder, loc, lhs, rhs);
      break;
    case CmpPredicate::sle:
      V_CMP_LE_I32::create(builder, loc, lhs, rhs);
      break;
    case CmpPredicate::sgt:
      V_CMP_GT_I32::create(builder, loc, lhs, rhs);
      break;
    case CmpPredicate::sge:
      V_CMP_GE_I32::create(builder, loc, lhs, rhs);
      break;
    case CmpPredicate::ult:
      V_CMP_LT_U32::create(builder, loc, lhs, rhs);
      break;
    case CmpPredicate::ule:
      V_CMP_LE_U32::create(builder, loc, lhs, rhs);
      break;
    case CmpPredicate::ugt:
      V_CMP_GT_U32::create(builder, loc, lhs, rhs);
      break;
    case CmpPredicate::uge:
      V_CMP_GE_U32::create(builder, loc, lhs, rhs);
      break;
    }
    // VCC-setting compares have no explicit result. Create a placeholder
    // constant for uses (v_cndmask_b32 reads VCC implicitly).
    auto immTy = ImmType::get(builder.getContext(), 1);
    auto placeholder = ConstantOp::create(builder, loc, immTy, 1);
    op.replaceAllUsesWith(placeholder.getResult());
  } else {
    // SALU: s_cmp sets SCC, modeled as an SGPR result.
    auto sregTy = SRegType::get(builder.getContext(), 1, 1);
    Value result;
    switch (pred) {
    case CmpPredicate::eq:
      result = S_CMP_EQ_I32::create(builder, loc, sregTy, lhs, rhs);
      break;
    case CmpPredicate::ne:
      result = S_CMP_NE_I32::create(builder, loc, sregTy, lhs, rhs);
      break;
    case CmpPredicate::slt:
      result = S_CMP_LT_I32::create(builder, loc, sregTy, lhs, rhs);
      break;
    case CmpPredicate::sle:
      result = S_CMP_LE_I32::create(builder, loc, sregTy, lhs, rhs);
      break;
    case CmpPredicate::sgt:
      result = S_CMP_GT_I32::create(builder, loc, sregTy, lhs, rhs);
      break;
    case CmpPredicate::sge:
      result = S_CMP_GE_I32::create(builder, loc, sregTy, lhs, rhs);
      break;
    case CmpPredicate::ult:
      result = S_CMP_LT_U32::create(builder, loc, sregTy, lhs, rhs);
      break;
    case CmpPredicate::ule:
      result = S_CMP_LE_U32::create(builder, loc, sregTy, lhs, rhs);
      break;
    case CmpPredicate::ugt:
      result = S_CMP_GT_U32::create(builder, loc, sregTy, lhs, rhs);
      break;
    case CmpPredicate::uge:
      result = S_CMP_GE_U32::create(builder, loc, sregTy, lhs, rhs);
      break;
    }
    op.replaceAllUsesWith(result);
  }
  op.erase();
}

static void legalizeSelect(Arith_SelectOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  Value falseVal = narrowToI32(op.getFalseVal(), builder, loc);
  Value trueVal = narrowToI32(op.getTrueVal(), builder, loc);
  Value cond = op.getCondition();

  // v_cndmask_b32: dst = cond ? trueVal : falseVal.
  auto vregTy = VRegType::get(builder.getContext());
  if (!isVGPRType(falseVal.getType()))
    falseVal = sgprToVgpr(falseVal, builder, loc);
  if (!isVGPRType(trueVal.getType()))
    trueVal = sgprToVgpr(trueVal, builder, loc);
  auto sel =
      V_CNDMASK_B32::create(builder, loc, vregTy, falseVal, trueVal, cond);
  op.replaceAllUsesWith(sel.getResult());
  op.erase();
}

static void legalizeTrunc(Arith_TruncOp op, OpBuilder &builder) {
  Value result = narrowToI32(op.getSrc(), builder, op.getLoc());
  op.replaceAllUsesWith(result);
  op.erase();
}

static void legalizeSExt(Arith_SExtOp op) {
  // For now, pass through (consumers narrow back to i32 anyway).
  // TODO: Produce a register pair {lo, ashr(lo, 31)}.
  op.replaceAllUsesWith(op.getSrc());
  op.erase();
}

static void legalizeZExt(Arith_ZExtOp op) {
  // For now, pass through (consumers narrow back to i32 anyway).
  // TODO: Produce a register pair {lo, 0}.
  op.replaceAllUsesWith(op.getSrc());
  op.erase();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {

struct ArithLegalizationPass
    : waveasm::impl::WAVEASMArithLegalizationBase<ArithLegalizationPass> {

  void runOnOperation() override {
    // Collect pseudo-ops first to avoid invalidating the walk iterator.
    SmallVector<Operation *> toLegalize;
    getOperation()->walk([&](Operation *op) {
      if (isa<Arith_AddOp, Arith_MulOp, Arith_CmpOp, Arith_SelectOp,
              Arith_TruncOp, Arith_SExtOp, Arith_ZExtOp>(op))
        toLegalize.push_back(op);
    });

    for (auto *op : toLegalize) {
      OpBuilder builder(op);
      TypeSwitch<Operation *>(op)
          .Case([&](Arith_AddOp o) { legalizeAdd(o, builder); })
          .Case([&](Arith_MulOp o) { legalizeMul(o, builder); })
          .Case([&](Arith_CmpOp o) { legalizeCmp(o, builder); })
          .Case([&](Arith_SelectOp o) { legalizeSelect(o, builder); })
          .Case([&](Arith_TruncOp o) { legalizeTrunc(o, builder); })
          .Case([&](Arith_SExtOp o) { legalizeSExt(o); })
          .Case([&](Arith_ZExtOp o) { legalizeZExt(o); });
    }
  }
};

} // namespace
