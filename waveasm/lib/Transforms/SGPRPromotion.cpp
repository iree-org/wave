// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// SGPR Promotion Pass
//
// Identifies VALU operations whose operands are all SGPR or immediate and
// replaces them with SALU equivalents.  This reduces VGPR pressure for
// uniform computations (e.g. workgroup staggering, SRD base address
// calculation) that the translation handlers unconditionally emitted as
// VALU because affine.apply always produces VGPRs.
//
// The pass walks the IR in program order so that promotions cascade: once
// a V_ADD_U32 becomes S_ADD_U32, its downstream users may also become
// promotable.
//
// After promotion, V_READFIRSTLANE_B32 ops whose source is already SGPR
// become identity operations and are eliminated.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "waveasm-sgpr-promotion"

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMSGPRPROMOTION
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

using namespace mlir;
using namespace waveasm;

namespace {

static bool isScalarOrImm(Value v) {
  Type ty = v.getType();
  return isSGPRType(ty) || isImmType(ty);
}

/// Check that all operands of an operation are scalar (SGPR or immediate).
static bool allOperandsScalar(Operation *op) {
  return llvm::all_of(op->getOperands(),
                      [](Value v) { return isScalarOrImm(v); });
}

/// Check whether any user of a value is a region-bearing op (loop, if).
/// Promoting such values would create a type mismatch between the init
/// operand (now SGPR) and the region block argument (still VGPR/AGPR).
static bool hasRegionBearingUser(Value v) {
  for (Operation *user : v.getUsers()) {
    if (user->getNumRegions() > 0)
      return true;
  }
  return false;
}

/// Check whether a value was produced by an op that is truly Pure
/// (no SCC side effects).  Only these are safe to expose to LICM
/// after readfirstlane elimination.  S_MUL_I32 is Pure; S_AND_B32,
/// S_LSHL_B32 etc. set SCC and are not.
static bool isProducedByPureOp(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  return isPure(def);
}

/// Materialize an operand for SALU use.  If the value is an immediate,
/// emit S_MOV_B32 to place it in an SGPR (SALU src0 must be SGPR;
/// src1 can be SGPR or inline constant, but literals need S_MOV).
static Value ensureSGPR(Value v, OpBuilder &builder, Location loc,
                        SRegType sregType) {
  if (isSGPRType(v.getType()))
    return v;
  if (isImmType(v.getType()))
    return S_MOV_B32::create(builder, loc, sregType, v);
  return v;
}

struct SGPRPromotionPass
    : public waveasm::impl::WAVEASMSGPRPromotionBase<SGPRPromotionPass> {
  using WAVEASMSGPRPromotionBase::WAVEASMSGPRPromotionBase;

  void runOnOperation() override {
    unsigned numPromoted = 0;
    unsigned numReadfirstlaneElim = 0;

    auto *ctx = &getContext();

    getOperation()->walk([&](Block *block) {
      for (Operation &op : llvm::make_early_inc_range(*block)) {
        // --- V_READFIRSTLANE_B32 with SGPR source ---
        // When an op feeding readfirstlane was promoted to SALU, the
        // source becomes SGPR and readfirstlane is semantically an
        // identity.  We can eliminate it IF the defining op is truly
        // Pure (doesn't set SCC) -- otherwise LICM might hoist the
        // SCC-setting op across a live SCC range, corrupting loop
        // control flow.  For non-Pure sources, insert V_MOV_B32 to
        // satisfy readfirstlane's VGPR operand constraint.
        if (auto rfl = dyn_cast<V_READFIRSTLANE_B32>(&op)) {
          if (isSGPRType(rfl.getSrc().getType())) {
            if (isProducedByPureOp(rfl.getSrc())) {
              rfl.getDst().replaceAllUsesWith(rfl.getSrc());
              op.erase();
              ++numReadfirstlaneElim;
            } else {
              auto vregTy = VRegType::get(ctx, 1, 1);
              OpBuilder rflBuilder(&op);
              Value vgprCopy = V_MOV_B32::create(rflBuilder, op.getLoc(),
                                                 vregTy, rfl.getSrc());
              op.setOperand(0, vgprCopy);
            }
          }
          continue;
        }

        // --- VALU → SALU promotion ---
        if (!allOperandsScalar(&op))
          continue;

        // Don't promote if the result feeds a region-bearing op (loop, if)
        // as an init operand -- the region block args have fixed types and
        // changing the init from VGPR to SGPR would cause a type mismatch.
        if (op.getNumResults() > 0 && hasRegionBearingUser(op.getResult(0)))
          continue;

        // Don't promote if any result is not a simple VGPR (skip AGPR, etc.)
        if (op.getNumResults() > 0 && !isVGPRType(op.getResult(0).getType()))
          continue;

        auto sregType = SRegType::get(ctx, 1, 1);
        auto loc = op.getLoc();
        OpBuilder builder(&op);

        // Binary VALU → SALU promotions
        auto promoteBinary = [&](Value src0, Value src1,
                                 auto saluCreateFn) -> bool {
          Value s0 = ensureSGPR(src0, builder, loc, sregType);
          Value s1 = ensureSGPR(src1, builder, loc, sregType);
          Value result = saluCreateFn(builder, loc, sregType, s0, s1);
          op.getResult(0).replaceAllUsesWith(result);
          op.erase();
          ++numPromoted;
          return true;
        };

        // S_ADD_U32 and S_SUB_U32 produce two results (dst, scc).
        // We promote V_ADD/V_SUB (single result) by creating the SALU
        // op and using its first result.
        auto promoteBinaryWithCarry = [&](Value src0, Value src1,
                                          auto saluCreateFn) -> bool {
          Value s0 = ensureSGPR(src0, builder, loc, sregType);
          Value s1 = ensureSGPR(src1, builder, loc, sregType);
          auto saluOp = saluCreateFn(builder, loc, sregType, sregType, s0, s1);
          op.getResult(0).replaceAllUsesWith(saluOp->getResult(0));
          op.erase();
          ++numPromoted;
          return true;
        };

        // V_LSHLREV/V_LSHRREV have reversed operands: (shift_amt, value)
        // SALU shifts have normal order: (value, shift_amt)
        auto promoteShiftRev = [&](Value shiftAmt, Value value,
                                   auto saluCreateFn) -> bool {
          Value s0 = ensureSGPR(value, builder, loc, sregType);
          Value s1 = ensureSGPR(shiftAmt, builder, loc, sregType);
          Value result = saluCreateFn(builder, loc, sregType, s0, s1);
          op.getResult(0).replaceAllUsesWith(result);
          op.erase();
          ++numPromoted;
          return true;
        };

        bool promoted = false;

        if (auto addOp = dyn_cast<V_ADD_U32>(&op)) {
          promoted = promoteBinaryWithCarry(
              addOp.getSrc0(), addOp.getSrc1(),
              [](OpBuilder &b, Location l, SRegType t1, SRegType t2, Value s0,
                 Value s1) { return S_ADD_U32::create(b, l, t1, t2, s0, s1); });
        } else if (auto subOp = dyn_cast<V_SUB_U32>(&op)) {
          promoted = promoteBinaryWithCarry(
              subOp.getSrc0(), subOp.getSrc1(),
              [](OpBuilder &b, Location l, SRegType t1, SRegType t2, Value s0,
                 Value s1) { return S_SUB_U32::create(b, l, t1, t2, s0, s1); });
        } else if (auto mulOp = dyn_cast<V_MUL_LO_U32>(&op)) {
          promoted = promoteBinary(mulOp.getSrc0(), mulOp.getSrc1(),
                                   [](OpBuilder &b, Location l, SRegType t,
                                      Value s0, Value s1) {
                                     return S_MUL_I32::create(b, l, t, s0, s1);
                                   });
        } else if (auto andOp = dyn_cast<V_AND_B32>(&op)) {
          promoted = promoteBinary(andOp.getSrc0(), andOp.getSrc1(),
                                   [](OpBuilder &b, Location l, SRegType t,
                                      Value s0, Value s1) {
                                     return S_AND_B32::create(b, l, t, s0, s1);
                                   });
        } else if (auto orOp = dyn_cast<V_OR_B32>(&op)) {
          promoted = promoteBinary(orOp.getSrc0(), orOp.getSrc1(),
                                   [](OpBuilder &b, Location l, SRegType t,
                                      Value s0, Value s1) {
                                     return S_OR_B32::create(b, l, t, s0, s1);
                                   });
        } else if (auto xorOp = dyn_cast<V_XOR_B32>(&op)) {
          promoted = promoteBinary(xorOp.getSrc0(), xorOp.getSrc1(),
                                   [](OpBuilder &b, Location l, SRegType t,
                                      Value s0, Value s1) {
                                     return S_XOR_B32::create(b, l, t, s0, s1);
                                   });
        } else if (auto lshlOp = dyn_cast<V_LSHLREV_B32>(&op)) {
          promoted = promoteShiftRev(lshlOp.getSrc0(), lshlOp.getSrc1(),
                                     [](OpBuilder &b, Location l, SRegType t,
                                        Value s0, Value s1) {
                                       return S_LSHL_B32::create(b, l, t, s0,
                                                                 s1);
                                     });
        } else if (auto lshrOp = dyn_cast<V_LSHRREV_B32>(&op)) {
          promoted = promoteShiftRev(lshrOp.getSrc0(), lshrOp.getSrc1(),
                                     [](OpBuilder &b, Location l, SRegType t,
                                        Value s0, Value s1) {
                                       return S_LSHR_B32::create(b, l, t, s0,
                                                                 s1);
                                     });
        } else if (auto ashrOp = dyn_cast<V_ASHRREV_I32>(&op)) {
          promoted = promoteShiftRev(ashrOp.getSrc0(), ashrOp.getSrc1(),
                                     [](OpBuilder &b, Location l, SRegType t,
                                        Value s0, Value s1) {
                                       return S_ASHR_I32::create(b, l, t, s0,
                                                                 s1);
                                     });
        } else if (auto movOp = dyn_cast<V_MOV_B32>(&op)) {
          // V_MOV_B32 can produce AGPR results (for accumulator init).
          // Only promote to S_MOV_B32 when the result is a VGPR.
          if (isVGPRType(movOp.getDst().getType())) {
            Value s0 = ensureSGPR(movOp.getSrc(), builder, loc, sregType);
            Value result = S_MOV_B32::create(builder, loc, sregType, s0);
            op.getResult(0).replaceAllUsesWith(result);
            op.erase();
            ++numPromoted;
            promoted = true;
          }
        } else if (auto lshlOrOp = dyn_cast<V_LSHL_OR_B32>(&op)) {
          // Decompose V_LSHL_OR_B32(src, shift, orend) into
          // S_LSHL_B32 + S_OR_B32 (no ternary SALU equivalent).
          Value src = ensureSGPR(lshlOrOp.getSrc0(), builder, loc, sregType);
          Value shift = ensureSGPR(lshlOrOp.getSrc1(), builder, loc, sregType);
          Value orend = ensureSGPR(lshlOrOp.getSrc2(), builder, loc, sregType);
          Value shifted = S_LSHL_B32::create(builder, loc, sregType, src, shift);
          Value result = S_OR_B32::create(builder, loc, sregType, shifted, orend);
          op.getResult(0).replaceAllUsesWith(result);
          op.erase();
          ++numPromoted;
          promoted = true;
        } else if (auto cndOp = dyn_cast<V_CNDMASK_B32>(&op)) {
          // V_CNDMASK_B32 reads VCC implicitly.  We can only promote
          // to S_CSELECT_B32 (which reads SCC) when the VCC was set
          // by an S_CMP (i.e. the vcc_dep operand traces back to an
          // S_CMP result that wrote SCC).  For now, skip this case --
          // the main benefit comes from the arithmetic promotions above.
          // TODO: Implement V_CNDMASK → S_CSELECT when safe.
          (void)cndOp;
        }

        (void)promoted;
      }
    });

    // --- Post-promotion fixup: insert V_MOV_B32 for SGPR values used ---
    // --- by VALU ops that can't accept SGPR in that operand position ---
    // Only run fixup when promotions actually happened.
    unsigned numFixups = 0;
    if (numPromoted == 0)
      return;
    auto vregType = VRegType::get(ctx, 1, 1);

    getOperation()->walk([&](Operation *valuOp) {
      // Only fix VALU ops (ops that produce VGPR results or have no results
      // but are known VALU instructions).
      bool isVALU = false;
      if (valuOp->getNumResults() > 0 &&
          isVGPRType(valuOp->getResult(0).getType()))
        isVALU = true;
      // V_CNDMASK_B32 produces VGPR
      if (isa<V_CNDMASK_B32>(valuOp))
        isVALU = true;
      if (!isVALU)
        return;

      // V_CNDMASK_B32: src1 (index 1) MUST be VGPR in VOP2 encoding.
      // src0 (index 0) CAN be SGPR or inline constant.
      // Also enforce: at most one SGPR/constant-bus source among src0/src1.
      if (auto cndOp = dyn_cast<V_CNDMASK_B32>(valuOp)) {
        Value src1 = cndOp.getSrc1();
        if (isSGPRType(src1.getType())) {
          llvm::errs() << "SGPR-FIXUP-CNDMASK: at ";
          valuOp->getLoc().print(llvm::errs());
          llvm::errs() << "\n";
          OpBuilder fixBuilder(valuOp);
          Value copy = V_MOV_B32::create(fixBuilder, valuOp->getLoc(),
                                         vregType, src1);
          valuOp->setOperand(1, copy);
          ++numFixups;
        }
        return;
      }

      // General VALU constant bus restriction: at most one SGPR source.
      // GFX9 VOP3 supports multiple SGPRs from the SAME pair, but different
      // SGPR pairs violate the restriction.  Conservatively copy extras.
      unsigned sgrpCount = 0;
      for (unsigned i = 0; i < valuOp->getNumOperands(); ++i) {
        if (isSGPRType(valuOp->getOperand(i).getType()))
          ++sgrpCount;
      }
      if (sgrpCount > 1) {
        bool seenFirst = false;
        for (unsigned i = 0; i < valuOp->getNumOperands(); ++i) {
          if (isSGPRType(valuOp->getOperand(i).getType())) {
            if (!seenFirst) {
              seenFirst = true;
              continue;
            }
            OpBuilder fixBuilder(valuOp);
            Value copy = V_MOV_B32::create(fixBuilder, valuOp->getLoc(),
                                           vregType, valuOp->getOperand(i));
            valuOp->setOperand(i, copy);
            ++numFixups;
          }
        }
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "SGPRPromotion: promoted " << numPromoted
                            << " VALU ops to SALU, eliminated "
                            << numReadfirstlaneElim
                            << " V_READFIRSTLANE_B32 ops, inserted "
                            << numFixups << " SGPR->VGPR fixup copies\n");
  }
};

} // namespace
