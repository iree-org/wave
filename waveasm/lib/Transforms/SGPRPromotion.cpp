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
        // --- V_READFIRSTLANE_B32 elimination ---
        if (auto rfl = dyn_cast<V_READFIRSTLANE_B32>(&op)) {
          if (isSGPRType(rfl.getSrc().getType())) {
            rfl.getDst().replaceAllUsesWith(rfl.getSrc());
            op.erase();
            ++numReadfirstlaneElim;
          }
          continue;
        }

        // --- VALU → SALU promotion ---
        if (!allOperandsScalar(&op))
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
          Value s0 = ensureSGPR(movOp.getSrc(), builder, loc, sregType);
          Value result = S_MOV_B32::create(builder, loc, sregType, s0);
          op.getResult(0).replaceAllUsesWith(result);
          op.erase();
          ++numPromoted;
          promoted = true;
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

    LLVM_DEBUG(llvm::dbgs() << "SGPRPromotion: promoted " << numPromoted
                            << " VALU ops to SALU, eliminated "
                            << numReadfirstlaneElim
                            << " V_READFIRSTLANE_B32 ops\n");
  }
};

} // namespace
