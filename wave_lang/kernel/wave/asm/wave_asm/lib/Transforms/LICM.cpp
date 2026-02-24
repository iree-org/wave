// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Loop-Invariant Code Motion (LICM) Pass
//
// Hoists VALU address-computation instructions whose operands all dominate the
// loop out of LoopOp bodies.  This matches the aiter pattern where all LDS
// address computation is in the prologue and zero VALU appears in the hot loop.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace waveasm;

namespace {

struct LICMPass : public PassWrapper<LICMPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LICMPass)

  LICMPass() = default;

  StringRef getArgument() const override { return "waveasm-licm"; }

  StringRef getDescription() const override {
    return "Loop-invariant code motion for WAVEASM IR";
  }

  void runOnOperation() override {
    // Post-order: process inner loops first so ops hoisted from inner to
    // outer loop bodies get a second chance to be hoisted further out.
    getOperation()->walk<WalkOrder::PostOrder>([](LoopOp loopOp) {
      Block &body = loopOp.getBodyBlock();
      Region *loopRegion = &loopOp.getBodyRegion();

      // Check if a value dominates the loop (defined outside the region).
      auto dominatesLoop = [&](Value val) -> bool {
        if (auto *defOp = val.getDefiningOp())
          return defOp->getParentRegion() != loopRegion;
        // Block arguments of the loop body do NOT dominate (they're
        // loop-carried values that change each iteration).
        if (auto blockArg = dyn_cast<BlockArgument>(val))
          return blockArg.getOwner()->getParentOp() != loopOp.getOperation();
        return false;
      };

      // Check if an op is a VALU (address computation) that's safe to hoist.
      auto isHoistableVALU = [](Operation *op) -> bool {
        return isa<V_LSHLREV_B32, V_ADD_U32, V_MUL_LO_U32, V_AND_B32, V_OR_B32,
                   V_LSHL_ADD_U32, V_LSHRREV_B32>(op);
      };

      // Iteratively collect and hoist: after hoisting a batch, new ops
      // may become hoistable (their producer was just hoisted).
      bool changed = true;
      while (changed) {
        changed = false;
        SmallVector<Operation *> toHoist;
        for (Operation &op : body) {
          if (!isHoistableVALU(&op))
            continue;
          bool allDominate = llvm::all_of(
              op.getOperands(), [&](Value v) { return dominatesLoop(v); });
          if (allDominate)
            toHoist.push_back(&op);
        }
        for (Operation *op : toHoist) {
          op->moveBefore(loopOp);
          changed = true;
        }
      }
    });
  }
};

} // namespace

namespace waveasm {

std::unique_ptr<mlir::Pass> createWAVEASMLICMPass() {
  return std::make_unique<LICMPass>();
}

} // namespace waveasm
