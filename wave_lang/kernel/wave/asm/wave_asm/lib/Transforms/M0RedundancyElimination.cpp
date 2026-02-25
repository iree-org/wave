// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// M0 Redundancy Elimination Pass
//
// Eliminates redundant s_mov_b32_m0 writes by tracking the last M0 source
// value within each basic block and erasing writes that set M0 to the same
// value it already holds.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMM0REDUNDANCYELIM
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

using namespace mlir;
using namespace waveasm;

namespace {

struct M0RedundancyEliminationPass
    : public waveasm::impl::WAVEASMM0RedundancyElimBase<
          M0RedundancyEliminationPass> {
  using WAVEASMM0RedundancyElimBase::WAVEASMM0RedundancyElimBase;

  void runOnOperation() override {
    getOperation()->walk([](Block *block) {
      Value lastM0Source;
      SmallVector<Operation *> toErase;

      for (Operation &op : *block) {
        auto m0Op = dyn_cast<S_MOV_B32_M0>(&op);
        if (!m0Op) {
          // Conservatively reset tracking across region-bearing ops
          // (loops, ifs) — their bodies may clobber M0.
          if (op.getNumRegions() > 0)
            lastM0Source = Value{};
          continue;
        }

        Value src = m0Op.getSrc();

        // If M0 already holds this value, the write is redundant.
        if (src == lastM0Source) {
          toErase.push_back(&op);
        } else {
          lastM0Source = src;
        }
      }

      for (Operation *op : toErase)
        op->erase();
    });
  }
};

} // namespace
