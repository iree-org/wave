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

using namespace mlir;
using namespace waveasm;

namespace {

struct M0RedundancyEliminationPass
    : public PassWrapper<M0RedundancyEliminationPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(M0RedundancyEliminationPass)

  M0RedundancyEliminationPass() = default;

  StringRef getArgument() const override {
    return "waveasm-m0-redundancy-elim";
  }

  StringRef getDescription() const override {
    return "Eliminate redundant M0 register writes in WAVEASM IR";
  }

  void runOnOperation() override {
    getOperation()->walk([](ProgramOp program) {
      program.walk([](Block *block) {
        Value lastM0Source;
        SmallVector<Operation *, 8> toErase;

        for (Operation &op : *block) {
          auto m0Op = dyn_cast<S_MOV_B32_M0>(&op);
          if (!m0Op)
            continue;

          if (m0Op->getNumOperands() < 1)
            continue;

          Value src = m0Op->getOperand(0);

          // If M0 already holds this value, the write is redundant.
          if (src == lastM0Source) {
            toErase.push_back(m0Op);
          } else {
            lastM0Source = src;
          }
        }

        for (auto *op : toErase)
          op->erase();
      });
    });
  }
};

} // namespace

namespace waveasm {

std::unique_ptr<mlir::Pass> createWAVEASMM0RedundancyElimPass() {
  return std::make_unique<M0RedundancyEliminationPass>();
}

} // namespace waveasm
