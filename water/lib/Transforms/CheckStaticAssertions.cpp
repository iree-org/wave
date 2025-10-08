// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::water {
#define GEN_PASS_DEF_WATERCHECKSTATICASSERTIONSPASS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

/// Simple pass that emits an error when it sees a `cf.assert` with a
/// known-failing condition.
class CheckStaticAssertionsPass
    : public water::impl::WaterCheckStaticAssertionsPassBase<
          CheckStaticAssertionsPass> {
public:
  void runOnOperation() override {
    WalkResult walkResult = getOperation()->walk([&](cf::AssertOp assertOp) {
      APInt value;
      if (matchPattern(assertOp.getArg(), m_ConstantInt(&value)) &&
          value.isZero()) {
        assertOp->emitError() << "assertion known to be false";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      signalPassFailure();
  }
};
