// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/Transforms/Passes.h"
#include "water/Dialect/Wave/Transforms/Utils.h"

#define DEBUG_TYPE "wave-propagate-defaults-from-constraints"

namespace wave {
#define GEN_PASS_DEF_WATERWAVEPROPAGATEDEFAULTSFROMCONSTRAINTSPASS
#include "water/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace wave

using namespace mlir;
using namespace wave;

namespace {

struct PropagateDefaultsFromConstraints
    : public wave::impl::WaterWavePropagateDefaultsFromConstraintsPassBase<
          PropagateDefaultsFromConstraints> {
  void runOnOperation() override {
    // Do a first pass looking for constraint top-down and, when found, visiting
    // all MmaOps to which the constraints apply. Then do a second pass to check
    // if we missed something. This is practically faster than looking up the
    // ancestor for every MmaOp.
    DenseMap<Operation *, Attribute> constraints;
    if (failed(collectWaveConstraints(getOperation(), constraints)))
      return signalPassFailure();

    for (auto [op, attr] : constraints) {
      auto it = llvm::find_if(cast<ArrayAttr>(attr).getValue(),
                              llvm::IsaPred<HardwareConstraintAttr>);
      if (it == cast<ArrayAttr>(attr).getValue().end())
        continue;
      op->walk([&](MmaOp mmaOp) {
        if (!mmaOp.getKindAttr())
          mmaOp.setKindAttr(cast<HardwareConstraintAttr>(*it).getMmaType());
      });
    }

    WalkResult walkResult = getOperation()->walk([&](MmaOp mmaOp) {
      if (mmaOp.getKindAttr())
        return WalkResult::advance();

      mmaOp.emitError("mma operation without kind attribute and no constraint "
                      "to derive it from");
      return WalkResult::interrupt();
    });
    if (walkResult.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace
