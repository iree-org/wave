// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/Transforms/Passes.h"

using namespace mlir;
using namespace wave;

namespace wave {
#define GEN_PASS_DEF_WATERWAVEDETECTNORMALFORMSPASS
#include "water/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace wave

namespace {

struct WaterWaveDetectNormalFormsPass
    : public wave::impl::WaterWaveDetectNormalFormsPassBase<
          WaterWaveDetectNormalFormsPass> {
  void runOnOperation() override {
    Operation *op = getOperation();

    uint64_t forms = 0;
    for (unsigned bit = 0, lastBit = WaveNormalFormAttr::getLastSetBit();
         bit <= lastBit; ++bit) {
      WaveNormalForm form =
          static_cast<WaveNormalForm>(static_cast<uint64_t>(1) << bit);
      if (succeeded(wave::detail::verifyNormalFormAttr(
              getOperation(), form, /*emitDiagnostics=*/false))) {
        forms |= static_cast<uint64_t>(form);
      }
    }
    op->setAttr(WaveDialect::kNormalFormAttrName,
                WaveNormalFormAttr::get(op->getContext(),
                                        static_cast<WaveNormalForm>(forms)));
  }
};

} // end anonymous namespace
