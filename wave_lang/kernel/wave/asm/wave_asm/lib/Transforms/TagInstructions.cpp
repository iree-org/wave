// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Tag Instructions Pass - Attach stable NameLoc tags to WaveASM ops.
//
// Each operation gets a tag like loc("buffer_load_dwordx4_0"),
// loc("v_mfma_f32_16x16x16_f16_3"), etc. Tags are stable across
// scheduling rounds (they follow the operation, not the position).
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "waveasm-tag-instructions"

using namespace mlir;
using namespace waveasm;

namespace {

struct TagInstructionsPass
    : public PassWrapper<TagInstructionsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TagInstructionsPass)

  StringRef getArgument() const override { return "waveasm-tag-instructions"; }

  StringRef getDescription() const override {
    return "Attach stable NameLoc tags to WaveASM instructions";
  }

  void runOnOperation() override {
    unsigned total = 0;
    MLIRContext *ctx = &getContext();
    llvm::DenseMap<StringRef, unsigned> counters;

    getOperation().walk([&](Operation *op) {
      if (op->getDialect() && op->getDialect()->getNamespace() == "waveasm") {
        StringRef opName = op->getName().stripDialect();
        unsigned idx = counters[opName]++;
        std::string tag = (opName + "_" + llvm::Twine(idx)).str();
        op->setLoc(NameLoc::get(StringAttr::get(ctx, tag)));
        ++total;
      }
    });

    LDBG() << "tagged " << total << " operations";
  }
};

} // namespace

namespace waveasm {

std::unique_ptr<mlir::Pass> createWAVEASMTagInstructionsPass() {
  return std::make_unique<TagInstructionsPass>();
}

} // namespace waveasm
