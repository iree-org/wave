// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::gpu;

namespace mlir::water {
#define GEN_PASS_DEF_WATERGPUMODULETOBINARY
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

namespace {
class WaterGPUModuleToBinaryPass
    : public water::impl::WaterGPUModuleToBinaryBase<
          WaterGPUModuleToBinaryPass> {
public:
  using Base::Base;
  void runOnOperation() final;
};
} // namespace

void WaterGPUModuleToBinaryPass::runOnOperation() {
  // Parse compilation target format
  auto targetFormat =
      llvm::StringSwitch<std::optional<CompilationTarget>>(compilationTarget)
          .Cases({"offloading", "llvm"}, CompilationTarget::Offload)
          .Cases({"assembly", "isa"}, CompilationTarget::Assembly)
          .Cases({"binary", "bin"}, CompilationTarget::Binary)
          .Cases({"fatbinary", "fatbin"}, CompilationTarget::Fatbin)
          .Default(std::nullopt);

  if (!targetFormat) {
    getOperation()->emitError()
        << "Invalid format specified: " << compilationTarget;
    return signalPassFailure();
  }

  // TODO: Implement the actual serialization logic
  // This is a stub that will be filled in with:
  // 1. Walk all GPUModuleOp instances
  // 2. For each module, serialize using target attributes
  // 3. Create gpu.binary ops with the serialized objects
  // 4. Erase the original gpu.module ops

  getOperation()->emitRemark() << "WaterGPUModuleToBinary pass stub executed";
}
