// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

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

private:
  LogicalResult serializeModule(GPUModuleOp module);
};
} // namespace

LogicalResult WaterGPUModuleToBinaryPass::serializeModule(GPUModuleOp module) {
  OpBuilder builder(module->getContext());

  // Check if module has target attributes
  if (!module.getTargetsAttr() || module.getTargetsAttr().empty()) {
    return module.emitError("GPU module has no target attributes");
  }

  // Collect serialized objects for each target
  SmallVector<Attribute> objects;

  for (auto targetAttr : module.getTargetsAttr()) {
    if (!targetAttr) {
      return module.emitError("Target attribute cannot be null");
    }

    auto target = dyn_cast<gpu::TargetAttrInterface>(targetAttr);
    if (!target) {
      return module.emitError(
          "Target attribute doesn't implement TargetAttrInterface");
    }

    // Build target options
    SmallVector<Attribute> librariesToLink;
    for (const std::string &path : linkFiles) {
      librariesToLink.push_back(StringAttr::get(&getContext(), path));
    }

    // Create lazy symbol table builder
    std::optional<SymbolTable> parentTable;
    auto lazyTableBuilder = [&]() -> SymbolTable * {
      if (!parentTable) {
        Operation *table = SymbolTable::getNearestSymbolTable(module);
        if (!table)
          return nullptr;
        parentTable = SymbolTable(table);
      }
      return &parentTable.value();
    };

    TargetOptions targetOptions(toolkitPath, librariesToLink, cmdOptions,
                                /*elfSection=*/"", CompilationTarget::Binary,
                                lazyTableBuilder);

    // Serialize the module to binary
    std::optional<SmallVector<char, 0>> serializedModule =
        target.serializeToObject(module, targetOptions);

    if (!serializedModule) {
      return module.emitError("Failed to serialize module to object");
    }

    // Create object attribute
    Attribute object =
        target.createObject(module, *serializedModule, targetOptions);
    if (!object) {
      return module.emitError("Failed to create object attribute");
    }

    objects.push_back(object);
  }

  // Create gpu.binary op
  builder.setInsertionPointAfter(module);
  gpu::BinaryOp::create(builder, module.getLoc(), module.getName(),
                        /*offloadingHandler=*/nullptr,
                        builder.getArrayAttr(objects));

  // Erase the original module
  module->erase();
  return success();
}

void WaterGPUModuleToBinaryPass::runOnOperation() {
  // Walk all regions and blocks looking for GPUModuleOp instances
  for (Region &region : getOperation()->getRegions()) {
    for (Block &block : region.getBlocks()) {
      // Use early_inc_range since we're erasing modules during iteration
      for (auto module :
           llvm::make_early_inc_range(block.getOps<GPUModuleOp>())) {
        if (failed(serializeModule(module))) {
          return signalPassFailure();
        }
      }
    }
  }
}
