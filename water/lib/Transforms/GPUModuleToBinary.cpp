// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

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

  // For now, we only support ROCDL targets
  auto rocdlTarget =
      dyn_cast_or_null<ROCDL::ROCDLTargetAttr>(module.getTargetsAttr()[0]);
  if (!rocdlTarget) {
    return module.emitError("Only ROCDL targets are currently supported");
  }

  // Step 1: Translate GPU module to LLVM IR
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(module, llvmContext);

  if (!llvmModule) {
    return module.emitError("Failed to translate GPU module to LLVM IR");
  }

  // TODO: Step 2: Link device libraries
  // TODO: Step 3: Optimize LLVM IR
  // TODO: Step 4: Compile to ISA
  // TODO: Step 5: Assemble to binary

  // For now, just create a placeholder binary
  SmallVector<char, 0> binaryData;

  // Create object attribute
  Builder attrBuilder(module.getContext());
  StringAttr binaryAttr = attrBuilder.getStringAttr(
      StringRef(binaryData.data(), binaryData.size()));

  DictionaryAttr properties{};
  gpu::KernelTableAttr kernels;

  Attribute objectAttr = attrBuilder.getAttr<gpu::ObjectAttr>(
      rocdlTarget, gpu::CompilationTarget::Binary, binaryAttr, properties,
      kernels);

  // Create gpu.binary op
  builder.setInsertionPointAfter(module);
  gpu::BinaryOp::create(builder, module.getLoc(), module.getName(),
                        /*offloadingHandler=*/nullptr,
                        builder.getArrayAttr({objectAttr}));

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
