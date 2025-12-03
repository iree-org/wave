// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/Internalize.h"

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

  // Helper methods
  std::unique_ptr<llvm::Module> loadBitcodeFile(llvm::LLVMContext &context,
                                                StringRef path);
  LogicalResult
  linkBitcodeFiles(llvm::Module &module,
                   SmallVector<std::unique_ptr<llvm::Module>> &&libs);
  std::optional<llvm::TargetMachine *>
  createTargetMachine(ROCDL::ROCDLTargetAttr target);
  LogicalResult optimizeModule(llvm::Module &module,
                               llvm::TargetMachine *targetMachine,
                               int optLevel);
};
} // namespace

LogicalResult WaterGPUModuleToBinaryPass::serializeModule(GPUModuleOp module) {
  OpBuilder builder(module->getContext());

  // Check if module has target attributes
  if (!module.getTargetsAttr() || module.getTargetsAttr().empty())
    return module.emitError("GPU module has no target attributes");

  // For now, we only support ROCDL targets
  auto rocdlTarget =
      dyn_cast_or_null<ROCDL::ROCDLTargetAttr>(module.getTargetsAttr()[0]);
  if (!rocdlTarget)
    return module.emitError("Only ROCDL targets are currently supported");

  // Step 1: Translate GPU module to LLVM IR
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(module, llvmContext);

  if (!llvmModule)
    return module.emitError("Failed to translate GPU module to LLVM IR");

  // Step 2: Load and link device libraries
  SmallVector<std::unique_ptr<llvm::Module>> bitcodeLibs;
  for (const std::string &path : linkFiles) {
    auto lib = loadBitcodeFile(llvmContext, path);
    if (!lib)
      return module.emitError("Failed to load bitcode file: " + path);
    bitcodeLibs.push_back(std::move(lib));
  }

  if (failed(linkBitcodeFiles(*llvmModule, std::move(bitcodeLibs))))
    return module.emitError("Failed to link bitcode libraries");

  // Step 3: Optimize LLVM IR
  auto targetMachine = createTargetMachine(rocdlTarget);
  if (!targetMachine)
    return module.emitError("Failed to create target machine");

  if (failed(optimizeModule(*llvmModule, *targetMachine, rocdlTarget.getO())))
    return module.emitError("Failed to optimize LLVM IR");

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

std::unique_ptr<llvm::Module>
WaterGPUModuleToBinaryPass::loadBitcodeFile(llvm::LLVMContext &context,
                                            StringRef path) {
  llvm::SMDiagnostic error;
  std::unique_ptr<llvm::Module> library =
      llvm::getLazyIRFileModule(path, error, context);
  if (!library) {
    getOperation()->emitError() << "Failed loading bitcode file from " << path
                                << ", error: " << error.getMessage();
    return nullptr;
  }
  return library;
}

LogicalResult WaterGPUModuleToBinaryPass::linkBitcodeFiles(
    llvm::Module &module, SmallVector<std::unique_ptr<llvm::Module>> &&libs) {
  if (libs.empty())
    return success();

  llvm::Linker linker(module);
  for (std::unique_ptr<llvm::Module> &libModule : libs) {
    // Link the library, importing only needed symbols
    bool err = linker.linkInModule(
        std::move(libModule), llvm::Linker::Flags::LinkOnlyNeeded,
        [](llvm::Module &m, const StringSet<> &gvs) {
          llvm::internalizeModule(m, [&gvs](const llvm::GlobalValue &gv) {
            return !gv.hasName() || (gvs.count(gv.getName()) == 0);
          });
        });

    if (err) {
      getOperation()->emitError("Failed during bitcode linking");
      return failure();
    }
  }
  return success();
}

std::optional<llvm::TargetMachine *>
WaterGPUModuleToBinaryPass::createTargetMachine(ROCDL::ROCDLTargetAttr target) {
  std::string error;
  llvm::Triple triple(llvm::Triple::normalize(target.getTriple()));
  const llvm::Target *llvmTarget =
      llvm::TargetRegistry::lookupTarget(triple, error);

  if (!llvmTarget) {
    getOperation()->emitError() << "Failed to lookup target for triple '"
                                << target.getTriple() << "': " << error;
    return std::nullopt;
  }

  std::unique_ptr<llvm::TargetMachine> targetMachine(
      llvmTarget->createTargetMachine(triple, target.getChip(),
                                      target.getFeatures(), {}, {}));
  if (!targetMachine)
    return std::nullopt;

  return targetMachine.release();
}

LogicalResult WaterGPUModuleToBinaryPass::optimizeModule(
    llvm::Module &module, llvm::TargetMachine *targetMachine, int optLevel) {
  if (optLevel < 0 || optLevel > 3) {
    getOperation()->emitError() << "Invalid optimization level: " << optLevel;
    return failure();
  }

  targetMachine->setOptLevel(static_cast<llvm::CodeGenOptLevel>(optLevel));

  auto transformer =
      makeOptimizingTransformer(optLevel, /*sizeLevel=*/0, targetMachine);
  auto error = transformer(&module);
  if (error) {
    InFlightDiagnostic mlirError = getOperation()->emitError();
    llvm::handleAllErrors(
        std::move(error), [&mlirError](const llvm::ErrorInfoBase &ei) {
          mlirError << "Failed to optimize LLVM IR: " << ei.message();
        });
    return failure();
  }
  return success();
}

void WaterGPUModuleToBinaryPass::runOnOperation() {
  // Walk all regions and blocks looking for GPUModuleOp instances
  for (Region &region : getOperation()->getRegions()) {
    for (Block &block : region.getBlocks()) {
      // Use early_inc_range since we're erasing modules during iteration
      for (auto module :
           llvm::make_early_inc_range(block.getOps<GPUModuleOp>())) {
        if (failed(serializeModule(module)))
          return signalPassFailure();
      }
    }
  }
}
