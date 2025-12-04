// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "AssembleISA.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
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
  LogicalResult serializeModule(GPUModuleOp mod);

  // Helper methods
  std::unique_ptr<llvm::Module> loadBitcodeFile(llvm::LLVMContext &context,
                                                StringRef path);
  LogicalResult
  linkBitcodeFiles(llvm::Module &mod,
                   SmallVector<std::unique_ptr<llvm::Module>> &&libs);
  FailureOr<std::unique_ptr<llvm::TargetMachine>>
  createTargetMachine(Attribute targetAttr);
  LogicalResult optimizeModule(llvm::Module &mod,
                               llvm::TargetMachine &targetMachine);
  FailureOr<std::string> compileToISA(llvm::Module &mod,
                                      llvm::TargetMachine &targetMachine);

  // Dump helpers
  LogicalResult dumpLLVMModule(llvm::Module &mod, StringRef modName,
                               StringRef suffix);
  LogicalResult dumpText(StringRef text, StringRef modName, StringRef suffix);
  LogicalResult dumpBinary(ArrayRef<char> data, StringRef modName,
                           StringRef suffix);

  // Override helpers
  FailureOr<std::unique_ptr<llvm::Module>>
  tryLoadOverrideLLVM(llvm::LLVMContext &context, StringRef modName,
                      StringRef suffix);
  FailureOr<std::optional<std::string>> tryLoadOverrideText(StringRef modName,
                                                            StringRef suffix);
  FailureOr<std::optional<SmallVector<char, 0>>>
  tryLoadOverrideBinary(StringRef modName, StringRef suffix);
};
} // namespace

LogicalResult WaterGPUModuleToBinaryPass::serializeModule(GPUModuleOp mod) {
  // Check that there is exactly one target.
  if (!mod.getTargetsAttr() || mod.getTargetsAttr().size() != 1)
    return mod.emitError("GPU module must have exactly one target attribute");

  // Get the target attribute.
  Attribute targetAttr = mod.getTargetsAttr()[0];
  if (!targetAttr)
    return mod.emitError("Target attribute cannot be null");

  // Step 1: Translate GPU module to LLVM IR.
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(mod, llvmContext);

  if (!llvmModule)
    return mod.emitError("Failed to translate GPU module to LLVM IR");

  // Create dump directory if specified.
  if (!dumpIntermediates.empty()) {
    std::error_code ec = llvm::sys::fs::create_directories(dumpIntermediates);
    if (ec)
      return mod.emitError()
             << "Failed to create dump directory: " << dumpIntermediates << ": "
             << ec.message();
  }

  auto dumpAndOverrideLLVM = [&](StringRef suffix) -> LogicalResult {
    StringRef modName = mod.getName();
    if (failed(dumpLLVMModule(*llvmModule, modName, suffix)))
      return failure();

    auto overrideLLVM = tryLoadOverrideLLVM(llvmContext, modName, suffix);
    if (failed(overrideLLVM))
      return failure();

    if (*overrideLLVM)
      llvmModule = std::move(*overrideLLVM);

    return success();
  };
  // Dump/override original LLVM IR.
  if (failed(dumpAndOverrideLLVM("_original")))
    return failure();

  // Step 2: Load and link device libraries.
  SmallVector<std::unique_ptr<llvm::Module>> bitcodeLibs;
  for (const std::string &path : linkFiles) {
    auto lib = loadBitcodeFile(llvmContext, path);
    if (!lib)
      return mod.emitError("Failed to load bitcode file: " + path);
    bitcodeLibs.push_back(std::move(lib));
  }

  if (failed(linkBitcodeFiles(*llvmModule, std::move(bitcodeLibs))))
    return mod.emitError("Failed to link bitcode libraries");

  // Dump/override linked LLVM IR.
  if (failed(dumpAndOverrideLLVM("_linked")))
    return failure();

  // Step 3: Create target machine and set data layout.
  FailureOr<std::unique_ptr<llvm::TargetMachine>> targetMachine =
      createTargetMachine(targetAttr);
  if (failed(targetMachine))
    return mod.emitError("Failed to create target machine");

  // Set the data layout and target triple to match the target machine.
  llvmModule->setDataLayout((*targetMachine)->createDataLayout());
  llvmModule->setTargetTriple((*targetMachine)->getTargetTriple());

  // Step 4: Optimize LLVM IR.
  if (failed(optimizeModule(*llvmModule, **targetMachine)))
    return mod.emitError("Failed to optimize LLVM IR");

  // Dump optimized LLVM IR.
  if (failed(dumpAndOverrideLLVM("_optimized")))
    return failure();

  // Step 5: Compile to ISA.
  FailureOr<std::string> isa = compileToISA(*llvmModule, **targetMachine);
  if (failed(isa))
    return mod.emitError("Failed to compile to ISA");

  auto dumpAndOverrideISA = [&](StringRef suffix) -> LogicalResult {
    StringRef modName = mod.getName();
    if (failed(dumpText(*isa, modName, suffix)))
      return failure();

    auto overrideISA = tryLoadOverrideText(modName, suffix);
    if (failed(overrideISA))
      return failure();

    if (*overrideISA)
      isa = std::move(**overrideISA);

    return success();
  };

  // Dump/override ISA.
  if (failed(dumpAndOverrideISA(".s")))
    return failure();

  // Step 6: Assemble to binary.
  // Use ROCM_PATH environment variable if toolkitPath is not provided.
  StringRef actualToolkitPath = toolkitPath;
  if (actualToolkitPath.empty())
    actualToolkitPath = ROCDL::getROCMPath();

  FailureOr<SmallVector<char, 0>> binary =
      water::assembleISAToHSACO(mod, *isa, **targetMachine, actualToolkitPath);
  if (failed(binary))
    return mod.emitError("Failed to assemble to binary");

  SmallVector<char, 0> binaryData = std::move(*binary);

  auto dumpAndOverrideBinary = [&](StringRef suffix) -> LogicalResult {
    StringRef modName = mod.getName();
    if (failed(dumpBinary(binaryData, modName, suffix)))
      return failure();

    auto overrideBinary = tryLoadOverrideBinary(modName, suffix);
    if (failed(overrideBinary))
      return failure();

    if (*overrideBinary)
      binaryData = std::move(**overrideBinary);

    return success();
  };

  // Dump/override HSACO binary.
  if (failed(dumpAndOverrideBinary(".hsaco")))
    return failure();

  // Create object attribute.
  Builder attrBuilder(mod.getContext());
  StringAttr binaryAttr = attrBuilder.getStringAttr(
      StringRef(binaryData.data(), binaryData.size()));

  DictionaryAttr properties{};
  gpu::KernelTableAttr kernels;

  Attribute objectAttr = attrBuilder.getAttr<gpu::ObjectAttr>(
      targetAttr, gpu::CompilationTarget::Binary, binaryAttr, properties,
      kernels);

  // Create gpu.binary op.
  OpBuilder builder(mod.getContext());
  builder.setInsertionPointAfter(mod);
  gpu::BinaryOp::create(builder, mod.getLoc(), mod.getName(),
                        /*offloadingHandler=*/nullptr,
                        builder.getArrayAttr({objectAttr}));

  // Erase the original module.
  mod->erase();
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
    llvm::Module &mod, SmallVector<std::unique_ptr<llvm::Module>> &&libs) {
  if (libs.empty())
    return success();

  llvm::Linker linker(mod);
  for (std::unique_ptr<llvm::Module> &libModule : libs) {
    // Link the library, importing only needed symbols.
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

FailureOr<std::unique_ptr<llvm::TargetMachine>>
WaterGPUModuleToBinaryPass::createTargetMachine(Attribute targetAttr) {
  water::initializeAMDGPUTarget();

  auto rocdlTarget = dyn_cast<ROCDL::ROCDLTargetAttr>(targetAttr);
  if (!rocdlTarget)
    return getOperation()->emitError(
        "Only ROCDL targets are currently supported");

  std::string error;
  llvm::Triple triple(llvm::Triple::normalize(rocdlTarget.getTriple()));
  const llvm::Target *llvmTarget =
      llvm::TargetRegistry::lookupTarget(triple, error);

  if (!llvmTarget)
    return getOperation()->emitError()
           << "Failed to lookup target for triple '" << rocdlTarget.getTriple()
           << "': " << error;

  std::unique_ptr<llvm::TargetMachine> targetMachine(
      llvmTarget->createTargetMachine(triple, rocdlTarget.getChip(),
                                      rocdlTarget.getFeatures(), {}, {}));
  if (!targetMachine)
    return getOperation()->emitError("Failed to create target machine");

  // Set optimization level from target attribute.
  targetMachine->setOptLevel(
      static_cast<llvm::CodeGenOptLevel>(rocdlTarget.getO()));

  return targetMachine;
}

LogicalResult
WaterGPUModuleToBinaryPass::optimizeModule(llvm::Module &mod,
                                           llvm::TargetMachine &targetMachine) {
  // Get optimization level from target machine.
  int optLevel = static_cast<int>(targetMachine.getOptLevel());

  auto transformer =
      makeOptimizingTransformer(optLevel, /*sizeLevel=*/0, &targetMachine);
  auto error = transformer(&mod);
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

FailureOr<std::string>
WaterGPUModuleToBinaryPass::compileToISA(llvm::Module &mod,
                                         llvm::TargetMachine &targetMachine) {
  SmallVector<char, 0> isaBuffer;
  llvm::raw_svector_ostream stream(isaBuffer);

  llvm::legacy::PassManager codegen;
  if (targetMachine.addPassesToEmitFile(codegen, stream, nullptr,
                                        llvm::CodeGenFileType::AssemblyFile))
    return getOperation()->emitError("Target machine cannot emit assembly");

  codegen.run(mod);
  return std::string(isaBuffer.begin(), isaBuffer.end());
}

LogicalResult WaterGPUModuleToBinaryPass::dumpLLVMModule(llvm::Module &mod,
                                                         StringRef modName,
                                                         StringRef suffix) {
  if (dumpIntermediates.empty())
    return success();

  SmallString<128> path(dumpIntermediates);
  llvm::sys::path::append(path, modName + suffix + ".ll");

  std::error_code ec;
  llvm::ToolOutputFile outputFile(path, ec, llvm::sys::fs::OF_None);
  if (ec)
    return getOperation()->emitError()
           << "Failed to open file for dumping: " << path << ": "
           << ec.message();

  mod.print(outputFile.os(), nullptr);
  outputFile.keep();
  return success();
}

LogicalResult WaterGPUModuleToBinaryPass::dumpText(StringRef text,
                                                   StringRef modName,
                                                   StringRef suffix) {
  if (dumpIntermediates.empty())
    return success();

  SmallString<128> path(dumpIntermediates);
  llvm::sys::path::append(path, modName + suffix);

  std::error_code ec;
  llvm::ToolOutputFile outputFile(path, ec, llvm::sys::fs::OF_None);
  if (ec)
    return getOperation()->emitError()
           << "Failed to open file for dumping: " << path << ": "
           << ec.message();

  outputFile.os() << text;
  outputFile.keep();
  return success();
}

LogicalResult WaterGPUModuleToBinaryPass::dumpBinary(ArrayRef<char> data,
                                                     StringRef modName,
                                                     StringRef suffix) {
  if (dumpIntermediates.empty())
    return success();

  SmallString<128> path(dumpIntermediates);
  llvm::sys::path::append(path, modName + suffix);

  std::error_code ec;
  llvm::ToolOutputFile outputFile(path, ec, llvm::sys::fs::OF_None);
  if (ec)
    return getOperation()->emitError()
           << "Failed to open file for dumping: " << path << ": "
           << ec.message();

  outputFile.os().write(data.data(), data.size());
  outputFile.keep();
  return success();
}

FailureOr<std::unique_ptr<llvm::Module>>
WaterGPUModuleToBinaryPass::tryLoadOverrideLLVM(llvm::LLVMContext &context,
                                                StringRef modName,
                                                StringRef suffix) {
  if (overrideIntermediates.empty())
    return std::unique_ptr<llvm::Module>(nullptr);

  SmallString<128> path(overrideIntermediates);
  llvm::sys::path::append(path, modName + suffix + ".ll");

  if (!llvm::sys::fs::exists(path))
    return std::unique_ptr<llvm::Module>(nullptr);

  llvm::SMDiagnostic error;
  std::unique_ptr<llvm::Module> module =
      llvm::parseIRFile(path, error, context);
  if (!module)
    return getOperation()->emitError()
           << "Failed to load override LLVM IR from " << path << ": "
           << error.getMessage();

  return module;
}

FailureOr<std::optional<std::string>>
WaterGPUModuleToBinaryPass::tryLoadOverrideText(StringRef modName,
                                                StringRef suffix) {
  if (overrideIntermediates.empty())
    return std::optional<std::string>(std::nullopt);

  SmallString<128> path(overrideIntermediates);
  llvm::sys::path::append(path, modName + suffix);

  if (!llvm::sys::fs::exists(path))
    return std::optional<std::string>(std::nullopt);

  auto bufferOrError = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrError)
    return getOperation()->emitError()
           << "Failed to load override file from " << path << ": "
           << bufferOrError.getError().message();

  return std::optional<std::string>(bufferOrError.get()->getBuffer().str());
}

FailureOr<std::optional<SmallVector<char, 0>>>
WaterGPUModuleToBinaryPass::tryLoadOverrideBinary(StringRef modName,
                                                  StringRef suffix) {
  if (overrideIntermediates.empty())
    return std::optional<SmallVector<char, 0>>(std::nullopt);

  SmallString<128> path(overrideIntermediates);
  llvm::sys::path::append(path, modName + suffix);

  if (!llvm::sys::fs::exists(path))
    return std::optional<SmallVector<char, 0>>(std::nullopt);

  auto bufferOrError = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrError)
    return getOperation()->emitError()
           << "Failed to load override binary from " << path << ": "
           << bufferOrError.getError().message();

  StringRef data = bufferOrError.get()->getBuffer();
  SmallVector<char, 0> result(data.begin(), data.end());
  return std::optional<SmallVector<char, 0>>(std::move(result));
}

void WaterGPUModuleToBinaryPass::runOnOperation() {
  // Walk all regions and blocks looking for GPUModuleOp instances.
  for (Region &region : getOperation()->getRegions()) {
    for (Block &block : region.getBlocks()) {
      // Use early_inc_range since we're erasing modules during iteration.
      for (auto module :
           llvm::make_early_inc_range(block.getOps<GPUModuleOp>())) {
        if (failed(serializeModule(module)))
          return signalPassFailure();
      }
    }
  }
}
