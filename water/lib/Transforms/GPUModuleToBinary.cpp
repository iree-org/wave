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
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
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
  LogicalResult serializeModule(GPUModuleOp module);

  // Helper methods
  std::unique_ptr<llvm::Module> loadBitcodeFile(llvm::LLVMContext &context,
                                                StringRef path);
  LogicalResult
  linkBitcodeFiles(llvm::Module &mod,
                   SmallVector<std::unique_ptr<llvm::Module>> &&libs);
  FailureOr<llvm::TargetMachine *> createTargetMachine(Attribute targetAttr);
  LogicalResult optimizeModule(llvm::Module &mod,
                               llvm::TargetMachine *targetMachine);
  FailureOr<std::string> compileToISA(llvm::Module &mod,
                                      llvm::TargetMachine &targetMachine);
  FailureOr<SmallVector<char, 0>>
  assembleToObject(StringRef isa, llvm::TargetMachine &targetMachine);
};
} // namespace

LogicalResult WaterGPUModuleToBinaryPass::serializeModule(GPUModuleOp module) {
  OpBuilder builder(module->getContext());

  // Check if module has target attributes
  if (!module.getTargetsAttr() || module.getTargetsAttr().empty())
    return module.emitError("GPU module has no target attributes");

  // Check that there is exactly one target
  if (module.getTargetsAttr().size() != 1)
    return module.emitError(
        "GPU module must have exactly one target attribute");

  // Get the target attribute
  Attribute targetAttr = module.getTargetsAttr()[0];
  if (!targetAttr)
    return module.emitError("Target attribute cannot be null");

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
  FailureOr<llvm::TargetMachine *> targetMachine =
      createTargetMachine(targetAttr);
  if (failed(targetMachine))
    return module.emitError("Failed to create target machine");

  if (failed(optimizeModule(*llvmModule, *targetMachine)))
    return module.emitError("Failed to optimize LLVM IR");

  // Step 4: Compile to ISA
  FailureOr<std::string> isa = compileToISA(*llvmModule, **targetMachine);
  if (failed(isa))
    return module.emitError("Failed to compile to ISA");

  // Step 5: Assemble to binary
  FailureOr<SmallVector<char, 0>> binary =
      assembleToObject(*isa, **targetMachine);
  if (failed(binary))
    return module.emitError("Failed to assemble to binary");

  SmallVector<char, 0> binaryData = std::move(*binary);

  // Create object attribute
  Builder attrBuilder(module.getContext());
  StringAttr binaryAttr = attrBuilder.getStringAttr(
      StringRef(binaryData.data(), binaryData.size()));

  DictionaryAttr properties{};
  gpu::KernelTableAttr kernels;

  Attribute objectAttr = attrBuilder.getAttr<gpu::ObjectAttr>(
      targetAttr, gpu::CompilationTarget::Binary, binaryAttr, properties,
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
    llvm::Module &mod, SmallVector<std::unique_ptr<llvm::Module>> &&libs) {
  if (libs.empty())
    return success();

  llvm::Linker linker(mod);
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

FailureOr<llvm::TargetMachine *>
WaterGPUModuleToBinaryPass::createTargetMachine(Attribute targetAttr) {
  // Check if this is a ROCDL target
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

  // Set optimization level from target attribute
  targetMachine->setOptLevel(
      static_cast<llvm::CodeGenOptLevel>(rocdlTarget.getO()));

  return targetMachine.release();
}

LogicalResult
WaterGPUModuleToBinaryPass::optimizeModule(llvm::Module &mod,
                                           llvm::TargetMachine *targetMachine) {
  // Get optimization level from target machine
  int optLevel = static_cast<int>(targetMachine->getOptLevel());

  auto transformer =
      makeOptimizingTransformer(optLevel, /*sizeLevel=*/0, targetMachine);
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

FailureOr<SmallVector<char, 0>> WaterGPUModuleToBinaryPass::assembleToObject(
    StringRef isa, llvm::TargetMachine &targetMachine) {
  // Step 1: Assemble ISA to object file using MC infrastructure
  llvm::Triple triple = targetMachine.getTargetTriple();
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target)
    return getOperation()->emitError() << "Failed to lookup target: " << error;

  // Set up MC infrastructure
  llvm::SourceMgr srcMgr;
  srcMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(isa),
                            llvm::SMLoc());

  const llvm::MCTargetOptions mcOptions;
  std::unique_ptr<llvm::MCRegisterInfo> mri(target->createMCRegInfo(triple));
  std::unique_ptr<llvm::MCAsmInfo> mai(
      target->createMCAsmInfo(*mri, triple, mcOptions));
  std::unique_ptr<llvm::MCSubtargetInfo> sti(
      target->createMCSubtargetInfo(triple, targetMachine.getTargetCPU(),
                                    targetMachine.getTargetFeatureString()));

  SmallVector<char, 0> objectBuffer;
  llvm::raw_svector_ostream os(objectBuffer);

  llvm::MCContext ctx(triple, mai.get(), mri.get(), sti.get(), &srcMgr,
                      &mcOptions);
  std::unique_ptr<llvm::MCObjectFileInfo> mofi(target->createMCObjectFileInfo(
      ctx, /*PIC=*/false, /*LargeCodeModel=*/false));
  ctx.setObjectFileInfo(mofi.get());

  std::unique_ptr<llvm::MCInstrInfo> mcii(target->createMCInstrInfo());
  llvm::MCCodeEmitter *ce = target->createMCCodeEmitter(*mcii, ctx);
  llvm::MCAsmBackend *mab = target->createMCAsmBackend(*sti, *mri, mcOptions);
  std::unique_ptr<llvm::MCStreamer> mcStreamer(target->createMCObjectStreamer(
      triple, ctx, std::unique_ptr<llvm::MCAsmBackend>(mab),
      mab->createObjectWriter(os), std::unique_ptr<llvm::MCCodeEmitter>(ce),
      *sti));

  std::unique_ptr<llvm::MCAsmParser> parser(
      createMCAsmParser(srcMgr, ctx, *mcStreamer, *mai));
  std::unique_ptr<llvm::MCTargetAsmParser> tap(
      target->createMCAsmParser(*sti, *parser, *mcii, mcOptions));

  if (!tap)
    return getOperation()->emitError("Assembler initialization error");

  parser->setTargetParser(*tap);
  parser->Run(false);

  // Step 2: Link object file to create HSACO
  // Write object to temporary file
  int tempObjFd = -1;
  SmallString<128> tempObjFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel%%", "o", tempObjFd,
                                         tempObjFilename))
    return getOperation()->emitError(
        "Failed to create temporary file for object");

  llvm::FileRemover cleanupObj(tempObjFilename);
  {
    llvm::raw_fd_ostream tempObjOs(tempObjFd, true);
    tempObjOs << StringRef(objectBuffer.data(), objectBuffer.size());
    tempObjOs.flush();
  }

  // Create temporary file for HSACO
  SmallString<128> tempHsacoFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel", "hsaco", tempHsacoFilename))
    return getOperation()->emitError(
        "Failed to create temporary file for HSACO");

  llvm::FileRemover cleanupHsaco(tempHsacoFilename);

  // Link using ld.lld
  SmallString<128> lldPath(toolkitPath);
  llvm::sys::path::append(lldPath, "llvm", "bin", "ld.lld");
  int lldResult = llvm::sys::ExecuteAndWait(
      lldPath, {"ld.lld", "-shared", tempObjFilename, "-o", tempHsacoFilename});
  if (lldResult != 0)
    return getOperation()->emitError("ld.lld invocation failed");

  // Read HSACO file
  auto hsacoFile =
      llvm::MemoryBuffer::getFile(tempHsacoFilename, /*IsText=*/false);
  if (!hsacoFile)
    return getOperation()->emitError(
        "Failed to read HSACO from temporary file");

  StringRef buffer = (*hsacoFile)->getBuffer();
  return SmallVector<char, 0>(buffer.begin(), buffer.end());
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
