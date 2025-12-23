// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AssembleISA.h"

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
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

using namespace mlir;

namespace mlir::water {

void initializeAMDGPUTarget() {
  static bool initialized = []() {
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
    return true;
  }();
  (void)initialized;
}

FailureOr<SmallVector<char, 0>>
assembleISAToHSACO(Operation *op, StringRef isa,
                   llvm::TargetMachine &targetMachine, StringRef toolkitPath) {
  initializeAMDGPUTarget();

  // Step 1: Assemble ISA to object file using MC infrastructure.
  llvm::Triple triple = targetMachine.getTargetTriple();
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target)
    return op->emitError() << "Failed to lookup target: " << error;

  // Set up MC infrastructure.
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
    return op->emitError("Assembler initialization error");

  parser->setTargetParser(*tap);
  if (parser->Run(false))
    return op->emitError("Assembly parsing failed");

  // Step 2: Link object file to create HSACO.
  // Write object to temporary file.
  int tempObjFd = -1;
  SmallString<128> tempObjFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel%%", "o", tempObjFd,
                                         tempObjFilename))
    return op->emitError("Failed to create temporary file for object");

  llvm::FileRemover cleanupObj(tempObjFilename);
  {
    llvm::raw_fd_ostream tempObjOs(tempObjFd, true);
    tempObjOs << StringRef(objectBuffer.data(), objectBuffer.size());
    tempObjOs.flush();
  }

  // Create temporary file for HSACO.
  SmallString<128> tempHsacoFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel", "hsaco", tempHsacoFilename))
    return op->emitError("Failed to create temporary file for HSACO");

  llvm::FileRemover cleanupHsaco(tempHsacoFilename);

  // Link using ld.lld.
  SmallString<128> lldPath(toolkitPath);
  llvm::sys::path::append(lldPath, "llvm", "bin", "ld.lld");
  int lldResult = llvm::sys::ExecuteAndWait(
      lldPath, {"ld.lld", "-shared", tempObjFilename, "-o", tempHsacoFilename});
  if (lldResult != 0)
    return op->emitError("ld.lld invocation failed");

  // Read HSACO file.
  auto hsacoFile =
      llvm::MemoryBuffer::getFile(tempHsacoFilename, /*IsText=*/false);
  if (!hsacoFile)
    return op->emitError("Failed to read HSACO from temporary file");

  StringRef buffer = (*hsacoFile)->getBuffer();
  return SmallVector<char, 0>(buffer.begin(), buffer.end());
}

} // namespace mlir::water
