// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AssembleISA.h"

#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "llvm/Support/TargetSelect.h"
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
  auto emitError = [&]() -> InFlightDiagnostic { return op->emitError(); };

  FailureOr<SmallVector<char, 0>> binary = ROCDL::assembleIsa(
      isa, targetMachine.getTargetTriple().str(), targetMachine.getTargetCPU(),
      targetMachine.getTargetFeatureString(), emitError);
  if (failed(binary))
    return failure();

  return ROCDL::linkObjectCode(*binary, toolkitPath, emitError);
}

} // namespace mlir::water
