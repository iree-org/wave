// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// WAVEASMGPUModuleToBinary: emit assembly, assemble, link, gpu.binary.
//
// Final pass in the WaveASM pipeline.  Expects gpu.module ops containing
// waveasm.program ops (already register-allocated and scheduled).  Emits
// AMDGCN assembly, assembles + links to HSACO, and replaces each gpu.module
// with a gpu.binary holding the code object.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Transforms/AssemblyEmitter.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "waveasm-gpu-module-to-binary"

using namespace mlir;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMGPUMODULETOBINARY
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

namespace {

static constexpr StringLiteral kTriple = "amdgcn-amd-amdhsa";

struct WAVEASMGPUModuleToBinaryPass
    : waveasm::impl::WAVEASMGPUModuleToBinaryBase<
          WAVEASMGPUModuleToBinaryPass> {
  using WAVEASMGPUModuleToBinaryBase::WAVEASMGPUModuleToBinaryBase;

  void runOnOperation() override {
    Operation *rootOp = getOperation();

    // Collect gpu.modules that contain waveasm.program ops.
    SmallVector<gpu::GPUModuleOp> gpuModules;
    rootOp->walk<WalkOrder::PreOrder>([&](gpu::GPUModuleOp m) -> WalkResult {
      if (!m.getOps<waveasm::ProgramOp>().empty())
        gpuModules.push_back(m);
      return WalkResult::skip();
    });

    if (gpuModules.empty())
      return;

    // Step 1: Emit assembly for all programs.
    waveasm::PhysicalMapping mapping;
    std::string asmText;
    llvm::raw_string_ostream asmStream(asmText);

    WalkResult walkResult =
        rootOp->walk<WalkOrder::PreOrder>([&](waveasm::ProgramOp program) {
          if (failed(waveasm::writeAssembly(program, mapping, asmStream)))
            return WalkResult::interrupt();
          return WalkResult::advance();
        });
    if (walkResult.wasInterrupted())
      return signalPassFailure();

    if (asmText.empty()) {
      rootOp->emitError("no assembly generated");
      return signalPassFailure();
    }

    // Step 2: Assemble + link to HSACO.
    ROCDL::SerializeGPUModuleBase::init();

    auto emitError = [&]() -> InFlightDiagnostic {
      return rootOp->emitError();
    };

    std::string gpuArch = targetArch.getValue();
    FailureOr<SmallVector<char, 0>> objectCode = ROCDL::assembleIsa(
        asmText, kTriple, gpuArch, /*features=*/"", emitError);
    if (failed(objectCode))
      return signalPassFailure();

    // Resolve lld path.
    SmallString<128> actualLldPath(lldPath.getValue());
    if (actualLldPath.empty() || !llvm::sys::fs::exists(actualLldPath)) {
      actualLldPath = ROCDL::getROCMPath();
      llvm::sys::path::append(actualLldPath, "llvm", "bin", "ld.lld");
    }
    if (!llvm::sys::fs::exists(actualLldPath)) {
      rootOp->emitError()
          << "ld.lld not found (set --lld-path or ROCM_PATH). Tried: "
          << actualLldPath;
      return signalPassFailure();
    }

    FailureOr<SmallVector<char, 0>> hsaco =
        ROCDL::linkObjectCode(*objectCode, actualLldPath, emitError);
    if (failed(hsaco))
      return signalPassFailure();

    // Step 3: Replace each gpu.module with gpu.binary.
    OpBuilder builder(rootOp->getContext());
    StringAttr binaryAttr =
        builder.getStringAttr(StringRef(hsaco->data(), hsaco->size()));

    for (gpu::GPUModuleOp gpuModule : gpuModules) {
      builder.setInsertionPointAfter(gpuModule);

      Attribute target;
      if (gpuModule.getTargetsAttr() && !gpuModule.getTargetsAttr().empty()) {
        target = gpuModule.getTargetsAttr()[0];
        if (gpuModule.getTargetsAttr().size() > 1)
          gpuModule.emitWarning("multiple targets specified, only the first "
                                "is used; remaining targets are ignored");
      }
      if (!target)
        target = ROCDL::ROCDLTargetAttr::get(rootOp->getContext(),
                                             /*optLevel=*/2, kTriple, gpuArch);

      auto objectAttr = builder.getAttr<gpu::ObjectAttr>(
          target, gpu::CompilationTarget::Binary, binaryAttr,
          /*properties=*/DictionaryAttr{}, /*kernels=*/gpu::KernelTableAttr{});

      gpu::BinaryOp::create(builder, gpuModule.getLoc(), gpuModule.getName(),
                            /*offloadingHandler=*/nullptr,
                            builder.getArrayAttr({objectAttr}));
      gpuModule->erase();
    }
  }
};

} // namespace
