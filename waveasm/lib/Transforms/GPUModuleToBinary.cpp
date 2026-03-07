// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// WAVEASMGPUModuleToBinary: compile LLVM dialect gpu.module → gpu.binary.
//
// End-to-end pass that translates LLVM dialect to WaveASM IR, runs the full
// optimization pipeline, emits AMDGCN assembly, assembles + links to HSACO
// via LLVM MC, and embeds the binary as a gpu.binary op.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Transforms/AssemblyEmitter.h"
#include "waveasm/Transforms/Passes.h"
#include "waveasm/Transforms/RegAlloc.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "mlir/Transforms/Passes.h"
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
    auto module = getOperation();

    // Check if there are LLVM kernels to compile.
    bool hasLLVMKernels = false;
    module.walk([&](LLVM::LLVMFuncOp func) {
      if (func->hasAttr("gpu.kernel") || func->hasAttr("rocdl.kernel"))
        hasLLVMKernels = true;
    });

    if (!hasLLVMKernels)
      return;

    // Snapshot gpu.module metadata before the inner pipeline erases them.
    struct GPUModuleInfo {
      StringAttr name;
      Location loc;
      Attribute target; // First target attr, if any.
    };
    SmallVector<GPUModuleInfo> gpuModuleInfos;
    module.walk([&](gpu::GPUModuleOp m) {
      Attribute target;
      if (m.getTargetsAttr() && !m.getTargetsAttr().empty())
        target = m.getTargetsAttr()[0];
      gpuModuleInfos.push_back({m.getNameAttr(), m.getLoc(), target});
    });

    // Step 1: Run the LLVM→WaveASM translation + optimization pipeline.
    PassManager pm(module.getContext());
    pm.addPass(waveasm::createWAVEASMTranslateFromLLVM(
        {/*targetArch=*/targetArch.getValue()}));
    pm.addPass(waveasm::createWAVEASMScopedCSE());
    pm.addPass(waveasm::createWAVEASMPeephole());
    pm.addPass(waveasm::createWAVEASMMemoryOffsetOpt());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(waveasm::createWAVEASMScopedCSE());
    pm.addPass(waveasm::createWAVEASMLinearScan(
        {/*maxVGPRs=*/512, /*maxSGPRs=*/104, /*maxAGPRs=*/512}));
    pm.addPass(waveasm::createWAVEASMInsertWaitcnt(
        {/*insertAfterLoads=*/false, /*ticketedWaitcnt=*/true}));
    pm.addPass(waveasm::createWAVEASMHazardMitigation(
        {/*targetArch=*/targetArch.getValue()}));

    if (failed(pm.run(module)))
      return signalPassFailure();

    // Step 2: Emit assembly for each program.
    waveasm::PhysicalMapping mapping;
    std::string asmText;
    llvm::raw_string_ostream asmStream(asmText);

    module.walk([&](waveasm::ProgramOp program) {
      if (failed(waveasm::writeAssembly(program, mapping, asmStream)))
        signalPassFailure();
    });

    if (asmText.empty()) {
      module.emitError("no assembly generated");
      return signalPassFailure();
    }

    // Step 3: Assemble + link to HSACO.
    ROCDL::SerializeGPUModuleBase::init();

    auto emitError = [&]() -> InFlightDiagnostic {
      return module.emitError();
    };

    std::string cpu = targetArch.getValue();
    FailureOr<SmallVector<char, 0>> objectCode = ROCDL::assembleIsa(
        asmText, kTriple, cpu, /*features=*/"", emitError);
    if (failed(objectCode))
      return signalPassFailure();

    // Resolve lld path.
    SmallString<128> actualLldPath(lldPath.getValue());
    if (actualLldPath.empty() || !llvm::sys::fs::exists(actualLldPath)) {
      actualLldPath = ROCDL::getROCMPath();
      llvm::sys::path::append(actualLldPath, "llvm", "bin", "ld.lld");
    }
    if (!llvm::sys::fs::exists(actualLldPath)) {
      module.emitError()
          << "ld.lld not found (set --lld-path or ROCM_PATH). Tried: "
          << actualLldPath;
      return signalPassFailure();
    }

    FailureOr<SmallVector<char, 0>> hsaco =
        ROCDL::linkObjectCode(*objectCode, actualLldPath, emitError);
    if (failed(hsaco))
      return signalPassFailure();

    // Step 4: Create gpu.binary from saved metadata and erase waveasm.program.
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    StringAttr binaryAttr = builder.getStringAttr(
        StringRef(hsaco->data(), hsaco->size()));

    for (auto &info : gpuModuleInfos) {
      Attribute target = info.target;
      if (!target)
        target = ROCDL::ROCDLTargetAttr::get(
            module.getContext(), /*optLevel=*/2, kTriple, cpu);

      auto objectAttr = builder.getAttr<gpu::ObjectAttr>(
          target, gpu::CompilationTarget::Binary, binaryAttr,
          /*properties=*/DictionaryAttr{}, /*kernels=*/gpu::KernelTableAttr{});

      gpu::BinaryOp::create(builder, info.loc, info.name,
                            /*offloadingHandler=*/nullptr,
                            builder.getArrayAttr({objectAttr}));
    }

    // Clean up waveasm.program ops (they've been serialized into the binary).
    SmallVector<waveasm::ProgramOp> programs;
    module.walk([&](waveasm::ProgramOp p) { programs.push_back(p); });
    for (auto p : programs)
      p->erase();
  }
};

} // namespace
