// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_TRANSFORMS_PASSES_H
#define WaveASM_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace waveasm {

//===----------------------------------------------------------------------===//
// Pass Creation Functions
//===----------------------------------------------------------------------===//

/// Create the SSA validation pass
std::unique_ptr<mlir::Pass> createWAVEASMValidateSSAPass();

/// Create the liveness analysis pass
std::unique_ptr<mlir::Pass> createWAVEASMLivenessPass();
std::unique_ptr<mlir::Pass> createWAVEASMLivenessPass(bool useCFG);

/// Create the linear scan register allocation pass
std::unique_ptr<mlir::Pass> createWAVEASMLinearScanPass();
std::unique_ptr<mlir::Pass> createWAVEASMLinearScanPass(int64_t maxVGPRs,
                                                        int64_t maxSGPRs);

/// Create the hazard mitigation pass
std::unique_ptr<mlir::Pass> createWAVEASMHazardMitigationPass();
std::unique_ptr<mlir::Pass> createWAVEASMHazardMitigationPass(llvm::StringRef targetArch);

/// Create the waitcnt insertion pass
std::unique_ptr<mlir::Pass> createWAVEASMInsertWaitcntPass();
std::unique_ptr<mlir::Pass> createWAVEASMInsertWaitcntPass(bool insertAfterLoads);

/// Create the assembly emission pass
std::unique_ptr<mlir::Pass> createWAVEASMEmitAssemblyPass();
std::unique_ptr<mlir::Pass> createWAVEASMEmitAssemblyPass(
    llvm::StringRef outputPath);

/// Create the MLIR translation pass
std::unique_ptr<mlir::Pass> createWAVEASMTranslateFromMLIRPass();
std::unique_ptr<mlir::Pass> createWAVEASMTranslateFromMLIRPass(
    llvm::StringRef targetId);

/// Create the scoped CSE pass
std::unique_ptr<mlir::Pass> createWAVEASMScopedCSEPass();

/// Create the peephole optimization pass
std::unique_ptr<mlir::Pass> createWAVEASMPeepholePass();

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Register all WAVEASM passes
void registerWAVEASMPasses();

#define GEN_PASS_DECL
#include "waveasm/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "waveasm/Transforms/Passes.h.inc"

} // namespace waveasm

#endif // WaveASM_TRANSFORMS_PASSES_H
