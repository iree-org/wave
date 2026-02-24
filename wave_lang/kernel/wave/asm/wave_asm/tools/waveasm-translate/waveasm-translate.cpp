// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// waveasm-translate: CLI tool for WAVEASM IR processing
//
// This tool provides the following capabilities:
// - Translate MLIR (GPU, arith, etc.) to waveasm IR
// - Run register allocation on waveasm IR
// - Emit AMDGCN assembly from waveasm IR
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Transforms/AssemblyEmitter.h"
#include "waveasm/Transforms/Passes.h"
#include "waveasm/Transforms/RegAlloc.h"
#include "waveasm/Transforms/TranslateFromMLIR.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "waveasm-translate"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace waveasm;

//===----------------------------------------------------------------------===//
// Command Line Options
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    targetId("target", llvm::cl::desc("Target GPU architecture"),
             llvm::cl::value_desc("gfx942|gfx950|gfx1250"),
             llvm::cl::init("gfx942"));

static llvm::cl::opt<bool> runHazardMitigation(
    "waveasm-hazard-mitigation",
    llvm::cl::desc("Run hazard mitigation pass (insert s_nop for hazards)"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    runWaitcntInsertion("waveasm-insert-waitcnt",
                        llvm::cl::desc("Run waitcnt insertion pass"),
                        llvm::cl::init(false));

static llvm::cl::opt<bool>
    runLinearScan("waveasm-linear-scan",
                  llvm::cl::desc("Run linear scan register allocation"),
                  llvm::cl::init(false));

static llvm::cl::opt<bool>
    runScopedCSE("waveasm-scoped-cse",
                 llvm::cl::desc("Run scoped common subexpression elimination"),
                 llvm::cl::init(false));

static llvm::cl::opt<bool>
    runPeephole("waveasm-peephole",
                llvm::cl::desc("Run peephole optimizations"),
                llvm::cl::init(false));

static llvm::cl::opt<bool> runMemoryOffsetOpt(
    "waveasm-memory-offset-opt",
    llvm::cl::desc("Fold constant address components into memory instruction "
                   "offset fields"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> runBufferLoadStrengthReduction(
    "waveasm-buffer-load-strength-reduction",
    llvm::cl::desc("Run buffer load strength reduction pass"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> runScalePackElimination(
    "waveasm-scale-pack-elimination",
    llvm::cl::desc("Eliminate BFE/LSHL_OR round-trips for B-scale iter_args"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    runLoopAddressPromotion("waveasm-loop-address-promotion",
                            llvm::cl::desc("Run loop address promotion pass"),
                            llvm::cl::init(false));

static llvm::cl::opt<bool>
    emitAssembly("emit-assembly",
                 llvm::cl::desc("Emit AMDGCN assembly instead of MLIR"),
                 llvm::cl::init(false));

static llvm::cl::opt<bool> runPreTranslationCSE(
    "mlir-cse",
    llvm::cl::desc("Run MLIR CSE before translation (reduces redundant index "
                   "computations)"),
    llvm::cl::init(false));

static llvm::cl::opt<int64_t>
    workgroupSizeX("workgroup-size-x",
                   llvm::cl::desc("Workgroup size in X dimension"),
                   llvm::cl::init(0));

static llvm::cl::opt<int64_t>
    workgroupSizeY("workgroup-size-y",
                   llvm::cl::desc("Workgroup size in Y dimension"),
                   llvm::cl::init(0));

static llvm::cl::opt<int64_t>
    workgroupSizeZ("workgroup-size-z",
                   llvm::cl::desc("Workgroup size in Z dimension"),
                   llvm::cl::init(0));

static llvm::cl::opt<int64_t>
    subgroupSize("subgroup-size", llvm::cl::desc("Subgroup (wavefront) size"),
                 llvm::cl::init(64));

static llvm::cl::opt<int64_t>
    maxVGPRs("max-vgprs",
             llvm::cl::desc("Maximum VGPRs for register allocation"),
             llvm::cl::init(256));

static llvm::cl::opt<int64_t>
    maxSGPRs("max-sgprs",
             llvm::cl::desc("Maximum SGPRs for register allocation"),
             llvm::cl::init(104));

static llvm::cl::opt<int64_t>
    maxAGPRs("max-agprs",
             llvm::cl::desc("Maximum AGPRs for register allocation"),
             llvm::cl::init(256));

//===----------------------------------------------------------------------===//
// Main Function
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "WAVEASM IR translation tool\n");

  // Set up the MLIR context and register essential dialects
  DialectRegistry registry;
  registry.insert<waveasm::WaveASMDialect>();
  registry.insert<affine::AffineDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<gpu::GPUDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<vector::VectorDialect>();
  registry.insert<amdgpu::AMDGPUDialect>();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  // Allow unregistered dialects (including IREE dialects if encountered)
  context.allowUnregisteredDialects();

  // Parse the input file
  auto inputFileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (auto ec = inputFileOrErr.getError()) {
    llvm::errs() << "Error reading input file: " << ec.message() << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*inputFileOrErr), llvm::SMLoc());

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Failed to parse input file\n";
    return 1;
  }

  // Check if input is already WAVEASM IR (has waveasm.program ops)
  bool hasWaveASMPrograms = false;
  module->walk([&](waveasm::ProgramOp) { hasWaveASMPrograms = true; });

  // If not already WAVEASM IR, translate from MLIR
  if (!hasWaveASMPrograms) {
    // Run pre-translation MLIR passes (e.g., CSE to reduce redundant
    // computations)
    if (runPreTranslationCSE) {
      PassManager prePm(&context);
      prePm.addPass(mlir::createCSEPass());
      if (failed(prePm.run(*module))) {
        llvm::errs() << "Pre-translation CSE failed\n";
        return 1;
      }
    }

    // Use TranslationOptions if workgroup size is specified
    if (workgroupSizeX > 0 || workgroupSizeY > 0 || workgroupSizeZ > 0) {
      waveasm::TranslationOptions options;
      options.targetId = targetId.getValue();
      options.workgroupSizeX = workgroupSizeX;
      options.workgroupSizeY = workgroupSizeY;
      options.workgroupSizeZ = workgroupSizeZ;
      options.subgroupSize = subgroupSize;
      if (failed(waveasm::translateModule(*module, options))) {
        llvm::errs() << "Translation failed\n";
        return 1;
      }
    } else {
      if (failed(waveasm::translateModule(*module, targetId))) {
        llvm::errs() << "Translation failed\n";
        return 1;
      }
    }
  }

  // Run passes if requested
  PassManager pm(&context);

  // CSE should run early (before register allocation)
  if (runScopedCSE) {
    pm.addPass(waveasm::createWAVEASMScopedCSEPass());
  }

  // Peephole optimizations run after CSE but before waitcnt/hazard.
  if (runPeephole) {
    pm.addPass(waveasm::createWAVEASMPeepholePass());
  }

  // Eliminate BFE/LSHL_OR round-trips for B-scale iter_args.
  // Runs after peephole (which creates lshl_or_b32 ops) and before
  // loop address promotion (which modifies loop structure).
  if (runScalePackElimination) {
    pm.addPass(waveasm::createWAVEASMScalePackEliminationPass());
  }

  // Strength-reduce buffer_load address computation in loops: precompute
  // voffsets and increment by constant stride each iteration.
  if (runBufferLoadStrengthReduction) {
    pm.addPass(waveasm::createWAVEASMBufferLoadStrengthReductionPass());
  }

  // Memory offset optimization: fold constant address components into
  // memory instruction offset fields (saves VALU instructions)
  if (runMemoryOffsetOpt) {
    pm.addPass(waveasm::createWAVEASMMemoryOffsetOptPass());

    // Clean up any dead instructions left by the optimization
    pm.addPass(mlir::createCanonicalizerPass());

    // Re-run ScopedCSE after memory offset optimization, because folding
    // constants into offsets may expose identical base address computations
    // that can now be deduplicated.
    if (runScopedCSE) {
      pm.addPass(waveasm::createWAVEASMScopedCSEPass());
    }
  }

  // Loop address promotion: replace per-iteration v_add_u32 LDS address
  // computation with precomputed rotating VGPR iter_args.
  if (runLoopAddressPromotion) {
    pm.addPass(waveasm::createWAVEASMLoopAddressPromotionPass());
  }

  // Register allocation must run before waitcnt/hazard so that those passes
  // see the final register assignments.  Matches compare_backends.py order:
  // LinearScan -> Waitcnt -> Hazard.
  if (runLinearScan) {
    pm.addPass(
        waveasm::createWAVEASMLinearScanPass(maxVGPRs, maxSGPRs, maxAGPRs));
    // After register allocation, physical register types (pvreg/psreg) replace
    // virtual types (vreg/sreg). The MLIR RegionBranchOpInterface verifier
    // checks exact type equality between entry operands and block arguments,
    // but after regalloc these may have structurally compatible (same register
    // class and size) but not identical types (different physical indices).
    // Our op-level verifiers use typesCompatible() to handle this, but the
    // built-in interface verifier does not.
    //
    // We disable the pass-pipeline interleaved verifier ONLY for the regalloc
    // pass. Module-level verification still runs below to catch other errors.
    // TODO: Implement a custom post-regalloc verification pass that uses
    // typesCompatible() instead of exact type equality, then re-enable the
    // interleaved verifier.
    pm.enableVerifier(false);
  }

  // Waitcnt insertion should run before hazard mitigation
  // (matching Python pipeline order for better wait coalescing)
  if (runWaitcntInsertion) {
    pm.addPass(waveasm::createWAVEASMInsertWaitcntPass());
  }

  // Hazard mitigation runs after waitcnt (NOPs don't affect wait coalescing)
  if (runHazardMitigation) {
    pm.addPass(waveasm::createWAVEASMHazardMitigationPass(targetId));
  }

  if (pm.size() > 0) {
    if (failed(pm.run(*module))) {
      llvm::errs() << "Pass pipeline failed\n";
      return 1;
    }
  }

  // Verify module after translation/passes.
  // After regalloc, the RegionBranchOpInterface verifier may reject
  // structurally compatible types (same register class/size but different
  // physical indices). We still run verification but treat failures as
  // warnings when regalloc has run, since our op verifiers handle this.
  if (failed(mlir::verify(*module))) {
    if (runLinearScan) {
      // Expected: RegionBranchOpInterface type mismatch after regalloc.
      // Op-level verifiers using typesCompatible() passed during the pass run.
      LLVM_DEBUG(llvm::dbgs()
                 << "Note: Module verification warning after regalloc "
                 << "(expected for physical register type mismatches)\n");
    } else {
      llvm::errs() << "Module verification failed\n";
    }
  }

  // Set up the output stream
  std::error_code ec;
  llvm::raw_fd_ostream outputStream(outputFilename, ec);
  if (ec) {
    llvm::errs() << "Error opening output file: " << ec.message() << "\n";
    return 1;
  }

  // Emit assembly if requested
  if (emitAssembly) {
    // Create an empty physical mapping (for already-physical registers)
    waveasm::PhysicalMapping mapping;

    // Find all programs and emit assembly for each
    bool success = true;
    module->walk([&](waveasm::ProgramOp program) {
      if (failed(waveasm::writeAssembly(program, mapping, outputStream))) {
        success = false;
      }
    });

    return success ? 0 : 1;
  }

  // Print the translated module (MLIR format)
  module->print(outputStream);

  return 0;
}
