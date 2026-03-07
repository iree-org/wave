// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// TranslateFromLLVM: Strict LLVM dialect → WaveASM translation.
//
// Consumes gpu.module { llvm.func @kernel ... } with rocdl intrinsics.
// Fails on any unhandled op — no silent fallthrough.
//===----------------------------------------------------------------------===//

#include "waveasm/Transforms/TranslateFromLLVM.h"
#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Passes.h"
#include "waveasm/Transforms/TranslateFromMLIR.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "waveasm-translate-llvm"

using namespace mlir;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMTRANSLATEFROMLLVM
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

namespace waveasm {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Extract workgroup size from llvm.func attributes.
static std::tuple<int64_t, int64_t, int64_t>
getWorkgroupSize(LLVM::LLVMFuncOp func) {
  // gpu.known_block_size = array<i32: X, Y, Z>
  if (auto attr =
          func->getAttrOfType<DenseI32ArrayAttr>("gpu.known_block_size")) {
    auto vals = attr.asArrayRef();
    int64_t x = vals.size() > 0 ? vals[0] : 64;
    int64_t y = vals.size() > 1 ? vals[1] : 1;
    int64_t z = vals.size() > 2 ? vals[2] : 1;
    return {x, y, z};
  }
  // rocdl.reqd_work_group_size = array<i32: X, Y, Z>
  if (auto attr = func->getAttrOfType<DenseI32ArrayAttr>(
          "rocdl.reqd_work_group_size")) {
    auto vals = attr.asArrayRef();
    int64_t x = vals.size() > 0 ? vals[0] : 64;
    int64_t y = vals.size() > 1 ? vals[1] : 1;
    int64_t z = vals.size() > 2 ? vals[2] : 1;
    return {x, y, z};
  }
  return {64, 1, 1};
}

/// Create a waveasm.program from an llvm.func kernel.
static ProgramOp createProgramFromLLVMFunc(LLVM::LLVMFuncOp func,
                                           OpBuilder &builder,
                                           StringRef targetId) {
  auto *ctx = builder.getContext();
  auto loc = func.getLoc();

  auto targetAttr = TargetAttr::get(ctx, getTargetKindAttr(ctx, targetId), 5);
  auto abiAttr =
      KernelABIAttr::get(ctx, 0, 0, std::nullopt, std::nullopt, std::nullopt);

  auto [wgX, wgY, wgZ] = getWorkgroupSize(func);
  SmallVector<Attribute, 3> sizes = {builder.getI64IntegerAttr(wgX),
                                     builder.getI64IntegerAttr(wgY),
                                     builder.getI64IntegerAttr(wgZ)};
  auto workgroupSizeAttr = builder.getArrayAttr(sizes);

  auto program =
      ProgramOp::create(builder, loc, func.getName(), targetAttr, abiAttr,
                        /*vgprs=*/int64_t{256},
                        /*sgprs=*/int64_t{104},
                        /*workgroup_size=*/workgroupSizeAttr,
                        /*lds_size=*/IntegerAttr{});

  if (program.getBody().empty())
    program.getBody().emplaceBlock();

  return program;
}

//===----------------------------------------------------------------------===//
// Op translation — stub that rejects everything unknown.
//===----------------------------------------------------------------------===//

static LogicalResult translateOp(Operation *op, TranslationContext &ctx) {
  return op->emitOpError("unhandled op in LLVM->WaveASM translation");
}

//===----------------------------------------------------------------------===//
// Core translation logic
//===----------------------------------------------------------------------===//

static LogicalResult translateLLVMModule(ModuleOp module, StringRef targetId) {
  auto target = getTargetKindAttr(module.getContext(), targetId);
  if (!target)
    return module.emitError() << "unknown target: " << targetId;

  // Collect llvm.func kernels inside gpu.module.
  SmallVector<LLVM::LLVMFuncOp> kernels;
  module.walk([&](LLVM::LLVMFuncOp func) {
    if (func->hasAttr("gpu.kernel") || func->hasAttr("rocdl.kernel"))
      kernels.push_back(func);
  });

  if (kernels.empty())
    return module.emitError() << "no llvm.func kernel found in module";

  for (auto func : kernels) {
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());

    auto program = createProgramFromLLVMFunc(func, builder, targetId);
    builder.setInsertionPointToStart(&program.getBodyBlock());
    TranslationContext ctx(builder, program, target);

    // Map llvm.func arguments — all are !llvm.ptr (bare pointers).
    for (auto arg : func.getBody().getArguments()) {
      int64_t argIdx = arg.getArgNumber();
      // Treat each ptr arg as a kernel buffer binding.
      // TODO: set up SRDs from bare pointers.
      ctx.queueSRDSetup(arg, argIdx, /*bufferSize=*/0x7FFFFFFC);
    }

    ctx.emitSRDPrologue();

    // Translate body ops (single block for now).
    for (Operation &op : func.getBody().front()) {
      // Skip the return terminator — we emit s_endpgm instead.
      if (isa<LLVM::ReturnOp>(op))
        continue;
      if (failed(translateOp(&op, ctx)))
        return failure();
    }

    S_ENDPGM::create(builder, func.getLoc());

    size_t numKernelArgs = ctx.getNumKernelArgs();
    program->setAttr(
        "num_kernel_args",
        builder.getI64IntegerAttr(static_cast<int64_t>(numKernelArgs)));

    int64_t ldsSize = ctx.getTotalLDSSize();
    if (ldsSize > 0)
      program->setAttr("lds_size", builder.getI64IntegerAttr(ldsSize));

    // Erase the original function.
    func.erase();
  }

  // Clean up empty gpu.module containers.
  SmallVector<gpu::GPUModuleOp> emptyModules;
  module.walk([&](gpu::GPUModuleOp gpuModule) {
    if (gpuModule.getBody()->getOperations().size() <= 1)
      emptyModules.push_back(gpuModule);
  });
  for (auto m : emptyModules)
    m.erase();

  return success();
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace {

struct WAVEASMTranslateFromLLVMPass
    : impl::WAVEASMTranslateFromLLVMBase<WAVEASMTranslateFromLLVMPass> {
  using WAVEASMTranslateFromLLVMBase::WAVEASMTranslateFromLLVMBase;

  void runOnOperation() override {
    auto module = getOperation();

    // Check if there are any LLVM kernels to translate.
    bool hasLLVMKernels = false;
    module.walk([&](LLVM::LLVMFuncOp func) {
      if (func->hasAttr("gpu.kernel") || func->hasAttr("rocdl.kernel"))
        hasLLVMKernels = true;
    });

    if (!hasLLVMKernels)
      return;

    if (failed(translateLLVMModule(module, targetArch)))
      return signalPassFailure();
  }
};

} // namespace

} // namespace waveasm
