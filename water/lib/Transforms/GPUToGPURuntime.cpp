// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "water-gpu-to-gpu-runtime"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace mlir::water {
#define GEN_PASS_DEF_WATERGPUTOGPURUNTIME
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

using namespace mlir;
using namespace mlir::water;

namespace {

struct WaterGPUtoGPURuntimePass
    : public water::impl::WaterGPUtoGPURuntimeBase<WaterGPUtoGPURuntimePass> {
  using WaterGPUtoGPURuntimeBase::WaterGPUtoGPURuntimeBase;

  void runOnOperation() override {}
};
} // namespace
