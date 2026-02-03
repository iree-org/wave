// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::water {
#define GEN_PASS_DEF_WATERFUSESCALELOADSPASS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

namespace {
struct FuseScaleLoadsPass
    : public water::impl::WaterFuseScaleLoadsPassBase<FuseScaleLoadsPass> {
  using WaterFuseScaleLoadsPassBase::WaterFuseScaleLoadsPassBase;

  void runOnOperation() override {
    // TODO: Implement scale load fusion.
  }
};
} // namespace
