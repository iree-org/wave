// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::water {
#define GEN_PASS_DEF_WATERLOWERMEMORYOPS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

namespace {

/// Pass that lowers high-level memory operations to AMDGPU-specific
/// memory operations (buffer loads/stores, LDS operations, etc.).
class WaterLowerMemoryOpsPass
    : public water::impl::WaterLowerMemoryOpsBase<WaterLowerMemoryOpsPass> {
public:
  void runOnOperation() override {
    // TODO: Implement the pass logic
    // This will involve:
    // 1. Pattern matching on vector.load/store, memref operations
    // 2. Lowering to amdgpu.raw_buffer_load/store
    // 3. Lowering to amdgpu.lds_barrier and related ops
    // 4. Handling address space conversions
  }
};

} // namespace
