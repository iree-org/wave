// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace mlir::water {
#define GEN_PASS_DEF_WATERWMMAMATRIXREUSEPASS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

namespace {

/// Returns true if the operation is a ROCDL WMMA operation that supports
/// reuseA/reuseB flags (gfx1250 variants).
static bool isWMMAWithReuse(Operation *op) {
  // clang-format off
  return isa<
      // ModsAll_Reuse ops (signA, signB, modC, reuseA, reuseB).
      ROCDL::wmma_f32_16x16x4_f32,
      ROCDL::wmma_f32_16x16x32_bf16,
      ROCDL::wmma_f32_16x16x32_f16,
      ROCDL::wmma_f16_16x16x32_f16,
      ROCDL::wmma_bf16_16x16x32_bf16,
      // ModsAll_Diff op.
      ROCDL::wmma_bf16f32_16x16x32_bf16,
      // ModsC ops (modC, reuseA, reuseB).
      ROCDL::wmma_f32_16x16x64_fp8_fp8,
      ROCDL::wmma_f32_16x16x64_fp8_bf8,
      ROCDL::wmma_f32_16x16x64_bf8_fp8,
      ROCDL::wmma_f32_16x16x64_bf8_bf8,
      ROCDL::wmma_f16_16x16x64_fp8_fp8,
      ROCDL::wmma_f16_16x16x64_fp8_bf8,
      ROCDL::wmma_f16_16x16x64_bf8_fp8,
      ROCDL::wmma_f16_16x16x64_bf8_bf8,
      ROCDL::wmma_f32_16x16x128_fp8_fp8,
      ROCDL::wmma_f32_16x16x128_fp8_bf8,
      ROCDL::wmma_f32_16x16x128_bf8_fp8,
      ROCDL::wmma_f32_16x16x128_bf8_bf8,
      ROCDL::wmma_f16_16x16x128_fp8_fp8,
      ROCDL::wmma_f16_16x16x128_fp8_bf8,
      ROCDL::wmma_f16_16x16x128_bf8_fp8,
      ROCDL::wmma_f16_16x16x128_bf8_bf8,
      // ModsABClamp op.
      ROCDL::wmma_i32_16x16x64_iu8,
      // Scale ops.
      ROCDL::wmma_scale_f32_16x16x128_f8f6f4,
      ROCDL::wmma_scale16_f32_16x16x128_f8f6f4,
      ROCDL::wmma_scale_f32_32x16x128_f4,
      ROCDL::wmma_scale16_f32_32x16x128_f4
      >(op);
  // clang-format on
}

/// Helper to get matrix A operand from any WMMA op with reuse support.
/// Returns the A matrix Value, or nullptr if not found.
static Value getMatrixA(Operation *op) {
  // Use TypeSwitch to handle different WMMA op signatures.
  // ModsAll_Reuse/ModsAll_Diff: operands are (a, b, c).
  // ModsC: operands are (a, b, c).
  // ModsABClamp: operands are (a, b, c).
  // Scale: operands are (a, b, c, scaleA, scaleB).
  // In all cases, A is the first operand.
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<
          ROCDL::wmma_f32_16x16x4_f32, ROCDL::wmma_f32_16x16x32_bf16,
          ROCDL::wmma_f32_16x16x32_f16, ROCDL::wmma_f16_16x16x32_f16,
          ROCDL::wmma_bf16_16x16x32_bf16, ROCDL::wmma_bf16f32_16x16x32_bf16,
          ROCDL::wmma_f32_16x16x64_fp8_fp8, ROCDL::wmma_f32_16x16x64_fp8_bf8,
          ROCDL::wmma_f32_16x16x64_bf8_fp8, ROCDL::wmma_f32_16x16x64_bf8_bf8,
          ROCDL::wmma_f16_16x16x64_fp8_fp8, ROCDL::wmma_f16_16x16x64_fp8_bf8,
          ROCDL::wmma_f16_16x16x64_bf8_fp8, ROCDL::wmma_f16_16x16x64_bf8_bf8,
          ROCDL::wmma_f32_16x16x128_fp8_fp8, ROCDL::wmma_f32_16x16x128_fp8_bf8,
          ROCDL::wmma_f32_16x16x128_bf8_fp8, ROCDL::wmma_f32_16x16x128_bf8_bf8,
          ROCDL::wmma_f16_16x16x128_fp8_fp8, ROCDL::wmma_f16_16x16x128_fp8_bf8,
          ROCDL::wmma_f16_16x16x128_bf8_fp8, ROCDL::wmma_f16_16x16x128_bf8_bf8,
          ROCDL::wmma_i32_16x16x64_iu8, ROCDL::wmma_scale_f32_16x16x128_f8f6f4,
          ROCDL::wmma_scale16_f32_16x16x128_f8f6f4,
          ROCDL::wmma_scale_f32_32x16x128_f4,
          ROCDL::wmma_scale16_f32_32x16x128_f4>(
          [](auto wmmaOp) { return wmmaOp.getA(); })
      .Default([](Operation *) { return Value(); });
}

/// Helper to get matrix B operand from any WMMA op with reuse support.
static Value getMatrixB(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<
          ROCDL::wmma_f32_16x16x4_f32, ROCDL::wmma_f32_16x16x32_bf16,
          ROCDL::wmma_f32_16x16x32_f16, ROCDL::wmma_f16_16x16x32_f16,
          ROCDL::wmma_bf16_16x16x32_bf16, ROCDL::wmma_bf16f32_16x16x32_bf16,
          ROCDL::wmma_f32_16x16x64_fp8_fp8, ROCDL::wmma_f32_16x16x64_fp8_bf8,
          ROCDL::wmma_f32_16x16x64_bf8_fp8, ROCDL::wmma_f32_16x16x64_bf8_bf8,
          ROCDL::wmma_f16_16x16x64_fp8_fp8, ROCDL::wmma_f16_16x16x64_fp8_bf8,
          ROCDL::wmma_f16_16x16x64_bf8_fp8, ROCDL::wmma_f16_16x16x64_bf8_bf8,
          ROCDL::wmma_f32_16x16x128_fp8_fp8, ROCDL::wmma_f32_16x16x128_fp8_bf8,
          ROCDL::wmma_f32_16x16x128_bf8_fp8, ROCDL::wmma_f32_16x16x128_bf8_bf8,
          ROCDL::wmma_f16_16x16x128_fp8_fp8, ROCDL::wmma_f16_16x16x128_fp8_bf8,
          ROCDL::wmma_f16_16x16x128_bf8_fp8, ROCDL::wmma_f16_16x16x128_bf8_bf8,
          ROCDL::wmma_i32_16x16x64_iu8, ROCDL::wmma_scale_f32_16x16x128_f8f6f4,
          ROCDL::wmma_scale16_f32_16x16x128_f8f6f4,
          ROCDL::wmma_scale_f32_32x16x128_f4,
          ROCDL::wmma_scale16_f32_32x16x128_f4>(
          [](auto wmmaOp) { return wmmaOp.getB(); })
      .Default([](Operation *) { return Value(); });
}

/// Sets the reuseA flag on a WMMA operation.
static void setReuseA(Operation *op, bool reuse) {
  llvm::TypeSwitch<Operation *>(op)
      .Case<
          ROCDL::wmma_f32_16x16x4_f32, ROCDL::wmma_f32_16x16x32_bf16,
          ROCDL::wmma_f32_16x16x32_f16, ROCDL::wmma_f16_16x16x32_f16,
          ROCDL::wmma_bf16_16x16x32_bf16, ROCDL::wmma_bf16f32_16x16x32_bf16,
          ROCDL::wmma_f32_16x16x64_fp8_fp8, ROCDL::wmma_f32_16x16x64_fp8_bf8,
          ROCDL::wmma_f32_16x16x64_bf8_fp8, ROCDL::wmma_f32_16x16x64_bf8_bf8,
          ROCDL::wmma_f16_16x16x64_fp8_fp8, ROCDL::wmma_f16_16x16x64_fp8_bf8,
          ROCDL::wmma_f16_16x16x64_bf8_fp8, ROCDL::wmma_f16_16x16x64_bf8_bf8,
          ROCDL::wmma_f32_16x16x128_fp8_fp8, ROCDL::wmma_f32_16x16x128_fp8_bf8,
          ROCDL::wmma_f32_16x16x128_bf8_fp8, ROCDL::wmma_f32_16x16x128_bf8_bf8,
          ROCDL::wmma_f16_16x16x128_fp8_fp8, ROCDL::wmma_f16_16x16x128_fp8_bf8,
          ROCDL::wmma_f16_16x16x128_bf8_fp8, ROCDL::wmma_f16_16x16x128_bf8_bf8,
          ROCDL::wmma_i32_16x16x64_iu8, ROCDL::wmma_scale_f32_16x16x128_f8f6f4,
          ROCDL::wmma_scale16_f32_16x16x128_f8f6f4,
          ROCDL::wmma_scale_f32_32x16x128_f4,
          ROCDL::wmma_scale16_f32_32x16x128_f4>(
          [reuse](auto wmmaOp) { wmmaOp.setReuseA(reuse); });
}

/// Sets the reuseB flag on a WMMA operation.
static void setReuseB(Operation *op, bool reuse) {
  llvm::TypeSwitch<Operation *>(op)
      .Case<
          ROCDL::wmma_f32_16x16x4_f32, ROCDL::wmma_f32_16x16x32_bf16,
          ROCDL::wmma_f32_16x16x32_f16, ROCDL::wmma_f16_16x16x32_f16,
          ROCDL::wmma_bf16_16x16x32_bf16, ROCDL::wmma_bf16f32_16x16x32_bf16,
          ROCDL::wmma_f32_16x16x64_fp8_fp8, ROCDL::wmma_f32_16x16x64_fp8_bf8,
          ROCDL::wmma_f32_16x16x64_bf8_fp8, ROCDL::wmma_f32_16x16x64_bf8_bf8,
          ROCDL::wmma_f16_16x16x64_fp8_fp8, ROCDL::wmma_f16_16x16x64_fp8_bf8,
          ROCDL::wmma_f16_16x16x64_bf8_fp8, ROCDL::wmma_f16_16x16x64_bf8_bf8,
          ROCDL::wmma_f32_16x16x128_fp8_fp8, ROCDL::wmma_f32_16x16x128_fp8_bf8,
          ROCDL::wmma_f32_16x16x128_bf8_fp8, ROCDL::wmma_f32_16x16x128_bf8_bf8,
          ROCDL::wmma_f16_16x16x128_fp8_fp8, ROCDL::wmma_f16_16x16x128_fp8_bf8,
          ROCDL::wmma_f16_16x16x128_bf8_fp8, ROCDL::wmma_f16_16x16x128_bf8_bf8,
          ROCDL::wmma_i32_16x16x64_iu8, ROCDL::wmma_scale_f32_16x16x128_f8f6f4,
          ROCDL::wmma_scale16_f32_16x16x128_f8f6f4,
          ROCDL::wmma_scale_f32_32x16x128_f4,
          ROCDL::wmma_scale16_f32_32x16x128_f4>(
          [reuse](auto wmmaOp) { wmmaOp.setReuseB(reuse); });
}

/// Checks if op1 must execute before op2 due to data dependencies.
static bool hasDependency(Operation *op1, Operation *op2) {
  // Check if any result of op1 is used by op2.
  for (Value result : op1->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (user == op2)
        return true;
    }
  }
  return false;
}

class WMMAMatrixReusePass
    : public water::impl::WaterWMMAMatrixReusePassBase<WMMAMatrixReusePass> {
public:
  void runOnOperation() override {
    Operation *rootOp = getOperation();

    // Process each block independently.
    rootOp->walk([&](Block *block) { processBlock(block); });
  }

private:
  /// Process a single basic block to reorder WMMA operations.
  void processBlock(Block *block) {
    // Collect all WMMA operations with reuse support in this block.
    SmallVector<Operation *> wmmaOps;
    for (Operation &op : *block) {
      if (isWMMAWithReuse(&op))
        wmmaOps.push_back(&op);
    }

    if (wmmaOps.size() < 2)
      return;

    // Build dependency graph between WMMA operations.
    llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>> successors;
    llvm::DenseMap<Operation *, unsigned> inDegree;
    for (Operation *op : wmmaOps) {
      successors[op] = {};
      inDegree[op] = 0;
    }

    for (size_t i = 0; i < wmmaOps.size(); ++i) {
      for (size_t j = i + 1; j < wmmaOps.size(); ++j) {
        if (hasDependency(wmmaOps[i], wmmaOps[j])) {
          successors[wmmaOps[i]].push_back(wmmaOps[j]);
          inDegree[wmmaOps[j]]++;
        }
      }
    }

    // Group operations by their A and B operands.
    llvm::DenseMap<Value, SmallVector<Operation *>> opsByMatrixA;
    llvm::DenseMap<Value, SmallVector<Operation *>> opsByMatrixB;
    for (Operation *op : wmmaOps) {
      if (Value a = getMatrixA(op))
        opsByMatrixA[a].push_back(op);
      if (Value b = getMatrixB(op))
        opsByMatrixB[b].push_back(op);
    }

    // TODO: Implement reordering algorithm using topological sort with
    // preference for grouping operations that share A or B operands.
    // For now, just set reuse flags based on current ordering.
    setReuseFlagsForCurrentOrder(wmmaOps);
  }

  /// Sets reuse flags based on the current operation ordering.
  void setReuseFlagsForCurrentOrder(ArrayRef<Operation *> wmmaOps) {
    for (size_t i = 0; i < wmmaOps.size(); ++i) {
      Operation *curr = wmmaOps[i];
      bool reuseA = false;
      bool reuseB = false;

      if (i > 0) {
        Operation *prev = wmmaOps[i - 1];
        Value currA = getMatrixA(curr);
        Value prevA = getMatrixA(prev);
        Value currB = getMatrixB(curr);
        Value prevB = getMatrixB(prev);

        if (currA && prevA && currA == prevA)
          reuseA = true;
        if (currB && prevB && currB == prevB)
          reuseB = true;
      }

      setReuseA(curr, reuseA);
      setReuseB(curr, reuseB);
    }
  }
};

} // namespace
