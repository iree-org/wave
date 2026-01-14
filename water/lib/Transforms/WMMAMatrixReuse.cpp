// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace mlir::water {
#define GEN_PASS_DEF_WATERWMMAMATRIXREUSEPASS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

namespace {

// clang-format off
/// List of all ROCDL WMMA operations that support reuseA/reuseB flags.
#define ROCDL_WMMA_OPS_WITH_REUSE                                              \
    ROCDL::wmma_f32_16x16x4_f32,                                               \
    ROCDL::wmma_f32_16x16x32_bf16,                                             \
    ROCDL::wmma_f32_16x16x32_f16,                                              \
    ROCDL::wmma_f16_16x16x32_f16,                                              \
    ROCDL::wmma_bf16_16x16x32_bf16,                                            \
    ROCDL::wmma_bf16f32_16x16x32_bf16,                                         \
    ROCDL::wmma_f32_16x16x64_fp8_fp8,                                          \
    ROCDL::wmma_f32_16x16x64_fp8_bf8,                                          \
    ROCDL::wmma_f32_16x16x64_bf8_fp8,                                          \
    ROCDL::wmma_f32_16x16x64_bf8_bf8,                                          \
    ROCDL::wmma_f16_16x16x64_fp8_fp8,                                          \
    ROCDL::wmma_f16_16x16x64_fp8_bf8,                                          \
    ROCDL::wmma_f16_16x16x64_bf8_fp8,                                          \
    ROCDL::wmma_f16_16x16x64_bf8_bf8,                                          \
    ROCDL::wmma_f32_16x16x128_fp8_fp8,                                         \
    ROCDL::wmma_f32_16x16x128_fp8_bf8,                                         \
    ROCDL::wmma_f32_16x16x128_bf8_fp8,                                         \
    ROCDL::wmma_f32_16x16x128_bf8_bf8,                                         \
    ROCDL::wmma_f16_16x16x128_fp8_fp8,                                         \
    ROCDL::wmma_f16_16x16x128_fp8_bf8,                                         \
    ROCDL::wmma_f16_16x16x128_bf8_fp8,                                         \
    ROCDL::wmma_f16_16x16x128_bf8_bf8,                                         \
    ROCDL::wmma_i32_16x16x64_iu8,                                              \
    ROCDL::wmma_scale_f32_16x16x128_f8f6f4,                                    \
    ROCDL::wmma_scale16_f32_16x16x128_f8f6f4,                                  \
    ROCDL::wmma_scale_f32_32x16x128_f4,                                        \
    ROCDL::wmma_scale16_f32_32x16x128_f4
// clang-format on

/// Returns true if the operation is a ROCDL WMMA operation that supports
/// reuseA/reuseB flags (gfx1250 variants).
static bool isWMMAWithReuse(Operation *op) {
  return isa<ROCDL_WMMA_OPS_WITH_REUSE>(op);
}

/// Helper to get matrix A operand from any WMMA op with reuse support.
static Value getMatrixA(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<ROCDL_WMMA_OPS_WITH_REUSE>(
          [](auto wmmaOp) { return wmmaOp.getA(); })
      .Default([](Operation *) { return Value(); });
}

/// Helper to get matrix B operand from any WMMA op with reuse support.
static Value getMatrixB(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<ROCDL_WMMA_OPS_WITH_REUSE>(
          [](auto wmmaOp) { return wmmaOp.getB(); })
      .Default([](Operation *) { return Value(); });
}

/// Helper to get matrix C (accumulator) operand from any WMMA op.
static Value getMatrixC(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<ROCDL_WMMA_OPS_WITH_REUSE>(
          [](auto wmmaOp) { return wmmaOp.getC(); })
      .Default([](Operation *) { return Value(); });
}

/// Sets the reuseA flag on a WMMA operation.
static void setReuseA(Operation *op, bool reuse) {
  llvm::TypeSwitch<Operation *>(op).Case<ROCDL_WMMA_OPS_WITH_REUSE>(
      [reuse](auto wmmaOp) { wmmaOp.setReuseA(reuse); });
}

/// Sets the reuseB flag on a WMMA operation.
static void setReuseB(Operation *op, bool reuse) {
  llvm::TypeSwitch<Operation *>(op).Case<ROCDL_WMMA_OPS_WITH_REUSE>(
      [reuse](auto wmmaOp) { wmmaOp.setReuseB(reuse); });
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
  /// Process a single basic block to find consecutive WMMA op sequences.
  void processBlock(Block *block) {
    SmallVector<Operation *> currentSequence;

    for (Operation &op : *block) {
      if (isWMMAWithReuse(&op)) {
        currentSequence.push_back(&op);
      } else if (!currentSequence.empty()) {
        // Non-WMMA op encountered, process the current sequence.
        processConsecutiveWMMAOps(currentSequence);
        currentSequence.clear();
      }
    }

    // Process any remaining sequence at the end of the block.
    if (!currentSequence.empty())
      processConsecutiveWMMAOps(currentSequence);
  }

  /// Check if candidate can be scheduled given already scheduled ops.
  /// Returns true if all operands of candidate that are defined by ops
  /// in the sequence have already been scheduled.
  static bool
  canSchedule(Operation *candidate,
              const llvm::SmallSetVector<Operation *, 16> &scheduled,
              const llvm::SmallDenseSet<Operation *> &opsInSequence) {
    for (Value operand : candidate->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (defOp && opsInSequence.contains(defOp) && !scheduled.contains(defOp))
        return false;
    }
    return true;
  }

  /// Process a sequence of consecutive WMMA operations.
  static void processConsecutiveWMMAOps(MutableArrayRef<Operation *> wmmaOps) {
    if (wmmaOps.size() < 2)
      return;

    llvm::SmallDenseSet<Operation *> opsInSequence(wmmaOps.begin(),
                                                   wmmaOps.end());
    llvm::SmallSetVector<Operation *, 16> scheduled;

    // First op stays in place.
    scheduled.insert(wmmaOps[0]);

    // Greedily select ops that maximize reuse while respecting dependencies.
    for ([[maybe_unused]] auto i : llvm::seq<size_t>(1, wmmaOps.size())) {
      Operation *prev = scheduled.back();
      Value prevA = getMatrixA(prev);
      Value prevB = getMatrixB(prev);

      Operation *bestCandidate = nullptr;
      int bestScore = -1;

      for (Operation *candidate : wmmaOps) {
        if (scheduled.contains(candidate))
          continue;
        if (!canSchedule(candidate, scheduled, opsInSequence))
          continue;

        // Score: +2 for matching A, +2 for matching B, +1 for chaining C.
        int score = 0;
        if (getMatrixA(candidate) == prevA)
          score += 2;
        if (getMatrixB(candidate) == prevB)
          score += 2;
        // Smaller bonus if candidate uses prev's result as its accumulator.
        if (getMatrixC(candidate) == prev->getResult(0))
          score += 1;

        if (score > bestScore) {
          bestScore = score;
          bestCandidate = candidate;
        }
      }

      // Fallback if no best found (pick first available).
      if (!bestCandidate) {
        for (Operation *candidate : wmmaOps) {
          if (!scheduled.contains(candidate) &&
              canSchedule(candidate, scheduled, opsInSequence)) {
            bestCandidate = candidate;
            break;
          }
        }
      }

      scheduled.insert(bestCandidate);
    }

    // Reorder operations in the IR.
    Operation *insertPoint = wmmaOps[0];
    for (Operation *op : scheduled) {
      if (op != insertPoint)
        op->moveAfter(insertPoint);
      insertPoint = op;
    }

    // Set reuse flags based on new order.
    setReuseFlagsForOrder(scheduled.getArrayRef());

    // Insert sched.barrier ops to prevent LLVM from reordering.
    insertSchedBarriers(scheduled.getArrayRef());
  }

  /// Insert rocdl.sched.barrier ops before, between, and after WMMA ops.
  static void insertSchedBarriers(ArrayRef<Operation *> wmmaOps) {
    if (wmmaOps.empty())
      return;

    OpBuilder builder(wmmaOps.front()->getContext());

    // Insert barrier before the first op.
    builder.setInsertionPoint(wmmaOps.front());
    ROCDL::SchedBarrier::create(builder, wmmaOps.front()->getLoc(),
                                /*mask=*/0);

    // Insert barrier after each op (covers between ops and after last).
    for (Operation *op : wmmaOps) {
      builder.setInsertionPointAfter(op);
      ROCDL::SchedBarrier::create(builder, op->getLoc(), /*mask=*/0);
    }
  }

  /// Sets reuse flags based on the operation ordering.
  static void setReuseFlagsForOrder(ArrayRef<Operation *> wmmaOps) {
    for (auto i : llvm::seq<size_t>(1, wmmaOps.size())) {
      Operation *curr = wmmaOps[i];
      Operation *prev = wmmaOps[i - 1];

      if (getMatrixA(curr) == getMatrixA(prev))
        setReuseA(curr, true);

      if (getMatrixB(curr) == getMatrixB(prev))
        setReuseB(curr, true);
    }
  }
};

} // namespace
