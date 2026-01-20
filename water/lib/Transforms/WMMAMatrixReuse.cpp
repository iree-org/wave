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
/// WMMA ops that support reuseA/reuseB flags (gfx1250).
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

/// WMMA ops without reuse flags (gfx11, gfx12).
#define ROCDL_WMMA_OPS_NO_REUSE                                                \
    ROCDL::wmma_f32_16x16x16_f16,                                              \
    ROCDL::wmma_f32_16x16x16_bf16,                                             \
    ROCDL::wmma_f16_16x16x16_f16,                                              \
    ROCDL::wmma_bf16_16x16x16_bf16,                                            \
    ROCDL::wmma_i32_16x16x16_iu8,                                              \
    ROCDL::wmma_i32_16x16x16_iu4,                                              \
    ROCDL::wmma_f32_16x16x16_fp8_fp8,                                          \
    ROCDL::wmma_f32_16x16x16_fp8_bf8,                                          \
    ROCDL::wmma_f32_16x16x16_bf8_bf8,                                          \
    ROCDL::wmma_f32_16x16x16_bf8_fp8,                                          \
    ROCDL::wmma_i32_16x16x32_iu4

/// MFMA ops (CDNA architectures).
#define ROCDL_MFMA_OPS                                                         \
    ROCDL::mfma_f32_32x32x1f32,                                                \
    ROCDL::mfma_f32_16x16x1f32,                                                \
    ROCDL::mfma_f32_4x4x1f32,                                                  \
    ROCDL::mfma_f32_32x32x2f32,                                                \
    ROCDL::mfma_f32_16x16x4f32,                                                \
    ROCDL::mfma_f32_32x32x4f16,                                                \
    ROCDL::mfma_f32_16x16x4f16,                                                \
    ROCDL::mfma_f32_4x4x4f16,                                                  \
    ROCDL::mfma_f32_32x32x8f16,                                                \
    ROCDL::mfma_f32_16x16x16f16,                                               \
    ROCDL::mfma_i32_32x32x4i8,                                                 \
    ROCDL::mfma_i32_16x16x4i8,                                                 \
    ROCDL::mfma_i32_4x4x4i8,                                                   \
    ROCDL::mfma_i32_32x32x8i8,                                                 \
    ROCDL::mfma_i32_16x16x16i8,                                                \
    ROCDL::mfma_f32_32x32x2bf16,                                               \
    ROCDL::mfma_f32_16x16x2bf16,                                               \
    ROCDL::mfma_f32_4x4x2bf16,                                                 \
    ROCDL::mfma_f32_32x32x4bf16,                                               \
    ROCDL::mfma_f32_16x16x8bf16,                                               \
    ROCDL::mfma_f32_32x32x4bf16_1k,                                            \
    ROCDL::mfma_f32_16x16x4bf16_1k,                                            \
    ROCDL::mfma_f32_4x4x4bf16_1k,                                              \
    ROCDL::mfma_f32_32x32x8bf16_1k,                                            \
    ROCDL::mfma_f32_16x16x16bf16_1k,                                           \
    ROCDL::mfma_f64_16x16x4f64,                                                \
    ROCDL::mfma_f64_4x4x4f64,                                                  \
    ROCDL::mfma_i32_16x16x32_i8,                                               \
    ROCDL::mfma_i32_32x32x16_i8,                                               \
    ROCDL::mfma_f32_16x16x8_xf32,                                              \
    ROCDL::mfma_f32_32x32x4_xf32,                                              \
    ROCDL::mfma_f32_16x16x32_bf8_bf8,                                          \
    ROCDL::mfma_f32_16x16x32_bf8_fp8,                                          \
    ROCDL::mfma_f32_16x16x32_fp8_bf8,                                          \
    ROCDL::mfma_f32_16x16x32_fp8_fp8,                                          \
    ROCDL::mfma_f32_32x32x16_bf8_bf8,                                          \
    ROCDL::mfma_f32_32x32x16_bf8_fp8,                                          \
    ROCDL::mfma_f32_32x32x16_fp8_bf8,                                          \
    ROCDL::mfma_f32_32x32x16_fp8_fp8,                                          \
    ROCDL::mfma_f32_16x16x32_bf16,                                             \
    ROCDL::mfma_i32_16x16x64_i8,                                               \
    ROCDL::mfma_f32_16x16x32_f16,                                              \
    ROCDL::mfma_f32_32x32x16_bf16,                                             \
    ROCDL::mfma_i32_32x32x32_i8,                                               \
    ROCDL::mfma_f32_32x32x16_f16,                                              \
    ROCDL::mfma_scale_f32_16x16x128_f8f6f4,                                    \
    ROCDL::mfma_scale_f32_32x32x64_f8f6f4

/// SMFMAC (sparse MFMA) ops.
#define ROCDL_SMFMAC_OPS                                                       \
    ROCDL::smfmac_f32_16x16x32_f16,                                            \
    ROCDL::smfmac_f32_32x32x16_f16,                                            \
    ROCDL::smfmac_f32_16x16x32_bf16,                                           \
    ROCDL::smfmac_f32_32x32x16_bf16,                                           \
    ROCDL::smfmac_i32_16x16x64_i8,                                             \
    ROCDL::smfmac_i32_32x32x32_i8,                                             \
    ROCDL::smfmac_f32_16x16x64_bf8_bf8,                                        \
    ROCDL::smfmac_f32_16x16x64_bf8_fp8,                                        \
    ROCDL::smfmac_f32_16x16x64_fp8_bf8,                                        \
    ROCDL::smfmac_f32_16x16x64_fp8_fp8,                                        \
    ROCDL::smfmac_f32_32x32x32_bf8_bf8,                                        \
    ROCDL::smfmac_f32_32x32x32_bf8_fp8,                                        \
    ROCDL::smfmac_f32_32x32x32_fp8_bf8,                                        \
    ROCDL::smfmac_f32_32x32x32_fp8_fp8,                                        \
    ROCDL::smfmac_f32_16x16x64_bf16,                                           \
    ROCDL::smfmac_f32_16x16x64_f16,                                            \
    ROCDL::smfmac_i32_16x16x128_i8,                                            \
    ROCDL::smfmac_f32_16x16x128_bf8_bf8,                                       \
    ROCDL::smfmac_f32_16x16x128_bf8_fp8,                                       \
    ROCDL::smfmac_f32_16x16x128_fp8_bf8,                                       \
    ROCDL::smfmac_f32_16x16x128_fp8_fp8,                                       \
    ROCDL::smfmac_f32_32x32x32_bf16,                                           \
    ROCDL::smfmac_f32_32x32x32_f16,                                            \
    ROCDL::smfmac_i32_32x32x64_i8,                                             \
    ROCDL::smfmac_f32_32x32x64_bf8_bf8,                                        \
    ROCDL::smfmac_f32_32x32x64_bf8_fp8,                                        \
    ROCDL::smfmac_f32_32x32x64_fp8_bf8,                                        \
    ROCDL::smfmac_f32_32x32x64_fp8_fp8

/// All WMMA ops (with and without reuse).
#define ROCDL_ALL_WMMA_OPS                                                     \
    ROCDL_WMMA_OPS_WITH_REUSE,                                                 \
    ROCDL_WMMA_OPS_NO_REUSE

/// All matrix multiply ops (WMMA, MFMA, SMFMAC).
#define ROCDL_ALL_MATMUL_OPS                                                   \
    ROCDL_ALL_WMMA_OPS,                                                        \
    ROCDL_MFMA_OPS,                                                            \
    ROCDL_SMFMAC_OPS
// clang-format on

/// Returns true if the operation is a matrix multiply op (WMMA/MFMA/SMFMAC).
static bool isMatrixMultiplyOp(Operation *op) {
  return isa<ROCDL_ALL_MATMUL_OPS>(op);
}

/// Returns true if the operation supports reuseA/reuseB flags.
static bool hasReuseFlags(Operation *op) {
  return isa<ROCDL_WMMA_OPS_WITH_REUSE>(op);
}

/// Helper to get matrix A operand from any matrix multiply op.
static Value getMatrixA(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      // WMMA ops have named operands.
      .Case<ROCDL_ALL_WMMA_OPS>([](auto wmmaOp) { return wmmaOp.getA(); })
      // MFMA/SMFMAC ops use variadic args: A is at index 0.
      .Case<ROCDL_MFMA_OPS, ROCDL_SMFMAC_OPS>(
          [](auto mfmaOp) { return mfmaOp.getArgs()[0]; })
      .Default([](Operation *) { return Value(); });
}

/// Helper to get matrix B operand from any matrix multiply op.
static Value getMatrixB(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      // WMMA ops have named operands.
      .Case<ROCDL_ALL_WMMA_OPS>([](auto wmmaOp) { return wmmaOp.getB(); })
      // MFMA/SMFMAC ops use variadic args: B is at index 1.
      .Case<ROCDL_MFMA_OPS, ROCDL_SMFMAC_OPS>(
          [](auto mfmaOp) { return mfmaOp.getArgs()[1]; })
      .Default([](Operation *) { return Value(); });
}

/// Helper to get matrix C (accumulator) operand from any matrix multiply op.
static Value getMatrixC(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      // WMMA ops have named operands.
      .Case<ROCDL_ALL_WMMA_OPS>([](auto wmmaOp) { return wmmaOp.getC(); })
      // MFMA/SMFMAC ops use variadic args: C is at index 2.
      .Case<ROCDL_MFMA_OPS, ROCDL_SMFMAC_OPS>(
          [](auto mfmaOp) { return mfmaOp.getArgs()[2]; })
      .Default([](Operation *) { return Value(); });
}

/// Sets the reuseA flag on a WMMA operation (only for ops that support it).
static void setReuseA(Operation *op, bool reuse) {
  llvm::TypeSwitch<Operation *>(op).Case<ROCDL_WMMA_OPS_WITH_REUSE>(
      [reuse](auto wmmaOp) { wmmaOp.setReuseA(reuse); });
}

/// Sets the reuseB flag on a WMMA operation (only for ops that support it).
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
  /// Process a single basic block to find consecutive matrix multiply op
  /// sequences.
  void processBlock(Block *block) {
    SmallVector<Operation *> currentSequence;
    llvm::SmallDenseSet<Value> producedValues;

    for (Operation &op : llvm::make_early_inc_range(*block)) {
      if (isMatrixMultiplyOp(&op)) {
        for (Value result : op.getResults())
          producedValues.insert(result);

        currentSequence.push_back(&op);
      } else if (isPure(&op) && !llvm::any_of(op.getOperands(), [&](Value arg) {
                   return producedValues.contains(arg);
                 })) {
        // Allow mmas to move around the pure ops.
        continue;
      } else if (!currentSequence.empty()) {
        // Non-matmul op encountered, process the current sequence.
        processConsecutiveMatmulOps(currentSequence);
        producedValues.clear();
        currentSequence.clear();
      }
    }

    // Process any remaining sequence at the end of the block.
    if (!currentSequence.empty())
      processConsecutiveMatmulOps(currentSequence);
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

  /// Process a sequence of consecutive matrix multiply operations.
  static void processConsecutiveMatmulOps(MutableArrayRef<Operation *> ops) {
    if (ops.size() < 2)
      return;

    llvm::SmallDenseSet<Operation *> opsInSequence(ops.begin(), ops.end());
    llvm::SmallSetVector<Operation *, 16> scheduled;

    // First op stays in place.
    scheduled.insert(ops[0]);

    // Greedily select ops that maximize reuse while respecting dependencies.
    for ([[maybe_unused]] auto i : llvm::seq<size_t>(1, ops.size())) {
      Operation *prev = scheduled.back();
      Value prevA = getMatrixA(prev);
      Value prevB = getMatrixB(prev);

      Operation *bestCandidate = nullptr;
      int bestScore = -1;

      for (Operation *candidate : ops) {
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
        for (Operation *candidate : ops) {
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
    Operation *insertPoint = ops[0];
    insertPoint->moveAfter(ops.back());
    for (Operation *op : scheduled) {
      if (op != insertPoint)
        op->moveAfter(insertPoint);
      insertPoint = op;
    }

    // Set reuse flags based on new order (only for ops that support it).
    setReuseFlagsForOrder(scheduled.getArrayRef());

    // Insert sched.barrier ops to prevent LLVM from reordering.
    insertSchedBarriers(scheduled.getArrayRef());
  }

  /// Insert rocdl.sched.barrier ops before, between, and after matmul ops.
  static void insertSchedBarriers(ArrayRef<Operation *> ops) {
    if (ops.empty())
      return;

    OpBuilder builder(ops.front()->getContext());

    // Insert barrier before the first op.
    builder.setInsertionPoint(ops.front());
    ROCDL::SchedBarrier::create(builder, ops.front()->getLoc(),
                                /*mask=*/0);

    // Insert barrier after each op (covers between ops and after last).
    for (Operation *op : ops) {
      builder.setInsertionPointAfter(op);
      ROCDL::SchedBarrier::create(builder, op->getLoc(), /*mask=*/0);
    }
  }

  /// Sets reuse flags based on the operation ordering.
  /// Only applies to ops that support reuse flags (gfx1250 WMMA variants).
  static void setReuseFlagsForOrder(ArrayRef<Operation *> ops) {
    for (auto i : llvm::seq<size_t>(1, ops.size())) {
      Operation *curr = ops[i];
      Operation *prev = ops[i - 1];

      // Only set reuse flags on ops that support them.
      if (!hasReuseFlags(prev))
        continue;

      if (getMatrixA(curr) == getMatrixA(prev))
        setReuseA(prev, true);

      if (getMatrixB(curr) == getMatrixB(prev))
        setReuseB(prev, true);
    }
  }
};

} // namespace
