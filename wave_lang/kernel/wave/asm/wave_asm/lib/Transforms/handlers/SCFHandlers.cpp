// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// SCF Dialect Handlers
//===----------------------------------------------------------------------===//
//
// This file implements handlers for SCF dialect operations:
//   - scf.for
//   - scf.if
//   - scf.yield
//
//===----------------------------------------------------------------------===//

#include "Handlers.h"

#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Transforms/TranslateFromMLIR.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace waveasm;

namespace waveasm {

LogicalResult handleSCFIf(Operation *op, TranslationContext &ctx) {
  auto ifOp = cast<scf::IfOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // Get condition
  auto cond = ctx.getMapper().getMapped(ifOp.getCondition());
  if (!cond) {
    return op->emitError("condition not mapped");
  }

  // Create labels for if/else/endif
  std::string baseName =
      "L_if_" + std::to_string(reinterpret_cast<uintptr_t>(op));
  std::string elseLabel = baseName + "_else";
  std::string endLabel = baseName + "_end";

  // Branch to else block if condition is false
  auto elseLabelRef = SymbolRefAttr::get(builder.getContext(), elseLabel);
  S_CBRANCH_SCC0::create(builder, loc, elseLabelRef);

  // Translate then region
  for (Operation &thenOp : ifOp.getThenRegion().front().without_terminator()) {
    if (failed(translateOperation(&thenOp, ctx))) {
      return failure();
    }
  }

  // Jump to end (skip else block)
  if (!ifOp.getElseRegion().empty()) {
    auto endLabelRef = SymbolRefAttr::get(builder.getContext(), endLabel);
    S_BRANCH::create(builder, loc, endLabelRef);
  }

  // Else label and block
  if (!ifOp.getElseRegion().empty()) {
    LabelOp::create(builder, loc, elseLabel);
    for (Operation &elseOp :
         ifOp.getElseRegion().front().without_terminator()) {
      if (failed(translateOperation(&elseOp, ctx))) {
        return failure();
      }
    }
  }

  // End label
  LabelOp::create(builder, loc, endLabel);

  return success();
}

LogicalResult handleSCFFor(Operation *op, TranslationContext &ctx) {
  auto forOp = cast<scf::ForOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // Generate unique loop label
  std::string labelName = ctx.generateLabel("L_loop");
  std::string endLabelName = labelName + "_end";

  // Initialize loop counter in a FIXED physical SGPR
  // Using a fixed register ensures the counter value persists across loop
  // iterations Reserve s32+ for loop counters (avoiding s0-s31 which may be
  // used for arguments/SRDs/cache-swizzle) Note: s[24:31] may be used for cache
  // swizzle SRDs in g2s kernels
  int64_t loopCounterPhysReg =
      32 + ctx.getLoopDepth(); // Use s32, s33, etc. for nested loops
  auto counterType =
      PSRegType::get(builder.getContext(), loopCounterPhysReg, 1);

  auto lb = ctx.getMapper().getMapped(forOp.getLowerBound());
  if (!lb) {
    auto immType = ctx.createImmType(0);
    lb = ConstantOp::create(builder, loc, immType, 0);
  }

  auto counter = S_MOV_B32::create(builder, loc, counterType, *lb);
  ctx.getMapper().mapValue(forOp.getInductionVar(), counter);

  // Handle iter_args (loop-carried values)
  // These are values passed from one iteration to the next (e.g., accumulators)
  // For vector-type iter_args (accumulators), we need to:
  // 1. Allocate VREGs before the loop
  // 2. Initialize them with v_mov_b32
  // 3. Map the region arg to those VREGs for in-place accumulation
  SmallVector<Value, 4> iterArgValues;
  for (auto [initArg, regionArg] :
       llvm::zip(forOp.getInitArgs(), forOp.getRegionIterArgs())) {
    auto mapped = ctx.getMapper().getMapped(initArg);

    // Check if this is a vector type (likely an accumulator for MFMA)
    Type regionArgType = regionArg.getType();
    if (auto vecType = dyn_cast<VectorType>(regionArgType)) {
      // For vector accumulators, we need to materialize VREGs
      int64_t numElems = vecType.getNumElements();

      // Check if the init value is a constant 0 (common for accumulators)
      bool isZeroInit = false;
      if (auto constOp = initArg.getDefiningOp<arith::ConstantOp>()) {
        if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
          if (denseAttr.isSplat()) {
            if (auto floatAttr =
                    dyn_cast<FloatAttr>(denseAttr.getSplatValue<Attribute>())) {
              isZeroInit = (floatAttr.getValueAsDouble() == 0.0);
            }
          }
        }
      }

      if (isZeroInit) {
        // Allocate VREGs for the accumulator and initialize to 0
        // The MFMA instruction will use these registers for in-place
        // accumulation
        auto vregType =
            ctx.createVRegType(numElems, 4); // Quad-aligned for MFMA
        auto zeroImm = ctx.createImmType(0);
        auto zero = ConstantOp::create(builder, loc, zeroImm, 0);

        // Create v_mov_b32 to initialize each element of the accumulator
        // Mark with "no_cse" to prevent CSE from merging multiple accumulators
        auto accMovOp = V_MOV_B32::create(builder, loc, vregType, zero);
        accMovOp->setAttr("no_cse", builder.getUnitAttr());
        Value accReg = accMovOp.getResult();

        // Map the region argument to this register
        ctx.getMapper().mapValue(regionArg, accReg);
        iterArgValues.push_back(accReg);
        continue;
      }
    }

    // Default: map to the initial value directly
    if (mapped) {
      ctx.getMapper().mapValue(regionArg, *mapped);
      iterArgValues.push_back(*mapped);
    }
  }

  // Set up loop context for nested operations
  LoopContext loopCtx;
  loopCtx.inductionVar = counter;
  loopCtx.iterArgs = iterArgValues;
  loopCtx.labelName = labelName;
  loopCtx.depth = ctx.getLoopDepth() + 1;
  ctx.pushLoopContext(loopCtx);

  // Clear expression cache at loop entry (loop-variant expressions must be
  // recomputed)
  ctx.clearExprCache();

  // Loop label
  LabelOp::create(builder, loc, labelName);

  // Translate loop body
  for (Operation &bodyOp : forOp.getBody()->without_terminator()) {
    if (failed(translateOperation(&bodyOp, ctx))) {
      ctx.popLoopContext();
      return failure();
    }
  }

  // Handle yield - update iter_args for next iteration
  if (auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator())) {
    for (auto [yieldedVal, regionArg] :
         llvm::zip(yieldOp.getOperands(), forOp.getRegionIterArgs())) {
      auto mapped = ctx.getMapper().getMapped(yieldedVal);
      if (mapped) {
        // Update the region argument mapping for the next iteration
        ctx.getMapper().mapValue(regionArg, *mapped);
      }
    }
  }

  // Increment counter - use the same physical register type as the counter
  // This ensures the increment writes to the same physical register
  auto step = ctx.getMapper().getMapped(forOp.getStep());
  if (!step) {
    auto immType = ctx.createImmType(1);
    step = ConstantOp::create(builder, loc, immType, 1);
  }

  // Get the counter type from the original counter (it is a physical register
  // type)
  auto counterPhysType = counter.getType();
  auto newCounter =
      S_ADD_U32::create(builder, loc, counterPhysType, counter, *step);

  // Since we are using physical registers, newCounter is in the same register
  // as counter Update the mapping for any post-loop uses
  ctx.getMapper().mapValue(forOp.getInductionVar(), newCounter);

  // Compare and branch back to loop header
  auto ub = ctx.getMapper().getMapped(forOp.getUpperBound());
  if (ub) {
    S_CMP_LT_U32::create(builder, loc, newCounter, *ub);
    auto labelRef = SymbolRefAttr::get(builder.getContext(), labelName);
    S_CBRANCH_SCC1::create(builder, loc, labelRef);
  }

  // End label for loop exit
  LabelOp::create(builder, loc, endLabelName);

  // Map loop results to final iter_arg values
  if (auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator())) {
    for (auto [result, yieldedVal] :
         llvm::zip(forOp.getResults(), yieldOp.getOperands())) {
      auto mapped = ctx.getMapper().getMapped(yieldedVal);
      if (mapped) {
        ctx.getMapper().mapValue(result, *mapped);
      }
    }
  }

  ctx.popLoopContext();
  return success();
}

/// Handle scf.yield - typically a no-op
LogicalResult handleSCFYield(Operation *op, TranslationContext &ctx) {
  // Yield values are handled by the parent op
  return success();
}

} // namespace waveasm
