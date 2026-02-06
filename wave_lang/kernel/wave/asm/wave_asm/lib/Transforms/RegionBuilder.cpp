// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Transforms/RegionBuilder.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "waveasm/Dialect/WaveASMOps.h"

using namespace mlir;
using namespace waveasm;

// Forward declaration
LogicalResult translateOperation(Operation *op, TranslationContext &ctx);

LogicalResult RegionBuilder::translateOp(Operation *op) {
  return translateOperation(op, ctx);
}

LoopOp RegionBuilder::buildLoopFromSCFFor(scf::ForOp forOp) {
  auto &builder = ctx.getBuilder();
  auto loc = forOp.getLoc();

  // Collect loop-carried values: [induction_var, iter_args...]
  SmallVector<Value> initArgs;
  auto lowerBound = ctx.getMapper().getMapped(forOp.getLowerBound());
  if (!lowerBound) {
    forOp.emitError("lower bound not mapped");
    return nullptr;
  }

  // Convert lower bound to sreg if it's an immediate (loop counter needs sreg
  // type)
  Value lowerBoundValue = *lowerBound;
  if (isa<ImmType>(lowerBoundValue.getType())) {
    auto sregType = ctx.createSRegType();
    lowerBoundValue =
        S_MOV_B32::create(builder, loc, sregType, lowerBoundValue);
  }
  initArgs.push_back(lowerBoundValue);

  for (Value arg : forOp.getInitArgs()) {
    if (auto mapped = ctx.getMapper().getMapped(arg)) {
      Value mappedValue = *mapped;
      // Convert immediate iter args to vregs (they'll be used in VALU ops)
      if (isa<ImmType>(mappedValue.getType())) {
        // Infer vreg size from the original SCF type.
        // For vector types (e.g. vector<4xf32> for MFMA accumulators),
        // we need a vreg with matching element count and alignment.
        int64_t vregSize = 1;
        if (auto vecType = dyn_cast<VectorType>(arg.getType())) {
          vregSize = vecType.getNumElements();
        }
        int64_t vregAlign = vregSize > 1 ? vregSize : 1;
        auto vregType = ctx.createVRegType(vregSize, vregAlign);
        mappedValue = V_MOV_B32::create(builder, loc, vregType, mappedValue);
      }
      initArgs.push_back(mappedValue);
    } else {
      forOp.emitError("init arg not mapped");
      return nullptr;
    }
  }

  // Create loop op (result types inferred from initArgs)
  auto loopOp = LoopOp::create(builder, loc, initArgs);
  Block &bodyBlock = loopOp.getBodyBlock();

  // Map loop induction variable and iter args to block arguments
  ctx.getMapper().mapValue(forOp.getInductionVar(), bodyBlock.getArgument(0));

  size_t argIdx = 1;
  for (Value origArg : forOp.getRegionIterArgs()) {
    if (argIdx < bodyBlock.getNumArguments()) {
      ctx.getMapper().mapValue(origArg, bodyBlock.getArgument(argIdx++));
    }
  }

  // Translate loop body
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&bodyBlock);

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (failed(translateOp(&op))) {
      forOp.emitError("failed to translate loop body operation");
      return nullptr;
    }
  }

  // Build loop increment and condition
  Value inductionVar = bodyBlock.getArgument(0);

  auto step = ctx.getMapper().getMapped(forOp.getStep());
  auto upperBound = ctx.getMapper().getMapped(forOp.getUpperBound());

  if (!step || !upperBound) {
    forOp.emitError("step or upper bound not mapped");
    return nullptr;
  }

  auto sregType = ctx.createSRegType();
  Value nextIV = S_ADD_U32::create(builder, loc, sregType, inductionVar, *step);
  // S_CMP now returns an SCC value explicitly
  Value cond =
      S_CMP_LT_U32::create(builder, loc, sregType, nextIV, *upperBound);

  // Collect iter args from yield
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  SmallVector<Value> iterArgs;
  iterArgs.push_back(nextIV);

  for (Value result : yieldOp.getResults()) {
    if (auto mapped = ctx.getMapper().getMapped(result)) {
      iterArgs.push_back(*mapped);
    } else {
      yieldOp.emitError("yield result not mapped");
      return nullptr;
    }
  }

  // Create condition terminator
  ConditionOp::create(builder, loc, cond, iterArgs);

  // Map loop results to for results.
  // scf.for results correspond 1:1 with iter_args (no induction variable).
  // waveasm.loop results include the induction variable at index 0,
  // then iter_args starting at index 1.
  assert(forOp.getResults().size() + 1 == loopOp.getResults().size() &&
         "result count mismatch between scf.for and waveasm.loop");
  size_t resIdx = 0;
  for (Value forRes : forOp.getResults()) {
    ctx.getMapper().mapValue(forRes, loopOp.getResults()[resIdx + 1]);
    resIdx++;
  }

  return loopOp;
}

IfOp RegionBuilder::buildIfFromSCFIf(scf::IfOp ifOp) {
  auto &builder = ctx.getBuilder();
  auto loc = ifOp.getLoc();

  // Get mapped condition
  auto condition = ctx.getMapper().getMapped(ifOp.getCondition());
  if (!condition) {
    ifOp.emitError("condition not mapped");
    return nullptr;
  }

  // Convert condition to sreg if it's an immediate
  // (arith.cmpi maps result to immediate placeholder)
  Value conditionValue = *condition;
  if (isa<ImmType>(conditionValue.getType())) {
    auto sregType = ctx.createSRegType();
    conditionValue = S_MOV_B32::create(builder, loc, sregType, conditionValue);
  }

  // Infer result types by peeking at what the then region will yield
  // This requires translating yield operands first to get their mapped types
  SmallVector<Type> resultTypes;
  auto thenYield =
      cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
  for (Value yieldVal : thenYield.getOperands()) {
    // If already mapped, use its type
    if (auto mapped = ctx.getMapper().getMapped(yieldVal)) {
      resultTypes.push_back(mapped->getType());
    } else {
      // Use a default vreg type if not yet mapped
      // (will be set properly during body translation)
      resultTypes.push_back(ctx.createVRegType());
    }
  }

  // Create if operation with inferred result types
  bool hasElse = !ifOp.getElseRegion().empty();
  auto waveIfOp =
      IfOp::create(builder, loc, resultTypes, conditionValue, hasElse);

  // Translate then region
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&waveIfOp.getThenBlock());

    for (Operation &op : ifOp.getThenRegion().front().without_terminator()) {
      if (failed(translateOp(&op))) {
        ifOp.emitError("failed to translate then region");
        return nullptr;
      }
    }

    // Get yield values and create waveasm.yield
    auto scfYield =
        cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());

    SmallVector<Value> yieldVals;
    for (Value res : scfYield.getResults()) {
      if (auto mapped = ctx.getMapper().getMapped(res)) {
        yieldVals.push_back(*mapped);
      } else {
        scfYield.emitError("yield result not mapped");
        return nullptr;
      }
    }

    YieldOp::create(builder, loc, yieldVals);
  }

  // Translate else region if present
  if (hasElse) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(waveIfOp.getElseBlock());

    for (Operation &op : ifOp.getElseRegion().front().without_terminator()) {
      if (failed(translateOp(&op))) {
        ifOp.emitError("failed to translate else region");
        return nullptr;
      }
    }

    auto scfYield =
        cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());

    SmallVector<Value> yieldVals;
    for (Value res : scfYield.getResults()) {
      if (auto mapped = ctx.getMapper().getMapped(res)) {
        yieldVals.push_back(*mapped);
      } else {
        scfYield.emitError("yield result not mapped");
        return nullptr;
      }
    }

    YieldOp::create(builder, loc, yieldVals);
  }

  // Map if results
  for (auto [ifRes, waveRes] :
       llvm::zip(ifOp.getResults(), waveIfOp.getResults())) {
    ctx.getMapper().mapValue(ifRes, waveRes);
  }

  return waveIfOp;
}
