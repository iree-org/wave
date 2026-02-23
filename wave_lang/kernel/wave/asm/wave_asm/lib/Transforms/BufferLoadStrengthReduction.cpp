// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Buffer Load Strength Reduction Pass
//
// Replaces per-iteration buffer_load voffset recomputation with precomputed
// voffsets and SGPR soffset bumping. For each buffer_load in a loop whose
// voffset depends on the induction variable:
//
//   1. Precompute the voffset at iv=initial_value (loop-invariant).
//   2. Compute the stride per SRD group (SGPR, via v_readfirstlane).
//   3. Carry one soffset per SRD group as SGPR iter_arg (starts at 0).
//   4. Each iteration: soffset += stride (one s_add_u32 per SRD group).
//   5. Set each buffer_load's soffset to the group's soffset.
//
// This eliminates ALL VALU address computation from the loop body.
//
// Why buffer-load-specific and not a general loop strength reduction:
//
// Buffer instructions have effective_address = SRD_base + voffset + soffset,
// where soffset is an SGPR-only scalar offset added at zero VALU cost by the
// hardware. The voffset captures the thread-specific part (loop-invariant),
// and soffset captures the iteration-dependent part (uniform across threads,
// lives in SGPR). This lets us go from N VALU/iter → 1 SALU/SRD-group/iter.
//
// A general strength reduction would produce v_add(base, accumulated_stride)
// — still 1 VALU per iteration — which a subsequent buffer-load peephole
// could split into voffset/soffset. But the only consumers of IV-dependent
// address chains in GEMM loops are buffer_loads and LDS ops (handled by
// LoopAddressPromotion with a different strategy: rotating precomputed VGPRs).
// Splitting into two passes adds an abstraction boundary for one consumer.
// If non-buffer IV-dependent VALU chains appear later, factor out the stride
// computation (clone-at-two-IVs, subtract, readfirstlane) as shared utility.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Passes.h"
#include "waveasm/Transforms/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "waveasm-buffer-load-strength-reduction"

using namespace mlir;
using namespace waveasm;

namespace {

static bool isAddressVALU(Operation *op) {
  return isa<V_LSHLREV_B32, V_LSHL_OR_B32, V_LSHL_ADD_U32, V_ADD_U32, V_SUB_U32,
             V_AND_B32, V_OR_B32, V_LSHRREV_B32, V_MUL_LO_U32, V_MOV_B32,
             V_XOR_B32, V_BFE_U32>(op);
}

static bool isBufferLoad(Operation *op) {
  return isa<BUFFER_LOAD_DWORD, BUFFER_LOAD_DWORDX2, BUFFER_LOAD_DWORDX3,
             BUFFER_LOAD_DWORDX4, BUFFER_LOAD_UBYTE, BUFFER_LOAD_USHORT>(op);
}

static bool isDefinedInLoop(Value val, Region *loopRegion) {
  if (auto *defOp = val.getDefiningOp())
    return defOp->getParentRegion() == loopRegion;
  if (auto ba = dyn_cast<BlockArgument>(val))
    return ba.getOwner()->getParent() == loopRegion;
  return false;
}

static void collectVoffsetDeps(Value voffset, Region *loopRegion,
                               llvm::SetVector<Operation *> &deps) {
  SmallVector<Value> worklist;
  worklist.push_back(voffset);
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    auto *defOp = v.getDefiningOp();
    if (!defOp)
      continue;
    if (defOp->getParentRegion() != loopRegion)
      continue;
    if (!deps.insert(defOp))
      continue;
    for (Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }
}

static std::optional<int64_t> findIVStep(ConditionOp condOp, Block &body) {
  ValueRange condArgs = condOp.getIterArgs();
  if (condArgs.empty())
    return std::nullopt;
  Value nextIV = condArgs[0];
  auto addOp = nextIV.getDefiningOp<S_ADD_U32>();
  if (!addOp)
    return std::nullopt;
  Value iv = body.getArgument(0);
  auto tryExtract = [&](Value maybeSrc,
                        Value maybeConst) -> std::optional<int64_t> {
    if (maybeSrc != iv)
      return std::nullopt;
    return getConstantValue(maybeConst);
  };
  if (auto step = tryExtract(addOp.getSrc0(), addOp.getSrc1()))
    return step;
  if (auto step = tryExtract(addOp.getSrc1(), addOp.getSrc0()))
    return step;
  return std::nullopt;
}

static bool dependsOnIV(const llvm::SetVector<Operation *> &deps, Value iv) {
  for (Operation *op : deps)
    for (Value operand : op->getOperands())
      if (operand == iv)
        return true;
  return false;
}

static Value cloneChainBeforeLoop(const llvm::SetVector<Operation *> &deps,
                                  Value targetVoffset, Value ivValue,
                                  LoopOp loopOp, Block &body,
                                  OpBuilder &builder) {
  IRMapping mapping;
  ValueRange initArgs = loopOp.getInitArgs();
  for (unsigned i = 0; i < body.getNumArguments(); ++i)
    mapping.map(body.getArgument(i), initArgs[i]);
  mapping.map(body.getArgument(0), ivValue);
  for (Operation &op : body)
    if (deps.contains(&op))
      builder.clone(op, mapping);
  return mapping.lookupOrDefault(targetVoffset);
}

struct BufferLoadInfo {
  Operation *loadOp;
  Value voffset;
  Value srd;
  llvm::SetVector<Operation *> deps;
};

static void applyStrengthReduction(LoopOp loopOp) {
  Block &body = loopOp.getBodyBlock();
  auto condOp = dyn_cast<ConditionOp>(body.getTerminator());
  if (!condOp)
    return;

  unsigned numArgs = body.getNumArguments();
  ValueRange condIterArgs = condOp.getIterArgs();
  if (numArgs == 0 || condIterArgs.size() != numArgs)
    return;

  auto ivStep = findIVStep(condOp, body);
  if (!ivStep)
    return;

  Region *loopRegion = &loopOp.getBodyRegion();
  Value iv = body.getArgument(0);

  SmallVector<BufferLoadInfo> candidates;
  llvm::SetVector<Operation *> allDeps;

  for (Operation &op : body) {
    if (!isBufferLoad(&op))
      continue;
    if (op.getNumOperands() < 3) // saddr, voffset, soffset.
      continue;

    Value voffset = op.getOperand(1);
    Value srd = op.getOperand(0);

    if (!isDefinedInLoop(voffset, loopRegion))
      continue;

    llvm::SetVector<Operation *> deps;
    collectVoffsetDeps(voffset, loopRegion, deps);

    if (!dependsOnIV(deps, iv))
      continue;

    bool allPure = true;
    for (Operation *dep : deps) {
      if (!isAddressVALU(dep) && !isa<ConstantOp>(dep)) {
        allPure = false;
        break;
      }
    }
    if (!allPure)
      continue;

    allDeps.insert(deps.begin(), deps.end());

    BufferLoadInfo info;
    info.loadOp = &op;
    info.voffset = voffset;
    info.srd = srd;
    info.deps = std::move(deps);
    candidates.push_back(std::move(info));
  }

  if (candidates.empty())
    return;

  LDBG() << "found " << candidates.size() << " buffer_loads to optimize";

  OpBuilder builder(loopOp);
  auto loc = loopOp.getLoc();
  ValueRange initArgs = loopOp.getInitArgs();
  Value ivInit = initArgs[0];

  auto stepImm = builder.getType<ImmType>(*ivStep);
  auto stepConst = ConstantOp::create(builder, loc, stepImm, *ivStep);
  auto sregType = builder.getType<SRegType>();
  auto vregType = builder.getType<VRegType>(1, 1);
  Value ivPlusStep =
      S_ADD_U32::create(builder, loc, sregType, ivInit, stepConst);

  // Group candidates by SRD. Compute one stride per group.
  struct SRDGroup {
    Value srd;
    Value strideVGPR; // stride as VGPR (from v_sub).
    Value strideSGPR; // stride as SGPR (from v_readfirstlane).
  };
  llvm::DenseMap<Value, unsigned> srdToGroupIdx;
  SmallVector<SRDGroup> groups;
  SmallVector<unsigned> candidateGroupIdx;

  for (unsigned i = 0; i < candidates.size(); ++i) {
    auto it = srdToGroupIdx.find(candidates[i].srd);
    if (it == srdToGroupIdx.end()) {
      // Compute stride for this SRD group using the first candidate.
      Value voff0 =
          cloneChainBeforeLoop(candidates[i].deps, candidates[i].voffset,
                               ivInit, loopOp, body, builder);
      Value voff1 =
          cloneChainBeforeLoop(candidates[i].deps, candidates[i].voffset,
                               ivPlusStep, loopOp, body, builder);
      Value strideVGPR =
          V_SUB_U32::create(builder, loc, vregType, voff1, voff0);
      Value strideSGPR =
          V_READFIRSTLANE_B32::create(builder, loc, sregType, strideVGPR);

      srdToGroupIdx[candidates[i].srd] = groups.size();
      candidateGroupIdx.push_back(groups.size());
      groups.push_back({candidates[i].srd, strideVGPR, strideSGPR});
    } else {
      candidateGroupIdx.push_back(it->second);
    }
  }

  // Precompute all initial voffsets (at iv=initial_value, soffset=0).
  SmallVector<Value> initialVoffsets;
  for (auto &info : candidates) {
    Value voff = cloneChainBeforeLoop(info.deps, info.voffset, ivInit, loopOp,
                                      body, builder);
    initialVoffsets.push_back(voff);
  }

  // Build expanded init args: old args + soffset per SRD group (starts at 0).
  SmallVector<Value> expandedInit(initArgs.begin(), initArgs.end());
  unsigned soffsetArgBase = expandedInit.size();
  auto zeroImm = builder.getType<ImmType>(0);
  auto zeroConst = ConstantOp::create(builder, loc, zeroImm, 0);
  auto zeroSoff = S_MOV_B32::create(builder, loc, sregType, zeroConst);
  for (unsigned g = 0; g < groups.size(); ++g)
    expandedInit.push_back(zeroSoff);

  // Build new loop.
  auto newLoop = LoopOp::create(builder, loc, expandedInit);
  Block &newBody = newLoop.getBodyBlock();

  // Map old block args to new block args.
  IRMapping mapping;
  for (unsigned i = 0; i < numArgs; ++i)
    mapping.map(body.getArgument(i), newBody.getArgument(i));

  // Clone loop body.
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(&newBody);
  for (Operation &op : body) {
    if (isa<ConditionOp>(&op))
      continue;
    bodyBuilder.clone(op, mapping);
  }

  // Patch buffer_loads: set voffset to precomputed value, soffset to iter_arg.
  for (unsigned i = 0; i < candidates.size(); ++i) {
    // Find the cloned buffer_load.
    Operation *clonedLoad = nullptr;
    for (Value result : candidates[i].loadOp->getResults()) {
      Value clonedResult = mapping.lookup(result);
      clonedLoad = clonedResult.getDefiningOp();
      break;
    }
    if (!clonedLoad)
      continue;

    // Replace voffset (operand 1) with precomputed initial voffset
    // (defined before the loop, accessible inside).
    clonedLoad->setOperand(1, initialVoffsets[i]);

    // Replace soffset (operand 2) with the group's soffset iter_arg.
    unsigned groupIdx = candidateGroupIdx[i];
    Value soffsetArg = newBody.getArgument(soffsetArgBase + groupIdx);
    clonedLoad->setOperand(2, soffsetArg);
  }

  // Compute next soffsets BEFORE the condition (s_add_u32 clobbers SCC,
  // and s_cmp -> s_cbranch must have no SCC-clobbering ops between them).
  // Find the cloned s_cmp/s_add for IV (the ops that produce the condition)
  // and insert soffset updates before them.
  Value newCond = mapping.lookup(condOp.getCondition());

  // Insert soffset updates before the s_cmp that produces the condition.
  Operation *condProducer = newCond.getDefiningOp();
  if (condProducer) {
    OpBuilder preCondBuilder(condProducer);
    SmallVector<Value> nextSoffs;
    for (unsigned g = 0; g < groups.size(); ++g) {
      Value currentSoff = newBody.getArgument(soffsetArgBase + g);
      Value nextSoff = S_ADD_U32::create(preCondBuilder, loc, sregType,
                                         currentSoff, groups[g].strideSGPR);
      nextSoffs.push_back(nextSoff);
    }

    SmallVector<Value> newCondIterArgs;
    for (Value v : condIterArgs)
      newCondIterArgs.push_back(mapping.lookup(v));
    for (Value s : nextSoffs)
      newCondIterArgs.push_back(s);

    ConditionOp::create(bodyBuilder, loc, newCond, newCondIterArgs);
  } else {
    // Fallback: no condition producer found, just append.
    SmallVector<Value> newCondIterArgs;
    for (Value v : condIterArgs)
      newCondIterArgs.push_back(mapping.lookup(v));
    for (unsigned g = 0; g < groups.size(); ++g) {
      Value currentSoff = newBody.getArgument(soffsetArgBase + g);
      Value nextSoff = S_ADD_U32::create(bodyBuilder, loc, sregType,
                                         currentSoff, groups[g].strideSGPR);
      newCondIterArgs.push_back(nextSoff);
    }
    ConditionOp::create(bodyBuilder, loc, newCond, newCondIterArgs);
  }

  // Replace old loop results.
  for (unsigned i = 0; i < numArgs; ++i)
    loopOp.getResult(i).replaceAllUsesWith(newLoop.getResult(i));

  // Verify no cross-references.
  bool hasCrossRefs = false;
  for (Operation &op : body) {
    if (isa<ConditionOp>(&op))
      continue;
    for (Value result : op.getResults()) {
      for (OpOperand &use : result.getUses()) {
        if (use.getOwner()->getParentRegion() != &loopOp.getBodyRegion()) {
          hasCrossRefs = true;
          break;
        }
      }
      if (hasCrossRefs)
        break;
    }
    if (hasCrossRefs)
      break;
  }

  if (hasCrossRefs) {
    LDBG() << "cross-references detected, reverting";
    for (unsigned i = 0; i < numArgs; ++i)
      newLoop.getResult(i).replaceAllUsesWith(loopOp.getResult(i));
    newLoop.erase();
    return;
  }

  loopOp.erase();
}

struct BufferLoadStrengthReductionPass
    : public PassWrapper<BufferLoadStrengthReductionPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BufferLoadStrengthReductionPass)

  StringRef getArgument() const override {
    return "waveasm-buffer-load-strength-reduction";
  }

  StringRef getDescription() const override {
    return "Replace per-iteration buffer_load address computation with "
           "soffset-based incremental addressing";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<LoopOp> loops;
    module.walk([&](LoopOp loopOp) { loops.push_back(loopOp); });
    for (auto loopOp : loops)
      applyStrengthReduction(loopOp);
  }
};

} // namespace

namespace waveasm {

std::unique_ptr<mlir::Pass> createWAVEASMBufferLoadStrengthReductionPass() {
  return std::make_unique<BufferLoadStrengthReductionPass>();
}

} // namespace waveasm
