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
//   1. Symbolically compute the constant stride per candidate by
//      differentiating the address chain w.r.t. the induction variable.
//   2. Group candidates by (SRD, stride); each group shares one soffset.
//   3. Precompute each voffset at iv=initial_value (loop-invariant).
//   4. Carry one soffset per group as SGPR iter_arg (starts at 0).
//   5. Each iteration: soffset += stride (one s_add_u32 per group).
//   6. Set each buffer_load's soffset to the group's soffset.
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
// computation (symbolic differentiation of address chain) as shared utility.
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

// Symbolically compute the stride of the voffset chain w.r.t. the IV.
// Returns the constant stride if determinable, nullopt otherwise.
// Works by computing the "derivative" of each op: IV -> ivStep,
// loop-invariant -> 0, add(a,b) -> da+db, lshl(a,c) -> da<<c, etc.
// The deps SetVector is in reverse topological order (DFS from voffset),
// so we iterate in reverse to process inputs before outputs.
//
// Because the result must be a compile-time integer, non-uniform strides
// are automatically rejected. E.g. v_mul_lo_u32(tid, iv) would need
// getConstantValue(tid) to succeed, which it cannot for a VGPR — so
// the candidate is skipped. No runtime readfirstlane needed.
static std::optional<int64_t>
computeStaticStride(const llvm::SetVector<Operation *> &deps, Value voffset,
                    Value iv, int64_t ivStep) {
  // Maps each Value to its per-IV-step delta (symbolic derivative).
  // Populated in topological order; absent entries are loop-invariant
  // (delta=0).
  llvm::DenseMap<Value, int64_t> delta;
  delta[iv] = ivStep;

  // Values not in the map are loop-invariant (delta=0): defined outside the
  // loop (constants, precolored VGPRs like tid) or non-IV block args
  // (accumulators). Delta=0 means "does not change across iterations", NOT
  // "has value 0" — so tid correctly gets delta=0 even though its per-thread
  // value is nonzero. For linear ops (add, sub, shift) the tid contribution
  // cancels in the derivative; for mul, we require getConstantValue on the
  // loop-invariant operand, which rejects VGPRs like tid.
  auto getDelta = [&](Value v) -> int64_t {
    auto it = delta.find(v);
    return it != delta.end() ? it->second : 0;
  };

  for (auto it = deps.rbegin(); it != deps.rend(); ++it) {
    Operation *op = *it;

    if (isa<ConstantOp>(op)) {
      for (Value r : op->getResults())
        delta[r] = 0;
      continue;
    }

    if (isa<V_MOV_B32>(op)) {
      for (Value r : op->getResults())
        delta[r] = getDelta(op->getOperand(0));
      continue;
    }

    // Validate shift amount is in [0, 31] (32-bit GPU ops) to avoid UB.
    auto validShift = [](std::optional<int64_t> amt) -> std::optional<int64_t> {
      if (!amt || *amt < 0 || *amt > 31)
        return std::nullopt;
      return amt;
    };

    if (isa<V_ADD_U32, V_LSHL_ADD_U32>(op)) {
      // v_add_u32(a, b) = a + b.
      // v_lshl_add_u32(a, b, c) = (a << b) + c.
      // For lshl_add: treat as add(lshl(a, b), c) — but b must be constant
      // and IV-independent for the shift to be linear.
      if (isa<V_LSHL_ADD_U32>(op)) {
        int64_t dSrc = getDelta(op->getOperand(0));
        int64_t dShift = getDelta(op->getOperand(1));
        int64_t dAdd = getDelta(op->getOperand(2));
        if (dShift != 0)
          return std::nullopt;
        auto shiftAmt = validShift(getConstantValue(op->getOperand(1)));
        if (!shiftAmt)
          return std::nullopt;
        for (Value r : op->getResults())
          delta[r] = (dSrc << *shiftAmt) + dAdd;
      } else {
        int64_t d = getDelta(op->getOperand(0)) + getDelta(op->getOperand(1));
        for (Value r : op->getResults())
          delta[r] = d;
      }
      continue;
    }

    if (isa<V_SUB_U32>(op)) {
      int64_t d = getDelta(op->getOperand(0)) - getDelta(op->getOperand(1));
      for (Value r : op->getResults())
        delta[r] = d;
      continue;
    }

    if (isa<V_LSHLREV_B32>(op)) {
      // lshlrev(amt, src) = src << amt.
      int64_t dAmt = getDelta(op->getOperand(0));
      int64_t dSrc = getDelta(op->getOperand(1));
      if (dAmt != 0)
        return std::nullopt;
      auto shiftAmt = validShift(getConstantValue(op->getOperand(0)));
      if (!shiftAmt)
        return std::nullopt;
      for (Value r : op->getResults())
        delta[r] = dSrc << *shiftAmt;
      continue;
    }

    if (isa<V_LSHL_OR_B32>(op)) {
      // lshl_or(src, amt, or_val) = (src << amt) | or_val.
      // In address computation, OR always has disjoint bits (packed fields),
      // so delta = (dSrc << amt) + dOr — same as lshl_add.
      int64_t dSrc = getDelta(op->getOperand(0));
      int64_t dShift = getDelta(op->getOperand(1));
      int64_t dOr = getDelta(op->getOperand(2));
      if (dShift != 0)
        return std::nullopt;
      auto shiftAmt = validShift(getConstantValue(op->getOperand(1)));
      if (!shiftAmt)
        return std::nullopt;
      for (Value r : op->getResults())
        delta[r] = (dSrc << *shiftAmt) + dOr;
      continue;
    }

    if (isa<V_MUL_LO_U32>(op)) {
      // Linear only if exactly one operand depends on IV.
      int64_t d0 = getDelta(op->getOperand(0));
      int64_t d1 = getDelta(op->getOperand(1));
      if (d0 != 0 && d1 != 0)
        return std::nullopt;
      if (d0 == 0 && d1 == 0) {
        for (Value r : op->getResults())
          delta[r] = 0;
        continue;
      }
      // One is IV-dependent, the other must be a known constant.
      Value constOperand = d0 == 0 ? op->getOperand(0) : op->getOperand(1);
      int64_t dVar = d0 != 0 ? d0 : d1;
      auto constVal = getConstantValue(constOperand);
      if (!constVal)
        return std::nullopt;
      for (Value r : op->getResults())
        delta[r] = dVar * *constVal;
      continue;
    }

    // Bitwise ops (AND, OR, XOR, BFE, LSHR): nonlinear if IV-dependent.
    bool hasIVDep = false;
    for (Value operand : op->getOperands())
      if (getDelta(operand) != 0)
        hasIVDep = true;
    if (hasIVDep)
      return std::nullopt;
    for (Value r : op->getResults())
      delta[r] = 0;
  }

  auto found = delta.find(voffset);
  return found != delta.end() ? std::optional<int64_t>(found->second)
                              : std::nullopt;
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
  auto sregType = builder.getType<SRegType>();

  // Compute static stride for each candidate by symbolically differentiating
  // the address chain w.r.t. IV. Drop candidates with non-constant strides
  // (the voffset delta may be non-uniform across iterations).
  SmallVector<int64_t> candidateStrides;
  {
    SmallVector<BufferLoadInfo> filtered;
    for (auto &info : candidates) {
      auto stride = computeStaticStride(info.deps, info.voffset, iv, *ivStep);
      if (stride) {
        candidateStrides.push_back(*stride);
        filtered.push_back(std::move(info));
      } else {
        LDBG() << "skipping candidate: cannot determine constant stride";
      }
    }
    candidates = std::move(filtered);
  }

  if (candidates.empty())
    return;

  // Group by (SRD, stride). Loads sharing the same SRD and same constant
  // stride share one soffset iter_arg; different strides get separate ones.
  struct SRDGroup {
    Value srd;
    int64_t stride;
    Value strideSGPR;
  };
  SmallVector<SRDGroup> groups;
  SmallVector<unsigned> candidateGroupIdx;

  for (unsigned i = 0; i < candidates.size(); ++i) {
    std::optional<unsigned> matchIdx;
    for (unsigned g = 0; g < groups.size(); ++g) {
      if (groups[g].srd == candidates[i].srd &&
          groups[g].stride == candidateStrides[i]) {
        matchIdx = g;
        break;
      }
    }
    if (matchIdx) {
      candidateGroupIdx.push_back(*matchIdx);
    } else {
      auto strideImm = builder.getType<ImmType>(candidateStrides[i]);
      auto strideConst =
          ConstantOp::create(builder, loc, strideImm, candidateStrides[i]);
      Value strideSGPR = S_MOV_B32::create(builder, loc, sregType, strideConst);
      candidateGroupIdx.push_back(groups.size());
      groups.push_back({candidates[i].srd, candidateStrides[i], strideSGPR});
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
    // Find the cloned buffer_load via the mapping. The lookup must succeed
    // because we cloned the entire loop body above.
    Operation *clonedLoad = nullptr;
    for (Value result : candidates[i].loadOp->getResults()) {
      Value clonedResult = mapping.lookupOrNull(result);
      assert(clonedResult && "cloned load result not found in mapping");
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
  // NOTE: s_add_u32 wraps at 2^32. For typical GEMM loops (stride < 2^20,
  // iterations < 2^12), soffset stays well within 32 bits. If the product
  // stride*iterations overflows 2^32, the buffer address will be wrong.
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
