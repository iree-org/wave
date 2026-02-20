// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Linear Scan Register Allocation Pass
//
// This pass runs the linear scan register allocator and transforms virtual
// register types to physical register types in the IR.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Liveness.h"
#include "waveasm/Transforms/Passes.h"
#include "waveasm/Transforms/RegAlloc.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace waveasm;

/// Convert a virtual register type to a physical register type.
/// Returns the original type unchanged if it's not a virtual register type
/// or if physReg < 0.
static Type makePhysicalType(MLIRContext *ctx, Type virtualType,
                             int64_t physReg) {
  if (physReg < 0)
    return virtualType;
  if (auto vreg = dyn_cast<VRegType>(virtualType))
    return PVRegType::get(ctx, physReg, vreg.getSize());
  if (auto sreg = dyn_cast<SRegType>(virtualType))
    return PSRegType::get(ctx, physReg, sreg.getSize());
  if (auto areg = dyn_cast<ARegType>(virtualType))
    return PARegType::get(ctx, physReg, areg.getSize());
  return virtualType;
}

namespace {

//===----------------------------------------------------------------------===//
// Linear Scan Pass
//===----------------------------------------------------------------------===//

struct LinearScanPass
    : public PassWrapper<LinearScanPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinearScanPass)

  LinearScanPass() = default;
  LinearScanPass(int64_t maxVGPRs, int64_t maxSGPRs, int64_t maxAGPRs)
      : maxVGPRs(maxVGPRs), maxSGPRs(maxSGPRs), maxAGPRs(maxAGPRs) {}

  StringRef getArgument() const override { return "waveasm-linear-scan"; }

  StringRef getDescription() const override {
    return "Linear scan register allocation";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Process each program
    module.walk([&](ProgramOp program) {
      if (failed(processProgram(program))) {
        signalPassFailure();
      }
    });
  }

private:
  int64_t maxVGPRs = 256;
  int64_t maxSGPRs = 104;
  int64_t maxAGPRs = 256;

  /// Create a fresh zero-initialized copy of a duplicate init arg to ensure
  /// unique physical registers. This is used when CSE merges identical
  /// zero-initialized accumulators, causing multiple loop block args to be
  /// tied to the same init value. Each block arg needs its own physical
  /// register, so we create a new v_mov_b32/s_mov_b32 from zero.
  ///
  /// PRECONDITION: This should only be called for zero-initialized init args
  /// (e.g., v_mov_b32 %vreg, 0). Calling it for non-zero init args will
  /// produce incorrect zero values silently.
  Value createZeroInitCopy(LoopOp loopOp, Value initArg) {
    OpBuilder copyBuilder(loopOp);
    auto loc = loopOp.getLoc();

    // Create a zero immediate. We always use 0 because this function is
    // only called for duplicate init args produced by CSE merging identical
    // zero-initialized values (e.g., v_mov_b32 vN, 0).
    auto immType = ImmType::get(loopOp->getContext(), 0);
    Value zeroImm = ConstantOp::create(copyBuilder, loc, immType, 0);

    if (isAGPRType(initArg.getType())) {
      // AGPR zero-init: V_MOV_B32 with ARegType destination.
      // The assembly emitter will produce v_accvgpr_write_b32 aN, 0.
      auto aregType = cast<ARegType>(initArg.getType());
      return V_MOV_B32::create(copyBuilder, loc, aregType, zeroImm);
    }
    if (isVGPRType(initArg.getType())) {
      auto vregType = cast<VRegType>(initArg.getType());
      return V_MOV_B32::create(copyBuilder, loc, vregType, zeroImm);
    }
    if (isSGPRType(initArg.getType())) {
      auto sregType = cast<SRegType>(initArg.getType());
      return S_MOV_B32::create(copyBuilder, loc, sregType, zeroImm);
    }
    return nullptr;
  }

  /// Get the accumulator operand from an MFMA op (always operand index 2).
  /// Returns nullptr if the operation doesn't have enough operands.
  Value getMFMAAccumulator(Operation *op) {
    if (op->getNumOperands() < 3) {
      return nullptr;
    }
    return op->getOperand(2); // acc is the third operand
  }

  LogicalResult processProgram(ProgramOp program) {
    // Collect precolored values from precolored.vreg and precolored.sreg ops
    llvm::DenseMap<Value, int64_t> precoloredValues;
    llvm::DenseSet<int64_t> reservedVGPRs;
    llvm::DenseSet<int64_t> reservedSGPRs;
    llvm::DenseSet<int64_t> reservedAGPRs;

    // Collect tied operand pairs from MFMA ops
    // tiedPairs[result] = accumulator (result should get same phys reg as acc)
    llvm::DenseMap<Value, Value> tiedPairs;

    // Reserve v15 as scratch VGPR for literal materialization in assembly
    // emitter. See AssemblyEmitter.h kScratchVGPR. VOP3 instructions like
    // v_mul_lo_u32 don't support large literal operands, so the emitter
    // generates v_mov_b32 v15, <literal> before such instructions.
    reservedVGPRs.insert(15);

    // Note: ABI SGPRs (kernarg ptr, preload regs, workgroup IDs, SRDs) are
    // reserved via PrecoloredSRegOp ops emitted during translation. The
    // collection loop below picks those up and adds their indices to
    // reservedSGPRs automatically -- no manual reservation needed here.

    bool collectFailed = false;
    program.walk([&](Operation *op) {
      if (collectFailed)
        return;
      if (auto precoloredVReg = dyn_cast<PrecoloredVRegOp>(op)) {
        int64_t physIdx = precoloredVReg.getIndex();
        int64_t size = precoloredVReg.getSize();
        precoloredValues[precoloredVReg.getResult()] = physIdx;
        for (int64_t i = 0; i < size; ++i) {
          reservedVGPRs.insert(physIdx + i);
        }
      } else if (auto precoloredSReg = dyn_cast<PrecoloredSRegOp>(op)) {
        int64_t physIdx = precoloredSReg.getIndex();
        int64_t size = precoloredSReg.getSize();
        precoloredValues[precoloredSReg.getResult()] = physIdx;
        for (int64_t i = 0; i < size; ++i) {
          reservedSGPRs.insert(physIdx + i);
        }
      } else if (auto precoloredAReg = dyn_cast<PrecoloredARegOp>(op)) {
        int64_t physIdx = precoloredAReg.getIndex();
        int64_t size = precoloredAReg.getSize();
        precoloredValues[precoloredAReg.getResult()] = physIdx;
        for (int64_t i = 0; i < size; ++i) {
          reservedAGPRs.insert(physIdx + i);
        }
      } else if (op->hasTrait<OpTrait::MFMAOp>() && op->getNumResults() > 0) {
        // For MFMA with VGPR accumulator, tie result to accumulator
        Value acc = getMFMAAccumulator(op);
        if (!acc) {
          op->emitError() << "MFMA operation must have at least 3 operands "
                          << "(A, B, accumulator), but found "
                          << op->getNumOperands();
          collectFailed = true;
          return;
        }
        if ((isVGPRType(acc.getType()) &&
             isVGPRType(op->getResult(0).getType())) ||
            (isAGPRType(acc.getType()) &&
             isAGPRType(op->getResult(0).getType()))) {
          // Result should be allocated to same physical register as accumulator
          tiedPairs[op->getResult(0)] = acc;
        }
      }
    });

    if (collectFailed)
      return failure();

    // Add tied constraints for loop ops: block args share registers with
    // init args, and condition iter_args share registers with block args.
    program.walk([&](LoopOp loopOp) {
      Block &bodyBlock = loopOp.getBodyBlock();
      auto condOp = dyn_cast<ConditionOp>(bodyBlock.getTerminator());
      if (!condOp)
        return;

      // Track which init arg values have been used, to detect duplicates.
      // If multiple block args are tied to the same init arg (e.g., after
      // CSE merges identical zero-initialized accumulators), insert copies
      // so each gets a unique physical register.
      llvm::DenseSet<Value> usedInitArgs;

      for (unsigned i = 0; i < bodyBlock.getNumArguments(); ++i) {
        BlockArgument blockArg = bodyBlock.getArgument(i);

        // Tie block arg to init arg (with copy if duplicate)
        if (i < loopOp.getInitArgs().size()) {
          Value initArg = loopOp.getInitArgs()[i];

          if (usedInitArgs.contains(initArg)) {
            // Duplicate init arg: create a fresh zero-init to get a unique
            // value. This ensures each block arg gets its own phys register.
            // We re-initialize from the zero immediate rather than copying
            // the multi-register init arg (v_mov_b32 can't copy a vector).
            Value copy = createZeroInitCopy(loopOp, initArg);
            if (copy) {
              loopOp.getInitArgsMutable()[i].set(copy);
              initArg = copy;
            }
          }
          usedInitArgs.insert(initArg);
          tiedPairs[blockArg] = initArg;
        }

        // Tie condition iter_arg to block arg (skip if MFMA already tied it)
        if (i < condOp.getIterArgs().size()) {
          Value iterArg = condOp.getIterArgs()[i];
          if (!tiedPairs.contains(iterArg)) {
            tiedPairs[iterArg] = blockArg;
          }
        }

        // Tie loop result to block arg (the result is the value that
        // exits the loop, must be same register as the block arg)
        if (i < loopOp->getNumResults()) {
          tiedPairs[loopOp->getResult(i)] = blockArg;
        }
      }
    });

    // Create allocator with precolored values and tied operands
    LinearScanRegAlloc allocator(maxVGPRs, maxSGPRs, maxAGPRs, reservedVGPRs,
                                 reservedSGPRs, reservedAGPRs);
    for (const auto &[value, physIdx] : precoloredValues) {
      allocator.precolorValue(value, physIdx);
    }
    for (const auto &[result, acc] : tiedPairs) {
      allocator.addTiedOperand(result, acc);
    }

    // Run allocation
    auto result = allocator.allocate(program);
    if (failed(result)) {
      return failure();
    }

    auto [mapping, stats] = *result;

    // Handle waveasm.extract ops: result = source[offset]
    // Set the extract result's physical register = source's physical register +
    // offset
    program.walk([&](ExtractOp extractOp) {
      Value source = extractOp.getVector();
      Value extractResult = extractOp.getResult();
      int64_t index = extractOp.getIndex();

      // Get source's physical register (may be precolored or allocated)
      int64_t sourcePhysReg = -1;
      Type srcType = source.getType();
      if (auto pvreg = dyn_cast<PVRegType>(srcType)) {
        sourcePhysReg = pvreg.getIndex();
      } else if (auto pareg = dyn_cast<PARegType>(srcType)) {
        sourcePhysReg = pareg.getIndex();
      } else {
        sourcePhysReg = mapping.getPhysReg(source);
      }

      if (sourcePhysReg >= 0) {
        // Set the extract result to source + offset
        mapping.setPhysReg(extractResult, sourcePhysReg + index);
      }
    });

    // Transform the IR: replace virtual register types with physical types
    OpBuilder builder(program.getContext());

    // Track values that need type updates
    llvm::DenseMap<Value, Value> valueReplacements;

    // For each operation, update result types from virtual to physical
    program.walk([&](Operation *op) {
      // Skip program op itself
      if (isa<ProgramOp>(op))
        return;

      bool needsUpdate = false;
      SmallVector<Type> newResultTypes;

      for (Value result : op->getResults()) {
        Type ty = result.getType();
        int64_t physReg = mapping.getPhysReg(result);
        Type newTy = makePhysicalType(op->getContext(), ty, physReg);
        newResultTypes.push_back(newTy);
        if (newTy != ty)
          needsUpdate = true;
      }

      // Update the operation's result types if needed
      // Note: MLIR operations typically require replacement, but we can
      // modify in place for dialect ops that support it
      if (needsUpdate && !newResultTypes.empty()) {
        for (size_t i = 0; i < op->getNumResults(); ++i) {
          op->getResult(i).setType(newResultTypes[i]);
        }
      }
    });

    // Update block arguments and result types for region-based control flow.
    // After the walk above, operation results inside loop/if bodies have
    // physical register types, but block arguments and the parent op's
    // result types still have virtual types. We need to propagate the
    // physical types to maintain consistency.
    //
    // Strategy: Use the allocation mapping to update block argument types
    // directly to their assigned physical register types. Then propagate
    // to loop result types and init arg types.
    program.walk([&](LoopOp loopOp) {
      Block &bodyBlock = loopOp.getBodyBlock();

      // Get the condition op (terminator of body)
      auto condOp = dyn_cast<ConditionOp>(bodyBlock.getTerminator());
      if (!condOp)
        return;

      // Update block argument types from the allocation mapping.
      // Block arguments are tied to init args, so they should get the same
      // physical register. Using the mapping directly is more robust than
      // inferring from condition iter_arg types (which may themselves be
      // block arguments not yet updated, e.g. in cross-swap patterns).
      for (unsigned i = 0; i < bodyBlock.getNumArguments(); ++i) {
        BlockArgument blockArg = bodyBlock.getArgument(i);
        Type ty = blockArg.getType();
        int64_t physReg = mapping.getPhysReg(blockArg);

        if (physReg >= 0) {
          blockArg.setType(makePhysicalType(loopOp->getContext(), ty, physReg));
        } else if (i < condOp.getIterArgs().size()) {
          // Fallback: infer from condition iter_arg type (for values not in
          // the mapping, e.g. precolored values)
          Type condType = condOp.getIterArgs()[i].getType();
          if (isa<PVRegType>(condType) || isa<PSRegType>(condType)) {
            blockArg.setType(condType);
          }
        }
      }

      // Update the loop op's result types to match block arg types
      for (unsigned i = 0; i < loopOp->getNumResults(); ++i) {
        if (i < bodyBlock.getNumArguments()) {
          loopOp->getResult(i).setType(bodyBlock.getArgument(i).getType());
        }
      }
    });

    // Also update if op result types from yield operand types
    program.walk([&](IfOp ifOp) {
      auto &thenBlock = ifOp.getThenBlock();
      if (auto yieldOp = dyn_cast<YieldOp>(thenBlock.getTerminator())) {
        for (unsigned i = 0; i < ifOp->getNumResults(); ++i) {
          if (i < yieldOp.getResults().size()) {
            ifOp->getResult(i).setType(yieldOp.getResults()[i].getType());
          }
        }
      }
    });

    return success();
  }
};

} // namespace

namespace waveasm {

std::unique_ptr<mlir::Pass> createWAVEASMLinearScanPass() {
  return std::make_unique<LinearScanPass>();
}

std::unique_ptr<mlir::Pass> createWAVEASMLinearScanPass(int64_t maxVGPRs,
                                                        int64_t maxSGPRs,
                                                        int64_t maxAGPRs) {
  return std::make_unique<LinearScanPass>(maxVGPRs, maxSGPRs, maxAGPRs);
}

} // namespace waveasm
