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

#include "waveasm/Transforms/RegAlloc.h"
#include "waveasm/Transforms/Liveness.h"
#include "waveasm/Transforms/Passes.h"
#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace waveasm;

namespace {

//===----------------------------------------------------------------------===//
// Linear Scan Pass
//===----------------------------------------------------------------------===//

struct LinearScanPass
    : public PassWrapper<LinearScanPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinearScanPass)

  LinearScanPass() = default;
  LinearScanPass(int64_t maxVGPRs, int64_t maxSGPRs)
      : maxVGPRs(maxVGPRs), maxSGPRs(maxSGPRs) {}

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

  /// Get the accumulator operand from an MFMA op (always operand index 2)
  Value getMFMAAccumulator(Operation *op) {
    if (op->getNumOperands() >= 3) {
      return op->getOperand(2);  // acc is the third operand
    }
    return Value();
  }

  LogicalResult processProgram(ProgramOp program) {
    // Collect precolored values from precolored.vreg and precolored.sreg ops
    llvm::DenseMap<Value, int64_t> precoloredValues;
    llvm::DenseSet<int64_t> reservedVGPRs;
    llvm::DenseSet<int64_t> reservedSGPRs;

    // Collect tied operand pairs from MFMA ops
    // tiedPairs[result] = accumulator (result should get same phys reg as acc)
    llvm::DenseMap<Value, Value> tiedPairs;

    // Reserve v15 as scratch VGPR for literal materialization in assembly emitter.
    // See AssemblyEmitter.h kScratchVGPR.
    // VOP3 instructions like v_mul_lo_u32 don't support large literal operands,
    // so the emitter generates v_mov_b32 v15, <literal> before such instructions.
    reservedVGPRs.insert(15);

    program.walk([&](Operation *op) {
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
      } else if (op->hasTrait<OpTrait::MFMAOp>() && op->getNumResults() > 0) {
        // For MFMA with VGPR accumulator, tie result to accumulator
        Value acc = getMFMAAccumulator(op);
        if (acc && isVGPRType(acc.getType())) {
          // Result should be allocated to same physical register as accumulator
          tiedPairs[op->getResult(0)] = acc;
        }
      }
    });

    // Create allocator with precolored values and tied operands
    LinearScanRegAlloc allocator(maxVGPRs, maxSGPRs, reservedVGPRs,
                                  reservedSGPRs);
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
    // Set the extract result's physical register = source's physical register + offset
    program.walk([&](ExtractOp extractOp) {
      Value source = extractOp.getVector();
      Value extractResult = extractOp.getResult();
      int64_t index = extractOp.getIndex();

      // Get source's physical register (may be precolored or allocated)
      int64_t sourcePhysReg = -1;
      Type srcType = source.getType();
      if (auto pvreg = dyn_cast<PVRegType>(srcType)) {
        sourcePhysReg = pvreg.getIndex();
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

        if (physReg >= 0) {
          // Convert virtual to physical type
          if (auto vreg = dyn_cast<VRegType>(ty)) {
            newResultTypes.push_back(
                PVRegType::get(op->getContext(), physReg, vreg.getSize()));
            needsUpdate = true;
          } else if (auto sreg = dyn_cast<SRegType>(ty)) {
            newResultTypes.push_back(
                PSRegType::get(op->getContext(), physReg, sreg.getSize()));
            needsUpdate = true;
          } else {
            newResultTypes.push_back(ty);
          }
        } else {
          newResultTypes.push_back(ty);
        }
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

    return success();
  }
};

} // namespace

namespace waveasm {

std::unique_ptr<mlir::Pass> createWAVEASMLinearScanPass() {
  return std::make_unique<LinearScanPass>();
}

std::unique_ptr<mlir::Pass> createWAVEASMLinearScanPass(int64_t maxVGPRs,
                                                         int64_t maxSGPRs) {
  return std::make_unique<LinearScanPass>(maxVGPRs, maxSGPRs);
}

} // namespace waveasm
