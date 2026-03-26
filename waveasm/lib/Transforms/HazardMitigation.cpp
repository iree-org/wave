// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Hazard Mitigation Pass - Insert s_nop instructions for hardware hazards
//
// This pass handles hardware-specific hazards that require NOP insertion:
// - VALU → v_readfirstlane hazard (gfx940+)
// - Trans → non-Trans VALU forwarding hazard (gfx940+)
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMAttrs.h"
#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Transforms/Liveness.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace waveasm;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMHAZARDMITIGATION
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

namespace {

//===----------------------------------------------------------------------===//
// Instruction Classification
//===----------------------------------------------------------------------===//

/// Check if an operation is a VALU instruction (writes VGPR, not memory)
bool isVALUOp(Operation *op) {
  // Must produce at least one VGPR result
  bool writesVGPR = false;
  for (Value result : op->getResults()) {
    if (isa<VRegType, PVRegType>(result.getType())) {
      writesVGPR = true;
      break;
    }
  }
  if (!writesVGPR)
    return false;

  // Exclude memory operations (VMEM, LDS, SMEM)
  if (isa<BUFFER_LOAD_DWORD, BUFFER_LOAD_DWORDX2, BUFFER_LOAD_DWORDX3,
          BUFFER_LOAD_DWORDX4, BUFFER_LOAD_UBYTE, BUFFER_LOAD_SBYTE,
          BUFFER_LOAD_USHORT, BUFFER_LOAD_SSHORT, GLOBAL_LOAD_DWORD,
          GLOBAL_LOAD_DWORDX2, GLOBAL_LOAD_DWORDX3, GLOBAL_LOAD_DWORDX4,
          GLOBAL_LOAD_UBYTE, GLOBAL_LOAD_SBYTE, GLOBAL_LOAD_USHORT,
          GLOBAL_LOAD_SSHORT, FLAT_LOAD_DWORD, FLAT_LOAD_DWORDX2,
          FLAT_LOAD_DWORDX3, FLAT_LOAD_DWORDX4, DS_READ_B32, DS_READ_B64,
          DS_READ_B128, DS_READ2_B32, DS_READ2_B64, DS_READ_U8, DS_READ_I8,
          DS_READ_U16, DS_READ_I16>(op))
    return false;

  // Exclude non-ALU ops that produce VGPRs
  if (isa<PrecoloredVRegOp, PackOp, ExtractOp>(op))
    return false;

  // Exclude v_readfirstlane (it's the consumer in the hazard, not the producer)
  if (isa<V_READFIRSTLANE_B32>(op))
    return false;

  return true;
}

/// Check if an operation is v_readfirstlane
bool isReadfirstlaneOp(Operation *op) { return isa<V_READFIRSTLANE_B32>(op); }

/// Check if an operation does NOT emit an assembly instruction.
/// These ops are lowered to register aliases or eliminated entirely,
/// so they don't create real inter-instruction delays.
bool isNonEmittingOp(Operation *op) {
  return isa<ExtractOp, PackOp, PrecoloredSRegOp, PrecoloredVRegOp, ConstantOp,
             DCEProtectOp>(op);
}

/// Check if an operation is a transcendental instruction (uses the Trans
/// pipeline which has different latency characteristics from the main VALU).
bool isTransOp(Operation *op) {
  return isa<V_RCP_F32, V_RCP_F64, V_RSQ_F32, V_RSQ_F64, V_SQRT_F32, V_SQRT_F64,
             V_EXP_F32, V_LOG_F32, V_SIN_F32, V_COS_F32>(op);
}

/// Get the set of VGPRs written by an operation (SSA values).
llvm::DenseSet<Value> getVGPRDefs(Operation *op) {
  llvm::DenseSet<Value> defs;
  for (Value result : op->getResults()) {
    if (isa<VRegType, PVRegType>(result.getType()))
      defs.insert(result);
  }
  return defs;
}

/// Get the set of VGPRs read by an operation (SSA values).
llvm::DenseSet<Value> getVGPRUses(Operation *op) {
  llvm::DenseSet<Value> uses;
  for (Value operand : op->getOperands()) {
    if (isa<VRegType, PVRegType>(operand.getType()))
      uses.insert(operand);
  }
  return uses;
}

/// Check if a VALU def set and a VALU use set have a conflicting VGPR.
/// After register allocation, different SSA values can share the same
/// physical register, so we compare physical indices when available and
/// fall back to SSA identity for virtual registers.
bool hasVGPRConflict(const llvm::DenseSet<Value> &defs,
                     const llvm::DenseSet<Value> &uses) {
  for (Value d : defs) {
    if (uses.contains(d))
      return true;
    if (auto pdReg = dyn_cast<PVRegType>(d.getType())) {
      int64_t dIdx = pdReg.getIndex();
      for (Value u : uses) {
        if (auto puReg = dyn_cast<PVRegType>(u.getType())) {
          if (puReg.getIndex() == dIdx)
            return true;
        }
      }
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Target-Specific Hazard Rules
//===----------------------------------------------------------------------===//

/// Check if target requires VALU → readfirstlane hazard mitigation
static bool needsVALUReadFirstLaneHazard(TargetAttrInterface target) {
  // gfx940+ (CDNA3/4) architectures need this hazard mitigation
  return isa<GFX942TargetAttr, GFX950TargetAttr, GFX1250TargetAttr>(target);
}

//===----------------------------------------------------------------------===//
// Hazard Mitigation Pass
//===----------------------------------------------------------------------===//

struct HazardMitigationPass
    : public waveasm::impl::WAVEASMHazardMitigationBase<HazardMitigationPass> {
  using WAVEASMHazardMitigationBase::WAVEASMHazardMitigationBase;

  void runOnOperation() override {
    Operation *module = getOperation();

    // Parse target arch from option.
    std::optional<TargetKind> parsed = symbolizeTargetKind(targetArch);
    if (!parsed) {
      module->emitError() << "Invalid target architecture: '" << targetArch
                          << "'. Supported targets: gfx942, gfx950, gfx1250";
      return signalPassFailure();
    }
    targetKindEnum = *parsed;

    // Process each program.
    module->walk([&](ProgramOp program) { processProgram(program); });
  }

private:
  TargetKind targetKindEnum = TargetKind::GFX942;
  unsigned numNopsInserted = 0;

  void processProgram(ProgramOp program) {
    TargetAttrInterface targetKind;
    // Get target from program if available.
    if (auto targetAttr = program.getTarget()) {
      targetKind = targetAttr.getTargetKind();
    } else {
      targetKind = getTargetKindAttr(program.getContext(), targetKindEnum);
    }

    // Check if this target needs VALU → readfirstlane hazard mitigation
    bool needsVALUHazard = needsVALUReadFirstLaneHazard(targetKind);
    if (!needsVALUHazard)
      return;

    // Collect operations in order, recursively walking into while/if bodies
    llvm::SmallVector<Operation *> ops;
    collectOpsRecursive(program.getBodyBlock(), ops);

    // Scan for hazards and collect insertion points.
    // Non-emitting ops (constants, extracts, precolored refs) don't produce
    // assembly instructions, so we look past them to find the preceding
    // emitting instruction that would actually be adjacent in the output.
    llvm::SmallVector<Operation *> insertionPoints;

    for (size_t i = 0; i < ops.size(); ++i) {
      Operation *op = ops[i];

      if (isReadfirstlaneOp(op)) {
        // Walk backwards to find the nearest emitting predecessor.
        Operation *pred = nullptr;
        for (size_t j = i; j > 0; --j) {
          Operation *candidate = ops[j - 1];
          if (!isNonEmittingOp(candidate)) {
            pred = candidate;
            break;
          }
        }
        if (pred && isVALUOp(pred)) {
          auto defs = getVGPRDefs(pred);
          auto uses = getVGPRUses(op);
          if (hasVGPRConflict(defs, uses))
            insertionPoints.push_back(op);
        }
      }

      // Check for Trans -> non-Trans VALU forwarding hazard (gfx940+).
      if (isVALUOp(op) && !isTransOp(op) && i > 0) {
        Operation *pred = nullptr;
        for (size_t j = i; j > 0; --j) {
          Operation *candidate = ops[j - 1];
          if (!isNonEmittingOp(candidate)) {
            pred = candidate;
            break;
          }
        }
        if (pred && isTransOp(pred)) {
          auto defs = getVGPRDefs(pred);
          auto uses = getVGPRUses(op);
          if (hasVGPRConflict(defs, uses))
            insertionPoints.push_back(op);
        }
      }
    }

    // Insert s_nop instructions
    for (Operation *insertBefore : insertionPoints) {
      OpBuilder builder(insertBefore);
      S_NOP::create(builder, insertBefore->getLoc(),
                    builder.getI32IntegerAttr(0));
      numNopsInserted++;
    }
  }
};

} // namespace
