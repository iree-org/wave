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
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMAttrs.h"
#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace waveasm;

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
          BUFFER_LOAD_USHORT, BUFFER_LOAD_SSHORT,
          GLOBAL_LOAD_DWORD, GLOBAL_LOAD_DWORDX2, GLOBAL_LOAD_DWORDX3,
          GLOBAL_LOAD_DWORDX4, GLOBAL_LOAD_UBYTE, GLOBAL_LOAD_SBYTE,
          GLOBAL_LOAD_USHORT, GLOBAL_LOAD_SSHORT,
          FLAT_LOAD_DWORD, FLAT_LOAD_DWORDX2, FLAT_LOAD_DWORDX3,
          FLAT_LOAD_DWORDX4,
          DS_READ_B32, DS_READ_B64, DS_READ_B128,
          DS_READ2_B32, DS_READ2_B64,
          DS_READ_U8, DS_READ_I8, DS_READ_U16, DS_READ_I16>(op))
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
bool isReadfirstlaneOp(Operation *op) {
  return isa<V_READFIRSTLANE_B32>(op);
}

/// Get the set of VGPRs written by an operation
llvm::DenseSet<Value> getVGPRDefs(Operation *op) {
  llvm::DenseSet<Value> defs;
  for (Value result : op->getResults()) {
    Type ty = result.getType();
    // Check if it's a VGPR type
    if (isa<VRegType, PVRegType>(ty)) {
      defs.insert(result);
    }
  }
  return defs;
}

/// Get the set of VGPRs read by an operation
llvm::DenseSet<Value> getVGPRUses(Operation *op) {
  llvm::DenseSet<Value> uses;
  for (Value operand : op->getOperands()) {
    Type ty = operand.getType();
    // Check if it's a VGPR type
    if (isa<VRegType, PVRegType>(ty)) {
      uses.insert(operand);
    }
  }
  return uses;
}

/// Check if two value sets have any intersection
bool hasIntersection(const llvm::DenseSet<Value> &a,
                     const llvm::DenseSet<Value> &b) {
  for (Value v : a) {
    if (b.contains(v))
      return true;
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
    : public PassWrapper<HazardMitigationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HazardMitigationPass)

  HazardMitigationPass() = default;
  HazardMitigationPass(StringRef target) {
    std::optional<TargetKind> targetKindEnum = symbolizeTargetKind(target);
    assert(targetKindEnum && "Invalid target");
    targetKindEnum = *targetKindEnum;
  }

  StringRef getArgument() const override { return "waveasm-hazard-mitigation"; }

  StringRef getDescription() const override {
    return "Insert s_nop instructions to mitigate hardware hazards";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Process each program
    module.walk([&](ProgramOp program) { processProgram(program); });
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

    // Collect operations in order
    llvm::SmallVector<Operation *> ops;
    for (Operation &op : program.getBodyBlock()) {
      ops.push_back(&op);
    }

    // Scan for hazards and collect insertion points
    llvm::SmallVector<Operation *> insertionPoints;

    for (size_t i = 0; i + 1 < ops.size(); ++i) {
      Operation *current = ops[i];
      Operation *next = ops[i + 1];

      // Check for VALU → v_readfirstlane hazard
      if (isVALUOp(current) && isReadfirstlaneOp(next)) {
        // Get VGPRs written by current and read by next
        auto defs = getVGPRDefs(current);
        auto uses = getVGPRUses(next);

        // If there's an intersection, we have a hazard
        if (hasIntersection(defs, uses)) {
          insertionPoints.push_back(next);
        }
      }
    }

    // Insert s_nop instructions
    for (Operation *insertBefore : insertionPoints) {
      OpBuilder builder(insertBefore);
      // Insert s_nop 0 (no extra wait states beyond the instruction itself)
      S_NOP::create(builder, insertBefore->getLoc(),
                    builder.getI32IntegerAttr(0));
      numNopsInserted++;
    }
  }
};

} // namespace

namespace waveasm {

std::unique_ptr<mlir::Pass> createWAVEASMHazardMitigationPass() {
  return std::make_unique<HazardMitigationPass>();
}

std::unique_ptr<mlir::Pass> createWAVEASMHazardMitigationPass(
    llvm::StringRef targetArch) {
  return std::make_unique<HazardMitigationPass>(targetArch);
}

} // namespace waveasm
