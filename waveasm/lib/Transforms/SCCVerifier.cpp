// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// SCC Verifier Pass
//
// Verifies that no SCC-clobbering SALU instruction is placed between an
// SCC-producing op and its consumer.  Uses isa<> checks instead of
// hasTrait<SCCDef>() because adding traits to existing op classes changes
// ODS-generated C++ and causes MLIR passes to produce different IR.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMInterfaces.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "waveasm-scc-verifier"

using namespace mlir;
using namespace waveasm;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMSCCVERIFIER
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

namespace {

/// Returns true if the operation writes the SCC flag on hardware.
/// Uses hasTrait<SCCDef> for ops that carry the trait (carry ops, cmp ops),
/// and isa<> for ops still on SALUBinaryOp/SALUUnaryOp (pending migration).
static bool writesSCC(Operation *op) {
  return op->hasTrait<OpTrait::SCCDef>();
}

struct SCCVerifierPass
    : public waveasm::impl::WAVEASMSCCVerifierBase<SCCVerifierPass> {
  using WAVEASMSCCVerifierBase::WAVEASMSCCVerifierBase;

  void runOnOperation() override {
    Operation *module = getOperation();
    unsigned errorCount = 0;
    module->walk([&](ProgramOp program) {
      for (Block &block : program.getBody())
        errorCount += verifyBlock(block);
    });
    if (errorCount > 0) {
      LLVM_DEBUG(llvm::dbgs() << "SCC verifier: found " << errorCount
                               << " SCC hazard(s)\n");
      signalPassFailure();
    }
  }

private:
  static SmallVector<Operation *> findSCCClobbersBetween(Operation *producer,
                                                          Operation *consumer) {
    SmallVector<Operation *> clobbers;
    if (!producer || !consumer || producer->getBlock() != consumer->getBlock())
      return clobbers;
    bool inRange = false;
    for (Operation &op : *producer->getBlock()) {
      if (&op == producer) { inRange = true; continue; }
      if (&op == consumer) break;
      if (inRange && writesSCC(&op))
        clobbers.push_back(&op);
    }
    return clobbers;
  }

  static void emitSCCClobberError(Operation *consumer, Operation *producer,
                                   ArrayRef<Operation *> clobbers) {
    auto diag = consumer->emitError()
                << "SCC hazard: " << clobbers.size()
                << " SCC-clobbering op(s) between SCC producer '"
                << producer->getName() << "' and consumer '"
                << consumer->getName() << "'";
    for (Operation *c : clobbers)
      diag.attachNote(c->getLoc())
          << "SCC clobbered here by '" << c->getName() << "'";
    diag.attachNote(producer->getLoc()) << "SCC defined here";
  }

  unsigned verifyBlock(Block &block) {
    unsigned errors = 0;
    Operation *lastSCCWriter = nullptr;
    for (Operation &op : block) {
      if (auto condOp = dyn_cast<ConditionOp>(&op)) {
        Value cond = condOp.getCondition();
        Operation *condDef = cond.getDefiningOp();
        if (condDef && lastSCCWriter && lastSCCWriter != condDef) {
          auto clobbers = findSCCClobbersBetween(condDef, &op);
          if (!clobbers.empty()) {
            emitSCCClobberError(&op, condDef, clobbers);
            ++errors;
          }
        }
      }
      if (auto ifOp = dyn_cast<IfOp>(&op)) {
        Value cond = ifOp.getCondition();
        Operation *condDef = cond.getDefiningOp();
        if (condDef && lastSCCWriter && lastSCCWriter != condDef) {
          auto clobbers = findSCCClobbersBetween(condDef, &op);
          if (!clobbers.empty()) {
            emitSCCClobberError(&op, condDef, clobbers);
            ++errors;
          }
        }
      }
      if (isa<S_CSELECT_B32>(&op) && !lastSCCWriter) {
        op.emitError()
            << "SCC hazard: s_cselect_b32 has no preceding SCC writer";
        ++errors;
      }
      if (isa<S_ADDC_U32>(&op) && !lastSCCWriter) {
        op.emitError()
            << "SCC hazard: s_addc_u32 has no preceding SCC writer";
        ++errors;
      }
      if (writesSCC(&op))
        lastSCCWriter = &op;
      for (Region &region : op.getRegions())
        for (Block &nestedBlock : region)
          errors += verifyBlock(nestedBlock);
    }
    return errors;
  }
};

} // namespace
