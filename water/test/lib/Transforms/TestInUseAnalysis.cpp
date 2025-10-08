// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Transforms/Passes.h"

#include "water/Analysis/InUseForSpeculation.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::water::test {
#define GEN_PASS_DEF_TESTINUSEANALYSISPASS
#include "Transforms/Passes.h.inc"
} // namespace mlir::water::test

using namespace mlir;
using namespace mlir::water;

/// Attach in-use information as attributes to operations defining values
/// (results and block arguments).
static LogicalResult testInUseAnalysis(Operation *top) {
  DataFlowConfig config;
  config.setInterprocedural(false);
  DataFlowSolver solver(config);
  SymbolTableCollection symbolTable;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
  solver.load<InUseAnalysis>(symbolTable);
  if (failed(solver.initializeAndRun(top)))
    return top->emitError() << "failed to complete in-use analysis";

  // Pre-create attributes so we don't take the lock all the time during the
  // walk.
  auto latticeBottom = StringAttr::get(top->getContext(), "bottom");
  auto used = StringAttr::get(top->getContext(), "used");
  auto unused = StringAttr::get(top->getContext(), "unused");
  auto latticeTop = StringAttr::get(top->getContext(), "top");
  auto missing = StringAttr::get(top->getContext(), "missing");

  auto getAttrForValueList = [&](ValueRange values) -> Attribute {
    return ArrayAttr::get(
        top->getContext(),
        llvm::map_to_vector(values, [&](Value result) -> Attribute {
          auto *inUseLattice = solver.lookupState<InUseLattice>(result);
          if (!inUseLattice)
            return missing;

          switch (inUseLattice->getValue().get()) {
          case InUseLatticeState::Uninitialized:
            return latticeBottom;
          case InUseLatticeState::Unused:
            return unused;
          case InUseLatticeState::Used:
            return used;
          case InUseLatticeState::Undecidable:
            return latticeTop;
          }
          llvm_unreachable("unhandled enum case");
        }));
  };

  top->walk([&](Operation *op) {
    if (isa<ModuleOp>(op))
      return;
    op->setAttr("__in_use_results", getAttrForValueList(op->getResults()));
    for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i) {
      for (auto &&[j, block] : llvm::enumerate(op->getRegion(i))) {
        op->setAttr("__in_use_blockarg_" + std::to_string(i) + "_" +
                        std::to_string(j),
                    getAttrForValueList(block.getArguments()));
      }
    }
  });
  return success();
}

class TestInUseAnalysisPass
    : public water::test::impl::TestInUseAnalysisPassBase<
          TestInUseAnalysisPass> {
public:
  void runOnOperation() override {
    if (failed(testInUseAnalysis(getOperation())))
      return signalPassFailure();
  }
};
