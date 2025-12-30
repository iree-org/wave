// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/Transforms/UniformityAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/Transforms/DataFlowAnalyses.h"
#include "water/Dialect/Wave/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::dataflow;

#define DEBUG_TYPE "wave-uniformity-analysis"

namespace wave {
#define GEN_PASS_DEF_WATERWAVEUNIFORMITYANALYSISPASS
#include "water/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace wave

namespace {

//===----------------------------------------------------------------------===//
// UniformityLatticeStorage
//===----------------------------------------------------------------------===//

// Lattice storage representing uniformity state of a value.
// The lattice has three states:
//   - Bottom (uninitialized): analysis hasn't visited this value yet.
//   - Uniform: value is the same across all threads in a wavefront.
//   - Divergent: value may differ between threads.
//
// Lattice order: Bottom < Uniform < Divergent (Top).
class UniformityLatticeStorage {
public:
  enum class State {
    Bottom,   // Uninitialized.
    Uniform,  // Value is uniform across threads.
    Divergent // Value is divergent (top of lattice).
  };

  UniformityLatticeStorage() : state(State::Bottom) {}
  UniformityLatticeStorage(State state) : state(state) {}

  State getState() const { return state; }

  bool isUniform() const { return state == State::Uniform; }
  bool isDivergent() const { return state == State::Divergent; }
  bool isBottom() const { return state == State::Bottom; }

  bool operator==(const UniformityLatticeStorage &other) const {
    return state == other.state;
  }

  bool operator!=(const UniformityLatticeStorage &other) const {
    return !(*this == other);
  }

  // Return the top lattice instance.
  static UniformityLatticeStorage top() {
    return UniformityLatticeStorage(State::Divergent);
  }

  // Return the uniform lattice instance.
  static UniformityLatticeStorage uniform() {
    return UniformityLatticeStorage(State::Uniform);
  }

  // Join operation for the lattice.
  // Bottom ⊔ x = x
  // Uniform ⊔ Uniform = Uniform
  // Uniform ⊔ Divergent = Divergent
  // Divergent ⊔ x = Divergent
  static UniformityLatticeStorage join(const UniformityLatticeStorage &lhs,
                                       const UniformityLatticeStorage &rhs) {
    if (lhs.state == rhs.state)
      return lhs;
    if (lhs.isBottom())
      return rhs;
    if (rhs.isBottom())
      return lhs;
    // One is uniform, the other is divergent -> divergent.
    return top();
  }

  // Meet is same as join for this lattice.
  static UniformityLatticeStorage meet(const UniformityLatticeStorage &lhs,
                                       const UniformityLatticeStorage &rhs) {
    return join(lhs, rhs);
  }

  void print(llvm::raw_ostream &os) const {
    switch (state) {
    case State::Bottom:
      os << "bottom";
      break;
    case State::Uniform:
      os << "uniform";
      break;
    case State::Divergent:
      os << "divergent";
      break;
    }
  }

private:
  State state;
};

// Typed lattice wrapper.
class UniformityLattice : public dataflow::Lattice<UniformityLatticeStorage> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UniformityLattice);
  using Lattice::Lattice;
};

//===----------------------------------------------------------------------===//
// UniformityAnalysis
//===----------------------------------------------------------------------===//

// Forward dataflow analysis to propagate uniformity information.
class UniformityAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<UniformityLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  // Helper: Mark all results as divergent.
  void setAllResultsDivergent(ArrayRef<UniformityLattice *> results) {
    for (UniformityLattice *result : results)
      propagateIfChanged(result, result->join(UniformityLatticeStorage::top()));
  }

  // Helper: Mark all results as uniform.
  void setAllResultsUniform(ArrayRef<UniformityLattice *> results) {
    for (UniformityLattice *result : results)
      propagateIfChanged(result,
                         result->join(UniformityLatticeStorage::uniform()));
  }

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const UniformityLattice *> operands,
                               ArrayRef<UniformityLattice *> results) override {
    // Handle GPU-specific operations.
    // Thread ID x and lane ID are divergent (different per lane in wavefront).
    if (auto threadIdOp = dyn_cast<gpu::ThreadIdOp>(op)) {
      bool isDivergent = threadIdOp.getDimension() == gpu::Dimension::x;
      if (isDivergent)
        setAllResultsDivergent(results);
      else
        setAllResultsUniform(results);
      return success();
    }

    // Lane ID is divergent (identifies individual lanes within wavefront).
    if (isa<gpu::LaneIdOp>(op)) {
      setAllResultsDivergent(results);
      return success();
    }

    // Block IDs, grid dims, and block dims are uniform within a block.
    if (isa<gpu::BlockIdOp, gpu::GridDimOp, gpu::BlockDimOp>(op)) {
      setAllResultsUniform(results);
      return success();
    }

    // Default propagation: mark results as divergent if any operand
    // is divergent, otherwise uniform.
    bool anyDivergent =
        llvm::any_of(operands, [](const UniformityLattice *lattice) {
          return lattice && lattice->getValue().isDivergent();
        });

    if (anyDivergent)
      setAllResultsDivergent(results);
    else
      setAllResultsUniform(results);

    return success();
  }

  void setToEntryState(UniformityLattice *lattice) override {
    // Entry state: mark as uniform by default.
    // Specific divergent sources (like thread IDs) should be handled
    // separately.
    propagateIfChanged(lattice,
                       lattice->join(UniformityLatticeStorage::uniform()));
  }
};

//===----------------------------------------------------------------------===//
// UniformityAnalysisPass
//===----------------------------------------------------------------------===//

struct UniformityAnalysisPass
    : public wave::impl::WaterWaveUniformityAnalysisPassBase<
          UniformityAnalysisPass> {
  void runOnOperation() override {
    Operation *op = getOperation();

    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();
    wave::addWaveUniformityAnalysis(solver);

    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    if (failed(wave::setWaveUniformityAnalysisResults(op, solver)))
      return signalPassFailure();
  }
};

} // namespace

namespace wave {

void addWaveUniformityAnalysis(DataFlowSolver &solver) {
  solver.load<UniformityAnalysis>();
}

LogicalResult setWaveUniformityAnalysisResults(Operation *top,
                                               const DataFlowSolver &solver) {
  // Walk all operations and attach uniformity attributes.
  top->walk([&](Operation *op) {
    // Check if all results are uniform.
    bool allResultsUniform = true;
    for (Value result : op->getResults()) {
      const UniformityLattice *lattice =
          solver.lookupState<UniformityLattice>(result);
      if (!lattice || !lattice->getValue().isUniform()) {
        allResultsUniform = false;
        break;
      }
    }

    // Attach unit attribute if all results are uniform.
    if (allResultsUniform && op->getNumResults() > 0)
      op->setAttr("wave.uniform", UnitAttr::get(op->getContext()));
  });

  return success();
}

} // namespace wave
