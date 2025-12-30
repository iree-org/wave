// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/Transforms/UniformityAnalysis.h"

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
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
// The lattice has four states:
//   - Bottom (uninitialized): analysis hasn't visited this value yet.
//   - Uniform: value is the same across all threads in a wavefront.
//   - SubgroupLinear(width): value varies linearly within subgroup with given
//   width.
//   - Divergent: value may differ arbitrarily between threads.
//
// Lattice order: Bottom < Uniform < SubgroupLinear(w) < Divergent (Top).
class UniformityLatticeStorage {
public:
  enum class State {
    Bottom,         // Uninitialized.
    Uniform,        // Value is uniform across threads.
    SubgroupLinear, // Value varies linearly within subgroup.
    Divergent       // Value is divergent (top of lattice).
  };

  UniformityLatticeStorage() : state(State::Bottom), width(0) {}
  UniformityLatticeStorage(State state) : state(state), width(0) {}
  UniformityLatticeStorage(State state, uint64_t width)
      : state(state), width(width) {}

  State getState() const { return state; }
  uint64_t getWidth() const { return width; }

  bool isUniform() const { return state == State::Uniform; }
  bool isDivergent() const { return state == State::Divergent; }
  bool isBottom() const { return state == State::Bottom; }
  bool isSubgroupLinear() const { return state == State::SubgroupLinear; }

  bool operator==(const UniformityLatticeStorage &other) const {
    if (state != other.state)
      return false;
    if (state == State::SubgroupLinear)
      return width == other.width;
    return true;
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

  // Return a subgroup linear lattice instance with given width.
  static UniformityLatticeStorage subgroupLinear(uint64_t width) {
    // Width <= 1 collapses to uniform.
    if (width <= 1)
      return uniform();
    return UniformityLatticeStorage(State::SubgroupLinear, width);
  }

  // Join operation for the lattice.
  // Bottom ⊔ x = x
  // Uniform ⊔ Uniform = Uniform
  // Uniform ⊔ SubgroupLinear(w) = SubgroupLinear(w)
  // SubgroupLinear(w1) ⊔ SubgroupLinear(w2) = SubgroupLinear(max(w1, w2))
  // x ⊔ Divergent = Divergent
  static UniformityLatticeStorage join(const UniformityLatticeStorage &lhs,
                                       const UniformityLatticeStorage &rhs) {
    if (lhs == rhs)
      return lhs;
    if (lhs.isBottom())
      return rhs;
    if (rhs.isBottom())
      return lhs;
    if (lhs.isDivergent() || rhs.isDivergent())
      return top();

    // At least one is SubgroupLinear or both are Uniform/SubgroupLinear.
    if (lhs.isUniform() && rhs.isUniform())
      return uniform();
    if (lhs.isUniform())
      return rhs;
    if (rhs.isUniform())
      return lhs;

    // Both are SubgroupLinear.
    return subgroupLinear(std::max(lhs.width, rhs.width));
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
    case State::SubgroupLinear:
      os << "subgroup_linear(" << width << ")";
      break;
    case State::Divergent:
      os << "divergent";
      break;
    }
  }

private:
  State state;
  uint64_t width; // Only used for SubgroupLinear state.
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

  // Attribute name for subgroup size on function operations.
  static constexpr StringLiteral SUBGROUP_SIZE_ATTR = "subgroup_size";

  // Helper: Get subgroup size from parent function attribute.
  std::optional<uint64_t> getSubgroupSize(Operation *op) {
    auto func = op->getParentOfType<FunctionOpInterface>();
    if (!func)
      return std::nullopt;

    auto attr = func->getAttrOfType<IntegerAttr>(SUBGROUP_SIZE_ATTR);
    if (!attr)
      return std::nullopt;

    return attr.getValue().getZExtValue();
  }

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

  // Helper: Set all results to subgroup linear with given width.
  void setAllResultsSubgroupLinear(ArrayRef<UniformityLattice *> results,
                                   uint64_t width) {
    for (UniformityLattice *result : results)
      propagateIfChanged(
          result,
          result->join(UniformityLatticeStorage::subgroupLinear(width)));
  }

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const UniformityLattice *> operands,
                               ArrayRef<UniformityLattice *> results) override {
    // Handle GPU-specific operations.
    // Thread ID x is subgroup linear (varies within wavefront).
    if (auto threadIdOp = dyn_cast<gpu::ThreadIdOp>(op)) {
      if (threadIdOp.getDimension() == gpu::Dimension::x) {
        if (auto subgroupSize = getSubgroupSize(op))
          setAllResultsSubgroupLinear(results, *subgroupSize);
        else
          setAllResultsDivergent(results);
      } else {
        setAllResultsUniform(results);
      }
      return success();
    }

    // Lane ID is subgroup linear (identifies individual lanes within
    // wavefront).
    if (isa<gpu::LaneIdOp>(op)) {
      if (auto subgroupSize = getSubgroupSize(op))
        setAllResultsSubgroupLinear(results, *subgroupSize);
      else
        setAllResultsDivergent(results);
      return success();
    }

    // Block IDs, grid dims, and block dims are uniform within a block.
    if (isa<gpu::BlockIdOp, gpu::GridDimOp, gpu::BlockDimOp>(op)) {
      setAllResultsUniform(results);
      return success();
    }

    // Subgroup broadcast always produces uniform result (broadcasts value to
    // all lanes).
    if (isa<gpu::SubgroupBroadcastOp>(op)) {
      setAllResultsUniform(results);
      return success();
    }

    // Handle division: SubgroupLinear(w) / N -> SubgroupLinear(w/N) if w % N
    // == 0.
    if (isa<arith::DivSIOp, arith::DivUIOp>(op)) {
      if (operands.size() == 2 && results.size() == 1) {
        const auto &lhs = operands[0]->getValue();
        const auto &rhs = operands[1]->getValue();

        // If LHS is SubgroupLinear and RHS is uniform constant.
        if (lhs.isSubgroupLinear() && rhs.isUniform()) {
          if (auto divisor = getConstantIntValue(op->getOperand(1))) {
            uint64_t divisorVal = *divisor;
            uint64_t width = lhs.getWidth();

            // Only remain SubgroupLinear if evenly divisible.
            if (divisorVal > 0 && width % divisorVal == 0) {
              setAllResultsSubgroupLinear(results, width / divisorVal);
              return success();
            }
          }
        }
        // Fall through to default handling.
      }
    }

    // Handle multiplication: SubgroupLinear(w) * N -> SubgroupLinear(w*N) if
    // no overflow.
    if (isa<arith::MulIOp>(op)) {
      if (operands.size() == 2 && results.size() == 1) {
        const auto &lhs = operands[0]->getValue();
        const auto &rhs = operands[1]->getValue();

        // If one is SubgroupLinear and other is uniform constant.
        const UniformityLatticeStorage *linear = nullptr;
        Value constOperand;

        if (lhs.isSubgroupLinear() && rhs.isUniform()) {
          linear = &lhs;
          constOperand = op->getOperand(1);
        } else if (rhs.isSubgroupLinear() && lhs.isUniform()) {
          linear = &rhs;
          constOperand = op->getOperand(0);
        }

        if (linear) {
          if (auto factor = getConstantIntValue(constOperand)) {
            uint64_t factorVal = *factor;
            uint64_t width = linear->getWidth();

            // Check for overflow.
            if (factorVal > 0 && width <= UINT64_MAX / factorVal) {
              setAllResultsSubgroupLinear(results, width * factorVal);
              return success();
            }
          }
        }
        // Fall through to default handling.
      }
    }

    // Default propagation: mark results as divergent if any operand
    // is divergent or subgroup linear, otherwise uniform.
    bool anyNonUniform =
        llvm::any_of(operands, [](const UniformityLattice *lattice) {
          return lattice && !lattice->getValue().isUniform();
        });

    if (anyNonUniform)
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

static void setWaveUniformityAnalysisResults(Operation *top,
                                             const DataFlowSolver &solver) {
  // Walk all operations and attach uniformity attributes.
  top->walk([&](Operation *op) {
    // Check if all results are uniform.
    bool allResultsUniform = true;
    for (Value result : op->getResults()) {
      if (!wave::isUniform(result, solver)) {
        allResultsUniform = false;
        break;
      }
    }

    // Attach unit attribute if all results are uniform.
    if (allResultsUniform && op->getNumResults() > 0)
      op->setAttr("wave.uniform", UnitAttr::get(op->getContext()));
  });
}

struct UniformityAnalysisPass
    : public wave::impl::WaterWaveUniformityAnalysisPassBase<
          UniformityAnalysisPass> {
  void runOnOperation() override {
    Operation *op = getOperation();

    DataFlowSolver solver;
    loadBaselineAnalyses(solver);
    wave::addWaveUniformityAnalysis(solver);

    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    setWaveUniformityAnalysisResults(op, solver);
  }
};

} // namespace

namespace wave {

void addWaveUniformityAnalysis(DataFlowSolver &solver) {
  solver.load<UniformityAnalysis>();
}

bool isUniform(Value value, const DataFlowSolver &solver) {
  const UniformityLattice *lattice =
      solver.lookupState<UniformityLattice>(value);
  return lattice && lattice->getValue().isUniform();
}

} // namespace wave
