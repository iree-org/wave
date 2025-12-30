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

  UniformityLatticeStorage(State state = State::Bottom) : state(state) {}
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
  uint64_t width = 0; // Only used for SubgroupLinear state.
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

  // Get subgroup size from parent function attribute.
  std::optional<uint64_t> getSubgroupSize(Operation *op) {
    auto func = op->getParentOfType<FunctionOpInterface>();
    if (!func)
      return std::nullopt;

    auto attr = func->getAttrOfType<IntegerAttr>(SUBGROUP_SIZE_ATTR);
    if (!attr)
      return std::nullopt;

    return attr.getValue().getZExtValue();
  }

  // Mark all results as divergent.
  void setAllResultsDivergent(ArrayRef<UniformityLattice *> results) {
    for (UniformityLattice *result : results)
      propagateIfChanged(result, result->join(UniformityLatticeStorage::top()));
  }

  // Mark all results as uniform.
  void setAllResultsUniform(ArrayRef<UniformityLattice *> results) {
    for (UniformityLattice *result : results)
      propagateIfChanged(result,
                         result->join(UniformityLatticeStorage::uniform()));
  }

  // Set all results to subgroup linear with given width.
  void setAllResultsSubgroupLinear(ArrayRef<UniformityLattice *> results,
                                   uint64_t width) {
    for (UniformityLattice *result : results)
      propagateIfChanged(
          result,
          result->join(UniformityLatticeStorage::subgroupLinear(width)));
  }

  // Handle division of SubgroupLinear by a uniform divisor.
  // Returns true if handled (uniform or subgroup linear), false otherwise.
  bool handleSubgroupLinearDivision(uint64_t width, uint64_t divisor,
                                    ArrayRef<UniformityLattice *> results) {
    if (divisor == 0)
      return false;

    // If divisor is a multiple of width and larger, result is uniform.
    if (divisor > width && divisor % width == 0) {
      setAllResultsUniform(results);
      return true;
    }

    // Remain SubgroupLinear if width evenly divisible by divisor.
    if (width % divisor == 0) {
      setAllResultsSubgroupLinear(results, width / divisor);
      return true;
    }

    return false;
  }

  // Handle multiplication of SubgroupLinear by a uniform multiplier.
  // Returns the result width on success, or 0 on failure (overflow).
  uint64_t handleSubgroupLinearMultiplication(uint64_t width,
                                              uint64_t multiplier) {
    if (multiplier == 0)
      return 0;

    // Check for overflow.
    if (width > (std::numeric_limits<decltype(width)>::max() / multiplier))
      return 0;

    return width * multiplier;
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
    // == 0, or Uniform if N % w == 0 and N > w.
    if (isa<arith::DivSIOp, arith::DivUIOp>(op)) {
      assert(operands.size() == 2 && results.size() == 1 &&
             "Division must have 2 operands and 1 result");
      const UniformityLatticeStorage &lhs = operands[0]->getValue();
      const UniformityLatticeStorage &rhs = operands[1]->getValue();

      // If LHS is SubgroupLinear and RHS is uniform constant.
      if (lhs.isSubgroupLinear() && rhs.isUniform()) {
        if (auto divisor = getConstantIntValue(op->getOperand(1))) {
          if (handleSubgroupLinearDivision(lhs.getWidth(), *divisor, results))
            return success();
        }
      }
      // Fall through to default handling.
    }

    // Handle multiplication: SubgroupLinear(w) * N -> SubgroupLinear(w*N) if
    // no overflow (nsw flag set).
    if (auto mulOp = dyn_cast<arith::MulIOp>(op)) {
      assert(operands.size() == 2 && results.size() == 1 &&
             "Multiplication must have 2 operands and 1 result");
      const UniformityLatticeStorage &lhs = operands[0]->getValue();
      const UniformityLatticeStorage &rhs = operands[1]->getValue();

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
          // Check if nsw flag is set.
          if (bitEnumContainsAny(mulOp.getOverflowFlags(),
                                 arith::IntegerOverflowFlags::nsw)) {
            if (uint64_t newWidth = handleSubgroupLinearMultiplication(
                    linear->getWidth(), *factor)) {
              setAllResultsSubgroupLinear(results, newWidth);
              return success();
            }
          }
        }
      }
      // Fall through to default handling.
    }

    // Handle shift left: SubgroupLinear(w) << N -> SubgroupLinear(w << N) if
    // nsw flag set (similar to multiplication by 2^N).
    if (auto shliOp = dyn_cast<arith::ShLIOp>(op)) {
      assert(operands.size() == 2 && results.size() == 1 &&
             "Shift left must have 2 operands and 1 result");
      const UniformityLatticeStorage &lhs = operands[0]->getValue();
      const UniformityLatticeStorage &rhs = operands[1]->getValue();

      if (lhs.isSubgroupLinear() && rhs.isUniform()) {
        if (auto shiftAmount = getConstantIntValue(op->getOperand(1))) {
          // Check if nsw flag is set.
          if (*shiftAmount > 0 && *shiftAmount < 64 &&
              bitEnumContainsAny(shliOp.getOverflowFlags(),
                                 arith::IntegerOverflowFlags::nsw)) {
            uint64_t multiplier = 1ULL << *shiftAmount;
            if (uint64_t newWidth = handleSubgroupLinearMultiplication(
                    lhs.getWidth(), multiplier)) {
              setAllResultsSubgroupLinear(results, newWidth);
              return success();
            }
          }
        }
      }
      // Fall through to default handling.
    }

    // Handle shift right: SubgroupLinear(w) >> N -> SubgroupLinear(w >> N) if
    // w is divisible by 2^N, or Uniform if 2^N > w and 2^N % w == 0.
    if (isa<arith::ShRUIOp, arith::ShRSIOp>(op)) {
      assert(operands.size() == 2 && results.size() == 1 &&
             "Shift right must have 2 operands and 1 result");
      const UniformityLatticeStorage &lhs = operands[0]->getValue();
      const UniformityLatticeStorage &rhs = operands[1]->getValue();

      if (lhs.isSubgroupLinear() && rhs.isUniform()) {
        if (auto shiftAmount = getConstantIntValue(op->getOperand(1))) {
          if (*shiftAmount > 0 && *shiftAmount < 64) {
            uint64_t divisor = 1ULL << *shiftAmount;
            if (handleSubgroupLinearDivision(lhs.getWidth(), divisor, results))
              return success();
          }
        }
      }
      // Fall through to default handling.
    }

    // Handle bitwise AND: SubgroupLinear(w) & mask -> Uniform if mask zeros all
    // width bits (mask & (w-1) == 0).
    if (isa<arith::AndIOp>(op)) {
      assert(operands.size() == 2 && results.size() == 1 &&
             "Bitwise AND must have 2 operands and 1 result");
      const UniformityLatticeStorage &lhs = operands[0]->getValue();
      const UniformityLatticeStorage &rhs = operands[1]->getValue();

      // If one is SubgroupLinear and other is uniform constant.
      const UniformityLatticeStorage *linear = nullptr;
      Value maskOperand;

      if (lhs.isSubgroupLinear() && rhs.isUniform()) {
        linear = &lhs;
        maskOperand = op->getOperand(1);
      } else if (rhs.isSubgroupLinear() && lhs.isUniform()) {
        linear = &rhs;
        maskOperand = op->getOperand(0);
      }

      if (linear) {
        if (auto mask = getConstantIntValue(maskOperand)) {
          uint64_t width = linear->getWidth();
          // If mask zeros all bits used by width, result is uniform (all
          // zeros).
          if ((*mask & (width - 1)) == 0) {
            setAllResultsUniform(results);
            return success();
          }
        }
      }
      // Fall through to default handling.
    }

    // Handle cast operations: preserve SubgroupLinear state.
    // These operations change representation but not value distribution.
    if (isa<arith::IndexCastOp, arith::TruncIOp, arith::ExtUIOp,
            arith::ExtSIOp>(op)) {
      assert(operands.size() == 1 && results.size() == 1 &&
             "Cast must have 1 operand and 1 result");
      const UniformityLatticeStorage &operand = operands[0]->getValue();

      if (operand.isSubgroupLinear()) {
        setAllResultsSubgroupLinear(results, operand.getWidth());
        return success();
      } else if (operand.isUniform()) {
        setAllResultsUniform(results);
        return success();
      }
      // Fall through to default handling for divergent operands.
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
