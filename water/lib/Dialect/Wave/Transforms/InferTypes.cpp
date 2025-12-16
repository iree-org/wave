// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "water/Dialect/Wave/IR/IndexExpr.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/Transforms/DataFlowAnalyses.h"
#include "water/Dialect/Wave/Transforms/Passes.h"
#include "water/Dialect/Wave/Transforms/Utils.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/FormatVariadic.h"
#include <type_traits>

#define DEBUG_TYPE "wave-infer-types"

using wave::ElementsPerThreadLatticeValue;
using wave::IndexExprsLatticeStorage;

namespace wave {
#define GEN_PASS_DEF_WATERWAVEINFERTYPESPASS
#define GEN_PASS_DEF_WATERWAVEPROPAGATEELEMENTSPERTHREADPASS
#define GEN_PASS_DEF_WATERWAVEINFERINDEXEXPRSPASS
#include "water/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace wave

namespace {

//-----------------------------------------------------------------------------
// WaveInferTypeOpInterface and implementation traits
//-----------------------------------------------------------------------------

// Core lattice for type/shape inference of wave tensors. In addition to the
// bottom and top states, it can represent a concrete type which may be
// a fully specified tensor type (specific) or an underspecified type (any). The
// JOIN function is defined by the following table:
//
// JOIN         top       specific       any         bottom
// top          top       top            top         top
// specific     top       specific|top*  specific    specific
// any          top       specific       any         any
// bottom       top       specific       any         bottom
//   * if two specific shapes are equal, their JOIN is equal to them, otherwise
//     it is an inference conflict and the result of a JOIN is the top state.
//
// The explicit bottom type is mostly there for debugging purposes: lattice
// instances are default-constructed in this state and them remaining in it
// after the analysis converges means the analysis hasn't updated or may not
// have explicitly initialized the state.
class InferTypeLatticeStorage {
public:
  InferTypeLatticeStorage() : value(nullptr, kUninitializedState) {}
  InferTypeLatticeStorage(const InferTypeLatticeStorage &value) = default;
  InferTypeLatticeStorage(wave::WaveTensorType concreteValue)
      : value(concreteValue, kSpecificTypeState) {}

  InferTypeLatticeStorage &
  operator=(const InferTypeLatticeStorage &other) = default;

  bool operator==(const InferTypeLatticeStorage &other) const {
    return value == other.value;
  }

  bool operator!=(const InferTypeLatticeStorage &other) const {
    return !(*this == other);
  }

  // Return true if this lattice instance is the bottom state.
  bool isBottom() const { return value.getInt() == kUninitializedState; }

  // Return true if this lattice instance is the top state.
  bool isTop() const { return value.getInt() == kUndecidableState; }

  // Returns the concrete type stored in the lattice instance, be it fully
  // specified or not, or null if the lattice instance is a top or a bottom.
  wave::WaveTensorType getConcreteValue() const {
    if (value.getInt() != kSpecificTypeState)
      return nullptr;
    return llvm::cast<wave::WaveTensorType>(value.getPointer());
  }

  // Return the top lattice instance.
  static InferTypeLatticeStorage top() {
    InferTypeLatticeStorage result;
    result.value.setPointer(nullptr);
    result.value.setInt(kUndecidableState);
    return result;
  }

  // Join the two lattice instances and return the result.
  static InferTypeLatticeStorage join(const InferTypeLatticeStorage &lhs,
                                      const InferTypeLatticeStorage &rhs) {
    if (lhs.value == rhs.value)
      return lhs;

    if (lhs.isTop() || rhs.isTop())
      return top();

    if (lhs.isBottom())
      return rhs;

    if (rhs.isBottom())
      return lhs;

    // If one of the types is under-specified, return the other type.
    wave::WaveTensorType lhsType = lhs.getConcreteValue();
    wave::WaveTensorType rhsType = rhs.getConcreteValue();
    if (!lhsType.getFullySpecified())
      return rhs;
    if (!rhsType.getFullySpecified())
      return lhs;

    // We only care about shape matches.
    if (lhsType.getShape() == rhsType.getShape())
      return lhsType;

    return top();
  }

  // XXX: backward analysis calls `meet` instead of `join`, but it isn't related
  // to the direction of the analysis. Just defer to join.
  static InferTypeLatticeStorage meet(const InferTypeLatticeStorage &lhs,
                                      const InferTypeLatticeStorage &rhs) {
    return join(lhs, rhs);
  }

  // Forcibly assign the current value of the lattice. This MUST NOT be used in
  // the transfer functions as it may be moving the instance back on the lattice
  // and therefore breaking the analysis convergence guarantees due to
  // non-monotonicity. This is useful during forceful initialization to override
  // the quirk of the dataflow framework using the same function
  // (`setToEntry/ExitState`) to both initialize the analysis and to indicate
  // failure to analyze. Those functions can keep setting the lattice to the top
  // state.
  void unsafeSet(const InferTypeLatticeStorage &value) {
    this->value = value.value;
  }

  void print(llvm::raw_ostream &os) const {
    if (isBottom())
      os << "<bottom>";
    else if (isTop())
      os << "<top>";
    else
      os << getConcreteValue();
  }

  LLVM_DUMP_METHOD void dump() const { print(llvm::errs()); }

private:
  // The internal storage is either a type or one of the top/bottom flags.
  llvm::PointerIntPair<mlir::Type, 2> value;

  // State flags.
  const static unsigned kUninitializedState = 0;
  const static unsigned kSpecificTypeState = 1;
  const static unsigned kUndecidableState = 2;
};

// Typed lattice object for type inference.
class InferTypeLattice
    : public mlir::dataflow::Lattice<InferTypeLatticeStorage> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InferTypeLattice);
  using Lattice::Lattice;
};

// Helper function for forward/backward inference handling ops that do not
// implement the inference interface. Returns success if the op doesn't
// manipulate wave tensor types, failure otherwise. Returns nullopt if the op
// does implement the interface.
static std::optional<llvm::LogicalResult>
handleNonInterfaceOpInferType(mlir::Operation *op) {
  if (llvm::isa<wave::WaveInferTypeOpInterface>(op))
    return std::nullopt;

  if (!llvm::any_of(op->getOperandTypes(),
                    llvm::IsaPred<wave::WaveTensorType>) &&
      !llvm::any_of(op->getResultTypes(),
                    llvm::IsaPred<wave::WaveTensorType>)) {
    return mlir::success();
  }
  return op->emitError()
         << "cannot propagate types across an operation not implementing "
            "the wave infer type interface";
}

// Dataflow analysis propagating type/shape information from operands to
// results. This is an optimistic sparse context-insensitive forward dataflow
// analysis intended for intra-procedural use and composition with the
// equivalent backward analysis. It starts by initializing a lattice instance
// for all wave tensor-typed operation results and block arguments to their
// current type and propagates shape information across operations using
// WaveInferTypeOpInterface as well as across control flow using regular control
// flow interfaces until convergence. In case of a type inference conflict, e.g.
// the pre-existing type is different from the type inferred by dataflow, a
// diagnostic is emitted and the lattice instance corresponding to the value
// with type conflict is set to the top state. If the analysis fails due to the
// lack of information, e.g., control flow operations not implementing the
// requested interfaces, the lattice instances may be set to the top state
// without diagnostics.
class InferTypeForwardAnalysis
    : public mlir::dataflow::SparseForwardDataFlowAnalysis<InferTypeLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  // Forcibly initializes lattice instances for wave tensor-typed operation
  // results and block arguments of supported operations to their current type.
  //
  // XXX: this works since we never revisit the function entry block again in
  // the intra-procedural case. Without this function, the framework would call
  // `setToEntryState` that would trigger setting to lattice top which we don't
  // want.
  mlir::LogicalResult initialize(mlir::Operation *top) override {
    if (getSolverConfig().isInterprocedural())
      return top->emitError() << "interprocedural analysis not supported";

    // Call the base class initialization in order to set up update listeners.
    // Note that this will initialize values at function/region entries to
    // lattice top.
    if (mlir::failed(AbstractSparseForwardDataFlowAnalysis::initialize(top)))
      return mlir::failure();

    // Reset the initialization of values that may have been initialized to
    // lattice top to the concrete type instead.
    top->walk([&](mlir::Operation *op) {
      if (auto iface = llvm::dyn_cast<wave::WaveInferTypeOpInterface>(op)) {
        initForResults(iface);
      } else if (auto iface = llvm::dyn_cast<mlir::FunctionOpInterface>(op)) {
        if (!iface.isDeclaration())
          initForBlockArguments(iface.getFunctionBody().front());
      } else if (auto iterate = llvm::dyn_cast<wave::IterateOp>(op)) {
        initForResults(op);
        initForBlockArguments(iterate.getBody().front());
      }
      return mlir::WalkResult::advance();
    });
    return mlir::success();
  }

  // Called by base class initialization and when the analysis fails to identify
  // lattices to join.
  void setToEntryState(InferTypeLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(InferTypeLatticeStorage::top()));
  }

  // Dataflow transfer function, defers to the WaveInferTypeOpInterface.
  mlir::LogicalResult
  visitOperation(mlir::Operation *op,
                 llvm::ArrayRef<const InferTypeLattice *> operands,
                 llvm::ArrayRef<InferTypeLattice *> results) override {
    std::optional<mlir::LogicalResult> res = handleNonInterfaceOpInferType(op);
    if (res)
      return *res;

    auto extractType = [](const InferTypeLattice *lattice) {
      return lattice->getValue().getConcreteValue();
    };
    llvm::SmallVector<wave::WaveTensorType> operandTypes =
        llvm::map_to_vector(operands, extractType);
    llvm::SmallVector<wave::WaveTensorType> resultTypes =
        llvm::map_to_vector(results, extractType);

    std::string errorMessage;
    llvm::raw_string_ostream errs(errorMessage);
    llvm::FailureOr<mlir::ChangeResult> result =
        llvm::cast<wave::WaveInferTypeOpInterface>(op).propagateForward(
            operandTypes, resultTypes, errs);
    if (mlir::failed(result)) {
      return op->emitError()
             << "failed to propagate type information forward: " << errs.str();
    }
    if (*result == mlir::ChangeResult::NoChange)
      return mlir::success();

    for (auto &&[result, lattice] : llvm::zip_equal(resultTypes, results)) {
      propagateIfChanged(lattice,
                         lattice->join(InferTypeLatticeStorage(result)));
    }
    return mlir::success();
  }

  void visitNonControlFlowArguments(mlir::Operation *op,
                                    const mlir::RegionSuccessor &successor,
                                    llvm::ArrayRef<InferTypeLattice *> lattices,
                                    unsigned firstIndex) override {
    auto iterateOp = llvm::dyn_cast<wave::IterateOp>(op);
    if (!iterateOp)
      return;

    // Technically, the non-captured arguments can be seen as forwarded from
    // operands or results, but they need special handling to remove
    // loop-specific parts of the index.
    assert(firstIndex == 0 &&
           "expected all arguments to be marked as non-control flow");
    assert((successor.isParent() ||
            successor.getSuccessor()->getRegionNumber() == 0) &&
           "unexpected control flow");

    auto yieldOp =
        llvm::cast<wave::YieldOp>(iterateOp.getLoopBody()->getTerminator());
    if (successor.getSuccessor()) {
      // When successor is the body region, propagate induction variable
      // lattices from their initial and yielded values.
      for (auto &&[terminatorOperand, iterArg, lattice] : llvm::zip_equal(
               yieldOp.getOperands(), iterateOp.getIterArgs(),
               lattices.take_front(iterateOp.getIterArgs().size()))) {
        // Fetch the lattice and create a dependecy to re-visit the program
        // point at the start of the loop body block when the lattice changes
        // since we know we are processing a branch into the loop body. Taking
        // the program point before the block / first operation will call
        // `visitBlock` which may ultimately dispatch here.
        // TODO: it's a shame we need to dig through implementation details of
        // the analysis framework to understand what needs to be done.
        const InferTypeLattice *iterArgLattice = getLatticeElementFor(
            getProgramPointBefore(iterateOp.getLoopBody()), iterArg);
        const InferTypeLattice *terminatorOperandLattice = getLatticeElementFor(
            getProgramPointBefore(iterateOp.getLoopBody()), terminatorOperand);
        mlir::ChangeResult changed = lattice->join(iterArgLattice->getValue());
        changed |= lattice->join(terminatorOperandLattice->getValue());
        propagateIfChanged(lattice, changed);
      }
    } else {
      // When successor is the iterate op itself, propagate lattices from iter
      // args and terminator operands to results while removing loop-specific
      // parts of the index.
      for (auto &&[terminatorOperand, iterArg, resultLattice] : llvm::zip_equal(
               yieldOp.getOperands(), iterateOp.getIterArgs(), lattices)) {
        // Fetch the lattice and create a dependency to re-visit the program
        // point after the iterate op when the lattice changes since we know we
        // are processing a branch back to the iterate op itself. Taking the
        // program point after the operation will call `visitOperation` on the
        // operation itself, which will ultimately dispatch here.
        // TODO: same as above, we shouldn't need to care about such
        // implementation details.
        const InferTypeLattice *terminatorOperandLattice = getLatticeElementFor(
            getProgramPointAfter(iterateOp), terminatorOperand);
        const InferTypeLattice *iterArgLattice =
            getLatticeElementFor(getProgramPointAfter(iterateOp), iterArg);
        mlir::ChangeResult changed =
            resultLattice->join(iterArgLattice->getValue());
        changed |= resultLattice->join(terminatorOperandLattice->getValue());
        propagateIfChanged(resultLattice, changed);
      }
    }
  }

private:
  // Initialize the lattice instance for the given value to its current type and
  // trigger dataflow propagation. Returns the lattice instance or null if the
  // value is not of wave tensor type.
  InferTypeLattice *initForValue(mlir::Value value) {
    auto tensorType = llvm::dyn_cast<wave::WaveTensorType>(value.getType());
    if (!tensorType)
      return nullptr;
    InferTypeLattice *lattice = getLatticeElement(value);
    lattice->getValue().unsafeSet(InferTypeLatticeStorage(tensorType));
    propagateIfChanged(lattice, mlir::ChangeResult::Change);
    return lattice;
  }

  // Initialize lattice instances for results of the given op.
  void initForResults(mlir::Operation *op) {
    for (mlir::Value result : op->getResults())
      initForValue(result);
  }

  // Initialize lattice instances for block arguments of the given block.
  void initForBlockArguments(mlir::Block &block) {
    for (mlir::Value arg : block.getArguments())
      initForValue(arg);
  }
};

// Dataflow analysis propagating type/shape information from results back to
// operands. This is an optimistic sparse context-insensitive backward dataflow
// analysis intended for intra-procedural use and composition with the
// equivalent forward analysis. It starts by initializing a lattice instance
// for all wave tensor-typed operands to their current type and propagates shape
// information across operations using WaveInferTypeOpInterface as well as
// across control flow using regular control flow interfaces until convergence.
// In case of a type inference conflict, e.g. the pre-existing type is different
// from the type inferred by dataflow, a diagnostic is emitted and the lattice
// instance corresponding to the value with type conflict is set to the top
// state. If the analysis fails due to the lack of information, e.g., control
// flow operations not implementing the requested interfaces, the lattice
// instances may be set to the top state without diagnostics.
class InferTypeBackwardAnalysis
    : public mlir::dataflow::SparseBackwardDataFlowAnalysis<InferTypeLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  // Forcibly initializes lattice instances for wave tensor-typed function
  // terminator operands to their current type.
  //
  // XXX: this works since we never revisit the function exit blocks again in
  // the intra-procedural case. Without this function, the framework would
  // call `setToEntryState` that would trigger setting to lattice top which we
  // don't want.
  mlir::LogicalResult initialize(mlir::Operation *top) override {
    if (getSolverConfig().isInterprocedural())
      return top->emitError() << "interprocedural analysis not supported";

    // Call the base class initialization in order to set up update listeners.
    // Note that this will initialize values at function/region entries to
    // lattice top.
    if (mlir::failed(SparseBackwardDataFlowAnalysis::initialize(top)))
      return mlir::failure();

    top->walk([this](mlir::Operation *op) {
      if (!op->hasTrait<mlir::OpTrait::ReturnLike>())
        return;
      if (!llvm::isa<mlir::FunctionOpInterface>(op->getParentOp()))
        return;
      for (mlir::Value operand : op->getOperands()) {
        auto tensorType =
            llvm::dyn_cast<wave::WaveTensorType>(operand.getType());
        if (!tensorType)
          continue;
        InferTypeLattice *lattice = getLatticeElement(operand);
        lattice->getValue().unsafeSet(InferTypeLatticeStorage(tensorType));
      }
    });
    return mlir::success();
  }

  // Called by base class initialization and when the analysis fails to identify
  // lattices to join.
  void setToExitState(InferTypeLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(InferTypeLatticeStorage::top()));
  }

  // Dataflow transfer function, defers to the WaveInferTypeOpInterface.
  mlir::LogicalResult
  visitOperation(mlir::Operation *op,
                 llvm::ArrayRef<InferTypeLattice *> operands,
                 llvm::ArrayRef<const InferTypeLattice *> results) override {
    std::optional<mlir::LogicalResult> res = handleNonInterfaceOpInferType(op);
    if (res)
      return *res;

    auto extractType = [](const InferTypeLattice *lattice) {
      return lattice->getValue().getConcreteValue();
    };
    llvm::SmallVector<wave::WaveTensorType> operandTypes =
        llvm::map_to_vector(operands, extractType);
    llvm::SmallVector<wave::WaveTensorType> resultTypes =
        llvm::map_to_vector(results, extractType);

    std::string errorMessage;
    llvm::raw_string_ostream errs(errorMessage);
    llvm::FailureOr<mlir::ChangeResult> result =
        llvm::cast<wave::WaveInferTypeOpInterface>(op).propagateBackward(
            operandTypes, resultTypes, errs);
    if (mlir::failed(result)) {
      return op->emitError()
             << "failed to propagate type information backward: " << errs.str();
    }
    if (*result == mlir::ChangeResult::NoChange)
      return mlir::success();

    for (auto &&[operand, lattice] : llvm::zip_equal(operandTypes, operands)) {
      propagateIfChanged(lattice,
                         lattice->join(InferTypeLatticeStorage(operand)));
    }
    return mlir::success();
  }

  // Specialization of the dataflow transfer function for control flow branch
  // operation that are not forwarded to the branching target. Wave::IterateOp
  // doesn't use the interface's capability for 1:1 lattice forwarding because
  // lattices of other analyses change along this dataflow path. Propagate
  // lattices manually here. For other ops, we do not expect this to happen so
  // move the lattice instance to the top state, which will result in a type
  // conflict diagnostic if this ever happens.
  void visitBranchOperand(mlir::OpOperand &opOperand) override {
    auto tensorType =
        llvm::dyn_cast<wave::WaveTensorType>(opOperand.get().getType());
    if (!tensorType)
      return;

    if (auto iterateOp =
            llvm::dyn_cast<wave::IterateOp>(opOperand.getOwner())) {
      unsigned position = opOperand.getOperandNumber();
      mlir::Value blockArgument =
          iterateOp.getLoopBody()->getArgument(position);
      const InferTypeLattice *blockArgLattice =
          getLatticeElement(blockArgument);
      InferTypeLattice *lattice = getLatticeElement(opOperand.get());
      // Manually add a dependency because `getLatticeElementFor` is not exposed
      // upstream for backwards analyses. Using the `after` point because the
      // logic of the analysis will re-visit the previous operation of the point
      // (the point itself being between operations), and we want the iterateOp
      // to be revisited since it is the one we are processing here.
      // TODO: consider having a `getCurrentProgramPoint` upstream so we don't
      // have to care about these arguably implementation detail.
      addDependency(const_cast<InferTypeLattice *>(blockArgLattice),
                    getProgramPointAfter(iterateOp));
      propagateIfChanged(lattice, lattice->join(blockArgLattice->getValue()));
      return;
    }

    if (auto yieldOp = llvm::dyn_cast<wave::YieldOp>(opOperand.getOwner())) {
      unsigned position = opOperand.getOperandNumber();
      mlir::Value result = yieldOp->getParentOp()->getResult(position);
      const InferTypeLattice *resultLattice = getLatticeElement(result);
      InferTypeLattice *lattice = getLatticeElement(opOperand.get());
      // Same as for the `addDependency` above.
      addDependency(const_cast<InferTypeLattice *>(resultLattice),
                    getProgramPointAfter(yieldOp));
      propagateIfChanged(lattice, lattice->join(resultLattice->getValue()));
      return;
    }

    InferTypeLattice *lattice = getLatticeElement(opOperand.get());
    propagateIfChanged(lattice, lattice->join(InferTypeLatticeStorage::top()));
  }

  // Specialization of the dataflow transfer function for call operands that are
  // not forwarded to the callee. We expect types to be fully specified at the
  // function boundary so just set to the type.
  void visitCallOperand(mlir::OpOperand &opOperand) override {
    auto tensorType =
        llvm::dyn_cast<wave::WaveTensorType>(opOperand.get().getType());
    if (!tensorType)
      return;
    assert(tensorType.getFullySpecified() &&
           "expected fully-specified types at the call boundary");
    InferTypeLattice *lattice = getLatticeElement(opOperand.get());
    propagateIfChanged(lattice,
                       lattice->join(InferTypeLatticeStorage(tensorType)));
  }
};
} // namespace

// Run the dataflow analyses and capture whether some diagnostics were emitted.
// Only emit a generic diagnostic if no more specific diagnostic was emitted.
// This is usually indicative of some deep internal problem in the dataflow
// solver.
llvm::LogicalResult
wave::runSolverAndCaptureErrors(mlir::DataFlowSolver &solver,
                                mlir::Operation *root, bool force) {
  bool emittedError = false;
  mlir::DiagnosticEngine::HandlerID handlerID =
      root->getContext()->getDiagEngine().registerHandler(
          [&](mlir::Diagnostic &diag) {
            if (diag.getSeverity() == mlir::DiagnosticSeverity::Error)
              emittedError = true;

            // Returning failure indicates that the diagnostic wan't handled
            // and it is forwarded to other registered handlers.
            return mlir::failure();
          });
  if (mlir::failed(solver.initializeAndRun(root))) {
    if (!emittedError)
      root->emitError() << "dataflow analysis failed";
    if (!force)
      return llvm::failure();
  }
  root->getContext()->getDiagEngine().eraseHandler(handlerID);
  return llvm::success();
}

// Walk over all value definitions (op results and block arguments) and directly
// set their types using the provided callback. Report error and stop if
// any type failed to infer. Inferred types are supposed to still be accepted by
// the op verifiers that will normally run after the pass.
static llvm::LogicalResult updateValueTypes(
    mlir::Operation *root,
    llvm::function_ref<llvm::LogicalResult(mlir::Value, llvm::StringRef)>
        updateType) {
  mlir::WalkResult walkResult = root->walk([&](mlir::Operation *op) {
    for (mlir::OpResult res : op->getResults()) {
      if (mlir::failed(updateType(
              res, "result #" + std::to_string(res.getResultNumber()))))
        return mlir::WalkResult::interrupt();
    }

    for (mlir::Region &region : op->getRegions()) {
      for (auto &&[blockNumber, block] : llvm::enumerate(region)) {
        for (mlir::BlockArgument arg : block.getArguments()) {
          auto fmt = llvm::formatv("argument #{0} of block #{1} in region #{2}",
                                   arg.getArgNumber(), blockNumber,
                                   region.getRegionNumber());
          if (mlir::failed(updateType(arg, fmt.str())))
            return mlir::WalkResult::interrupt();
        }
      }
    }

    return mlir::WalkResult::advance();
  });

  return llvm::failure(walkResult.wasInterrupted());
}

namespace {
// Type inference pass implementation.
class InferTypes : public wave::impl::WaterWaveInferTypesPassBase<InferTypes> {
public:
  using WaterWaveInferTypesPassBase::WaterWaveInferTypesPassBase;

  void runOnOperation() override {
    if (llvm::failed(verifyNormalFormPassPrecondition(
            wave::WaveNormalForm::FunctionBoundarySpecified, getOperation(),
            getArgument())))
      return signalPassFailure();

    // Configure the analyses. The dead code and SCP analyses are required by
    // the logic of the solver currently.
    mlir::SymbolTableCollection symbolTable;
    mlir::DataFlowConfig dataFlowConfig;
    dataFlowConfig.setInterprocedural(false);
    mlir::DataFlowSolver solver(dataFlowConfig);
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<mlir::dataflow::SparseConstantPropagation>();
    solver.load<InferTypeForwardAnalysis>();
    solver.load<InferTypeBackwardAnalysis>(symbolTable);
    mlir::Operation *root = getOperation();

    if (llvm::failed(wave::runSolverAndCaptureErrors(solver, root, force)))
      return signalPassFailure();

    // Update the type of the value given the lattice. Don't return failure
    // after emitting error if force-processing is requested.
    auto updateType = [&](mlir::Value value, llvm::StringRef description) {
      if (!llvm::isa<wave::WaveTensorType>(value.getType()))
        return mlir::success();

      auto *lattice = solver.lookupState<InferTypeLattice>(value);
      if (!lattice || lattice->getValue().isBottom()) {
        emitError(value.getLoc()) << "couldn't infer type for " << description;
        return mlir::failure(!force);
      }
      if (lattice->getValue().isTop()) {
        emitError(value.getLoc())
            << "type conflict was detected for " << description;
        return mlir::failure(!force);
      }

      value.setType(lattice->getValue().getConcreteValue());
      return mlir::success();
    };

    if (llvm::failed(updateValueTypes(getOperation(), updateType)))
      return signalPassFailure();

    llvm::LogicalResult result = setNormalFormPassPostcondition(
        wave::WaveNormalForm::AllTypesSpecified, getOperation());
    if (llvm::failed(result) && !force) {
      emitError(getOperation()->getLoc())
          << "failed to produce code with the expected normal form";
      return signalPassFailure();
    }
  }
};

//-----------------------------------------------------------------------------
// WaveInferTypeOpInterface and implementation traits
//-----------------------------------------------------------------------------

// Typed lattice object for elements-per-thread dataflow propagation.
class ElementsPerThreadLattice
    : public mlir::dataflow::Lattice<ElementsPerThreadLatticeValue> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ElementsPerThreadLattice);
  using Lattice::Lattice;
};

// Helper function for forward/backward inference handling ops that do not
// implement the elements per thread propagation interface. Returns success if
// the op doesn't manipulate register-resident wave tensor types, failure
// otherwise. Returns nullopt if the op does implement the interface.
static std::optional<llvm::LogicalResult>
handleNonInterfaceOpElementsPerThread(mlir::Operation *op) {
  if (llvm::isa<wave::WaveElementsPerThreadOpInterface>(op))
    return std::nullopt;

  if (llvm::none_of(op->getOperandTypes(), wave::isaTensorInRegister) &&
      llvm::none_of(op->getResultTypes(), wave::isaTensorInRegister))
    return llvm::success();

  return op->emitError()
         << "cannot propagate elements per thread information across an "
            "operation not implementing the corresponding interface";
}

// Dataflow analysis propagating elements-per-thread information from operands
// to results. This is an optimistic sparse context-insensitive forward dataflow
// analysis intended for intra-procedural use and composition with the
// equivalent backward analysis. It propagates information based on per-op
// implementations of WaveElementsPerThreadOpInterface,
// NoOpElementsPerThreadOpTrait as well as regular control flow interfaces until
// convergence. In case of a conflict, e.g., the same value is used with
// different number of elements per thread in two different IR contexts, reports
// a diagnostic and sets the value lattice to the top state. If the analysis
// fails due to the lack of information, e.g., control flow operations not
// implementing the requested interfaces, the lattice instances may be set to
// the top state without diagnostics. If insufficient information is available
// in the IR, e.g., memory-related operations do not provide an explicit number
// of elements per thread and there is no context allowing to infer them, the
// lattice value remains set to bottom.
class ElementsPerThreadForwardAnalysis
    : public mlir::dataflow::SparseForwardDataFlowAnalysis<
          ElementsPerThreadLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  // Basic initialization and configuration filtering.
  mlir::LogicalResult initialize(mlir::Operation *top) override {
    if (getSolverConfig().isInterprocedural())
      return top->emitError() << "interprocedural analysis not supported";

    // Call the base class initialization in order to set up update listeners.
    // Note that this will initialize values at function/region entries to
    // lattice top.
    if (mlir::failed(AbstractSparseForwardDataFlowAnalysis::initialize(top)))
      return mlir::failure();

    return mlir::success();
  }

  // Called by base class initialization and when the analysis fails to identify
  // lattices to join.
  void setToEntryState(ElementsPerThreadLattice *lattice) override {
    propagateIfChanged(lattice,
                       lattice->join(ElementsPerThreadLatticeValue::top()));
  }

  // Dataflow transfer function, defers to either
  // WaveElementsPerThreadOpInterface or NoOpElementsPerThreadOpTrait.
  mlir::LogicalResult
  visitOperation(mlir::Operation *op,
                 llvm::ArrayRef<const ElementsPerThreadLattice *> operands,
                 llvm::ArrayRef<ElementsPerThreadLattice *> results) override {
    if (op->hasTrait<wave::NoOpElementsPerThreadOpTrait>())
      return llvm::success();

    std::optional<mlir::LogicalResult> res =
        handleNonInterfaceOpElementsPerThread(op);
    if (res)
      return *res;

    auto extractValue = [](const ElementsPerThreadLattice *lattice) {
      return lattice->getValue();
    };
    llvm::SmallVector<ElementsPerThreadLatticeValue> operandElements =
        llvm::map_to_vector(operands, extractValue);
    llvm::SmallVector<ElementsPerThreadLatticeValue> resultElements =
        llvm::map_to_vector(results, extractValue);

    std::string errorMessage;
    llvm::raw_string_ostream errs(errorMessage);
    llvm::FailureOr<mlir::ChangeResult> result =
        llvm::cast<wave::WaveElementsPerThreadOpInterface>(op)
            .propagateElementsPerThreadForward(operandElements, resultElements,
                                               errs);
    if (llvm::failed(result)) {
      return op->emitError()
             << "failed to propagate elements per thread forward: "
             << errs.str();
    }
    if (*result == mlir::ChangeResult::NoChange)
      return mlir::success();

    for (auto &&[result, lattice] : llvm::zip_equal(resultElements, results)) {
      propagateIfChanged(lattice,
                         lattice->join(ElementsPerThreadLatticeValue(result)));
    }
    return mlir::success();
  }

  void visitNonControlFlowArguments(
      mlir::Operation *op, const mlir::RegionSuccessor &successor,
      llvm::ArrayRef<ElementsPerThreadLattice *> lattices,
      unsigned firstIndex) override {
    auto iterateOp = llvm::dyn_cast<wave::IterateOp>(op);
    if (!iterateOp)
      return;

    // Technically, the non-captured arguments can be seen as forwarded from
    // operands or results, but they need special handling to remove
    // loop-specific parts of the index.
    assert(firstIndex == 0 &&
           "expected all arguments to be marked as non-control flow");
    assert((successor.isParent() ||
            successor.getSuccessor()->getRegionNumber() == 0) &&
           "unexpected control flow");

    auto yieldOp =
        llvm::cast<wave::YieldOp>(iterateOp.getLoopBody()->getTerminator());
    if (successor.getSuccessor()) {
      // When successor is the body region, propagate induction variable
      // lattices from their initial and yielded values.
      for (auto &&[terminatorOperand, iterArg, lattice] : llvm::zip_equal(
               yieldOp.getOperands(), iterateOp.getIterArgs(),
               lattices.take_front(iterateOp.getIterArgs().size()))) {
        // Fetch the lattice and create a dependency to re-visit the program
        // point at the start of the loop body block when the lattice changes
        // since we know we are processing a branch into the loop body. Taking
        // the program point before the block / first operation will call
        // `visitBlock` which may ultimately dispatch here.
        const ElementsPerThreadLattice *iterArgLattice = getLatticeElementFor(
            getProgramPointBefore(iterateOp.getLoopBody()), iterArg);
        const ElementsPerThreadLattice *terminatorOperandLattice =
            getLatticeElementFor(getProgramPointBefore(iterateOp.getLoopBody()),
                                 terminatorOperand);
        mlir::ChangeResult changed = lattice->join(iterArgLattice->getValue());
        changed |= lattice->join(terminatorOperandLattice->getValue());
        propagateIfChanged(lattice, changed);
      }
    } else {
      // When successor is the iterate op itself, propagate lattices from iter
      // args and terminator operands to results while removing loop-specific
      // parts of the index.
      for (auto &&[terminatorOperand, iterArg, resultLattice] : llvm::zip_equal(
               yieldOp.getOperands(), iterateOp.getIterArgs(), lattices)) {
        // Fetch the lattice and create a dependency to re-visit the program
        // point after the iterate op when the lattice changes since we know we
        // are processing a branch back to the iterate op itself. Taking the
        // program point after the operation will call `visitOperation` on the
        // operation itself, which will ultimately dispatch here.
        const ElementsPerThreadLattice *terminatorOperandLattice =
            getLatticeElementFor(getProgramPointAfter(iterateOp),
                                 terminatorOperand);
        const ElementsPerThreadLattice *iterArgLattice =
            getLatticeElementFor(getProgramPointAfter(iterateOp), iterArg);
        mlir::ChangeResult changed =
            resultLattice->join(iterArgLattice->getValue());
        changed |= resultLattice->join(terminatorOperandLattice->getValue());
        propagateIfChanged(resultLattice, changed);
      }
    }
  }
};

// Dataflow analysis propagating elements-per-thread information from results
// to operands. This is an optimistic sparse context-insensitive forward
// dataflow analysis intended for intra-procedural use and composition with the
// equivalent backward analysis. It propagates information based on per-op
// implementations of WaveElementsPerThreadOpInterface,
// NoOpElementsPerThreadOpTrait as well as regular control flow interfaces until
// convergence. In case of a conflict, e.g., the same value is used with
// different number of elements per thread in two different IR contexts, reports
// a diagnostic and sets the value lattice to the top state. If the analysis
// fails due to the lack of information, e.g., control flow operations not
// implementing the requested interfaces, the lattice instances may be set to
// the top state without diagnostics. If insufficient information is available
// in the IR, e.g., memory-related operations do not provide an explicit number
// of elements per thread and there is no context allowing to infer them, the
// lattice value remains set to bottom.
class ElementsPerThreadBackwardAnalysis
    : public mlir::dataflow::SparseBackwardDataFlowAnalysis<
          ElementsPerThreadLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  // Basic initialization and configuration filtering.
  mlir::LogicalResult initialize(mlir::Operation *top) override {
    assert(!getSolverConfig().isInterprocedural() &&
           "interprocedural analysis not supported");

    if (mlir::failed(SparseBackwardDataFlowAnalysis::initialize(top)))
      return mlir::failure();

    return mlir::success();
  }

  // Called by base class initialization and when the analysis fails to identify
  // lattices to join.
  void setToExitState(ElementsPerThreadLattice *lattice) override {
    propagateIfChanged(lattice,
                       lattice->join(ElementsPerThreadLatticeValue::top()));
  }

  // Specialization of the dataflow transfer function for control flow branch
  // operation that are not forwarded to the branching target. Wave::IterateOp
  // doesn't use the interface's capability for 1:1 lattice forwarding because
  // lattices of other analyses change along this dataflow path. Propagate
  // lattices manually here. For other ops, we do not expect this to happen so
  // move the lattice instance to the top state, which will result in a
  // conflict diagnostic if this ever happens.
  void visitBranchOperand(mlir::OpOperand &opOperand) override {
    if (!wave::isaTensorInRegister(opOperand.get().getType()))
      return;

    if (auto iterateOp =
            llvm::dyn_cast<wave::IterateOp>(opOperand.getOwner())) {
      unsigned position = opOperand.getOperandNumber();
      mlir::Value blockArgument =
          iterateOp.getLoopBody()->getArgument(position);
      const ElementsPerThreadLattice *blockArgLattice =
          getLatticeElement(blockArgument);
      ElementsPerThreadLattice *lattice = getLatticeElement(opOperand.get());
      // Manually add a dependency because `getLatticeElementFor` is not exposed
      // upstream for backwards analyses. Using the `after` point because the
      // logic of the analysis will re-visit the previous operation of the point
      // (the point itself being between operations), and we want the iterateOp
      // to be revisited since it is the one we are processing here.
      addDependency(const_cast<ElementsPerThreadLattice *>(blockArgLattice),
                    getProgramPointAfter(iterateOp));
      propagateIfChanged(lattice, lattice->join(blockArgLattice->getValue()));
      return;
    }

    if (auto yieldOp = llvm::dyn_cast<wave::YieldOp>(opOperand.getOwner())) {
      unsigned position = opOperand.getOperandNumber();
      mlir::Value result = yieldOp->getParentOp()->getResult(position);
      const ElementsPerThreadLattice *resultLattice = getLatticeElement(result);
      ElementsPerThreadLattice *lattice = getLatticeElement(opOperand.get());
      // Same as for the `addDependency` above.
      addDependency(const_cast<ElementsPerThreadLattice *>(resultLattice),
                    getProgramPointAfter(yieldOp));
      propagateIfChanged(lattice, lattice->join(resultLattice->getValue()));
      return;
    }

    setToExitState(getLatticeElement(opOperand.get()));
  }

  // Specialization of the dataflow transfer function for call operands that are
  // not forwarded to the callee. We do not expect register-resident types
  // handled by this analysis to be present at the function boundary so we move
  // the lattice instance to the top state, indicating a conflict if this ever
  // happens.
  void visitCallOperand(mlir::OpOperand &opOperand) override {
    if (!wave::isaTensorInRegister(opOperand.get().getType()))
      return;

    setToExitState(getLatticeElement(opOperand.get()));
  }

  // Dataflow transfer function, defers to either
  // WaveElementsPerThreadOpInterface or NoOpElementsPerThreadOpTrait.
  llvm::LogicalResult visitOperation(
      mlir::Operation *op, llvm::ArrayRef<ElementsPerThreadLattice *> operands,
      llvm::ArrayRef<const ElementsPerThreadLattice *> results) override {
    if (op->hasTrait<wave::NoOpElementsPerThreadOpTrait>())
      return llvm::success();

    std::optional<mlir::LogicalResult> res =
        handleNonInterfaceOpElementsPerThread(op);
    if (res)
      return *res;

    auto extractValue = [](const ElementsPerThreadLattice *lattice) {
      return lattice->getValue();
    };
    llvm::SmallVector<ElementsPerThreadLatticeValue> operandElements =
        llvm::map_to_vector(operands, extractValue);
    llvm::SmallVector<ElementsPerThreadLatticeValue> resultElements =
        llvm::map_to_vector(results, extractValue);

    std::string errorMessage;
    llvm::raw_string_ostream errs(errorMessage);
    llvm::FailureOr<mlir::ChangeResult> result =
        llvm::cast<wave::WaveElementsPerThreadOpInterface>(op)
            .propagateElementsPerThreadBackward(operandElements, resultElements,
                                                errs);
    if (llvm::failed(result)) {
      return op->emitError()
             << "failed to propagate elements per thread backward: "
             << errs.str();
    }
    if (*result == mlir::ChangeResult::NoChange)
      return llvm::success();

    for (auto &&[operand, lattice] :
         llvm::zip_equal(operandElements, operands)) {
      propagateIfChanged(lattice,
                         lattice->join(ElementsPerThreadLatticeValue(operand)));
    }
    return llvm::success();
  }
};

// Elements-per-thread propagation pass implementation.
class PropagateElementsPerThread
    : public wave::impl::WaterWavePropagateElementsPerThreadPassBase<
          PropagateElementsPerThread> {
public:
  using WaterWavePropagateElementsPerThreadPassBase::
      WaterWavePropagateElementsPerThreadPassBase;

  void runOnOperation() override {
    // Configure the analyses. The dead code and SCP analyses are required by
    // the logic of the solver currently.
    mlir::SymbolTableCollection symbolTable;
    mlir::DataFlowConfig dataFlowConfig;
    dataFlowConfig.setInterprocedural(false);
    mlir::DataFlowSolver solver(dataFlowConfig);
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<mlir::dataflow::SparseConstantPropagation>();
    solver.load<ElementsPerThreadForwardAnalysis>();
    solver.load<ElementsPerThreadBackwardAnalysis>(symbolTable);

    if (llvm::failed(
            wave::runSolverAndCaptureErrors(solver, getOperation(), false)))
      return signalPassFailure();

    auto updateType = [&](mlir::Value value, llvm::StringRef description) {
      auto tensorType = llvm::dyn_cast<wave::WaveTensorType>(value.getType());
      if (!tensorType ||
          tensorType.getAddressSpaceValue() != wave::WaveAddressSpace::Register)
        return llvm::success();

      const auto *lattice = solver.lookupState<ElementsPerThreadLattice>(value);
      if (!lattice || lattice->getValue().isBottom()) {
        emitError(value.getLoc())
            << "couldn't identify elements per thread for " << description;
        return llvm::failure();
      }
      if (lattice->getValue().isTop()) {
        emitError(value.getLoc())
            << "elements per thread conflict was detected for " << description;
        return llvm::failure();
      }

      auto vectorType = mlir::VectorType::get(
          {static_cast<int64_t>(lattice->getValue().getValue())},
          tensorType.getElementType());
      value.setType(vectorType);
      return llvm::success();
    };

    if (llvm::failed(updateValueTypes(getOperation(), updateType)))
      return signalPassFailure();

    if (llvm::failed(wave::setNormalFormPassPostcondition(
            wave::WaveNormalForm::MemoryOnlyTypes, getOperation())))
      return signalPassFailure();
  }
};
} // namespace

// Lattice object for index expressions for analysis compatibility, the actual
// logic is in the IndexExprsLatticeStorage class.
class IndexExprsLattice
    : public mlir::dataflow::Lattice<IndexExprsLatticeStorage> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IndexExprsLattice);
  using Lattice::Lattice;
};

namespace {
// Wrapper to print operations without regions. Use as `llvm::outs() <<
// PrintNoRegions(op)`.
class PrintNoRegions {
public:
  PrintNoRegions(mlir::Operation *op) : operation(op) {}

  void print(llvm::raw_ostream &os) const {
    if (!operation) {
      os << "<null>";
      return;
    }
    operation->print(os, mlir::OpPrintingFlags().skipRegions());
  }

private:
  mlir::Operation *operation;
};

} // namespace

// Support operator<< for OperationPrinterWithoutRegions.
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const PrintNoRegions &printer) {
  printer.print(os);
  return os;
}

// Helper function to walk the IR and collect wave constraints attributes.
static llvm::LogicalResult collectWaveConstraints(
    mlir::Operation *top,
    llvm::DenseMap<mlir::Operation *, mlir::Attribute> &constraints) {
  auto *waveDialect = top->getContext()->getLoadedDialect<wave::WaveDialect>();
  auto walkResult =
      top->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
        if (auto attr = op->getAttrOfType<mlir::ArrayAttr>(
                wave::WaveDialect::kWaveConstraintsAttrName)) {
          constraints[op] = attr;
          return mlir::WalkResult::skip();
        }
        if (op->getDialect() == waveDialect) {
          op->emitError()
              << "wave dialect operation without constraints on an ancestor";
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
  if (walkResult.wasInterrupted())
    return llvm::failure();
  return llvm::success();
}

class IndexExprsForwardAnalysis
    : public mlir::dataflow::SparseForwardDataFlowAnalysis<IndexExprsLattice> {
private:
  void unsafeSet(IndexExprsLattice *lattice, IndexExprsLatticeStorage value) {
    if (lattice->getValue() == value)
      return;
    lattice->getValue().unsafeSet(value);
    propagateIfChanged(lattice, mlir::ChangeResult::Change);
  }

public:
  explicit IndexExprsForwardAnalysis(
      mlir::DataFlowSolver &solver,
      wave::OverrideInitializationFn overrideInitialization = nullptr)
      : SparseForwardDataFlowAnalysis(solver),
        overrideInitialization(overrideInitialization) {}

  mlir::LogicalResult initialize(mlir::Operation *top) override {
    assert(!getSolverConfig().isInterprocedural() &&
           "interprocedural analysis not supported");

    // Call the base class initialization in order to set up update listeners.
    // Note that this will initialize values at function/region entries to
    // lattice top.
    if (mlir::failed(SparseForwardDataFlowAnalysis::initialize(top)))
      return mlir::failure();

    llvm::DenseMap<mlir::Operation *, mlir::Attribute> constraints;
    if (llvm::failed(collectWaveConstraints(top, constraints)))
      return llvm::failure();

    for (auto &&[parent, attr] : constraints) {
      auto initObject =
          wave::IndexExprsAnalysisInit::create(parent->getLoc(), attr);
      if (llvm::failed(initObject))
        return llvm::failure();
      mlir::WalkResult walkResult =
          parent->walk([&](mlir::Operation *op) -> mlir::WalkResult {
            if (auto iface =
                    llvm::dyn_cast<wave::WaveInferIndexExprsOpInterface>(op)) {
              llvm::SmallVector<wave::IndexExprsLatticeStorage> resultExprs =
                  llvm::map_to_vector(op->getResults(), [&](mlir::Value v) {
                    return getLatticeElement(v)->getValue();
                  });
              auto emitError = [op]() { return op->emitError(); };
              if (llvm::failed(iface.initializeIndexExprsForward(
                      resultExprs, *initObject, emitError)))
                return mlir::WalkResult::interrupt();

              for (auto &&[result, lattice] :
                   llvm::zip_equal(op->getResults(), resultExprs)) {
                IndexExprsLattice *latticeObject = getLatticeElement(result);
                unsafeSet(latticeObject, lattice);
              }
            }

            // Set block arguments to bottom initially so they can be join'ed
            // with actual lattices coming from other operations.
            for (mlir::Region &region : op->getRegions()) {
              for (mlir::Block &block : region) {
                for (mlir::Value value : block.getArguments()) {
                  if (!llvm::isa<wave::WaveTensorType>(value.getType()))
                    continue;

                  LDBG() << "setting block argument lattice " << value
                         << " from " << PrintNoRegions(op) << " to bottom";
                  unsafeSet(getLatticeElement(value),
                            IndexExprsLatticeStorage::bottom());
                }
              }
            }

            if (auto iterateOp = llvm::dyn_cast<wave::IterateOp>(op)) {
              // Set lattices of captured block arguments to the relevant tiling
              // constraint, it will be then propagated by joining with
              // expressions induced by other constraints.
              wave::WaveSymbolAttr iterSymbolAttr = iterateOp.getIterator();
              llvm::SmallVector<mlir::Attribute> symbolConstraints =
                  initObject->symbolConstraints.lookup(iterSymbolAttr);
              auto it = llvm::find_if(
                  symbolConstraints, llvm::IsaPred<wave::TilingConstraintAttr>);
              if (it != symbolConstraints.end()) {
                wave::TilingConstraintAttr tilingConstraint =
                    llvm::cast<wave::TilingConstraintAttr>(*it);
                for (mlir::Value capture : iterateOp.getCaptureBlockArgs()) {
                  if (!llvm::isa<wave::WaveTensorType>(capture.getType()))
                    continue;
                  auto dict = mlir::DictionaryAttr::get(
                      iterSymbolAttr.getContext(),
                      {{iterSymbolAttr.getName(),
                        wave::applyConstraint(tilingConstraint)}});
                  LDBG() << "setting iterate block argument lattice " << capture
                         << " from " << PrintNoRegions(iterateOp) << " to "
                         << dict;
                  unsafeSet(getLatticeElement(capture), dict);
                }
              }
            }
            return llvm::success();
          });
      if (walkResult.wasInterrupted())
        return llvm::failure();
    }

    if (overrideInitialization) {
      if (llvm::failed(overrideInitialization(
              top, [&](mlir::Value value, mlir::DictionaryAttr dict) {
                if (!dict)
                  return unsafeSet(getLatticeElement(value),
                                   IndexExprsLatticeStorage::top());
                unsafeSet(getLatticeElement(value), dict);
              })))
        return llvm::failure();
    }

    initialized = true;
    return llvm::success();
  }

  void setToEntryState(IndexExprsLattice *lattice) override {
    // Default logic will call this function on arguments of a callable
    // operation since we are running in a non-interprocedural analysis. Setting
    // them to top would propagate everywhere. Instead, just do nothing here and
    // let them converge to whatever value needed by backward analysis.
    auto arg = llvm::dyn_cast<mlir::BlockArgument>(lattice->getAnchor());
    if (arg &&
        llvm::isa<mlir::CallableOpInterface>(arg.getOwner()->getParentOp())) {
      return;
    }

    // Default initialization calls `setToEntryState` on block arguments, we
    // don't want to set it to the top state because it will propagate
    // everywhere. Set/join with bottom instead so it can be overridden. Once
    // initialization is done, `setToEntryState` may be called for unanalyzable
    // cases, where we actually want to set it to the (pessimistic fixpoint) top
    // state.
    propagateIfChanged(lattice,
                       lattice->join(initialized
                                         ? IndexExprsLatticeStorage::top()
                                         : IndexExprsLatticeStorage::bottom()));
    if (initialized) {
      LDBG() << "top fixpoint for " << lattice->getAnchor() << " "
             << (arg ? PrintNoRegions(arg.getOwner()->getParentOp())
                     : PrintNoRegions(nullptr));
    }
  }

  llvm::LogicalResult
  visitOperation(mlir::Operation *op,
                 llvm::ArrayRef<const IndexExprsLattice *> operands,
                 llvm::ArrayRef<IndexExprsLattice *> results) override {

    LLVM_DEBUG({
      LDBG() << "visiting operation forward " << PrintNoRegions(op);
      LDBG() << "  operand lattices:";
      for (auto [i, operand] : llvm::enumerate(operands)) {
        LDBG() << "    operand #" << i << ": " << *operand;
      }
      // Print all result lattices.
      LDBG() << "  result lattices:";
      for (auto [i, result] : llvm::enumerate(results)) {
        LDBG() << "    result #" << i << ": " << *result;
      }
    });
    auto scope = llvm::make_scope_exit([&] {
      LLVM_DEBUG({
        LDBG() << "  updated result lattices:";
        for (auto [i, result] : llvm::enumerate(results)) {
          LDBG() << "    result #" << i << ": " << *result;
        }
      });
    });

    // Check if the operation implements the interface.
    if (!llvm::isa<wave::WaveInferIndexExprsOpInterface>(op)) {
      // Operations without the interface should not manipulate WaveTensorType.
      if (!llvm::any_of(op->getOperandTypes(),
                        llvm::IsaPred<wave::WaveTensorType>) &&
          !llvm::any_of(op->getResultTypes(),
                        llvm::IsaPred<wave::WaveTensorType>)) {
        return llvm::success();
      }
      return op->emitError()
             << "cannot propagate index expressions across an operation not "
                "implementing the wave infer index expressions interface";
    }

    auto extractLattice = [](const IndexExprsLattice *lattice) {
      return lattice->getValue();
    };
    llvm::SmallVector<IndexExprsLatticeStorage> operandLattices =
        llvm::map_to_vector(operands, extractLattice);
    llvm::SmallVector<IndexExprsLatticeStorage> resultLattices =
        llvm::map_to_vector(results, extractLattice);

    auto reportError = [op]() { return op->emitError(); };
    llvm::FailureOr<mlir::ChangeResult> result =
        llvm::cast<wave::WaveInferIndexExprsOpInterface>(op)
            .propagateIndexExprsForward(operandLattices, resultLattices,
                                        reportError);
    if (llvm::failed(result))
      return llvm::failure();
    if (*result == mlir::ChangeResult::NoChange)
      return llvm::success();

    for (auto &&[resultLattice, lattice] :
         llvm::zip_equal(resultLattices, results)) {
      // In release mode, just set the lattice value instead of calling join.
      // The interface should have returned the correctly joined lattice and we
      // don't want to re-join it and don't need the expensive check of the
      // lattice direction.
#ifndef NDEBUG
      propagateIfChanged(lattice, lattice->join(resultLattice));
#else
      unsafeSet(lattice, resultLattice);
#endif
    }
    return llvm::success();
  }

  void
  visitNonControlFlowArguments(mlir::Operation *op,
                               const mlir::RegionSuccessor &successor,
                               llvm::ArrayRef<IndexExprsLattice *> lattices,
                               unsigned firstIndex) override {
    auto iterateOp = llvm::dyn_cast<wave::IterateOp>(op);
    if (!iterateOp)
      return;

    // Technically, the non-captured arguments can be seen as forwarded from
    // operands or results, but they need special handling to remove
    // loop-specific parts of the index.
    assert(firstIndex == 0 &&
           "expected all arguments to be marked as non-control flow");
    assert((successor.isParent() ||
            successor.getSuccessor()->getRegionNumber() == 0) &&
           "unexpected control flow");

    auto yieldOp =
        llvm::cast<wave::YieldOp>(iterateOp.getLoopBody()->getTerminator());

    LDBG() << "visiting " << PrintNoRegions(iterateOp);
    if (successor.getSuccessor()) {
      LDBG() << " propagating to region #"
             << successor.getSuccessor()->getRegionNumber();

      // When successor is the body region, propagate induction variable
      // lattices from their initial and yielded values.
      for (auto &&[terminatorOperand, iterArg, lattice] : llvm::zip_equal(
               yieldOp.getOperands(), iterateOp.getIterArgs(),
               lattices.take_front(iterateOp.getIterArgs().size()))) {
        // See comments in
        // InferTypeForwardAnalysis::visitNonControlFlowArguments.
        const IndexExprsLattice *iterArgLattice = getLatticeElementFor(
            getProgramPointBefore(iterateOp.getLoopBody()), iterArg);
        const IndexExprsLattice *terminatorOperandLattice =
            getLatticeElementFor(getProgramPointBefore(iterateOp.getLoopBody()),
                                 terminatorOperand);
        LDBG() << "iter arg lattice: " << *iterArgLattice;
        LDBG() << "terminator operand lattice: " << *terminatorOperandLattice;
        LDBG() << "block lattice: " << *lattice;
        mlir::ChangeResult changed = lattice->join(iterArgLattice->getValue());
        changed |= lattice->join(terminatorOperandLattice->getValue());
        propagateIfChanged(lattice, changed);
        LDBG() << "new block lattice: " << *lattice;
      }

      // And also propagate lattices for captured values, which only need to be
      // propagated from the initial values.
      for (auto &&[capture, lattice] : llvm::zip_equal(
               iterateOp.getCaptures(),
               lattices.take_back(iterateOp.getCaptures().size()))) {
        // See comments in
        // InferTypeForwardAnalysis::visitNonControlFlowArguments.
        const IndexExprsLattice *captureLattice = getLatticeElementFor(
            getProgramPointBefore(iterateOp.getLoopBody()), capture);

        LDBG() << "captured lattice: " << *captureLattice;
        LDBG() << "block lattice: " << *lattice;
        propagateIfChanged(lattice, lattice->join(captureLattice->getValue()));
        LDBG() << "new block lattice: " << *lattice;
      }
    } else {
      LDBG() << "propagating to parent";

      // When successor is the iterate op itself, propagate lattices from iter
      // args and terminator operands to results while removing loop-specific
      // parts of the index.
      for (auto &&[terminatorOperand, iterArg, resultLattice] : llvm::zip_equal(
               yieldOp.getOperands(), iterateOp.getIterArgs(), lattices)) {
        // See comments in
        // InferTypeForwardAnalysis::visitNonControlFlowArguments.
        const IndexExprsLattice *terminatorOperandLattice =
            getLatticeElementFor(getProgramPointAfter(iterateOp),
                                 terminatorOperand);
        const IndexExprsLattice *iterArgLattice =
            getLatticeElementFor(getProgramPointAfter(iterateOp), iterArg);
        LDBG() << "iter arg lattice: " << *iterArgLattice;
        LDBG() << "terminator operand lattice: " << *terminatorOperandLattice;
        LDBG() << "result lattice: " << *resultLattice;
        mlir::ChangeResult changed =
            resultLattice->join(iterArgLattice->getValue());
        changed |= resultLattice->join(
            terminatorOperandLattice->getValue().withoutIterSymbols(
                iterateOp.getIterator()));
        propagateIfChanged(resultLattice, changed);
        LDBG() << "new result lattice: " << *resultLattice;
      }
    }
  }

private:
  bool initialized = false;
  wave::OverrideInitializationFn overrideInitialization;
};

class IndexExprsBackwardAnalysis
    : public mlir::dataflow::SparseBackwardDataFlowAnalysis<IndexExprsLattice> {
private:
  void unsafeSet(IndexExprsLattice *lattice, IndexExprsLatticeStorage value) {
    if (lattice->getValue() == value)
      return;
    lattice->getValue().unsafeSet(value);
    propagateIfChanged(lattice, mlir::ChangeResult::Change);
  }

public:
  IndexExprsBackwardAnalysis(
      mlir::DataFlowSolver &solver, mlir::SymbolTableCollection &symbolTable,
      wave::OverrideInitializationFn overrideInitialization = nullptr)
      : SparseBackwardDataFlowAnalysis(solver, symbolTable),
        overrideInitialization(overrideInitialization) {}

  llvm::LogicalResult initialize(mlir::Operation *top) override {
    assert(!getSolverConfig().isInterprocedural() &&
           "interprocedural analysis not supported");

    // Call the base class initialization in order to set up update listeners.
    // Note that this will initialize values at function/region entries to
    // lattice top.
    if (llvm::failed(SparseBackwardDataFlowAnalysis::initialize(top)))
      return llvm::failure();

    llvm::DenseMap<mlir::Operation *, mlir::Attribute> constraints;
    if (llvm::failed(collectWaveConstraints(top, constraints)))
      return llvm::failure();
    for (auto &&[parent, attr] : constraints) {
      auto initObject =
          wave::IndexExprsAnalysisInit::create(parent->getLoc(), attr);
      if (llvm::failed(initObject))
        return llvm::failure();

      parent->walk([&](mlir::Operation *op) -> mlir::WalkResult {
        if (auto iface =
                llvm::dyn_cast<wave::WaveInferIndexExprsOpInterface>(op)) {
          llvm::SmallVector<wave::IndexExprsLatticeStorage> operandExprs =
              llvm::map_to_vector(op->getOperands(), [&](mlir::Value v) {
                return getLatticeElement(v)->getValue();
              });
          auto emitError = [op]() { return op->emitError(); };

          if (llvm::failed(iface.initializeIndexExprsBackward(
                  operandExprs, *initObject, emitError)))
            return mlir::WalkResult::interrupt();
          for (auto &&[operand, lattice] :
               llvm::zip_equal(op->getOperands(), operandExprs)) {
            IndexExprsLattice *latticeObject = getLatticeElement(operand);
            unsafeSet(latticeObject, lattice);
          }
          return mlir::WalkResult::advance();
        } else if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
          // Set terminator operands to bottom initially so they can be join'ed
          // with actual lattices coming from other operations.
          for (mlir::Value operand : op->getOperands()) {
            if (!llvm::isa<wave::WaveTensorType>(operand.getType()))
              continue;
            unsafeSet(getLatticeElement(operand),
                      IndexExprsLatticeStorage::bottom());
          }
        }

        return mlir::WalkResult::advance();
      });
    }

    if (overrideInitialization) {
      if (llvm::failed(overrideInitialization(
              top, [&](mlir::Value value, mlir::DictionaryAttr dict) {
                if (!dict)
                  return unsafeSet(getLatticeElement(value),
                                   IndexExprsLatticeStorage::top());
                unsafeSet(getLatticeElement(value), dict);
              })))
        return llvm::failure();
    }

    initialized = true;
    return llvm::success();
  }

  void visitBranchOperand(mlir::OpOperand &opOperand) override {
    if (!llvm::isa<wave::WaveTensorType>(opOperand.get().getType()))
      return;

    // Captures of the iterate need to be propagated from the corresponding
    // block arguments manually without the tiling constraint.
    if (auto iterateOp =
            llvm::dyn_cast<wave::IterateOp>(opOperand.getOwner())) {
      unsigned position = opOperand.getOperandNumber();
      mlir::Value blockArgument =
          iterateOp.getLoopBody()->getArgument(position);
      IndexExprsLattice *blockArgLattice = getLatticeElement(blockArgument);
      IndexExprsLattice *lattice = getLatticeElement(opOperand.get());
      // Explicitly add a dependency to update this analysis at the program
      // point before the iterate if the block argument lattice changes. Using
      // localized const cast here to avoid accidentally modifying this lattice
      // in the function.
      // TODO: expose it upstream and use.
      addDependency(blockArgLattice, getProgramPointAfter(iterateOp));
      LDBG() << "propagating backwards from block argument #" << position
             << " to op operand " << PrintNoRegions(iterateOp);
      LDBG() << "block argument lattice: " << *blockArgLattice;
      LDBG() << "lattice: " << *lattice;
#ifndef NDEBUG
      propagateIfChanged(
          lattice, lattice->join(blockArgLattice->getValue().withoutIterSymbols(
                       iterateOp.getIterator())));
#else
      IndexExprsLatticeStorage joined = IndexExprsLatticeStorage::join(
          lattice->getValue(), blockArgLattice->getValue().withoutIterSymbols(
                                   iterateOp.getIterator()));
      unsafeSet(lattice, joined);
#endif
      LDBG() << "new lattice: " << *lattice;
      return;
    }

    // Terminator operands are propagated from op results as is.
    if (auto yieldOp = llvm::dyn_cast<wave::YieldOp>(opOperand.getOwner())) {
      unsigned position = opOperand.getOperandNumber();
      mlir::Value result = yieldOp->getParentOp()->getResult(position);
      const IndexExprsLattice *resultLattice = getLatticeElement(result);
      IndexExprsLattice *lattice = getLatticeElement(opOperand.get());
      // Explicitly add a dependency to update this analysis at the program
      // point before the terminator if the result lattice changes. Using
      // localized const cast here to avoid accidentally modifying this lattice.
      // TODO: expose it upstream and use.
      addDependency(const_cast<IndexExprsLattice *>(resultLattice),
                    getProgramPointAfter(yieldOp));
      LDBG() << "propagating backwards from region-carrying op result #"
             << position << " to terminator operand " << yieldOp;
      LDBG() << "result lattice: " << *resultLattice;
      LDBG() << "lattice: " << *lattice;
#ifndef NDEBUG
      propagateIfChanged(lattice, lattice->join(resultLattice->getValue()));
#else
      IndexExprsLatticeStorage joined = IndexExprsLatticeStorage::join(
          lattice->getValue(), resultLattice->getValue());
      unsafeSet(lattice, joined);
#endif
      LDBG() << "new lattice: " << *lattice;
      return;
    }

    setToExitState(getLatticeElement(opOperand.get()));
  }

  void visitCallOperand(mlir::OpOperand &opOperand) override {
    if (!llvm::isa<wave::WaveTensorType>(opOperand.get().getType()))
      return;
    setToExitState(getLatticeElement(opOperand.get()));
  }

  void setToExitState(IndexExprsLattice *lattice) override {
    // Default initialization calls `setToExitState` on terminator and call
    // operands, we don't want to set it to the top state because it will
    // propagate everywhere. Set/join with bottom instead so it can be
    // overridden. Once initialization is done, `setToExitState` may be called
    // for unanalyzable cases, where we actually want to set it to the
    // (pessimistic fixpoint) top state.
    propagateIfChanged(lattice,
                       lattice->join(initialized
                                         ? IndexExprsLatticeStorage::top()
                                         : IndexExprsLatticeStorage::bottom()));
    if (initialized)
      LDBG() << "top fixpoint (backward) for " << lattice->getAnchor();
  }

  llvm::LogicalResult
  visitOperation(mlir::Operation *op,
                 llvm::ArrayRef<IndexExprsLattice *> operands,
                 llvm::ArrayRef<const IndexExprsLattice *> results) override {
    LLVM_DEBUG({
      LDBG() << "visiting operation backward " << PrintNoRegions(op);
      LDBG() << "  operand lattices:";
      for (auto [i, operand] : llvm::enumerate(operands)) {
        LDBG() << "    operand #" << i << ": " << *operand;
      }
      LDBG() << "  results lattices:";
      for (auto [i, result] : llvm::enumerate(results)) {
        LDBG() << "    result #" << i << ": " << *result;
      }
    });
    auto scope = llvm::make_scope_exit([&] {
      LLVM_DEBUG({
        LDBG() << "  updated operand lattices:";
        for (auto [i, operand] : llvm::enumerate(operands)) {
          LDBG() << "    operand #" << i << ": " << *operand;
        }
      });
    });

    // Check if the operation implements the interface.
    if (!llvm::isa<wave::WaveInferIndexExprsOpInterface>(op)) {
      // Operations without the interface should not manipulate WaveTensorType.
      if (!llvm::any_of(op->getOperandTypes(),
                        llvm::IsaPred<wave::WaveTensorType>) &&
          !llvm::any_of(op->getResultTypes(),
                        llvm::IsaPred<wave::WaveTensorType>)) {
        return llvm::success();
      }
      return op->emitError()
             << "cannot propagate index expressions across an operation not "
                "implementing the wave infer index expressions interface";
    }

    auto extractLattice = [](const IndexExprsLattice *lattice) {
      return lattice->getValue();
    };
    llvm::SmallVector<IndexExprsLatticeStorage> operandLattices =
        llvm::map_to_vector(operands, extractLattice);
    llvm::SmallVector<IndexExprsLatticeStorage> resultLattices =
        llvm::map_to_vector(results, extractLattice);

    auto reportError = [op]() { return op->emitError(); };
    llvm::FailureOr<mlir::ChangeResult> result =
        llvm::cast<wave::WaveInferIndexExprsOpInterface>(op)
            .propagateIndexExprsBackward(operandLattices, resultLattices,
                                         reportError);
    if (llvm::failed(result))
      return llvm::failure();
    if (*result == mlir::ChangeResult::NoChange)
      return llvm::success();

    for (auto &&[operandLattice, lattice] :
         llvm::zip_equal(operandLattices, operands)) {
      // In release mode, just set the lattice value instead of calling join.
      // The interface should have returned the correctly joined lattice and we
      // don't want to re-join it and don't need the expensive check of the
      // lattice direction.
#ifndef NDEBUG
      propagateIfChanged(lattice, lattice->join(operandLattice));
#else
      unsafeSet(lattice, operandLattice);
#endif
    }
    return llvm::success();
  }

private:
  bool initialized = false;
  wave::OverrideInitializationFn overrideInitialization;
};

namespace {
class InferIndexExprsPass
    : public wave::impl::WaterWaveInferIndexExprsPassBase<InferIndexExprsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    if (llvm::failed(verifyNormalFormPassPrecondition(
            wave::WaveNormalForm::AllTypesSpecified, getOperation(),
            getArgument())))
      return signalPassFailure();

    mlir::IRRewriter rewriter(&getContext());
    getOperation()->walk(
        [&](wave::IterateOp iterateOp) { iterateOp.makeIsolated(rewriter); });

    mlir::SymbolTableCollection symbolTable;
    mlir::DataFlowConfig config;
    config.setInterprocedural(false);
    mlir::DataFlowSolver solver(config);

    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<mlir::dataflow::SparseConstantPropagation>();
    wave::addWaveIndexExprsAnalyses(solver, symbolTable);

    if (llvm::failed(
            wave::runSolverAndCaptureErrors(solver, getOperation(), false)))
      return signalPassFailure();

    if (llvm::failed(
            wave::setWaveIndexExprAnalysisResults(getOperation(), solver)))
      return signalPassFailure();

    getOperation()->walk([&](wave::IterateOp iterateOp) {
      iterateOp.makeNonIsolated(rewriter);
    });

    if (llvm::failed(wave::setNormalFormPassPostcondition(
            wave::WaveNormalForm::IndexExprsSpecified, getOperation())))
      return signalPassFailure();
  }
};
} // namespace

void wave::addWaveIndexExprsAnalyses(
    mlir::DataFlowSolver &solver, mlir::SymbolTableCollection &symbolTable,
    wave::WaveIndexExprsAnalysisOptions options) {
  if (!options.disableForward) {
    solver.load<IndexExprsForwardAnalysis>(options.overrideInitialization);
  }
  if (!options.disableBackward) {
    solver.load<IndexExprsBackwardAnalysis>(symbolTable,
                                            options.overrideInitialization);
  }
}

mlir::LogicalResult
wave::setWaveIndexExprAnalysisResults(mlir::Operation *top,
                                      const mlir::DataFlowSolver &solver) {
  mlir::WalkResult walkResult =
      top->walk([&](wave::WaveInferIndexExprsOpInterface iface) {
        auto getLatticeValue = [&](mlir::Value value) {
          auto *latticeObject = solver.lookupState<IndexExprsLattice>(value);
          return latticeObject ? latticeObject->getValue()
                               : IndexExprsLatticeStorage::bottom();
        };
        llvm::SmallVector<wave::IndexExprsLatticeStorage> operandExprs =
            llvm::map_to_vector(iface->getOperands(), getLatticeValue);
        llvm::SmallVector<wave::IndexExprsLatticeStorage> resultExprs =
            llvm::map_to_vector(iface->getResults(), getLatticeValue);

        if (llvm::failed(iface.setIndexFromLattices(operandExprs, resultExprs)))
          return mlir::WalkResult::interrupt();

        return mlir::WalkResult::advance();
      });
  return llvm::failure(walkResult.wasInterrupted());
}
