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
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/Transforms/Passes.h"
#include "water/Dialect/Wave/Transforms/Utils.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/FormatVariadic.h"
#include <type_traits>

#define DEBUG_TYPE "wave-infer-types"

using wave::ElementsPerThreadLatticeValue;

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
  // operation that are not forwarded to the branching target, so they cannot be
  // backpropagated from there. We do not expect this to happen so move the
  // lattice instance to the top state, indicating a if this ever happens.
  void visitBranchOperand(mlir::OpOperand &opOperand) override {
    auto tensorType =
        llvm::dyn_cast<wave::WaveTensorType>(opOperand.get().getType());
    if (!tensorType)
      return;
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

// Run the dataflow analyses and capture whether some diagnostics were emitted.
// Only emit a generic diagnostic if no more specific diagnostic was emitted.
// This is usually indicative of some deep internal problem in the dataflow
// solver.
static llvm::LogicalResult
runSolverAndCaptureErrors(mlir::DataFlowSolver &solver, mlir::Operation *root,
                          bool force) {
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

    if (llvm::failed(runSolverAndCaptureErrors(solver, root, force)))
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
    if (llvm::failed(result) && !force)
      return signalPassFailure();
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
    if (getSolverConfig().isInterprocedural())
      return top->emitError() << "interprocedural analysis not supported";

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
  // operation that are not forwarded to the branching target, so they cannot be
  // backpropagated from there. We do not expect this to happen so move the
  // lattice instance to the top state, indicating a conflict if this ever
  // happens.
  void visitBranchOperand(mlir::OpOperand &opOperand) override {
    if (!wave::isaTensorInRegister(opOperand.get().getType()))
      return;

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

    if (llvm::failed(runSolverAndCaptureErrors(solver, getOperation(), false)))
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

template <typename RangeT>
static void
aggregateAllSymbolNames(RangeT &&symbolNameLists,
                        llvm::SmallVectorImpl<llvm::StringRef> &symbolNames,
                        llvm::StringMap<unsigned> &symbolNamesToIdx) {
  llvm::SetVector<llvm::StringRef> allSymbolNames;
  for (auto &&symbolNameList : symbolNameLists)
    allSymbolNames.insert_range(symbolNameList);
  for (auto &&[i, symbolName] : llvm::enumerate(allSymbolNames))
    symbolNamesToIdx[symbolName] = i;
  symbolNames = allSymbolNames.takeVector();
}

static mlir::AffineMap
permuteMapSymbols(mlir::AffineMap map,
                  llvm::ArrayRef<llvm::StringRef> symbolNames,
                  llvm::ArrayRef<llvm::StringRef> allSymbolNames,
                  const llvm::StringMap<unsigned> &symbolNamesToIdx) {
  assert(map.getNumDims() == 0 && "maps should not involve dimensions");
  mlir::MLIRContext *ctx = map.getContext();
  unsigned newNumSyms = allSymbolNames.size();

  auto newSymbols = llvm::map_to_vector(symbolNames, [&](llvm::StringRef name) {
    return mlir::getAffineSymbolExpr(symbolNamesToIdx.at(name), ctx);
  });

  llvm::SmallVector<mlir::AffineExpr> remapped;
  remapped.reserve(map.getNumResults());
  for (mlir::AffineExpr expr : map.getResults())
    remapped.push_back(expr.replaceSymbols(newSymbols));

  return mlir::AffineMap::get(/*dimCount=*/0, newNumSyms, remapped, ctx);
};

// index sequence propagation:
//   initial values comes from (1) kernel-level constraints and (2) mma
//   operations (though still driven by kernel level constraints) set for all
//   mma, read/write and (?) reduction operations and then propagate do we need
//   to differentiate thread-dependent and thread-independent parts of the index
//   sequence? join is still a match/mismatch kind of operation

class IndexExprsLatticeStorage {
public:
  IndexExprsLatticeStorage() : value(nullptr, kUninitializedState) {}
  IndexExprsLatticeStorage(const IndexExprsLatticeStorage &value) = default;
  IndexExprsLatticeStorage(mlir::DictionaryAttr concreteValue)
      : value(concreteValue, kSpecificTypeState) {}

  IndexExprsLatticeStorage &
  operator=(const IndexExprsLatticeStorage &other) = default;

  bool operator==(const IndexExprsLatticeStorage &other) const {
    return value == other.value;
  }

  // Return true if this lattice instance is the bottom state.
  bool isBottom() const { return value.getInt() == kUninitializedState; }

  // Return true if this lattice instance is the top state.
  bool isTop() const { return value.getInt() == kUndecidableState; }

  // Returns the concrete value stored in the lattice instance, be it fully
  // specified or not, or null if the lattice instance is a top or a bottom.
  mlir::DictionaryAttr getConcreteValue() const {
    if (value.getInt() != kSpecificTypeState)
      return nullptr;
    return llvm::cast<mlir::DictionaryAttr>(value.getPointer());
  }

  // Return the top lattice instance.
  static IndexExprsLatticeStorage top() {
    IndexExprsLatticeStorage result;
    result.value.setPointer(nullptr);
    result.value.setInt(kUndecidableState);
    return result;
  }

  // Return the bottom lattice instance.
  static IndexExprsLatticeStorage bottom() {
    IndexExprsLatticeStorage result;
    result.value.setPointer(nullptr);
    result.value.setInt(kUninitializedState);
    return result;
  }

  // Join two lattice instances and return the result.
  static IndexExprsLatticeStorage
  join(const IndexExprsLatticeStorage &lhs, const IndexExprsLatticeStorage &rhs,
       // TODO: we'd want a WaveSymbolAttr here, but we are actually using
       // StringAttr below because of interaction with DictAttr. We want a
       // custom attribute instead, which would also keep entries sorted.
       llvm::ArrayRef<llvm::StringRef> ignoredRhsSymbols = {}) {
    if (lhs.value == rhs.value)
      return lhs;

    if (lhs.isTop() || rhs.isTop())
      return top();

    if (lhs.isBottom()) {
      if (ignoredRhsSymbols.empty() || rhs.isBottom())
        return rhs;

      llvm::SmallVector<mlir::NamedAttribute> filtered = llvm::filter_to_vector(
          rhs.getConcreteValue(), [&](mlir::NamedAttribute attr) {
            return !llvm::is_contained(ignoredRhsSymbols,
                                       attr.getName().getValue());
          });
      return IndexExprsLatticeStorage(mlir::DictionaryAttr::get(
          rhs.getConcreteValue().getContext(), filtered));
    }

    if (rhs.isBottom())
      return lhs;

    mlir::MLIRContext *ctx = lhs.getConcreteValue().getContext();
    mlir::DictionaryAttr lhsValue = lhs.getConcreteValue();
    mlir::DictionaryAttr rhsValue = rhs.getConcreteValue();

    llvm::DenseMap<mlir::StringAttr, mlir::Attribute> result;
    for (mlir::NamedAttribute namedAttr : lhsValue) {
      result[namedAttr.getName()] = namedAttr.getValue();
    }
    for (mlir::NamedAttribute namedAttr : rhsValue) {
      if (llvm::find_if(ignoredRhsSymbols, [&](llvm::StringRef symbol) {
            return symbol == namedAttr.getName().getValue();
          }) != ignoredRhsSymbols.end()) {
        continue;
      }

      auto it = result.find(namedAttr.getName());
      if (it == result.end()) {
        result[namedAttr.getName()] = namedAttr.getValue();
        continue;
      }

      auto lhsValue = llvm::cast<wave::WaveIndexMappingAttr>(it->getSecond());
      auto rhsValue =
          llvm::cast<wave::WaveIndexMappingAttr>(namedAttr.getValue());
      if (lhsValue == rhsValue)
        continue;

      // TODO: fix this string-based abomination.
      llvm::SmallVector<llvm::StringRef> fixmeMagicThreadDependentNames = {
          "_T0", "_T1", "_T2", "_GPR_NUM"};
      auto hasThreadSymbols = [&](mlir::AffineMap map,
                                  llvm::ArrayRef<llvm::StringRef> names) {
        for (auto &&[i, symbol] : llvm::enumerate(names)) {
          if (!llvm::is_contained(fixmeMagicThreadDependentNames, symbol))
            continue;
          if (map.isFunctionOfSymbol(i))
            return true;
        }
        return false;
      };

      auto isThreadDependent = [&](wave::WaveIndexMappingAttr val) -> bool {
        return llvm::any_of(
            llvm::ArrayRef{val.getStart(), val.getStep(), val.getStride()},
            [&](mlir::AffineMap map) {
              return hasThreadSymbols(map, val.getAllSymbolNames());
            });
      };

      // If both are thread-dependent or thread-independent, the only acceptable
      // join is when they are equal, which was handled above.
      bool lhsIsThreadDependent = isThreadDependent(lhsValue);
      bool rhsIsThreadDependent = isThreadDependent(rhsValue);
      if (!(lhsIsThreadDependent ^ rhsIsThreadDependent))
        return top();

      wave::WaveIndexMappingAttr threadDependentMapping =
          lhsIsThreadDependent ? lhsValue : rhsValue;
      wave::WaveIndexMappingAttr threadIndependentMapping =
          lhsIsThreadDependent ? rhsValue : lhsValue;

      // Collect all unique symbol names from both index mappings in order.
      llvm::SmallVector<llvm::StringRef> allSymbolNames;
      llvm::StringMap<unsigned> symbolNamesToIdx;
      auto threadDependentSymbolNames =
          threadDependentMapping.getAllSymbolNames();
      auto threadIndependentSymbolNames =
          threadIndependentMapping.getAllSymbolNames();
      aggregateAllSymbolNames(llvm::ArrayRef{threadIndependentSymbolNames,
                                             threadDependentSymbolNames},
                              allSymbolNames, symbolNamesToIdx);

      mlir::AffineMap threadDependentStart = permuteMapSymbols(
          threadDependentMapping.getStart(), threadDependentSymbolNames,
          allSymbolNames, symbolNamesToIdx);
      mlir::AffineMap threadIndependentStart = permuteMapSymbols(
          threadIndependentMapping.getStart(), threadIndependentSymbolNames,
          allSymbolNames, symbolNamesToIdx);

      mlir::AffineMap threadDependentStep = permuteMapSymbols(
          threadDependentMapping.getStep(), threadDependentSymbolNames,
          allSymbolNames, symbolNamesToIdx);
      mlir::AffineMap threadIndependentStep = permuteMapSymbols(
          threadIndependentMapping.getStep(), threadIndependentSymbolNames,
          allSymbolNames, symbolNamesToIdx);

      mlir::AffineMap threadDependentStride = permuteMapSymbols(
          threadDependentMapping.getStride(), threadDependentSymbolNames,
          allSymbolNames, symbolNamesToIdx);
      mlir::AffineMap threadIndependentStride = permuteMapSymbols(
          threadIndependentMapping.getStride(), threadIndependentSymbolNames,
          allSymbolNames, symbolNamesToIdx);

      // Subtract the thread-independent from thread-dependent for each.
      mlir::MLIRContext *ctx = threadDependentMapping.getContext();
      auto subtractMaps = [&](mlir::AffineMap a,
                              mlir::AffineMap b) -> mlir::AffineMap {
        // Assert there is only one result expression in each map.
        assert(a.getNumResults() == 1 &&
               "expected a single result expression in affine map 'a'");
        assert(b.getNumResults() == 1 &&
               "expected a single result expression in affine map 'b'");
        mlir::AffineExpr subtracted = a.getResult(0) - b.getResult(0);
        return mlir::AffineMap::get(a.getNumDims(), a.getNumSymbols(),
                                    subtracted, ctx);
      };
      mlir::AffineMap newStart =
          subtractMaps(threadDependentStart, threadIndependentStart);
      mlir::AffineMap newStep =
          subtractMaps(threadDependentStep, threadIndependentStep);
      mlir::AffineMap newStride =
          subtractMaps(threadDependentStride, threadIndependentStride);

      llvm::DenseSet<unsigned> allowedSymbols;
      for (llvm::StringRef symbolName : fixmeMagicThreadDependentNames) {
        auto it = symbolNamesToIdx.find(symbolName);
        if (it != symbolNamesToIdx.end())
          allowedSymbols.insert(it->second);
      }
      auto isOnlyThreadDependent = [&](mlir::AffineMap map) {
        mlir::WalkResult walkResult =
            map.getResult(0).walk([&](mlir::AffineExpr expr) {
              auto symExpr = llvm::dyn_cast<mlir::AffineSymbolExpr>(expr);
              if (!symExpr)
                return mlir::WalkResult::advance();
              if (!allowedSymbols.contains(symExpr.getPosition()))
                return mlir::WalkResult::interrupt();
              return mlir::WalkResult::advance();
            });
        return !walkResult.wasInterrupted();
      };

      if (!isOnlyThreadDependent(newStart) || !isOnlyThreadDependent(newStep) ||
          !isOnlyThreadDependent(newStride))
        return top();

      result[namedAttr.getName()] = threadDependentMapping;
    }
    return IndexExprsLatticeStorage(mlir::DictionaryAttr::get(
        ctx, llvm::map_to_vector(result, [](auto &&pair) {
          return mlir::NamedAttribute(pair.first, pair.second);
        })));
  }

  // XXX: backward analysis calls `meet` instead of `join`, but it isn't related
  // to the direction of the analysis. Just defer to join.
  static IndexExprsLatticeStorage meet(const IndexExprsLatticeStorage &lhs,
                                       const IndexExprsLatticeStorage &rhs) {
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
  void unsafeSet(const IndexExprsLatticeStorage &value) {
    this->value = value.value;
  }

  // Return a new lattice instance with only the provided symbols present.
  IndexExprsLatticeStorage
  keepOnlySymbols(llvm::ArrayRef<wave::WaveSymbolAttr> symbols) const {
    if (isBottom() || isTop())
      return *this;

    llvm::StringSet<> symbolNames;
    for (wave::WaveSymbolAttr symbol : symbols)
      symbolNames.insert(symbol.getName());

    llvm::SmallVector<mlir::NamedAttribute> filtered = llvm::filter_to_vector(
        getConcreteValue(), [&](mlir::NamedAttribute attr) {
          return symbolNames.contains(attr.getName().getValue());
        });

    if (filtered.empty())
      return bottom();

    return IndexExprsLatticeStorage(
        mlir::DictionaryAttr::get(getConcreteValue().getContext(), filtered));
  }

  void print(llvm::raw_ostream &os) const {
    if (isBottom()) {
      os << "<bottom>";
    } else if (isTop()) {
      os << "<top>";
    } else {
      os << getConcreteValue();
    }
  }

  LLVM_DUMP_METHOD void dump() const { print(llvm::errs()); }

private:
  // The internal storage is either a dictionary attribute with one entry per
  // symbol indexing the value or one of the top/bottom flags.
  llvm::PointerIntPair<mlir::Attribute, 2> value;

  // State flags.
  const static unsigned kUninitializedState = 0;
  const static unsigned kSpecificTypeState = 1;
  const static unsigned kUndecidableState = 2;
};

void operator<<(mlir::Diagnostic &diag, const IndexExprsLatticeStorage &value) {
  std::string str;
  llvm::raw_string_ostream os(str);
  value.print(os);
  diag << os.str();
}

#include "water/Dialect/Wave/IR/WaveOps.h"

class IndexExprsLattice
    : public mlir::dataflow::Lattice<IndexExprsLatticeStorage> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IndexExprsLattice);
  using Lattice::Lattice;
};

// Return the list of symbols indexing the given operation, i.e. the union of
// symbols present in operand and result tensors.
static llvm::SmallVector<wave::WaveSymbolAttr>
getIndexingSymbols(mlir::Operation *op) {
  llvm::SetVector<wave::WaveSymbolAttr> symbols;
  auto append = [&](mlir::Value value) {
    auto tensorType = llvm::dyn_cast<wave::WaveTensorType>(value.getType());
    if (!tensorType)
      return;
    symbols.insert_range(tensorType.getShape());
  };
  llvm::for_each(op->getOperands(), append);
  llvm::for_each(op->getResults(), append);
  return symbols.takeVector();
}

static mlir::AffineExpr getOrInsertSymbolExpr(
    wave::WaveSymbolAttr symbol,
    llvm::SmallVectorImpl<wave::WaveSymbolAttr> &symbolNames) {
  auto it = llvm::find(symbolNames, symbol);
  unsigned position = [&] {
    if (it != symbolNames.end())
      return static_cast<unsigned>(std::distance(symbolNames.begin(), it));
    symbolNames.push_back(symbol);
    return static_cast<unsigned>(symbolNames.size() - 1);
  }();
  return mlir::getAffineSymbolExpr(position, symbol.getContext());
}

template <typename ConstraintAttrT>
static wave::WaveIndexMappingAttr
applyConstraint(ConstraintAttrT constraint,
                wave::WaveIndexMappingAttr baseMapping = nullptr) {
  static_assert(llvm::is_one_of<ConstraintAttrT, wave::WorkgroupConstraintAttr,
                                wave::TilingConstraintAttr>(),
                "unsupported constraint type for applyConstraint");

  llvm::SmallVector<wave::WaveSymbolAttr> symbolNames =
      llvm::to_vector(constraint.getTileSize().getSymbols());

  mlir::AffineExpr symbolExpr;
  mlir::MLIRContext *context = constraint.getContext();

  if constexpr (std::is_same_v<ConstraintAttrT,
                               wave::WorkgroupConstraintAttr>) {
    // TODO: remove this string-based abomination in favor of first-class
    // attributes.
    std::string symbolString =
        "_WG" + std::to_string(static_cast<uint32_t>(
                    constraint.getWorkgroupDim().getValue()));
    wave::WaveSymbolAttr symbolNameAttr =
        wave::WaveSymbolAttr::get(context, symbolString);
    symbolExpr = getOrInsertSymbolExpr(symbolNameAttr, symbolNames);
  } else if constexpr (std::is_same_v<ConstraintAttrT,
                                      wave::TilingConstraintAttr>) {
    symbolExpr = getOrInsertSymbolExpr(constraint.getDim(), symbolNames);
  }

  assert(constraint.getTileSize().getMap().getNumResults() == 1 &&
         "expected a single result expression in affine map");
  mlir::AffineMap map = mlir::AffineMap::get(
      /*dimCount=*/0, symbolNames.size(),
      symbolExpr * constraint.getTileSize().getMap().getResult(0));
  if (baseMapping == nullptr)
    return wave::WaveIndexMappingAttr::get(
        context, symbolNames, map, mlir::AffineMap::getConstantMap(1, context),
        mlir::AffineMap::getConstantMap(1, context));

  // TODO: there's too much of stringRef flying around here, use proper symbols
  // instead.
  llvm::SmallVector<llvm::StringRef> symbolNameStrings =
      llvm::map_to_vector(symbolNames, [](wave::WaveSymbolAttr symbol) {
        return symbol.getName();
      });
  llvm::SmallVector<llvm::StringRef> baseSymbolNames =
      baseMapping.getAllSymbolNames();
  llvm::SmallVector<llvm::StringRef> allSymbolNames;
  llvm::StringMap<unsigned int> symbolNamesToIdx;
  aggregateAllSymbolNames(llvm::ArrayRef{baseSymbolNames, symbolNameStrings},
                          allSymbolNames, symbolNamesToIdx);
  mlir::AffineMap baseStart =
      permuteMapSymbols(baseMapping.getStart(), baseSymbolNames, allSymbolNames,
                        symbolNamesToIdx);
  map = permuteMapSymbols(map, symbolNameStrings, allSymbolNames,
                          symbolNamesToIdx);
  map = mlir::AffineMap::get(/*dimCount=*/0, allSymbolNames.size(),
                             baseStart.getResult(0) + map.getResult(0));
  return wave::WaveIndexMappingAttr::get(context, symbolNames, map,
                                         baseMapping.getStep(),
                                         baseMapping.getStride());
}

#include "llvm/ADT/TypeSwitch.h"

static wave::WaveIndexMappingAttr
applyConstraintGeneric(mlir::Attribute constraint,
                       wave::WaveIndexMappingAttr baseMapping = nullptr) {
  return llvm::TypeSwitch<mlir::Attribute, wave::WaveIndexMappingAttr>(
             constraint)
      .Case<wave::WorkgroupConstraintAttr, wave::TilingConstraintAttr>(
          [&](auto constraint) {
            // This double dispatching is necessary in absence of interfaces to
            // dispatch to a class method based on a specific type.
            return applyConstraint(constraint, baseMapping);
          })
      .Default([&](mlir::Attribute constraint) { return nullptr; });
}

/// Applies thread-independent constraints to symbol mappings.
///
/// For each symbol in indexingSymbols, this function looks up the symbol in
/// symbolConstraints and applies all constraints to the corresponding mapping
/// in symbolMappings.
static void mixInThreadIndependentConstraints(
    llvm::ArrayRef<wave::WaveSymbolAttr> indexingSymbols,
    const llvm::DenseMap<wave::WaveSymbolAttr,
                         llvm::SmallVector<mlir::Attribute>> &symbolConstraints,
    llvm::SmallVector<mlir::NamedAttribute> &symbolMappings) {
  for (wave::WaveSymbolAttr symbol : indexingSymbols) {
    auto it = symbolConstraints.find(symbol);
    if (it == symbolConstraints.end())
      continue;

    auto mappingIt =
        llvm::find_if(symbolMappings, [&](mlir::NamedAttribute attr) {
          return attr.getName() == symbol.getName();
        });
#ifndef NDEBUG
    llvm::errs() << "symbol: " << symbol.getName() << "\n";
    assert(mappingIt != symbolMappings.end() &&
           "expected a mapping for the symbol");
#endif // NDEBUG
    wave::WaveIndexMappingAttr mapping =
        llvm::cast<wave::WaveIndexMappingAttr>(mappingIt->getValue());
    for (mlir::Attribute constraint : it->second) {
      mapping = applyConstraintGeneric(constraint, mapping);
    }
    mappingIt->setValue(mapping);
  }
}

static mlir::MLIRContext *getAnySymbolContext(wave::WaveSymbolAttr mSymbol,
                                              wave::WaveSymbolAttr nSymbol,
                                              wave::WaveSymbolAttr kSymbol) {
  mlir::MLIRContext *context = nullptr;
  for (wave::WaveSymbolAttr symbol : {mSymbol, nSymbol, kSymbol})
    if (!context && symbol)
      context = symbol.getContext();
  assert(context && "expected at least one symbol name to be provided");
  return context;
}

namespace {

struct MmaIndexingExprBuilder;

struct MmaSingleIndexExprBuilder {
  MmaSingleIndexExprBuilder(MmaIndexingExprBuilder &parent, bool enabled)
      : parent(parent), enabled(enabled) {}

  MmaSingleIndexExprBuilder &offset(mlir::AffineExpr expr);
  MmaSingleIndexExprBuilder &size(int64_t value);
  MmaSingleIndexExprBuilder &stride(int64_t value);
  MmaSingleIndexExprBuilder &m();
  MmaSingleIndexExprBuilder &n();
  MmaSingleIndexExprBuilder &k();
  void populate(llvm::SmallVectorImpl<mlir::NamedAttribute> &attributes) const;

  MmaIndexingExprBuilder &parent;
  mlir::AffineExpr offsetExpr, sizeExpr, strideExpr;
  bool enabled;
};

struct MmaIndexingExprBuilder {
  MmaIndexingExprBuilder(llvm::ArrayRef<wave::WaveSymbolAttr> symbols,
                         wave::WaveSymbolAttr mSymbol,
                         wave::WaveSymbolAttr nSymbol,
                         wave::WaveSymbolAttr kSymbol)
      : symbols(symbols), mBuilder(*this, mSymbol != nullptr),
        nBuilder(*this, nSymbol != nullptr),
        kBuilder(*this, kSymbol != nullptr), mSymbol(mSymbol), nSymbol(nSymbol),
        kSymbol(kSymbol) {}

  MmaSingleIndexExprBuilder &m() { return mBuilder; }
  MmaSingleIndexExprBuilder &n() { return nBuilder; }
  MmaSingleIndexExprBuilder &k() { return kBuilder; }

  void populate(llvm::SmallVectorImpl<mlir::NamedAttribute> &attributes) const {
    mlir::MLIRContext *ctx = getAnySymbolContext(mSymbol, nSymbol, kSymbol);

    auto buildMap = [&](mlir::AffineExpr expr) {
      assert(expr &&
             "expected offset/size/stride to be set up for all symbols");
      return mlir::AffineMap::get(/*dimCount=*/0,
                                  /*symbolCount=*/symbols.size(), expr, ctx);
    };
    auto buildOne = [&](const MmaSingleIndexExprBuilder &builder) {
      return wave::WaveIndexMappingAttr::get(
          ctx, symbols, buildMap(builder.offsetExpr),
          buildMap(builder.sizeExpr), buildMap(builder.strideExpr));
    };

    if (mSymbol)
      attributes.emplace_back(mSymbol.getName(), buildOne(mBuilder));
    if (nSymbol)
      attributes.emplace_back(nSymbol.getName(), buildOne(nBuilder));
    if (kSymbol)
      attributes.emplace_back(kSymbol.getName(), buildOne(kBuilder));
  }

  llvm::ArrayRef<wave::WaveSymbolAttr> symbols;
  MmaSingleIndexExprBuilder mBuilder, nBuilder, kBuilder;
  wave::WaveSymbolAttr mSymbol, nSymbol, kSymbol;
};

MmaSingleIndexExprBuilder &
MmaSingleIndexExprBuilder::offset(mlir::AffineExpr expr) {
  if (!enabled)
    return *this;
  assert(!offsetExpr && "expected offset to be set only once");
  offsetExpr = expr;
  return *this;
}

MmaSingleIndexExprBuilder &MmaSingleIndexExprBuilder::size(int64_t value) {
  if (!enabled)
    return *this;
  assert(offsetExpr && "expected offset to be set before size");
  assert(!sizeExpr && "expected size to be set only once");
  sizeExpr = mlir::getAffineConstantExpr(value, offsetExpr.getContext());
  return *this;
}

MmaSingleIndexExprBuilder &MmaSingleIndexExprBuilder::stride(int64_t value) {
  if (!enabled)
    return *this;
  assert(offsetExpr && "expected offset to be set before stride");
  assert(!strideExpr && "expected stride to be set only once");
  strideExpr = mlir::getAffineConstantExpr(value, offsetExpr.getContext());
  return *this;
}

MmaSingleIndexExprBuilder &MmaSingleIndexExprBuilder::m() { return parent.m(); }
MmaSingleIndexExprBuilder &MmaSingleIndexExprBuilder::n() { return parent.n(); }
MmaSingleIndexExprBuilder &MmaSingleIndexExprBuilder::k() { return parent.k(); }
void MmaSingleIndexExprBuilder::populate(
    llvm::SmallVectorImpl<mlir::NamedAttribute> &attributes) const {
  parent.populate(attributes);
}

// Wrapper to print operations without regions. Use as `llvm::outs() <<
// PrintNoRegions(op)`.
class PrintNoRegions {
public:
  PrintNoRegions(mlir::Operation *op) : operation(op) {}

  void print(llvm::raw_ostream &os) const {
    operation->print(os, mlir::OpPrintingFlags().skipRegions());
    os << "\n";
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

static llvm::LogicalResult populateMmaIndexingExpr(
    wave::WaveMmaKind kind, bool isAccumulator,
    llvm::ArrayRef<unsigned> wavesPerWorkgroup, int64_t threadsPerWave,
    wave::WaveSymbolAttr mSymbol, wave::WaveSymbolAttr nSymbol,
    wave::WaveSymbolAttr kSymbol,
    llvm::SmallVectorImpl<mlir::NamedAttribute> &attributes) {
  mlir::MLIRContext *ctx = getAnySymbolContext(mSymbol, nSymbol, kSymbol);

  // TODO: fix the string-based abomination in favor of first-class symbols.
  llvm::SmallVector<wave::WaveSymbolAttr> symbolNames = {
      wave::WaveSymbolAttr::get(ctx, "_T0"),
      wave::WaveSymbolAttr::get(ctx, "_T1"),
      wave::WaveSymbolAttr::get(ctx, "_T2"),
      wave::WaveSymbolAttr::get(ctx, "_GPR_NUM"),
  };
  mlir::AffineExpr threadX, threadY, threadZ, gprNum;
  mlir::bindSymbols(ctx, threadX, threadY, threadZ, gprNum);

  mlir::AffineExpr linearizedThreadId =
      threadX + threadY * wavesPerWorkgroup[0] +
      threadZ * wavesPerWorkgroup[1] * wavesPerWorkgroup[0];
  mlir::AffineExpr laneId = linearizedThreadId % threadsPerWave;
  MmaIndexingExprBuilder builder(symbolNames, mSymbol, nSymbol, kSymbol);

  switch (kind) {
  case wave::WaveMmaKind::F32_16x16x16_F16:
  case wave::WaveMmaKind::I32_16x16x16_I8:
    builder.m()
        .offset(isAccumulator ? 4 * laneId.floorDiv(16) : laneId % 16)
        .size(isAccumulator ? 4 : 1)
        .stride(isAccumulator ? 16 : 1)
        .n()
        .offset(laneId % 16)
        .size(1)
        .stride(1)
        .k()
        .offset(4 * laneId.floorDiv(16))
        .size(4)
        .stride(1)
        .populate(attributes);
    return llvm::LogicalResult::success();

  case wave::WaveMmaKind::F32_32x32x8_F16:
  case wave::WaveMmaKind::I32_32x32x8_I8:
    builder.m()
        .offset(isAccumulator ? (8 * gprNum.floorDiv(4) % 32) +
                                    4 * laneId.floorDiv(32) + (gprNum % 4)
                              : laneId % 32)
        .size(isAccumulator ? 16 : 1)
        .stride(isAccumulator ? 32 : 1)
        .n()
        .offset(laneId % 32)
        .size(1)
        .stride(1)
        .k()
        .offset(4 * laneId.floorDiv(32))
        .size(4)
        .stride(1)
        .populate(attributes);
    return llvm::LogicalResult::success();

  case wave::WaveMmaKind::F32_16x16x32_F8:
  case wave::WaveMmaKind::F32_16x16x32_BF16:
  case wave::WaveMmaKind::F32_16x16x32_F16:
  case wave::WaveMmaKind::F32_16x16x32_K8_F16:
  case wave::WaveMmaKind::I32_16x16x32_I8:
    builder.m()
        .offset(isAccumulator ? 4 * laneId.floorDiv(16) : laneId % 16)
        .size(isAccumulator ? 4 : 1)
        .stride(isAccumulator ? 16 : 1)
        .n()
        .offset(laneId % 16)
        .size(1)
        .stride(1)
        .k()
        .offset(8 * laneId.floorDiv(16))
        .size(8)
        .stride(1)
        .populate(attributes);
    return llvm::LogicalResult::success();
  case wave::WaveMmaKind::F32_16x16x32_K4_F8:
    builder.m()
        .offset(isAccumulator ? 4 * laneId.floorDiv(16) : laneId % 16)
        .size(isAccumulator ? 4 : 1)
        .stride(isAccumulator ? 16 : 1)
        .n()
        .offset(laneId % 16)
        .size(1)
        .stride(1)
        .k()
        .offset(16 * gprNum.floorDiv(4) + 4 * laneId.floorDiv(16) +
                (gprNum % 4))
        .size(8)
        .stride(1)
        .populate(attributes);
    return llvm::LogicalResult::success();
  case wave::WaveMmaKind::F32_32x32x16_F8:
  case wave::WaveMmaKind::F32_32x32x16_BF16:
  case wave::WaveMmaKind::F32_32x32x16_F16:
  case wave::WaveMmaKind::F32_32x32x16_K8_F16:
  case wave::WaveMmaKind::I32_32x32x16_I8:
    builder.m()
        .offset(isAccumulator ? (8 * gprNum.floorDiv(4) % 32) +
                                    4 * laneId.floorDiv(32) + (gprNum % 4)
                              : laneId % 32)
        .size(isAccumulator ? 16 : 1)
        .stride(isAccumulator ? 32 : 1)
        .n()
        .offset(laneId % 32)
        .size(1)
        .stride(1)
        .k()
        .offset(8 * laneId.floorDiv(32))
        .size(8)
        .stride(1)
        .populate(attributes);
    return llvm::LogicalResult::success();
  case wave::WaveMmaKind::F32_32x32x16_K4_F8:
    builder.m()
        .offset(isAccumulator ? (8 * gprNum.floorDiv(4) % 32) +
                                    4 * laneId.floorDiv(32) + (gprNum % 4)
                              : laneId % 32)
        .size(isAccumulator ? 16 : 1)
        .stride(isAccumulator ? 32 : 1)
        .n()
        .offset(laneId % 32)
        .size(1)
        .stride(1)
        .k()
        .offset(8 * gprNum.floorDiv(4) + 4 * laneId.floorDiv(32) + (gprNum % 4))
        .size(8)
        .stride(1)
        .populate(attributes);
    return llvm::LogicalResult::success();
  default:
    return llvm::LogicalResult::failure();
  }
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

/// Parse and validate wave constraints from an attribute array.
/// Returns the hardware constraint or nullptr on failure.
static wave::HardwareConstraintAttr parseWaveConstraints(
    mlir::Operation *parent, mlir::Attribute constraints,
    llvm::DenseMap<wave::WaveSymbolAttr, llvm::SmallVector<mlir::Attribute>>
        &symbolConstraints) {
  wave::HardwareConstraintAttr hardwareConstraint;
  for (mlir::Attribute constraint : llvm::cast<mlir::ArrayAttr>(constraints)) {
    if (auto workgroup =
            llvm::dyn_cast<wave::WorkgroupConstraintAttr>(constraint)) {
      symbolConstraints[workgroup.getDim()].push_back(workgroup);
    } else if (auto tiling =
                   llvm::dyn_cast<wave::TilingConstraintAttr>(constraint)) {
      symbolConstraints[tiling.getDim()].push_back(tiling);
    } else if (auto hardware =
                   llvm::dyn_cast<wave::HardwareConstraintAttr>(constraint)) {
      if (!hardwareConstraint) {
        hardwareConstraint = hardware;
      } else {
        // TODO: this should be checked by the verifier.
        parent->emitError()
            << "multiple hardware constraints are not supported";
        return nullptr;
      }
    } else {
      parent->emitError() << "unsupported constraint type: " << constraint;
      return nullptr;
    }
  }

  if (!hardwareConstraint) {
    parent->emitError() << "expected a hardware constraint";
    return nullptr;
  }
  // TODO: compute waves_per_block from wave constraints; this should be
  // done in the attribute itself. Maybe move this to the attribute
  // verifier.
  llvm::ArrayRef<unsigned> wavesPerBlock =
      hardwareConstraint.getWavesPerBlock();
  if (wavesPerBlock.size() != 3) {
    parent->emitError() << "expected a waves_per_block entry with three "
                           "elements in the hardware constraint";
    return nullptr;
  }

  return hardwareConstraint;
}

class IndexExprsForwardAnalysis
    : public mlir::dataflow::SparseForwardDataFlowAnalysis<IndexExprsLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  mlir::LogicalResult initialize(mlir::Operation *top) override {
    if (getSolverConfig().isInterprocedural())
      return top->emitError() << "interprocedural analysis not supported";

    // Call the base class initialization in order to set up update listeners.
    // Note that this will initialize values at function/region entries to
    // lattice top.
    if (mlir::failed(SparseForwardDataFlowAnalysis::initialize(top)))
      return mlir::failure();

    llvm::DenseMap<mlir::Operation *, mlir::Attribute> constraints;
    if (llvm::failed(collectWaveConstraints(top, constraints)))
      return llvm::failure();

    for (auto &&[parent, attr] : constraints) {
      llvm::DenseMap<wave::WaveSymbolAttr, llvm::SmallVector<mlir::Attribute>>
          symbolConstraints;
      wave::HardwareConstraintAttr hardwareConstraint =
          parseWaveConstraints(parent, attr, symbolConstraints);
      if (!hardwareConstraint)
        return mlir::failure();

      llvm::ArrayRef<unsigned> wavesPerBlock =
          hardwareConstraint.getWavesPerBlock();

      mlir::WalkResult walkResult =
          parent->walk([&](mlir::Operation *op) -> mlir::WalkResult {
            if (auto mma = llvm::dyn_cast<wave::MmaOp>(op)) {
              llvm::ArrayRef<wave::WaveSymbolAttr> indexingSymbols =
                  llvm::cast<wave::WaveTensorType>(mma.getResult().getType())
                      .getShape();
              llvm::SmallVector<mlir::NamedAttribute> symbolMappings;
              symbolMappings.reserve(indexingSymbols.size());

              // TODO: consider whether we want to allow for some batched MMA
              // operations at this level and in general. If not, disallow this
              // at the operation verifier level instead.
              if (indexingSymbols.size() != 2)
                return op->emitError()
                       << "only 2 indexing symbols are currently "
                          "supported for MMA result";
              wave::WaveSymbolAttr mSymbol = indexingSymbols[0];
              wave::WaveSymbolAttr nSymbol = indexingSymbols[1];

              // TODO: propagate MMA kinds from hardware constraints as a
              // separate step...
              wave::WaveMmaKind mmaKind =
                  mma.getKindAttr()
                      ? mma.getKind()
                      : hardwareConstraint.getMmaType().getValue();
              if (llvm::failed(populateMmaIndexingExpr(
                      mmaKind,
                      /*isAccumulator=*/true, wavesPerBlock,
                      hardwareConstraint.getThreadsPerWave(), mSymbol, nSymbol,
                      /*kSymbol=*/nullptr, symbolMappings))) {
                return mma->emitError()
                       << "MMA kind not supported by index deduction";
              }

              mixInThreadIndependentConstraints(
                  indexingSymbols, symbolConstraints, symbolMappings);
              getLatticeElement(mma.getResult())
                  ->getValue()
                  .unsafeSet(mlir::DictionaryAttr::get(top->getContext(),
                                                       symbolMappings));
            }

            // Set block arguments to bottom initially so they can be join'ed
            // with actual lattices coming from other operations.
            for (mlir::Region &region : op->getRegions()) {
              for (mlir::Block &block : region) {
                for (mlir::Value value : block.getArguments()) {
                  if (!llvm::isa<wave::WaveTensorType>(value.getType()))
                    continue;

                  getLatticeElement(value)->getValue().unsafeSet(
                      IndexExprsLatticeStorage::bottom());
                }
              }
            }
            return llvm::success();
          });
      if (walkResult.wasInterrupted())
        return llvm::failure();
    }

    printDiagnosticInVisit = true;
    return llvm::success();
  }

  void setToEntryState(IndexExprsLattice *lattice) override {
    // TODO: rename to "is initialized".
    // Unclear if this doesn't lead to infinite loops if set to entry state
    // called on failure-to-analyze...
    if (printDiagnosticInVisit) {
      propagateIfChanged(lattice,
                         lattice->join(IndexExprsLatticeStorage::top()));
    } else {
      propagateIfChanged(lattice,
                         lattice->join(IndexExprsLatticeStorage::bottom()));
    }
  }

  llvm::LogicalResult
  visitOperation(mlir::Operation *op,
                 llvm::ArrayRef<const IndexExprsLattice *> operands,
                 llvm::ArrayRef<IndexExprsLattice *> results) override {

    LLVM_DEBUG({
      LDBG() << "visiting operation " << PrintNoRegions(op) << "\n";
      LDBG() << "  Operands lattices:\n";
      for (auto [i, operand] : llvm::enumerate(operands)) {
        LDBG() << "    operand #" << i << ": ";
        operand->getValue().print(LDBG_STREAM);
        LDBG() << "";
      }
      // Print all result lattices
      LDBG() << "  Results lattices:\n";
      for (auto [i, result] : llvm::enumerate(results)) {
        LDBG() << "    result #" << i << ": ";
        result->getValue().print(LDBG_STREAM);
        LDBG() << "";
      }
    });

    auto resultLattice = IndexExprsLatticeStorage::bottom();
    if (auto mma = llvm::dyn_cast<wave::MmaOp>(op)) {
      // TODO: move this to a function in the MmaOp itself.
      auto lhsType = llvm::cast<wave::WaveTensorType>(mma.getLhs().getType());
      llvm::StringRef mSymbol = lhsType.getShape()[0].getName();

      // Do not propagate indexing of the M symbol in MxNxK MMA, they are
      // indexed differently between LHS and accumulator/result.
      if (llvm::isa<wave::WaveTensorType>(mma.getLhs().getType())) {
        resultLattice = IndexExprsLatticeStorage::join(
            resultLattice, operands[0]->getValue(), {mSymbol});
      }
      if (llvm::isa<wave::WaveTensorType>(mma.getRhs().getType())) {
        resultLattice = IndexExprsLatticeStorage::join(resultLattice,
                                                       operands[1]->getValue());
      }
      if (llvm::isa<wave::WaveTensorType>(mma.getAccumulator().getType())) {
        resultLattice = IndexExprsLatticeStorage::join(resultLattice,
                                                       operands[2]->getValue());
      }
    } else {
      // Default to propagating from all operands to all results through join.
      for (auto &&[operand, lattice] :
           llvm::zip_equal(op->getOperands(), operands)) {
        if (!llvm::isa<wave::WaveTensorType>(operand.getType()))
          continue;

        resultLattice =
            IndexExprsLatticeStorage::join(resultLattice, lattice->getValue());
      }
    }

    for (auto &&[result, lattice] :
         llvm::zip_equal(op->getResults(), results)) {
      // If the result lattice is already lattice top, it will not change
      // anymore, so don't propagate. We also want to avoid the error message
      // from below.
      if (lattice->getValue().isTop())
        continue;

      auto resultType = llvm::dyn_cast<wave::WaveTensorType>(result.getType());
      if (!resultType)
        continue;

      std::string originalLatticeStr;
      llvm::raw_string_ostream originalLatticeOs(originalLatticeStr);
      lattice->getValue().print(originalLatticeOs);

      propagateIfChanged(lattice, lattice->join(resultLattice.keepOnlySymbols(
                                      resultType.getShape())));
      if (!lattice->getValue().isTop())
        continue;

      // TODO: turn this into error and stop? Make configurable? Remember the
      // place so we can resolve automatically by inserting some sort of
      // transpose/shuffle?
      if (printDiagnosticInVisit) {
        mlir::InFlightDiagnostic diag =
            op->emitWarning() << "conflict when propagating index expressions "
                                 "forward through this operation for result #"
                              << result.getResultNumber();
        diag.attachNote() << "original lattice: " << originalLatticeOs.str();
        diag.attachNote() << "result of joining operand lattices: "
                          << resultLattice;
        for (auto &&[i, operandLattice] : llvm::enumerate(operands)) {
          diag.attachNote()
              << "operand #" << i << " lattice: " << operandLattice->getValue();
        }
      }
    }

    return llvm::success();
  }

private:
  bool printDiagnosticInVisit = false;
};

class IndexExprsBackwardAnalysis
    : public mlir::dataflow::SparseBackwardDataFlowAnalysis<IndexExprsLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  llvm::LogicalResult initialize(mlir::Operation *top) override {
    if (getSolverConfig().isInterprocedural())
      return top->emitError() << "interprocedural analysis not supported";

    // Call the base class initialization in order to set up update listeners.
    // Note that this will initialize values at function/region entries to
    // lattice top.
    if (llvm::failed(SparseBackwardDataFlowAnalysis::initialize(top)))
      return llvm::failure();

    llvm::DenseMap<mlir::Operation *, mlir::Attribute> constraints;
    if (llvm::failed(collectWaveConstraints(top, constraints)))
      return llvm::failure();
    for (auto &&[parent, attr] : constraints) {
      llvm::DenseMap<wave::WaveSymbolAttr, llvm::SmallVector<mlir::Attribute>>
          symbolConstraints;
      wave::HardwareConstraintAttr hardwareConstraint =
          parseWaveConstraints(parent, attr, symbolConstraints);
      if (!hardwareConstraint)
        return mlir::failure();

      llvm::ArrayRef<unsigned> wavesPerBlock =
          hardwareConstraint.getWavesPerBlock();

      parent->walk([&](mlir::Operation *op) -> mlir::WalkResult {
        if (auto mma = llvm::dyn_cast<wave::MmaOp>(op)) {
          auto resultType =
              llvm::cast<wave::WaveTensorType>(mma.getResult().getType());
          auto lhsType =
              llvm::cast<wave::WaveTensorType>(mma.getLhs().getType());
          // TODO: check whether this is actually the case in op verifier
          assert(resultType.getRank() == lhsType.getRank() &&
                 lhsType.getRank() == 2 &&
                 "only 2D MMA operations are supported");
          wave::WaveSymbolAttr mSymbol = resultType.getShape()[0];
          wave::WaveSymbolAttr nSymbol = resultType.getShape()[1];
          wave::WaveSymbolAttr kSymbol = lhsType.getShape()[1];

          // TODO: propagate MMA kinds from hardware constraints as a
          // separate step...
          wave::WaveMmaKind mmaKind =
              mma.getKindAttr() ? mma.getKind()
                                : hardwareConstraint.getMmaType().getValue();

          llvm::SmallVector<mlir::NamedAttribute> operandSymbolMappings;
          if (llvm::failed(populateMmaIndexingExpr(
                  mmaKind, /*isAccumulator=*/false, wavesPerBlock,
                  hardwareConstraint.getThreadsPerWave(), mSymbol, nSymbol,
                  kSymbol, operandSymbolMappings))) {
            return mma->emitError()
                   << "MMA kind not supported by index deduction";
          }

          llvm::SmallVector<mlir::NamedAttribute> accumulatorSymbolMappings;
          if (llvm::failed(populateMmaIndexingExpr(
                  mmaKind,
                  /*isAccumulator=*/true, wavesPerBlock,
                  hardwareConstraint.getThreadsPerWave(), mSymbol, nSymbol,
                  nullptr, accumulatorSymbolMappings))) {
            return mma->emitError()
                   << "MMA kind not supported by index deduction";
          }

          mixInThreadIndependentConstraints({mSymbol, nSymbol, kSymbol},
                                            symbolConstraints,
                                            operandSymbolMappings);
          mixInThreadIndependentConstraints(
              {mSymbol, nSymbol}, symbolConstraints, accumulatorSymbolMappings);

          // Create the LHS and RHS mappings that are not using symbols
          // irrelevant for them.
          llvm::SmallVector<mlir::NamedAttribute> lhsSymbolMappings =
              llvm::filter_to_vector(
                  operandSymbolMappings, [&](mlir::NamedAttribute attr) {
                    return attr.getName() != nSymbol.getName();
                  });
          llvm::SmallVector<mlir::NamedAttribute> rhsSymbolMappings =
              llvm::filter_to_vector(
                  operandSymbolMappings, [&](mlir::NamedAttribute attr) {
                    return attr.getName() != mSymbol.getName();
                  });

          getLatticeElement(mma.getLhs())
              ->getValue()
              .unsafeSet(mlir::DictionaryAttr::get(op->getContext(),
                                                   lhsSymbolMappings));
          getLatticeElement(mma.getRhs())
              ->getValue()
              .unsafeSet(mlir::DictionaryAttr::get(op->getContext(),
                                                   rhsSymbolMappings));
          getLatticeElement(mma.getAccumulator())
              ->getValue()
              .unsafeSet(mlir::DictionaryAttr::get(op->getContext(),
                                                   accumulatorSymbolMappings));
        } else if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
          // Set terminator operands to bottom initially so they can be join'ed
          // with actual lattices coming from other operations.
          for (mlir::Value operand : op->getOperands()) {
            if (!llvm::isa<wave::WaveTensorType>(operand.getType()))
              continue;
            getLatticeElement(operand)->getValue().unsafeSet(
                IndexExprsLatticeStorage::bottom());
          }
        }

        return mlir::WalkResult::advance();
      });
    }

    printDiagnosticInVisit = true;
    return llvm::success();
  }

  void visitBranchOperand(mlir::OpOperand &opOperand) override {
    if (!llvm::isa<wave::WaveTensorType>(opOperand.get().getType()))
      return;
    setToExitState(getLatticeElement(opOperand.get()));
  }

  void visitCallOperand(mlir::OpOperand &opOperand) override {
    if (!llvm::isa<wave::WaveTensorType>(opOperand.get().getType()))
      return;
    setToExitState(getLatticeElement(opOperand.get()));
  }

  void setToExitState(IndexExprsLattice *lattice) override {
    if (printDiagnosticInVisit) {
      propagateIfChanged(lattice,
                         lattice->join(IndexExprsLatticeStorage::top()));
    } else {
      propagateIfChanged(lattice,
                         lattice->join(IndexExprsLatticeStorage::bottom()));
    }
  }

  llvm::LogicalResult
  visitOperation(mlir::Operation *op,
                 llvm::ArrayRef<IndexExprsLattice *> operands,
                 llvm::ArrayRef<const IndexExprsLattice *> results) override {
    LLVM_DEBUG({
      LDBG() << "visiting operation backward " << PrintNoRegions(op) << "\n";
      LDBG() << "  Operands lattices:\n";
      for (auto [i, operand] : llvm::enumerate(operands)) {
        LDBG() << "    operand #" << i << ": ";
        operand->getValue().print(llvm::dbgs());
        LDBG() << "\n";
      }
      LDBG() << "  Results lattices:\n";
      for (auto [i, result] : llvm::enumerate(results)) {
        LDBG() << "    result #" << i << ": ";
        result->getValue().print(llvm::dbgs());
        LDBG() << "\n";
      }
    });

    auto operandLattice = IndexExprsLatticeStorage::bottom();
    auto accumulatorLattice = IndexExprsLatticeStorage::bottom();
    auto mma = llvm::dyn_cast<wave::MmaOp>(op);
    for (auto &&[result, lattice] :
         llvm::zip_equal(op->getResults(), results)) {
      if (!llvm::isa<wave::WaveTensorType>(result.getType()))
        continue;

      // Do not propagate indexing of the M symbol in MxNxK MMA, they are
      // indexed differently between LHS and accumulator/result.
      llvm::SmallVector<llvm::StringRef> ignoredRhsSymbols;
      if (mma) {
        // TODO: move this to a function in the MmaOp itself.
        auto lhsType = llvm::cast<wave::WaveTensorType>(mma.getLhs().getType());
        llvm::StringRef mSymbol = lhsType.getShape()[0].getName();
        ignoredRhsSymbols.push_back(mSymbol);
      }

      operandLattice = IndexExprsLatticeStorage::join(
          operandLattice, lattice->getValue(), ignoredRhsSymbols);

      if (mma) {
        accumulatorLattice = IndexExprsLatticeStorage::join(
            accumulatorLattice, lattice->getValue());
      }
    }

    // Propagate lattices "sideways" between operands for write, this is done in
    // the backward analysis because it has operand lattices mutable.
    if (auto write = llvm::dyn_cast<wave::WriteOp>(op)) {
      IndexExprsLatticeStorage sidewaysPropagationLattice =
          IndexExprsLatticeStorage::bottom();
      for (IndexExprsLattice *operand : operands) {
        sidewaysPropagationLattice = IndexExprsLatticeStorage::join(
            sidewaysPropagationLattice, operand->getValue());
      }
      unsigned valueToStoreOperandNumber =
          write.getValueToStoreMutable().getOperandNumber();
      unsigned memoryOperandNumber =
          write.getMemoryMutable().getOperandNumber();
      if (sidewaysPropagationLattice.isTop()) {
        mlir::InFlightDiagnostic diag =
            op->emitError() << "conflict between operand index expressions";
        diag.attachNote() << "value to store lattice: "
                          << operands[valueToStoreOperandNumber]->getValue();
        diag.attachNote() << "memory lattice: "
                          << operands[memoryOperandNumber]->getValue();
        return diag;
      }
      propagateIfChanged(operands[valueToStoreOperandNumber],
                         operands[valueToStoreOperandNumber]->join(
                             sidewaysPropagationLattice));
      propagateIfChanged(
          operands[memoryOperandNumber],
          operands[memoryOperandNumber]->join(sidewaysPropagationLattice));
    }

    for (auto &&[operand, lattice] :
         llvm::zip_equal(op->getOpOperands(), operands)) {
      if (lattice->getValue().isTop())
        continue;

      auto operandType =
          llvm::dyn_cast<wave::WaveTensorType>(operand.get().getType());
      if (!operandType)
        continue;

      std::string originalLatticeStr;
      llvm::raw_string_ostream originalLatticeOs(originalLatticeStr);
      lattice->getValue().print(originalLatticeOs);

      IndexExprsLatticeStorage *latticeToJoin =
          (mma && operand.getOperandNumber() == 2) ? &accumulatorLattice
                                                   : &operandLattice;
      propagateIfChanged(lattice, lattice->join(latticeToJoin->keepOnlySymbols(
                                      operandType.getShape())));
      if (!lattice->getValue().isTop())
        continue;

      // TODO: turn this into error and stop? Make configurable? Remember the
      // place so we can resolve automatically by inserting some sort of
      // transpose/shuffle?
      // TODO: now that we initialize to bottom, this should not be needed.
      if (printDiagnosticInVisit) {
        mlir::InFlightDiagnostic diag =
            op->emitWarning() << "conflict when propagating index expressions "
                                 "backward through this operation for operand #"
                              << operand.getOperandNumber();
        diag.attachNote() << "original lattice: " << originalLatticeOs.str();
        for (auto &&[i, resultLattice] : llvm::enumerate(results)) {
          diag.attachNote()
              << "result #" << i << " lattice: " << resultLattice->getValue();
        }
      }
    }
    return llvm::success();
  }

private:
  bool printDiagnosticInVisit = false;
};

class InferIndexExprsPass
    : public wave::impl::WaterWaveInferIndexExprsPassBase<InferIndexExprsPass> {
public:
  using Base::Base;

  llvm::LogicalResult
  appendIndexExprForValue(mlir::Location loc, mlir::Value value,
                          llvm::Twine description,
                          const mlir::DataFlowSolver &solver,
                          llvm::SmallVectorImpl<mlir::Attribute> &indexExprs) {
    auto *lattice = solver.lookupState<IndexExprsLattice>(value);
    if (!lattice || lattice->getValue().isBottom()) {
      emitError(loc) << "failed to infer index expressions for " << description;
      return llvm::failure();
    }
    if (lattice->getValue().isTop()) {
      emitError(loc) << "conflict detected in index expressions for "
                     << description;
      return llvm::failure();
    }
    indexExprs.push_back(lattice->getValue().getConcreteValue());
    return llvm::success();
  }

  void runOnOperation() override {
    if (llvm::failed(verifyNormalFormPassPrecondition(
            wave::WaveNormalForm::AllTypesSpecified, getOperation(),
            getArgument())))
      return signalPassFailure();

    mlir::SymbolTableCollection symbolTable;
    mlir::DataFlowConfig config;
    config.setInterprocedural(false);
    mlir::DataFlowSolver solver(config);

    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<mlir::dataflow::SparseConstantPropagation>();
    solver.load<IndexExprsForwardAnalysis>();
    solver.load<IndexExprsBackwardAnalysis>(symbolTable);

    if (llvm::failed(runSolverAndCaptureErrors(solver, getOperation(), false)))
      return signalPassFailure();

    auto *waveDialect =
        getOperation()->getContext()->getLoadedDialect<wave::WaveDialect>();

    // TODO: we want an interface for this...
    mlir::WalkResult walkResult =
        getOperation()->walk([&](mlir::Operation *op) {
          llvm::SmallVector<mlir::Attribute> indexExprs;

          if (op->getDialect() != waveDialect)
            return mlir::WalkResult::advance();
          if (op->hasTrait<mlir::OpTrait::IsTerminator>())
            return mlir::WalkResult::advance();

          // Special case for MMA where we also want to have index expressions
          // for the operands.
          // TODO: this shouldn't be strictly necessary in a purely MLIR flow,
          // but is kept for Python compatibility.
          if (wave::MmaOp mma = llvm::dyn_cast<wave::MmaOp>(op)) {
            for (mlir::OpOperand &operand : mma->getOpOperands()) {
              if (llvm::failed(appendIndexExprForValue(
                      mma->getLoc(), operand.get(),
                      "operand #" + llvm::Twine(operand.getOperandNumber()),
                      solver, indexExprs)))
                return mlir::WalkResult::interrupt();
            }
          }

          // Special case for WriteOp where we want an index expression even
          // though it doesn't have results.
          // TODO: this shouldn't be necessary in a purely MLIR form since
          // mappings are a property of the SSA value (conversely, changing the
          // mapping should create a new value), but keeping for compatibility.
          if (wave::WriteOp write = llvm::dyn_cast<wave::WriteOp>(op)) {
            if (llvm::failed(appendIndexExprForValue(
                    write->getLoc(), write.getValueToStore(), "value to store",
                    solver, indexExprs)))
              return mlir::WalkResult::interrupt();
          }

          for (mlir::OpResult result : op->getResults()) {
            if (!llvm::isa<wave::WaveTensorType>(result.getType()))
              continue;
            if (llvm::failed(appendIndexExprForValue(
                    result.getLoc(), result,
                    "result #" + llvm::Twine(result.getResultNumber()), solver,
                    indexExprs)))
              return mlir::WalkResult::interrupt();
          }
          op->setAttr(
              wave::WaveDialect::kIndexWaveExprListAttrName,
              mlir::ArrayAttr::get(getOperation()->getContext(), indexExprs));
          return mlir::WalkResult::advance();
        });
    if (walkResult.wasInterrupted())
      return signalPassFailure();

    if (llvm::failed(wave::setNormalFormPassPostcondition(
            wave::WaveNormalForm::IndexExprsSpecified, getOperation())))
      return signalPassFailure();
  }
};
