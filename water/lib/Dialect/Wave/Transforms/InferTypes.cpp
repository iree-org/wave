// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/Transforms/Passes.h"
#include "water/Dialect/Wave/Transforms/Utils.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "wave-infer-types"

using wave::ElementsPerThreadLatticeValue;

namespace wave {
#define GEN_PASS_DEF_WATERWAVEINFERTYPESPASS
#define GEN_PASS_DEF_WATERWAVEPROPAGATEELEMENTSPERTHREADPASS
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
