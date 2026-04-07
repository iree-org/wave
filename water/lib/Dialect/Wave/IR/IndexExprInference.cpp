// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/IndexExprInference.h"
#include "water/Dialect/Wave/IR/WaveOps.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "water/Dialect/Wave/IR/IndexExpr.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"
#include "water/Dialect/Wave/Transforms/DataFlowAnalyses.h"
#include "water/Dialect/Wave/Transforms/Utils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/FormatVariadic.h"

#include <functional>

using namespace mlir;
using namespace wave;

using wave::IndexExprsLatticeStorage;

#define DEBUG_TYPE "wave-index-expr-inference"

//-----------------------------------------------------------------------------
// Reduction index expression propagation (from WaveInterfaces.cpp)
//-----------------------------------------------------------------------------

FailureOr<ChangeResult> wave::detail::propagateReductionIndexExprsForward(
    TypeRange operandTypes, Value result,
    llvm::ArrayRef<IndexExprsLatticeStorage> operandExprs,
    llvm::MutableArrayRef<IndexExprsLatticeStorage> resultExprs,
    EmitErrorFn emitError) {
  auto targetTensorType = dyn_cast<WaveTensorType>(result.getType());
  if (!targetTensorType) {
    emitError() << "expected result tensor type, got " << result.getType();
    return failure();
  }

  // Multiple operands appear after expansion, which requires index inference to
  // work, so this is not expected to happen with correct pass pipeline setup.
  if (operandExprs.size() != 2) {
    emitError()
        << "index inference not supported for reduction with multiple operands";
    return failure();
  }

  // Forward propagation is identity only for symbols that are present.
  return identityIndexExprsPropagate(
             operandExprs[0].keepOnlySymbols(targetTensorType.getShape()),
             resultExprs, result, "reduced value", "result", emitError) |
         identityIndexExprsPropagate(operandExprs[1], resultExprs, result,
                                     "init", "result", emitError);
}

FailureOr<ChangeResult> wave::detail::propagateReductionIndexExprsBackward(
    ValueRange operands,
    llvm::MutableArrayRef<IndexExprsLatticeStorage> operandExprs,
    llvm::ArrayRef<IndexExprsLatticeStorage> resultExprs,
    EmitErrorFn emitError) {
  auto initTensorType = dyn_cast<WaveTensorType>(operands[1].getType());
  if (!initTensorType) {
    emitError() << "expected init tensor type, got " << operands[1].getType();
    return failure();
  }
  if (operandExprs.size() != 2) {
    emitError()
        << "index inference not supported for reduction with multiple operands";
    return failure();
  }

  // Backward propagation to the reduced is identity: it will propagate
  // expressions for symbols present in both target and source, but additional
  // propagation from the op defining the operand is needed to cover reduction
  // dimensions. Backward propagation to the init is full identity.
  return identityIndexExprsPropagate(resultExprs[0], operandExprs, operands,
                                     "result", "operands", emitError) |
         identityIndexExprsPropagate(
             operandExprs[0].keepOnlySymbols(initTensorType.getShape()),
             operandExprs[1], operands[1], "operand", "init", emitError) |
         identityIndexExprsPropagate(operandExprs[1], operandExprs[0],
                                     operands[0], "init", "operand", emitError);
}

//-----------------------------------------------------------------------------
// Index expression initialization (from WaveInterfaces.cpp)
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Index expr initialization
//-----------------------------------------------------------------------------

wave::WaveSymbolMappingAttr
wave::detail::filterVectorShape(wave::WaveSymbolMappingAttr vectorShape,
                                llvm::ArrayRef<wave::WaveSymbolAttr> symbols) {
  if (!vectorShape)
    return vectorShape;
  SmallVector<std::pair<wave::WaveSymbolAttr, Attribute>> filtered;
  filtered.reserve(vectorShape.size());
  for (auto [key, value] : vectorShape) {
    if (!llvm::is_contained(symbols, key))
      continue;
    filtered.emplace_back(key, value);
  }
  return wave::WaveSymbolMappingAttr::get(vectorShape.getContext(), filtered);
}

static void initializeIndexExprsWithThreadIndependentConstraints(
    Operation *op, Type type, wave::IndexExprsLatticeStorage &storage,
    const wave::IndexExprsAnalysisInit &initObject) {
  SmallVector<std::pair<wave::WaveSymbolAttr, wave::WaveIndexMappingAttr>>
      entries;
  if (failed(wave::detail::buildThreadIndependentIndexMappings(
          op, type, initObject, entries)))
    return;

  auto tensorType = cast<wave::WaveTensorType>(type);
  SmallVector<std::pair<wave::WaveSymbolAttr, Attribute>> indexEntries(
      entries.begin(), entries.end());
  storage.unsafeSet(wave::IndexExprsLatticeStorage(
      wave::WaveSymbolMappingAttr::get(op->getContext(), indexEntries),
      wave::IndexExprsLatticeStorage::kLowestPriority,
      wave::detail::filterVectorShape(
          initObject.hardwareConstraint.getVectorShapes(),
          tensorType.getShape())));
}

LogicalResult wave::detail::defaultInitializeIndexExprsForward(
    WaveInferIndexExprsOpInterface iface,
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> resultExprs,
    const wave::IndexExprsAnalysisInit &initObject,
    wave::EmitErrorFn emitError) {
  SmallVector<Value> indexedValues;
  iface.getIndexExprValuesAndDescriptions(indexedValues);
  for (Value value : indexedValues) {
    auto opResult = dyn_cast<OpResult>(value);
    if (!opResult || opResult.getOwner() != iface.getOperation())
      continue;

    initializeIndexExprsWithThreadIndependentConstraints(
        iface, opResult.getType(), resultExprs[opResult.getResultNumber()],
        initObject);
  }
  return success();
}

LogicalResult wave::detail::defaultInitializeIndexExprsBackward(
    WaveInferIndexExprsOpInterface iface,
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> operandExprs,
    const wave::IndexExprsAnalysisInit &initObject, wave::EmitErrorFn emitError,
    wave::EmitDelayedErrorFn &delayedErrorEmitter) {
  SmallVector<Value> indexedValues;
  iface.getIndexExprValuesAndDescriptions(indexedValues);
  for (Value value : indexedValues) {
    auto opResult = dyn_cast<OpResult>(value);
    if (opResult && opResult.getOwner() == iface.getOperation())
      continue;

    // A value may be used repeatedly as operand, make sure to update all
    // corresponding operand expression even though they go to the same lattice
    // after all.
    [[maybe_unused]] bool updated = false;
    for (unsigned i = 0, e = operandExprs.size(); i < e; ++i) {
      if (iface->getOperand(i) != value)
        continue;

      initializeIndexExprsWithThreadIndependentConstraints(
          iface, iface->getOperand(i).getType(), operandExprs[i], initObject);
      updated = true;
    }
    assert(updated && "value declared in getIndexExprValuesAndDescriptions is "
                      "neither op operand nor result");
  }
  return success();
}

//-----------------------------------------------------------------------------
// IndexExprsLatticeStorage implementation (from WaveInterfaces.cpp)
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Lattice implementation
//-----------------------------------------------------------------------------

wave::IndexExprsLatticeStorage::IndexExprsLatticeStorage()
    : value(nullptr, kUninitializedState), vectorShape(nullptr) {}

wave::IndexExprsLatticeStorage::IndexExprsLatticeStorage(
    WaveSymbolMappingAttr concreteValue, int32_t priority,
    WaveSymbolMappingAttr vectorShape)
    : value(concreteValue, kSpecificTypeState), vectorShape(vectorShape),
      sourceVectorShape(priority == kLowestPriority ? nullptr : vectorShape),
      sourceVectorShapePriority(priority) {
  MLIRContext *ctx = concreteValue.getContext();
  IntegerType i32 = IntegerType::get(ctx, 32);
  IntegerAttr priAttr = IntegerAttr::get(i32, priority);
  SmallVector<std::pair<wave::WaveSymbolAttr, Attribute>> priEntries;
  priEntries.reserve(concreteValue.size());
  for (auto [key, _] : concreteValue)
    priEntries.emplace_back(key, priAttr);
  priorities = WaveSymbolMappingAttr::get(ctx, priEntries);
}

wave::IndexExprsLatticeStorage::IndexExprsLatticeStorage(
    WaveSymbolMappingAttr concreteValue, WaveSymbolMappingAttr priorities,
    WaveSymbolMappingAttr vectorShape)
    : value(concreteValue, kSpecificTypeState), priorities(priorities),
      vectorShape(vectorShape), sourceVectorShape(vectorShape) {
  if (priorities) {
    for (auto [_, priVal] : priorities)
      sourceVectorShapePriority = std::max<int32_t>(
          sourceVectorShapePriority, llvm::cast<IntegerAttr>(priVal).getInt());
  }
  if (sourceVectorShapePriority == kLowestPriority)
    sourceVectorShape = nullptr;
}

wave::IndexExprsLatticeStorage::IndexExprsLatticeStorage(
    WaveSymbolMappingAttr concreteValue, WaveSymbolMappingAttr priorities,
    WaveSymbolMappingAttr vectorShape, WaveSymbolMappingAttr sourceVectorShape,
    int32_t sourceVectorShapePriority)
    : value(concreteValue, kSpecificTypeState), priorities(priorities),
      vectorShape(vectorShape),
      sourceVectorShape(sourceVectorShapePriority == kLowestPriority
                            ? nullptr
                            : sourceVectorShape),
      sourceVectorShapePriority(sourceVectorShapePriority) {}

bool wave::IndexExprsLatticeStorage::operator==(
    const IndexExprsLatticeStorage &other) const {
  return value == other.value && priorities == other.priorities &&
         vectorShape == other.vectorShape &&
         sourceVectorShape == other.sourceVectorShape &&
         sourceVectorShapePriority == other.sourceVectorShapePriority;
}

bool wave::IndexExprsLatticeStorage::operator!=(
    const IndexExprsLatticeStorage &other) const {
  return !(*this == other);
}

bool wave::IndexExprsLatticeStorage::isBottom() const {
  return value.getInt() == kUninitializedState;
}

bool wave::IndexExprsLatticeStorage::isTop() const {
  return value.getInt() == kUndecidableState;
}

wave::WaveSymbolMappingAttr
wave::IndexExprsLatticeStorage::getConcreteValue() const {
  if (value.getInt() != kSpecificTypeState)
    return nullptr;
  return llvm::cast<WaveSymbolMappingAttr>(value.getPointer());
}

wave::WaveSymbolMappingAttr
wave::IndexExprsLatticeStorage::getVectorShape() const {
  if (value.getInt() != kSpecificTypeState)
    return nullptr;
  return vectorShape;
}

wave::WaveSymbolMappingAttr
wave::IndexExprsLatticeStorage::getSourceVectorShape() const {
  if (value.getInt() != kSpecificTypeState)
    return nullptr;
  return sourceVectorShape;
}

int32_t wave::IndexExprsLatticeStorage::getSourceVectorShapePriority() const {
  return sourceVectorShapePriority;
}

wave::IndexExprsLatticeStorage
wave::IndexExprsLatticeStorage::withSourceVectorShape(
    wave::WaveSymbolMappingAttr shape, int32_t priority) const {
  IndexExprsLatticeStorage copy = *this;
  copy.sourceVectorShape = shape;
  copy.sourceVectorShapePriority = priority;
  return copy;
}

int32_t
wave::IndexExprsLatticeStorage::getPriorityForKey(WaveSymbolAttr key) const {
  assert(getConcreteValue() && "no priority for lattice top/bottom");
  if (!priorities)
    return kLowestPriority;
  auto val = priorities.lookup<IntegerAttr>(key);
  if (!val)
    return kLowestPriority;
  return val.getInt();
}

wave::IndexExprsLatticeStorage wave::IndexExprsLatticeStorage::top() {
  IndexExprsLatticeStorage result;
  result.value.setPointer(nullptr);
  result.value.setInt(kUndecidableState);
  result.vectorShape = nullptr;
  result.sourceVectorShape = nullptr;
  result.sourceVectorShapePriority = 0;
  return result;
}

wave::IndexExprsLatticeStorage wave::IndexExprsLatticeStorage::bottom() {
  IndexExprsLatticeStorage result;
  result.value.setPointer(nullptr);
  result.value.setInt(kUninitializedState);
  result.vectorShape = nullptr;
  result.sourceVectorShape = nullptr;
  result.sourceVectorShapePriority = 0;
  return result;
}

/// Parse and validate wave constraints from an attribute array.
/// Returns the hardware constraint or nullptr on failure.
static wave::HardwareConstraintAttr parseWaveConstraints(
    Location loc, Attribute constraints,
    llvm::DenseMap<wave::WaveSymbolAttr, llvm::SmallVector<Attribute>>
        &symbolConstraints) {
  wave::HardwareConstraintAttr hardwareConstraint;
  for (Attribute constraint : llvm::cast<ArrayAttr>(constraints)) {
    if (auto workgroup =
            llvm::dyn_cast<wave::WorkgroupConstraintAttr>(constraint)) {
      symbolConstraints[workgroup.getDim()].push_back(workgroup);
    } else if (auto tiling =
                   llvm::dyn_cast<wave::TilingConstraintAttr>(constraint)) {
      symbolConstraints[tiling.getDim()].push_back(tiling);
    } else if (auto waveConstraint =
                   llvm::dyn_cast<wave::WaveConstraintAttr>(constraint)) {
      symbolConstraints[waveConstraint.getDim()].push_back(waveConstraint);
    } else if (auto hardware =
                   llvm::dyn_cast<wave::HardwareConstraintAttr>(constraint)) {
      assert(hardwareConstraint == nullptr &&
             "multiple hardware constraints are not supported");
      hardwareConstraint = hardware;
    } else {
      emitError(loc) << "unsupported constraint type: " << constraint;
      return nullptr;
    }
  }

  if (!hardwareConstraint) {
    emitError(loc) << "expected a hardware constraint";
    return nullptr;
  }

  return hardwareConstraint;
}

llvm::FailureOr<wave::IndexExprsAnalysisInit>
wave::IndexExprsAnalysisInit::create(Operation *parent,
                                     Attribute constraintsAttr,
                                     wave::WaveHyperparameterAttr hyperparams) {
  Location loc = parent->getLoc();
  wave::IndexExprsAnalysisInit initObject;
  initObject.hardwareConstraint =
      parseWaveConstraints(loc, constraintsAttr, initObject.symbolConstraints);
  if (initObject.hardwareConstraint == nullptr)
    return llvm::failure();

  parent->walk(
      [&](Operation *op) { initObject.deterministicOpOrder.push_back(op); });

  // If waves_per_block is explicitly provided, copy it to storage. Note that we
  // have verified they match the result of dividing block tiles with wave tiles
  // previously.
  if (!initObject.hardwareConstraint.getWavesPerBlock().empty()) {
    assert(initObject.hardwareConstraint.getWavesPerBlock().size() == 3 &&
           "expected waves_per_block to have 3 elements");
    llvm::ArrayRef<unsigned> explicitWpb =
        initObject.hardwareConstraint.getWavesPerBlock();
    initObject.wavesPerBlock.assign(explicitWpb);
    return initObject;
  }

  // Otherwise, compute waves_per_block from wave constraints.
  // First, extract wave and workgroup constraints from symbolConstraints.
  llvm::SmallDenseMap<wave::WaveSymbolAttr, wave::WorkgroupConstraintAttr>
      workgroupConstraints;
  llvm::SmallDenseMap<wave::WaveSymbolAttr, wave::WaveConstraintAttr>
      waveConstraints;

  for (auto &&[symbol, constraints] : initObject.symbolConstraints) {
    for (Attribute constraint : constraints) {
      if (auto wg = llvm::dyn_cast<wave::WorkgroupConstraintAttr>(constraint))
        workgroupConstraints[symbol] = wg;
      else if (auto wv = llvm::dyn_cast<wave::WaveConstraintAttr>(constraint))
        waveConstraints[symbol] = wv;
    }
  }

  // If there are no wave constraints, default to [1, 1, 1].
  if (waveConstraints.empty()) {
    return emitError(loc) << "expected either waves_per_block in the hardware "
                             "constraint or wave constraints on an ancestor op";
  }

  if (!hyperparams) {
    return emitError(loc) << "cannot compute waves_per_block: missing "
                             "hyperparameters attribute";
  }

  if (failed(wave::computeWavesPerBlockFromConstraints(
          workgroupConstraints, waveConstraints, hyperparams,
          initObject.wavesPerBlock))) {
    return emitError(loc)
           << "failed to compute waves_per_block from wave constraints";
  }
  return initObject;
}

// Create a new map with per-result sum between a and b maps.
static AffineMap addMaps(AffineMap a, AffineMap b) {
  assert(a.getNumResults() == b.getNumResults() &&
         "expected both maps to have the same number of expressions");
  SmallVector<AffineExpr> subtracted = llvm::map_to_vector(
      llvm::zip_equal(a.getResults(), b.getResults()),
      [&](auto &&pair) { return std::get<0>(pair) + std::get<1>(pair); });
  return AffineMap::get(a.getNumDims(), a.getNumSymbols(), subtracted,
                        a.getContext());
}

// Create a new map with per-result difference between a and b maps.
static AffineMap subtractMaps(AffineMap a, AffineMap b) {
  assert(a.getNumResults() == b.getNumResults() &&
         "expected both maps to have the same number of expressions");
  SmallVector<AffineExpr> subtracted = llvm::map_to_vector(
      llvm::zip_equal(a.getResults(), b.getResults()),
      [&](auto &&pair) { return std::get<0>(pair) - std::get<1>(pair); });
  return AffineMap::get(a.getNumDims(), a.getNumSymbols(), subtracted,
                        a.getContext());
}

const static wave::WaveIndexSymbol threadLikeIndexSymbols[] = {
    wave::WaveIndexSymbol::THREAD_0, wave::WaveIndexSymbol::THREAD_1,
    wave::WaveIndexSymbol::THREAD_2, wave::WaveIndexSymbol::GPR_NUMBER};

// Return the list of thread-like index symbols.
// TODO: It would be nice to cache the list somehow, but we need to tie it to
// the context and ensure thread safety, potentially by storing it as an
// attribute or some other named/typed entity in the context object.
static SmallVector<Attribute> getThreadLikeIndexSymbols(MLIRContext *ctx) {
  return llvm::map_to_vector(
      threadLikeIndexSymbols, [&](wave::WaveIndexSymbol symbol) -> Attribute {
        return wave::WaveIndexSymbolAttr::get(ctx, symbol);
      });
}

// Return the list of index symbols other than thread-like.
static SmallVector<Attribute> getNonThreadLikeIndexSymbols(MLIRContext *ctx) {
  return llvm::map_to_vector(ArrayRef{wave::WaveIndexSymbol::WORKGROUP_0,
                                      wave::WaveIndexSymbol::WORKGROUP_1,
                                      wave::WaveIndexSymbol::WORKGROUP_2,
                                      wave::WaveIndexSymbol::DEVICE_DIM_0,
                                      wave::WaveIndexSymbol::DEVICE_DIM_1,
                                      wave::WaveIndexSymbol::DEVICE_DIM_2},
                             [&](wave::WaveIndexSymbol symbol) -> Attribute {
                               return wave::WaveIndexSymbolAttr::get(ctx,
                                                                     symbol);
                             });
}

// Get the positions of `symbols` in `allSymbols`. Missing symbols are ignored.
SmallVector<unsigned> getPositionsOfSymbols(ArrayRef<Attribute> symbols,
                                            ArrayRef<Attribute> allSymbols) {
  // Find positions of threadLikeIndexSymbols in symbols
  SmallVector<unsigned> positions;
  for (Attribute symbol : symbols) {
    auto it = llvm::find(allSymbols, symbol);
    if (it != allSymbols.end())
      positions.push_back(std::distance(allSymbols.begin(), it));
  }
  return positions;
}

// Return true if any expression in the map is a function of a symbol at any of
// the given positions.
static bool isIndexExprMapFunctionOf(AffineMap map,
                                     ArrayRef<unsigned> positions) {
  return llvm::any_of(positions, [&](unsigned position) {
    return map.isFunctionOfSymbol(position);
  });
}

// Affine maps used in an index expression conceptually consists of multiple
// additive components:
//
//   - thread independent component (workgroup and device indices)
//   - thread dependent component (thread and GPR indices)
//   - one component per iter dimension
//
// Two start maps can be joined if, for all pairwise components:
//
//   - the components are equal;
//   - the component is absent from one of the maps.
//
// The join is then the sum of unique components from both maps.
//
// We check this by examining the difference between the two maps, which should
// only contain symbols absent from one of the maps, i.e. symbols from the
// symmetric difference of the symbol sets or, alternatively, not contain any
// symbols from the intersection of the symbol sets. The difference should also
// not contain a constant term.
//
// Additional care is taken for index (non-iter) dimensions as they must appear
// or not appear simultaneously. For example, lhs may only have thread 0 index
// and rhs may only have thread 1 index, so the difference will depend on both
// thread 0 and thread 1 indices without either of them being common, so the
// regular check won't detect that. Check for dependency on any such symbol
// instead.
//
// The function takes the list of symbols used in LHS and RHS and separately a
// list containing a union thereof and a list of positions in that list of the
// common symbols. This is done for efficiency reasons to avoid re-computing
// these several times when handling start/size/stride maps that share the
// symbol lists.
//
// TODO: consider having separate expressions for each component for simplicity;
// even further, consider having a lattice that is a collection of constraints
// applicable to the value + metadata (like it being used in LHS/RHS/Acc of an
// MMA) without creating the expression immediately.
static FailureOr<AffineMap> getIndexExprStartJoinedMap(
    AffineMap lhs, AffineMap rhs, ArrayRef<Attribute> lhsSymbols,
    ArrayRef<Attribute> rhsSymbols, ArrayRef<Attribute> allSymbols,
    ArrayRef<Attribute> commonSymbols) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;

  lhs = wave::alignMapSymbols(lhs, lhsSymbols, allSymbols);
  rhs = wave::alignMapSymbols(rhs, rhsSymbols, allSymbols);

  if (lhs == rhs)
    return lhs;

  AffineMap difference = simplifyAffineMap(subtractMaps(lhs, rhs));

  MLIRContext *ctx = rhs.getContext();
  SmallVector<unsigned> threadLikePositions =
      getPositionsOfSymbols(getThreadLikeIndexSymbols(ctx), allSymbols);
  SmallVector<unsigned> nonThreadLikePositions =
      getPositionsOfSymbols(getNonThreadLikeIndexSymbols(ctx), allSymbols);
  if (isIndexExprMapFunctionOf(difference, threadLikePositions) &&
      isIndexExprMapFunctionOf(lhs, threadLikePositions) &&
      isIndexExprMapFunctionOf(rhs, threadLikePositions)) {
    return failure();
  }
  if (isIndexExprMapFunctionOf(difference, nonThreadLikePositions) &&
      isIndexExprMapFunctionOf(lhs, nonThreadLikePositions) &&
      isIndexExprMapFunctionOf(rhs, nonThreadLikePositions)) {
    return failure();
  }

  // The symbolic part of the difference should not depend on any of the
  // disallowed symbols (usually meaning symbols appearing in both).
  for (AffineExpr expr : difference.getResults()) {
    for (Attribute symbol : commonSymbols) {
      auto it = llvm::find(allSymbols, symbol);
      assert(it != allSymbols.end() &&
             "expected common symbols to be a subset of all symbols");
      unsigned position = std::distance(allSymbols.begin(), it);
      if (expr.isFunctionOfSymbol(position) &&
          !llvm::is_contained(threadLikePositions, position))
        return failure();
    }
  }

  // The constant parts of the expression must be equal.
  // TODO: consider whether we want to allow one of the sides being 0 here. If
  // we do, we will have to be more careful to construct a constant difference
  // map here instead of taking the RHS constant in subtraction below.
  AffineExpr zeroExpr = getAffineConstantExpr(0, ctx);
  SmallVector<AffineExpr> symReplacements(allSymbols.size(), zeroExpr);
  AffineMap lhsConstant =
      lhs.replaceDimsAndSymbols(/*dimReplacements=*/{}, symReplacements,
                                /*numResultDims=*/0, /*numResultSyms=*/0);
  AffineMap rhsConstant =
      rhs.replaceDimsAndSymbols(/*dimReplacements=*/{}, symReplacements,
                                /*numResultDims=*/0, /*numResultSyms=*/0);
  for (auto [lhsConstantExpr, rhsConstantExpr] :
       llvm::zip_equal(lhsConstant.getResults(), rhsConstant.getResults())) {
    auto lhsConstantExprCast = cast<AffineConstantExpr>(lhsConstantExpr);
    auto rhsConstantExprCast = cast<AffineConstantExpr>(rhsConstantExpr);
    if (lhsConstantExprCast.getValue() != rhsConstantExprCast.getValue())
      return failure();
  }

  // Obtain a part of the RHS map that is only a function of RHS-specific
  // symbols. For this, we replace all symbols appearing in the LHS map with
  // zero. Symbol replacements contain zeros at this point. Reuse that but set
  // RHS-only symbols to be replaced with themselves. Don't forget to subtract
  // the constant part of RHS, which is known to be identical to that of RHS, so
  // we don't duplicate it. At this point, we expect the caller to have removed
  // unused symbols from the symbol list.
  SmallVector<unsigned> lhsSymbolPositions =
      getPositionsOfSymbols(lhsSymbols, allSymbols);
  for (unsigned i = 0, e = symReplacements.size(); i < e; ++i) {
    if (llvm::is_contained(lhsSymbolPositions, i))
      continue;
    symReplacements[i] = getAffineSymbolExpr(i, ctx);
  }
  AffineMap rhsOnly = rhs.replaceDimsAndSymbols(
      {}, symReplacements, /*numResultDims=*/0, rhs.getNumSymbols());
  rhsOnly = subtractMaps(rhsOnly, rhsConstant);
  return simplifyAffineMap(addMaps(lhs, rhsOnly));
}

// Two step/stride maps can be joined if one of them is a constant one, at which
// point the join is the other map, or if they are equal. All other combinations
// join to lattice top.
static FailureOr<AffineMap> getIndexExprStepStrideJoinedMap(
    AffineMap lhs, AffineMap rhs, ArrayRef<Attribute> lhsSymbols,
    ArrayRef<Attribute> rhsSymbols, ArrayRef<Attribute> allSymbols) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;

  lhs = wave::alignMapSymbols(lhs, lhsSymbols, allSymbols);
  rhs = wave::alignMapSymbols(rhs, rhsSymbols, allSymbols);

  if (lhs == rhs)
    return lhs;

  AffineExpr lhsExpr = lhs.getResult(0);
  AffineExpr rhsExpr = rhs.getResult(0);
  auto isConstantOne = [](AffineExpr expr) -> bool {
    if (auto constantExpr = llvm::dyn_cast<AffineConstantExpr>(expr)) {
      return constantExpr.getValue() == 1;
    }
    return false;
  };
  bool lhsIsConstantOne = isConstantOne(lhsExpr);
  bool rhsIsConstantOne = isConstantOne(rhsExpr);
  if (lhsIsConstantOne)
    return rhs;
  if (rhsIsConstantOne)
    return lhs;
  return failure();
}

// Join two concrete index expressions mappings either by picking the
// higher-priority one or by joining their start/step/stride maps independently.
// See getIndexExprStartJoinedMap and getIndexExprStepStrideJoinedMap for more
// details on independent joining.
static wave::WaveIndexMappingAttr
getIndexExprsJoinMappings(wave::WaveIndexMappingAttr lhs,
                          wave::WaveIndexMappingAttr rhs, int32_t lhsPriority,
                          int32_t rhsPriority) {
  if (lhsPriority > rhsPriority)
    return lhs;
  if (rhsPriority > lhsPriority)
    return rhs;

  // Collect all unique symbol names from both index mappings in order.
  llvm::SmallVector<Attribute> allSymbols;
  llvm::SetVector<Attribute> lhsSymbols(llvm::from_range, lhs.getSymbols());
  llvm::SetVector<Attribute> rhsSymbols(llvm::from_range, rhs.getSymbols());
  wave::aggregateAllSymbols(llvm::ArrayRef{lhsSymbols, rhsSymbols}, allSymbols);
  llvm::SetVector<Attribute> commonSymbols =
      llvm::set_intersection(lhsSymbols, rhsSymbols);

  FailureOr<AffineMap> joinedStart = getIndexExprStartJoinedMap(
      lhs.getStart(), rhs.getStart(), lhsSymbols.getArrayRef(),
      rhsSymbols.getArrayRef(), allSymbols, commonSymbols.getArrayRef());
  if (failed(joinedStart))
    return nullptr;
  FailureOr<AffineMap> joinedStep = getIndexExprStepStrideJoinedMap(
      lhs.getStep(), rhs.getStep(), lhsSymbols.getArrayRef(),
      rhsSymbols.getArrayRef(), allSymbols);
  if (failed(joinedStep))
    return nullptr;
  FailureOr<AffineMap> joinedStride = getIndexExprStepStrideJoinedMap(
      lhs.getStride(), rhs.getStride(), lhsSymbols.getArrayRef(),
      rhsSymbols.getArrayRef(), allSymbols);
  if (failed(joinedStride))
    return nullptr;

  return wave::WaveIndexMappingAttr::get(
      lhs.getContext(), allSymbols, *joinedStart, *joinedStep, *joinedStride);
}

// Returns a joined vector shape using per-key priorities. For each key, the
// higher-priority entry wins. If priorities are equal and values differ,
// returns nullptr indicating the failure to join (reaching top).
static wave::WaveSymbolMappingAttr
getJoinedVectorShape(wave::WaveSymbolMappingAttr lhs,
                     wave::WaveSymbolMappingAttr rhs,
                     wave::WaveSymbolMappingAttr lhsPriorities,
                     wave::WaveSymbolMappingAttr rhsPriorities) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;

  if (lhs == rhs)
    return lhs;

  auto getPriority = [](wave::WaveSymbolMappingAttr map,
                        wave::WaveSymbolAttr key) -> int32_t {
    if (!map)
      return wave::IndexExprsLatticeStorage::kLowestPriority;
    auto val = map.lookup<IntegerAttr>(key);
    if (!val)
      return wave::IndexExprsLatticeStorage::kLowestPriority;
    return val.getInt();
  };

  SmallVector<std::pair<wave::WaveSymbolAttr, Attribute>> joined;

  llvm::DenseSet<wave::WaveSymbolAttr> visited;
  for (auto [key, value] : lhs) {
    visited.insert(key);
    Attribute rhsValue = rhs.lookup(key);
    if (!rhsValue) {
      joined.emplace_back(key, value);
      continue;
    }
    int32_t lhsPriority = getPriority(lhsPriorities, key);
    int32_t rhsPriority = getPriority(rhsPriorities, key);
    if (lhsPriority > rhsPriority) {
      joined.emplace_back(key, value);
    } else if (rhsPriority > lhsPriority) {
      joined.emplace_back(key, rhsValue);
    } else {
      if (value != rhsValue)
        return nullptr;
      joined.emplace_back(key, value);
    }
  }

  for (auto [key, value] : rhs) {
    if (!visited.contains(key))
      joined.emplace_back(key, value);
  }

  return wave::WaveSymbolMappingAttr::get(lhs.getContext(), joined);
}

FailureOr<wave::WaveSymbolMappingAttr>
wave::IndexExprsLatticeStorage::getJoinedVectorShape(
    const IndexExprsLatticeStorage &lhs, const IndexExprsLatticeStorage &rhs) {
  if (lhs.isBottom() && rhs.isBottom())
    return wave::WaveSymbolMappingAttr();
  if (lhs.isTop() || rhs.isTop())
    return failure();
  if (lhs.isBottom())
    return rhs.getVectorShape();
  if (rhs.isBottom())
    return lhs.getVectorShape();
  if (wave::WaveSymbolMappingAttr result =
          ::getJoinedVectorShape(lhs.getVectorShape(), rhs.getVectorShape(),
                                 lhs.getPriorities(), rhs.getPriorities()))
    return result;
  return failure();
}

FailureOr<std::pair<wave::WaveSymbolMappingAttr, int32_t>>
wave::IndexExprsLatticeStorage::getJoinedSourceVectorShape(
    const IndexExprsLatticeStorage &lhs, const IndexExprsLatticeStorage &rhs) {
  wave::WaveSymbolMappingAttr lhsSVS = lhs.getSourceVectorShape();
  wave::WaveSymbolMappingAttr rhsSVS = rhs.getSourceVectorShape();
  int32_t lhsPri = lhs.getSourceVectorShapePriority();
  int32_t rhsPri = rhs.getSourceVectorShapePriority();

  if (!lhsSVS && !rhsSVS)
    return std::make_pair(wave::WaveSymbolMappingAttr(), int32_t(0));
  if (!lhsSVS)
    return std::make_pair(rhsSVS, rhsPri);
  if (!rhsSVS)
    return std::make_pair(lhsSVS, lhsPri);
  if (lhsPri > rhsPri)
    return std::make_pair(lhsSVS, lhsPri);
  if (rhsPri > lhsPri)
    return std::make_pair(rhsSVS, rhsPri);
  if (lhsSVS == rhsSVS)
    return std::make_pair(lhsSVS, lhsPri);
  return failure();
}

wave::IndexExprsLatticeStorage
wave::IndexExprsLatticeStorage::join(const IndexExprsLatticeStorage &lhs,
                                     const IndexExprsLatticeStorage &rhs) {
  if (lhs == rhs)
    return lhs;

  // Top is saturating.
  if (lhs.isTop() || rhs.isTop())
    return top();

  // Bottom is neutral.
  if (lhs.isBottom())
    return rhs;
  if (rhs.isBottom())
    return lhs;

  MLIRContext *ctx = lhs.getConcreteValue().getContext();
  WaveSymbolMappingAttr lhsMapping = lhs.getConcreteValue();
  WaveSymbolMappingAttr rhsMapping = rhs.getConcreteValue();

  // Join specific values per symbol using per-key priorities.
  IntegerType i32 = IntegerType::get(ctx, 32);
  llvm::MapVector<wave::WaveSymbolAttr, Attribute> result;
  llvm::MapVector<wave::WaveSymbolAttr, int32_t> resultPriorities;
  for (auto [key, val] : lhsMapping) {
    result[key] = val;
    resultPriorities[key] = lhs.getPriorityForKey(key);
  }
  for (auto [key, val] : rhsMapping) {
    int32_t rhsPriority = rhs.getPriorityForKey(key);

    auto [it, inserted] = result.try_emplace(key, val);
    if (inserted) {
      resultPriorities[key] = rhsPriority;
      continue;
    }

    int32_t lhsPriority = lhs.getPriorityForKey(key);

    auto lhsIdx = llvm::cast<wave::WaveIndexMappingAttr>(it->second);
    auto rhsIdx = llvm::cast<wave::WaveIndexMappingAttr>(val);
    if (lhsIdx == rhsIdx) {
      resultPriorities[key] = std::max(lhsPriority, rhsPriority);
      continue;
    }

    wave::WaveIndexMappingAttr joinedIdx =
        getIndexExprsJoinMappings(lhsIdx, rhsIdx, lhsPriority, rhsPriority);
    if (!joinedIdx)
      return IndexExprsLatticeStorage::top();

    result[key] = joinedIdx;
    resultPriorities[key] = std::max(lhsPriority, rhsPriority);
  }

  wave::WaveSymbolMappingAttr joinedVectorShape =
      ::getJoinedVectorShape(lhs.getVectorShape(), rhs.getVectorShape(),
                             lhs.getPriorities(), rhs.getPriorities());
  if (!joinedVectorShape && (lhs.getVectorShape() || rhs.getVectorShape()))
    return IndexExprsLatticeStorage::top();

  // Join source vector shapes using priority: higher priority wins. If
  // priorities are equal, two different values cause top; two identical values
  // join cleanly.
  auto joinedSVSResult = getJoinedSourceVectorShape(lhs, rhs);
  if (failed(joinedSVSResult))
    return IndexExprsLatticeStorage::top();
  auto [joinedSourceVectorShape, joinedSourceVectorShapePriority] =
      *joinedSVSResult;

  SmallVector<std::pair<wave::WaveSymbolAttr, Attribute>> priEntries;
  priEntries.reserve(resultPriorities.size());
  for (auto &[key, pri] : resultPriorities)
    priEntries.emplace_back(key, IntegerAttr::get(i32, pri));

  SmallVector<std::pair<wave::WaveSymbolAttr, Attribute>> joinedEntries;
  joinedEntries.reserve(result.size());
  for (auto &[key, val] : result)
    joinedEntries.emplace_back(key, val);

  IndexExprsLatticeStorage joined(
      WaveSymbolMappingAttr::get(ctx, joinedEntries),
      WaveSymbolMappingAttr::get(ctx, priEntries), joinedVectorShape,
      joinedSourceVectorShape, joinedSourceVectorShapePriority);
  return joined;
}

wave::IndexExprsLatticeStorage
wave::IndexExprsLatticeStorage::meet(const IndexExprsLatticeStorage &lhs,
                                     const IndexExprsLatticeStorage &rhs) {
  return join(lhs, rhs);
}

void wave::IndexExprsLatticeStorage::unsafeSet(
    const IndexExprsLatticeStorage &value) {
  operator=(value);
}

wave::IndexExprsLatticeStorage wave::IndexExprsLatticeStorage::keepOnlySymbols(
    llvm::ArrayRef<wave::WaveSymbolAttr> symbols) const {
  if (isBottom() || isTop())
    return *this;

  llvm::DenseSet<wave::WaveSymbolAttr> symbolSet(symbols.begin(),
                                                 symbols.end());

  WaveSymbolMappingAttr mapping = getConcreteValue();
  MLIRContext *ctx = getConcreteValue().getContext();
  IntegerType i32 = IntegerType::get(ctx, 32);
  SmallVector<std::pair<wave::WaveSymbolAttr, Attribute>> filtered;
  SmallVector<std::pair<wave::WaveSymbolAttr, Attribute>> filteredPri;
  for (auto [key, val] : mapping) {
    if (symbolSet.contains(key)) {
      filtered.emplace_back(key, val);
      filteredPri.emplace_back(key,
                               IntegerAttr::get(i32, getPriorityForKey(key)));
    }
  }
  wave::WaveSymbolMappingAttr filteredVectorShape =
      detail::filterVectorShape(getVectorShape(), symbols);

  if (filtered.empty())
    return bottom();

  return IndexExprsLatticeStorage(WaveSymbolMappingAttr::get(ctx, filtered),
                                  WaveSymbolMappingAttr::get(ctx, filteredPri),
                                  filteredVectorShape, sourceVectorShape,
                                  sourceVectorShapePriority);
}

wave::IndexExprsLatticeStorage
wave::IndexExprsLatticeStorage::withoutIterSymbols(
    llvm::ArrayRef<wave::WaveSymbolAttr> iterSymbols) const {
  if (isBottom() || isTop())
    return *this;

  MLIRContext *ctx = getConcreteValue().getContext();
  WaveSymbolMappingAttr mapping = getConcreteValue();
  SmallVector<std::pair<wave::WaveSymbolAttr, Attribute>> updated;
  updated.reserve(mapping.size());
  for (auto [key, val] : mapping) {
    auto idxMapping = llvm::cast<wave::WaveIndexMappingAttr>(val);
    for (wave::WaveSymbolAttr iterSymbol : iterSymbols) {
      auto actualIterSymbol =
          wave::WaveIterSymbolAttr::get(ctx, iterSymbol.getName());
      idxMapping = idxMapping.removeInput(actualIterSymbol);
    }
    updated.emplace_back(key, idxMapping);
  }
  IndexExprsLatticeStorage result(WaveSymbolMappingAttr::get(ctx, updated),
                                  getPriorities(), getVectorShape());
  result.sourceVectorShape = sourceVectorShape;
  result.sourceVectorShapePriority = sourceVectorShapePriority;
  return result;
}

void wave::IndexExprsLatticeStorage::print(llvm::raw_ostream &os) const {
  if (isBottom()) {
    os << "<bottom>";
  } else if (isTop()) {
    os << "<top>";
  } else {
    os << "[pri: {";
    bool first = true;
    for (auto [key, _] : getConcreteValue()) {
      if (!first)
        os << ", ";
      first = false;
      os << key.getName() << "=" << getPriorityForKey(key);
    }
    os << "}] " << getConcreteValue();
    if (auto shape = getVectorShape())
      os << " vectorShape: " << shape;
    if (auto svs = getSourceVectorShape())
      os << " sourceVectorShape(pri=" << getSourceVectorShapePriority()
         << "): " << svs;
  }
}

void wave::IndexExprsLatticeStorage::dump() const { print(llvm::errs()); }

void wave::operator<<(Diagnostic &diag, const IndexExprsLatticeStorage &value) {
  std::string str;
  llvm::raw_string_ostream os(str);
  value.print(os);
  diag << os.str();
}

llvm::raw_ostream &
llvm::operator<<(llvm::raw_ostream &os,
                 const wave::IndexExprsLatticeStorage &lattice) {
  lattice.print(os);
  return os;
}

llvm::FailureOr<ChangeResult> wave::detail::identityIndexExprsPropagate(
    llvm::ArrayRef<IndexExprsLatticeStorage> from,
    llvm::MutableArrayRef<IndexExprsLatticeStorage> to, ValueRange toValues,

    llvm::StringRef fromName, llvm::StringRef toName,
    wave::EmitErrorFn emitError) {
  ChangeResult changeResult = ChangeResult::NoChange;
  for (auto &&[toNum, toLattice, toValue] : llvm::enumerate(to, toValues)) {
    if (toLattice.isTop())
      continue;

    auto toTensorType = dyn_cast<WaveTensorType>(toValue.getType());
    if (!toTensorType || !toTensorType.getFullySpecified())
      continue;

    for (auto &&[fromNum, fromLattice] : llvm::enumerate(from)) {
      bool fromTop = false;

      // If one of the from lattices reached the top, no need to keep joining,
      // the result is known to be top. The error will have been reported
      // already, no need to repeat it.
      if (fromLattice.isTop()) {
        fromTop = true;
        toLattice = IndexExprsLatticeStorage::top();
        changeResult = ChangeResult::Change;
        break;
      }

      // XXX: a more efficient way would have been to join all "from" lattices
      // first, and then join that into each "to" lattice. But this heuristic
      // would not work in that case.
      if (!wave::detail::shouldPropagateIndexExprs(fromLattice, toLattice,
                                                   toValue)) {
        LLVM_DEBUG(LDBG() << "not propagating index expressions from "
                          << fromName << " #" << fromNum << " to " << toName
                          << " #" << toNum << "\n";);
        continue;
      }

      IndexExprsLatticeStorage joined =
          IndexExprsLatticeStorage::join(toLattice, fromLattice);
      if (joined.isTop() && !fromTop) {
        StringRef conflictKind = "index expressions";
        if (failed(IndexExprsLatticeStorage::getJoinedVectorShape(
                toLattice, fromLattice)) &&
            toLattice.getVectorShape() && fromLattice.getVectorShape())
          conflictKind = "vector shapes";
        else if (failed(IndexExprsLatticeStorage::getJoinedSourceVectorShape(
                     toLattice, fromLattice)))
          conflictKind = "source vector shapes";
        InFlightDiagnostic diag =
            emitError() << "conflict when propagating " << conflictKind
                        << " from " << fromName << " #" << fromNum << " to "
                        << toName << " #" << toNum;
        diag.attachNote() << "original " << toName << " lattice: " << toLattice;
        diag.attachNote() << fromName << " #" << fromNum
                          << " lattice: " << fromLattice;
        return diag;
      }
      joined = joined.keepOnlySymbols(toTensorType.getShape());
      if (joined != toLattice) {
        changeResult = ChangeResult::Change;
        toLattice = joined;
      }
    }
  }

  return changeResult;
}

llvm::LogicalResult wave::detail::checkAndAppendIndexExpr(
    Location loc, const IndexExprsLatticeStorage &expr,
    const llvm::Twine &description,
    llvm::SmallVectorImpl<Attribute> &indexExprs) {
  if (expr.isBottom()) {
    emitError(loc) << "failed to infer index expressions for " << description;
    return llvm::failure();
  }
  if (expr.isTop()) {
    InFlightDiagnostic diag = emitError(loc)
                              << "conflict detected in index expressions for "
                              << description;
    diag.attachNote() << "PLEASE REPORT this as a bug in absence of further "
                         "information about the conflict";
    return llvm::failure();
  }
  indexExprs.push_back(expr.getConcreteValue());
  return llvm::success();
}

//-----------------------------------------------------------------------------
// Op-specific index expression inference (from WaveOps.cpp)
//-----------------------------------------------------------------------------

llvm::FailureOr<ChangeResult> wave::IterateOp::propagateIndexExprsForward(
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> /*operands*/,
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> /*results*/,
    wave::EmitErrorFn /*emitError*/) {
  llvm_unreachable("IterateOp should be handled by control flow interfaces");
}

llvm::FailureOr<ChangeResult> wave::IterateOp::propagateIndexExprsBackward(
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> /*operands*/,
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> /*results*/,
    wave::EmitErrorFn /*emitError*/) {
  llvm_unreachable("IterateOp should be handled by control flow interfaces");
}

// Set the value of `lattice` to `newLattice` and return whether a change
// happened. Note that this does NOT verify whether the lattice change goes into
// the direction of top or bottom.
static ChangeResult
updateIfChanged(wave::IndexExprsLatticeStorage &lattice,
                const wave::IndexExprsLatticeStorage &newLattice) {
  if (newLattice == lattice)
    return ChangeResult::NoChange;
  lattice = newLattice;
  return ChangeResult::Change;
}

// No propagation through MMA. The index expressions remain the same as set by
// initialization since MMAs require very specific index expressions. If there
// is a conflict with operands that were propagated from another MMA (other
// operations have lower priority), it will be resolved in a separate pass after
// the analysis completes.
llvm::FailureOr<ChangeResult> wave::MmaOp::propagateIndexExprsForward(
    llvm::ArrayRef<wave::IndexExprsLatticeStorage>,
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage>, wave::EmitErrorFn) {
  return ChangeResult::NoChange;
}
llvm::FailureOr<ChangeResult> wave::MmaOp::propagateIndexExprsBackward(
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage>,
    llvm::ArrayRef<wave::IndexExprsLatticeStorage>, wave::EmitErrorFn) {
  return ChangeResult::NoChange;
}
llvm::FailureOr<ChangeResult> wave::ScaledMmaOp::propagateIndexExprsForward(
    llvm::ArrayRef<wave::IndexExprsLatticeStorage>,
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage>, wave::EmitErrorFn) {
  return ChangeResult::NoChange;
}
llvm::FailureOr<ChangeResult> wave::ScaledMmaOp::propagateIndexExprsBackward(
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage>,
    llvm::ArrayRef<wave::IndexExprsLatticeStorage>, wave::EmitErrorFn) {
  return ChangeResult::NoChange;
}

// Extract the context from the first symbol that is not null.
static MLIRContext *getAnySymbolContext(wave::WaveSymbolAttr mSymbol,
                                        wave::WaveSymbolAttr nSymbol,
                                        wave::WaveSymbolAttr kSymbol) {
  MLIRContext *context = nullptr;
  for (wave::WaveSymbolAttr symbol : {mSymbol, nSymbol, kSymbol})
    if (!context && symbol)
      context = symbol.getContext();
  assert(context && "expected at least one symbol name to be provided");
  return context;
}

namespace {

struct MmaIndexingExprBuilder;

// Fluent-style API builder for index expressions of an MMA operation. See
// MmaIndexingExprBuilder for details.
struct MmaSingleIndexExprBuilder {
  MmaSingleIndexExprBuilder(MmaIndexingExprBuilder &parent, bool enabled)
      : parent(parent), enabled(enabled) {}

  // Set the parameter of the index expression for the currently selected
  // dimension.
  MmaSingleIndexExprBuilder &offset(AffineExpr expr);
  MmaSingleIndexExprBuilder &size(int64_t value);
  MmaSingleIndexExprBuilder &stride(int64_t value);

  // Select the dimension.
  MmaSingleIndexExprBuilder &m();
  MmaSingleIndexExprBuilder &n();
  MmaSingleIndexExprBuilder &k();

  // Populate the attributes with all index expressions.
  void populate(SmallVectorImpl<std::pair<WaveSymbolAttr, WaveIndexMappingAttr>>
                    &entries) const;

  MmaIndexingExprBuilder &parent;
  AffineExpr offsetExpr, sizeExpr, strideExpr;
  bool enabled;
};

// Fluent-style API builder for index expressions of an MMA operation. Usage:
//   1. Create an instance of this class.
//   2. Use `m`, `n` or `k` to select the MMA dimension to build an index
//   expression for.
//   3. After selecting the dimension, use `offset`, `size` or `stride` to set
//   the corresponding quantities of the index expression.
//   4. Proceed with the next dimension until all dimensions are set.
//   5. Call `populate` to populate the attributes of the MMA operation.
//
// Example:
//
// ```
//   MmaIndexingExprBuilder builder(symbols, mSymbol, nSymbol, kSymbol);
//   builder.m().offset(offset_m).size(size_m).stride(stride_m)
//          .n().offset(offset_n).size(size_n).stride(stride_n)
//          .k().offset(offset_k).size(size_k).stride(stride_k)
//          .populate(attributes);
// ```
struct MmaIndexingExprBuilder {
  MmaIndexingExprBuilder(llvm::ArrayRef<Attribute> symbols,
                         wave::WaveSymbolAttr mSymbol,
                         wave::WaveSymbolAttr nSymbol,
                         wave::WaveSymbolAttr kSymbol)
      : symbols(symbols), mBuilder(*this, mSymbol != nullptr),
        nBuilder(*this, nSymbol != nullptr),
        kBuilder(*this, kSymbol != nullptr), mSymbol(mSymbol), nSymbol(nSymbol),
        kSymbol(kSymbol) {
    assert(
        llvm::all_of(
            symbols,
            llvm::IsaPred<wave::WaveSymbolAttr, wave::WaveIndexSymbolAttr>) &&
        "expected symbols to be a range of WaveSymbolAttr or "
        "WaveIndexSymbolAttr attributes");
  }

  // Select the dimension.
  MmaSingleIndexExprBuilder &m() { return mBuilder; }
  MmaSingleIndexExprBuilder &n() { return nBuilder; }
  MmaSingleIndexExprBuilder &k() { return kBuilder; }

  // Populate the attributes with all index expressions.
  void populate(SmallVectorImpl<std::pair<WaveSymbolAttr, WaveIndexMappingAttr>>
                    &entries) const {
    MLIRContext *ctx = getAnySymbolContext(mSymbol, nSymbol, kSymbol);

    auto buildMap = [&](AffineExpr expr) {
      assert(expr &&
             "expected offset/size/stride to be set up for all symbols");
      return AffineMap::get(/*dimCount=*/0,
                            /*symbolCount=*/symbols.size(), expr, ctx);
    };
    auto buildOne = [&](const MmaSingleIndexExprBuilder &builder) {
      return wave::WaveIndexMappingAttr::get(
          ctx, symbols, buildMap(builder.offsetExpr),
          buildMap(builder.sizeExpr), buildMap(builder.strideExpr));
    };

    if (mSymbol)
      entries.emplace_back(mSymbol, buildOne(mBuilder));
    if (nSymbol)
      entries.emplace_back(nSymbol, buildOne(nBuilder));
    if (kSymbol)
      entries.emplace_back(kSymbol, buildOne(kBuilder));
  }

  llvm::ArrayRef<Attribute> symbols;
  MmaSingleIndexExprBuilder mBuilder, nBuilder, kBuilder;
  wave::WaveSymbolAttr mSymbol, nSymbol, kSymbol;
};

MmaSingleIndexExprBuilder &MmaSingleIndexExprBuilder::offset(AffineExpr expr) {
  if (!enabled)
    return *this;
  assert(!offsetExpr && "expected offset to be set only once");
  offsetExpr = expr;
  return *this;
}

MmaSingleIndexExprBuilder &MmaSingleIndexExprBuilder::size(int64_t value) {
  if (!enabled)
    return *this;
  assert(!sizeExpr && "expected size to be set only once");
  sizeExpr = getAffineConstantExpr(value, offsetExpr.getContext());
  return *this;
}

MmaSingleIndexExprBuilder &MmaSingleIndexExprBuilder::stride(int64_t value) {
  if (!enabled)
    return *this;
  assert(!strideExpr && "expected stride to be set only once");
  strideExpr = getAffineConstantExpr(value, offsetExpr.getContext());
  return *this;
}

[[maybe_unused]] MmaSingleIndexExprBuilder &MmaSingleIndexExprBuilder::m() {
  return parent.m();
}
[[maybe_unused]] MmaSingleIndexExprBuilder &MmaSingleIndexExprBuilder::n() {
  return parent.n();
}
[[maybe_unused]] MmaSingleIndexExprBuilder &MmaSingleIndexExprBuilder::k() {
  return parent.k();
}
void MmaSingleIndexExprBuilder::populate(
    SmallVectorImpl<std::pair<WaveSymbolAttr, WaveIndexMappingAttr>> &entries)
    const {
  parent.populate(entries);
}
} // namespace

/// MXFP group scale factor for GFX950 and GFX1250.
static constexpr unsigned kScaleGroupSize = 32;

// Get the attribute representing the vector shape implied by the given MMA
// operation kind for its M, N, K dimensions.
static wave::WaveSymbolMappingAttr
getMmaVectorShape(Location loc, wave::WaveMmaKind kind,
                  wave::WaveSymbolAttr mSymbol, wave::WaveSymbolAttr nSymbol,
                  wave::WaveSymbolAttr kSymbol,
                  wave::WaveSymbolAttr kScaledSymbol,
                  ArrayRef<wave::WaveSymbolAttr> batchSymbols,
                  wave::WaveSymbolMappingAttr hwVectorShape,
                  wave::EmitDelayedErrorFn *delayedErrorEmitter) {
  int m, n, k, kScaled = 0;
  switch (kind) {
  case wave::WaveMmaKind::F32_16x16x16_F16:
  case wave::WaveMmaKind::I32_16x16x16_I8:
    m = 16;
    n = 16;
    k = 16;
    break;
  case wave::WaveMmaKind::F32_32x32x8_F16:
  case wave::WaveMmaKind::I32_32x32x8_I8:
    m = 32;
    n = 32;
    k = 8;
    break;
  case wave::WaveMmaKind::F32_16x16x32_F8:
  case wave::WaveMmaKind::F32_16x16x32_BF16:
  case wave::WaveMmaKind::F32_16x16x32_F16:
  case wave::WaveMmaKind::F32_16x16x32_K8_F16:
  case wave::WaveMmaKind::I32_16x16x32_I8:
  case wave::WaveMmaKind::F32_16x16x32_K4_F8:
    m = 16;
    n = 16;
    k = 32;
    break;
  case wave::WaveMmaKind::F32_32x32x16_F8:
  case wave::WaveMmaKind::F32_32x32x16_BF16:
  case wave::WaveMmaKind::F32_32x32x16_F16:
  case wave::WaveMmaKind::F32_32x32x16_K8_F16:
  case wave::WaveMmaKind::I32_32x32x16_I8:
  case wave::WaveMmaKind::F32_32x32x16_K4_F8:
    m = 32;
    n = 32;
    k = 16;
    break;
  case wave::WaveMmaKind::F32_16x16x128_F8F6F4:
    m = 16;
    n = 16;
    k = 128;
    kScaled = k / kScaleGroupSize;
    break;
  case wave::WaveMmaKind::F32_32x32x64_F8F6F4:
    m = 32;
    n = 32;
    k = 64;
    kScaled = k / kScaleGroupSize;
    break;
  }

  MLIRContext *ctx = loc->getContext();
  auto iAttr = [&](int64_t value) {
    return IntegerAttr::get(IntegerType::get(ctx, 64), value);
  };
  SmallVector<std::pair<wave::WaveSymbolAttr, Attribute>> entries;
  if (mSymbol)
    entries.emplace_back(mSymbol, iAttr(m));
  if (nSymbol)
    entries.emplace_back(nSymbol, iAttr(n));
  if (kSymbol)
    entries.emplace_back(kSymbol, iAttr(k));
  if (kScaledSymbol)
    entries.emplace_back(kScaledSymbol, iAttr(kScaled));

  if (hwVectorShape) {
    for (auto [key, value] : hwVectorShape) {
      if (mSymbol && key == mSymbol) {
        if (delayedErrorEmitter && cast<IntegerAttr>(value).getValue() != m) {
          wave::EmitDelayedErrorFn previous = *delayedErrorEmitter;
          *delayedErrorEmitter = [mSymbol, m,
                                  previous](InFlightDiagnostic &diag) {
            if (previous)
              previous(diag);
            diag << "overriding vector shape for " << mSymbol << " to " << m
                 << " implied by the MMA operation";
          };
        }
        continue;
      }
      if (nSymbol && key == nSymbol) {
        if (delayedErrorEmitter && cast<IntegerAttr>(value).getValue() != n) {
          wave::EmitDelayedErrorFn previous = *delayedErrorEmitter;
          *delayedErrorEmitter = [nSymbol, n,
                                  previous](InFlightDiagnostic &diag) {
            if (previous)
              previous(diag);
            diag << "overriding vector shape for " << nSymbol << " to " << n
                 << " implied by the MMA operation";
          };
        }
        continue;
      }
      if (kSymbol && key == kSymbol) {
        if (delayedErrorEmitter && cast<IntegerAttr>(value).getValue() != k) {
          wave::EmitDelayedErrorFn previous = *delayedErrorEmitter;
          *delayedErrorEmitter = [kSymbol, k,
                                  previous](InFlightDiagnostic &diag) {
            if (previous)
              previous(diag);
            diag << "overriding vector shape for " << kSymbol << " to " << k
                 << " implied by the MMA operation";
          };
        }
        continue;
      }
      if (kScaledSymbol && key == kScaledSymbol) {
        if (delayedErrorEmitter &&
            cast<IntegerAttr>(value).getValue() != kScaled) {
          wave::EmitDelayedErrorFn previous = *delayedErrorEmitter;
          *delayedErrorEmitter = [kScaledSymbol, kScaled,
                                  previous](InFlightDiagnostic &diag) {
            if (previous)
              previous(diag);
            diag << "overriding vector shape for " << kScaledSymbol << " to "
                 << kScaled << " implied by the Scaled MMA operation";
          };
        }
        continue;
      }
      if (llvm::is_contained(batchSymbols, key))
        entries.emplace_back(key, value);
    }
  }

  return wave::WaveSymbolMappingAttr::get(ctx, entries);
}

// Populate `attributes` with index expressions for the symbols associated with
// M, N, K dimensions of the given MMA operation kind provided the configuration
// of wavefronts in the workgroup. Any symbol may be omitted as long as at least
// one is provided, e.g., for the LHS of the operation, only M and N symbols may
// be provided. If `isAccumulator` is set, the index expressions are created for
// the accumulator/result of an MMA, which may affect the expression for the M
// dimension. If `isScaled` is set, the K expression uses the scale operand
// mapping (one scale per kScaleGroupSize K-elements). If `isFP4` is set,
// the K expression uses the FP4 data mapping. Both `isScaled` and `isFP4`
// currently produce the same K offset formula (they are ORed together), but
// are kept separate because they represent distinct semantic cases: `isScaled`
// refers to E8M0 scale operands while `isFP4` refers to FP4 data operands.
static llvm::LogicalResult populateMmaIndexingExpr(
    wave::WaveMmaKind kind, bool isAccumulator, bool isScaled, bool isFP4,
    llvm::ArrayRef<unsigned> wavesPerWorkgroup, int64_t threadsPerWave,
    wave::WaveSymbolAttr mSymbol, wave::WaveSymbolAttr nSymbol,
    wave::WaveSymbolAttr kSymbol,
    SmallVectorImpl<std::pair<WaveSymbolAttr, WaveIndexMappingAttr>> &entries) {
  MLIRContext *ctx = getAnySymbolContext(mSymbol, nSymbol, kSymbol);

  llvm::SmallVector<Attribute> symbolNames = {
      wave::WaveIndexSymbolAttr::get(ctx, wave::WaveIndexSymbol::THREAD_0),
      wave::WaveIndexSymbolAttr::get(ctx, wave::WaveIndexSymbol::THREAD_1),
      wave::WaveIndexSymbolAttr::get(ctx, wave::WaveIndexSymbol::THREAD_2),
      wave::WaveIndexSymbolAttr::get(ctx, wave::WaveIndexSymbol::GPR_NUMBER),
  };
  AffineExpr threadX, threadY, threadZ, gprNum;
  bindSymbols(ctx, threadX, threadY, threadZ, gprNum);

  AffineExpr linearizedThreadId =
      threadX + threadY * wavesPerWorkgroup[0] * threadsPerWave +
      threadZ * wavesPerWorkgroup[1] * wavesPerWorkgroup[0] * threadsPerWave;
  AffineExpr laneId = linearizedThreadId % threadsPerWave;
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
        .populate(entries);
    return llvm::success();

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
        .populate(entries);
    return llvm::success();

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
        .populate(entries);
    return llvm::success();
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
        .populate(entries);
    return llvm::success();
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
        .populate(entries);
    return llvm::success();
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
        .populate(entries);
    return llvm::success();

  case wave::WaveMmaKind::F32_16x16x128_F8F6F4:
    builder.m()
        .offset(isAccumulator ? 4 * laneId.floorDiv(16) : laneId % 16)
        .size(isAccumulator ? 4 : 1)
        .stride(isAccumulator ? 16 : 1)
        .n()
        .offset(laneId % 16)
        .size(1)
        .stride(1)
        .k()
        .offset((isScaled || isFP4)
                    ? 32 * laneId.floorDiv(16)
                    : 64 * gprNum.floorDiv(16) + 16 * laneId.floorDiv(16) +
                          (gprNum % 16))
        .size(isScaled ? 1 : 32)
        .stride(1)
        .populate(entries);
    return llvm::success();
  case wave::WaveMmaKind::F32_32x32x64_F8F6F4:
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
        .offset(32 * laneId.floorDiv(32))
        .size(32)
        .stride(1)
        .populate(entries);
    return llvm::success();
  }
}

/// Create per-symbol thread-independent index expressions for `indexingSymbols`
/// given constraints on them and put them into `symbols` and `indexExprs`.
/// Thread-independent means affected by workgroup, tiling and device
/// constraints, and NOT affected by wave constraints or MMA shapes. The first
/// argument indicates for which operation the constraints are being used, which
/// is in particular necessary to only apply tiling constraints inside the
/// relevant loops.
template <typename RangeT>
static void mixInThreadIndependentConstraints(
    Operation *where, uint64_t threadsPerWave, RangeT &&indexingSymbols,
    const llvm::DenseMap<wave::WaveSymbolAttr, llvm::SmallVector<Attribute>>
        &symbolConstraints,
    SmallVectorImpl<std::pair<wave::WaveSymbolAttr, wave::WaveIndexMappingAttr>>
        &entries) {

  static_assert(
      std::is_same_v<std::decay_t<decltype(*std::declval<RangeT>().begin())>,
                     wave::WaveSymbolAttr>,
      "expected a range of WaveSymbolAttr");

  auto zero = AffineMap::get(/*dimCount=*/0, /*numSymbols=*/0,
                             getAffineConstantExpr(0, where->getContext()));
  auto one = AffineMap::get(/*dimCount=*/0, /*numSymbols=*/0,
                            getAffineConstantExpr(1, where->getContext()));
  for (wave::WaveSymbolAttr symbol : indexingSymbols) {
    auto entryIt =
        llvm::find_if(entries, [&](auto &p) { return p.first == symbol; });
    if (entryIt == entries.end()) {
      auto mapping = wave::WaveIndexMappingAttr::get(
          where->getContext(), /*symbols=*/{}, zero, one, one);
      entryIt = entries.insert(entries.end(), {symbol, mapping});
    }

    auto it = symbolConstraints.find(symbol);
    if (it == symbolConstraints.end()) {
      continue;
    }

    wave::WaveIndexMappingAttr &mapping = entryIt->second;

    // There is interaction between constraints of different kinds for the same
    // symbol, find them all upfront.
    wave::WorkgroupConstraintAttr workgroupConstraint;
    wave::WaveConstraintAttr waveConstraint;
    wave::TilingConstraintAttr tilingConstraint;
    for (Attribute constraint : it->second) {
      if (auto maybeWorkgroupConstraint =
              dyn_cast<wave::WorkgroupConstraintAttr>(constraint)) {
        workgroupConstraint = maybeWorkgroupConstraint;
      } else if (auto maybeWaveConstraint =
                     dyn_cast<wave::WaveConstraintAttr>(constraint)) {
        waveConstraint = maybeWaveConstraint;
      } else if (auto maybeTilingConstraint =
                     dyn_cast<wave::TilingConstraintAttr>(constraint)) {
        tilingConstraint = maybeTilingConstraint;
      } else {
        llvm_unreachable("unsupported constraint type");
      }
    }

    if (tilingConstraint) {
      // Tiling constraints should only be applied inside the corresponding
      // parent iterate op.
      for (Operation *parent = where->getParentOp(); parent;
           parent = parent->getParentOp()) {
        auto iterateOp = dyn_cast<wave::IterateOp>(parent);
        if (!iterateOp)
          continue;
        wave::WaveSymbolAttr iterSymbol = iterateOp.getIterator();
        if (iterSymbol == symbol) {
          mapping = applyConstraint(tilingConstraint, mapping);
          break;
        }
      }
    }

    if (workgroupConstraint)
      mapping = applyConstraint(workgroupConstraint, mapping);

    if (waveConstraint) {
      assert(workgroupConstraint && "workgroup constraint must be present if a "
                                    "wave constraint for the same symbol is");
      mapping = applyConstraint(
          waveConstraint, workgroupConstraint.getWorkgroupDim().getValue(),
          threadsPerWave, mapping);
    }
  }
}

// Joins the given lattice with another lattice in place and handles conflicts.
// If the lattice is already top, or the join result does not change the
// lattice, the function returns success. If joining produces top (i.e., an
// undecidable or conflicting result) that differs from the original, an error
// is emitted using the provided emitError function.
static LogicalResult
joinIndexExprsLatticeInPlace(wave::IndexExprsLatticeStorage &lattice,
                             StringRef latticeName,
                             const wave::IndexExprsLatticeStorage &other,
                             StringRef otherName, wave::EmitErrorFn emitError) {
  if (lattice.isTop())
    return success();

  IndexExprsLatticeStorage joined =
      IndexExprsLatticeStorage::join(lattice, other);
  if (joined == lattice)
    return success();
  // When newly reached top, report an error.
  if (joined.isTop() && !other.isTop()) {
    StringRef conflictKind = " index expression";
    if (failed(IndexExprsLatticeStorage::getJoinedVectorShape(lattice, other)))
      conflictKind = " vector shape";
    else if (failed(IndexExprsLatticeStorage::getJoinedSourceVectorShape(
                 lattice, other)))
      conflictKind = " source vector shape";
    InFlightDiagnostic diag =
        emitError() << "conflict for " << latticeName << conflictKind
                    << " when propagating from " << otherName << " lattice";
    diag.attachNote() << "original " << latticeName << " lattice: " << lattice;
    diag.attachNote() << otherName << " lattice: " << other;
    return diag;
  }

#ifndef NDEBUG
  assert(IndexExprsLatticeStorage::join(joined, lattice) == joined &&
         "join should not move the lattice backward");
  assert(IndexExprsLatticeStorage::join(joined, other) == joined &&
         "join should not move the lattice forward");
#endif

  lattice.unsafeSet(joined);
  return success();
}

bool wave::detail::shouldPropagateIndexExprs(
    const wave::IndexExprsLatticeStorage &from,
    const wave::IndexExprsLatticeStorage &to, Value toValue) {
  if (from.isBottom() || from.isTop() || to.isTop() || to.isBottom())
    return true;

  // MMA results have fixed index expressions set during initialization; never
  // overwrite them via backward propagation from users.
  if (isa_and_nonnull<wave::MmaOp, wave::ScaledMmaOp>(
          toValue.getDefiningOp())) {
    LLVM_DEBUG(LDBG() << "skipping update to mma");
    return false;
  }

  wave::WaveSymbolMappingAttr sourceVectorShapes = from.getSourceVectorShape();
  if (!sourceVectorShapes)
    return true;

  auto toType = cast<WaveTensorType>(toValue.getType());
  ArrayRef<wave::WaveSymbolAttr> toShape = toType.getShape();
  if (from.getSourceVectorShapePriority() > 0 &&
      !llvm::all_of(toShape, [&](wave::WaveSymbolAttr symbol) {
        return sourceVectorShapes.lookup(symbol) != nullptr;
      })) {
    LLVM_DEBUG(LDBG() << "skipping update to " << toValue
                      << " because source vector shape does not include all "
                         "target dimensions");
    return false;
  }

  SmallVector<StringAttr> nonBatchDims;
  for (WaveSymbolAttr attr : toType.getShape()) {
    Attribute dimSize = sourceVectorShapes.lookup(attr);
    if (!dimSize)
      continue;
    if (llvm::cast<IntegerAttr>(dimSize).getValue().getSExtValue() > 1) {
      nonBatchDims.push_back(
          StringAttr::get(attr.getContext(), attr.getName()));
    }
  }

  SmallVector<StringAttr> toKeys =
      llvm::map_to_vector(to.getConcreteValue(), [](auto pair) {
        return StringAttr::get(pair.first.getContext(), pair.first.getName());
      });
  auto isSubset = [](ArrayRef<StringAttr> subset,
                     ArrayRef<StringAttr> superset) {
    return llvm::all_of(subset, [&](StringAttr name) {
      return llvm::is_contained(superset, name);
    });
  };
  if ((toKeys.size() < nonBatchDims.size() &&
       !isSubset(toKeys, nonBatchDims)) ||
      !isSubset(nonBatchDims, toKeys)) {
    LLVM_DEBUG(LDBG() << "skipping update to " << toValue
                      << " because of the subset condition");
    return false;
  }
  return true;
}

LogicalResult wave::detail::buildThreadIndependentIndexMappings(
    Operation *op, Type type, const wave::IndexExprsAnalysisInit &initObject,
    SmallVectorImpl<std::pair<wave::WaveSymbolAttr, wave::WaveIndexMappingAttr>>
        &entries) {
  auto tensorType = dyn_cast<wave::WaveTensorType>(type);
  if (!tensorType)
    return failure();

  ArrayRef<wave::WaveSymbolAttr> indexingSymbols = tensorType.getShape();
  mixInThreadIndependentConstraints(
      op, initObject.hardwareConstraint.getThreadsPerWave(), indexingSymbols,
      initObject.symbolConstraints, entries);
  return success();
}

// Initialize the index expression lattices for the result of the MMA operation.
// This sets index expressions to values derived from the MMA operation kind and
// wavefront-in-workgroup configuration (thread-dependent) as well as workgroup
// constraints (thread-independent).
LogicalResult MmaOp::initializeIndexExprsForward(
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> resultExprs,
    const IndexExprsAnalysisInit &initObject, wave::EmitErrorFn emitError) {
  ArrayRef<wave::WaveSymbolAttr> indexingSymbols =
      cast<wave::WaveTensorType>(getResult().getType()).getShape();
  SmallVector<std::pair<wave::WaveSymbolAttr, wave::WaveIndexMappingAttr>>
      entries;
  entries.reserve(indexingSymbols.size());

  assert(indexingSymbols.size() >= 2 &&
         "at least 2 indexing symbols are required for MMA result");
  wave::WaveSymbolAttr mSymbol = indexingSymbols.drop_back().back();
  wave::WaveSymbolAttr nSymbol = indexingSymbols.back();

  std::optional<wave::WaveMmaKind> mmaKind = getKind();
  if (!mmaKind)
    return emitError() << "MMA operation without kind attribute not supported";
  if (llvm::failed(populateMmaIndexingExpr(
          *mmaKind,
          /*isAccumulator=*/true, /*isScaled=*/false, /*isFP4=*/false,
          initObject.wavesPerBlock,
          initObject.hardwareConstraint.getThreadsPerWave(), mSymbol, nSymbol,
          /*kSymbol=*/nullptr, entries))) {
    return emitError() << "MMA kind not supported by index deduction";
  }

  // Set the priority based on the order of operations: earlier MMAs have higher
  // priority.
  auto orderedAllMmas = llvm::make_filter_range(
      initObject.deterministicOpOrder, llvm::IsaPred<MmaOp, ScaledMmaOp>);
  int32_t priority = wave::IndexExprsLatticeStorage::kMmaPriority +
                     std::distance(llvm::find(orderedAllMmas, getOperation()),
                                   orderedAllMmas.end()) -
                     1;
  mixInThreadIndependentConstraints(
      *this, initObject.hardwareConstraint.getThreadsPerWave(), indexingSymbols,
      initObject.symbolConstraints, entries);
  SmallVector<std::pair<wave::WaveSymbolAttr, mlir::Attribute>> resultEntries(
      entries.begin(), entries.end());
  return joinIndexExprsLatticeInPlace(
      resultExprs[0], "MMA result",
      wave::IndexExprsLatticeStorage(
          wave::WaveSymbolMappingAttr::get(getContext(), resultEntries),
          priority,
          getMmaVectorShape(getLoc(), *mmaKind, mSymbol, nSymbol,
                            /*kSymbol=*/nullptr,
                            /*kScaledSymbol=*/nullptr,
                            /*batchSymbols=*/indexingSymbols.drop_back(2),
                            initObject.hardwareConstraint.getVectorShapes(),
                            /*delayedErrorEmitter=*/nullptr)),
      "implied by MMA kind", emitError);
}

// Populate the filteredSymbols and filteredIndexExprs with the symbols and
// index expressions except the excluded one.
static void filterSymbol(
    llvm::ArrayRef<std::pair<wave::WaveSymbolAttr, wave::WaveIndexMappingAttr>>
        entries,
    WaveSymbolAttr exclude,
    SmallVectorImpl<std::pair<wave::WaveSymbolAttr, wave::WaveIndexMappingAttr>>
        &filtered) {
  filtered.reserve(filtered.size() + entries.size());
  for (auto &entry : entries)
    if (entry.first != exclude)
      filtered.push_back(entry);
}

// Initialize the index expression lattices for the operands of the MMA
// operation. This sets index expressions to values derived from the MMA
// operation kind and wavefront-in-workgroup configuration (thread-dependent) as
// well as workgroup constraints (thread-independent).
LogicalResult MmaOp::initializeIndexExprsBackward(
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> operandExprs,
    const wave::IndexExprsAnalysisInit &initObject, wave::EmitErrorFn emitError,
    wave::EmitDelayedErrorFn &delayedErrorEmitter) {
  auto resultType = llvm::cast<wave::WaveTensorType>(getResult().getType());
  auto lhsType = llvm::cast<wave::WaveTensorType>(getLhs().getType());
  assert(resultType.getRank() == lhsType.getRank() && lhsType.getRank() >= 2 &&
         "at least 2D MMA operations are supported");
  wave::WaveSymbolAttr mSymbol = resultType.getShape().drop_back().back();
  wave::WaveSymbolAttr nSymbol = resultType.getShape().back();
  wave::WaveSymbolAttr kSymbol = lhsType.getShape().back();

  std::optional<wave::WaveMmaKind> mmaKind = getKind();
  if (!mmaKind)
    return emitError() << "MMA operation without kind attribute not supported";

  // Reserve space for 1 more since the list will initially contain m,n,k along
  // with batch dimensions until we drop either m or n for each operand.
  SmallVector<std::pair<wave::WaveSymbolAttr, wave::WaveIndexMappingAttr>>
      operandEntries;
  operandEntries.reserve(lhsType.getShape().size() + 1);
  if (llvm::failed(populateMmaIndexingExpr(
          *mmaKind, /*isAccumulator=*/false, /*isScaled=*/false,
          /*isFP4=*/false, initObject.wavesPerBlock,
          initObject.hardwareConstraint.getThreadsPerWave(), mSymbol, nSymbol,
          kSymbol, operandEntries))) {
    return emitError() << "MMA kind not supported by index deduction";
  }

  SmallVector<std::pair<wave::WaveSymbolAttr, wave::WaveIndexMappingAttr>>
      accumulatorEntries;
  accumulatorEntries.reserve(resultType.getShape().size());
  if (llvm::failed(populateMmaIndexingExpr(
          *mmaKind,
          /*isAccumulator=*/true, /*isScaled=*/false, /*isFP4=*/false,
          initObject.wavesPerBlock,
          initObject.hardwareConstraint.getThreadsPerWave(), mSymbol, nSymbol,
          nullptr, accumulatorEntries))) {
    return emitError() << "MMA kind not supported by index deduction";
  }

  ArrayRef<wave::WaveSymbolAttr> batchSymbols =
      resultType.getShape().drop_back(2);
  mixInThreadIndependentConstraints(
      *this, initObject.hardwareConstraint.getThreadsPerWave(),
      llvm::concat<const WaveSymbolAttr>(batchSymbols,
                                         ArrayRef{mSymbol, nSymbol, kSymbol}),
      initObject.symbolConstraints, operandEntries);
  mixInThreadIndependentConstraints(
      *this, initObject.hardwareConstraint.getThreadsPerWave(),
      llvm::concat<const WaveSymbolAttr>(batchSymbols,
                                         ArrayRef{mSymbol, nSymbol}),
      initObject.symbolConstraints, accumulatorEntries);

  SmallVector<std::pair<wave::WaveSymbolAttr, wave::WaveIndexMappingAttr>>
      lhsEntries;
  filterSymbol(operandEntries, nSymbol, lhsEntries);
  SmallVector<std::pair<wave::WaveSymbolAttr, wave::WaveIndexMappingAttr>>
      rhsEntries;
  filterSymbol(operandEntries, mSymbol, rhsEntries);

  // Set the priority based on the order of operations: earlier MMAs have higher
  // priority.
  auto orderedAllMmas = llvm::make_filter_range(
      initObject.deterministicOpOrder, llvm::IsaPred<MmaOp, ScaledMmaOp>);
  int32_t priority = wave::IndexExprsLatticeStorage::kMmaPriority +
                     std::distance(llvm::find(orderedAllMmas, getOperation()),
                                   orderedAllMmas.end()) -
                     1;
  SmallVector<std::pair<wave::WaveSymbolAttr, mlir::Attribute>>
      lhsMappingEntries(lhsEntries.begin(), lhsEntries.end());
  if (failed(joinIndexExprsLatticeInPlace(
          operandExprs[getLhsMutable().getOperandNumber()], "LHS",
          wave::IndexExprsLatticeStorage(
              wave::WaveSymbolMappingAttr::get(getContext(), lhsMappingEntries),
              priority,
              getMmaVectorShape(getLoc(), *mmaKind, mSymbol,
                                /*nSymbol=*/nullptr, kSymbol,
                                /*kScaledSymbol=*/nullptr, batchSymbols,
                                initObject.hardwareConstraint.getVectorShapes(),
                                &delayedErrorEmitter)),
          "implied by MMA kind", emitError)))
    return failure();
  SmallVector<std::pair<wave::WaveSymbolAttr, mlir::Attribute>>
      rhsMappingEntries(rhsEntries.begin(), rhsEntries.end());
  if (failed(joinIndexExprsLatticeInPlace(
          operandExprs[getRhsMutable().getOperandNumber()], "RHS",
          wave::IndexExprsLatticeStorage(
              wave::WaveSymbolMappingAttr::get(getContext(), rhsMappingEntries),
              priority,
              getMmaVectorShape(getLoc(), *mmaKind,
                                /*mSymbol=*/nullptr, nSymbol, kSymbol,
                                /*kScaledSymbol=*/nullptr, batchSymbols,
                                initObject.hardwareConstraint.getVectorShapes(),
                                &delayedErrorEmitter)),
          "implied by MMA kind", emitError)))
    return failure();
  SmallVector<std::pair<wave::WaveSymbolAttr, mlir::Attribute>>
      accMappingEntries(accumulatorEntries.begin(), accumulatorEntries.end());
  if (failed(joinIndexExprsLatticeInPlace(
          operandExprs[getAccumulatorMutable().getOperandNumber()],
          "accumulator",
          wave::IndexExprsLatticeStorage(
              wave::WaveSymbolMappingAttr::get(getContext(), accMappingEntries),
              priority,
              getMmaVectorShape(getLoc(), *mmaKind, mSymbol, nSymbol,
                                /*kSymbol=*/nullptr,
                                /*kScaledSymbol=*/nullptr, batchSymbols,
                                initObject.hardwareConstraint.getVectorShapes(),
                                &delayedErrorEmitter)),
          "implied by MMA kind", emitError)))
    return failure();
  return success();
}

// Special case for MMA where we also want to have index expressions
// for the operands.
// TODO: this shouldn't be strictly necessary in a purely MLIR flow,
// but is kept for Python compatibility.
std::function<void(raw_ostream &, unsigned)>
MmaOp::getIndexExprValuesAndDescriptions(llvm::SmallVectorImpl<Value> &values) {
  values.reserve(4);
  llvm::append_range(values, getOperands());
  values.push_back(getResult());
  unsigned lhsPosition = getLhsMutable().getOperandNumber();
  unsigned rhsPosition = getRhsMutable().getOperandNumber();
  unsigned accumulatorPosition = getAccumulatorMutable().getOperandNumber();
  return [lhsPosition, rhsPosition, accumulatorPosition](raw_ostream &os,
                                                         unsigned i) {
    assert(i < 4 && "unexpected position");
    if (i == lhsPosition)
      os << "lhs";
    else if (i == rhsPosition)
      os << "rhs";
    else if (i == accumulatorPosition)
      os << "accumulator";
    else
      os << "result";
  };
}

LogicalResult wave::ScaledMmaOp::initializeIndexExprsForward(
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> resultExprs,
    const IndexExprsAnalysisInit &initObject, wave::EmitErrorFn emitError) {
  ArrayRef<wave::WaveSymbolAttr> indexingSymbols =
      cast<wave::WaveTensorType>(getResult().getType()).getShape();
  SmallVector<std::pair<wave::WaveSymbolAttr, wave::WaveIndexMappingAttr>>
      entries;
  entries.reserve(indexingSymbols.size());

  assert(indexingSymbols.size() >= 2 &&
         "at least 2 indexing symbols are required for MMA result");
  wave::WaveSymbolAttr mSymbol = indexingSymbols.drop_back().back();
  wave::WaveSymbolAttr nSymbol = indexingSymbols.back();

  std::optional<wave::WaveMmaKind> mmaKind = getKind();
  if (!mmaKind)
    return emitError() << "scaled MMA without kind attribute not supported";
  if (llvm::failed(populateMmaIndexingExpr(
          *mmaKind, /*isAccumulator=*/true, /*isScaled=*/false,
          /*isFP4=*/false, initObject.wavesPerBlock,
          initObject.hardwareConstraint.getThreadsPerWave(), mSymbol, nSymbol,
          /*kSymbol=*/nullptr, entries))) {
    return emitError() << "scaled MMA kind not supported by index deduction";
  }

  // Set the priority based on the order of operations: earlier scaled MMAs have
  // higher priority.
  auto orderedAllMmas = llvm::make_filter_range(
      initObject.deterministicOpOrder, llvm::IsaPred<ScaledMmaOp, MmaOp>);
  int32_t priority = wave::IndexExprsLatticeStorage::kMmaPriority +
                     std::distance(llvm::find(orderedAllMmas, getOperation()),
                                   orderedAllMmas.end()) -
                     1;
  mixInThreadIndependentConstraints(
      *this, initObject.hardwareConstraint.getThreadsPerWave(), indexingSymbols,
      initObject.symbolConstraints, entries);
  SmallVector<std::pair<wave::WaveSymbolAttr, mlir::Attribute>>
      scaledResultEntries(entries.begin(), entries.end());
  return joinIndexExprsLatticeInPlace(
      resultExprs[0], "scaled MMA result",
      wave::IndexExprsLatticeStorage(
          wave::WaveSymbolMappingAttr::get(getContext(), scaledResultEntries),
          priority,
          getMmaVectorShape(
              getLoc(), *mmaKind, mSymbol, nSymbol,
              /*kSymbol=*/nullptr,
              /*kScaledSymbol=*/nullptr, indexingSymbols.drop_back(2),
              initObject.hardwareConstraint.getVectorShapes(), nullptr)),
      "implied by scaled MMA kind", emitError);
}

LogicalResult wave::ScaledMmaOp::initializeIndexExprsBackward(
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> operandExprs,
    const IndexExprsAnalysisInit &initObject, wave::EmitErrorFn emitError,
    wave::EmitDelayedErrorFn &delayedErrorEmitter) {
  auto resultType = llvm::cast<wave::WaveTensorType>(getResult().getType());
  auto lhsType = llvm::cast<wave::WaveTensorType>(getLhs().getType());
  assert(resultType.getRank() == lhsType.getRank() && lhsType.getRank() >= 2 &&
         "at least 2D scaled MMA operations are supported");
  wave::WaveSymbolAttr mSymbol = resultType.getShape().drop_back().back();
  wave::WaveSymbolAttr nSymbol = resultType.getShape().back();
  wave::WaveSymbolAttr kSymbol = lhsType.getShape().back();

  auto lhsScaleType = llvm::cast<wave::WaveTensorType>(getLhsScale().getType());
  wave::WaveSymbolAttr kScaleSymbol = lhsScaleType.getShape().back();

  std::optional<wave::WaveMmaKind> mmaKind = getKind();
  if (!mmaKind)
    return emitError() << "scaled MMA without kind attribute not supported";

  SmallVector<std::pair<wave::WaveSymbolAttr, wave::WaveIndexMappingAttr>>
      operandEntries;
  operandEntries.reserve(lhsType.getShape().size() + 1);
  if (llvm::failed(populateMmaIndexingExpr(
          *mmaKind, /*isAccumulator=*/false, /*isScaled=*/false,
          /*isFP4=*/false, initObject.wavesPerBlock,
          initObject.hardwareConstraint.getThreadsPerWave(), mSymbol, nSymbol,
          kSymbol, operandEntries))) {
    return emitError() << "scaled MMA kind not supported by index deduction";
  }

  SmallVector<std::pair<wave::WaveSymbolAttr, wave::WaveIndexMappingAttr>>
      scaleEntries;
  scaleEntries.reserve(lhsScaleType.getShape().size() + 1);
  if (llvm::failed(populateMmaIndexingExpr(
          *mmaKind, /*isAccumulator=*/false, /*isScaled=*/true,
          /*isFP4=*/false, initObject.wavesPerBlock,
          initObject.hardwareConstraint.getThreadsPerWave(), mSymbol, nSymbol,
          kScaleSymbol, scaleEntries))) {
    return emitError()
           << "scaled MMA kind not supported by scale index deduction";
  }

  SmallVector<std::pair<wave::WaveSymbolAttr, wave::WaveIndexMappingAttr>>
      accumulatorEntries;
  accumulatorEntries.reserve(resultType.getShape().size());
  if (llvm::failed(populateMmaIndexingExpr(
          *mmaKind, /*isAccumulator=*/true, /*isScaled=*/false,
          /*isFP4=*/false, initObject.wavesPerBlock,
          initObject.hardwareConstraint.getThreadsPerWave(), mSymbol, nSymbol,
          nullptr, accumulatorEntries))) {
    return emitError() << "scaled MMA kind not supported by index deduction";
  }

  ArrayRef<wave::WaveSymbolAttr> batchSymbols =
      resultType.getShape().drop_back(2);

  mixInThreadIndependentConstraints(
      *this, initObject.hardwareConstraint.getThreadsPerWave(),
      llvm::concat<const WaveSymbolAttr>(batchSymbols,
                                         ArrayRef{mSymbol, nSymbol, kSymbol}),
      initObject.symbolConstraints, operandEntries);
  mixInThreadIndependentConstraints(
      *this, initObject.hardwareConstraint.getThreadsPerWave(),
      llvm::concat<const WaveSymbolAttr>(
          batchSymbols, ArrayRef{mSymbol, nSymbol, kScaleSymbol}),
      initObject.symbolConstraints, scaleEntries);
  mixInThreadIndependentConstraints(
      *this, initObject.hardwareConstraint.getThreadsPerWave(),
      llvm::concat<const WaveSymbolAttr>(batchSymbols,
                                         ArrayRef{mSymbol, nSymbol}),
      initObject.symbolConstraints, accumulatorEntries);

  SmallVector<std::pair<wave::WaveSymbolAttr, wave::WaveIndexMappingAttr>>
      lhsEntries;
  filterSymbol(operandEntries, nSymbol, lhsEntries);
  SmallVector<std::pair<wave::WaveSymbolAttr, wave::WaveIndexMappingAttr>>
      rhsEntries;
  filterSymbol(operandEntries, mSymbol, rhsEntries);
  SmallVector<std::pair<wave::WaveSymbolAttr, wave::WaveIndexMappingAttr>>
      lhsScaleEntries;
  filterSymbol(scaleEntries, nSymbol, lhsScaleEntries);
  SmallVector<std::pair<wave::WaveSymbolAttr, wave::WaveIndexMappingAttr>>
      rhsScaleEntries;
  filterSymbol(scaleEntries, mSymbol, rhsScaleEntries);

  // Set the priority based on the order of operations: earlier scaled MMAs have
  // higher priority.
  auto orderedAllMmas = llvm::make_filter_range(
      initObject.deterministicOpOrder, llvm::IsaPred<MmaOp, ScaledMmaOp>);
  int32_t priority = wave::IndexExprsLatticeStorage::kMmaPriority +
                     std::distance(llvm::find(orderedAllMmas, getOperation()),
                                   orderedAllMmas.end()) -
                     1;
  SmallVector<std::pair<wave::WaveSymbolAttr, mlir::Attribute>>
      lhsMappingEntries(lhsEntries.begin(), lhsEntries.end());
  if (failed(joinIndexExprsLatticeInPlace(
          operandExprs[getLhsMutable().getOperandNumber()], "LHS",
          wave::IndexExprsLatticeStorage(
              wave::WaveSymbolMappingAttr::get(getContext(), lhsMappingEntries),
              priority,
              getMmaVectorShape(getLoc(), *mmaKind, mSymbol,
                                /*nSymbol=*/nullptr, kSymbol,
                                /*kScaledSymbol=*/nullptr, batchSymbols,
                                initObject.hardwareConstraint.getVectorShapes(),
                                &delayedErrorEmitter)),
          "implied by scaled MMA kind", emitError)))
    return failure();
  SmallVector<std::pair<wave::WaveSymbolAttr, mlir::Attribute>>
      lhsScaleMappingEntries(lhsScaleEntries.begin(), lhsScaleEntries.end());
  if (failed(joinIndexExprsLatticeInPlace(
          operandExprs[getLhsScaleMutable().getOperandNumber()], "LHS scale",
          wave::IndexExprsLatticeStorage(
              wave::WaveSymbolMappingAttr::get(getContext(),
                                               lhsScaleMappingEntries),
              priority,
              getMmaVectorShape(getLoc(), *mmaKind, mSymbol,
                                /*nSymbol=*/nullptr, /*kSymbol=*/nullptr,
                                kScaleSymbol, batchSymbols,
                                initObject.hardwareConstraint.getVectorShapes(),
                                &delayedErrorEmitter)),
          "implied by scaled MMA kind", emitError)))
    return failure();
  SmallVector<std::pair<wave::WaveSymbolAttr, mlir::Attribute>>
      rhsMappingEntries(rhsEntries.begin(), rhsEntries.end());
  if (failed(joinIndexExprsLatticeInPlace(
          operandExprs[getRhsMutable().getOperandNumber()], "RHS",
          wave::IndexExprsLatticeStorage(
              wave::WaveSymbolMappingAttr::get(getContext(), rhsMappingEntries),
              priority,
              getMmaVectorShape(getLoc(), *mmaKind,
                                /*mSymbol=*/nullptr, nSymbol, kSymbol,
                                /*kScaledSymbol=*/nullptr, batchSymbols,
                                initObject.hardwareConstraint.getVectorShapes(),
                                &delayedErrorEmitter)),
          "implied by scaled MMA kind", emitError)))
    return failure();
  SmallVector<std::pair<wave::WaveSymbolAttr, mlir::Attribute>>
      rhsScaleMappingEntries(rhsScaleEntries.begin(), rhsScaleEntries.end());
  if (failed(joinIndexExprsLatticeInPlace(
          operandExprs[getRhsScaleMutable().getOperandNumber()], "RHS scale",
          wave::IndexExprsLatticeStorage(
              wave::WaveSymbolMappingAttr::get(getContext(),
                                               rhsScaleMappingEntries),
              priority,
              getMmaVectorShape(getLoc(), *mmaKind,
                                /*mSymbol=*/nullptr, nSymbol,
                                /*kSymbol=*/nullptr, kScaleSymbol, batchSymbols,
                                initObject.hardwareConstraint.getVectorShapes(),
                                &delayedErrorEmitter)),
          "implied by scaled MMA kind", emitError)))
    return failure();
  SmallVector<std::pair<wave::WaveSymbolAttr, mlir::Attribute>>
      accMappingEntries(accumulatorEntries.begin(), accumulatorEntries.end());
  if (failed(joinIndexExprsLatticeInPlace(
          operandExprs[getAccumulatorMutable().getOperandNumber()],
          "accumulator",
          wave::IndexExprsLatticeStorage(
              wave::WaveSymbolMappingAttr::get(getContext(), accMappingEntries),
              priority,
              getMmaVectorShape(getLoc(), *mmaKind, mSymbol, nSymbol,
                                /*kSymbol=*/nullptr,
                                /*kScaledSymbol=*/nullptr, batchSymbols,
                                initObject.hardwareConstraint.getVectorShapes(),
                                &delayedErrorEmitter)),
          "implied by scaled MMA kind", emitError)))
    return failure();
  return llvm::success();
}

std::function<void(raw_ostream &, unsigned)>
wave::ScaledMmaOp::getIndexExprValuesAndDescriptions(
    llvm::SmallVectorImpl<Value> &values) {
  values.reserve(6);
  llvm::append_range(values, getOperands());
  values.push_back(getResult());
  unsigned lhsPos = getLhsMutable().getOperandNumber();
  unsigned lhsScalePos = getLhsScaleMutable().getOperandNumber();
  unsigned rhsPos = getRhsMutable().getOperandNumber();
  unsigned rhsScalePos = getRhsScaleMutable().getOperandNumber();
  unsigned accPos = getAccumulatorMutable().getOperandNumber();
  return [lhsPos, lhsScalePos, rhsPos, rhsScalePos, accPos](raw_ostream &os,
                                                            unsigned i) {
    assert(i < 6 && "unexpected position");
    if (i == lhsPos)
      os << "lhs";
    else if (i == lhsScalePos)
      os << "lhs scale";
    else if (i == rhsPos)
      os << "rhs";
    else if (i == rhsScalePos)
      os << "rhs scale";
    else if (i == accPos)
      os << "accumulator";
    else
      os << "result";
  };
}

// Propagate index expressions forward from the operands to the result of the
// WriteOp. Since WriteOp has no results, this is a no-op.
llvm::FailureOr<ChangeResult> wave::WriteOp::propagateIndexExprsForward(
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> operandExprs,
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> resultExprs,
    wave::EmitErrorFn emitError) {
  // WriteOp has no results, so just return NoChange.
  return ChangeResult::NoChange;
}

/// Computes the vector stride for each dimension: stride[i] is the product of
/// vector shapes for dimensions i+1 .. rank-1 (so the last dimension has
/// stride 1). A dimension is contiguous iff its stride is 1.
static FailureOr<SmallVector<int64_t>>
getVectorStrides(wave::WaveTensorType tensorType,
                 HardwareConstraintAttr hardwareConstraint) {
  assert(tensorType.getFullySpecified() &&
         "expected fully-specified tensor type");
  wave::WaveSymbolMappingAttr vectorShapes =
      hardwareConstraint.getVectorShapes();
  if (!vectorShapes)
    return failure();
  int64_t rank = tensorType.getRank();
  SmallVector<int64_t> strides(rank);
  if (rank == 0)
    return strides;
  strides[rank - 1] = 1;
  for (int64_t i = rank - 2; i >= 0; --i) {
    Attribute vectorShape = vectorShapes.lookup(tensorType.getShape()[i + 1]);
    if (!vectorShape)
      return failure();
    int64_t shape = cast<IntegerAttr>(vectorShape).getValue().getSExtValue();
    strides[i] = shape * strides[i + 1];
  }
  return strides;
}

LogicalResult WriteOp::initializeIndexExprsBackward(
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> operandExprs,
    const wave::IndexExprsAnalysisInit &initObject, wave::EmitErrorFn emitError,
    wave::EmitDelayedErrorFn &delayedErrorEmitter) {

  // TODO: figure out how to propagate elements per threads from constraints to
  // operations while avoiding the clash with index sequences. When propagating,
  // we don't have sequences yet.
  WaveTensorType tensorType = cast<WaveTensorType>(getValueToStore().getType());
  HardwareConstraintAttr hardwareConstraint = initObject.hardwareConstraint;

  assert(tensorType.getFullySpecified());
  FailureOr<SmallVector<int64_t>> stridesOr =
      getVectorStrides(tensorType, hardwareConstraint);
  // XXX: don't report this error immediately since we may be able to proceed
  // without it, e.g., when index expressions may be propagated from
  // operations with higher priority operations to this one. This is a
  // questionable design choice carried over from the initial Python
  // prototype, but is needed for initial consistency. Consider revising.
  if (failed(stridesOr)) {
    delayedErrorEmitter = [](InFlightDiagnostic &diag) {
      diag << "couldn't find vector shapes in the contiguity check";
    };
    return success();
  }
  llvm::ArrayRef<int64_t> strides = *stridesOr;
  // XXX: pywave confusingly calls this "contiguous" but it is actually the
  // dimension along which SIMD vectorization is applied, i.e., the deepest
  // dimension for which the per-thread vector shape is not 1 or, alternatively,
  // the product of vector shapes for trailing dimensions remains 1.
  int64_t vectorizedDimPos = -1;
  for (int64_t i = 0, e = tensorType.getRank(); i < e; ++i)
    if (strides[i] == 1) {
      vectorizedDimPos = i;
      break;
    }

  SmallVector<std::pair<WaveSymbolAttr, WaveIndexMappingAttr>> indexEntries;
  for (int64_t i = 0, e = tensorType.getRank(); i < e; ++i) {
    AffineExpr elementsPerThread = nullptr;
    bool isVectorized = (i == vectorizedDimPos);

    // The absence of constraints for a dimension means it is not mapped to
    // workgroups/wave/items, so there is nothing to do here.
    // We expect it to be handled by thread-independent constraints setting
    // the default (0, 1, 1) index expression or following the tiling
    // constraint.
    auto it = initObject.symbolConstraints.find(tensorType.getShape()[i]);
    if (it == initObject.symbolConstraints.end())
      continue;
    auto wgConstraintIt =
        llvm::find_if(it->second, llvm::IsaPred<WorkgroupConstraintAttr>);
    if (wgConstraintIt == it->second.end())
      continue;
    WorkgroupConstraintAttr wgConstraint =
        cast<WorkgroupConstraintAttr>(*wgConstraintIt);

    // The innermost dimension with vectorized size other than 1 is the one we
    // want to vectorize along.
    std::optional<int64_t> opElementsPerThread = getElementsPerThread();
    SmallVector<Attribute> symbols;
    if (isVectorized) {
      if (opElementsPerThread) {
        elementsPerThread =
            getAffineConstantExpr(*opElementsPerThread, getContext());
      } else {
        AffineMap tileSizeMap = wgConstraint.getTileSize().getMap();
        assert(tileSizeMap.getNumResults() == 1 &&
               "expected a single expression in tile size affine map");
        unsigned numThreads = [&]() {
          switch (wgConstraint.getWorkgroupDim().getValue()) {
          case WaveWorkgroupDim::X:
            return initObject.wavesPerBlock[0] *
                   initObject.hardwareConstraint.getThreadsPerWave();
          case WaveWorkgroupDim::Y:
            return initObject.wavesPerBlock[1];
          case WaveWorkgroupDim::Z:
            return initObject.wavesPerBlock[2];
          }
        }();

        elementsPerThread = tileSizeMap.getResult(0).ceilDiv(numThreads);
        llvm::append_range(symbols, wgConstraint.getTileSize().getSymbols());
      }
    } else {
      elementsPerThread = getAffineConstantExpr(1, getContext());
    }

    int64_t stride = strides[i];
    assert(stride > 0 && "stride should be positive");

    WaveIndexSymbol threadSymbol = [&]() {
      switch (wgConstraint.getWorkgroupDim().getValue()) {
      case WaveWorkgroupDim::X:
        return WaveIndexSymbol::THREAD_0;
      case WaveWorkgroupDim::Y:
        return WaveIndexSymbol::THREAD_1;
      case WaveWorkgroupDim::Z:
        return WaveIndexSymbol::THREAD_2;
      }
    }();
    WaveIndexSymbolAttr threadSymbolAttr =
        WaveIndexSymbolAttr::get(getContext(), threadSymbol);
    symbols.push_back(threadSymbolAttr);

    AffineExpr startExpr =
        getAffineSymbolExpr(symbols.size() - 1, getContext());
    if (wgConstraint.getWorkgroupDim().getValue() == WaveWorkgroupDim::X) {
      startExpr = startExpr % hardwareConstraint.getThreadsPerWave();
    } else {
      // TODO: in pywave, we always do `startExpr % threadsPerWave` where
      // threadsPerWave == 1 for workgroup dims other than X, making it
      // always zero. It mentions an assumption about the (64, 1, 1) thread
      // shape, but it is unclear whether that assumption always holds.
      // It looks like the intention for this was to express lane ID rather
      // than thread ID, but it is unclear how it accounts for multiple
      // wavefronts running in parallel.
      startExpr = getAffineConstantExpr(0, getContext());
    }

    auto indexMapping = WaveIndexMappingAttr::get(
        getContext(), symbols,
        AffineMap::get(/*dimCount=*/0, symbols.size(),
                       startExpr * elementsPerThread),
        AffineMap::get(/*dimCount=*/0, symbols.size(), elementsPerThread),
        AffineMap::get(/*dimCount=*/0, symbols.size(),
                       getAffineConstantExpr(stride, getContext())));
    indexEntries.emplace_back(tensorType.getShape()[i], indexMapping);
  }
  mixInThreadIndependentConstraints(
      *this, initObject.hardwareConstraint.getThreadsPerWave(),
      tensorType.getShape(), initObject.symbolConstraints, indexEntries);
  SmallVector<std::pair<wave::WaveSymbolAttr, mlir::Attribute>> writeEntries(
      indexEntries.begin(), indexEntries.end());
  auto writeMapping =
      wave::WaveSymbolMappingAttr::get(getContext(), writeEntries);
  if (failed(joinIndexExprsLatticeInPlace(
          operandExprs[getValueToStoreMutable().getOperandNumber()],
          "value to store",
          IndexExprsLatticeStorage(
              writeMapping, IndexExprsLatticeStorage::kWritePriority,
              detail::filterVectorShape(
                  hardwareConstraint.getVectorShapes(),
                  cast<WaveTensorType>(getValueToStore().getType())
                      .getShape())),
          "implied by write operation", emitError)))
    return failure();
  if (failed(joinIndexExprsLatticeInPlace(
          operandExprs[getMemoryMutable().getOperandNumber()], "memory",
          IndexExprsLatticeStorage(
              writeMapping, IndexExprsLatticeStorage::kWritePriority,
              detail::filterVectorShape(
                  hardwareConstraint.getVectorShapes(),
                  cast<WaveTensorType>(getMemory().getType()).getShape())),
          "implied by write operation", emitError)))
    return failure();

  return success();
}

// Propagating "sideways" between operands, but only if this would not result
// in conflicts.
llvm::FailureOr<ChangeResult> wave::WriteOp::propagateIndexExprsBackward(
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> operandExprs,
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> resultExprs,
    wave::EmitErrorFn emitError) {
  // XXX: If both are propagating from MMAs, temporarily assume equal priority
  // (ignore order of MMAs). If sideways propagation would result in a new
  // conflict in this case, don't propagate. This is a questionable design
  // carried over from the initial Python prototype.
  auto forceMmaPriority = [](const IndexExprsLatticeStorage &lattice) {
    if (lattice.isBottom() || lattice.isTop())
      return lattice;
    WaveSymbolMappingAttr priorityMap = lattice.getPriorities();
    bool needsCapping = priorityMap && llvm::any_of(priorityMap, [](auto pair) {
                          return llvm::cast<IntegerAttr>(pair.second).getInt() >
                                 IndexExprsLatticeStorage::kMmaPriority;
                        });
    if (!needsCapping)
      return lattice;
    MLIRContext *ctx = priorityMap.getContext();
    IntegerType i32 = IntegerType::get(ctx, 32);
    SmallVector<std::pair<wave::WaveSymbolAttr, Attribute>> capped;
    capped.reserve(priorityMap.size());
    for (auto [key, val] : priorityMap) {
      int32_t priority = llvm::cast<IntegerAttr>(val).getInt();
      capped.emplace_back(
          key,
          IntegerAttr::get(
              i32, std::min(priority, IndexExprsLatticeStorage::kMmaPriority)));
    }
    IndexExprsLatticeStorage result(lattice.getConcreteValue(),
                                    WaveSymbolMappingAttr::get(ctx, capped),
                                    lattice.getVectorShape());
    return result.withSourceVectorShape(
        lattice.getSourceVectorShape(),
        std::min(lattice.getSourceVectorShapePriority(),
                 IndexExprsLatticeStorage::kMmaPriority));
  };
  IndexExprsLatticeStorage lhs = forceMmaPriority(operandExprs[0]);
  IndexExprsLatticeStorage rhs = forceMmaPriority(operandExprs[1]);
  auto joined = IndexExprsLatticeStorage::join(lhs, rhs);
  if (joined.isTop() && !(lhs.isTop() || rhs.isTop())) {
    return ChangeResult::NoChange;
  }

  // Re-join with the original expressions to get the right priority.
  joined = IndexExprsLatticeStorage::join(operandExprs[0], operandExprs[1]);

  unsigned valueToStoreOperandNumber =
      getValueToStoreMutable().getOperandNumber();
  unsigned memoryOperandNumber = getMemoryMutable().getOperandNumber();
  ChangeResult changeResult = ChangeResult::NoChange;
  if (operandExprs[valueToStoreOperandNumber] != joined &&
      wave::detail::shouldPropagateIndexExprs(
          operandExprs[valueToStoreOperandNumber], joined, getValueToStore())) {
    operandExprs[valueToStoreOperandNumber] = joined;
    changeResult = ChangeResult::Change;
  }
  if (operandExprs[memoryOperandNumber] != joined &&
      wave::detail::shouldPropagateIndexExprs(operandExprs[memoryOperandNumber],
                                              joined, getMemory())) {
    operandExprs[memoryOperandNumber] = joined;
    changeResult = ChangeResult::Change;
  }
  return changeResult;
}

// Special case for WriteOp where we want an index expression even
// though it doesn't have results.
// TODO: this shouldn't be necessary in a purely MLIR form since
// mappings are a property of the SSA value (conversely, changing the
// mapping should create a new value), but keeping for compatibility.
std::function<void(raw_ostream &, unsigned)>
wave::WriteOp::getIndexExprValuesAndDescriptions(
    llvm::SmallVectorImpl<Value> &values) {
  values.push_back(getValueToStore());
  return [](raw_ostream &os, unsigned i) {
    assert(i == 0 && "unexpected position");
    os << "value to store";
  };
}

FailureOr<ChangeResult> wave::BroadcastOp::propagateIndexExprsForward(
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> operandIndexExprs,
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> resultIndexExprs,
    wave::EmitErrorFn emitError) {
  // Forward propagation is identity: it will propagate expressions for symbols
  // present in the source to the result and make sure they are joined with
  // those. Additional propagation backward from the result users will be needed
  // to cover all symbols.
  return detail::identityIndexExprsPropagate(operandIndexExprs,
                                             resultIndexExprs, getResult(),
                                             "operand", "result", emitError);
}

FailureOr<ChangeResult> wave::BroadcastOp::propagateIndexExprsBackward(
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> operandIndexExprs,
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> resultIndexExprs,
    wave::EmitErrorFn emitError) {
  auto sourceTensorType = dyn_cast<WaveTensorType>(getSource().getType());
  if (!sourceTensorType) {
    emitError() << "expected source tensor type, got " << getSource().getType();
    return failure();
  }

  // Backward propagation is identity only for symbols that are present
  return detail::identityIndexExprsPropagate(
      resultIndexExprs[0].keepOnlySymbols(sourceTensorType.getShape()),
      operandIndexExprs, getSource(), "result", "operand", emitError);
}

// Helper to permute strides in an index expressions lattice according to
// the permutation from source shape to target shape.
//
// The permute operation swaps the strides of the permuted indices.
// For example, if we have a permute operation that swaps [B, M, N] to
// [M, N, B], then for each dimension k, we keep its start and step,
// but take the stride from the dimension at the same position in
// target_shape.
static IndexExprsLatticeStorage
permuteIndexExprsStrides(const IndexExprsLatticeStorage &inputLattice,
                         llvm::ArrayRef<wave::WaveSymbolAttr> srcShape,
                         llvm::ArrayRef<wave::WaveSymbolAttr> targetShape,
                         MLIRContext *ctx, wave::EmitErrorFn emitError) {
  if (inputLattice.isBottom() || inputLattice.isTop())
    return inputLattice;

  assert(srcShape.size() == targetShape.size() &&
         "source shape rank does not match target shape rank");

  WaveSymbolMappingAttr inputMapping = inputLattice.getConcreteValue();

  llvm::DenseMap<WaveSymbolAttr, WaveIndexMappingAttr> symbolToMapping;
  for (auto [key, val] : inputMapping) {
    if (auto mapping = llvm::dyn_cast<WaveIndexMappingAttr>(val))
      symbolToMapping[key] = mapping;
  }

  // Create the permuted index expressions.
  // For each dimension k in src_shape:
  //   - Keep start and step from the original mapping for k
  //   - Take stride from the mapping for src_to_target[k]
  SmallVector<std::pair<WaveSymbolAttr, Attribute>> permutedMappings;
  permutedMappings.reserve(srcShape.size());
  for (auto [srcSymbol, targetSymbol] :
       llvm::zip_equal(srcShape, targetShape)) {
    auto srcMappingIt = symbolToMapping.find(srcSymbol);
    auto targetMappingIt = symbolToMapping.find(targetSymbol);

    assert(srcMappingIt != symbolToMapping.end() &&
           "source mapping not found for symbol");
    assert(targetMappingIt != symbolToMapping.end() &&
           "target mapping not found for symbol");

    WaveIndexMappingAttr srcMapping = srcMappingIt->second;
    WaveIndexMappingAttr targetMapping = targetMappingIt->second;

    SmallVector<Attribute> allSymbols(srcMapping.getSymbols());
    for (Attribute sym : targetMapping.getSymbols()) {
      if (!llvm::is_contained(allSymbols, sym))
        allSymbols.push_back(sym);
    }

    AffineMap alignedStart = alignMapSymbols(
        srcMapping.getStart(), srcMapping.getSymbols(), allSymbols);
    AffineMap alignedStep = alignMapSymbols(
        srcMapping.getStep(), srcMapping.getSymbols(), allSymbols);
    AffineMap alignedStride = alignMapSymbols(
        targetMapping.getStride(), targetMapping.getSymbols(), allSymbols);

    auto newMapping = WaveIndexMappingAttr::get(ctx, allSymbols, alignedStart,
                                                alignedStep, alignedStride);

    permutedMappings.emplace_back(srcSymbol, newMapping);
  }

  return IndexExprsLatticeStorage(
             WaveSymbolMappingAttr::get(ctx, permutedMappings),
             inputLattice.getPriorities(), inputLattice.getVectorShape())
      .withSourceVectorShape(inputLattice.getSourceVectorShape(),
                             inputLattice.getSourceVectorShapePriority());
}

llvm::FailureOr<ChangeResult> wave::PermuteOp::propagateIndexExprsForward(
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> operandExprs,
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> resultExprs,
    wave::EmitErrorFn emitError) {
  auto inputType = llvm::dyn_cast<WaveTensorType>(getValue().getType());
  if (!inputType || !inputType.getFullySpecified())
    return ChangeResult::NoChange;

  auto resultType = llvm::dyn_cast<WaveTensorType>(getResult().getType());
  if (!resultType || !resultType.getFullySpecified())
    return ChangeResult::NoChange;

  ArrayRef<WaveSymbolAttr> targetShape = resultType.getShape();
  ArrayRef<WaveSymbolAttr> srcShape = inputType.getShape();

  IndexExprsLatticeStorage permuted = permuteIndexExprsStrides(
      operandExprs[0], srcShape, targetShape, getContext(), emitError);

  IndexExprsLatticeStorage newResultLattice =
      IndexExprsLatticeStorage::join(resultExprs[0], permuted);

  if (newResultLattice.isTop() && !resultExprs[0].isTop() &&
      !permuted.isTop()) {
    InFlightDiagnostic diag =
        emitError()
        << "conflict when propagating forward to the result lattice in "
           "PermuteOp";
    diag.attachNote() << "Result lattice: " << resultExprs[0];
    diag.attachNote() << "Operand lattice: " << operandExprs[0];
    return diag;
  }

  return updateIfChanged(resultExprs[0], newResultLattice);
}

llvm::FailureOr<ChangeResult> wave::PermuteOp::propagateIndexExprsBackward(
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> operandExprs,
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> resultExprs,
    wave::EmitErrorFn emitError) {
  auto inputType = llvm::dyn_cast<WaveTensorType>(getValue().getType());
  if (!inputType || !inputType.getFullySpecified())
    return ChangeResult::NoChange;

  auto resultType = llvm::dyn_cast<WaveTensorType>(getResult().getType());
  if (!resultType || !resultType.getFullySpecified())
    return ChangeResult::NoChange;

  if (!wave::detail::shouldPropagateIndexExprs(resultExprs[0], operandExprs[0],
                                               getValue())) {
    return ChangeResult::NoChange;
  }

  ArrayRef<WaveSymbolAttr> resultShape = resultType.getShape();
  ArrayRef<WaveSymbolAttr> srcShape = inputType.getShape();

  IndexExprsLatticeStorage permuted = permuteIndexExprsStrides(
      resultExprs[0], resultShape, srcShape, getContext(), emitError);

  IndexExprsLatticeStorage newOperandLattice =
      IndexExprsLatticeStorage::join(operandExprs[0], permuted);

  if (newOperandLattice.isTop() && !operandExprs[0].isTop() &&
      !permuted.isTop()) {
    InFlightDiagnostic diag =
        emitError()
        << "conflict when propagating backward to the operand lattice in "
           "PermuteOp";
    diag.attachNote() << "Operand lattice: " << operandExprs[0];
    diag.attachNote() << "Result lattice: " << resultExprs[0];
    return diag;
  }

  return updateIfChanged(operandExprs[0], newOperandLattice);
}

//-----------------------------------------------------------------------------
// BitcastOp index expression propagation
//-----------------------------------------------------------------------------

static bool isExprListDim(DictionaryAttr mapping, wave::WaveSymbolAttr sym) {
  Attribute attr = mapping.get(sym.getName());
  return llvm::isa_and_nonnull<wave::WaveExprListAttr>(attr);
}

static SmallVector<unsigned> getScaledDimensions(wave::BitcastOp op) {
  wave::WaveHyperparameterAttr hyper =
      wave::getHyperparameters(op.getOperation());
  if (!hyper)
    return {};
  auto srcType =
      llvm::dyn_cast<wave::WaveTensorType>(op.getValueToCast().getType());
  auto dstType = llvm::dyn_cast<wave::WaveTensorType>(op.getResult().getType());
  if (!srcType || !dstType)
    return {};
  ArrayRef<wave::WaveSymbolAttr> srcShape = srcType.getShape();
  ArrayRef<wave::WaveSymbolAttr> dstShape = dstType.getShape();
  assert(srcShape.size() == dstShape.size() &&
         "bitcast shapes must have equal rank");
  DictionaryAttr mapping = hyper.getMapping();
  SmallVector<unsigned> scaledDims;
  for (unsigned i = 0, e = srcShape.size(); i < e; ++i) {
    if (isExprListDim(mapping, srcShape[i]) ||
        isExprListDim(mapping, dstShape[i]))
      scaledDims.push_back(i);
  }
  return scaledDims;
}

// Remap the index expression lattice for bitcast: non-scaled dimensions pass
// through as identity, and the scaled dimension gets its symbol renamed and its
// step (and start/stride if present) scaled by the element bitwidth ratio.
// \p scaledDim is the index of the dimension that carries the scaling.
static IndexExprsLatticeStorage
scaleBitcastIndexExprs(const IndexExprsLatticeStorage &inputLattice,
                       ArrayRef<wave::WaveSymbolAttr> fromShape,
                       ArrayRef<wave::WaveSymbolAttr> toShape,
                       unsigned fromBits, unsigned toBits, unsigned scaledDim,
                       MLIRContext *ctx) {
  if (inputLattice.isBottom() || inputLattice.isTop())
    return inputLattice;

  assert(fromShape.size() == toShape.size() &&
         "bitcast shapes must have equal rank");

  WaveSymbolMappingAttr inputMapping = inputLattice.getConcreteValue();

  unsigned ratio =
      (fromBits > toBits) ? (fromBits / toBits) : (toBits / fromBits);
  bool scaleUp = fromBits > toBits;

  SmallVector<std::pair<WaveSymbolAttr, Attribute>> newMappings;
  newMappings.reserve(inputMapping.size());

  WaveSymbolAttr fromScaledSym = fromShape[scaledDim];
  WaveSymbolAttr toScaledSym = toShape[scaledDim];

  for (auto [key, val] : inputMapping) {
    auto mapping = llvm::cast<WaveIndexMappingAttr>(val);

    if (key != fromScaledSym) {
      newMappings.emplace_back(key, val);
      continue;
    }

    // Scaled dimension: rename the symbol key and scale the step by the
    // bitwidth ratio.  If the step has no thread dimensions (thread-independent
    // initialization), skip this lattice to avoid producing a zero-step mapping
    // that would conflict with a real propagated value.
    AffineMap stepMap = mapping.getStep();
    if (!stepMap || (stepMap.getNumDims() == 0 && stepMap.getNumSymbols() == 0))
      return IndexExprsLatticeStorage::bottom();

    auto scaleMap = [&](AffineMap map) -> AffineMap {
      AffineExpr scaled =
          scaleUp ? map.getResult(0) * ratio : map.getResult(0).floorDiv(ratio);
      return AffineMap::get(map.getNumDims(), map.getNumSymbols(), scaled, ctx);
    };

    AffineMap newStep = scaleMap(stepMap);
    auto newMapping =
        WaveIndexMappingAttr::get(ctx, mapping.getSymbols(), mapping.getStart(),
                                  newStep, mapping.getStride());

    newMappings.emplace_back(toScaledSym, newMapping);
  }

  if (newMappings.empty())
    return IndexExprsLatticeStorage::bottom();

  return IndexExprsLatticeStorage(WaveSymbolMappingAttr::get(ctx, newMappings),
                                  inputLattice.getPriorities(),
                                  inputLattice.getVectorShape());
}

// Shared propagation logic for BitcastOp index expressions in both directions.
// \p fromExprs is the source lattice, \p toExprs is the destination to update.
// \p fromType / \p toType are the wave tensor types on the respective sides.
// \p fromBits / \p toBits are the element bitwidths on the respective sides.
static llvm::FailureOr<ChangeResult> propagateBitcastIndexExprs(
    wave::BitcastOp op, llvm::ArrayRef<IndexExprsLatticeStorage> fromExprs,
    llvm::MutableArrayRef<IndexExprsLatticeStorage> toExprs,
    WaveTensorType fromType, WaveTensorType toType, unsigned fromBits,
    unsigned toBits, StringRef fromName, StringRef toName, mlir::Value toValue,
    wave::EmitErrorFn emitError) {
  if (!fromType || !fromType.getFullySpecified() || !toType ||
      !toType.getFullySpecified())
    return ChangeResult::NoChange;

  if (fromBits == toBits)
    return wave::detail::identityIndexExprsPropagate(
        fromExprs, toExprs, toValue, fromName, toName, emitError);

  SmallVector<unsigned> scaledDims = getScaledDimensions(op);
  std::optional<unsigned> scaledDim =
      scaledDims.empty() ? std::nullopt : std::optional(scaledDims[0]);

  if (!scaledDim) {
    InFlightDiagnostic diag = emitError()
                              << "could not determine scaled dimension for "
                                 "BitcastOp with differing bitwidths";
    return diag;
  }

  IndexExprsLatticeStorage scaled = scaleBitcastIndexExprs(
      fromExprs[0], fromType.getShape(), toType.getShape(), fromBits, toBits,
      *scaledDim, op.getContext());

  IndexExprsLatticeStorage newLattice =
      IndexExprsLatticeStorage::join(toExprs[0], scaled);

  if (newLattice.isTop() && !toExprs[0].isTop() && !scaled.isTop()) {
    InFlightDiagnostic diag = emitError()
                              << "conflict when propagating " << fromName
                              << " to " << toName << " lattice in BitcastOp";
    diag.attachNote() << toName << " lattice: " << toExprs[0];
    diag.attachNote() << fromName << " lattice: " << fromExprs[0];
    return diag;
  }

  return updateIfChanged(toExprs[0], newLattice);
}

llvm::FailureOr<ChangeResult> wave::BitcastOp::propagateIndexExprsForward(
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> operandExprs,
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> resultExprs,
    wave::EmitErrorFn emitError) {
  WaveTensorType inputType =
      llvm::dyn_cast<WaveTensorType>(getValueToCast().getType());
  WaveTensorType resultType =
      llvm::dyn_cast<WaveTensorType>(getResult().getType());
  unsigned srcBits =
      wave::getElementType(getValueToCast().getType()).getIntOrFloatBitWidth();
  unsigned dstBits =
      wave::getElementType(getResult().getType()).getIntOrFloatBitWidth();
  return propagateBitcastIndexExprs(*this, operandExprs, resultExprs, inputType,
                                    resultType, srcBits, dstBits, "operand",
                                    "result", getResult(), emitError);
}

llvm::FailureOr<ChangeResult> wave::BitcastOp::propagateIndexExprsBackward(
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> operandExprs,
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> resultExprs,
    wave::EmitErrorFn emitError) {
  WaveTensorType inputType =
      llvm::dyn_cast<WaveTensorType>(getValueToCast().getType());
  WaveTensorType resultType =
      llvm::dyn_cast<WaveTensorType>(getResult().getType());
  unsigned srcBits =
      wave::getElementType(getValueToCast().getType()).getIntOrFloatBitWidth();
  unsigned dstBits =
      wave::getElementType(getResult().getType()).getIntOrFloatBitWidth();
  return propagateBitcastIndexExprs(
      *this, resultExprs, operandExprs, resultType, inputType, dstBits, srcBits,
      "result", "operand", getValueToCast(), emitError);
}

//-----------------------------------------------------------------------------
// Index expression inference analysis and pass (from InferTypes.cpp)
//-----------------------------------------------------------------------------

namespace {
// PrintNoRegions(op)`.
class PrintNoRegions {
public:
  PrintNoRegions(Operation *op) : operation(op) {}

  void print(llvm::raw_ostream &os) const {
    if (!operation) {
      os << "<null>";
      return;
    }
    operation->print(os, OpPrintingFlags().skipRegions());
  }

private:
  Operation *operation;
};
} // namespace

// Support operator<< for PrintNoRegions.
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const PrintNoRegions &printer) {
  printer.print(os);
  return os;
}

inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os, const ElementsPerThreadLatticeValue &value) {
  value.print(os);
  return os;
}

// Lattice object for index expressions for analysis compatibility, the actual
// logic is in the IndexExprsLatticeStorage class.
class IndexExprsLattice : public dataflow::Lattice<IndexExprsLatticeStorage> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IndexExprsLattice);
  using Lattice::Lattice;
};

class IndexExprsForwardAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<IndexExprsLattice> {
private:
  void unsafeSet(IndexExprsLattice *lattice, IndexExprsLatticeStorage value) {
    if (lattice->getValue() == value)
      return;
    lattice->getValue().unsafeSet(value);
    propagateIfChanged(lattice, ChangeResult::Change);
  }

  void safeSet(IndexExprsLattice *lattice, IndexExprsLatticeStorage value) {
    if (lattice->getValue() == value)
      return;
#ifndef NDEBUG
    IndexExprsLatticeStorage joined =
        IndexExprsLatticeStorage::join(lattice->getValue(), value);
    assert(IndexExprsLatticeStorage::join(joined, lattice->getValue()) ==
               joined &&
           "join should not move the lattice backward, did you forget to join "
           "with the original lattice value in an interface method "
           "implementation?");
    assert(
        IndexExprsLatticeStorage::join(joined, value) == joined &&
        "join should not move the lattice forward, did you forget to join with "
        "the original lattice value in an interface method implementation?");
#endif
    unsafeSet(lattice, value);
  }

public:
  explicit IndexExprsForwardAnalysis(
      DataFlowSolver &solver,
      wave::OverrideInitializationFn overrideInitialization = nullptr)
      : SparseForwardDataFlowAnalysis(solver),
        overrideInitialization(overrideInitialization) {}

  LogicalResult initialize(Operation *top) override {
    assert(!getSolverConfig().isInterprocedural() &&
           "interprocedural analysis not supported");

    // Call the base class initialization in order to set up update listeners.
    // Note that this will initialize values at function/region entries to
    // lattice top.
    if (failed(SparseForwardDataFlowAnalysis::initialize(top)))
      return failure();

    llvm::DenseMap<Operation *, Attribute> constraints;
    if (llvm::failed(wave::collectWaveConstraints(top, constraints)))
      return llvm::failure();

    for (auto &&[parent, attr] : constraints) {
      wave::WaveHyperparameterAttr hyperparams =
          wave::getHyperparameters(parent);
      auto initObject =
          wave::IndexExprsAnalysisInit::create(parent, attr, hyperparams);
      if (llvm::failed(initObject))
        return llvm::failure();
      WalkResult walkResult = parent->walk([&](Operation *op) -> WalkResult {
        if (auto iface =
                llvm::dyn_cast<wave::WaveInferIndexExprsOpInterface>(op)) {
          llvm::SmallVector<wave::IndexExprsLatticeStorage> resultExprs =
              llvm::map_to_vector(op->getResults(), [&](Value v) {
                return getLatticeElement(v)->getValue();
              });
          auto emitError = [op]() { return op->emitError(); };
          LDBG() << "initializing index expressions forward for "
                 << PrintNoRegions(op);
          if (llvm::failed(iface.initializeIndexExprsForward(
                  resultExprs, *initObject, emitError)))
            return WalkResult::interrupt();

          for (auto &&[result, lattice] :
               llvm::zip_equal(op->getResults(), resultExprs)) {
            IndexExprsLattice *latticeObject = getLatticeElement(result);
            LDBG() << "  result #" << result.getResultNumber()
                   << " original: " << *latticeObject;
            safeSet(latticeObject, lattice);
            LDBG() << "  result #" << result.getResultNumber()
                   << " updated: " << *latticeObject;
          }
        }

        if (auto iterateOp = llvm::dyn_cast<wave::IterateOp>(op)) {
          // Set lattices of captured block arguments to the relevant tiling
          // constraint, it will be then propagated by joining with
          // expressions induced by other constraints.
          wave::WaveSymbolAttr iterSymbolAttr = iterateOp.getIterator();
          llvm::SmallVector<Attribute> symbolConstraints =
              initObject->symbolConstraints.lookup(iterSymbolAttr);
          auto it = llvm::find_if(symbolConstraints,
                                  llvm::IsaPred<wave::TilingConstraintAttr>);
          if (it != symbolConstraints.end()) {
            wave::TilingConstraintAttr tilingConstraint =
                llvm::cast<wave::TilingConstraintAttr>(*it);
            for (Value capture : iterateOp.getCaptureBlockArgs()) {
              auto captureType =
                  dyn_cast<wave::WaveTensorType>(capture.getType());
              if (!captureType)
                continue;
              if (!llvm::is_contained(captureType.getShape(), iterSymbolAttr))
                continue;
              auto mapping = wave::WaveSymbolMappingAttr::get(
                  iterSymbolAttr.getContext(),
                  {{iterSymbolAttr, wave::applyConstraint(tilingConstraint)}});
              LDBG() << "setting iterate block argument lattice " << capture
                     << " from " << PrintNoRegions(iterateOp) << " to "
                     << mapping;
              IndexExprsLattice *captureLattice = getLatticeElement(capture);
              safeSet(
                  captureLattice,
                  wave::IndexExprsLatticeStorage::join(
                      captureLattice->getValue(),
                      wave::IndexExprsLatticeStorage(
                          mapping,
                          wave::IndexExprsLatticeStorage::kLowestPriority,
                          wave::detail::filterVectorShape(
                              initObject->hardwareConstraint.getVectorShapes(),
                              captureType.getShape()))));
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
              top, [&](Value value, wave::WaveSymbolMappingAttr mapping,
                       wave::WaveSymbolMappingAttr priorities,
                       wave::WaveSymbolMappingAttr vecShape) {
                if (!mapping)
                  return unsafeSet(getLatticeElement(value),
                                   IndexExprsLatticeStorage::top());
                unsafeSet(getLatticeElement(value),
                          wave::IndexExprsLatticeStorage(mapping, priorities,
                                                         vecShape));
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
    auto arg = llvm::dyn_cast<BlockArgument>(lattice->getAnchor());
    if (arg && llvm::isa<CallableOpInterface>(arg.getOwner()->getParentOp())) {
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
  visitOperation(Operation *op,
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
    llvm::scope_exit scope([&] {
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
    llvm::FailureOr<ChangeResult> result =
        llvm::cast<wave::WaveInferIndexExprsOpInterface>(op)
            .propagateIndexExprsForward(operandLattices, resultLattices,
                                        reportError);
    if (llvm::failed(result))
      return llvm::failure();
    if (*result == ChangeResult::NoChange)
      return llvm::success();

    for (auto &&[resultLattice, lattice] :
         llvm::zip_equal(resultLattices, results)) {
      safeSet(lattice, resultLattice);
    }
    return llvm::success();
  }

  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor,
      ValueRange nonSuccessorInputs,
      llvm::ArrayRef<IndexExprsLattice *> lattices) override {
    auto iterateOp = llvm::dyn_cast<wave::IterateOp>(op);
    if (!iterateOp)
      return;

    // Technically, the non-captured arguments can be seen as forwarded from
    // operands or results, but they need special handling to remove
    // loop-specific parts of the index.
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
        ChangeResult changed = lattice->join(iterArgLattice->getValue());
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
        ChangeResult changed = resultLattice->join(iterArgLattice->getValue());
        changed |= resultLattice->join(
            terminatorOperandLattice->getValue().withoutIterSymbols(
                iterateOp.getIterator()));
        propagateIfChanged(resultLattice, changed);
        LDBG() << "new result lattice: " << *resultLattice;
      }
    }
  }

  // Return true if there are pending error reports.
  bool hasDelayedErrors() const { return !delayedErrors.empty(); }

  // Return the emitter of a pending error report for the given operation.
  wave::EmitDelayedErrorFn getDelayedError(Operation *op) const {
    return delayedErrors.lookup_or(op, wave::EmitDelayedErrorFn());
  }

private:
  bool initialized = false;
  wave::OverrideInitializationFn overrideInitialization;
  llvm::SmallDenseMap<Operation *, wave::EmitDelayedErrorFn> delayedErrors;
};

class IndexExprsBackwardAnalysis
    : public dataflow::SparseBackwardDataFlowAnalysis<IndexExprsLattice> {
private:
  void unsafeSet(IndexExprsLattice *lattice, IndexExprsLatticeStorage value) {
    if (lattice->getValue() == value)
      return;
    lattice->getValue().unsafeSet(value);
    propagateIfChanged(lattice, ChangeResult::Change);
  }

  void safeSet(IndexExprsLattice *lattice, IndexExprsLatticeStorage value) {
    if (lattice->getValue() == value)
      return;
#ifndef NDEBUG
    IndexExprsLatticeStorage joined =
        IndexExprsLatticeStorage::join(lattice->getValue(), value);
    assert(IndexExprsLatticeStorage::join(joined, lattice->getValue()) ==
               joined &&
           "join should not move the lattice backward, did you forget to join "
           "with the original lattice value in an interface method "
           "implementation?");
    assert(
        IndexExprsLatticeStorage::join(joined, value) == joined &&
        "join should not move the lattice forward, did you forget to join with "
        "the original lattice value in an interface method implementation?");
#endif
    unsafeSet(lattice, value);
  }

public:
  IndexExprsBackwardAnalysis(
      DataFlowSolver &solver, SymbolTableCollection &symbolTable,
      wave::OverrideInitializationFn overrideInitialization = nullptr)
      : SparseBackwardDataFlowAnalysis(solver, symbolTable),
        overrideInitialization(overrideInitialization) {}

  llvm::LogicalResult initialize(Operation *top) override {
    assert(!getSolverConfig().isInterprocedural() &&
           "interprocedural analysis not supported");

    // Call the base class initialization in order to set up update listeners.
    // Note that this will initialize values at function/region entries to
    // lattice top.
    if (llvm::failed(SparseBackwardDataFlowAnalysis::initialize(top)))
      return llvm::failure();

    llvm::DenseMap<Operation *, Attribute> constraints;
    if (llvm::failed(wave::collectWaveConstraints(top, constraints)))
      return llvm::failure();
    for (auto &&[parent, attr] : constraints) {
      wave::WaveHyperparameterAttr hyperparams =
          wave::getHyperparameters(parent);
      auto initObject =
          wave::IndexExprsAnalysisInit::create(parent, attr, hyperparams);
      if (llvm::failed(initObject))
        return llvm::failure();

      WalkResult walkResult = parent->walk([&](Operation *op) -> WalkResult {
        if (op->hasTrait<wave::RequiresSidewaysBackwardPropagationOpTrait>()) {
          for (Value operand : op->getOperands())
            addDependency(getLatticeElement(operand), getProgramPointAfter(op));
        }
        if (auto iface =
                llvm::dyn_cast<wave::WaveInferIndexExprsOpInterface>(op)) {
          llvm::SmallVector<wave::IndexExprsLatticeStorage> operandExprs =
              llvm::map_to_vector(op->getOperands(), [&](Value v) {
                return getLatticeElement(v)->getValue();
              });
          auto emitError = [op]() { return op->emitError(); };

          LDBG() << "initializing index expressions backward for "
                 << PrintNoRegions(op);
          wave::EmitDelayedErrorFn delayedErrorEmitter = nullptr;
          if (llvm::failed(iface.initializeIndexExprsBackward(
                  operandExprs, *initObject, emitError, delayedErrorEmitter)))
            return WalkResult::interrupt();
          if (delayedErrorEmitter) {
            LDBG() << "delayed error recorded\n";
            delayedErrors[op] = delayedErrorEmitter;
          }
          for (auto &&[i, operand, lattice] :
               llvm::enumerate(op->getOperands(), operandExprs)) {
            IndexExprsLattice *latticeObject = getLatticeElement(operand);
            LDBG() << "  operand #" << i << " original: " << *latticeObject;
            safeSet(latticeObject, lattice);
            LDBG() << "  operand #" << i << " updated: " << *latticeObject;
          }
          return WalkResult::advance();
        }

        return WalkResult::advance();
      });
      if (walkResult.wasInterrupted())
        return failure();
    }

    if (overrideInitialization) {
      if (llvm::failed(overrideInitialization(
              top, [&](Value value, wave::WaveSymbolMappingAttr mapping,
                       wave::WaveSymbolMappingAttr priorities,
                       wave::WaveSymbolMappingAttr vecShape) {
                if (!mapping)
                  return unsafeSet(getLatticeElement(value),
                                   IndexExprsLatticeStorage::top());
                unsafeSet(getLatticeElement(value),
                          wave::IndexExprsLatticeStorage(mapping, priorities,
                                                         vecShape));
              })))
        return llvm::failure();
    }

    initialized = true;
    return llvm::success();
  }

  void visitBranchOperand(OpOperand &opOperand) override {
    if (!llvm::isa<wave::WaveTensorType>(opOperand.get().getType()))
      return;

    // Captures of the iterate need to be propagated from the corresponding
    // block arguments manually without the tiling constraint.
    if (auto iterateOp =
            llvm::dyn_cast<wave::IterateOp>(opOperand.getOwner())) {
      unsigned position = opOperand.getOperandNumber();
      Value blockArgument = iterateOp.getLoopBody()->getArgument(position);
      IndexExprsLattice *blockArgLattice = getLatticeElement(blockArgument);
      IndexExprsLattice *lattice = getLatticeElement(opOperand.get());
      // Explicitly add a dependency to update this analysis at the program
      // point before the iterate if the block argument lattice changes. Using
      // localized const cast here to avoid accidentally modifying this lattice
      // in the function.
      // TODO: expose it upstream and use.
      addDependency(blockArgLattice, getProgramPointAfter(iterateOp));

      if (!wave::detail::shouldPropagateIndexExprs(blockArgLattice->getValue(),
                                                   lattice->getValue(),
                                                   opOperand.get())) {
        return;
      }

      LDBG() << "propagating backwards from block argument #" << position
             << " to op operand " << PrintNoRegions(iterateOp);
      LDBG() << "block argument lattice: " << *blockArgLattice;
      LDBG() << "lattice: " << *lattice;
      IndexExprsLatticeStorage joined = IndexExprsLatticeStorage::join(
          lattice->getValue(), blockArgLattice->getValue().withoutIterSymbols(
                                   iterateOp.getIterator()));
      safeSet(lattice, joined);
      LDBG() << "new lattice: " << *lattice;
      return;
    }

    // Terminator operands are propagated from op results as is.
    if (auto yieldOp = llvm::dyn_cast<wave::YieldOp>(opOperand.getOwner())) {
      unsigned position = opOperand.getOperandNumber();
      Value result = yieldOp->getParentOp()->getResult(position);
      const IndexExprsLattice *resultLattice = getLatticeElement(result);
      IndexExprsLattice *lattice = getLatticeElement(opOperand.get());
      // Explicitly add a dependency to update this analysis at the program
      // point before the terminator if the result lattice changes. Using
      // localized const cast here to avoid accidentally modifying this lattice.
      // TODO: expose it upstream and use.
      addDependency(const_cast<IndexExprsLattice *>(resultLattice),
                    getProgramPointAfter(yieldOp));

      if (!wave::detail::shouldPropagateIndexExprs(resultLattice->getValue(),
                                                   lattice->getValue(),
                                                   opOperand.get())) {
        return;
      }

      LDBG() << "propagating backwards from region-carrying op result #"
             << position << " to terminator operand " << yieldOp;
      LDBG() << "result lattice: " << *resultLattice;
      LDBG() << "lattice: " << *lattice;
      IndexExprsLatticeStorage joined = IndexExprsLatticeStorage::join(
          lattice->getValue(), resultLattice->getValue());
      safeSet(lattice, joined);
      LDBG() << "new lattice: " << *lattice;
      return;
    }

    setToExitState(getLatticeElement(opOperand.get()));
  }

  void visitCallOperand(OpOperand &opOperand) override {
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
  visitOperation(Operation *op, llvm::ArrayRef<IndexExprsLattice *> operands,
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
    llvm::scope_exit scope([&] {
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
    if (*result == ChangeResult::NoChange)
      return llvm::success();

    for (auto &&[operandLattice, lattice] :
         llvm::zip_equal(operandLattices, operands)) {
      safeSet(lattice, operandLattice);
    }
    return llvm::success();
  }

  // Visit the non-forwarded arguments of a region, such as the
  // induction variables of a loop.
  void
  visitNonControlFlowArguments(RegionSuccessor & /*successor*/,
                               ArrayRef<BlockArgument> /*arguments*/) override {
    // This is called for induction variables of an IterateOp, which is handled
    // by the forward analysis.
  }

  // Returns true if there are any delayed errors.
  bool hasDelayedErrors() const { return !delayedErrors.empty(); }

  // Returns the delayed error emitter for the given operation.
  wave::EmitDelayedErrorFn getDelayedError(Operation *op) const {
    return delayedErrors.lookup_or(op, wave::EmitDelayedErrorFn());
  }

private:
  bool initialized = false;
  wave::OverrideInitializationFn overrideInitialization;
  llvm::SmallDenseMap<Operation *, wave::EmitDelayedErrorFn, 4> delayedErrors;
};

wave::DelayedErrorEmitterInfo
wave::addWaveIndexExprsAnalyses(DataFlowSolver &solver,
                                SymbolTableCollection &symbolTable,
                                wave::WaveIndexExprsAnalysisOptions options) {
  IndexExprsForwardAnalysis *forward = nullptr;
  if (!options.disableForward) {
    forward =
        solver.load<IndexExprsForwardAnalysis>(options.overrideInitialization);
  }
  IndexExprsBackwardAnalysis *backward = nullptr;
  if (!options.disableBackward) {
    backward = solver.load<IndexExprsBackwardAnalysis>(
        symbolTable, options.overrideInitialization);
  }

  // Note that these lambdas are stored and used later so they must not capture
  // anything that has a function-level lifetime.
  wave::DelayedErrorEmitterInfo delayedErrorEmitterInfo;
  delayedErrorEmitterInfo.getDelayedError =
      [forward, backward](Operation *op) -> wave::EmitDelayedErrorFn {
    if (forward) {
      if (wave::EmitDelayedErrorFn delayedError = forward->getDelayedError(op))
        return delayedError;
    }
    if (backward) {
      return backward->getDelayedError(op);
    }
    return nullptr;
  };
  delayedErrorEmitterInfo.hasDelayedErrors = [forward, backward]() {
    return (forward && forward->hasDelayedErrors()) ||
           (backward && backward->hasDelayedErrors());
  };
  return delayedErrorEmitterInfo;
}

/// If any slot has a vector shape, every slot must have one; otherwise emit an
/// error. Empty shape dictionaries are not used as placeholders on the IR.
static LogicalResult collectPerValueVectorShapeAttrs(
    Operation *op, llvm::ArrayRef<wave::IndexExprsLatticeStorage> slots,
    llvm::function_ref<void(llvm::raw_ostream &, unsigned)> describeSlot,
    llvm::SmallVectorImpl<Attribute> &shapeDicts) {
  bool anyPresent =
      llvm::any_of(slots, [](const wave::IndexExprsLatticeStorage &l) {
        return l.getVectorShape() != nullptr;
      });
  if (!anyPresent)
    return success();
  shapeDicts.reserve(shapeDicts.size() + slots.size());
  for (const auto [i, lat] : llvm::enumerate(slots)) {
    wave::WaveSymbolMappingAttr vs = lat.getVectorShape();
    if (!vs) {
      llvm::SmallString<64> buf;
      llvm::raw_svector_ostream os(buf);
      describeSlot(os, i);
      return op->emitError() << "missing vector shape for " << buf.str();
    }
    shapeDicts.push_back(vs);
  }
  return success();
}

LogicalResult wave::setWaveIndexExprAnalysisResults(
    Operation *top, const DataFlowSolver &solver,
    const DelayedErrorEmitterInfo &delayedErrorInfo,
    llvm::function_ref<LogicalResult(Operation *,
                                     ArrayRef<IndexExprsLatticeStorage>)>
        extraHandler) {
  bool hadFailures = false;
  WalkResult walkResult =
      top->walk([&](wave::WaveInferIndexExprsOpInterface iface) {
        auto getLatticeValue = [&](Value value) {
          auto *latticeObject = solver.lookupState<IndexExprsLattice>(value);
          return latticeObject ? latticeObject->getValue()
                               : IndexExprsLatticeStorage::bottom();
        };

        SmallVector<Value> valuesForIndexExpr;
        SmallVector<Attribute> indexExprs;

        // Special case for MMA operations. We always set their index attribute
        // to whatever is implied by the MMA kind regardless of the values
        // inferred for operands because MMAs must retain their specific index
        // expressions.
        // TODO: this shouldn't strictly necessary in a purely MLIR flow and is
        // kept for Python compatibility.
        std::function<void(raw_ostream &, unsigned)> descriptionGenerator =
            iface.getIndexExprValuesAndDescriptions(valuesForIndexExpr);
        indexExprs.reserve(valuesForIndexExpr.size());

        // Shared handler for MMA-family ops: initializes operand lattices from
        // MMA-kind-implied index expressions and sets the result index
        // attribute. `numOperands` is the number of MMA operands (3 for MmaOp,
        // 5 for ScaledMmaOp). `resultDescription` is used in error messages.
        auto handleMmaOp = [&](wave::WaveInferIndexExprsOpInterface mmaIface,
                               unsigned numOperands,
                               StringRef resultDescription) -> WalkResult {
          Operation *mmaOp = mmaIface.getOperation();
          SmallVector<wave::IndexExprsLatticeStorage> operandLattices(
              numOperands, wave::IndexExprsLatticeStorage::bottom());
          wave::EmitDelayedErrorFn delayedError;
          Attribute constraints;
          // TODO(#1049): this is not ideal, especially when combined with
          // getting hyperparameters below, we could just have a double walk
          // with a kernel operation first if we had one, or even do a
          // per-kernel pass.
          for (Operation *parent = mmaOp->getParentOp(); parent && !constraints;
               parent = parent->getParentOp()) {
            constraints = parent->getAttrOfType<ArrayAttr>(
                wave::WaveDialect::kWaveConstraintsAttrName);
          }
          if (!constraints) {
            mmaOp->emitError("constraints not found");
            return WalkResult::interrupt();
          }
          FailureOr<wave::IndexExprsAnalysisInit> init =
              wave::IndexExprsAnalysisInit::create(
                  mmaOp, constraints, wave::getHyperparameters(mmaOp));
          if (failed(init))
            return WalkResult::interrupt();
          if (failed(mmaIface.initializeIndexExprsBackward(
                  operandLattices, *init, [&]() { return mmaOp->emitError(); },
                  delayedError))) {
            if (delayedError) {
              InFlightDiagnostic diag = mmaOp->emitError();
              delayedError(diag);
            }
            return WalkResult::interrupt();
          }

          for (auto &&[i, lattice] : llvm::enumerate(operandLattices)) {
            SmallString<32> description;
            llvm::raw_svector_ostream os(description);
            descriptionGenerator(os, i);
            [[maybe_unused]] LogicalResult logicalResult =
                detail::checkAndAppendIndexExpr(mmaOp->getLoc(), lattice,
                                                description, indexExprs);
            assert(succeeded(logicalResult) &&
                   "failed to append implied index expression, it must not be "
                   "bottom/top");
          }
          Value result = mmaOp->getResult(0);
          if (failed(detail::checkAndAppendIndexExpr(
                  mmaOp->getLoc(), getLatticeValue(result), resultDescription,
                  indexExprs)))
            return WalkResult::interrupt();

          MLIRContext *ctx = mmaIface.getContext();
          SmallVector<wave::IndexExprsLatticeStorage> slots(operandLattices);
          slots.push_back(getLatticeValue(result));
          SmallVector<Attribute> shapeDicts;
          if (failed(collectPerValueVectorShapeAttrs(
                  mmaOp, slots, descriptionGenerator, shapeDicts)))
            return WalkResult::interrupt();
          mmaIface->setAttr(wave::WaveDialect::kIndexWaveExprListAttrName,
                            ArrayAttr::get(ctx, indexExprs));
          if (!shapeDicts.empty())
            mmaIface->setAttr(wave::WaveDialect::kVectorShapeAttrName,
                              ArrayAttr::get(ctx, shapeDicts));
          if (extraHandler) {
            if (failed(extraHandler(mmaOp, slots)))
              return WalkResult::interrupt();
          }
          return WalkResult::advance();
        };

        if (auto mma = dyn_cast<wave::MmaOp>(iface.getOperation()))
          return handleMmaOp(mma, /*numOperands=*/3, "mma result");

        if (auto scaledMma = dyn_cast<wave::ScaledMmaOp>(iface.getOperation()))
          return handleMmaOp(scaledMma, /*numOperands=*/5, "scaled_mma result");

        for (auto &&[i, value] : llvm::enumerate(valuesForIndexExpr)) {
          llvm::SmallString<32> description;
          llvm::raw_svector_ostream os(description);
          descriptionGenerator(os, i);
          if (failed(detail::checkAndAppendIndexExpr(iface->getLoc(),
                                                     getLatticeValue(value),
                                                     os.str(), indexExprs))) {
            // Don't stop on the first reported error if there are some delayed
            // errors that would be useful to report here. We need to wait and
            // see whether the operation they are attached to actually has had
            // inference issues as some errors may be corrected.
            if (!delayedErrorInfo.hasDelayedErrors())
              return WalkResult::interrupt();

            hadFailures = true;
            if (auto delayedError = delayedErrorInfo.getDelayedError(iface)) {
              InFlightDiagnostic diag =
                  iface->emitError()
                  << "the error above may be caused by the following: ";
              delayedError(diag);
            }
          }
        }

        SmallVector<wave::IndexExprsLatticeStorage> slots =
            llvm::map_to_vector(valuesForIndexExpr, getLatticeValue);
        // Only set the index expressions if there were no failures.
        if (!hadFailures) {
          MLIRContext *ctx = iface->getContext();
          SmallVector<Attribute> shapeDicts;
          if (failed(collectPerValueVectorShapeAttrs(
                  iface, slots, descriptionGenerator, shapeDicts)))
            return WalkResult::interrupt();
          iface->setAttr(wave::WaveDialect::kIndexWaveExprListAttrName,
                         ArrayAttr::get(ctx, indexExprs));
          if (!shapeDicts.empty())
            iface->setAttr(wave::WaveDialect::kVectorShapeAttrName,
                           ArrayAttr::get(ctx, shapeDicts));
        }

        if (extraHandler && failed(extraHandler(iface.getOperation(), slots)))
          return WalkResult::interrupt();

        return WalkResult::advance();
      });
  return llvm::failure(hadFailures || walkResult.wasInterrupted());
}
