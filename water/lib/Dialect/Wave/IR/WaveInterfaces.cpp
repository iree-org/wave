// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "mlir/IR/AffineExpr.h"
#include "water/Dialect/Wave/IR/IndexExpr.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

//-----------------------------------------------------------------------------
// Index attribute verification
//-----------------------------------------------------------------------------

LogicalResult wave::verifyWaveIndexMappings(Operation *op) {
  // The attribute is optional.
  Attribute attribute =
      op->getAttr(wave::WaveDialect::kIndexWaveExprListAttrName);
  if (!attribute)
    return success();

  auto arr = dyn_cast<ArrayAttr>(attribute);
  if (!arr)
    return op->emitError("'index' attribute must be an array of dictionaries");
  SmallVector<DictionaryAttr> dicts;
  dicts.reserve(arr.size());
  for (Attribute nestedAttr : arr) {
    auto dict = dyn_cast<DictionaryAttr>(nestedAttr);
    if (!dict)
      return op->emitError("'index' array elements must be dictionaries");
    dicts.push_back(dict);
  }

  for (DictionaryAttr dictAttr : dicts) {
    for (auto named : dictAttr) {
      auto val = named.getValue();
      if (!isa<wave::WaveIndexMappingAttr>(val))
        return op->emitError("'index' attribute value for key ")
               << named.getName() << " must be WaveIndexMappingAttr, got "
               << val;

      auto mapping = cast<wave::WaveIndexMappingAttr>(val);
      auto checkNoDims = [&](AffineMap map, StringRef which) -> LogicalResult {
        if (map.getNumDims() != 0)
          return op->emitError(
                     "wave indexing " + which +
                     " map should have no dimensions, only symbols, got ")
                 << map.getNumDims() << " dimensions for symbol "
                 << named.getName();
        return success();
      };

      AffineMap startMap = mapping.getStart();
      AffineMap stepMap = mapping.getStep();
      AffineMap strideMap = mapping.getStride();
      if (failed(checkNoDims(startMap, "start")) ||
          failed(checkNoDims(stepMap, "step")) ||
          failed(checkNoDims(strideMap, "stride")))
        return failure();

      unsigned declared = mapping.getSymbols().size();
      if (startMap.getNumSymbols() != declared ||
          stepMap.getNumSymbols() != declared ||
          strideMap.getNumSymbols() != declared) {
        return op->emitError(
                   "inconsistent symbol count between symbol_names and "
                   "affine maps for index symbol '")
               << named.getName() << "' (expected " << declared
               << ", got start=" << startMap.getNumSymbols()
               << ", step=" << stepMap.getNumSymbols()
               << ", stride=" << strideMap.getNumSymbols() << ")";
      }

      for (auto symbol : mapping.getSymbols()) {
        auto iterSymbol = dyn_cast<wave::WaveIterSymbolAttr>(symbol);
        if (!iterSymbol)
          continue;

        bool found = false;
        for (Operation *currentOp = op->getParentOp(); currentOp != nullptr;
             currentOp = currentOp->getParentOp()) {
          // TODO: we don't want to depend on the IterateOp specifically from
          // the interface (though technically we can), so we use the opaque
          // attribute name. We should add something like a "wave control flow
          // interface" that would provide it without hardcoded strings.
          auto parentIterSymbol =
              currentOp->getAttrOfType<wave::WaveSymbolAttr>("iterator");
          if (!parentIterSymbol)
            continue;
          if (parentIterSymbol.getName() == iterSymbol.getName()) {
            found = true;
            break;
          }
        }
        if (!found) {
          return op->emitError("index expression uses iterator symbol ")
                 << iterSymbol.getName()
                 << " which is not defined by any parent op";
        }
      }
    }
  }
  return success();
}

//-----------------------------------------------------------------------------
// Custom printing/parsing components
//-----------------------------------------------------------------------------

// ODS custom directive: parseWaveIndexDict/printWaveIndexDict
ParseResult wave::parseWaveIndexDict(OpAsmParser &parser, ArrayAttr &out) {
  auto parseSingleDict = [&](DictionaryAttr &out) -> ParseResult {
    SmallVector<NamedAttribute, 4> entries;
    if (parser.parseLBrace())
      return failure();
    auto parseEntry = [&]() -> ParseResult {
      StringRef symbolName;
      if (parser.parseKeyword(&symbolName) || parser.parseColon())
        return failure();
      Attribute mapping = wave::WaveIndexMappingAttr::parse(parser, Type{});
      if (!mapping)
        return failure();
      entries.emplace_back(parser.getBuilder().getStringAttr(symbolName),
                           mapping);
      return success();
    };
    if (parser.parseCommaSeparatedList(parseEntry) || parser.parseRBrace())
      return failure();
    out = parser.getBuilder().getDictionaryAttr(entries);
    return success();
  };

  SmallVector<Attribute> dicts;
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Square,
                                     [&]() -> ParseResult {
                                       DictionaryAttr dict;
                                       if (failed(parseSingleDict(dict)))
                                         return failure();
                                       dicts.push_back(dict);
                                       return success();
                                     }))
    return failure();
  out = parser.getBuilder().getArrayAttr(dicts);
  return success();
}

void wave::printWaveIndexDict(OpAsmPrinter &printer, Operation *op,
                              ArrayAttr arr) {
  auto printOne = [&](DictionaryAttr dict) {
    printer.getStream() << "{";
    llvm::interleaveComma(
        dict, printer.getStream(), [&](NamedAttribute namedAttr) {
          printer.getStream() << namedAttr.getName().getValue() << " : ";
          if (auto mappingAttr = llvm::dyn_cast<wave::WaveIndexMappingAttr>(
                  namedAttr.getValue())) {
            mappingAttr.print(printer);
          } else {
            printer.printAttribute(namedAttr.getValue());
          }
        });
    printer.getStream() << "}";
  };
  // Always print as an array to match the parser syntax.
  printer.getStream() << "[";
  llvm::interleaveComma(arr, printer.getStream(), [&](Attribute a) {
    printOne(llvm::cast<DictionaryAttr>(a));
  });
  printer.getStream() << "]";
}

//-----------------------------------------------------------------------------
// WaveInferTypeOpInterface helpers
//-----------------------------------------------------------------------------

// Return `true` if two tensor types have the same shape. Null types are
// considered to have different shapes.
static bool hasSameShape(wave::WaveTensorType lhs, wave::WaveTensorType rhs) {
  // TODO: this may require more advanced checking if shapes are more complex
  // than a single symbol.
  return lhs && rhs && lhs.getShape() == rhs.getShape();
}

llvm::FailureOr<ChangeResult> wave::detail::checkPropagateShapeConflict(
    wave::WaveTensorType from, wave::WaveTensorType to,
    llvm::StringRef fromName, llvm::StringRef toName, llvm::raw_ostream &errs) {
  if (!from || !to || hasSameShape(from, to))
    return ChangeResult::NoChange;

  if (!to.getFullySpecified())
    return ChangeResult::Change;

  errs << "irreconcilable types during type inference from " << fromName << "("
       << from << ") to " << toName << "(" << to << ")";
  return failure();
}

llvm::FailureOr<ChangeResult> wave::detail::propagateShapeInformation(
    wave::WaveTensorType from, wave::WaveTensorType &to,
    llvm::StringRef fromName, llvm::StringRef toName, llvm::raw_ostream &errs) {
  llvm::FailureOr<ChangeResult> res =
      checkPropagateShapeConflict(from, to, fromName, toName, errs);
  if (failed(res) || *res == ChangeResult::NoChange)
    return res;

  to = to.copyShapeFrom(from);
  return ChangeResult::Change;
}

llvm::FailureOr<ChangeResult> wave::detail::identityTypeInferencePropagate(
    llvm::ArrayRef<wave::WaveTensorType> from,
    llvm::MutableArrayRef<wave::WaveTensorType> to, llvm::StringRef fromName,
    llvm::StringRef toName, llvm::raw_ostream &errs) {
  auto it = llvm::find_if(from, [](wave::WaveTensorType type) {
    return type && type.getFullySpecified();
  });
  if (it == from.end())
    return ChangeResult::NoChange;

  // Expect all fully-specified "from" types to have the same shape.
  for (auto [i, fr] : llvm::enumerate(from)) {
    llvm::FailureOr<ChangeResult> res =
        checkPropagateShapeConflict(*it, fr, fromName, toName, errs);
    if (failed(res)) {
      errs << " for " << fromName << " #" << i;
      return res;
    }
  }

  ChangeResult changeResult = ChangeResult::NoChange;
  for (auto &&[i, toType] : llvm::enumerate(to)) {
    llvm::FailureOr<ChangeResult> res =
        propagateShapeInformation(*it, toType, fromName, toName, errs);
    if (failed(res)) {
      errs << " for " << fromName << " #" << i;
      return failure();
    }

    changeResult |= *res;
  }
  return changeResult;
}

//-----------------------------------------------------------------------------
// WaveElementsPerThreadOpInterface helpers
//-----------------------------------------------------------------------------

// Print the error message indicating a mismatch between the two lattices.
static void printElementsPerThreadMismatchMsg(
    llvm::raw_ostream &errs, const wave::ElementsPerThreadLatticeValue &from,
    const wave::ElementsPerThreadLatticeValue &to, llvm::StringRef fromName,
    llvm::StringRef toName, size_t toIndex) {
  errs << "mismatch between " << fromName << " (";
  from.print(errs);
  errs << ") and " << toName << " #" << toIndex << " (";
  to.print(errs);
  errs << ")";
}

llvm::FailureOr<ChangeResult>
wave::detail::checkAndPropagateElementsPerThreadFromConstant(
    const ElementsPerThreadLatticeValue &from,
    llvm::ArrayRef<ElementsPerThreadLatticeValue> immutableValues,
    llvm::MutableArrayRef<ElementsPerThreadLatticeValue> mutableValues,
    llvm::StringRef fromName, llvm::StringRef immutableName,
    llvm::StringRef mutableName, llvm::raw_ostream &errs) {
  for (auto [i, fr] : llvm::enumerate(immutableValues)) {
    if (fr.isBottom() || ElementsPerThreadLatticeValue::join(from, fr) == fr)
      continue;

    printElementsPerThreadMismatchMsg(errs, from, fr, fromName, immutableName,
                                      i);
    return failure();
  }

  ChangeResult changeResult = ChangeResult::NoChange;
  for (auto &&[i, toType] : llvm::enumerate(mutableValues)) {
    ElementsPerThreadLatticeValue joined =
        ElementsPerThreadLatticeValue::join(from, toType);

    if (joined.isTop() && !from.isTop() && !toType.isTop()) {
      printElementsPerThreadMismatchMsg(errs, from, toType, fromName,
                                        mutableName, i);
      toType = ElementsPerThreadLatticeValue::top();
      return failure();
    }

    if (joined != toType) {
      changeResult = ChangeResult::Change;
      toType = joined;
    }
  }
  return changeResult;
}

llvm::FailureOr<ChangeResult> wave::detail::identityElementsPerThreadPropagate(
    llvm::ArrayRef<ElementsPerThreadLatticeValue> from,
    llvm::MutableArrayRef<ElementsPerThreadLatticeValue> to,
    llvm::StringRef fromName, llvm::StringRef toName, llvm::raw_ostream &errs) {
  assert(!from.empty());
  assert(!to.empty());
  return checkAndPropagateElementsPerThreadFromConstant(
      from[0], from, to, (fromName + " #0").str(), fromName, toName, errs);
}

wave::ElementsPerThreadLatticeValue wave::ElementsPerThreadLatticeValue::join(
    const wave::ElementsPerThreadLatticeValue &lhs,
    const wave::ElementsPerThreadLatticeValue &rhs) {
  if (lhs.isBottom())
    return rhs;
  if (rhs.isBottom())
    return lhs;
  if (lhs.isTop())
    return lhs;
  if (rhs.isTop())
    return rhs;

  // At this point, this is a specific lattice value.
  if (lhs.value == rhs.value)
    return lhs;
  return top();
}

void wave::ElementsPerThreadLatticeValue::print(llvm::raw_ostream &os) const {
  if (isBottom())
    os << "<bottom>";
  else if (isTop())
    os << "<top>";
  else
    os << value;
}

//-----------------------------------------------------------------------------
// Verification helpers
//-----------------------------------------------------------------------------

// Update negative indices in the array to positive equivalents given the total
// rank, e.g. -1 and -3 get updated to 3 and 1, respectively, for the rank of 4.
static void updateNegativeIndices(llvm::MutableArrayRef<int> indices,
                                  int rank) {
  for (int &index : indices) {
    if (index < 0)
      index += rank;
  }
}

llvm::LogicalResult wave::detail::verifyTypesMatchingDimensions(
    std::optional<Location> loc, llvm::StringRef lhsName,
    wave::WaveTensorType lhs, llvm::ArrayRef<int> lhsDims,
    llvm::StringRef rhsName, wave::WaveTensorType rhs,
    llvm::ArrayRef<int> rhsDims) {
  assert(lhsDims.size() == rhsDims.size() &&
         "expected lhs and rhs dim lists to be co-indexed");

  // Under-specified types are okay everywhere.
  if (!lhs.getFullySpecified() || !rhs.getFullySpecified())
    return success();

  llvm::SmallVector<int> lhsDimsVec(lhsDims), rhsDimsVec(rhsDims);
  updateNegativeIndices(lhsDimsVec, lhs.getRank());
  updateNegativeIndices(rhsDimsVec, rhs.getRank());
  for (auto &&[lhsDim, rhsDim] : llvm::zip_equal(lhsDimsVec, rhsDimsVec)) {
    wave::WaveSymbolAttr lhsExpr = lhs.getShape()[lhsDim];
    wave::WaveSymbolAttr rhsExpr = rhs.getShape()[rhsDim];
    if (lhsExpr == rhsExpr)
      continue;

    if (loc) {
      emitError(*loc) << "expected " << lhsName << " dimension #" << lhsDim
                      << " (" << lhsExpr << ") to match " << rhsName
                      << " dimension #" << rhsDim << " (" << rhsExpr << ")";
    }
    return failure();
  }
  return success();
}

llvm::LogicalResult wave::detail::verifyElementTypesMatch(
    std::optional<Location> loc, llvm::StringRef lhsName,
    wave::WaveTensorType lhs, llvm::StringRef rhsName,
    wave::WaveTensorType rhs) {
  if (lhs.getElementType() == rhs.getElementType())
    return success();

  if (loc) {
    emitError(*loc) << "expected " << lhsName << " and " << rhsName
                    << " elemental types to match, got " << lhs.getElementType()
                    << ", " << rhs.getElementType();
  }
  return failure();
}

llvm::LogicalResult wave::detail::verifyTypesCompatible(
    wave::WaveTensorType lhs, wave::WaveTensorType rhs,
    bool includeAddressSpace, std::optional<Location> errorLocation,
    llvm::StringRef lhsName, llvm::StringRef rhsName) {
  // Fast and cheap path.
  if (lhs == rhs)
    return success();

  if (errorLocation) {
    assert(!lhsName.empty() && !rhsName.empty() &&
           "expected names when location is provided");
  }

  if (includeAddressSpace) {
    if (lhs.getAddressSpaceValue() != rhs.getAddressSpaceValue() &&
        lhs.getAddressSpaceValue() != wave::WaveAddressSpace::Unspecified &&
        rhs.getAddressSpaceValue() != wave::WaveAddressSpace::Unspecified) {
      if (errorLocation) {
        emitError(*errorLocation) << "address space mismatch between "
                                  << lhsName << " and " << rhsName;
      }
      return failure();
    }
  }

  if (failed(
          verifyElementTypesMatch(errorLocation, lhsName, lhs, rhsName, rhs)))
    return failure();

  if (!lhs.getFullySpecified() || !rhs.getFullySpecified())
    return success();

  if (lhs.getRank() != rhs.getRank()) {
    if (errorLocation) {
      emitError(*errorLocation)
          << "rank mismatch between " << lhsName << " and " << rhsName;
    }
    return failure();
  }

  auto allDims = llvm::to_vector(llvm::iota_range<int>(0, lhs.getRank(),
                                                       /*Inclusive=*/false));
  return verifyTypesMatchingDimensions(errorLocation, lhsName, lhs, allDims,
                                       rhsName, rhs, allDims);
}

static llvm::LogicalResult
verifyTypeRange(Location loc, TypeRange range,
                wave::WaveTensorType referenceType, bool includeAddressSpace,
                llvm::StringRef rangeDescriptionPrefix,
                llvm::StringRef referenceDescription) {
  llvm::SmallString<16> rangeDescription(rangeDescriptionPrefix);
  for (auto &&[i, type] : llvm::enumerate(range)) {
    auto tensorType = llvm::dyn_cast<wave::WaveTensorType>(type);
    if (!tensorType)
      continue;

    rangeDescription.resize(rangeDescriptionPrefix.size());
    llvm::raw_svector_ostream os(rangeDescription);
    os << i;

    if (failed(wave::detail::verifyTypesCompatible(
            tensorType, referenceType, includeAddressSpace, loc, os.str(),
            referenceDescription))) {
      return llvm::failure();
    }
  }
  return llvm::success();
}

llvm::LogicalResult wave::detail::verifyCompatibleOperandsAndResultsOpTrait(
    Operation *op, bool includeAddressSpace) {
  const llvm::StringLiteral kOperandNamePrefix = "operand #";
  const llvm::StringLiteral kResultNamePrefix = "result #";
  std::string referenceDescription;
  llvm::raw_string_ostream os(referenceDescription);
  wave::WaveTensorType referenceType;
  auto it =
      llvm::find_if(op->getOperandTypes(), llvm::IsaPred<wave::WaveTensorType>);
  if (it != op->getOperandTypes().end()) {
    referenceType = llvm::cast<wave::WaveTensorType>(*it);
    os << kOperandNamePrefix
       << std::distance(op->getOperandTypes().begin(), it);
  } else {
    auto it2 = llvm::find_if(op->getResultTypes(),
                             llvm::IsaPred<wave::WaveTensorType>);
    // No tensor-typed operands or results, nothing to verify.
    if (it2 == op->getResultTypes().end())
      return llvm::success();

    referenceType = llvm::cast<wave::WaveTensorType>(*it2);
    os << kResultNamePrefix << std::distance(op->getResultTypes().begin(), it2);
  }
  assert(referenceType);

  if (llvm::failed(verifyTypeRange(op->getLoc(), op->getOperandTypes(),
                                   referenceType, includeAddressSpace,
                                   kOperandNamePrefix, os.str())))
    return llvm::failure();

  return verifyTypeRange(op->getLoc(), op->getResultTypes(), referenceType,
                         includeAddressSpace, kResultNamePrefix, os.str());
}

//-----------------------------------------------------------------------------
// Lattice implementation
//-----------------------------------------------------------------------------

wave::IndexExprsLatticeStorage::IndexExprsLatticeStorage()
    : value(nullptr, kUninitializedState) {}

wave::IndexExprsLatticeStorage::IndexExprsLatticeStorage(
    DictionaryAttr concreteValue)
    : value(concreteValue, kSpecificTypeState) {}

bool wave::IndexExprsLatticeStorage::operator==(
    const IndexExprsLatticeStorage &other) const {
  return value == other.value;
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

DictionaryAttr wave::IndexExprsLatticeStorage::getConcreteValue() const {
  if (value.getInt() != kSpecificTypeState)
    return nullptr;
  return llvm::cast<DictionaryAttr>(value.getPointer());
}

wave::IndexExprsLatticeStorage wave::IndexExprsLatticeStorage::top() {
  IndexExprsLatticeStorage result;
  result.value.setPointer(nullptr);
  result.value.setInt(kUndecidableState);
  return result;
}

wave::IndexExprsLatticeStorage wave::IndexExprsLatticeStorage::bottom() {
  IndexExprsLatticeStorage result;
  result.value.setPointer(nullptr);
  result.value.setInt(kUninitializedState);
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
  // TODO: compute waves_per_block from wave constraints; this should be
  // done in the attribute itself. Maybe move this to the attribute
  // verifier.
  llvm::ArrayRef<unsigned> wavesPerBlock =
      hardwareConstraint.getWavesPerBlock();
  if (wavesPerBlock.size() != 3) {
    emitError(loc) << "expected a waves_per_block entry with three "
                      "elements in the hardware constraint";
    return nullptr;
  }

  return hardwareConstraint;
}

llvm::FailureOr<wave::IndexExprsAnalysisInit>
wave::IndexExprsAnalysisInit::create(Location loc, Attribute constraintsAttr) {
  wave::IndexExprsAnalysisInit initObject;
  initObject.hardwareConstraint =
      parseWaveConstraints(loc, constraintsAttr, initObject.symbolConstraints);
  if (initObject.hardwareConstraint == nullptr)
    return llvm::failure();
  initObject.wavesPerBlock = initObject.hardwareConstraint.getWavesPerBlock();
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

// Return the list of thread-like index symbols.
// TODO: It would be nice to cache the list somehow, but we need to tie it to
// the context and ensure thread safety, potentially by storing it as an
// attribute or some other named/typed entity in the context object.
static SmallVector<Attribute> getThreadLikeIndexSymbols(MLIRContext *ctx) {
  return llvm::map_to_vector(
      ArrayRef{wave::WaveIndexSymbol::THREAD_0, wave::WaveIndexSymbol::THREAD_1,
               wave::WaveIndexSymbol::THREAD_2,
               wave::WaveIndexSymbol::GPR_NUMBER},
      [&](wave::WaveIndexSymbol symbol) -> Attribute {
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
    ArrayRef<unsigned> disallowedSymbols) {
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
    for (unsigned symbol : disallowedSymbols) {
      if (expr.isFunctionOfSymbol(symbol))
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
    ArrayRef<Attribute> rhsSymbols, ArrayRef<Attribute> allSymbols,
    ArrayRef<unsigned> disallowedSymbols) {
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

// Join two concrete index expressions mappings by joining their
// start/step/stride maps independently. See getIndexExprStartJoinedMap and
// getIndexExprStepStrideJoinedMap for more details.
static wave::WaveIndexMappingAttr
getIndexExprsJoinMappings(wave::WaveIndexMappingAttr lhs,
                          wave::WaveIndexMappingAttr rhs) {

  lhs = lhs.removeUnusedInputs();
  rhs = rhs.removeUnusedInputs();

  // Collect all unique symbol names from both index mappings in order.
  llvm::SmallVector<Attribute> allSymbols;
  llvm::SetVector<Attribute> lhsSymbols(llvm::from_range, lhs.getSymbols());
  llvm::SetVector<Attribute> rhsSymbols(llvm::from_range, rhs.getSymbols());
  wave::aggregateAllSymbols(llvm::ArrayRef{lhsSymbols, rhsSymbols}, allSymbols);
  llvm::SetVector<Attribute> commonSymbols =
      llvm::set_intersection(lhsSymbols, rhsSymbols);
  llvm::SmallVector<unsigned> commonSymbolPositions;
  for (Attribute symbol : commonSymbols) {
    auto it = llvm::find(lhsSymbols, symbol);
    assert(it != lhsSymbols.end());
    commonSymbolPositions.push_back(std::distance(lhsSymbols.begin(), it));
  }

  FailureOr<AffineMap> joinedStart = getIndexExprStartJoinedMap(
      lhs.getStart(), rhs.getStart(), lhsSymbols.getArrayRef(),
      rhsSymbols.getArrayRef(), allSymbols, commonSymbolPositions);
  if (failed(joinedStart))
    return nullptr;
  FailureOr<AffineMap> joinedStep = getIndexExprStepStrideJoinedMap(
      lhs.getStep(), rhs.getStep(), lhsSymbols.getArrayRef(),
      rhsSymbols.getArrayRef(), allSymbols, commonSymbolPositions);
  if (failed(joinedStep))
    return nullptr;
  FailureOr<AffineMap> joinedStride = getIndexExprStepStrideJoinedMap(
      lhs.getStride(), rhs.getStride(), lhsSymbols.getArrayRef(),
      rhsSymbols.getArrayRef(), allSymbols, commonSymbolPositions);
  if (failed(joinedStride))
    return nullptr;

  return wave::WaveIndexMappingAttr::get(
      lhs.getContext(), allSymbols, *joinedStart, *joinedStep, *joinedStride);
}

wave::IndexExprsLatticeStorage wave::IndexExprsLatticeStorage::join(
    const IndexExprsLatticeStorage &lhs, const IndexExprsLatticeStorage &rhs,
    llvm::ArrayRef<Attribute> ignoredRhsSymbols) {
  if (lhs.value == rhs.value)
    return lhs;

  // Top is saturating.
  if (lhs.isTop() || rhs.isTop())
    return top();

  // Only named symbols may be ignored.
  llvm::StringSet<> ignoredRhsSymbolNames;
  for (Attribute attr : ignoredRhsSymbols) {
    auto symbolAttr = llvm::dyn_cast<wave::WaveSymbolAttr>(attr);
    if (!symbolAttr)
      continue;
    ignoredRhsSymbolNames.insert(symbolAttr.getName());
  }

  // Even if LHS is bottom, we still need to filter out ignored symbols.
  if (lhs.isBottom()) {
    if (ignoredRhsSymbols.empty() || rhs.isBottom())
      return rhs;

    llvm::SmallVector<NamedAttribute> filtered = llvm::filter_to_vector(
        rhs.getConcreteValue(), [&](NamedAttribute namedAttr) {
          return !ignoredRhsSymbolNames.contains(
              namedAttr.getName().getValue());
        });
    return IndexExprsLatticeStorage(
        DictionaryAttr::get(rhs.getConcreteValue().getContext(), filtered));
  }

  if (rhs.isBottom())
    return lhs;

  MLIRContext *ctx = lhs.getConcreteValue().getContext();
  DictionaryAttr lhsValue = lhs.getConcreteValue();
  DictionaryAttr rhsValue = rhs.getConcreteValue();

  // Join specific values per symbol.
  llvm::DenseMap<StringAttr, Attribute> result;
  for (NamedAttribute namedAttr : lhsValue) {
    result[namedAttr.getName()] = namedAttr.getValue();
  }
  for (NamedAttribute namedAttr : rhsValue) {
    if (ignoredRhsSymbolNames.contains(namedAttr.getName().getValue()))
      continue;

    // If the mapping for the symbol doesn't exist in the result yet, just take
    // it from the RHS.
    auto [it, inserted] =
        result.try_emplace(namedAttr.getName(), namedAttr.getValue());
    if (inserted)
      continue;

    // The symbol has a mapping on both LHS and RHS, join them.
    auto lhsValue = llvm::cast<wave::WaveIndexMappingAttr>(it->getSecond());
    auto rhsValue =
        llvm::cast<wave::WaveIndexMappingAttr>(namedAttr.getValue());
    if (lhsValue == rhsValue)
      continue;

    wave::WaveIndexMappingAttr joinedMapping =
        getIndexExprsJoinMappings(lhsValue, rhsValue);
    if (!joinedMapping)
      return IndexExprsLatticeStorage::top();

    result[namedAttr.getName()] = joinedMapping;
  }
  return IndexExprsLatticeStorage(
      DictionaryAttr::get(ctx, llvm::map_to_vector(result, [](auto &&pair) {
                            return NamedAttribute(pair.first, pair.second);
                          })));
}

wave::IndexExprsLatticeStorage
wave::IndexExprsLatticeStorage::meet(const IndexExprsLatticeStorage &lhs,
                                     const IndexExprsLatticeStorage &rhs) {
  return join(lhs, rhs);
}

void wave::IndexExprsLatticeStorage::unsafeSet(
    const IndexExprsLatticeStorage &value) {
  this->value = value.value;
}

wave::IndexExprsLatticeStorage wave::IndexExprsLatticeStorage::keepOnlySymbols(
    llvm::ArrayRef<wave::WaveSymbolAttr> symbols) const {
  if (isBottom() || isTop())
    return *this;

  llvm::StringSet<> symbolNames;
  for (wave::WaveSymbolAttr symbol : symbols)
    symbolNames.insert(symbol.getName());

  llvm::SmallVector<NamedAttribute> filtered =
      llvm::filter_to_vector(getConcreteValue(), [&](NamedAttribute attr) {
        return symbolNames.contains(attr.getName().getValue());
      });

  if (filtered.empty())
    return bottom();

  return IndexExprsLatticeStorage(
      DictionaryAttr::get(getConcreteValue().getContext(), filtered));
}

wave::IndexExprsLatticeStorage
wave::IndexExprsLatticeStorage::withoutIterSymbols(
    llvm::ArrayRef<wave::WaveSymbolAttr> iterSymbols) const {
  if (isBottom() || isTop())
    return *this;

  MLIRContext *ctx = getConcreteValue().getContext();
  llvm::SmallVector<NamedAttribute> updated =
      llvm::map_to_vector(getConcreteValue(), [&](NamedAttribute attr) {
        auto value = llvm::cast<wave::WaveIndexMappingAttr>(attr.getValue());
        for (wave::WaveSymbolAttr iterSymbol : iterSymbols) {
          auto actualIterSymbol =
              wave::WaveIterSymbolAttr::get(ctx, iterSymbol.getName());
          value = value.removeInput(actualIterSymbol).removeUnusedInputs();
        }
        return NamedAttribute(attr.getName(), value);
      });
  return IndexExprsLatticeStorage(DictionaryAttr::get(ctx, updated));
}

void wave::IndexExprsLatticeStorage::print(llvm::raw_ostream &os) const {
  if (isBottom()) {
    os << "<bottom>";
  } else if (isTop()) {
    os << "<top>";
  } else {
    os << getConcreteValue();
  }
}

void wave::IndexExprsLatticeStorage::dump() const { print(llvm::errs()); }

void wave::operator<<(Diagnostic &diag, const IndexExprsLatticeStorage &value) {
  std::string str;
  llvm::raw_string_ostream os(str);
  value.print(os);
  diag << os.str();
}

llvm::FailureOr<ChangeResult> wave::detail::identityIndexExprsPropagate(
    llvm::ArrayRef<IndexExprsLatticeStorage> from,
    llvm::MutableArrayRef<IndexExprsLatticeStorage> to, TypeRange toTypes,
    llvm::StringRef fromName, llvm::StringRef toName,
    wave::EmitErrorFn emitError) {
  // Join all "from" lattices.
  IndexExprsLatticeStorage joined = IndexExprsLatticeStorage::bottom();
  bool fromTop = false;
  for (const IndexExprsLatticeStorage &fromLattice : from) {
    // If one of the from lattices reached the top, no need to keep joining, the
    // result is known to be top.
    if (fromLattice.isTop()) {
      fromTop = true;
      joined = IndexExprsLatticeStorage::top();
      break;
    }
    joined = IndexExprsLatticeStorage::join(joined, fromLattice);
  }

  // Report if joining non-top "from" lattices reached the top as this is
  // indicative of a "sideways" conflict.
  if (joined.isTop() && !fromTop) {
    InFlightDiagnostic diag = emitError()
                              << "incompatible " << fromName << " lattices"
                              << " when propagating from those to " << toName;
    for (auto &&[i, fromLattice] : llvm::enumerate(from)) {
      diag.attachNote() << fromName << " #" << i << " lattice: " << fromLattice;
    }
    return diag;
  }

  // Propagate to all "to" lattices.
  ChangeResult changeResult = ChangeResult::NoChange;
  for (auto &&[i, toLattice] : llvm::enumerate(to)) {
    auto tensorType = llvm::dyn_cast<wave::WaveTensorType>(toTypes[i]);
    if (!tensorType) {
      toLattice = IndexExprsLatticeStorage::top();
      continue;
    }
    IndexExprsLatticeStorage filtered =
        joined.keepOnlySymbols(tensorType.getShape());
    IndexExprsLatticeStorage newLattice =
        IndexExprsLatticeStorage::join(toLattice, filtered);

    // Report an error only when the lattice moved to the top state, which is
    // indicative of a conflict, not when it was in the top state to start with.
    // Also avoid reporting errors when joined lattice is in the top state: it
    // is either reported above or it means the conflict was found elsewhere and
    // we are just propagating it.
    if (newLattice.isTop() && !toLattice.isTop() && !filtered.isTop()) {
      InFlightDiagnostic diag =
          emitError() << "conflict when propagating index expressions from "
                      << fromName << " to " << toName << " #" << i;
      diag.attachNote() << "original " << toName << " lattice: " << toLattice;
      for (auto &&[j, fromLattice] : llvm::enumerate(from)) {
        diag.attachNote() << fromName << " #" << j
                          << " lattice: " << fromLattice;
      }
      return diag;
    }

    if (newLattice != toLattice) {
      changeResult = ChangeResult::Change;
      toLattice = newLattice;
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

llvm::LogicalResult wave::detail::identitySetIndexFromLattices(
    Operation *op,
    [[maybe_unused]] llvm::ArrayRef<IndexExprsLatticeStorage> operandExprs,
    llvm::ArrayRef<IndexExprsLatticeStorage> resultExprs) {
  llvm::SmallVector<Attribute> indexExprs;
  indexExprs.reserve(resultExprs.size());
  for (auto &&[i, expr] : llvm::enumerate(resultExprs)) {
    if (!llvm::isa<wave::WaveTensorType>(op->getResult(i).getType()))
      continue;
    if (failed(checkAndAppendIndexExpr(op->getLoc(), resultExprs[i],
                                       "result #" + llvm::Twine(i),
                                       indexExprs)))
      return llvm::failure();
  }
  op->setAttr(wave::WaveDialect::kIndexWaveExprListAttrName,
              ArrayAttr::get(op->getContext(), indexExprs));
  return llvm::success();
}

#include "water/Dialect/Wave/IR/WaveOpInterfaces.cpp.inc"
