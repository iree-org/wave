// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "water/Dialect/Wave/IR/IndexExpr.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SmallString.h"
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

llvm::FailureOr<mlir::ChangeResult> wave::detail::checkPropagateShapeConflict(
    wave::WaveTensorType from, wave::WaveTensorType to,
    llvm::StringRef fromName, llvm::StringRef toName, llvm::raw_ostream &errs) {
  if (!from || !to || hasSameShape(from, to))
    return mlir::ChangeResult::NoChange;

  if (!to.getFullySpecified())
    return mlir::ChangeResult::Change;

  errs << "irreconcilable types during type inference from " << fromName << "("
       << from << ") to " << toName << "(" << to << ")";
  return mlir::failure();
}

llvm::FailureOr<mlir::ChangeResult> wave::detail::propagateShapeInformation(
    wave::WaveTensorType from, wave::WaveTensorType &to,
    llvm::StringRef fromName, llvm::StringRef toName, llvm::raw_ostream &errs) {
  llvm::FailureOr<mlir::ChangeResult> res =
      checkPropagateShapeConflict(from, to, fromName, toName, errs);
  if (mlir::failed(res) || *res == mlir::ChangeResult::NoChange)
    return res;

  to = to.copyShapeFrom(from);
  return mlir::ChangeResult::Change;
}

llvm::FailureOr<mlir::ChangeResult>
wave::detail::identityTypeInferencePropagate(
    llvm::ArrayRef<wave::WaveTensorType> from,
    llvm::MutableArrayRef<wave::WaveTensorType> to, llvm::StringRef fromName,
    llvm::StringRef toName, llvm::raw_ostream &errs) {
  auto it = llvm::find_if(from, [](wave::WaveTensorType type) {
    return type && type.getFullySpecified();
  });
  if (it == from.end())
    return mlir::ChangeResult::NoChange;

  // Expect all fully-specified "from" types to have the same shape.
  for (auto [i, fr] : llvm::enumerate(from)) {
    llvm::FailureOr<mlir::ChangeResult> res =
        checkPropagateShapeConflict(*it, fr, fromName, toName, errs);
    if (mlir::failed(res)) {
      errs << " for " << fromName << " #" << i;
      return res;
    }
  }

  mlir::ChangeResult changeResult = mlir::ChangeResult::NoChange;
  for (auto &&[i, toType] : llvm::enumerate(to)) {
    llvm::FailureOr<mlir::ChangeResult> res =
        propagateShapeInformation(*it, toType, fromName, toName, errs);
    if (mlir::failed(res)) {
      errs << " for " << fromName << " #" << i;
      return mlir::failure();
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

llvm::FailureOr<mlir::ChangeResult>
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
    return mlir::failure();
  }

  mlir::ChangeResult changeResult = mlir::ChangeResult::NoChange;
  for (auto &&[i, toType] : llvm::enumerate(mutableValues)) {
    ElementsPerThreadLatticeValue joined =
        ElementsPerThreadLatticeValue::join(from, toType);

    if (joined.isTop() && !from.isTop() && !toType.isTop()) {
      printElementsPerThreadMismatchMsg(errs, from, toType, fromName,
                                        mutableName, i);
      toType = ElementsPerThreadLatticeValue::top();
      return mlir::failure();
    }

    if (joined != toType) {
      changeResult = mlir::ChangeResult::Change;
      toType = joined;
    }
  }
  return changeResult;
}

llvm::FailureOr<mlir::ChangeResult>
wave::detail::identityElementsPerThreadPropagate(
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
    std::optional<mlir::Location> loc, llvm::StringRef lhsName,
    wave::WaveTensorType lhs, llvm::ArrayRef<int> lhsDims,
    llvm::StringRef rhsName, wave::WaveTensorType rhs,
    llvm::ArrayRef<int> rhsDims) {
  assert(lhsDims.size() == rhsDims.size() &&
         "expected lhs and rhs dim lists to be co-indexed");

  // Under-specified types are okay everywhere.
  if (!lhs.getFullySpecified() || !rhs.getFullySpecified())
    return mlir::success();

  llvm::SmallVector<int> lhsDimsVec(lhsDims), rhsDimsVec(rhsDims);
  updateNegativeIndices(lhsDimsVec, lhs.getRank());
  updateNegativeIndices(rhsDimsVec, rhs.getRank());
  for (auto &&[lhsDim, rhsDim] : llvm::zip_equal(lhsDimsVec, rhsDimsVec)) {
    wave::WaveSymbolAttr lhsExpr = lhs.getShape()[lhsDim];
    wave::WaveSymbolAttr rhsExpr = rhs.getShape()[rhsDim];
    if (lhsExpr == rhsExpr)
      continue;

    if (loc) {
      mlir::emitError(*loc)
          << "expected " << lhsName << " dimension #" << lhsDim << " ("
          << lhsExpr << ") to match " << rhsName << " dimension #" << rhsDim
          << " (" << rhsExpr << ")";
    }
    return mlir::failure();
  }
  return mlir::success();
}

llvm::LogicalResult wave::detail::verifyElementTypesMatch(
    std::optional<mlir::Location> loc, llvm::StringRef lhsName,
    wave::WaveTensorType lhs, llvm::StringRef rhsName,
    wave::WaveTensorType rhs) {
  if (lhs.getElementType() == rhs.getElementType())
    return mlir::success();

  if (loc) {
    mlir::emitError(*loc) << "expected " << lhsName << " and " << rhsName
                          << " elemental types to match, got "
                          << lhs.getElementType() << ", "
                          << rhs.getElementType();
  }
  return mlir::failure();
}

llvm::LogicalResult wave::detail::verifyTypesCompatible(
    wave::WaveTensorType lhs, wave::WaveTensorType rhs,
    bool includeAddressSpace, std::optional<mlir::Location> errorLocation,
    llvm::StringRef lhsName, llvm::StringRef rhsName) {
  // Fast and cheap path.
  if (lhs == rhs)
    return mlir::success();

  if (errorLocation) {
    assert(!lhsName.empty() && !rhsName.empty() &&
           "expected names when location is provided");
  }

  if (includeAddressSpace) {
    if (lhs.getAddressSpaceValue() != rhs.getAddressSpaceValue() &&
        lhs.getAddressSpaceValue() != wave::WaveAddressSpace::Unspecified &&
        rhs.getAddressSpaceValue() != wave::WaveAddressSpace::Unspecified) {
      if (errorLocation) {
        emitError(*errorLocation) << "address space mismatch between" << lhsName
                                  << " and " << rhsName;
      }
      return mlir::failure();
    }
  }

  if (mlir::failed(
          verifyElementTypesMatch(errorLocation, lhsName, lhs, rhsName, rhs)))
    return mlir::failure();

  if (!lhs.getFullySpecified() || !rhs.getFullySpecified())
    return mlir::success();

  if (lhs.getRank() != rhs.getRank()) {
    if (errorLocation) {
      emitError(*errorLocation)
          << "rank mismatch between " << lhsName << " and " << rhsName;
    }
    return mlir::failure();
  }

  auto allDims = llvm::to_vector(llvm::iota_range<int>(0, lhs.getRank(),
                                                       /*Inclusive=*/false));
  return verifyTypesMatchingDimensions(errorLocation, lhsName, lhs, allDims,
                                       rhsName, rhs, allDims);
}

static llvm::LogicalResult
verifyTypeRange(mlir::Location loc, mlir::TypeRange range,
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

    if (mlir::failed(wave::detail::verifyTypesCompatible(
            tensorType, referenceType, includeAddressSpace, loc, os.str(),
            referenceDescription))) {
      return llvm::failure();
    }
  }
  return llvm::success();
}

llvm::LogicalResult wave::detail::verifyCompatibleOperandsAndResultsOpTrait(
    mlir::Operation *op, bool includeAddressSpace) {
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
    mlir::DictionaryAttr concreteValue)
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

mlir::DictionaryAttr wave::IndexExprsLatticeStorage::getConcreteValue() const {
  if (value.getInt() != kSpecificTypeState)
    return nullptr;
  return llvm::cast<mlir::DictionaryAttr>(value.getPointer());
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
    mlir::Location loc, mlir::Attribute constraints,
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
wave::IndexExprsAnalysisInit::create(mlir::Location loc,
                                     mlir::Attribute constraintsAttr) {
  wave::IndexExprsAnalysisInit initObject;
  initObject.hardwareConstraint =
      parseWaveConstraints(loc, constraintsAttr, initObject.symbolConstraints);
  if (initObject.hardwareConstraint == nullptr)
    return llvm::failure();
  initObject.wavesPerBlock = initObject.hardwareConstraint.getWavesPerBlock();
  return initObject;
}

// Populate `positions` with the positions of thread and GPR-related symbols in
// the given list.
static void
getThreadLikeSymbolPositions(llvm::ArrayRef<mlir::Attribute> symbols,
                             llvm::SmallVectorImpl<unsigned> &positions) {
  for (auto &&[i, symbol] : llvm::enumerate(symbols)) {
    auto indexSymbol = llvm::dyn_cast<wave::WaveIndexSymbolAttr>(symbol);
    if (!indexSymbol)
      continue;
    if (llvm::is_contained({wave::WaveIndexSymbol::THREAD_0,
                            wave::WaveIndexSymbol::THREAD_1,
                            wave::WaveIndexSymbol::THREAD_2,
                            wave::WaveIndexSymbol::GPR_NUMBER},
                           indexSymbol.getValue()))
      positions.push_back(i);
  }
}

wave::IndexExprsLatticeStorage wave::IndexExprsLatticeStorage::join(
    const IndexExprsLatticeStorage &lhs, const IndexExprsLatticeStorage &rhs,
    llvm::ArrayRef<mlir::Attribute> ignoredRhsSymbols) {
  if (lhs.value == rhs.value)
    return lhs;

  if (lhs.isTop() || rhs.isTop())
    return top();

  llvm::StringSet<> ignoredRhsSymbolNames;
  for (mlir::Attribute attr : ignoredRhsSymbols) {
    auto symbolAttr = llvm::dyn_cast<wave::WaveSymbolAttr>(attr);
    if (!symbolAttr)
      continue;
    ignoredRhsSymbolNames.insert(symbolAttr.getName());
  }

  if (lhs.isBottom()) {
    if (ignoredRhsSymbols.empty() || rhs.isBottom())
      return rhs;

    llvm::SmallVector<mlir::NamedAttribute> filtered = llvm::filter_to_vector(
        rhs.getConcreteValue(), [&](mlir::NamedAttribute namedAttr) {
          return !ignoredRhsSymbolNames.contains(
              namedAttr.getName().getValue());
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
    if (ignoredRhsSymbolNames.contains(namedAttr.getName().getValue()))
      continue;

    // If the mapping for the symbol doesn't exist in the result yet, just take
    // it from the RHS.
    auto [it, inserted] =
        result.try_emplace(namedAttr.getName(), namedAttr.getValue());
    if (inserted)
      continue;

    // Actually join otherwise.
    auto lhsValue = llvm::cast<wave::WaveIndexMappingAttr>(it->getSecond());
    auto rhsValue =
        llvm::cast<wave::WaveIndexMappingAttr>(namedAttr.getValue());
    if (lhsValue == rhsValue)
      continue;

    auto isThreadDependent = [&](wave::WaveIndexMappingAttr val) -> bool {
      llvm::SmallVector<unsigned> threadLikeSymbolPositions;
      getThreadLikeSymbolPositions(val.getSymbols(), threadLikeSymbolPositions);
      return llvm::any_of(
          llvm::ArrayRef{val.getStart(), val.getStep(), val.getStride()},
          [&](mlir::AffineMap map) {
            return llvm::any_of(threadLikeSymbolPositions, [&](unsigned pos) {
              return map.isFunctionOfSymbol(pos);
            });
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
    llvm::SmallVector<mlir::Attribute> allSymbols;
    llvm::ArrayRef<mlir::Attribute> threadDependentSymbols =
        threadDependentMapping.getSymbols();
    llvm::ArrayRef<mlir::Attribute> threadIndependentSymbols =
        threadIndependentMapping.getSymbols();
    wave::aggregateAllSymbols(
        llvm::ArrayRef{threadIndependentSymbols, threadDependentSymbols},
        allSymbols);

    mlir::AffineMap threadDependentStart = alignMapSymbols(
        threadDependentMapping.getStart(), threadDependentSymbols, allSymbols);
    mlir::AffineMap threadIndependentStart =
        alignMapSymbols(threadIndependentMapping.getStart(),
                        threadIndependentSymbols, allSymbols);

    mlir::AffineMap threadDependentStep = alignMapSymbols(
        threadDependentMapping.getStep(), threadDependentSymbols, allSymbols);
    mlir::AffineMap threadIndependentStep =
        alignMapSymbols(threadIndependentMapping.getStep(),
                        threadIndependentSymbols, allSymbols);

    mlir::AffineMap threadDependentStride = alignMapSymbols(
        threadDependentMapping.getStride(), threadDependentSymbols, allSymbols);
    mlir::AffineMap threadIndependentStride =
        alignMapSymbols(threadIndependentMapping.getStride(),
                        threadIndependentSymbols, allSymbols);

    // Subtract the thread-independent from thread-dependent for each.
    auto subtractMaps = [&](mlir::AffineMap a,
                            mlir::AffineMap b) -> mlir::AffineMap {
      // Assert there is only one result expression in each map.
      assert(a.getNumResults() == 1 &&
             "expected a single result expression in affine map 'a'");
      assert(b.getNumResults() == 1 &&
             "expected a single result expression in affine map 'b'");
      mlir::AffineExpr subtracted = a.getResult(0) - b.getResult(0);
      return mlir::AffineMap::get(a.getNumDims(), a.getNumSymbols(), subtracted,
                                  ctx);
    };
    mlir::AffineMap newStart =
        subtractMaps(threadDependentStart, threadIndependentStart);
    mlir::AffineMap newStep =
        subtractMaps(threadDependentStep, threadIndependentStep);
    mlir::AffineMap newStride =
        subtractMaps(threadDependentStride, threadIndependentStride);

    llvm::SmallVector<unsigned> threadLikeSymbolPositions;
    getThreadLikeSymbolPositions(allSymbols, threadLikeSymbolPositions);
    auto isOnlyThreadDependent = [&](mlir::AffineMap map) {
      mlir::WalkResult walkResult =
          map.getResult(0).walk([&](mlir::AffineExpr expr) {
            auto symExpr = llvm::dyn_cast<mlir::AffineSymbolExpr>(expr);
            if (!symExpr)
              return mlir::WalkResult::advance();
            if (!llvm::is_contained(threadLikeSymbolPositions,
                                    symExpr.getPosition()))
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

  llvm::SmallVector<mlir::NamedAttribute> filtered = llvm::filter_to_vector(
      getConcreteValue(), [&](mlir::NamedAttribute attr) {
        return symbolNames.contains(attr.getName().getValue());
      });

  if (filtered.empty())
    return bottom();

  return IndexExprsLatticeStorage(
      mlir::DictionaryAttr::get(getConcreteValue().getContext(), filtered));
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

void wave::operator<<(mlir::Diagnostic &diag,
                      const IndexExprsLatticeStorage &value) {
  std::string str;
  llvm::raw_string_ostream os(str);
  value.print(os);
  diag << os.str();
}

llvm::FailureOr<mlir::ChangeResult> wave::detail::identityIndexExprsPropagate(
    llvm::ArrayRef<IndexExprsLatticeStorage> from,
    llvm::MutableArrayRef<IndexExprsLatticeStorage> to, mlir::TypeRange toTypes,
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
    mlir::InFlightDiagnostic diag =
        emitError() << "incompatible " << fromName
                    << " when propagating from those to " << toName;
    for (auto &&[i, fromLattice] : llvm::enumerate(from)) {
      diag.attachNote() << fromName << " #" << i << " lattice: " << fromLattice;
    }
    return diag;
  }

  // Propagate to all "to" lattices.
  mlir::ChangeResult changeResult = mlir::ChangeResult::NoChange;
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
      mlir::InFlightDiagnostic diag =
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
      changeResult = mlir::ChangeResult::Change;
      toLattice = newLattice;
    }
  }
  return changeResult;
}

llvm::LogicalResult wave::detail::checkAndAppendIndexExpr(
    mlir::Location loc, const IndexExprsLatticeStorage &expr,
    const llvm::Twine &description,
    llvm::SmallVectorImpl<mlir::Attribute> &indexExprs) {
  if (expr.isBottom()) {
    emitError(loc) << "failed to infer index expressions for " << description;
    return llvm::failure();
  }
  if (expr.isTop()) {
    mlir::InFlightDiagnostic diag =
        emitError(loc) << "conflict detected in index expressions for "
                       << description;
    diag.attachNote() << "PLEASE REPORT this as a bug in absence of further "
                         "information about the conflict";
    return llvm::failure();
  }
  indexExprs.push_back(expr.getConcreteValue());
  return llvm::success();
}

llvm::LogicalResult wave::detail::identitySetIndexFromLattices(
    mlir::Operation *op,
    [[maybe_unused]] llvm::ArrayRef<IndexExprsLatticeStorage> operandExprs,
    llvm::ArrayRef<IndexExprsLatticeStorage> resultExprs) {
  llvm::SmallVector<mlir::Attribute> indexExprs;
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
              mlir::ArrayAttr::get(op->getContext(), indexExprs));
  return llvm::success();
}

#include "water/Dialect/Wave/IR/WaveOpInterfaces.cpp.inc"
