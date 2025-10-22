// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SmallString.h"

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

  auto dictAttr = dyn_cast<DictionaryAttr>(attribute);
  if (!dictAttr)
    return op->emitError("'index' attribute must be a dictionary");

  for (auto named : dictAttr) {
    auto val = named.getValue();
    if (!isa<wave::WaveIndexMappingAttr>(val))
      return op->emitError(
                 "index attribute values must be WaveIndexMappingAttr, got ")
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

    unsigned declared = mapping.getSymbolNames().size();
    if (startMap.getNumSymbols() != declared ||
        stepMap.getNumSymbols() != declared ||
        strideMap.getNumSymbols() != declared)
      return op->emitError("inconsistent symbol count between symbol_names and "
                           "affine maps for index symbol '")
             << named.getName() << "' (expected " << declared
             << ", got start=" << startMap.getNumSymbols()
             << ", step=" << stepMap.getNumSymbols()
             << ", stride=" << strideMap.getNumSymbols() << ")";
  }
  return success();
}

//-----------------------------------------------------------------------------
// Custom printing/parsing components
//-----------------------------------------------------------------------------

// ODS custom directive: parseWaveIndexDict/printWaveIndexDict
ParseResult wave::parseWaveIndexDict(OpAsmParser &parser, DictionaryAttr &out) {
  llvm::SmallVector<NamedAttribute, 4> mappingDict;
  if (parser.parseLBrace())
    return failure();

  auto parseOne = [&]() -> ParseResult {
    llvm::StringRef symbolName;
    if (parser.parseKeyword(&symbolName) || parser.parseColon())
      return failure();

    // Parse WaveIndexMappingAttr using its custom parser to allow concise
    // syntax.
    auto indexMapping = wave::WaveIndexMappingAttr::parse(parser, Type{});
    if (!indexMapping)
      return failure();

    mappingDict.emplace_back(parser.getBuilder().getStringAttr(symbolName),
                             indexMapping);
    return success();
  };

  if (parser.parseCommaSeparatedList(parseOne) || parser.parseRBrace())
    return failure();

  out = parser.getBuilder().getDictionaryAttr(mappingDict);
  return success();
}

void wave::printWaveIndexDict(OpAsmPrinter &printer, Operation *op,
                              DictionaryAttr dict) {
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

#include "water/Dialect/Wave/IR/WaveOpInterfaces.cpp.inc"
