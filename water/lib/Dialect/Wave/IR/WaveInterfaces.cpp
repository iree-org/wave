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
#include "water/Dialect/Wave/IR/WaveUtils.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "wave-interfaces"

using namespace mlir;

//-----------------------------------------------------------------------------
// getHyperparameters
//-----------------------------------------------------------------------------

wave::WaveHyperparameterAttr wave::getHyperparameters(Operation *op) {
  for (Operation *current = op; current; current = current->getParentOp()) {
    if (auto hyperparams = current->getAttrOfType<WaveHyperparameterAttr>(
            WaveDialect::kHyperparameterAttrName))
      return hyperparams;
  }
  return nullptr;
}

//-----------------------------------------------------------------------------
// Index attribute verification
//-----------------------------------------------------------------------------

LogicalResult wave::verifyWaveIndexMappings(Operation *op) {
  // Expected number of per-value index / vector_shape slots (same convention as
  // the `index` attribute).
  size_t expectedSlotCount = 0;
  if (auto iface = dyn_cast<wave::WaveInferIndexExprsOpInterface>(op)) {
    llvm::SmallVector<Value> values;
    iface.getIndexExprValuesAndDescriptions(values);
    expectedSlotCount = values.size();
  } else {
    expectedSlotCount = op->getNumResults();
  }

  auto verifyVectorShapeArray = [&](ArrayAttr vsArr) -> LogicalResult {
    if (vsArr.size() != expectedSlotCount)
      return op->emitError("'vector_shape' attribute length (")
             << vsArr.size() << ") does not match the number of per-value "
             << "slots (" << expectedSlotCount << ")";
    for (Attribute nestedAttr : vsArr) {
      auto mapping = dyn_cast<wave::WaveSymbolMappingAttr>(nestedAttr);
      if (!mapping)
        return op->emitError(
            "'vector_shape' array elements must be WaveSymbolMappingAttr");
      for (auto [key, value] : mapping.getMapping()) {
        auto intAttr = dyn_cast<IntegerAttr>(value);
        if (!intAttr)
          return op->emitError("vector_shape entry ")
                 << key << " must be an integer attribute";
        if (!intAttr.getType().isSignlessInteger(64))
          return op->emitError("vector_shape entry ")
                 << key << " must be a 64-bit signless integer attribute, got "
                 << intAttr.getType();
      }
    }
    return success();
  };

  if (Attribute vsAttr = op->getAttr(WaveDialect::kVectorShapeAttrName)) {
    auto vsArr = dyn_cast<ArrayAttr>(vsAttr);
    if (!vsArr)
      return op->emitError("'vector_shape' attribute must be an array of "
                           "WaveSymbolMappingAttr");
    if (failed(verifyVectorShapeArray(vsArr)))
      return failure();
  }

  // The index attribute is optional.
  Attribute attribute =
      op->getAttr(wave::WaveDialect::kIndexWaveExprListAttrName);
  if (!attribute) {
    // `vector_shape` without `index` is still validated above against slot
    // count.
    return success();
  }

  auto arr = dyn_cast<ArrayAttr>(attribute);
  if (!arr)
    return op->emitError(
        "'index' attribute must be an array of symbol mappings");

  SmallVector<wave::WaveSymbolMappingAttr> mappings;
  mappings.reserve(arr.size());
  for (Attribute nestedAttr : arr) {
    auto mapping = dyn_cast<wave::WaveSymbolMappingAttr>(nestedAttr);
    if (!mapping)
      return op->emitError(
          "'index' array elements must be WaveSymbolMappingAttr");
    mappings.push_back(mapping);
  }

  for (wave::WaveSymbolMappingAttr indexMapping : mappings) {
    for (auto &&[key, val] : indexMapping.getMapping()) {
      if (!isa<wave::WaveIndexMappingAttr>(val))
        return op->emitError("'index' attribute value for key ")
               << key.getName() << " must be WaveIndexMappingAttr, got " << val;

      auto mapping = cast<wave::WaveIndexMappingAttr>(val);
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
          WaveSymbolAttr parentIterSymbol =
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

  // For ops with the index attribute, verify that (1) each index expression has
  // at most one dimension whose step evaluates to a static value different from
  // 1 (with hyperparameters substituted), and (2) when step or stride can be
  // evaluated to a concrete value, that value is strictly positive. Be
  // defensive because we may not have verified anything but the basic
  // well-formedness yet, e.g., the op verifier checking for single-result
  // affine expressions in mappings did not run yet.
  wave::WaveHyperparameterAttr hyperparams = wave::getHyperparameters(op);
  for (wave::WaveSymbolMappingAttr indexMapping : mappings) {
    int nonUnitCount = 0;
    for (auto &&[key, val] : indexMapping.getMapping()) {
      auto mapping = dyn_cast<wave::WaveIndexMappingAttr>(val);
      if (!mapping)
        continue;

      if (AffineMap stepMap = mapping.getStep()) {
        std::optional<SmallVector<int64_t>> stepValues =
            wave::evaluateMapWithHyperparams(stepMap, mapping.getSymbols(),
                                             hyperparams);
        if (stepValues && stepValues->size() == 1) {
          int64_t step = (*stepValues)[0];
          if (step != ShapedType::kDynamic && step <= 0) {
            return op->emitOpError()
                   << "step in index expression must be strictly positive, got "
                   << step << " for dimension " << key.getName();
          }
          if (step != 1 && step != ShapedType::kDynamic && ++nonUnitCount > 1) {
            InFlightDiagnostic diag =
                op->emitOpError()
                << "'" << WaveDialect::kIndexWaveExprListAttrName
                << "' has more than one entry with non-unit step";
            diag.attachNote()
                << "second non-unit step dimension: " << key.getName();
            return failure();
          }
        }
      }

      if (AffineMap strideMap = mapping.getStride()) {
        std::optional<SmallVector<int64_t>> strideValues =
            wave::evaluateMapWithHyperparams(strideMap, mapping.getSymbols(),
                                             hyperparams);
        if (strideValues && strideValues->size() == 1) {
          int64_t stride = (*strideValues)[0];
          if (stride != ShapedType::kDynamic && stride <= 0) {
            return op->emitOpError()
                   << "stride in index expression must be strictly positive, "
                      "got "
                   << stride << " for dimension " << key.getName();
          }
        }
      }
    }
  }

  // When the operation implements WaveInferIndexExprsOpInterface, the index
  // attribute length must match the number of values from
  // getIndexExprValuesAndDescriptions. Otherwise, default to the number of op
  // results.

  if (arr.size() != expectedSlotCount) {
    return op->emitError() << WaveDialect::kIndexWaveExprListAttrName
                           << " attribute length (" << arr.size()
                           << ") does not match the number of per-value index "
                           << "slots (" << expectedSlotCount << ")";
  }

  if (Attribute vsAttr = op->getAttr(WaveDialect::kVectorShapeAttrName)) {
    auto vsArr = cast<ArrayAttr>(vsAttr);
    if (vsArr.size() != arr.size())
      return op->emitError("'vector_shape' attribute length (")
             << vsArr.size() << ") does not match 'index' attribute length ("
             << arr.size() << ")";
  }

  return success();
}

//-----------------------------------------------------------------------------
// Custom printing/parsing components
//-----------------------------------------------------------------------------

// ODS custom directive: parseWaveIndexDict/printWaveIndexDict
ParseResult wave::parseWaveIndexDict(OpAsmParser &parser, ArrayAttr &out) {
  auto parseSingleMapping =
      [&](wave::WaveSymbolMappingAttr &out) -> ParseResult {
    SmallVector<std::pair<wave::WaveSymbolAttr, Attribute>> entries;
    MLIRContext *ctx = parser.getContext();
    auto parseEntry = [&]() -> ParseResult {
      StringRef symbolName;
      if (parser.parseKeyword(&symbolName) || parser.parseColon())
        return failure();
      WaveIndexMappingAttr mapping;
      if (failed(parser.parseCustomAttributeWithFallback(mapping)))
        return failure();
      entries.emplace_back(wave::WaveSymbolAttr::get(ctx, symbolName), mapping);
      return success();
    };
    if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Braces,
                                       parseEntry))
      return failure();
    out = wave::WaveSymbolMappingAttr::get(ctx, entries);
    return success();
  };

  SmallVector<Attribute> mappings;
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Square,
                                     [&]() -> ParseResult {
                                       wave::WaveSymbolMappingAttr mapping;
                                       if (failed(parseSingleMapping(mapping)))
                                         return failure();
                                       mappings.push_back(mapping);
                                       return success();
                                     }))
    return failure();
  out = parser.getBuilder().getArrayAttr(mappings);
  return success();
}

void wave::printWaveIndexDict(OpAsmPrinter &printer, Operation *op,
                              ArrayAttr arr) {
  auto printOne = [&](wave::WaveSymbolMappingAttr mapping) {
    printer.getStream() << "{";
    llvm::interleaveComma(
        mapping.getMapping(), printer.getStream(), [&](auto pair) {
          auto [key, value] = pair;
          printer.getStream() << key.getName() << " : ";
          if (auto mappingAttr =
                  llvm::dyn_cast<wave::WaveIndexMappingAttr>(value)) {
            mappingAttr.print(printer);
          } else {
            printer.printAttribute(value);
          }
        });
    printer.getStream() << "}";
  };
  printer.getStream() << "[";
  llvm::interleaveComma(arr, printer.getStream(), [&](Attribute a) {
    printOne(llvm::cast<wave::WaveSymbolMappingAttr>(a));
  });
  printer.getStream() << "]";
}

// ODS custom directive: parseWaveVectorShapeDictList /
// printWaveVectorShapeDictList
ParseResult wave::parseWaveVectorShapeDictList(OpAsmParser &parser,
                                               ArrayAttr &out) {
  SmallVector<Attribute> mappings;
  if (parser.parseCommaSeparatedList(
          OpAsmParser::Delimiter::Square, [&]() -> ParseResult {
            Attribute attr;
            if (parser.parseAttribute(attr))
              return failure();
            if (!isa<wave::WaveSymbolMappingAttr>(attr))
              return parser.emitError(parser.getCurrentLocation())
                     << "expected a "
                        "WaveSymbolMappingAttr";
            mappings.push_back(attr);
            return success();
          }))
    return failure();
  out = parser.getBuilder().getArrayAttr(mappings);
  return success();
}

void wave::printWaveVectorShapeDictList(OpAsmPrinter &printer, Operation *op,
                                        ArrayAttr arr) {
  printer.getStream() << "[";
  llvm::interleaveComma(arr, printer.getStream(),
                        [&](Attribute a) { printer.printAttribute(a); });
  printer.getStream() << "]";
}

//-----------------------------------------------------------------------------
// WaveInferTypeOpInterface helpers
//-----------------------------------------------------------------------------

// Check whether the shape of the `to` tensor is reconcilable with the shape
// provided in the `from` array and print error messages to errs otherwise. The
// error message uses toName and fromName to describe from and to tensors. If
// shapes are reconcilable, returns an indicator whether the to type will have
// to be updated. This version avoids constructing a tensor type, which may
// be expensive.
static FailureOr<ChangeResult>
checkPropagateShapeConflict(ArrayRef<wave::WaveSymbolAttr> from,
                            wave::WaveTensorType to, llvm::StringRef fromName,
                            llvm::StringRef toName, llvm::raw_ostream &errs) {
  if (!to || from == to.getShape())
    return ChangeResult::NoChange;

  if (!to.getFullySpecified())
    return ChangeResult::Change;

  errs << "irreconcilable types during type inference from " << fromName << "(";
  llvm::interleaveComma(from, errs);
  errs << ") to " << toName << "(" << to << ")";
  return failure();
}

llvm::FailureOr<ChangeResult> wave::detail::checkPropagateShapeConflict(
    wave::WaveTensorType from, wave::WaveTensorType to,
    llvm::StringRef fromName, llvm::StringRef toName, llvm::raw_ostream &errs) {
  if (!from)
    return ChangeResult::NoChange;

  FailureOr<ChangeResult> res = ::checkPropagateShapeConflict(
      from.getShape(), to, fromName, toName, llvm::nulls());
  if (succeeded(res))
    return res;

  errs << "irreconcilable types during type inference from " << fromName << "("
       << from << ") to " << toName << "(" << to << ")";
  return failure();
}

llvm::FailureOr<ChangeResult> wave::detail::propagateShapeInformation(
    wave::WaveTensorType from, wave::WaveTensorType &to,
    llvm::StringRef fromName, llvm::StringRef toName, llvm::raw_ostream &errs) {
  if (!from || !from.getFullySpecified())
    return ChangeResult::NoChange;
  llvm::FailureOr<ChangeResult> res =
      checkPropagateShapeConflict(from, to, fromName, toName, errs);
  if (failed(res) || *res == ChangeResult::NoChange)
    return res;

  to = to.copyShapeFrom(from);
  return ChangeResult::Change;
}

FailureOr<ChangeResult> wave::detail::propagateShapeInformation(
    ArrayRef<wave::WaveSymbolAttr> from, wave::WaveTensorType &to,
    llvm::StringRef fromName, llvm::StringRef toName, llvm::raw_ostream &errs) {
  llvm::FailureOr<ChangeResult> res =
      ::checkPropagateShapeConflict(from, to, fromName, toName, errs);
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

// Propagate type information from the reduction input type by removing the
// reduction axis from it to the given type. Report errors to `errs` using
// `toName` to identify the target type.
static FailureOr<ChangeResult>
propagateFromReductionInput(wave::WaveTensorType inputType,
                            wave::WaveSymbolAttr axis, wave::WaveTensorType &to,
                            StringRef toName, raw_ostream &errs) {
  if (!inputType || !inputType.getFullySpecified())
    return ChangeResult::NoChange;

  SmallVector<wave::WaveSymbolAttr> filteredShape = llvm::filter_to_vector(
      inputType.getShape(),
      [&](wave::WaveSymbolAttr dim) { return dim != axis; });
  assert(inputType.getRank() - 1 == filteredShape.size() &&
         "expected rank to be reduced by 1 in reduction");
  auto inferredType = wave::WaveTensorType::get(
      inputType.getContext(), filteredShape, /*fully_specified=*/true,
      inputType.getElementType(), inputType.getAddressSpace());

  return wave::detail::propagateShapeInformation(inferredType, to, "input",
                                                 toName, errs);
}

FailureOr<ChangeResult> wave::detail::propagateShapeDropTrailingDims(
    wave::WaveTensorType source, wave::WaveTensorType &target,
    StringRef sourceName, StringRef targetName, unsigned n,
    llvm::raw_ostream &errs) {
  if (!source || !source.getFullySpecified())
    return ChangeResult::NoChange;

  ArrayRef<wave::WaveSymbolAttr> expectedShape = source.getShape().drop_back(n);
  FailureOr<ChangeResult> res = ::checkPropagateShapeConflict(
      expectedShape, target, sourceName, targetName, errs);
  if (failed(res) || *res == ChangeResult::NoChange)
    return res;

  target = target.copyShapeFrom(expectedShape);
  return ChangeResult::Change;
}

FailureOr<ChangeResult> wave::detail::propagateShapeAddTrailingDims(
    wave::WaveTensorType source, wave::WaveTensorType &target,
    StringRef sourceName, StringRef targetName,
    llvm::ArrayRef<wave::WaveSymbolAttr> newDims, llvm::raw_ostream &errs) {
  if (!source || !source.getFullySpecified())
    return ChangeResult::NoChange;

  SmallVector<wave::WaveSymbolAttr> resultShape(source.getShape());
  llvm::append_range(resultShape, newDims);
  llvm::FailureOr<ChangeResult> res = ::checkPropagateShapeConflict(
      resultShape, target, sourceName, targetName, errs);
  if (failed(res) || *res == ChangeResult::NoChange)
    return res;
  target = target.copyShapeFrom(resultShape);
  return ChangeResult::Change;
}

llvm::FailureOr<ChangeResult> wave::detail::propagateReductionTypesForward(
    wave::WaveSymbolAttr axis, unsigned initOperandNum,
    unsigned inputOperandNum, llvm::ArrayRef<wave::WaveTensorType> operandTypes,
    llvm::MutableArrayRef<wave::WaveTensorType> resultTypes,
    llvm::raw_ostream &errs) {
  // If init is present, we can propagate from it directly,
  // otherwise propagate from input after removing the axis.
  FailureOr<ChangeResult> maybeChangeResult =
      wave::detail::propagateShapeInformation(
          operandTypes[initOperandNum], resultTypes[0], "init", "result", errs);
  if (failed(maybeChangeResult))
    return failure();

  wave::WaveTensorType inputType = operandTypes[inputOperandNum];
  maybeChangeResult =
      maybeChangeResult | propagateFromReductionInput(
                              inputType, axis, resultTypes[0], "result", errs);
  maybeChangeResult = maybeChangeResult | propagateShapeDropTrailingDims(
                                              inputType, resultTypes[0],
                                              "input", "result", 1, errs);
  return maybeChangeResult;
}

llvm::FailureOr<ChangeResult> wave::detail::propagateReductionTypesBackward(
    wave::WaveSymbolAttr axis, unsigned initOperandNum,
    unsigned inputOperandNum,
    llvm::MutableArrayRef<wave::WaveTensorType> operandTypes,
    llvm::ArrayRef<wave::WaveTensorType> resultTypes, llvm::raw_ostream &errs) {
  FailureOr<ChangeResult> maybeChangeResult =
      wave::detail::propagateShapeInformation(
          resultTypes[0], operandTypes[initOperandNum], "result", "init", errs);
  if (failed(maybeChangeResult))
    return failure();

  // Propagate "sideways" from input to init operand.
  wave::WaveTensorType inputType = operandTypes[inputOperandNum];
  maybeChangeResult =
      maybeChangeResult |
      propagateFromReductionInput(inputType, axis, operandTypes[initOperandNum],
                                  "init", errs);

  // Since we only reduce trailing dimensions, we can infer the operand shape by
  // adding the reduction axis back to the result.
  maybeChangeResult =
      maybeChangeResult | propagateShapeAddTrailingDims(
                              resultTypes[0], operandTypes[inputOperandNum],
                              "result", "input", {axis}, errs);

  return maybeChangeResult;
}

bool wave::detail::isReductionTypeInferenceComplete(Value input, Value init,
                                                    Value result) {
  return llvm::all_of(
      llvm::ArrayRef<Value>{input, init, result}, [&](Value value) {
        return llvm::cast<WaveTensorType>(value.getType()).getFullySpecified();
      });
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

FailureOr<ChangeResult>
wave::detail::propagateReductionElementsPerThreadForward(
    wave::WaveSymbolAttr axis,
    llvm::ArrayRef<ElementsPerThreadLatticeValue> operandElements,
    llvm::MutableArrayRef<ElementsPerThreadLatticeValue> resultElements,
    llvm::raw_ostream &errs, const wave::ElementsPerThreadInit &init) {
  if (init.threadXDimension == axis) {
    // Reducing along the thread X, so mapped to lanes, means we will have one
    // element per thread.
    // TODO: not sure about that, it feels more like one element in general, not
    // per thread.
    wave::ElementsPerThreadLatticeValue expectedResult(1);
    return wave::detail::checkAndPropagateElementsPerThreadFromConstant(
        expectedResult, {}, resultElements,
        "reduction along thread X dimension", "", "result", errs);
  }
  return wave::detail::identityElementsPerThreadPropagate(
      operandElements, resultElements, "operands", "results", errs);
}

FailureOr<ChangeResult>
wave::detail::propagateReductionElementsPerThreadBackward(
    wave::WaveSymbolAttr axis, unsigned int initOperandNum,
    llvm::MutableArrayRef<ElementsPerThreadLatticeValue> operandElements,
    llvm::ArrayRef<ElementsPerThreadLatticeValue> resultElements,
    llvm::raw_ostream &errs, const wave::ElementsPerThreadInit &init) {
  if (init.threadXDimension == axis) {
    // Reducing along the thread X, so mapped to lanes, means we will have one
    // element per thread.
    // TODO: same as above.
    wave::ElementsPerThreadLatticeValue expectedOperand(1);
    return wave::detail::checkAndPropagateElementsPerThreadFromConstant(
        expectedOperand, {}, operandElements.slice(initOperandNum, 1),
        "reduction along thread X dimension", "", "operands", errs);

    // TODO: do we need to have elements per thread attribute here so we can set
    // it as lattice value for input?
  }
  return wave::detail::identityElementsPerThreadPropagate(
      resultElements, operandElements, "operands", "results", errs);
}

llvm::FailureOr<ChangeResult> wave::detail::identityElementsPerThreadPropagate(
    llvm::ArrayRef<ElementsPerThreadLatticeValue> from,
    llvm::MutableArrayRef<ElementsPerThreadLatticeValue> to,
    llvm::StringRef fromName, llvm::StringRef toName, llvm::raw_ostream &errs) {
  assert(!from.empty());
  assert(!to.empty());
  auto source = ElementsPerThreadLatticeValue::bottom();
  unsigned sourcePos = 0;
  for (auto &&[i, fromValue] : llvm::enumerate(from)) {
    if (fromValue.isBottom())
      continue;
    source = fromValue;
    sourcePos = i;
    break;
  }
  if (source.isBottom())
    return ChangeResult::NoChange;

  return checkAndPropagateElementsPerThreadFromConstant(
      source, from, to, (fromName + " #" + llvm::Twine(sourcePos)).str(),
      fromName, toName, errs);
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

llvm::LogicalResult
wave::detail::verifyElementTypesMatch(std::optional<Location> loc,
                                      llvm::StringRef lhsName, Type lhs,
                                      llvm::StringRef rhsName, Type rhs) {
  if (getElementType(lhs) == getElementType(rhs))
    return success();

  if (loc) {
    emitError(*loc) << "expected " << lhsName << " and " << rhsName
                    << " elemental types to match, got " << getElementType(lhs)
                    << ", " << getElementType(rhs);
  }
  return failure();
}

llvm::LogicalResult wave::detail::verifyTensorShapesCompatible(
    wave::WaveTensorType lhs, wave::WaveTensorType rhs,
    std::optional<Location> errorLocation, llvm::StringRef lhsName,
    llvm::StringRef rhsName) {
  if (lhs == rhs)
    return success();

  if (!lhs || !rhs || !lhs.getFullySpecified() || !rhs.getFullySpecified())
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

llvm::LogicalResult wave::detail::verifyTypesCompatible(
    Type lhs, Type rhs, bool includeAddressSpace, bool includeElementalType,
    std::optional<Location> errorLocation, llvm::StringRef lhsName,
    llvm::StringRef rhsName) {
  // Fast and cheap path.
  if (lhs == rhs)
    return success();

  if (errorLocation) {
    assert(!lhsName.empty() && !rhsName.empty() &&
           "expected names when location is provided");
  }

  if (includeElementalType) {
    if (failed(
            verifyElementTypesMatch(errorLocation, lhsName, lhs, rhsName, rhs)))
      return failure();
  }

  auto lhsTensor = llvm::dyn_cast<wave::WaveTensorType>(lhs);
  auto rhsTensor = llvm::dyn_cast<wave::WaveTensorType>(rhs);
  if (!lhsTensor || !rhsTensor)
    return success();

  if (includeAddressSpace) {
    if (lhsTensor.getAddressSpaceValue() != rhsTensor.getAddressSpaceValue() &&
        lhsTensor.getAddressSpaceValue() !=
            wave::WaveAddressSpace::Unspecified &&
        rhsTensor.getAddressSpaceValue() !=
            wave::WaveAddressSpace::Unspecified) {
      if (errorLocation) {
        emitError(*errorLocation) << "address space mismatch between "
                                  << lhsName << " and " << rhsName;
      }
      return failure();
    }
  }

  return verifyTensorShapesCompatible(lhsTensor, rhsTensor, errorLocation,
                                      lhsName, rhsName);
}

static llvm::LogicalResult
verifyTypeRange(Location loc, TypeRange range, Type referenceType,
                bool includeAddressSpace, bool includeElementalType,
                llvm::StringRef rangeDescriptionPrefix,
                llvm::StringRef referenceDescription) {
  llvm::SmallString<16> rangeDescription(rangeDescriptionPrefix);
  for (auto &&[i, type] : llvm::enumerate(range)) {
    rangeDescription.resize(rangeDescriptionPrefix.size());
    llvm::raw_svector_ostream os(rangeDescription);
    os << i;

    if (failed(wave::detail::verifyTypesCompatible(
            type, referenceType, includeAddressSpace, includeElementalType, loc,
            os.str(), referenceDescription))) {
      return llvm::failure();
    }
  }
  return llvm::success();
}

llvm::LogicalResult wave::detail::verifyCompatibleOperandsAndResultsOpTrait(
    Operation *op, bool includeAddressSpace, bool includeElementalType) {
  const llvm::StringLiteral kOperandNamePrefix = "operand #";
  const llvm::StringLiteral kResultNamePrefix = "result #";
  std::string referenceDescription;
  llvm::raw_string_ostream os(referenceDescription);
  Type referenceType;
  auto it =
      llvm::find_if(op->getOperandTypes(), llvm::IsaPred<wave::WaveTensorType>);
  auto it2 =
      llvm::find_if(op->getResultTypes(), llvm::IsaPred<wave::WaveTensorType>);
  if (it != op->getOperandTypes().end()) {
    referenceType = *it;
    os << kOperandNamePrefix
       << std::distance(op->getOperandTypes().begin(), it);
  } else if (it2 != op->getResultTypes().end()) {
    referenceType = *it2;
    os << kResultNamePrefix << std::distance(op->getResultTypes().begin(), it2);
  } else if (op->getNumOperands() > 0) {
    referenceType = op->getOperandTypes()[0];
    os << kOperandNamePrefix << 0;
  } else if (op->getNumResults() > 0) {
    referenceType = op->getResultTypes()[0];
    os << kResultNamePrefix << 0;
  } else {
    return llvm::success();
  }

  if (llvm::failed(verifyTypeRange(op->getLoc(), op->getOperandTypes(),
                                   referenceType, includeAddressSpace,
                                   includeElementalType, kOperandNamePrefix,
                                   os.str())))
    return llvm::failure();

  return verifyTypeRange(op->getLoc(), op->getResultTypes(), referenceType,
                         includeAddressSpace, includeElementalType,
                         kResultNamePrefix, os.str());
}

// ----------------------------------------------------------------------------
// Reduction operation traits
// ----------------------------------------------------------------------------

wave::WaveSymbolAttr
wave::detail::getReducedSymbol(Operation *op, wave::WaveSymbolAttr axisAttr,
                               Type inputType) {
  if (axisAttr)
    return axisAttr;

  auto tensor = dyn_cast<wave::WaveTensorType>(inputType);
  if (tensor && tensor.getFullySpecified()) {
    return tensor.getShape().back();
  }
  return {};
}

LogicalResult wave::detail::verifyReductionOperation(Operation *op,
                                                     Type inputTypeBase,
                                                     Type initTypeBase,
                                                     Type resultTypeBase,
                                                     Attribute axisAttr) {
  if (failed(wave::detail::verifyElementTypesMatch(
          op->getLoc(), "input", inputTypeBase, "init", initTypeBase))) {
    return failure();
  }
  if (failed(wave::detail::verifyTypesCompatible(
          initTypeBase, resultTypeBase, /*includeAddressSpace=*/true,
          /*includeElementalType=*/true, op->getLoc(), "init", "result"))) {
    return failure();
  }

  auto inputType = dyn_cast<WaveTensorType>(inputTypeBase);
  auto initType = dyn_cast<WaveTensorType>(initTypeBase);
  auto resultType = dyn_cast<WaveTensorType>(resultTypeBase);

  if (inputType && !inputType.getFullySpecified() && !axisAttr) {
    return op->emitOpError() << "expected axis attribute when input type is "
                             << "not fully specified";
  }

  if (inputType && inputType.getFullySpecified()) {
    if (axisAttr) {
      return op->emitOpError() << "did not expect axis attribute when input "
                                  "type is fully specified";
    }

    if (initType && initType.getFullySpecified()) {
      if (inputType.getRank() - 1 != initType.getRank()) {
        return op->emitOpError()
               << "init tensor rank (" << initType.getRank()
               << ") must be one less than input tensor rank ("
               << inputType.getRank() << ")";
      }
      auto leadingDims = llvm::to_vector(llvm::seq<int>(initType.getRank()));
      if (failed(wave::detail::verifyTypesMatchingDimensions(
              op->getLoc(), "init", initType, leadingDims, "input", inputType,
              leadingDims)))
        return failure();
    }

    if (resultType && resultType.getFullySpecified()) {
      if (inputType.getRank() - 1 != resultType.getRank()) {
        return op->emitOpError()
               << "result tensor rank (" << resultType.getRank()
               << ") must be one less than input tensor rank ("
               << inputType.getRank() << ")";
      }
      auto leadingDims = llvm::to_vector(llvm::seq<int>(resultType.getRank()));
      if (failed(wave::detail::verifyTypesMatchingDimensions(
              op->getLoc(), "input", inputType, leadingDims, "result",
              resultType, leadingDims)))
        return failure();
    }
  }

  if (initType && initType.getFullySpecified()) {
    if (axisAttr && llvm::is_contained(initType.getShape(), axisAttr)) {
      return op->emitOpError()
             << "init tensor shape must not contain the reduced axis";
    }
  }

  if (resultType && resultType.getFullySpecified()) {
    if (axisAttr && llvm::is_contained(resultType.getShape(), axisAttr)) {
      return op->emitOpError()
             << "result tensor shape must not contain the reduced axis";
    }
  }

  return success();
}

#include "water/Dialect/Wave/IR/WaveOpInterfaces.cpp.inc"
