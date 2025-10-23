// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "water/Dialect/Wave/IR/WaveEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "water/Dialect/Wave/IR/WaveAttrs.cpp.inc"

using namespace mlir;
using namespace wave;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Helper function to parse a `WaveSymbolAttr`, keeping track of both the
/// parse WaveSymbolAttr and the symbol name string.
static ParseResult parseSymbol(SmallVectorImpl<WaveSymbolAttr> &symbolNameAttrs,
                               SmallVectorImpl<StringRef> &symbolNames,
                               AsmParser &parser) {
  MLIRContext *context = parser.getContext();
  StringRef symbolName;
  if (failed(parser.parseKeyword(&symbolName)))
    return failure();
  symbolNameAttrs.push_back(WaveSymbolAttr::get(context, symbolName));
  symbolNames.push_back(symbolName);
  return success();
};

/// Helper function to parse an affine wave expression with the wave
/// symbol names passed in `names`.
static ParseResult parseExprWithNames(ArrayRef<StringRef> names,
                                      AffineExpr &outExpr, AsmParser &parser) {
  MLIRContext *context = parser.getContext();
  SmallVector<std::pair<StringRef, AffineExpr>> symbolSet;
  symbolSet.reserve(names.size());
  for (auto [i, nm] : llvm::enumerate(names))
    symbolSet.emplace_back(nm, getAffineSymbolExpr(i, context));
  return parser.parseAffineExpr(symbolSet, outExpr);
};

//===----------------------------------------------------------------------===//
// WaveIndexMappingAttr
//===----------------------------------------------------------------------===//

// Helper function to render an affine map result to a string.
// It textually substitutes 's<i>' occurrences with the corresponding names from
// the provided `names` array.
std::string stringifyWithNames(AffineMap map, ArrayRef<StringRef> names) {
  AffineExpr expr = map.getResult(0);
  std::string exprStr;
  llvm::raw_string_ostream os(exprStr);
  expr.print(os);
  os.flush();
  for (auto [i, nm] : llvm::enumerate(names)) {
    std::string pattern = "s" + std::to_string(i);
    size_t pos = 0;
    while ((pos = exprStr.find(pattern, pos)) != std::string::npos) {
      // Replace only when 'pattern' is a complete token (not embedded
      // inside a longer identifier or number). We approximate token
      // boundaries by checking that adjacent characters are non-alphanumeric.
      bool isWhole = (pos == 0 || !std::isalnum(exprStr[pos - 1])) &&
                     (pos + pattern.length() == exprStr.length() ||
                      !std::isalnum(exprStr[pos + pattern.length()]));
      if (isWhole) {
        exprStr.replace(pos, pattern.length(), nm.str());
        pos += nm.size();
      } else {
        pos += pattern.length();
      }
    }
  }
  return exprStr;
}

Attribute WaveIndexMappingAttr::parse(AsmParser &parser, Type type) {
  // Parse custom syntax: '[' symbol-names ']' '->' '(' start, step, stride ')'
  // This preserves meaningful symbol names while leveraging the existing
  // affine parser.

  SmallVector<WaveSymbolAttr> symbolNameAttrs;
  SmallVector<StringRef> symbolNames;

  // Parse '[' symbol-names ']' allowing empty or non-empty lists.
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
        return parseSymbol(symbolNameAttrs, symbolNames, parser);
      }))
    return {};

  // Parse affine expr triple: '->' '(' start_expr ',' step_expr ',' stride_expr
  // ')'
  if (parser.parseArrow() || parser.parseLParen())
    return {};

  MLIRContext *context = parser.getContext();
  AffineExpr startExpr;
  AffineExpr stepExpr;
  AffineExpr strideExpr;
  if (failed(parseExprWithNames(symbolNames, startExpr, parser)) ||
      parser.parseComma() ||
      failed(parseExprWithNames(symbolNames, stepExpr, parser)) ||
      parser.parseComma() ||
      failed(parseExprWithNames(symbolNames, strideExpr, parser)) ||
      parser.parseRParen()) {
    parser.emitError(
        parser.getCurrentLocation(),
        "expected three affine expressions for '(start, step, stride)'");
    return {};
  }

  // Build maps
  auto startMap = AffineMap::get(
      /*numDims=*/0, /*numSymbols=*/symbolNames.size(), startExpr, context);
  auto stepMap = AffineMap::get(
      /*numDims=*/0, /*numSymbols=*/symbolNames.size(), stepExpr, context);
  auto strideMap = AffineMap::get(
      /*numDims=*/0, /*numSymbols=*/symbolNames.size(), strideExpr, context);

  return get(context, symbolNameAttrs, startMap, stepMap, strideMap);
}

void WaveIndexMappingAttr::print(AsmPrinter &printer) const {
  // Print '[' symbol-names '] -> (start, step, stride)'.
  // We keep one global symbol list (symbol_names) for all three expressions.
  // Each expression is an affine map with the same numSymbols; we substitute
  // s0, s1, ... using the shared names when rendering each expression.
  printer << "[";
  llvm::interleaveComma(getSymbolNames(), printer,
                        [&](WaveSymbolAttr s) { printer << s.getName(); });
  printer << "] -> ";

  SmallVector<StringRef> allNames = getAllSymbolNames();
  // All three maps share the same symbol set and order.
  std::string startStr = stringifyWithNames(getStart(), allNames);
  std::string stepStr = stringifyWithNames(getStep(), allNames);
  std::string strideStr = stringifyWithNames(getStride(), allNames);

  printer << "(" << startStr << ", " << stepStr << ", " << strideStr << ")";
}

//===----------------------------------------------------------------------===//
// WaveHyperparameterAttr
//===----------------------------------------------------------------------===//

std::optional<int64_t>
WaveHyperparameterAttr::getSymbolValue(StringRef symbolName) const {
  DictionaryAttr mapping = getMapping();
  Attribute attr = mapping.get(symbolName);
  if (!attr)
    return std::nullopt;

  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getInt();

  return std::nullopt;
}

bool WaveHyperparameterAttr::hasSymbol(StringRef symbolName) const {
  return getMapping().get(symbolName) != nullptr;
}

//===----------------------------------------------------------------------===//
// WaveExprListAttr
//===----------------------------------------------------------------------===//

std::optional<llvm::SmallVector<int64_t>>
wave::WaveExprListAttr::getResolvedShape(
    wave::WaveHyperparameterAttr hyper) const {
  return wave::evaluateMapWithHyperparams(getMap(), getSymbols(), hyper);
}

llvm::LogicalResult
wave::verifyExprAttrsSameRank(llvm::ArrayRef<wave::WaveExprListAttr> exprs) {
  if (exprs.size() < 2)
    return mlir::success();

  unsigned expectedRank = exprs[0].getRank();

  for (size_t i = 1; i < exprs.size(); ++i) {
    if (exprs[i].getRank() != expectedRank)
      return mlir::failure();
  }

  return mlir::success();
}

llvm::LogicalResult
wave::verifyExprAttrsNoSymbols(llvm::ArrayRef<wave::WaveExprListAttr> exprs) {
  for (const auto &expr : exprs) {
    if (expr.getNumSymbols() != 0)
      return mlir::failure();
  }

  return mlir::success();
}

Attribute WaveExprListAttr::parse(AsmParser &parser, Type) {
  if (parser.parseLess())
    return {};

  SmallVector<WaveSymbolAttr> symbolNameAttrs;
  SmallVector<StringRef> symbolNames;

  // Parse '[' symbol-names ']' allowing empty or non-empty lists.
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
        return parseSymbol(symbolNameAttrs, symbolNames, parser);
      }))
    return {};

  // Parse: '->' '(' expr (',' expr)* ')'
  if (parser.parseArrow())
    return {};
  MLIRContext *context = parser.getContext();

  SmallVector<AffineExpr> results;
  auto parseOneExpr = [&]() -> ParseResult {
    AffineExpr e;
    if (failed(parseExprWithNames(symbolNames, e, parser)))
      return failure();
    results.push_back(e);
    return success();
  };
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, parseOneExpr))
    return {};

  if (results.empty()) {
    parser.emitError(
        parser.getCurrentLocation(),
        "wave expression attribute must have at least one dimension");
    return {};
  }
  if (parser.parseGreater())
    return {};

  // Build a single map with all result expressions.
  auto shape = AffineMap::get(/*numDims=*/0, /*numSymbols=*/symbolNames.size(),
                              results, context);

  return get(parser.getContext(), symbolNameAttrs, shape);
}

void WaveExprListAttr::print(mlir::AsmPrinter &printer) const {
  // Print symbol names like: [M, K] -> ( ... )
  printer << "<[";
  llvm::SmallVector<llvm::StringRef> names;
  names.reserve(getSymbols().size());
  llvm::interleaveComma(getSymbols(), printer.getStream(),
                        [&](wave::WaveSymbolAttr s) {
                          names.push_back(s.getName());
                          printer << s.getName();
                        });
  printer << "] -> (";

  // We have one map with N results. For each result expr, make a 1-result map
  // so we can reuse the identical stringifyWithNames(map,names) helper.
  mlir::AffineMap full = getMap(); // your stored map
  for (unsigned i = 0, e = full.getNumResults(); i < e; ++i) {
    if (i)
      printer << ", ";
    mlir::AffineMap one =
        mlir::AffineMap::get(/*numDims=*/0,
                             /*numSymbols=*/full.getNumSymbols(),
                             /*results=*/full.getResult(i), full.getContext());
    printer << stringifyWithNames(one, names);
  }
  printer << ")>";
}

//-----------------------------------------------------------------------------
// Constraint attributes
//-----------------------------------------------------------------------------

LogicalResult HardwareConstraintAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, unsigned threadsPerWave,
    ArrayRef<unsigned> wavesPerBlock, WaveMmaKindAttr mmaType,
    DictionaryAttr vectorShapes, unsigned maxBitsPerLoad) {

  if (vectorShapes && wavesPerBlock.size() != vectorShapes.size())
    return emitError() << "waves_per_block (" << wavesPerBlock
                       << ") does should have the same size as vector_shapes ("
                       << vectorShapes << ")";

  if (vectorShapes) {
    for (NamedAttribute attr : vectorShapes) {
      // TODO: verify that attr.getName() is a valid WaveSymbol
      Attribute value = attr.getValue();

      if (!isa<IntegerAttr>(value))
        return emitError() << attr.getName()
                           << " is not an IntegerAttr: " << attr.getValue();
    }
  }

  return success();
}

LogicalResult
WorkgroupConstraintAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                WaveSymbolAttr dim, WaveExprListAttr tile_size,
                                WaveWorkgroupDimAttr workgroup_dim,
                                bool primary) {
  if (tile_size.getSize() != 1) {
    return emitError() << "invalid ExpressionList size, expected 1";
  }
  return success();
}

LogicalResult
WaveConstraintAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           WaveSymbolAttr dim, WaveExprListAttr tile_size,
                           WorkgroupConstraintAttr wg_constraint) {
  if (wg_constraint && wg_constraint.getDim() != dim)
    return emitError()
           << "the dimension of the workgroup constraint in wg_constraint: "
           << wg_constraint.getDim()
           << " should match the dimension of the wave "
              "constraint: "
           << dim;

  if (tile_size.getSize() != 1) {
    return emitError() << "invalid ExpressionList size, expected 1";
  }

  return success();
}

LogicalResult
TilingConstraintAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                             WaveSymbolAttr dim, WaveExprListAttr tile_size) {
  if (tile_size.getSize() != 1) {
    return emitError() << "invalid ExpressionList size, expected 1";
  }
  return success();
}

LogicalResult
DeviceConstraintAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                             WaveSymbolAttr dim, WaveExprListAttr tile_size,
                             unsigned int device_dim) {
  if (tile_size.getSize() != 1) {
    return emitError() << "invalid ExpressionList size, expected 1";
  }
  return success();
}

void wave::WaveDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "water/Dialect/Wave/IR/WaveAttrs.cpp.inc"
      >();
}

// Verify that wave tensor types in the given range are fully specified. Emit a
// diagnostic with the given message at the location provided, if present,
// otherwise just return failure.
static llvm::LogicalResult
verifyTypesFullySpecified(std::optional<mlir::Location> loc,
                          mlir::TypeRange types, llvm::StringRef message) {
  for (mlir::Type type : types) {
    auto tensorType = llvm::dyn_cast<wave::WaveTensorType>(type);
    if (!tensorType || tensorType.getFullySpecified())
      continue;

    if (loc)
      mlir::emitError(*loc) << message;
    return llvm::failure();
  }
  return llvm::success();
}

// Verify that wave tensor types in the given range have memory-related address
// spaces, i.e., no unspecified and no register space.
static llvm::LogicalResult
verifyMemoryOnlyAddressSpaces(std::optional<mlir::Location> loc,
                              mlir::TypeRange types, llvm::StringRef message) {
  for (mlir::Type type : types) {
    auto tensorType = llvm::dyn_cast<wave::WaveTensorType>(type);
    if (!tensorType || llvm::is_contained({wave::WaveAddressSpace::Global,
                                           wave::WaveAddressSpace::Shared},
                                          tensorType.getAddressSpaceValue()))
      continue;

    if (loc)
      mlir::emitError(*loc) << message;
    return llvm::failure();
  }
  return llvm::success();
}

// Type of a callback visiting and verifying types of operation operands,
// results and block argument lists. The callback is expected to emit a
// diagnostic with the given message if the location is provided, and return
// failure in any case if verification fails.
using OpTypeRangeVisitor = llvm::function_ref<llvm::LogicalResult(
    std::optional<mlir::Location>, mlir::TypeRange, llvm::StringRef)>;

// Call `visitor` to verify type ranges of operation operands, results and block
// argument lists in immediately nested regions.
static llvm::LogicalResult visitOpRelatedTypes(mlir::Operation *op,
                                               OpTypeRangeVisitor visitor,
                                               llvm::StringRef message,
                                               bool emitDiagnostics) {
  std::optional<mlir::Location> optionalLoc =
      emitDiagnostics ? std::optional(op->getLoc()) : std::nullopt;
  if (llvm::failed(visitor(optionalLoc, op->getOperandTypes(), message)))
    return llvm::failure();
  if (llvm::failed(visitor(optionalLoc, op->getResultTypes(), message)))
    return llvm::failure();
  for (mlir::Region &region : op->getRegions()) {
    for (mlir::Block &block : region) {
      if (llvm::failed(visitor(optionalLoc, block.getArgumentTypes(), message)))
        return llvm::failure();
    }
  }
  return llvm::success();
}

llvm::LogicalResult wave::detail::verifyNormalFormAttr(
    mlir::Operation *root, wave::WaveNormalForm form, bool emitDiagnostics) {
  // No normal form required.
  if (form == wave::WaveNormalForm::None)
    return llvm::success();

  // Walk in pre-order so we hit functions sooner and verify them for the first
  // form.
  mlir::WalkResult walkResult =
      root->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
        std::optional<mlir::Location> optionalLoc;
        if (emitDiagnostics)
          optionalLoc = op->getLoc();

        if (auto func = llvm::dyn_cast<mlir::FunctionOpInterface>(op)) {
          if (wave::bitEnumContainsAll(
                  form, wave::WaveNormalForm::FunctionBoundarySpecified)) {
            const llvm::StringLiteral kMessage =
                "normal form requires tensor types to be fully specified at "
                "function boundaries";
            if (llvm::failed(verifyTypesFullySpecified(
                    optionalLoc, func.getArgumentTypes(), kMessage)))
              return mlir::WalkResult::interrupt();
            if (llvm::failed(verifyTypesFullySpecified(
                    optionalLoc, func->getResultTypes(), kMessage)))
              return mlir::WalkResult::interrupt();
          }
        }

        if (wave::bitEnumContainsAll(form,
                                     wave::WaveNormalForm::OpTypesSpecified)) {
          if (llvm::failed(visitOpRelatedTypes(
                  op, verifyTypesFullySpecified,
                  "normal form requires tensor types to be fully specified",
                  emitDiagnostics))) {
            return mlir::WalkResult::interrupt();
          }
        }

        if (wave::bitEnumContainsAll(form,
                                     wave::WaveNormalForm::MemoryOnlyTypes)) {
          if (llvm::failed(visitOpRelatedTypes(
                  op, verifyMemoryOnlyAddressSpaces,
                  "normal form requires tensor types to have only memory "
                  "address spaces (elements per thread propagation missing?)",
                  emitDiagnostics))) {
            return mlir::WalkResult::interrupt();
          }
        }

        if (wave::bitEnumContainsAll(
                form, wave::WaveNormalForm::IndexExprsSpecified)) {
          if (op->hasTrait<wave::HasWaveIndexMapping>() &&
              !op->getAttr(wave::WaveDialect::kIndexWaveExprListAttrName)) {
            op->emitError()
                << "normal form requires index expressions to be "
                   "provided for all supported wave dialect operations";
            return mlir::WalkResult::interrupt();
          }
        }

        return mlir::WalkResult::advance();
      });

  return llvm::failure(walkResult.wasInterrupted());
}
