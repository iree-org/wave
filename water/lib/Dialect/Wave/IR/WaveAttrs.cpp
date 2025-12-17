// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
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

constexpr llvm::StringLiteral kNullExpr = "NULL";

/// Helper function to parse an affine wave expression with the wave
/// symbol names passed in `names`.
static ParseResult parseExprWithNames(ArrayRef<StringRef> names,
                                      AffineExpr &outExpr, AsmParser &parser,
                                      bool allowNull = false) {
  MLIRContext *context = parser.getContext();
  if (allowNull && succeeded(parser.parseOptionalLess()) &&
      succeeded(parser.parseKeyword(kNullExpr)) &&
      succeeded(parser.parseGreater())) {
    outExpr = nullptr;
    return success();
  }

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
  if (!map)
    return "<" + kNullExpr.str() + ">";
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

// Populate `names` with the names of symbols to be used in the expression
// syntax. If new names are generated on-the-fly, store them in
// `owningSymbolNames`. It is the caller's responsibility to keep those names
// alive at least as long as it needs `names`, which only stores references.
LogicalResult
getExprSymbolNames(ArrayRef<Attribute> symbols,
                   SmallVectorImpl<StringRef> &names,
                   SmallVectorImpl<llvm::SmallString<16>> &owningSymbolNames,
                   function_ref<void(StringRef)> emitError) {
  names.reserve(names.size() + symbols.size());
  for (Attribute attr : symbols) {
    if (auto sym = dyn_cast<WaveSymbolAttr>(attr)) {
      names.push_back(sym.getName());
    } else if (auto sym = dyn_cast<WaveIndexSymbolAttr>(attr)) {
      names.push_back(stringifyWaveIndexSymbol(sym.getValue()));
    } else if (auto sym = dyn_cast<WaveIterSymbolAttr>(attr)) {
      llvm::raw_svector_ostream os(owningSymbolNames.emplace_back());
      os << "_Iter_" << sym.getName();
      names.push_back(os.str());
    } else {
      emitError("expected symbol names to be one of WaveSymbolAttr, "
                "WaveIndexSymbolAttr or WaveIterSymbolAttr");
      return failure();
    }
  }
  return success();
}

Attribute WaveIndexMappingAttr::parse(AsmParser &parser, Type type) {
  // Parse custom syntax: '[' symbol-names ']' '->' '(' start, step, stride ')'
  // This preserves meaningful symbol names while leveraging the existing
  // affine parser.

  SmallVector<Attribute> symbolNameAttrs;
  // Parse '[' symbol-names ']' allowing empty or non-empty lists.
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
        Attribute attr;
        if (failed(parser.parseAttribute(attr)))
          return llvm::failure();
        symbolNameAttrs.push_back(attr);
        return llvm::success();
      }))
    return {};

  // Parse affine expr triple: '->' '(' start_expr ',' step_expr ',' stride_expr
  // ')'
  if (parser.parseArrow() || parser.parseLParen())
    return {};

  MLIRContext *context = parser.getContext();
  SmallVector<StringRef> symbolNames;
  SmallVector<llvm::SmallString<16>> owningSymbolNames;
  if (failed(getExprSymbolNames(symbolNameAttrs, symbolNames, owningSymbolNames,
                                [&](StringRef message) {
                                  parser.emitError(parser.getCurrentLocation(),
                                                   message);
                                }))) {
    return {};
  }

  AffineExpr startExpr;
  AffineExpr stepExpr;
  AffineExpr strideExpr;
  if (failed(parseExprWithNames(symbolNames, startExpr, parser,
                                /*allowNull=*/true)) ||
      parser.parseComma() ||
      failed(parseExprWithNames(symbolNames, stepExpr, parser,
                                /*allowNull=*/true)) ||
      parser.parseComma() ||
      failed(parseExprWithNames(symbolNames, strideExpr, parser,
                                /*allowNull=*/true)) ||
      parser.parseRParen()) {
    parser.emitError(
        parser.getCurrentLocation(),
        "expected three affine expressions for '(start, step, stride)'");
    return {};
  }

  auto startMap = startExpr
                      ? AffineMap::get(
                            /*numDims=*/0, /*numSymbols=*/symbolNames.size(),
                            startExpr, context)
                      : AffineMap();
  auto stepMap = stepExpr
                     ? AffineMap::get(
                           /*numDims=*/0, /*numSymbols=*/symbolNames.size(),
                           stepExpr, context)
                     : AffineMap();
  auto strideMap = strideExpr
                       ? AffineMap::get(
                             /*numDims=*/0, /*numSymbols=*/symbolNames.size(),
                             strideExpr, context)
                       : AffineMap();

  return get(context, symbolNameAttrs, startMap, stepMap, strideMap);
}

void WaveIndexMappingAttr::print(AsmPrinter &printer) const {
  // Print '[' symbol-names '] -> (start, step, stride)'.
  // We keep one global symbol list (symbol_names) for all three expressions.
  // Each expression is an affine map with the same numSymbols; we substitute
  // s0, s1, ... using the shared names when rendering each expression.
  printer << "[";
  ArrayRef<Attribute> symbols = getSymbols();
  llvm::interleaveComma(symbols, printer, [&](mlir::Attribute attr) {
    printer.printAttribute(attr);
  });
  printer << "] -> ";

  SmallVector<StringRef> names;
  SmallVector<llvm::SmallString<16>> owningSymbolNames;
  if (failed(getExprSymbolNames(
          symbols, names, owningSymbolNames, [&](StringRef message) {
            // TODO: double-check that printer doesn't crash on malformed
            // attributes.
            llvm_unreachable("Unexpected symbol attribute type");
          }))) {
    return;
  }
  // All three maps share the same symbol set and order.
  std::string startStr = stringifyWithNames(getStart(), names);
  std::string stepStr = stringifyWithNames(getStep(), names);
  std::string strideStr = stringifyWithNames(getStride(), names);

  printer << "(" << startStr << ", " << stepStr << ", " << strideStr << ")";
}

LogicalResult
WaveIndexMappingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                             ArrayRef<Attribute> symbols, AffineMap start,
                             AffineMap step, AffineMap stride) {
  if (!llvm::all_of(symbols, llvm::IsaPred<WaveSymbolAttr, WaveIndexSymbolAttr,
                                           WaveIterSymbolAttr>)) {
    return emitError() << "expected all symbols to be a WaveSymbolAttr, "
                          "WaveIndexSymbolAttr or WaveIterSymbolAttr";
  }

  return success();
}

WaveIndexMappingAttr WaveIndexMappingAttr::removeInput(Attribute input) const {
  SmallVector<Attribute> symbols = llvm::to_vector(getSymbols());
  auto it = llvm::find(symbols, input);
  if (it == symbols.end())
    return *this;

  size_t position = std::distance(symbols.begin(), it);
  MLIRContext *ctx = getContext();
  DenseMap<AffineExpr, AffineExpr> replacement;
  replacement.try_emplace(getAffineSymbolExpr(position, ctx),
                          getAffineConstantExpr(0, ctx));
  for (size_t i = position + 1, e = symbols.size(); i < e; ++i) {
    replacement.try_emplace(getAffineSymbolExpr(i, ctx),
                            getAffineSymbolExpr(i - 1, ctx));
  }

  symbols.erase(it);
  return get(ctx, symbols,
             getStart() ? getStart().replace(replacement) : AffineMap(),
             getStep() ? getStep().replace(replacement) : AffineMap(),
             getStride() ? getStride().replace(replacement) : AffineMap());
}

SmallVector<unsigned> WaveIndexMappingAttr::getUsedSymbolPositions() const {
  llvm::SetVector<unsigned> usedSymbolPositions;
  auto collect = [&](AffineMap map) {
    map.walkExprs([&](AffineExpr expr) {
      if (auto symbolExpr = dyn_cast<AffineSymbolExpr>(expr))
        usedSymbolPositions.insert(symbolExpr.getPosition());
    });
  };
  if (getStart())
    collect(getStart());
  if (getStep())
    collect(getStep());
  if (getStride())
    collect(getStride());
  return usedSymbolPositions.takeVector();
}

WaveIndexMappingAttr WaveIndexMappingAttr::removeUnusedInputs() const {
  SmallVector<unsigned> usedSymbolPositions = getUsedSymbolPositions();
  llvm::sort(usedSymbolPositions);
  SmallVector<Attribute> newSymbols;
  DenseMap<AffineExpr, AffineExpr> replacement;
  for (auto [oldPosition, symbol] : llvm::enumerate(getSymbols())) {
    if (!llvm::is_contained(usedSymbolPositions, oldPosition))
      continue;

    replacement.try_emplace(
        getAffineSymbolExpr(oldPosition, getContext()),
        getAffineSymbolExpr(newSymbols.size(), getContext()));
    newSymbols.push_back(symbol);
  }
  assert(newSymbols.size() == usedSymbolPositions.size());
  AffineMap start = getStart() ? getStart().replace(replacement) : AffineMap();
  AffineMap step = getStep() ? getStep().replace(replacement) : AffineMap();
  AffineMap stride =
      getStride() ? getStride().replace(replacement) : AffineMap();
  return get(getContext(), newSymbols, start, step, stride);
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
// WaveSymbolAttr
//===----------------------------------------------------------------------===//

LogicalResult
WaveSymbolAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       StringRef name) {
  static const llvm::StringSet<> indexSymbols(llvm::from_range,
                                              WaveIndexSymbolAttr::AllCases);
  if (indexSymbols.contains(name)) {
    auto diag = emitError() << "illegal wave symbol name '" << name << "'";
    diag.attachNote() << "Did you mean to use WaveIndexSymbolAttr instead?";
    return diag;
  }
  if (name.starts_with("_"))
    return emitError()
           << "symbols names starting with '_' are reserved for internal use";
  return llvm::success();
}

//===----------------------------------------------------------------------===//
// WaveExprListAttr
//===----------------------------------------------------------------------===//

std::optional<llvm::SmallVector<int64_t>>
wave::WaveExprListAttr::getResolvedShape(
    wave::WaveHyperparameterAttr hyper) const {
  ArrayRef<Attribute> symbolAttrs(getSymbols().data(), getSymbols().size());
  return wave::evaluateMapWithHyperparams(getMap(), symbolAttrs, hyper);
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

  SmallVector<Attribute> symbolNameAttrs;
  SmallVector<StringRef> symbolNames;

  // Parse '[' symbol-names ']' allowing empty or non-empty lists.
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
        Attribute attr;
        if (failed(parser.parseAttribute(attr)))
          return llvm::failure();
        symbolNameAttrs.push_back(attr);
        return llvm::success();
      }))
    return {};

  // Parse: '->' '(' expr (',' expr)* ')'
  if (parser.parseArrow())
    return {};
  MLIRContext *context = parser.getContext();
  SmallVector<llvm::SmallString<16>> owningSymbolNames;
  if (failed(getExprSymbolNames(symbolNameAttrs, symbolNames, owningSymbolNames,
                                [&](StringRef message) {
                                  parser.emitError(parser.getCurrentLocation(),
                                                   message);
                                }))) {
    return {};
  }

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
  // Print symbol names like: [#wave.symbol<"M">, #wave.symbol<"K">] -> ( ... )
  printer << "<[";
  ArrayRef<Attribute> symbols = getSymbols();
  SmallVector<StringRef> names;
  llvm::interleaveComma(symbols, printer, [&](mlir::Attribute attr) {
    printer.printAttribute(attr);
  });
  printer << "] -> (";

  SmallVector<llvm::SmallString<16>> owningSymbolNames;
  if (failed(getExprSymbolNames(
          symbols, names, owningSymbolNames, [&](StringRef message) {
            // TODO: double-check that printer doesn't crash on malformed
            // attributes.
            llvm_unreachable("Unexpected symbol attribute type");
          }))) {
    return;
  }

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

LogicalResult
WaveExprListAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                         ArrayRef<Attribute> symbols, AffineMap map) {
  if (!llvm::all_of(symbols, llvm::IsaPred<WaveSymbolAttr, WaveIndexSymbolAttr,
                                           WaveIterSymbolAttr>))
    return emitError() << "expected all symbols to be a WaveSymbolAttr, "
                          "WaveIndexSymbolAttr or WaveIterSymbolAttr";

  return success();
}

//-----------------------------------------------------------------------------
// Constraint attributes
//-----------------------------------------------------------------------------

LogicalResult HardwareConstraintAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, unsigned threadsPerWave,
    ArrayRef<unsigned> wavesPerBlock, WaveMmaKindAttr mmaType,
    DictionaryAttr vectorShapes, unsigned maxBitsPerLoad) {

  if (!(wavesPerBlock.empty() || wavesPerBlock.size() == 3))
    return emitError() << "waves_per_block (" << wavesPerBlock
                       << ") should have 3 elements";

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
  if (!llvm::all_of(tile_size.getSymbols(),
                    llvm::IsaPred<wave::WaveSymbolAttr>)) {
    return emitError() << "tile_size must only contain WaveSymbolAttr";
  }
  return success();
}

LogicalResult
WaveConstraintAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           WaveSymbolAttr dim, WaveExprListAttr tile_size) {
  if (tile_size.getSize() != 1) {
    return emitError() << "invalid ExpressionList size, expected 1";
  }
  if (!llvm::all_of(tile_size.getSymbols(),
                    llvm::IsaPred<wave::WaveSymbolAttr>)) {
    return emitError() << "tile_size must only contain WaveSymbolAttr";
  }

  return success();
}

LogicalResult
TilingConstraintAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                             WaveSymbolAttr dim, WaveExprListAttr tile_size) {
  if (tile_size.getSize() != 1) {
    return emitError() << "invalid ExpressionList size, expected 1";
  }
  if (!llvm::all_of(tile_size.getSymbols(),
                    llvm::IsaPred<wave::WaveSymbolAttr>)) {
    return emitError() << "tile_size must only contain WaveSymbolAttr";
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
  if (!llvm::all_of(tile_size.getSymbols(),
                    llvm::IsaPred<wave::WaveSymbolAttr>)) {
    return emitError() << "tile_size must only contain WaveSymbolAttr";
  }
  return success();
}

void wave::WaveDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "water/Dialect/Wave/IR/WaveAttrs.cpp.inc"
      >();
}

//-----------------------------------------------------------------------------
// WaveMmaKindAttr
//-----------------------------------------------------------------------------

// MMA spec mapping — we adopt IREE’s 0xABCD encoding.
// AB encodes vendor/arch, C encodes a type-class (e.g. f16/bf16, fp8),
// D is a variant. This is not sufficient to derive M/N/K or exact operand
// types uniformly across families, so we use an explicit switch to return
// exact (M,N,K) and concrete element types. For fp8/packed encodings we
// default accumulator to f32, lhs/rhs to Float8E4M3FNUZ (CDNA3) or Float8E4M3FN
// (CDNA4).
// TODO: Extend WaveMmaKind to avoid conflating fp8/fp6/fp4 variants
// (E4M3FN/E5M2,etc.).

wave::WaveMmaSpec wave::WaveMmaKindAttr::getSpec(mlir::MLIRContext *ctx,
                                                 wave::WaveMmaKind kind) {
  auto f16 = [&]() -> mlir::Type { return mlir::Float16Type::get(ctx); };
  auto bf16 = [&]() -> mlir::Type { return mlir::BFloat16Type::get(ctx); };
  auto f32 = [&]() -> mlir::Type { return mlir::Float32Type::get(ctx); };
  auto i8 = [&]() -> mlir::Type { return mlir::IntegerType::get(ctx, 8); };
  auto i32 = [&]() -> mlir::Type { return mlir::IntegerType::get(ctx, 32); };
  auto f8E4M3FNUZ = [&]() -> mlir::Type {
    return mlir::Float8E4M3FNUZType::get(ctx);
  };
  auto f8E4M3FN = [&]() -> mlir::Type {
    return mlir::Float8E4M3FNType::get(ctx);
  };

  switch (kind) {
  // CDNA1
  case wave::WaveMmaKind::F32_16x16x16_F16:
    return {16, 16, 16, f16(), f16(), f32()};
  case wave::WaveMmaKind::F32_32x32x8_F16:
    return {32, 32, 8, f16(), f16(), f32()};
  case wave::WaveMmaKind::F32_16x16x32_K8_F16:
    return {16, 16, 32, f16(), f16(), f32()};
  case wave::WaveMmaKind::F32_32x32x16_K8_F16:
    return {32, 32, 16, f16(), f16(), f32()};
  case wave::WaveMmaKind::I32_16x16x16_I8:
    return {16, 16, 16, i8(), i8(), i32()};
  case wave::WaveMmaKind::I32_32x32x8_I8:
    return {32, 32, 8, i8(), i8(), i32()};

  // CDNA3
  case wave::WaveMmaKind::F32_16x16x32_F8:
    return {16, 16, 32, f8E4M3FNUZ(), f8E4M3FNUZ(), f32()};
  case wave::WaveMmaKind::F32_32x32x16_F8:
    return {32, 32, 16, f8E4M3FNUZ(), f8E4M3FNUZ(), f32()};
  case wave::WaveMmaKind::F32_16x16x32_K4_F8:
    return {16, 16, 32, f8E4M3FNUZ(), f8E4M3FNUZ(), f32()};
  case wave::WaveMmaKind::F32_32x32x16_K4_F8:
    return {32, 32, 16, f8E4M3FNUZ(), f8E4M3FNUZ(), f32()};
  case wave::WaveMmaKind::I32_16x16x32_I8:
    return {16, 16, 32, i8(), i8(), i32()};
  case wave::WaveMmaKind::I32_32x32x16_I8:
    return {32, 32, 16, i8(), i8(), i32()};

  // CDNA4
  case wave::WaveMmaKind::F32_16x16x128_F8F6F4:
    return {16, 16, 128, f8E4M3FN(), f8E4M3FN(), f32()};
  case wave::WaveMmaKind::F32_32x32x64_F8F6F4:
    return {32, 32, 64, f8E4M3FN(), f8E4M3FN(), f32()};
  case wave::WaveMmaKind::F32_16x16x32_F16:
    return {16, 16, 32, f16(), f16(), f32()};
  case wave::WaveMmaKind::F32_32x32x16_F16:
    return {32, 32, 16, f16(), f16(), f32()};
  case wave::WaveMmaKind::F32_16x16x32_BF16:
    return {16, 16, 32, bf16(), bf16(), f32()};
  case wave::WaveMmaKind::F32_32x32x16_BF16:
    return {32, 32, 16, bf16(), bf16(), f32()};
  }

  llvm_unreachable("unhandled WaveMmaKind in WaveMmaKindAttr::getSpec");
}

std::tuple<uint32_t, uint32_t, uint32_t>
wave::WaveMmaKindAttr::getShape(mlir::MLIRContext *ctx,
                                wave::WaveMmaKind kind) {
  wave::WaveMmaSpec spec = wave::WaveMmaKindAttr::getSpec(ctx, kind);
  return {static_cast<uint32_t>(spec.m), static_cast<uint32_t>(spec.n),
          static_cast<uint32_t>(spec.k)};
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
            if (emitDiagnostics) {
              op->emitError()
                  << "normal form requires index expressions to be "
                     "provided for all supported wave dialect operations";
            }
            return mlir::WalkResult::interrupt();
          }
        }

        return mlir::WalkResult::advance();
      });

  return llvm::failure(walkResult.wasInterrupted());
}
