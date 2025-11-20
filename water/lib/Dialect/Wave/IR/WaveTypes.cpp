// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveTypes.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

constexpr const static llvm::StringLiteral kwAny = "any";

static mlir::ParseResult
parseTensorShape(mlir::AsmParser &parser,
                 llvm::SmallVectorImpl<wave::WaveSymbolAttr> &attr,
                 bool &fullySpecified) {
  if (parser.parseOptionalKeyword(kwAny).succeeded()) {
    fullySpecified = false;
    return mlir::success();
  }

  fullySpecified = true;
  auto parseOne = [&]() -> mlir::ParseResult {
    mlir::StringAttr stringAttr;
    if (parser.parseSymbolName(stringAttr).failed())
      return mlir::failure();
    attr.emplace_back(wave::WaveSymbolAttr::get(stringAttr.getContext(),
                                                stringAttr.getValue()));
    return mlir::success();
  };

  return parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Square,
                                        parseOne);
}

static void printTensorShape(mlir::AsmPrinter &printer,
                             ::llvm::ArrayRef<::wave::WaveSymbolAttr> shape,
                             bool fullySpecified) {
  if (!fullySpecified) {
    printer.printKeywordOrString(kwAny);
    return;
  }

  auto printOne = [&](wave::WaveSymbolAttr attr) {
    attr.printAsSymbolRef(printer.getStream());
  };

  printer.getStream() << "[";
  llvm::interleaveComma(shape, printer.getStream(), printOne);
  printer.getStream() << "]";
}

#define GET_TYPEDEF_CLASSES
#include "water/Dialect/Wave/IR/WaveTypes.cpp.inc"

void wave::WaveDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "water/Dialect/Wave/IR/WaveTypes.cpp.inc"
      >();
}

mlir::LogicalResult wave::WaveTensorType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<wave::WaveSymbolAttr> shape, bool fullySpecified,
    mlir::Type elementType, wave::WaveAddressSpaceAttr addressSpace) {
  if (!fullySpecified && !shape.empty()) {
    return emitError() << "shape not expected for non-fully specified tensors";
  }

  if (!elementType.isIntOrIndexOrFloat()) {
    return emitError() << "expected element type to be integer, index or "
                          "floating point scalar";
  }
  return mlir::success();
}

std::optional<llvm::SmallVector<int64_t>>
wave::WaveTensorType::getResolvedShape(
    wave::WaveHyperparameterAttr hyper) const {
  if (!getFullySpecified()) {
    return {};
  }
  llvm::ArrayRef<mlir::Attribute> symbols(getShape().begin(), getShape().end());
  return wave::resolveSymbolNames(symbols, hyper);
}
