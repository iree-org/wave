// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"
#include "llvm/ADT/STLExtras.h"

//-----------------------------------------------------------------------------
// Custom parsing and printing hooks. These must be defined before including the
// op classes.
//-----------------------------------------------------------------------------

// Parse types of the `wave.register` op and perform type inference. The syntax
// is simply the tensor type from which the elemental type is extract for the
// initializer type.
static mlir::ParseResult parseRegisterOpTypes(mlir::OpAsmParser &parser,
                                              mlir::Type &initType,
                                              mlir::Type &resultType) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (mlir::failed(parser.parseType(resultType)))
    return mlir::failure();

  auto tensorType = llvm::dyn_cast<wave::WaveTensorType>(resultType);
  if (!tensorType)
    return parser.emitError(loc)
           << "expected wave tensor type, got " << resultType;

  initType = tensorType.getElementType();
  return mlir::success();
}

// Print types of the `wave.register` operation.
static void printRegisterOpTypes(mlir::OpAsmPrinter &printer, mlir::Operation *,
                                 mlir::Type initType, mlir::Type resultType) {
  assert(initType ==
             llvm::cast<wave::WaveTensorType>(resultType).getElementType() &&
         "expected equal types");
  (void)initType;
  printer.printType(resultType);
}

// Parse an @-symbol and interpret it as a wave symbol.
static mlir::ParseResult parseSingleSymbol(mlir::OpAsmParser &parser,
                                           wave::WaveSymbolAttr &symbolAttr) {
  mlir::StringAttr strAttr;
  if (mlir::failed(parser.parseSymbolName(strAttr)))
    return mlir::failure();

  symbolAttr =
      wave::WaveSymbolAttr::get(parser.getContext(), strAttr.getValue());
  return mlir::success();
}

// Print a wave symbol as an MLIR @-symbol.
static void printSingleSymbol(mlir::OpAsmPrinter &printer, mlir::Operation *,
                              wave::WaveSymbolAttr symbolAttr) {
  printer.printSymbolName(symbolAttr.getName());
}

using namespace mlir;
using namespace wave;

#define GET_OP_CLASSES
#include "water/Dialect/Wave/IR/WaveOps.cpp.inc"

//-----------------------------------------------------------------------------
// AllocateOp
//-----------------------------------------------------------------------------

llvm::LogicalResult wave::AllocateOp::verify() {
  bool hasParent = getParent() != Value();
  bool hasOffset = getOffset() != std::nullopt;
  if (hasParent ^ hasOffset) {
    return emitOpError()
           << "expects parent and offset to be present simultaneously";
  }

  return llvm::success();
}

//-----------------------------------------------------------------------------
// IterateOp
//-----------------------------------------------------------------------------

bool wave::IterateOp::areTypesCompatible(mlir::Type lhs, mlir::Type rhs) {
  return detail::verifyTypesCompatible(llvm::cast<wave::WaveTensorType>(lhs),
                                       llvm::cast<wave::WaveTensorType>(rhs),
                                       /*includeAddressSpace=*/true)
      .succeeded();
}

mlir::OperandRange
wave::IterateOp::getEntrySuccessorOperands(mlir::RegionBranchPoint point) {
  return getIterArgs();
}

void wave::IterateOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions) {
  // May branch into the region or bypass it regardless of the source.
  regions.emplace_back(mlir::RegionSuccessor(getResults()));
  regions.emplace_back(
      mlir::RegionSuccessor(&getBody(), getBody().front().getArguments()));
}

mlir::LogicalResult wave::IterateOp::verify() {
  mlir::TypeRange iterArgTypes = getIterArgs().getTypes();
  mlir::TypeRange resultTypes = getResultTypes();
  if (iterArgTypes.size() != resultTypes.size()) {
    return emitOpError() << "expects the same number of iter_args ("
                         << iterArgTypes.size() << ") and results ("
                         << resultTypes.size() << ")";
  }
  for (auto &&[i, iterArg, result] :
       llvm::enumerate(iterArgTypes, resultTypes)) {
    auto iterArgTensor = llvm::cast<wave::WaveTensorType>(iterArg);
    auto resultTensor = llvm::cast<wave::WaveTensorType>(result);
    if (!iterArgTensor.getFullySpecified() || !resultTensor.getFullySpecified())
      continue;

    auto allDims =
        llvm::to_vector(llvm::iota_range<int>(0, iterArgTensor.getRank(),
                                              /*Inclusive=*/false));
    auto istr = std::to_string(i);
    if (mlir::failed(detail::verifyTypesMatchingDimensions(
            getLoc(), "iter_args #" + istr, iterArgTensor, allDims,
            "result #" + istr, resultTensor, allDims)))
      return mlir::failure();
  }

  return mlir::success();
}

//-----------------------------------------------------------------------------
// MmaOp
//-----------------------------------------------------------------------------

llvm::FailureOr<mlir::ChangeResult> wave::MmaOp::propagateForward(
    llvm::ArrayRef<wave::WaveTensorType> operandTypes,
    llvm::MutableArrayRef<wave::WaveTensorType> resultTypes,
    llvm::raw_ostream &errs) {
  // TODO: probably upstream, we want a value-less adaptor or some way of
  // querying positions of named arguments instead of magic values here. It is
  // currently possible by doing get<OperandName>Mutable().getOperandNumber(),
  // but a bit annoying.
  return detail::propagateShapeInformation(operandTypes[2], resultTypes[0],
                                           "accumulator", "result", errs);
}

llvm::FailureOr<mlir::ChangeResult> wave::MmaOp::propagateBackward(
    llvm::MutableArrayRef<wave::WaveTensorType> operandTypes,
    llvm::ArrayRef<wave::WaveTensorType> resultTypes, llvm::raw_ostream &errs) {
  // TODO: we may consider doing partial type propagation, but we can't infer
  // the reduction dimension from the result type since it is absent from it.
  return detail::propagateShapeInformation(resultTypes[0], operandTypes[2],
                                           "result", "accumulator", errs);
}

// Check if the given type is one of the allowed types provided as template
// arguments and report an error at the given location otherwise.
template <typename... AllowedTypes>
static mlir::LogicalResult
checkAllowedTypes(mlir::Location loc, mlir::Type type, llvm::StringRef name,
                  wave::WaveMmaKind kind) {
  if (llvm::isa<AllowedTypes...>(type))
    return mlir::success();

  return mlir::emitError(loc)
         << "unexpected " << name << " elemental type " << type
         << " for MMA kind " << wave::stringifyEnum(kind);
}

// Check if the given type is an integer type with one of the provided bitwidths
// and report and error at the given location otherwise.
template <typename T>
static std::enable_if_t<std::is_same_v<T, mlir::IntegerType>,
                        mlir::LogicalResult>
checkAllowedTypes(mlir::Location loc, mlir::Type type, llvm::StringRef name,
                  wave::WaveMmaKind kind, llvm::ArrayRef<unsigned> bitwidths) {
  if (auto intType = llvm::dyn_cast<mlir::IntegerType>(type)) {
    if (llvm::is_contained(bitwidths, intType.getIntOrFloatBitWidth()))
      return mlir::success();
  }
  return mlir::emitError(loc)
         << "unexpected " << name << " elemental type " << type
         << " for MMA kind " << wave::stringifyEnum(kind);
}

// Check if the types used for multiplication and accumulation in a `wave.mma`
// operation are compatible with the specified MMA kind and report an error
// otherwise.
static mlir::LogicalResult checkMmaTypeCompatibility(mlir::Location loc,
                                                     wave::WaveMmaKind kind,
                                                     mlir::Type mulType,
                                                     mlir::Type accType) {
  bool success = false;
  switch (kind) {
  case wave::WaveMmaKind::F32_16x16x16_F16:
  case wave::WaveMmaKind::F32_32x32x8_F16:
  case wave::WaveMmaKind::F32_16x16x32_K8_F16:
  case wave::WaveMmaKind::F32_32x32x16_F16:
  case wave::WaveMmaKind::F32_16x16x32_F16:
  case wave::WaveMmaKind::F32_32x32x16_K8_F16:
  case wave::WaveMmaKind::F32_32x32x16_BF16:
  case wave::WaveMmaKind::F32_16x16x32_BF16: {
    success = mlir::succeeded(checkAllowedTypes<mlir::Float32Type>(
                  loc, accType, "accumulator/result", kind)) &&
              mlir::succeeded(
                  checkAllowedTypes<mlir::Float16Type, mlir::BFloat16Type>(
                      loc, mulType, "lhs/rhs", kind));
    break;
  }

  case wave::WaveMmaKind::I32_16x16x16_I8:
  case wave::WaveMmaKind::I32_32x32x8_I8:
  case wave::WaveMmaKind::I32_16x16x32_I8:
  case wave::WaveMmaKind::I32_32x32x16_I8:
    success = mlir::succeeded(checkAllowedTypes<mlir::IntegerType>(
                  loc, accType, "accumulator/result", kind, {32})) &&
              mlir::succeeded(checkAllowedTypes<mlir::IntegerType>(
                  loc, mulType, "lhs/rhs", kind, {8}));
    break;

  case wave::WaveMmaKind::F32_16x16x32_F8:
  case wave::WaveMmaKind::F32_32x32x16_F8:
  case wave::WaveMmaKind::F32_16x16x32_K4_F8:
  case wave::WaveMmaKind::F32_32x32x16_K4_F8:
    success = mlir::succeeded(checkAllowedTypes<mlir::Float32Type>(
                  loc, accType, "accumulator/result", kind)) &&
              mlir::succeeded(
                  checkAllowedTypes<mlir::Float8E3M4Type, mlir::Float8E5M2Type>(
                      loc, mulType, "lhs/rhs", kind));
    break;

  case wave::WaveMmaKind::F32_16x16x128_F8F6F4:
  case wave::WaveMmaKind::F32_32x32x64_F8F6F4:
    success =
        mlir::succeeded(checkAllowedTypes<mlir::Float32Type>(
            loc, accType, "accumulator/result", kind)) &&
        mlir::succeeded(
            checkAllowedTypes<mlir::Float8E3M4Type, mlir::Float8E5M2Type,
                              mlir::Float6E2M3FNType, mlir::Float6E3M2FNType,
                              mlir::Float4E2M1FNType>(loc, mulType, "lhs/rhs",
                                                      kind));
    break;
  }

  return mlir::success(success);
}

//===----------------------------------------------------------------------===//
// MmaOp
//===----------------------------------------------------------------------===//

LogicalResult MmaOp::verify() {
  WaveTensorType lhsType = getLhs().getType();
  WaveTensorType rhsType = getRhs().getType();
  WaveTensorType accumulatorType = getAccumulator().getType();
  WaveTensorType resultType = getResult().getType();

  if (failed(detail::verifyElementTypesMatch(getLoc(), "LHS", lhsType, "RHS",
                                             rhsType)) ||
      failed(detail::verifyElementTypesMatch(getLoc(), "result", resultType,
                                             "accumulator", accumulatorType)))
    return mlir::failure();

  if (detail::verifyTypesMatchingDimensions(getLoc(), "LHS", lhsType, {1},
                                            "RHS", rhsType, {1})
          .failed() ||
      detail::verifyTypesMatchingDimensions(getLoc(), "LHS", lhsType, {0},
                                            "accumulator", accumulatorType, {0})
          .failed() ||
      detail::verifyTypesMatchingDimensions(getLoc(), "RHS", rhsType, {0},
                                            "accumulator", accumulatorType, {1})
          .failed()) {
    return mlir::failure();
  }

  return checkMmaTypeCompatibility(getLoc(), getKind(),
                                   lhsType.getElementType(),
                                   accumulatorType.getElementType());
}

//-----------------------------------------------------------------------------
// RegisterOp
//-----------------------------------------------------------------------------

mlir::LogicalResult wave::RegisterOp::verify() {
  WaveTensorType tensorType = getResult().getType();
  mlir::Type initType = getInit().getType();
  if (tensorType.getElementType() != initType) {
    return emitOpError() << "expected the type of the init value to match the "
                            "elemental type of the tensor";
  }
  if (!tensorType.getFullySpecified()) {
    return emitOpError() << "expected fully-specified tensor type";
  }
  return mlir::success();
}

//-----------------------------------------------------------------------------
// YieldOp
//-----------------------------------------------------------------------------

mlir::MutableOperandRange
wave::YieldOp::getMutableSuccessorOperands(mlir::RegionBranchPoint) {
  return getValuesMutable();
}
