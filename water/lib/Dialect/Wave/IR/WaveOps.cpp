// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "water/Dialect/Wave/IR/IndexExpr.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace wave;

//-----------------------------------------------------------------------------
// Custom parsing and printing hooks. These must be defined before including the
// op classes.
//-----------------------------------------------------------------------------

// Parse types of the `wave.register` op and perform type inference. The syntax
// is simply the tensor type from which the elemental type is extract for the
// initializer type.
static ParseResult parseRegisterOpTypes(OpAsmParser &parser, Type &initType,
                                        Type &resultType) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (failed(parser.parseType(resultType)))
    return failure();

  initType =
      llvm::TypeSwitch<Type, Type>(resultType)
          .Case<wave::WaveTensorType, VectorType>(
              [](auto containerType) { return containerType.getElementType(); })
          .Default([](Type) { return nullptr; });

  if (!initType)
    return parser.emitError(loc)
           << "expected wave tensor or vector type, got " << resultType;

  return success();
}

// Print types of the `wave.register` operation.
static void printRegisterOpTypes(OpAsmPrinter &printer, Operation *,
                                 Type initType, Type resultType) {
#ifndef NDEBUG
  auto tensorType = llvm::dyn_cast<wave::WaveTensorType>(resultType);
  Type elementType = tensorType
                         ? tensorType.getElementType()
                         : llvm::cast<VectorType>(resultType).getElementType();
  assert(initType == elementType && "expected equal types");
#endif // NDEBUG
  (void)initType;
  printer.printType(resultType);
}

// Parse an @-symbol and interpret it as a wave symbol.
static ParseResult parseSingleSymbol(OpAsmParser &parser,
                                     wave::WaveSymbolAttr &symbolAttr) {
  StringAttr strAttr;
  if (failed(parser.parseSymbolName(strAttr)))
    return failure();

  symbolAttr =
      wave::WaveSymbolAttr::get(parser.getContext(), strAttr.getValue());
  return success();
}

// Print a wave symbol as an MLIR @-symbol.
static void printSingleSymbol(OpAsmPrinter &printer, Operation *,
                              wave::WaveSymbolAttr symbolAttr) {
  printer.printSymbolName(symbolAttr.getName());
}

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

  if (!llvm::all_of(getDistributedShape().getSymbols(),
                    llvm::IsaPred<wave::WaveSymbolAttr>)) {
    return emitOpError()
           << "distributed_shape must only contain WaveSymbolAttr";
  }

  return llvm::success();
}

//-----------------------------------------------------------------------------
// IterateOp
//-----------------------------------------------------------------------------

void wave::IterateOp::makeIsolated(RewriterBase &rewriter) {
  // Find all uses inside the body of values defined above.
  llvm::SetVector<Value> captures;
  llvm::SmallVector<OpOperand *> captureOperands;
  llvm::SmallPtrSet<Region *, 8> ancestorRegions;
  for (Operation *op = getOperation(); op != nullptr; op = op->getParentOp()) {
    ancestorRegions.insert(op->getParentRegion());
  }
  getOperation()->walk([&](Operation *op) {
    if (op == getOperation())
      return WalkResult::advance();

    for (OpOperand &operand : op->getOpOperands()) {
      if (!ancestorRegions.contains(operand.get().getParentRegion()))
        continue;
      captureOperands.push_back(&operand);
      captures.insert(operand.get());
    }
    return WalkResult::advance();
  });

  // Capture values defined above.
  llvm::SmallVector<Location> newCaptureLocs = llvm::map_to_vector(
      captures, [&](Value value) { return value.getLoc(); });
  rewriter.modifyOpInPlace(
      *this, [&] { getCapturesMutable().append(captures.getArrayRef()); });

  // Add trailing block arguments for captured values. The little dance with the
  // rewriter is a way to append block arguments.
  llvm::SmallVector<Type> allTypes(getLoopBody()->getArgumentTypes());
  llvm::append_range(allTypes, ValueRange(captures.getArrayRef()).getTypes());
  llvm::SmallVector<Location> allLocs =
      llvm::map_to_vector(getLoopBody()->getArguments(),
                          [](Value value) { return value.getLoc(); });
  llvm::append_range(allLocs, newCaptureLocs);
  Block *originalBlock = getLoopBody();
  Block *newBlock = rewriter.createBlock(originalBlock, allTypes, allLocs);
  rewriter.mergeBlocks(originalBlock, newBlock,
                       newBlock->getArguments().drop_back(captures.size()));
  ValueRange innerValues = newBlock->getArguments().take_back(captures.size());

  // Update uses in the body to use block arguments instead of the captured
  // values.
  for (OpOperand *opOperand : captureOperands) {
    rewriter.modifyOpInPlace(opOperand->getOwner(), [&] {
      auto it = llvm::find(captures, opOperand->get());
      assert(it != captures.end() && "expected capture to be found");
      size_t position = std::distance(captures.begin(), it);
      opOperand->set(innerValues[position]);
    });
  }
}

void wave::IterateOp::makeNonIsolated(RewriterBase &rewriter) {
  // Replace uses of block arguments with the captured values, these uses can
  // only be inside the body in well-formed SSA.
  for (auto &&[captureBlockArg, captured] :
       llvm::zip_equal(getCaptureBlockArgs(), getCaptures())) {
    rewriter.replaceAllUsesWith(captureBlockArg, captured);
  }

  // Remove block arguments for captured values that are no longer necessary.
  // The little dance is needed because the rewriter can't directly remove block
  // arguments. Note that it is fine to replace them with nullptr as they have
  // no uses at this point.
  unsigned numCaptures = getCaptures().size();
  rewriter.modifyOpInPlace(*this, [&] { getCapturesMutable().clear(); });
  Block *originalBlock = getLoopBody();
  auto types =
      TypeRange(originalBlock->getArgumentTypes()).drop_back(numCaptures);
  llvm::SmallVector<Location> locations =
      llvm::map_to_vector(originalBlock->getArguments().drop_back(numCaptures),
                          [](Value value) { return value.getLoc(); });
  Block *newBlock = rewriter.createBlock(getLoopBody(), types, locations);
  SmallVector<Value> replacementValues(newBlock->getArguments());
  replacementValues.append(numCaptures, Value());
  rewriter.mergeBlocks(originalBlock, newBlock, replacementValues);
}

bool wave::IterateOp::areTypesCompatible(Type lhs, Type rhs) {
  return detail::verifyTypesCompatible(llvm::cast<wave::WaveTensorType>(lhs),
                                       llvm::cast<wave::WaveTensorType>(rhs),
                                       /*includeAddressSpace=*/true)
      .succeeded();
}

OperandRange wave::IterateOp::getEntrySuccessorOperands(RegionSuccessor) {
  return getOperands().drop_back(getNumOperands());
}

void wave::IterateOp::getSuccessorRegions(
    RegionBranchPoint point,
    ::llvm::SmallVectorImpl<::RegionSuccessor> &regions) {
  // May branch into the region or bypass it regardless of the source.
  regions.emplace_back(
      RegionSuccessor(getOperation(), getResults().drop_back(getNumResults())));
  regions.emplace_back(
      RegionSuccessor(&getBody(), getLoopBody()->getArguments().drop_back(
                                      getLoopBody()->getNumArguments())));
}

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

llvm::LogicalResult wave::IterateOp::setIndexFromLattices(
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> operands,
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> resultExprs) {
  return detail::identitySetIndexFromLattices(*this, operands, resultExprs);
}

LogicalResult wave::IterateOp::verify() {
  if (getNumOperands() != getLoopBody()->getNumArguments()) {
    return emitOpError() << "expects the same number of operands ("
                         << getNumOperands() << ") and block arguments ("
                         << getLoopBody()->getNumArguments() << ")";
  }
  TypeRange blockIterArgTypes = getIterArgs().getTypes();
  TypeRange iterArgTypes =
      getOperands().drop_back(getCaptures().size()).getTypes();
  TypeRange captureTypes = getCaptures().getTypes();
  TypeRange captureBlockArgTypes = getCaptureBlockArgs().getTypes();
  TypeRange resultTypes = getResultTypes();
  if (iterArgTypes.size() != blockIterArgTypes.size()) {
    return emitOpError() << "expects the same number if iter_args ("
                         << iterArgTypes.size()
                         << ") and corresponding block arguments ("
                         << blockIterArgTypes.size() << ")";
  }
  if (blockIterArgTypes.size() != resultTypes.size()) {
    return emitOpError() << "expects the same number of iter_args ("
                         << blockIterArgTypes.size() << ") and results ("
                         << resultTypes.size() << ")";
  }
  for (auto &&[i, iterArg, blockIterArg, result] :
       llvm::enumerate(iterArgTypes, blockIterArgTypes, resultTypes)) {
    auto iterArgTensor = llvm::cast<wave::WaveTensorType>(iterArg);
    auto blockIterArgTensor = llvm::cast<wave::WaveTensorType>(blockIterArg);
    auto resultTensor = llvm::cast<wave::WaveTensorType>(result);

    auto istr = std::to_string(i);
    if (llvm::failed(detail::verifyTypesCompatible(
            iterArgTensor, blockIterArgTensor, /*includeAddressSpace=*/true,
            getLoc(), "operand iter_arg #" + istr,
            "block argument #" + istr))) {
      return llvm::failure();
    }
    if (llvm::failed(detail::verifyTypesCompatible(
            iterArgTensor, resultTensor, /*includeAddressSpace=*/true, getLoc(),
            "operand iter_arg #" + istr, "result #" + istr))) {
      return llvm::failure();
    }
  }
  for (auto &&[i, capture, captureBlockArg] :
       llvm::enumerate(captureTypes, captureBlockArgTypes)) {
    auto captureTensor = llvm::cast<wave::WaveTensorType>(capture);
    auto captureBlockArgTensor =
        llvm::cast<wave::WaveTensorType>(captureBlockArg);
    if (captureTensor != captureBlockArgTensor) {
      return emitOpError() << "expects the same type for capture #" << i
                           << " and block argument #"
                           << (getIterArgs().size() + i);
    }
  }

  return llvm::success();
}

llvm::LogicalResult wave::IterateOp::verifyRegions() {
  // Use the region hook since it runs after we verified the terminator itself
  // and know it is well-formed.
  TypeRange iterArgTypes = getIterArgs().getTypes();
  TypeRange blockIterArgTypes = TypeRange(getLoopBody()->getArgumentTypes())
                                    .take_front(iterArgTypes.size());
  TypeRange resultTypes = getResultTypes();
  TypeRange terminatorOperandTypes =
      getLoopBody()->getTerminator()->getOperands().getTypes();
  if (resultTypes.size() != terminatorOperandTypes.size()) {
    return emitOpError() << "expects the same number of results ("
                         << resultTypes.size() << ") and terminator operands ("
                         << terminatorOperandTypes.size() << ")";
  }
  for (auto &&[i, result, terminatorOperand, iterArg, blockIterArg] :
       llvm::enumerate(resultTypes, terminatorOperandTypes, iterArgTypes,
                       blockIterArgTypes)) {
    auto istr = std::to_string(i);
    if (llvm::failed(detail::verifyTypesCompatible(
            llvm::cast<wave::WaveTensorType>(result),
            llvm::cast<wave::WaveTensorType>(terminatorOperand),
            /*includeAddressSpace=*/true, getLoc(), "result #" + istr,
            "terminator operand #" + istr))) {
      return llvm::failure();
    }
  }
  return llvm::success();
}

//-----------------------------------------------------------------------------
// MmaOp
//-----------------------------------------------------------------------------

llvm::FailureOr<ChangeResult> wave::MmaOp::propagateForward(
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

llvm::FailureOr<ChangeResult> wave::MmaOp::propagateBackward(
    llvm::MutableArrayRef<wave::WaveTensorType> operandTypes,
    llvm::ArrayRef<wave::WaveTensorType> resultTypes, llvm::raw_ostream &errs) {
  // TODO: we may consider doing partial type propagation, but we can't infer
  // the reduction dimension from the result type since it is absent from it.
  return detail::propagateShapeInformation(resultTypes[0], operandTypes[2],
                                           "result", "accumulator", errs);
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

namespace llvm {
// Combine two potentially failing ChangeResults: if any of them failed, the
// result of the combination is also failure.
FailureOr<ChangeResult> operator|(FailureOr<ChangeResult> lhs,
                                  FailureOr<ChangeResult> rhs) {
  if (failed(lhs) || failed(rhs))
    return failure();
  return *lhs | *rhs;
}
} // namespace llvm

// Update index expressions of the result of the MMA operation.
llvm::FailureOr<ChangeResult> wave::MmaOp::propagateIndexExprsForward(
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> operandExprs,
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> resultExprs,
    wave::EmitErrorFn emitError) {
  auto resultType = dyn_cast<wave::WaveTensorType>(getResult().getType());
  if (!resultType)
    return updateIfChanged(resultExprs[0],
                           wave::IndexExprsLatticeStorage::top());

  // Join LHS (ignoring M symbol), RHS, and accumulator lattices into result.
  unsigned lhsOperandNumber = getLhsMutable().getOperandNumber();
  unsigned rhsOperandNumber = getRhsMutable().getOperandNumber();
  unsigned accumulatorOperandNumber =
      getAccumulatorMutable().getOperandNumber();

  wave::IndexExprsLatticeStorage resultLattice =
      wave::IndexExprsLatticeStorage::bottom();

  // LHS: ignore M symbol since it has different indexing in LHS vs result.
  if (auto lhsType = dyn_cast<wave::WaveTensorType>(getLhs().getType())) {
    Attribute mSymbol = lhsType.getShape()[0];
    resultLattice = wave::IndexExprsLatticeStorage::join(
        resultLattice, operandExprs[lhsOperandNumber], {mSymbol});
  }

  // RHS: propagate all symbols.
  if (llvm::isa<wave::WaveTensorType>(getRhs().getType())) {
    resultLattice = wave::IndexExprsLatticeStorage::join(
        resultLattice, operandExprs[rhsOperandNumber]);
  }

  // Accumulator: propagate all symbols.
  if (llvm::isa<wave::WaveTensorType>(getAccumulator().getType())) {
    resultLattice = wave::IndexExprsLatticeStorage::join(
        resultLattice, operandExprs[accumulatorOperandNumber]);
  }

  resultLattice = resultLattice.keepOnlySymbols(resultType.getShape());
  wave::IndexExprsLatticeStorage newResultLattice =
      wave::IndexExprsLatticeStorage::join(resultExprs[0], resultLattice);

  if (newResultLattice.isTop() && !resultExprs[0].isTop()) {
    InFlightDiagnostic diag =
        emitError()
        << "conflict when propagating forward to the result lattice in MmaOp";
    diag.attachNote() << "Result lattice: " << resultExprs[0];
    diag.attachNote() << "LHS lattice: " << operandExprs[lhsOperandNumber];
    diag.attachNote() << "RHS lattice: " << operandExprs[rhsOperandNumber];
    diag.attachNote() << "Accumulator lattice: "
                      << operandExprs[accumulatorOperandNumber];
    return diag;
  }

  return updateIfChanged(resultExprs[0], newResultLattice);
}

// Update index expressions of the operands of the MMA operation.
llvm::FailureOr<ChangeResult> wave::MmaOp::propagateIndexExprsBackward(
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> operandExprs,
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> resultExprs,
    wave::EmitErrorFn emitError) {
  const unsigned lhsOperandNumber = getLhsMutable().getOperandNumber();
  const unsigned rhsOperandNumber = getRhsMutable().getOperandNumber();
  const unsigned accumulatorOperandNumber =
      getAccumulatorMutable().getOperandNumber();

  // Create separate lattices for operands (ignoring M symbol from results)
  // and accumulator (with all symbols).
  wave::IndexExprsLatticeStorage operandLattice =
      wave::IndexExprsLatticeStorage::bottom();
  wave::IndexExprsLatticeStorage accumulatorLattice =
      wave::IndexExprsLatticeStorage::bottom();

  for (const wave::IndexExprsLatticeStorage &resultExpr : resultExprs) {
    auto resultType = dyn_cast<wave::WaveTensorType>(getResult().getType());
    if (!resultType)
      continue;

    // For LHS/RHS operands, ignore M symbol.
    Attribute mSymbol = resultType.getShape()[0];
    operandLattice = wave::IndexExprsLatticeStorage::join(
        operandLattice, resultExpr, {mSymbol});

    // For accumulator, use all symbols.
    accumulatorLattice =
        wave::IndexExprsLatticeStorage::join(accumulatorLattice, resultExpr);
  }

  ChangeResult changeResult = ChangeResult::NoChange;

  // Propagate to LHS (operand 0).
  if (auto lhsType = llvm::dyn_cast<wave::WaveTensorType>(getLhs().getType())) {
    wave::IndexExprsLatticeStorage filtered =
        operandLattice.keepOnlySymbols(lhsType.getShape());
    wave::IndexExprsLatticeStorage newLattice =
        wave::IndexExprsLatticeStorage::join(operandExprs[lhsOperandNumber],
                                             filtered);

    if (newLattice.isTop() && !operandExprs[lhsOperandNumber].isTop()) {
      InFlightDiagnostic diag =
          emitError()
          << "conflict when propagating to LHS from result in MmaOp";
      diag.attachNote() << "LHS lattice: " << operandExprs[lhsOperandNumber];
      diag.attachNote() << "result lattice: " << resultExprs[0];
      return diag;
    }

    if (newLattice != operandExprs[lhsOperandNumber]) {
      operandExprs[lhsOperandNumber] = newLattice;
      changeResult = ChangeResult::Change;
    }
  }

  // Propagate to RHS (operand 1).
  if (auto rhsType = llvm::dyn_cast<wave::WaveTensorType>(getRhs().getType())) {
    wave::IndexExprsLatticeStorage filtered =
        operandLattice.keepOnlySymbols(rhsType.getShape());
    wave::IndexExprsLatticeStorage newLattice =
        wave::IndexExprsLatticeStorage::join(operandExprs[rhsOperandNumber],
                                             filtered);

    if (newLattice.isTop() && !operandExprs[rhsOperandNumber].isTop()) {
      InFlightDiagnostic diag =
          emitError()
          << "conflict when propagating to RHS from result in MmaOp";
      diag.attachNote() << "RHS lattice: " << operandExprs[rhsOperandNumber];
      diag.attachNote() << "result lattice: " << resultExprs[0];
      return diag;
    }

    if (newLattice != operandExprs[rhsOperandNumber]) {
      operandExprs[rhsOperandNumber] = newLattice;
      changeResult = ChangeResult::Change;
    }
  }

  // Propagate to accumulator (operand 2).
  if (auto accType =
          llvm::dyn_cast<wave::WaveTensorType>(getAccumulator().getType())) {
    wave::IndexExprsLatticeStorage filtered =
        accumulatorLattice.keepOnlySymbols(accType.getShape());
    wave::IndexExprsLatticeStorage newLattice =
        wave::IndexExprsLatticeStorage::join(
            operandExprs[accumulatorOperandNumber], filtered);

    if (newLattice.isTop() && !operandExprs[accumulatorOperandNumber].isTop()) {
      InFlightDiagnostic diag =
          emitError()
          << "conflict when propagating to accumulator from result in MmaOp";
      diag.attachNote() << "accumulator lattice: "
                        << operandExprs[accumulatorOperandNumber];
      diag.attachNote() << "result lattice: " << resultExprs[0];
      return diag;
    }

    if (newLattice != operandExprs[accumulatorOperandNumber]) {
      operandExprs[accumulatorOperandNumber] = newLattice;
      changeResult = ChangeResult::Change;
    }
  }

  return changeResult;
}

// Check if the given type is one of the allowed types provided as template
// arguments and report an error at the given location otherwise.
template <typename... AllowedTypes>
static LogicalResult checkAllowedTypes(Location loc, Type type,
                                       llvm::StringRef name,
                                       wave::WaveMmaKind kind) {
  if (llvm::isa<AllowedTypes...>(type))
    return success();

  return emitError(loc) << "unexpected " << name << " elemental type " << type
                        << " for MMA kind " << wave::stringifyEnum(kind);
}

// Check if the given type is an integer type with one of the provided bitwidths
// and report and error at the given location otherwise.
template <typename T>
static std::enable_if_t<std::is_same_v<T, IntegerType>, LogicalResult>
checkAllowedTypes(Location loc, Type type, llvm::StringRef name,
                  wave::WaveMmaKind kind, llvm::ArrayRef<unsigned> bitwidths) {
  if (auto intType = llvm::dyn_cast<IntegerType>(type)) {
    if (llvm::is_contained(bitwidths, intType.getIntOrFloatBitWidth()))
      return success();
  }
  return emitError(loc) << "unexpected " << name << " elemental type " << type
                        << " for MMA kind " << wave::stringifyEnum(kind);
}

// Check if the types used for multiplication and accumulation in a `wave.mma`
// operation are compatible with the specified MMA kind and report an error
// otherwise.
static LogicalResult checkMmaTypeCompatibility(Location loc,
                                               wave::WaveMmaKind kind,
                                               Type mulType, Type accType) {
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
    success = succeeded(checkAllowedTypes<Float32Type>(
                  loc, accType, "accumulator/result", kind)) &&
              succeeded(checkAllowedTypes<Float16Type, BFloat16Type>(
                  loc, mulType, "lhs/rhs", kind));
    break;
  }

  case wave::WaveMmaKind::I32_16x16x16_I8:
  case wave::WaveMmaKind::I32_32x32x8_I8:
  case wave::WaveMmaKind::I32_16x16x32_I8:
  case wave::WaveMmaKind::I32_32x32x16_I8:
    success = succeeded(checkAllowedTypes<IntegerType>(
                  loc, accType, "accumulator/result", kind, {32})) &&
              succeeded(checkAllowedTypes<IntegerType>(loc, mulType, "lhs/rhs",
                                                       kind, {8}));
    break;

  case wave::WaveMmaKind::F32_16x16x32_F8:
  case wave::WaveMmaKind::F32_32x32x16_F8:
  case wave::WaveMmaKind::F32_16x16x32_K4_F8:
  case wave::WaveMmaKind::F32_32x32x16_K4_F8:
    success = succeeded(checkAllowedTypes<Float32Type>(
                  loc, accType, "accumulator/result", kind)) &&
              succeeded(checkAllowedTypes<Float8E3M4Type, Float8E5M2Type>(
                  loc, mulType, "lhs/rhs", kind));
    break;

  case wave::WaveMmaKind::F32_16x16x128_F8F6F4:
  case wave::WaveMmaKind::F32_32x32x64_F8F6F4:
    success =
        succeeded(checkAllowedTypes<Float32Type>(loc, accType,
                                                 "accumulator/result", kind)) &&
        succeeded(
            checkAllowedTypes<Float8E3M4Type, Float8E5M2Type, Float6E2M3FNType,
                              Float6E3M2FNType, Float4E2M1FNType>(
                loc, mulType, "lhs/rhs", kind));
    break;
  }

  return llvm::success(success);
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
  void populate(llvm::SmallVectorImpl<NamedAttribute> &attributes) const;

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
  void populate(llvm::SmallVectorImpl<NamedAttribute> &attributes) const {
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
                 buildMap(builder.sizeExpr), buildMap(builder.strideExpr))
          .removeUnusedInputs();
    };

    if (mSymbol)
      attributes.emplace_back(mSymbol.getName(), buildOne(mBuilder));
    if (nSymbol)
      attributes.emplace_back(nSymbol.getName(), buildOne(nBuilder));
    if (kSymbol)
      attributes.emplace_back(kSymbol.getName(), buildOne(kBuilder));
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

MmaSingleIndexExprBuilder &MmaSingleIndexExprBuilder::m() { return parent.m(); }
MmaSingleIndexExprBuilder &MmaSingleIndexExprBuilder::n() { return parent.n(); }
MmaSingleIndexExprBuilder &MmaSingleIndexExprBuilder::k() { return parent.k(); }
void MmaSingleIndexExprBuilder::populate(
    llvm::SmallVectorImpl<NamedAttribute> &attributes) const {
  parent.populate(attributes);
}
} // namespace

// Populate `attributes` with index expressions for the symbols associated with
// M, N, K dimensions of the given MMA operation kind provided the configuration
// of wavefronts in the workgroup. Any symbol may be omitted as long as at least
// one is provided, e.g., for the LHS of the operation, only M and N symbols may
// be provided. If `isAccumulator` is set, the index expressions are created for
// the accumulator/result of an MMA, which may affect the expression for the M
// dimension.
static llvm::LogicalResult
populateMmaIndexingExpr(wave::WaveMmaKind kind, bool isAccumulator,
                        llvm::ArrayRef<unsigned> wavesPerWorkgroup,
                        int64_t threadsPerWave, wave::WaveSymbolAttr mSymbol,
                        wave::WaveSymbolAttr nSymbol,
                        wave::WaveSymbolAttr kSymbol,
                        llvm::SmallVectorImpl<NamedAttribute> &attributes) {
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
        .populate(attributes);
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
        .populate(attributes);
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
        .populate(attributes);
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
        .populate(attributes);
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
        .populate(attributes);
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
        .populate(attributes);
    return llvm::success();
  default:
    return llvm::failure();
  }
}

/// Create per-symbol thread-independent index expressions for `indexingSymbols`
/// given constraints on them and put them into `symbolMappings` as named pairs
/// (symbol, index mapping attribute). Thread-independent means affected by
/// workgroup, tiling and device constraints, and NOT affected by wave
/// constraints or MMA shapes.
static void mixInThreadIndependentConstraints(
    llvm::ArrayRef<wave::WaveSymbolAttr> indexingSymbols,
    const llvm::DenseMap<wave::WaveSymbolAttr, llvm::SmallVector<Attribute>>
        &symbolConstraints,
    llvm::SmallVector<NamedAttribute> &symbolMappings) {
  for (wave::WaveSymbolAttr symbol : indexingSymbols) {
    auto it = symbolConstraints.find(symbol);
    if (it == symbolConstraints.end())
      continue;

    auto mappingIt = llvm::find_if(symbolMappings, [&](NamedAttribute attr) {
      return attr.getName() == symbol.getName();
    });
#ifndef NDEBUG
    if (mappingIt == symbolMappings.end()) {
      llvm::errs() << "symbol: " << symbol.getName() << "\n";
      assert(false && "expected a mapping for the symbol");
    }
#endif // NDEBUG
    wave::WaveIndexMappingAttr mapping =
        llvm::cast<wave::WaveIndexMappingAttr>(mappingIt->getValue());
    for (Attribute constraint : it->second) {
      // Tiling constraints are handled separately in loop propagation.
      if (!isa<wave::TilingConstraintAttr>(constraint))
        mapping = wave::applyConstraintGeneric(constraint, mapping);
    }
    mappingIt->setValue(mapping);
  }
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
  SmallVector<NamedAttribute> symbolMappings;
  symbolMappings.reserve(indexingSymbols.size());

  assert(indexingSymbols.size() == 2 &&
         "only 2 indexing symbols are currently supported for MMA result");
  wave::WaveSymbolAttr mSymbol = indexingSymbols[0];
  wave::WaveSymbolAttr nSymbol = indexingSymbols[1];

  std::optional<wave::WaveMmaKind> mmaKind = getKind();
  if (!mmaKind)
    return emitError() << "MMA operation without kind attribute not supported";
  if (llvm::failed(populateMmaIndexingExpr(
          *mmaKind,
          /*isAccumulator=*/true, initObject.wavesPerBlock,
          initObject.hardwareConstraint.getThreadsPerWave(), mSymbol, nSymbol,
          /*kSymbol=*/nullptr, symbolMappings))) {
    return emitError() << "MMA kind not supported by index deduction";
  }

  mixInThreadIndependentConstraints(
      indexingSymbols, initObject.symbolConstraints, symbolMappings);
  resultExprs[0].unsafeSet(DictionaryAttr::get(getContext(), symbolMappings));

  return llvm::success();
}

// Initialize the index expression lattices for the operands of the MMA
// operation. This sets index expressions to values derived from the MMA
// operation kind and wavefront-in-workgroup configuration (thread-dependent) as
// well as workgroup constraints (thread-independent).
LogicalResult MmaOp::initializeIndexExprsBackward(
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> operandExprs,
    const wave::IndexExprsAnalysisInit &initObject,
    wave::EmitErrorFn emitError) {
  auto resultType = llvm::cast<wave::WaveTensorType>(getResult().getType());
  auto lhsType = llvm::cast<wave::WaveTensorType>(getLhs().getType());
  assert(resultType.getRank() == lhsType.getRank() && lhsType.getRank() == 2 &&
         "only 2D MMA operations are supported");
  wave::WaveSymbolAttr mSymbol = resultType.getShape()[0];
  wave::WaveSymbolAttr nSymbol = resultType.getShape()[1];
  wave::WaveSymbolAttr kSymbol = lhsType.getShape()[1];

  std::optional<wave::WaveMmaKind> mmaKind = getKind();
  if (!mmaKind)
    return emitError() << "MMA operation without kind attribute not supported";

  llvm::SmallVector<NamedAttribute> operandSymbolMappings;
  if (llvm::failed(populateMmaIndexingExpr(
          *mmaKind, /*isAccumulator=*/false, initObject.wavesPerBlock,
          initObject.hardwareConstraint.getThreadsPerWave(), mSymbol, nSymbol,
          kSymbol, operandSymbolMappings))) {
    return emitError() << "MMA kind not supported by index deduction";
  }

  llvm::SmallVector<NamedAttribute> accumulatorSymbolMappings;
  if (llvm::failed(populateMmaIndexingExpr(
          *mmaKind,
          /*isAccumulator=*/true, initObject.wavesPerBlock,
          initObject.hardwareConstraint.getThreadsPerWave(), mSymbol, nSymbol,
          nullptr, accumulatorSymbolMappings))) {
    return emitError() << "MMA kind not supported by index deduction";
  }

  mixInThreadIndependentConstraints({mSymbol, nSymbol, kSymbol},
                                    initObject.symbolConstraints,
                                    operandSymbolMappings);
  mixInThreadIndependentConstraints({mSymbol, nSymbol},
                                    initObject.symbolConstraints,
                                    accumulatorSymbolMappings);

  // Create the LHS and RHS mappings that are not using symbols
  // irrelevant for them.
  llvm::SmallVector<NamedAttribute> lhsSymbolMappings =
      llvm::filter_to_vector(operandSymbolMappings, [&](NamedAttribute attr) {
        return attr.getName() != nSymbol.getName();
      });
  llvm::SmallVector<NamedAttribute> rhsSymbolMappings =
      llvm::filter_to_vector(operandSymbolMappings, [&](NamedAttribute attr) {
        return attr.getName() != mSymbol.getName();
      });

  operandExprs[getLhsMutable().getOperandNumber()] =
      wave::IndexExprsLatticeStorage(
          DictionaryAttr::get(getContext(), lhsSymbolMappings));
  operandExprs[getRhsMutable().getOperandNumber()] =
      wave::IndexExprsLatticeStorage(
          DictionaryAttr::get(getContext(), rhsSymbolMappings));
  operandExprs[getAccumulatorMutable().getOperandNumber()] =
      wave::IndexExprsLatticeStorage(
          DictionaryAttr::get(getContext(), accumulatorSymbolMappings));
  return llvm::success();
}

// Special case for MMA where we also want to have index expressions
// for the operands.
// TODO: this shouldn't be strictly necessary in a purely MLIR flow,
// but is kept for Python compatibility.
LogicalResult MmaOp::setIndexFromLattices(
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> operandExprs,
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> resultExprs) {
  llvm::SmallVector<Attribute> indexExprs;
  indexExprs.reserve(operandExprs.size() + resultExprs.size());
  for (OpOperand &operand : getOperation()->getOpOperands()) {
    if (llvm::failed(detail::checkAndAppendIndexExpr(
            getLoc(), operandExprs[operand.getOperandNumber()],
            "operand #" + llvm::Twine(operand.getOperandNumber()), indexExprs)))
      return failure();
  }
  for (auto &&[i, expr] : llvm::enumerate(resultExprs)) {
    if (llvm::failed(detail::checkAndAppendIndexExpr(
            getLoc(), resultExprs[i], "result #" + llvm::Twine(i), indexExprs)))
      return failure();
  }
  getOperation()->setAttr(wave::WaveDialect::kIndexWaveExprListAttrName,
                          ArrayAttr::get(getContext(), indexExprs));
  return llvm::success();
}

LogicalResult MmaOp::verify() {
  Type lhsTypeGeneric = getLhs().getType();
  Type rhsTypeGeneric = getRhs().getType();
  Type accumulatorTypeGeneric = getAccumulator().getType();
  Type resultTypeGeneric = getResult().getType();

  WaveTensorType lhsType = dyn_cast<wave::WaveTensorType>(lhsTypeGeneric);
  WaveTensorType rhsType = dyn_cast<wave::WaveTensorType>(rhsTypeGeneric);
  WaveTensorType accumulatorType =
      dyn_cast<wave::WaveTensorType>(accumulatorTypeGeneric);
  WaveTensorType resultType = dyn_cast<wave::WaveTensorType>(resultTypeGeneric);

  // TODO: need to verify vector types, but for that, we need to know what they
  // must look like based on the MMA enum.
  if (!lhsType || !rhsType || !accumulatorType || !resultType) {
    return success();
  }

  if (failed(detail::verifyElementTypesMatch(getLoc(), "LHS", lhsType, "RHS",
                                             rhsType)) ||
      failed(detail::verifyElementTypesMatch(getLoc(), "result", resultType,
                                             "accumulator", accumulatorType)))
    return failure();

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      accumulatorType.getRank() != 2) {
    return emitError() << "only 2D MMA operations are supported";
  }

  if (detail::verifyTypesMatchingDimensions(getLoc(), "LHS", lhsType, {1},
                                            "RHS", rhsType, {1})
          .failed() ||
      detail::verifyTypesMatchingDimensions(getLoc(), "LHS", lhsType, {0},
                                            "accumulator", accumulatorType, {0})
          .failed() ||
      detail::verifyTypesMatchingDimensions(getLoc(), "RHS", rhsType, {0},
                                            "accumulator", accumulatorType, {1})
          .failed()) {
    return failure();
  }

  if (!getKind())
    return success();

  return checkMmaTypeCompatibility(getLoc(), *getKind(),
                                   lhsType.getElementType(),
                                   accumulatorType.getElementType());
}

/// Compute the expected elements per thread for a specific MMA operand.
/// operandIndex: 0=LHS, 1=RHS, 2=Accumulator/Result
/// Returns failure if no constraints are found.
llvm::FailureOr<unsigned>
wave::MmaOp::computeElementsPerThreadForOperand(unsigned operandIndex) {
  std::optional<wave::WaveMmaKind> mmaKind = getKind();
  if (!mmaKind)
    return mlir::failure();
  wave::WaveMmaSpec spec =
      wave::WaveMmaKindAttr::getSpec(getContext(), *mmaKind);

  // Extract threads per wave from hardware constraint by walking up the
  // ancestry.
  mlir::Operation *op = getOperation();
  while (op) {
    if (auto constraints = op->getAttrOfType<mlir::ArrayAttr>(
            wave::WaveDialect::kWaveConstraintsAttrName)) {
      for (mlir::Attribute constraint : constraints) {
        if (auto hardwareConstraint =
                llvm::dyn_cast<wave::HardwareConstraintAttr>(constraint)) {
          unsigned totalElements;
          switch (operandIndex) {
          case 0: // LHS: M x K
            totalElements = spec.m * spec.k;
            break;
          case 1: // RHS: N x K
            totalElements = spec.n * spec.k;
            break;
          case 2: // Accumulator/Result: M x N
            totalElements = spec.m * spec.n;
            break;
          default:
            return mlir::failure();
          }
          return totalElements / hardwareConstraint.getThreadsPerWave();
        }
      }
    }
    op = op->getParentOp();
  }

  // Return failure if no constraints found.
  return mlir::failure();
}

llvm::FailureOr<mlir::ChangeResult>
wave::MmaOp::propagateElementsPerThreadForward(
    llvm::ArrayRef<wave::ElementsPerThreadLatticeValue> operandElements,
    llvm::MutableArrayRef<wave::ElementsPerThreadLatticeValue> resultElements,
    llvm::raw_ostream &errs) {
  llvm::FailureOr<unsigned> expectedElementsPerThreadResult =
      computeElementsPerThreadForOperand(
          getAccumulatorMutable().getOperandNumber());
  if (llvm::failed(expectedElementsPerThreadResult)) {
    errs << "MMA operation has no hardware constraints available";
    return mlir::failure();
  }
  unsigned expectedElementsPerThread = *expectedElementsPerThreadResult;
  wave::ElementsPerThreadLatticeValue expectedResult(expectedElementsPerThread);
  return wave::detail::checkAndPropagateElementsPerThreadFromConstant(
      expectedResult, llvm::ArrayRef<wave::ElementsPerThreadLatticeValue>(),
      resultElements, "computed from MMA kind", "", "result", errs);
}

llvm::FailureOr<mlir::ChangeResult>
wave::MmaOp::propagateElementsPerThreadBackward(
    llvm::MutableArrayRef<wave::ElementsPerThreadLatticeValue> operandElements,
    llvm::ArrayRef<wave::ElementsPerThreadLatticeValue>,
    llvm::raw_ostream &errs) {
  // For MMA, the accumulator should have the same elements per thread as the
  // result. The LHS and RHS operands may have different constraints based on
  // their dimensions.
  // MMA operation always has exactly 3 operands: LHS, RHS, Accumulator
  assert(operandElements.size() == 3 &&
         "MMA operation must have exactly 3 operands");

  unsigned lhsOperandNumber = getLhsMutable().getOperandNumber();
  unsigned rhsOperandNumber = getRhsMutable().getOperandNumber();
  unsigned accumulatorOperandNumber =
      getAccumulatorMutable().getOperandNumber();

  // Compute expected elements per thread for each operand
  llvm::FailureOr<unsigned> expectedLhsElementsPerThreadResult =
      computeElementsPerThreadForOperand(lhsOperandNumber);
  llvm::FailureOr<unsigned> expectedRhsElementsPerThreadResult =
      computeElementsPerThreadForOperand(rhsOperandNumber);
  llvm::FailureOr<unsigned> expectedAccumulatorElementsPerThreadResult =
      computeElementsPerThreadForOperand(accumulatorOperandNumber);

  if (llvm::failed(expectedLhsElementsPerThreadResult) ||
      llvm::failed(expectedRhsElementsPerThreadResult) ||
      llvm::failed(expectedAccumulatorElementsPerThreadResult)) {
    errs << "MMA operation has no hardware constraints available";
    return mlir::failure();
  }

  unsigned expectedLhsElementsPerThread = *expectedLhsElementsPerThreadResult;
  unsigned expectedRhsElementsPerThread = *expectedRhsElementsPerThreadResult;
  unsigned expectedAccumulatorElementsPerThread =
      *expectedAccumulatorElementsPerThreadResult;

  wave::ElementsPerThreadLatticeValue expectedLhs(expectedLhsElementsPerThread);
  wave::ElementsPerThreadLatticeValue expectedRhs(expectedRhsElementsPerThread);
  wave::ElementsPerThreadLatticeValue expectedAccumulator(
      expectedAccumulatorElementsPerThread);

  // Propagate elements_per_thread to LHS operand using the helper function
  llvm::MutableArrayRef<wave::ElementsPerThreadLatticeValue> lhsOnly =
      operandElements.slice(lhsOperandNumber, 1);

  llvm::FailureOr<mlir::ChangeResult> lhsResult =
      wave::detail::checkAndPropagateElementsPerThreadFromConstant(
          expectedLhs, llvm::ArrayRef<wave::ElementsPerThreadLatticeValue>(),
          lhsOnly, "computed from MMA kind", "", "LHS operand", errs);

  if (llvm::failed(lhsResult)) {
    return llvm::failure();
  }

  // Propagate elements_per_thread to RHS operand using the helper function
  llvm::MutableArrayRef<wave::ElementsPerThreadLatticeValue> rhsOnly =
      operandElements.slice(rhsOperandNumber, 1);

  llvm::FailureOr<mlir::ChangeResult> rhsResult =
      wave::detail::checkAndPropagateElementsPerThreadFromConstant(
          expectedRhs, llvm::ArrayRef<wave::ElementsPerThreadLatticeValue>(),
          rhsOnly, "computed from MMA kind", "", "RHS operand", errs);

  if (llvm::failed(rhsResult)) {
    return mlir::failure();
  }

  // Propagate to the accumulator operand.
  llvm::MutableArrayRef<wave::ElementsPerThreadLatticeValue> accumulatorOnly =
      operandElements.slice(accumulatorOperandNumber, 1);

  llvm::FailureOr<mlir::ChangeResult> accumulatorResult =
      wave::detail::checkAndPropagateElementsPerThreadFromConstant(
          expectedAccumulator,
          llvm::ArrayRef<wave::ElementsPerThreadLatticeValue>(),
          accumulatorOnly, "computed from MMA kind", "", "accumulator operand",
          errs);

  if (llvm::failed(accumulatorResult)) {
    return mlir::failure();
  }

  // Return Change if any operand changed
  return (*lhsResult == mlir::ChangeResult::Change ||
          *rhsResult == mlir::ChangeResult::Change ||
          *accumulatorResult == mlir::ChangeResult::Change)
             ? mlir::ChangeResult::Change
             : mlir::ChangeResult::NoChange;
}

//-----------------------------------------------------------------------------
// ReadOp
//-----------------------------------------------------------------------------

// Check the well-formedness of the index attribute (must have at most one
// non-unit dimension) and its correspondence with the explicit elements per
// thread, if provided, and with the number of elements in the vector type.
static LogicalResult
verifyIndexElementsPerThread(Operation *op, ArrayAttr indexAttr,
                             std::optional<int64_t> elementsPerThread,
                             wave::WaveTensorType tensorType,
                             Type maybeVectorType) {
  auto vectorType = dyn_cast<VectorType>(maybeVectorType);
  auto vectorSize = vectorType
                        ? std::optional<int64_t>(vectorType.getDimSize(0))
                        : std::nullopt;

  if (elementsPerThread && vectorSize && *elementsPerThread != *vectorSize) {
    return op->emitOpError()
           << "expected result vector type to have the "
              "number of elements per thread matching the attribute ("
           << *elementsPerThread << "), got " << vectorType.getDimSize(0);
  }

  // The 'index' attribute is optional. For non-MMA ops (read/write), we only
  // use a single index expression, which is stored as the first (and only)
  // dictionary inside the array attribute.
  ArrayAttr arr = dyn_cast_or_null<ArrayAttr>(indexAttr);
  if (!arr)
    return success();
  if (!llvm::hasSingleElement(arr.getValue()))
    return op->emitError() << "'index' attribute must contain exactly one "
                              "dictionary for this op, got "
                           << arr.size();
  DictionaryAttr indexDict = dyn_cast<DictionaryAttr>(arr[0]);
  if (!indexDict)
    return success();

  wave::WaveHyperparameterAttr hyper = wave::WaveHyperparameterAttr();
  for (Operation *cur = op; cur != nullptr && !hyper;
       cur = cur->getParentOp()) {
    hyper = cur->getAttrOfType<wave::WaveHyperparameterAttr>(
        WaveDialect::kHyperparameterAttrName);
  }
  // Default to empty hyperparameter set, sometimes we can run checks even in
  // absence of these.
  if (!hyper)
    hyper = wave::WaveHyperparameterAttr::get(
        op->getContext(), DictionaryAttr::get(op->getContext()));

  SmallVector<int64_t> shape =
      getUncollapsedVectorShape(tensorType.getShape(), indexDict, hyper);
  int64_t nonUnit = 1;
  bool hadDynamic = false;
  for (auto [i, size] : llvm::enumerate(shape)) {
    if (ShapedType::isDynamic(size)) {
      hadDynamic = true;
      continue;
    }

    if (size == 1) {
      continue;
    }
    if (nonUnit == 1) {
      nonUnit = size;
      continue;
    }

    InFlightDiagnostic diag =
        op->emitError() << "'index' has more than one entry with non-unit step";
    diag.attachNote() << "second non-unit step dimension: " << i;
    return diag;
  }

  // If there were unevaluated steps, they may end up matching later on.
  if (hadDynamic)
    return success();

  if (elementsPerThread && nonUnit != *elementsPerThread) {
    return op->emitError() << "vectorized dimension step in the index "
                              "expression with current hyperparameters ("
                           << nonUnit
                           << ") doesn't match the explicitly specified "
                              "elements per thread value ("
                           << *elementsPerThread << ")";
  }

  if (vectorSize && nonUnit != *vectorSize) {
    return op->emitError() << "vectorized dimension step in the index "
                              "expression with current hyperparameters ("
                           << nonUnit << ") doesn't match the vector size ("
                           << *vectorSize << ")";
  }
  return success();
}

// Check that if the given read/write operation has bound expressions specified,
// each symbolic dimension of the WaveTensorType has exactly one bound
// expression.
static LogicalResult verifyReadWriteBounds(Location loc,
                                           wave::WaveTensorType boundedType,
                                           DictionaryAttr bounds) {
  assert(bounds && "expected non-null bounds");
  assert(boundedType && "expected non-null type");

  // We need a fixed iteration order of names for determinism of error messages,
  // so using a vector instead of a StringSet.
  // TODO: consider refactoring bounds and other dictionary-like attributes to
  // be indexed by symbol expressions rather than string attributes to avoid
  // string comparisons everywhere.
  SmallVector<StringRef> requiredSymbolNames = llvm::map_to_vector(
      boundedType.getShape(),
      [](wave::WaveSymbolAttr symbol) { return symbol.getName(); });
  llvm::StringSet<> knownSymbolNames;
  for (NamedAttribute value : bounds) {
    if (!llvm::is_contained(requiredSymbolNames, value.getName().strref())) {
      return emitError(loc)
             << "'bounds' specified for a symbol " << value.getName()
             << " not used in the "
                "indexed memory tensor";
    }

    // Value type must be WaveExprListAttr.
    if (!isa<wave::WaveExprListAttr>(value.getValue()))
      return emitError(loc) << "'bounds' values must be WaveExprListAttr, got "
                            << value.getValue();

    knownSymbolNames.insert(value.getName().strref());
  }
  for (StringRef requiredName : requiredSymbolNames) {
    if (knownSymbolNames.contains(requiredName))
      continue;

    return emitError(loc) << "bounds not provided for memory tensor symbol '"
                          << requiredName << "'";
  }

  return success();
}

LogicalResult ReadOp::verify() {
  if (failed(verifyIndexElementsPerThread(
          *this, getIndexAttr(), getElementsPerThread(), getMemory().getType(),
          getResult().getType())))
    return failure();

  wave::WaveReadWriteBoundsAttr bounds =
      getBounds().value_or(wave::WaveReadWriteBoundsAttr());
  if (!bounds)
    return success();

  return verifyReadWriteBounds(getLoc(), getMemory().getType(),
                               bounds.getMapping());
}

llvm::FailureOr<mlir::ChangeResult>
wave::ReadOp::propagateElementsPerThreadForward(
    llvm::ArrayRef<wave::ElementsPerThreadLatticeValue>,
    llvm::MutableArrayRef<wave::ElementsPerThreadLatticeValue> resultElements,
    llvm::raw_ostream &errs) {
  // ReadOp only propagates elements_per_thread attribute to result (register).
  // Memory operand is ignored for propagation - you can read any number of
  // elements from memory regardless of how many were written.
  std::optional<int64_t> elementsPerThread = getElementsPerThread();
  if (!elementsPerThread)
    return mlir::ChangeResult::NoChange;

  wave::ElementsPerThreadLatticeValue expectedResult(*elementsPerThread);
  return wave::detail::checkAndPropagateElementsPerThreadFromConstant(
      expectedResult, llvm::ArrayRef<wave::ElementsPerThreadLatticeValue>(),
      resultElements, "elements_per_thread attribute", "", "result", errs);
}

llvm::FailureOr<mlir::ChangeResult>
wave::ReadOp::propagateElementsPerThreadBackward(
    llvm::MutableArrayRef<wave::ElementsPerThreadLatticeValue>,
    llvm::ArrayRef<wave::ElementsPerThreadLatticeValue> resultElements,
    llvm::raw_ostream &) {
  // ReadOp doesn't propagate backward to memory operand.
  // Memory is decoupled from register dataflow for elements_per_thread.
  return mlir::ChangeResult::NoChange;
}

//-----------------------------------------------------------------------------
// RegisterOp
//-----------------------------------------------------------------------------

LogicalResult wave::RegisterOp::verify() {
  Type type = getResult().getType();
  auto tensorType = dyn_cast<WaveTensorType>(type);
  auto elementType = tensorType ? tensorType.getElementType()
                                : cast<VectorType>(type).getElementType();
  Type initType = getInit().getType();
  if (elementType != initType) {
    return emitOpError() << "expected the type of the init value to match the "
                            "elemental type of the result";
  }
  if (!tensorType)
    return success();

  if (!tensorType.getFullySpecified()) {
    return emitOpError() << "expected fully-specified tensor type";
  }
  return success();
}

//-----------------------------------------------------------------------------
// ExtractSliceOp
//-----------------------------------------------------------------------------

LogicalResult ExtractSliceOp::verify() {
  wave::WaveExprListAttr offset = getOffset();
  wave::WaveExprListAttr size = getSize();
  wave::WaveExprListAttr stride = getStride();

  if (failed(wave::verifyExprAttrsSameRank({offset, size, stride}))) {
    return emitOpError() << "offset, size, and stride must all have the same "
                            "rank, but got offset rank "
                         << offset.getRank() << ", size rank " << size.getRank()
                         << ", and stride rank " << stride.getRank();
  }

  if (failed(wave::verifyExprAttrsNoSymbols({offset, size, stride}))) {
    return emitOpError() << "offset, size, and stride must be constant "
                            "expressions with no symbols, but got offset with "
                         << offset.getNumSymbols() << " symbols, size with "
                         << size.getNumSymbols() << " symbols, and stride with "
                         << stride.getNumSymbols() << " symbols";
  }

  if (!llvm::all_of(offset.getSymbols(), llvm::IsaPred<wave::WaveSymbolAttr>)) {
    return emitOpError() << "offset must only contain WaveSymbolAttr";
  }

  if (!llvm::all_of(size.getSymbols(), llvm::IsaPred<wave::WaveSymbolAttr>)) {
    return emitOpError() << "size must only contain WaveSymbolAttr";
  }

  if (!llvm::all_of(stride.getSymbols(), llvm::IsaPred<wave::WaveSymbolAttr>)) {
    return emitOpError() << "stride must only contain WaveSymbolAttr";
  }

  return success();
}

//-----------------------------------------------------------------------------
// WriteOp
//-----------------------------------------------------------------------------

LogicalResult WriteOp::verify() {
  if (failed(verifyIndexElementsPerThread(
          *this, getIndexAttr(), getElementsPerThread(), getMemory().getType(),
          getValueToStore().getType())))
    return failure();

  wave::WaveReadWriteBoundsAttr bounds =
      getBounds().value_or(wave::WaveReadWriteBoundsAttr());
  if (!bounds)
    return success();

  return verifyReadWriteBounds(getLoc(), getMemory().getType(),
                               bounds.getMapping());
}

llvm::FailureOr<mlir::ChangeResult>
wave::WriteOp::propagateElementsPerThreadForward(
    llvm::ArrayRef<wave::ElementsPerThreadLatticeValue> operandElements,
    llvm::MutableArrayRef<wave::ElementsPerThreadLatticeValue>,
    llvm::raw_ostream &errs) {
  // WriteOp only validates that elements_per_thread attribute matches register
  // operand. Memory operand is ignored for propagation - you can write to
  // memory with any layout.
  std::optional<int64_t> elementsPerThread = getElementsPerThread();
  if (!elementsPerThread)
    return mlir::ChangeResult::NoChange;

  // Validate register operand (value_to_store) matches attribute.
  wave::ElementsPerThreadLatticeValue expectedValue(*elementsPerThread);
  llvm::ArrayRef<wave::ElementsPerThreadLatticeValue> valueOnly =
      operandElements.slice(getValueToStoreMutable().getOperandNumber(), 1);

  return wave::detail::checkAndPropagateElementsPerThreadFromConstant(
      expectedValue, valueOnly,
      llvm::MutableArrayRef<wave::ElementsPerThreadLatticeValue>(),
      "elements_per_thread attribute", "operand", "", errs);
}

llvm::FailureOr<mlir::ChangeResult>
wave::WriteOp::propagateElementsPerThreadBackward(
    llvm::MutableArrayRef<wave::ElementsPerThreadLatticeValue> operandElements,
    llvm::ArrayRef<wave::ElementsPerThreadLatticeValue>,
    llvm::raw_ostream &errs) {
  // WriteOp only propagates backward to register operand (value_to_store).
  // Memory operand is ignored - you can write any layout to memory.
  std::optional<int64_t> elementsPerThread = getElementsPerThread();
  if (!elementsPerThread)
    return mlir::ChangeResult::NoChange;

  // Propagate to register operand only.
  wave::ElementsPerThreadLatticeValue expectedValue(*elementsPerThread);
  llvm::MutableArrayRef<wave::ElementsPerThreadLatticeValue> valueOnly =
      operandElements.slice(getValueToStoreMutable().getOperandNumber(), 1);

  return wave::detail::checkAndPropagateElementsPerThreadFromConstant(
      expectedValue, llvm::ArrayRef<wave::ElementsPerThreadLatticeValue>(),
      valueOnly, "elements_per_thread attribute", "", "operand", errs);
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

// WriteOp has no results, nothing to propagate to. Not propagating sideways
// between operands either, the same index expression is used for the memory
// operand, we can then read from the same memory with different index.
llvm::FailureOr<ChangeResult> wave::WriteOp::propagateIndexExprsBackward(
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> operandExprs,
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> resultExprs,
    wave::EmitErrorFn emitError) {
  return ChangeResult::NoChange;
}

// Special case for WriteOp where we want an index expression even
// though it doesn't have results.
// TODO: this shouldn't be necessary in a purely MLIR form since
// mappings are a property of the SSA value (conversely, changing the
// mapping should create a new value), but keeping for compatibility.
llvm::LogicalResult wave::WriteOp::setIndexFromLattices(
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> operandExprs,
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> resultExprs) {
  llvm::SmallVector<Attribute> indexExprs;
  indexExprs.reserve(resultExprs.size() + 1);
  if (llvm::failed(detail::checkAndAppendIndexExpr(
          getLoc(), operandExprs[getValueToStoreMutable().getOperandNumber()],
          "value to store", indexExprs)))
    return llvm::failure();
  for (auto &&[i, expr] : llvm::enumerate(resultExprs)) {
    if (llvm::failed(detail::checkAndAppendIndexExpr(
            getLoc(), resultExprs[i], "result #" + llvm::Twine(i), indexExprs)))
      return llvm::failure();
  }
  getOperation()->setAttr(wave::WaveDialect::kIndexWaveExprListAttrName,
                          ArrayAttr::get(getContext(), indexExprs));
  return llvm::success();
}

//-----------------------------------------------------------------------------
// YieldOp
//-----------------------------------------------------------------------------

MutableOperandRange
wave::YieldOp::getMutableSuccessorOperands(RegionSuccessor) {
  // Create an empty mutable operand range (it has no default constructor).
  return getValuesMutable().slice(/*subStart=*/0, /*subLen=*/0);
}

//-----------------------------------------------------------------------------
// CastOp
//-----------------------------------------------------------------------------

LogicalResult wave::CastOp::verify() {
  Type valueType = getValueToCast().getType();
  Type resultType = getResult().getType();

  wave::WaveTensorType valueTensor =
      llvm::dyn_cast<wave::WaveTensorType>(valueType);
  wave::WaveTensorType resultTensor =
      llvm::dyn_cast<wave::WaveTensorType>(resultType);
  VectorType valueVec = llvm::dyn_cast<VectorType>(valueType);
  VectorType resultVec = llvm::dyn_cast<VectorType>(resultType);
  if (valueTensor && resultTensor && valueTensor.getFullySpecified() &&
      resultTensor.getFullySpecified() &&
      valueTensor.getShape() != resultTensor.getShape()) {
    return emitOpError("shape of input (")
           << valueTensor.getShape() << ") must match shape of result ("
           << resultTensor.getShape() << ")";
  }

  if (valueVec && resultVec && valueVec.getShape() != resultVec.getShape()) {
    return emitOpError("shape of input (")
           << valueVec.getShape() << ") must match shape of result ("
           << resultVec.getShape() << ")";
  }

  return success();
}
