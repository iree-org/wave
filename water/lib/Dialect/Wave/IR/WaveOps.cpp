// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveOps.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "water/Dialect/Wave/IR/IndexExpr.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVectorExtras.h"
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

// Parse an array of wave symbols like [@M, @N, @K].
// Custom parsing is required because MLIR's default parser creates
// SymbolRefAttr for @Name syntax, but Wave requires WaveSymbolAttr for type
// system consistency.
static ParseResult parseSymbolArray(OpAsmParser &parser,
                                    ArrayAttr &symbolArrayAttr) {
  SmallVector<Attribute> symbols;
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Square, [&]() {
        StringAttr strAttr;
        if (failed(parser.parseSymbolName(strAttr)))
          return failure();
        symbols.push_back(
            wave::WaveSymbolAttr::get(parser.getContext(), strAttr.getValue()));
        return success();
      }))
    return failure();
  symbolArrayAttr = parser.getBuilder().getArrayAttr(symbols);
  return success();
}

// Print an array of wave symbols like [@M, @N, @K].
static void printSymbolArray(OpAsmPrinter &printer, Operation *,
                             ArrayAttr symbolArrayAttr) {
  printer << "[";
  llvm::interleaveComma(symbolArrayAttr, printer, [&](Attribute attr) {
    auto sym = llvm::cast<wave::WaveSymbolAttr>(attr);
    printer.printSymbolName(sym.getName());
  });
  printer << "]";
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

  if (hasParent && getTailPadding())
    return emitOpError() << "only top-level allocations can have tail_padding";

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

bool wave::IterateOp::areTypesCompatible(mlir::Type lhs, mlir::Type rhs) {
  // Handle both WaveTensorType and VectorType combinations.
  auto lhsTensor = llvm::dyn_cast<wave::WaveTensorType>(lhs);
  auto rhsTensor = llvm::dyn_cast<wave::WaveTensorType>(rhs);
  auto lhsVector = llvm::dyn_cast<mlir::VectorType>(lhs);
  auto rhsVector = llvm::dyn_cast<mlir::VectorType>(rhs);

  // Both are wave tensors - check shape and address space compatibility.
  if (lhsTensor && rhsTensor) {
    return detail::verifyTypesCompatible(lhsTensor, rhsTensor,
                                         /*includeAddressSpace=*/true)
        .succeeded();
  }

  // Both are vectors - simple equality check.
  if (lhsVector && rhsVector) {
    return lhsVector == rhsVector;
  }

  // Mixed types are not compatible.
  return false;
}

OperandRange wave::IterateOp::getEntrySuccessorOperands(RegionSuccessor) {
  return getOperands().drop_back(getNumOperands());
}

void wave::IterateOp::getSuccessorRegions(
    RegionBranchPoint point,
    ::llvm::SmallVectorImpl<::RegionSuccessor> &regions) {
  // May branch into the region or bypass it regardless of the source.
  regions.emplace_back(RegionSuccessor::parent());
  regions.emplace_back(RegionSuccessor(&getBody()));
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
  for (auto &&[i, iterArg, result] :
       llvm::enumerate(iterArgTypes, resultTypes)) {
    // Handle verification for both wave tensors and vectors.
    auto iterArgTensor = llvm::dyn_cast<wave::WaveTensorType>(iterArg);
    auto resultTensor = llvm::dyn_cast<wave::WaveTensorType>(result);

    auto istr = std::to_string(i);

    // Both are wave tensors - verify shapes match across all dimensions.
    if (iterArgTensor && resultTensor) {
      if (!iterArgTensor.getFullySpecified() ||
          !resultTensor.getFullySpecified())
        continue;

      auto allDims =
          llvm::to_vector(llvm::iota_range<int>(0, iterArgTensor.getRank(),
                                                /*Inclusive=*/false));
      if (mlir::failed(detail::verifyTypesMatchingDimensions(
              getLoc(), "iter_args #" + istr, iterArgTensor, allDims,
              "result #" + istr, resultTensor, allDims)))
        return mlir::failure();
    }
    // Both are vectors - check exact type equality.
    else if (isa<VectorType>(iterArg) && isa<VectorType>(result)) {
      if (iterArg != result) {
        return emitOpError()
               << "iter_args #" << i << " type (" << iterArg
               << ") must match result #" << i << " type (" << result << ")";
      }
    }
    // Mixed types are not allowed.
    else {
      return emitOpError() << "iter_args #" << i << " and result #" << i
                           << " must be the same category of types (both wave "
                              "tensors or both vectors)";
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

    auto iterArgTensor = llvm::dyn_cast<wave::WaveTensorType>(iterArg);
    auto resultTensor = llvm::dyn_cast<wave::WaveTensorType>(result);
    auto blockIterArgTensor =
        llvm::dyn_cast<wave::WaveTensorType>(blockIterArg);
    auto terminatorOperandTensor =
        llvm::dyn_cast<wave::WaveTensorType>(terminatorOperand);

    // Verify result type vs terminator operand type.
    if (resultTensor && terminatorOperandTensor) {
      if (llvm::failed(detail::verifyTypesCompatible(
              resultTensor, terminatorOperandTensor,
              /*includeAddressSpace=*/true, getLoc(), "result #" + istr,
              "terminator operand #" + istr))) {
        return llvm::failure();
      }
    } else if (isa<VectorType>(result) && isa<VectorType>(terminatorOperand)) {
      // For vector types, just check that they are exactly equal.
      if (result != terminatorOperand) {
        return emitOpError() << "result #" << i << " type (" << result
                             << ") does not match terminator operand #" << i
                             << " type (" << terminatorOperand << ")";
      }
    } else if (result != terminatorOperand) {
      return emitOpError() << "result #" << i << " type (" << result
                           << ") and terminator operand #" << i << " type ("
                           << terminatorOperand << ") are not compatible types";
    }

    // Verify iter arg type vs block arg type.
    if (iterArgTensor && blockIterArgTensor) {
      if (llvm::failed(detail::verifyTypesCompatible(
              iterArgTensor, blockIterArgTensor,
              /*includeAddressSpace=*/true, getLoc(), "iter arg #" + istr,
              "block iter arg #" + istr))) {
        return llvm::failure();
      }
    } else if (isa<VectorType>(iterArg) && isa<VectorType>(blockIterArg)) {
      // For vector types, just check that they are exactly equal.
      if (iterArg != blockIterArg) {
        return emitOpError() << "iter arg #" << i << " type (" << iterArg
                             << ") does not match block iter arg #" << i
                             << " type (" << blockIterArg << ")";
      }
    } else if (iterArg != blockIterArg) {
      return emitOpError() << "iter arg #" << i << " type (" << iterArg
                           << ") and block iter arg #" << i << " type ("
                           << blockIterArg << ") are not compatible types";
    }
  }

  // Verify capture types match their corresponding block arguments.
  TypeRange captureTypes = getCaptures().getTypes();
  TypeRange captureBlockArgTypes = TypeRange(getLoopBody()->getArgumentTypes())
                                       .take_back(captureTypes.size());
  for (auto &&[i, capture, captureBlockArg] :
       llvm::enumerate(captureTypes, captureBlockArgTypes)) {
    if (capture != captureBlockArg) {
      return emitOpError() << "expects the same type for capture #" << i
                           << " and block argument #"
                           << (getIterArgs().size() + i);
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

LogicalResult wave::MmaOp::finalizeTypeInference() { return success(); }

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
          buildMap(builder.sizeExpr), buildMap(builder.strideExpr));
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
/// constraints or MMA shapes. The first argument indicates for which operation
/// the constraints are being used, which is in particular necessary to only
/// apply tiling constraints inside the relevant loops.
static void mixInThreadIndependentConstraints(
    Operation *where, uint64_t threadsPerWave,
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
    wave::WaveIndexMappingAttr mapping =
        mappingIt != symbolMappings.end()
            ? llvm::cast<wave::WaveIndexMappingAttr>(mappingIt->getValue())
            : nullptr;

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
        if (iterSymbol.getName() == symbol.getName()) {
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

    if (mappingIt != symbolMappings.end())
      mappingIt->setValue(mapping);
    else if (mapping)
      symbolMappings.emplace_back(symbol.getName(), mapping);
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
      *this, initObject.hardwareConstraint.getThreadsPerWave(), indexingSymbols,
      initObject.symbolConstraints, symbolMappings);
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

  mixInThreadIndependentConstraints(
      *this, initObject.hardwareConstraint.getThreadsPerWave(),
      {mSymbol, nSymbol, kSymbol}, initObject.symbolConstraints,
      operandSymbolMappings);
  mixInThreadIndependentConstraints(
      *this, initObject.hardwareConstraint.getThreadsPerWave(),
      {mSymbol, nSymbol}, initObject.symbolConstraints,
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
    llvm::raw_ostream &errs, const wave::ElementsPerThreadInit &) {
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
    llvm::raw_ostream &errs, const wave::ElementsPerThreadInit &) {
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
    auto exprListAttr = dyn_cast<wave::WaveExprListAttr>(value.getValue());
    if (!exprListAttr)
      return emitError(loc) << "'bounds' values must be WaveExprListAttr, got "
                            << value.getValue();
    if (exprListAttr.getRank() != 1) {
      return emitError(loc)
             << "'bounds' must only contain single-result expressions";
    }

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

/// Common verification logic for ReadOp and WriteOp.
static LogicalResult verifyReadWriteOp(Operation *op, ArrayAttr indexAttr,
                                       std::optional<int64_t> elementsPerThread,
                                       Type memoryType, Type valueType,
                                       WaveReadWriteBoundsAttr bounds,
                                       ArrayAttr orderedSyms) {

  if (failed(wave::detail::verifyElementTypesMatch(
          op->getLoc(), "memory", memoryType, "register", valueType)))
    return failure();

  // Skip the rest of the verification if memory is already resolved to
  // MemRefType.
  auto tensorType = dyn_cast<WaveTensorType>(memoryType);
  if (!tensorType)
    return success();

  // When tensor type is present, verify ordered_syms matches if specified.
  if (orderedSyms) {
    ArrayRef<WaveSymbolAttr> shape = tensorType.getShape();
    if (orderedSyms.size() != shape.size()) {
      return op->emitOpError()
             << "'ordered_syms' size (" << orderedSyms.size()
             << ") does not match memory tensor rank (" << shape.size() << ")";
    }
    for (auto [i, pair] : llvm::enumerate(llvm::zip(orderedSyms, shape))) {
      auto orderedSym = cast<WaveSymbolAttr>(std::get<0>(pair));
      WaveSymbolAttr shapeSym = std::get<1>(pair);
      if (orderedSym.getName() != shapeSym.getName()) {
        return op->emitOpError()
               << "'ordered_syms' symbol at index " << i << " ('"
               << orderedSym.getName()
               << "') does not match memory tensor shape symbol ('"
               << shapeSym.getName() << "')";
      }
    }
  }

  if (failed(verifyIndexElementsPerThread(op, indexAttr, elementsPerThread,
                                          tensorType, valueType)))
    return failure();

  if (!bounds)
    return success();

  return verifyReadWriteBounds(op->getLoc(), tensorType, bounds.getMapping());
}

LogicalResult ReadOp::verify() {
  return verifyReadWriteOp(*this, getIndexAttr(), getElementsPerThread(),
                           getMemory().getType(), getResult().getType(),
                           getBoundsAttr(), getOrderedSymsAttr());
}

llvm::FailureOr<mlir::ChangeResult>
wave::ReadOp::propagateElementsPerThreadForward(
    llvm::ArrayRef<wave::ElementsPerThreadLatticeValue>,
    llvm::MutableArrayRef<wave::ElementsPerThreadLatticeValue> resultElements,
    llvm::raw_ostream &errs, const wave::ElementsPerThreadInit &) {
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
    llvm::raw_ostream &, const wave::ElementsPerThreadInit &) {
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
// ExtractOp
//-----------------------------------------------------------------------------

FailureOr<ChangeResult> wave::ExtractOp::propagateElementsPerThreadForward(
    llvm::ArrayRef<wave::ElementsPerThreadLatticeValue> operandElements,
    llvm::MutableArrayRef<wave::ElementsPerThreadLatticeValue> resultElements,
    llvm::raw_ostream &errs, const wave::ElementsPerThreadInit &) {

  return detail::checkAndPropagateElementsPerThreadFromConstant(
      wave::ElementsPerThreadLatticeValue(1), /*immutableValues=*/{},
      resultElements, "op semantics", "", "result", errs);
}

FailureOr<ChangeResult> wave::ExtractOp::propagateElementsPerThreadBackward(
    llvm::MutableArrayRef<wave::ElementsPerThreadLatticeValue> operandElements,
    llvm::ArrayRef<wave::ElementsPerThreadLatticeValue> resultElements,
    llvm::raw_ostream &errs, const wave::ElementsPerThreadInit &) {
  // We don't have enough information to propagate backwards here.
  return ChangeResult::NoChange;
}

LogicalResult ExtractOp::verify() {
  wave::WaveExprListAttr position = getPosition();
  if (position.getRank() != 1) {
    return emitOpError() << "position must contain exactly one expression, but "
                            "got "
                         << position.getRank();
  }

  if (failed(detail::verifyElementTypesMatch(getLoc(), "source",
                                             getSource().getType(), "result",
                                             getResult().getType()))) {
    return failure();
  }

  if (auto resultVectorType = dyn_cast<VectorType>(getResult().getType())) {
    if (resultVectorType.getShape()[0] != 1) {
      return emitOpError() << "result must be a 1-element vector, got "
                           << resultVectorType;
    }
    return success();
  }

  auto resultTensorType = cast<WaveTensorType>(getResult().getType());
  if (!resultTensorType.getFullySpecified() ||
      resultTensorType.getRank() != 1) {
    return emitOpError() << "result must be a 1-dimensional tensor, got "
                         << resultTensorType;
  }

  auto sourceTensorType = dyn_cast<WaveTensorType>(getSource().getType());
  // For mixed types, cannot do anything here.
  if (!sourceTensorType)
    return success();

  if (!sourceTensorType.getFullySpecified())
    return emitOpError() << "source tensor type must be fully specified";

  if (!llvm::is_contained(sourceTensorType.getShape(),
                          resultTensorType.getShape()[0])) {
    return emitOpError() << "source tensor type dimensions must contain the "
                            "result tensor type dimension";
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

  return success();
}

//-----------------------------------------------------------------------------
// WriteOp
//-----------------------------------------------------------------------------

LogicalResult WriteOp::verify() {
  return verifyReadWriteOp(*this, getIndexAttr(), getElementsPerThread(),
                           getMemory().getType(), getValueToStore().getType(),
                           getBoundsAttr(), getOrderedSymsAttr());
}

llvm::FailureOr<ChangeResult> wave::WriteOp::propagateElementsPerThreadForward(
    llvm::ArrayRef<wave::ElementsPerThreadLatticeValue> operandElements,
    llvm::MutableArrayRef<wave::ElementsPerThreadLatticeValue>,
    llvm::raw_ostream &errs, const wave::ElementsPerThreadInit &) {
  // WriteOp only validates that elements_per_thread attribute matches register
  // operand. Memory operand is ignored for propagation - you can write to
  // memory with any layout.
  std::optional<int64_t> elementsPerThread = getElementsPerThread();
  if (!elementsPerThread)
    return ChangeResult::NoChange;

  // Validate register operand (value_to_store) matches attribute.
  wave::ElementsPerThreadLatticeValue expectedValue(*elementsPerThread);
  llvm::ArrayRef<wave::ElementsPerThreadLatticeValue> valueOnly =
      operandElements.slice(getValueToStoreMutable().getOperandNumber(), 1);

  return wave::detail::checkAndPropagateElementsPerThreadFromConstant(
      expectedValue, valueOnly,
      llvm::MutableArrayRef<wave::ElementsPerThreadLatticeValue>(),
      "elements_per_thread attribute", "operand", "", errs);
}

llvm::FailureOr<ChangeResult> wave::WriteOp::propagateElementsPerThreadBackward(
    llvm::MutableArrayRef<wave::ElementsPerThreadLatticeValue> operandElements,
    llvm::ArrayRef<wave::ElementsPerThreadLatticeValue>,
    llvm::raw_ostream &errs, const wave::ElementsPerThreadInit &) {
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

// Propagating "sideways" between operands, but only if this would not result in
// conflicts.
llvm::FailureOr<ChangeResult> wave::WriteOp::propagateIndexExprsBackward(
    llvm::MutableArrayRef<wave::IndexExprsLatticeStorage> operandExprs,
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> resultExprs,
    wave::EmitErrorFn emitError) {
  auto joined =
      IndexExprsLatticeStorage::join(operandExprs[0], operandExprs[1]);

  // XXX: if sideways propagation would result in a new conflict, don't
  // propagate. This is a questionable design carried over from the initial
  // Python prototype.
  if (joined.isTop() && !(operandExprs[0].isTop() || operandExprs[1].isTop())) {
    return ChangeResult::NoChange;
  }

  ChangeResult changeResult = ChangeResult::NoChange;
  if (operandExprs[0] != joined) {
    operandExprs[0] = joined;
    changeResult = ChangeResult::Change;
  }
  if (operandExprs[1] != joined) {
    operandExprs[1] = joined;
    changeResult = ChangeResult::Change;
  }
  return changeResult;
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

//-----------------------------------------------------------------------------
// ReciprocalOp
//-----------------------------------------------------------------------------

LogicalResult wave::ReciprocalOp::verify() {
  Type argType = getArgument().getType();
  Type elementType = wave::getElementType(argType);
  if (!isa<FloatType>(elementType))
    return emitOpError("requires float element type, but got ") << elementType;

  return success();
}

//-----------------------------------------------------------------------------
// ApplyExprOp
//-----------------------------------------------------------------------------

LogicalResult wave::ApplyExprOp::verify() {
  for (Type operandType : getOperands().getTypes()) {
    auto waveTensorType = dyn_cast<WaveTensorType>(operandType);
    if (!waveTensorType)
      continue;

    if (waveTensorType.getAddressSpaceValue() !=
            WaveAddressSpace::Unspecified &&
        waveTensorType.getAddressSpaceValue() != WaveAddressSpace::Register)
      return emitOpError() << "tensor operands must be in register or "
                              "unspecified address space";
  }

  unsigned numResults = getExpr().getMap().getNumResults();
  if (numResults != 1)
    return emitOpError("expression must produce exactly one result, but got ")
           << numResults;

  if (!isa<IntegerType>(getElementType(getResult().getType())))
    return emitOpError() << "operates on integers only";

  llvm::BitVector usedOperandAttrs(getArguments().size());
  for (Attribute sym : getExpr().getSymbols()) {
    if (auto operand = dyn_cast<WaveOperandAttr>(sym)) {
      if (operand.getOperandNumber() >= getArguments().size()) {
        return emitOpError()
               << "expression uses operand #" << operand.getOperandNumber()
               << " but there are only " << getArguments().size()
               << " operands";
      }
      usedOperandAttrs.set(operand.getOperandNumber());
    }
  }
  usedOperandAttrs.flip();
  for (unsigned position : usedOperandAttrs.set_bits()) {
    emitWarning() << "operand #" << position
                  << " is not used in the expression";
  }

  return success();
}

//-----------------------------------------------------------------------------
// SelectOp
//-----------------------------------------------------------------------------

LogicalResult wave::SelectOp::verify() {
  if (failed(detail::verifyTypesCompatible(
          getLhs().getType(), getRhs().getType(), /*includeAddressSpace=*/false,
          getLoc(), "LHS", "RHS")))
    return failure();

  if (failed(detail::verifyTypesCompatible(
          getLhs().getType(), getResult().getType(),
          /*includeAddressSpace=*/false, getLoc(), "LHS", "result")))
    return failure();

  auto intType =
      dyn_cast<IntegerType>(getElementType(getCondition().getType()));
  if (!intType || intType.getWidth() != 1)
    return emitOpError("condition must be a tensor or vector of i1");

  SmallVector<int64_t> vecShapes;
  if (auto conditionVec = dyn_cast<VectorType>(getCondition().getType()))
    vecShapes.push_back(conditionVec.getNumElements());
  if (auto lhsVec = dyn_cast<VectorType>(getLhs().getType()))
    vecShapes.push_back(lhsVec.getNumElements());
  if (auto rhsVec = dyn_cast<VectorType>(getRhs().getType()))
    vecShapes.push_back(rhsVec.getNumElements());
  if (auto resultVec = dyn_cast<VectorType>(getResult().getType()))
    vecShapes.push_back(resultVec.getNumElements());

  for (int64_t i = 0, e = vecShapes.size(); i < e; ++i) {
    for (int64_t j = i + 1; j < e; ++j) {
      if (vecShapes[i] == vecShapes[j])
        continue;

      emitError() << "expects all vector shapes to be equal, got "
                  << vecShapes[i] << " and " << vecShapes[j];
    }
  }

  auto conditionTensor = dyn_cast<WaveTensorType>(getCondition().getType());
  auto lhsTensor = dyn_cast<WaveTensorType>(getLhs().getType());
  auto rhsTensor = dyn_cast<WaveTensorType>(getRhs().getType());
  auto resultTensor = dyn_cast<WaveTensorType>(getResult().getType());
  if (failed(detail::verifyTensorShapesCompatible(
          conditionTensor, lhsTensor, getLoc(), "condition", "LHS")))
    return failure();
  if (failed(detail::verifyTensorShapesCompatible(
          conditionTensor, rhsTensor, getLoc(), "condition", "RHS")))
    return failure();
  if (failed(detail::verifyTensorShapesCompatible(
          conditionTensor, resultTensor, getLoc(), "condition", "result")))
    return failure();

  return success();
}

//-----------------------------------------------------------------------------
// SelfIndexOp
//-----------------------------------------------------------------------------

LogicalResult wave::SelfIndexOp::verify() {
  Type elementType = getElementType(getResult().getType());

  if (!isa<IntegerType>(elementType))
    return emitOpError() << "result element type must be an integer type, got "
                         << elementType;

  auto tensorType = dyn_cast<WaveTensorType>(getResult().getType());
  if (!tensorType)
    return success();

  if (!tensorType.getFullySpecified())
    return success();

  if (tensorType.getRank() != 1)
    return emitOpError() << "result must be a 1-dimensional tensor, got rank "
                         << tensorType.getRank();

  if (tensorType.getShape()[0] != getDim())
    return emitOpError() << "result dimension '"
                         << tensorType.getShape()[0].getName()
                         << "' must match the specified dimension '"
                         << getDim().getName() << "'";

  return verifyIndexElementsPerThread(
      getOperation(), getIndexAttr(), getElementsPerThread(),
      dyn_cast<WaveTensorType>(getResult().getType()), getResult().getType());
}

//-----------------------------------------------------------------------------
// BroadcastOp
//-----------------------------------------------------------------------------

llvm::SmallVector<WaveSymbolAttr> wave::BroadcastOp::inferBroadcastDims() {
  if (auto dims = getBroadcastDims())
    return llvm::to_vector(dims->getAsRange<WaveSymbolAttr>());

  WaveTensorType sourceType = llvm::cast<WaveTensorType>(getSource().getType());
  WaveTensorType resultType = llvm::cast<WaveTensorType>(getResult().getType());
  assert(sourceType.getFullySpecified() && resultType.getFullySpecified() &&
         "expected source and result types to be fully specified");

  llvm::DenseSet<WaveSymbolAttr> sourceSymbols;
  for (WaveSymbolAttr sym : sourceType.getShape())
    sourceSymbols.insert(sym);

  llvm::SmallVector<WaveSymbolAttr> broadcastDims;
  for (WaveSymbolAttr sym : resultType.getShape()) {
    if (!sourceSymbols.contains(sym))
      broadcastDims.push_back(sym);
  }
  return broadcastDims;
}

LogicalResult wave::BroadcastOp::verify() {
  if (failed(detail::verifyElementTypesMatch(getLoc(), "source",
                                             getSource().getType(), "result",
                                             getResult().getType())))
    return failure();

  auto sourceType = llvm::dyn_cast<WaveTensorType>(getSource().getType());
  auto resultType = llvm::dyn_cast<WaveTensorType>(getResult().getType());

  if (!sourceType || !resultType)
    return success();

  if (!sourceType.getFullySpecified() || !resultType.getFullySpecified())
    return success();

  if (getBroadcastDims()) {
    return emitOpError("does not expect explicit dims when source and result "
                       "types are fully specified");
  }

  // Check all source symbols are in result.
  for (WaveSymbolAttr sym : sourceType.getShape()) {
    if (!llvm::is_contained(resultType.getShape(), sym))
      return emitOpError("source dimension '")
             << sym.getName() << "' not found in result shape";
  }

  return success();
}

llvm::FailureOr<ChangeResult> wave::BroadcastOp::propagateForward(
    llvm::ArrayRef<WaveTensorType> operandTypes,
    llvm::MutableArrayRef<WaveTensorType> resultTypes,
    llvm::raw_ostream &errs) {
  if (!getBroadcastDims())
    return ChangeResult::NoChange;

  unsigned operandNo = getSourceMutable().getOperandNumber();
  WaveTensorType operandType = operandTypes[operandNo];
  SmallVector<WaveSymbolAttr> broadcastDims =
      llvm::to_vector(getBroadcastDims()->getAsRange<WaveSymbolAttr>());
  return detail::propagateShapeAddTrailingDims(
      operandType, resultTypes[0], "operand", "result", broadcastDims, errs);
}

llvm::FailureOr<ChangeResult> wave::BroadcastOp::propagateBackward(
    llvm::MutableArrayRef<WaveTensorType> operandTypes,
    llvm::ArrayRef<WaveTensorType> resultTypes, llvm::raw_ostream &errs) {
  if (!getBroadcastDims())
    return ChangeResult::NoChange;

  WaveTensorType resultType = resultTypes[0];
  unsigned operandNo = getSourceMutable().getOperandNumber();
  return detail::propagateShapeDropTrailingDims(
      resultType, operandTypes[operandNo], "result", "operand",
      getBroadcastDims()->size(), errs);
}

LogicalResult wave::BroadcastOp::finalizeTypeInference() {
  if (cast<WaveTensorType>(getOperand().getType()).getFullySpecified() &&
      cast<WaveTensorType>(getResult().getType()).getFullySpecified())
    removeBroadcastDimsAttr();
  return success();
}

// Check if the broadcast operation is broadcasting along the thread X
// dimension. Returns true if types are fully specified and one of the
// broadcast dims matches the thread X dimension, meaning EPT propagation
// should be blocked.
static bool isBroadcastingAlongThreadX(wave::BroadcastOp op,
                                       const ElementsPerThreadInit &init) {
  auto sourceType = llvm::cast<WaveTensorType>(op.getSource().getType());
  auto resultType = llvm::cast<WaveTensorType>(op.getResult().getType());
  assert(sourceType.getFullySpecified() && resultType.getFullySpecified() &&
         "expected source and result types to be fully specified");

  SmallVector<WaveSymbolAttr> broadcastDims = op.inferBroadcastDims();
  return llvm::any_of(broadcastDims, llvm::equal_to(init.threadXDimension));
}

llvm::FailureOr<ChangeResult>
wave::BroadcastOp::propagateElementsPerThreadForward(
    llvm::ArrayRef<ElementsPerThreadLatticeValue> operandElements,
    llvm::MutableArrayRef<ElementsPerThreadLatticeValue> resultElements,
    llvm::raw_ostream &errs, const ElementsPerThreadInit &init) {
  if (isBroadcastingAlongThreadX(*this, init))
    return ChangeResult::NoChange;
  return detail::identityElementsPerThreadPropagate(
      operandElements, resultElements, "operands", "results", errs);
}

llvm::FailureOr<ChangeResult>
wave::BroadcastOp::propagateElementsPerThreadBackward(
    llvm::MutableArrayRef<ElementsPerThreadLatticeValue> operandElements,
    llvm::ArrayRef<ElementsPerThreadLatticeValue> resultElements,
    llvm::raw_ostream &errs, const ElementsPerThreadInit &init) {
  if (isBroadcastingAlongThreadX(*this, init))
    return ChangeResult::NoChange;
  return detail::identityElementsPerThreadPropagate(
      resultElements, operandElements, "results", "operands", errs);
}

//-----------------------------------------------------------------------------
// PermuteOp
//-----------------------------------------------------------------------------

/// Helper to validate the input type of a permute operation.
/// Checks if the input shape is a permutation of the result shape.
static LogicalResult validatePermutationInput(WaveTensorType inputType,
                                              WaveTensorType resultType,
                                              llvm::raw_ostream &errs) {
  // We cannot validate unspecified types.
  if (!inputType.getFullySpecified() || !resultType.getFullySpecified())
    return llvm::success();

  if (inputType.getShape().size() != resultType.getShape().size()) {
    errs << "input shape rank (" << inputType.getShape().size()
         << ") does not match target shape rank ("
         << resultType.getShape().size() << ")";
    return failure();
  }

  llvm::SmallDenseSet<WaveSymbolAttr, 4> resultShapeSet;
  resultShapeSet.insert_range(resultType.getShape());

  for (auto inputDim : inputType.getShape()) {
    if (!resultShapeSet.contains(inputDim)) {
      errs << "input dimension '" << inputDim.getName()
           << "' is not present in result shape";
      return failure();
    }
  }

  return llvm::success();
}

LogicalResult wave::PermuteOp::verify() {
  Value input = getValue();
  Value result = getResult();

  if (failed(detail::verifyElementTypesMatch(getLoc(), "input", input.getType(),
                                             "result", result.getType())))
    return failure();

  auto inputType = dyn_cast<WaveTensorType>(input.getType());
  auto resultType = dyn_cast<WaveTensorType>(result.getType());

  // If result / input is a vector (post-lowering phase), skip wave tensor
  // checks.
  if (!resultType || !inputType)
    return success();

  if (!inputType.getFullySpecified() || !resultType.getFullySpecified())
    return success();

  std::string errorMessage;
  llvm::raw_string_ostream errs(errorMessage);
  if (failed(validatePermutationInput(inputType, resultType, errs))) {
    return emitOpError() << errorMessage;
  }

  return success();
}

llvm::FailureOr<ChangeResult> wave::PermuteOp::propagateForward(
    llvm::ArrayRef<wave::WaveTensorType> operandTypes,
    llvm::MutableArrayRef<wave::WaveTensorType> resultTypes,
    llvm::raw_ostream &errs) {
  unsigned inputOperandPosition = getValueMutable().getOperandNumber();
  WaveTensorType inputType = operandTypes[inputOperandPosition];
  WaveTensorType &resultType = resultTypes[0];

  // Skip validation if either type is not fully specified.
  if (!inputType || !inputType.getFullySpecified() || !resultType ||
      !resultType.getFullySpecified())
    return ChangeResult::NoChange;

  if (failed(validatePermutationInput(inputType, resultType, errs)))
    return llvm::failure();

  return ChangeResult::NoChange;
}

llvm::FailureOr<ChangeResult> wave::PermuteOp::propagateBackward(
    llvm::MutableArrayRef<wave::WaveTensorType> operandTypes,
    llvm::ArrayRef<wave::WaveTensorType> resultTypes, llvm::raw_ostream &errs) {
  unsigned inputOperandPosition = getValueMutable().getOperandNumber();
  WaveTensorType inputType = operandTypes[inputOperandPosition];
  WaveTensorType resultType = resultTypes[0];

  if (!resultType || !resultType.getFullySpecified() || !inputType ||
      !inputType.getFullySpecified())
    return ChangeResult::NoChange;

  if (failed(validatePermutationInput(inputType, resultType, errs)))
    return llvm::failure();

  // Cannot propagate shape information backward for permute operations
  // because the input shape ordering is not determined by the result.
  return ChangeResult::NoChange;
}

// Helper to permute strides in an index expressions lattice according to
// the permutation from source shape to target shape.
//
// The permute operation swaps the strides of the permuted indices.
// For example, if we have a permute operation that swaps [B, M, N] to
// [M, N, B], then for each dimension k, we keep its start and step,
// but take the stride from the dimension at the same position in target_shape.
static IndexExprsLatticeStorage
permuteIndexExprsStrides(const IndexExprsLatticeStorage &inputLattice,
                         llvm::ArrayRef<wave::WaveSymbolAttr> srcShape,
                         llvm::ArrayRef<wave::WaveSymbolAttr> targetShape,
                         MLIRContext *ctx, wave::EmitErrorFn emitError) {
  if (inputLattice.isBottom() || inputLattice.isTop())
    return inputLattice;

  assert(srcShape.size() == targetShape.size() &&
         "source shape rank does not match target shape rank");

  DictionaryAttr inputDict = inputLattice.getConcreteValue();

  llvm::StringMap<WaveIndexMappingAttr> symbolToMapping;
  for (NamedAttribute namedAttr : inputDict) {
    if (auto mapping =
            llvm::dyn_cast<WaveIndexMappingAttr>(namedAttr.getValue())) {
      symbolToMapping[namedAttr.getName().getValue()] = mapping;
    }
  }

  // Create the permuted index expressions.
  // For each dimension k in src_shape:
  //   - Keep start and step from the original mapping for k
  //   - Take stride from the mapping for src_to_target[k]
  SmallVector<NamedAttribute> permutedMappings;
  permutedMappings.reserve(srcShape.size());
  for (auto [srcSymbol, targetSymbol] :
       llvm::zip_equal(srcShape, targetShape)) {
    llvm::StringRef srcName = srcSymbol.getName();
    auto srcMappingIt = symbolToMapping.find(srcName);

    llvm::StringRef targetName = targetSymbol.getName();
    auto targetMappingIt = symbolToMapping.find(targetName);

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

    permutedMappings.push_back(
        NamedAttribute(StringAttr::get(ctx, srcName), newMapping));
  }

  return IndexExprsLatticeStorage(DictionaryAttr::get(ctx, permutedMappings));
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

llvm::LogicalResult wave::PermuteOp::setIndexFromLattices(
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> operandExprs,
    llvm::ArrayRef<wave::IndexExprsLatticeStorage> resultExprs) {
  return detail::identitySetIndexFromLattices(*this, operandExprs, resultExprs);
}

llvm::LogicalResult wave::PermuteOp::finalizeTypeInference() {
  return llvm::success();
}
