// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/Transforms/LoweringPatterns.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "water/Dialect/Wave/IR/WaveOps.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// AllocateOp
//===----------------------------------------------------------------------===//

namespace {

class AllocateOpLoweringPattern : public OpConversionPattern<wave::AllocateOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wave::AllocateOp op, wave::AllocateOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    wave::WaveTensorType resultType = op.getResult().getType();
    wave::WaveExprListAttr distributedShape = op.getDistributedShape();
    auto *typeConverter =
        static_cast<const wave::WaveTypeConverter *>(getTypeConverter());
    Type convertedResultType = typeConverter->convertTensorFromComponents(
        distributedShape.getSymbols(), distributedShape.getMap(),
        resultType.getElementType(), resultType.getAddressSpaceValue());
    if (!convertedResultType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to construct resulting type");
    }

    auto memrefType = dyn_cast<MemRefType>(convertedResultType);
    if (!memrefType)
      return rewriter.notifyMatchFailure(
          op, "expected memref type for allocate op");
    Location loc = op.getLoc();

    // If operation contains a parent op, emit a view into this parent
    // allocation
    Value parent = adaptor.getParent();
    int64_t byteOffset = 0;
    if (parent) {
      // We need to ignore the automatically inserted cast. It is due to the
      // fact that the memref shape coming out of the allocation has a shape
      // different from one that would be obtained by converting the result
      // type... This is an annoying repercussion of the strange design choice
      // in original Wave of having "allocate" represent both allocations and
      // views, which makes it impossible to have a consistent result type: for
      // allocations, the shape is the distributed shape, but for views it is
      // the result shape.
      auto cast = parent.getDefiningOp<UnrealizedConversionCastOp>();
      assert(cast && "unexpected type conversion configuration");

      byteOffset = static_cast<int64_t>(*op.getOffset());
      mlir::Value off =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, byteOffset);
      rewriter.replaceOpWithNewOp<mlir::memref::ViewOp>(
          op, memrefType, cast.getOperand(0), off, mlir::ValueRange());

      return success();
    }

    // No parent : emit plain memref.alloc
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, memrefType);

    return success();
  }
};

} // namespace

void wave::populateWaveAllocateOpLoweringPatterns(
    WaveTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<AllocateOpLoweringPattern>(typeConverter, patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Binary ops
//===----------------------------------------------------------------------===//

namespace {

/// Lowers binary ops into arith counterpart.
template <typename WaveOp, typename FloatOp, typename IntOp>
class BinaryOpLoweringPattern : public OpConversionPattern<WaveOp> {
public:
  using OpConversionPattern<WaveOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WaveOp op, typename WaveOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedType) {
      return rewriter.notifyMatchFailure(op,
                                         "WaveTensorType conversion failed");
    }
    auto vectorType = cast<VectorType>(convertedType);

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Type elementType = vectorType.getElementType();

    if (isa<FloatType>(elementType)) {
      rewriter.replaceOpWithNewOp<FloatOp>(op, vectorType, lhs, rhs);
    } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
      rewriter.replaceOpWithNewOp<IntOp>(op, vectorType, lhs, rhs);
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }

    return success();
  }
};

} // namespace

void wave::populateWaveBinaryOpLoweringPatterns(
    WaveTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns
      .add<BinaryOpLoweringPattern<wave::AddOp, arith::AddFOp, arith::AddIOp>,
           BinaryOpLoweringPattern<wave::MulOp, arith::MulFOp, arith::MulIOp>,
           BinaryOpLoweringPattern<wave::DivOp, arith::DivFOp, arith::DivSIOp>>(
          typeConverter, patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Unary ops
//===----------------------------------------------------------------------===//

namespace {

/// Generic lowering for Wave unary ops to a target unary op operating on a
/// float-like operand.
template <typename WaveOp, typename TargetOp>
class UnaryFPOpLoweringPattern : public OpConversionPattern<WaveOp> {
public:
  using OpConversionPattern<WaveOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WaveOp op, typename WaveOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedType)
      return rewriter.notifyMatchFailure(
          op, "type conversion failed for unary op result");

    // Require float-like result (scalar float, vector-of-float, or
    // tensor-of-float).
    Type elemType = mlir::getElementTypeOrSelf(convertedType);
    if (!isa<FloatType>(elemType))
      return rewriter.notifyMatchFailure(
          op, "unary op requires float-like result type after conversion");

    rewriter.replaceOpWithNewOp<TargetOp>(op, convertedType,
                                          adaptor.getArgument());
    return success();
  }
};

} // namespace

void wave::populateWaveUnaryFPOpLoweringPatterns(
    WaveTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<UnaryFPOpLoweringPattern<wave::Exp2Op, mlir::math::Exp2Op>>(
      typeConverter, patterns.getContext());
}

//===----------------------------------------------------------------------===//
// RegisterOp
//===----------------------------------------------------------------------===//

namespace {

/// Lowers `wave.register` into a constant vector to avoid memory allocation.
class RegisterOpLoweringPattern : public OpConversionPattern<wave::RegisterOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wave::RegisterOp op, wave::RegisterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedType) {
      return rewriter.notifyMatchFailure(op,
                                         "WaveTensorType conversion failed");
    }
    auto vectorType = cast<VectorType>(convertedType);

    TypedAttr splatAttr;
    Value initValue = op.getInit();
    Type elementType = vectorType.getElementType();

    if (isa<FloatType>(elementType)) {
      if (auto cst = initValue.getDefiningOp<arith::ConstantFloatOp>()) {
        splatAttr = DenseFPElementsAttr::get(vectorType, cst.value());
      } else {
        return rewriter.notifyMatchFailure(op,
                                           "init value must be constant float");
      }
    } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
      if (auto cst = initValue.getDefiningOp<arith::ConstantIntOp>()) {
        if (!intType.isSignless()) {
          return rewriter.notifyMatchFailure(
              op, "only signless integer types are supported");
        }
        APInt valueInt(elementType.getIntOrFloatBitWidth(), cst.value(),
                       /*isSigned=*/true);
        splatAttr = DenseIntElementsAttr::get(vectorType, valueInt);
      } else {
        return rewriter.notifyMatchFailure(
            op, "init value must be constant integer");
      }
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }

    // Create vector.constant operation
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, vectorType, splatAttr);

    return success();
  }
};

} // namespace

void wave::populateWaveRegisterLoweringPatterns(
    WaveTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<RegisterOpLoweringPattern>(typeConverter, patterns.getContext());
}
