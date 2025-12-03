// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/Transforms/LoweringPatterns.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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
          mlir::arith::ConstantIndexOp::create(rewriter, loc, byteOffset);
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

class CastOpLoweringPattern : public OpConversionPattern<wave::CastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wave::CastOp op, wave::CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getValueToCast();
    Type srcType = input.getType();
    Type dstType = getTypeConverter()->convertType(op.getResult().getType());

    if (!dstType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    // Only support 1-D vectors here.
    VectorType srcVecType = dyn_cast<VectorType>(srcType);
    VectorType dstVecType = dyn_cast<VectorType>(dstType);
    if (!srcVecType || !dstVecType)
      return rewriter.notifyMatchFailure(
          op, "expected vector types for cast lowering");

    Type srcElemType = srcVecType.getElementType();
    Type dstElemType = dstVecType.getElementType();

    Value castResult = nullptr;

    if (isa<FloatType>(srcElemType) && isa<FloatType>(dstElemType)) {
      // Floating point to floating point.
      unsigned srcWidth = srcElemType.getIntOrFloatBitWidth();
      unsigned dstWidth = dstElemType.getIntOrFloatBitWidth();
      if (dstWidth < srcWidth)
        castResult = arith::TruncFOp::create(rewriter, loc, dstVecType, input);
      else if (dstWidth > srcWidth)
        castResult = arith::ExtFOp::create(rewriter, loc, dstVecType, input);
      else
        castResult = input;
    } else if (isa<IntegerType>(srcElemType) && isa<IntegerType>(dstElemType)) {
      // Integer to integer.
      unsigned srcWidth = srcElemType.getIntOrFloatBitWidth();
      unsigned dstWidth = dstElemType.getIntOrFloatBitWidth();
      if (dstWidth < srcWidth)
        castResult = arith::TruncIOp::create(rewriter, loc, dstVecType, input);
      else if (dstWidth > srcWidth)
        castResult = arith::ExtSIOp::create(rewriter, loc, dstVecType, input);
      else
        castResult = input;
    } else if (isa<FloatType>(srcElemType) && isa<IntegerType>(dstElemType)) {
      // Float to integer.
      castResult = arith::FPToSIOp::create(rewriter, loc, dstVecType, input);
    } else if (isa<IntegerType>(srcElemType) && isa<FloatType>(dstElemType)) {
      // Integer to float.
      castResult = arith::SIToFPOp::create(rewriter, loc, dstVecType, input);
    } else {
      return rewriter.notifyMatchFailure(
          op, "unsupported cast element type combination");
    }

    rewriter.replaceOp(op, castResult);
    return success();
  }
};

/// Evaluate a Wave expression list to constant integer values.
/// Since the Wave dialect verifier ensures expressions contain no symbols,
/// this only handles pure constant expressions.
static FailureOr<SmallVector<int64_t>>
evaluateWaveExprList(wave::WaveExprListAttr exprListAttr) {
  AffineMap map = exprListAttr.getMap();
  if (!map.isConstant())
    return failure();

  return map.getConstantResults();
}

class ExtractSliceOpLoweringPattern
    : public OpConversionPattern<wave::ExtractSliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wave::ExtractSliceOp op,
                  wave::ExtractSliceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Evaluate offset, size, and stride expressions to constant values.
    FailureOr<SmallVector<int64_t>> offsetValues =
        evaluateWaveExprList(op.getOffset());
    if (failed(offsetValues))
      return rewriter.notifyMatchFailure(
          op, "failed to evaluate offset expression to constants");

    FailureOr<SmallVector<int64_t>> sizeValues =
        evaluateWaveExprList(op.getSize());
    if (failed(sizeValues))
      return rewriter.notifyMatchFailure(
          op, "failed to evaluate size expression to constants");

    FailureOr<SmallVector<int64_t>> strideValues =
        evaluateWaveExprList(op.getStride());
    if (failed(strideValues))
      return rewriter.notifyMatchFailure(
          op, "failed to evaluate stride expression to constants");

    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedResultType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    // Direct mapping from wave.extract_slice to vector.extract_strided_slice.
    // The offset, size, and stride values are copied directly.
    auto extractOp = vector::ExtractStridedSliceOp::create(
        rewriter, op.getLoc(), adaptor.getMemory(),
        ArrayRef<int64_t>(*offsetValues), ArrayRef<int64_t>(*sizeValues),
        ArrayRef<int64_t>(*strideValues));
    rewriter.replaceOp(op, extractOp);

    return success();
  }
};

} // namespace

void wave::populateWaveMiscellaneousOpsLoweringPatterns(
    WaveTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<RegisterOpLoweringPattern, CastOpLoweringPattern,
               ExtractSliceOpLoweringPattern>(typeConverter,
                                              patterns.getContext());
}

//===----------------------------------------------------------------------===//
// MmaOp
//===----------------------------------------------------------------------===//

namespace {

class MmaOpLoweringPattern : public OpConversionPattern<wave::MmaOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wave::MmaOp op, wave::MmaOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value acc = adaptor.getAccumulator();

    auto lhsType = cast<VectorType>(lhs.getType());
    auto rhsType = cast<VectorType>(rhs.getType());
    auto accType = cast<VectorType>(acc.getType());
    assert(lhsType.getRank() == 1 && rhsType.getRank() == 1 &&
           accType.getRank() == 1 &&
           "only 1-D vectors supported for mma lowering");

    wave::WaveMmaKind kind = op.getKind();
    auto [M, N, K] =
        wave::WaveMmaKindAttr::getShape(rewriter.getContext(), kind);

    // TODO: Extend lowering for ops beyond MFMA, e.g. WMMA
    auto mfma = mlir::amdgpu::MFMAOp::create(rewriter, loc, acc.getType(),
                                             /*m=*/M,
                                             /*n=*/N,
                                             /*k=*/K,
                                             /*blocks=*/1,
                                             /*sourceA=*/lhs, /*sourceB=*/rhs,
                                             /*destC=*/acc);
    rewriter.replaceOp(op, mfma.getResult());
    return success();
  }
};

} // namespace

void wave::populateWaveMmaLoweringPatterns(WaveTypeConverter &typeConverter,
                                           RewritePatternSet &patterns) {
  patterns.add<MmaOpLoweringPattern>(typeConverter, patterns.getContext());
}
