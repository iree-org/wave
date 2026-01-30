// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/Transforms/LoweringPatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "water/Dialect/Wave/IR/WaveOps.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// materializeAffine
//===----------------------------------------------------------------------===//

/// Find parent operation of type OpTy starting from the given block.
template <typename OpTy> static OpTy findParentOfType(Block *currentBlock) {
  auto parentOp = currentBlock->getParentOp();
  if (auto op = dyn_cast<OpTy>(parentOp)) {
    return op;
  }
  return parentOp->getParentOfType<OpTy>();
}

FailureOr<SmallVector<Value>>
wave::materializeAffine(Location loc, ArrayRef<Attribute> symbols,
                        AffineMap map, PatternRewriter &rewriter,
                        wave::WaveHyperparameterAttr hyper) {
  // NOTE: This helper assumes 0 dims in `map`. If you add dims, prepend
  // the dim operands before the symbol operands below.
  assert(map.getNumDims() == 0 && "expected 0 dims");

  auto threadId = [&](gpu::Dimension d) -> Value {
    return gpu::ThreadIdOp::create(rewriter, loc, rewriter.getIndexType(), d);
  };
  auto blockId = [&](gpu::Dimension d) -> Value {
    return gpu::BlockIdOp::create(rewriter, loc, rewriter.getIndexType(), d);
  };

  SmallVector<Value> baseSymVals;
  baseSymVals.reserve(map.getNumSymbols());
  for (Attribute attr : symbols) {
    if (auto symbol = dyn_cast<wave::WaveSymbolAttr>(attr)) {
      StringRef name = symbol.getName();
      std::optional<int64_t> value = hyper.getSymbolValue(name);
#ifndef NDEBUG
      if (!value) {
        llvm::errs() << "symbol: " << name << "\n";
        assert(false && "unknown symbol, should have been caught by verifiers");
      }
#endif
      baseSymVals.emplace_back(
          arith::ConstantIndexOp::create(rewriter, loc, *value));
      continue;
    }

    if (auto indexSymbol = dyn_cast<wave::WaveIndexSymbolAttr>(attr)) {
      switch (indexSymbol.getValue()) {
      case wave::WaveIndexSymbol::THREAD_0:
        baseSymVals.emplace_back(threadId(gpu::Dimension::x));
        break;
      case wave::WaveIndexSymbol::THREAD_1:
        baseSymVals.emplace_back(threadId(gpu::Dimension::y));
        break;
      case wave::WaveIndexSymbol::THREAD_2:
        baseSymVals.emplace_back(threadId(gpu::Dimension::z));
        break;
      case wave::WaveIndexSymbol::WORKGROUP_0:
        baseSymVals.emplace_back(blockId(gpu::Dimension::x));
        break;
      case wave::WaveIndexSymbol::WORKGROUP_1:
        baseSymVals.emplace_back(blockId(gpu::Dimension::y));
        break;
      case wave::WaveIndexSymbol::WORKGROUP_2:
        baseSymVals.emplace_back(blockId(gpu::Dimension::z));
        break;
      case wave::WaveIndexSymbol::DEVICE_DIM_0:
      case wave::WaveIndexSymbol::DEVICE_DIM_1:
      case wave::WaveIndexSymbol::DEVICE_DIM_2:
        return rewriter.notifyMatchFailure(
            loc, "materialization of affine expressions containing device "
                 "dimension symbols is not implemented.");
      case wave::WaveIndexSymbol::GPR_NUMBER:
        return rewriter.notifyMatchFailure(
            loc, "materialization of affine expressions containing gpr number "
                 "symbols is not implemented.");
      }
      continue;
    }

    if (auto iterSymbol = dyn_cast<wave::WaveIterSymbolAttr>(attr)) {
      // Check if we're inside an scf.for loop that corresponds to this
      // iteration symbol.
      Block *currentBlock = rewriter.getInsertionBlock();

      if (findParentOfType<wave::IterateOp>(currentBlock)) {
        return rewriter.notifyMatchFailure(
            loc, "iteration symbol found inside wave.iterate - "
                 "please run lower-wave-control-flow pass first");
      }

      scf::ForOp parentFor = findParentOfType<scf::ForOp>(currentBlock);
      assert(parentFor &&
             "iteration symbol found but no iteration context available");

      // Get the induction variable from the scf.for loop.
      Value inductionVar = parentFor.getInductionVar();

      // Pass the induction variable directly to the affine map. The index
      // expressions are designed as affine maps that already incorporate tile
      // size scaling. Pre-multiplying here would cause double multiplication
      // when the affine map applies its own scaling.  For example, if the map
      // is (s0 * 32), it expects s0 = iteration, not s0 = iteration *
      // tile_size.
      baseSymVals.push_back(inductionVar);
      continue;
    }
  }

  // In case map contains multiple results, create one apply per result.
  SmallVector<Value> results;
  results.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults()) {
    AffineMap submap =
        AffineMap::get(map.getNumDims(), map.getNumSymbols(), expr);
    SmallVector<Value> symVals = baseSymVals;

    affine::canonicalizeMapAndOperands(&submap, &symVals);

    Value apply = affine::AffineApplyOp::create(rewriter, loc, submap, symVals);
    results.push_back(apply);
  }

  return results;
}

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
    // ResolveDistributedAllocations must run before this pass to convert
    // the result type from WaveTensorType to MemRefType.
    auto memrefType = dyn_cast<MemRefType>(op.getResult().getType());
    if (!memrefType)
      return rewriter.notifyMatchFailure(
          op, "expected memref type; ResolveDistributedAllocations must run "
              "before lowering");
    Location loc = op.getLoc();

    // If operation contains a parent op, emit a view into this parent
    // allocation.
    Value parent = adaptor.getParent();
    if (parent) {
      int64_t byteOffset = static_cast<int64_t>(*op.getOffset());
      Value off = arith::ConstantIndexOp::create(rewriter, loc, byteOffset);
      rewriter.replaceOpWithNewOp<memref::ViewOp>(op, memrefType, parent, off,
                                                  ValueRange());
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
           BinaryOpLoweringPattern<wave::SubOp, arith::SubFOp, arith::SubIOp>,
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
    Type elemType = getElementTypeOrSelf(convertedType);
    if (!isa<FloatType>(elemType))
      return rewriter.notifyMatchFailure(
          op, "unary op requires float-like result type after conversion");

    rewriter.replaceOpWithNewOp<TargetOp>(op, convertedType,
                                          adaptor.getArgument());
    return success();
  }
};

/// Lowers ReciprocalOp to 1.0 / x using arith.divf.
class ReciprocalOpLoweringPattern
    : public OpConversionPattern<wave::ReciprocalOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wave::ReciprocalOp op, wave::ReciprocalOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedType)
      return rewriter.notifyMatchFailure(
          op, "type conversion failed for reciprocal op result");

    auto vectorType = cast<VectorType>(convertedType);

    Location loc = op.getLoc();
    Type elemType = vectorType.getElementType();

    FloatAttr oneAttr = rewriter.getFloatAttr(elemType, 1.0);
    DenseElementsAttr splatAttr = SplatElementsAttr::get(vectorType, oneAttr);

    Value one = arith::ConstantOp::create(rewriter, loc, vectorType, splatAttr);
    rewriter.replaceOpWithNewOp<arith::DivFOp>(op, vectorType, one,
                                               adaptor.getArgument());
    return success();
  }
};

} // namespace

void wave::populateWaveUnaryFPOpLoweringPatterns(
    WaveTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<UnaryFPOpLoweringPattern<wave::Exp2Op, math::Exp2Op>,
               ReciprocalOpLoweringPattern>(typeConverter,
                                            patterns.getContext());
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

/// Lowers ExtractOp to vector.extract + vector.broadcast.
/// Supports both static and dynamic positions.
class ExtractOpLoweringPattern : public OpConversionPattern<wave::ExtractOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wave::ExtractOp op, wave::ExtractOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedResultType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    auto resultVectorType = cast<VectorType>(convertedResultType);

    Value source = adaptor.getSource();
    wave::WaveExprListAttr positionAttr = op.getPosition();
    auto *waveTypeConverter =
        static_cast<const wave::WaveTypeConverter *>(getTypeConverter());
    wave::WaveHyperparameterAttr hyper =
        waveTypeConverter->getHyperparameters();

    FailureOr<SmallVector<Value>> positions = wave::materializeAffine(
        loc, positionAttr.getSymbols(), positionAttr.getMap(), rewriter, hyper);
    if (failed(positions))
      return rewriter.notifyMatchFailure(
          op, "failed to materialize position expression");

    // Use dynamic extract with kDynamic sentinel. Constants will fold later.
    SmallVector<int64_t> staticPositions = {ShapedType::kDynamic};
    Value element = vector::ExtractOp::create(rewriter, loc, source, *positions,
                                              staticPositions);

    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, resultVectorType,
                                                     element);
    return success();
  }
};

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

//===----------------------------------------------------------------------===//
// IterateOp
//===----------------------------------------------------------------------===//

/// Lower `wave.iterate` to `scf.for`.
class IterateOpLoweringPattern : public OpConversionPattern<wave::IterateOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wave::IterateOp op, wave::IterateOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get hyperparameters from type converter.
    auto *waveTypeConverter =
        static_cast<const wave::WaveTypeConverter *>(getTypeConverter());
    wave::WaveHyperparameterAttr hyperparams =
        waveTypeConverter->getHyperparameters();

    // Get the iterator symbol (e.g., "K").
    wave::WaveSymbolAttr iteratorSymbol = op.getIterator();
    StringRef symbolName = iteratorSymbol.getName();

    // Find the tiling constraint for this iterator symbol.
    func::FuncOp parentFunc = op->getParentOfType<func::FuncOp>();
    if (!parentFunc) {
      return rewriter.notifyMatchFailure(op, "iterate op not in function");
    }

    // Look for tiling constraints in function attributes.
    ArrayAttr constraints = parentFunc->getAttrOfType<ArrayAttr>(
        wave::WaveDialect::kWaveConstraintsAttrName);
    if (!constraints) {
      return rewriter.notifyMatchFailure(
          op, "no wave constraints found in function");
    }

    // Get the dimension size (e.g., K = 640) from hyperparameters.
    std::optional<SmallVector<int64_t>> resolvedDims =
        wave::resolveSymbolNames(iteratorSymbol, hyperparams);
    if (!resolvedDims || resolvedDims->size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "iterator symbol not found in hyperparameters");
    }
    int64_t dimSize = resolvedDims->front();

    // Find tiling constraint for this dimension to get tile_size.
    std::optional<int64_t> tileSize;
    for (Attribute constraintAttr : constraints) {
      auto tilingConstraint =
          dyn_cast<wave::TilingConstraintAttr>(constraintAttr);
      if (!tilingConstraint)
        continue;

      wave::WaveSymbolAttr constraintDim = tilingConstraint.getDim();
      if (constraintDim.getName() != symbolName)
        continue;

      wave::WaveExprListAttr tileSizeAttr = tilingConstraint.getTileSize();
      AffineMap tileSizeMap = tileSizeAttr.getMap();
      ArrayRef<Attribute> tileSizeSymbols = tileSizeAttr.getSymbols();

      // Evaluate the tile size using hyperparameters.
      std::optional<SmallVector<int64_t>> evaluatedTileSize =
          wave::evaluateMapWithHyperparams(tileSizeMap, tileSizeSymbols,
                                           hyperparams);
      if (!evaluatedTileSize) {
        return rewriter.notifyMatchFailure(
            op, "failed to evaluate tile size from tiling constraint");
      }
      if (evaluatedTileSize->size() != 1) {
        return rewriter.notifyMatchFailure(op,
                                           "tile size must be single value");
      }
      tileSize = (*evaluatedTileSize)[0];
      break;
    }

    if (!tileSize) {
      return rewriter.notifyMatchFailure(
          op, "no tiling constraint found for iterator symbol");
    }

    // TODO(tyb): we reject non-exact division for now, which should require
    // peeling or padding to be correct.
    // TODO(tyb): make these errors better visible to the caller from python.
    if (*tileSize == 0) {
      return rewriter.notifyMatchFailure(op, "tile size cannot be zero");
    }
    if (dimSize % *tileSize != 0) {
      return op.emitOpError("non-exact division not supported to prevent "
                            "potential out-of-bounds access");
    }
    int64_t numIterations = dimSize / *tileSize;

    // Create loop bounds.
    Value lowerBound = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value upperBound =
        arith::ConstantIndexOp::create(rewriter, loc, numIterations);
    Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);

    // Create the scf.for loop.
    auto forOp = scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step,
                                    adaptor.getIterArgs());

    // Convert the body.
    Block &waveBody = op.getBody().front();
    Block &scfBody = *forOp.getBody();

    // Set up insertion point inside the loop body.
    rewriter.setInsertionPointToStart(&scfBody);

    // Create mapping from old block arguments to new ones.
    IRMapping mapping;

    // Map iter_args and captures separately.
    // Note: wave.iterate doesn't expose the induction variable, so we skip it.
    size_t numIterArgs = adaptor.getIterArgs().size();

    // Map iter_args: wave iter_args -> scf.for iter_args (skip induction var)
    for (auto [waveIterArg, scfIterArg] :
         llvm::zip_equal(waveBody.getArguments().take_front(numIterArgs),
                         scfBody.getArguments().drop_front())) {
      mapping.map(waveIterArg, scfIterArg);
    }

    // Map captures: wave capture args -> original capture SSA values
    for (auto [waveCaptureArg, originalCapture] :
         llvm::zip_equal(waveBody.getArguments().drop_front(numIterArgs),
                         adaptor.getCaptures())) {
      mapping.map(waveCaptureArg, originalCapture);
    }

    // Clone all operations except the terminator.
    for (Operation &bodyOp : waveBody.without_terminator()) {
      rewriter.clone(bodyOp, mapping);
    }

    // Convert wave.yield to scf.yield.
    auto yieldOp = cast<wave::YieldOp>(waveBody.getTerminator());
    SmallVector<Value> yieldValues;
    yieldValues.reserve(yieldOp.getValues().size());
    for (Value value : yieldOp.getValues()) {
      yieldValues.push_back(mapping.lookup(value));
    }
    scf::YieldOp::create(rewriter, yieldOp.getLoc(), yieldValues);

    // Replace the original op with the for loop results.
    rewriter.replaceOp(op, forOp.getResults());

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ShuffleOp
//===----------------------------------------------------------------------===//

namespace {

/// Convert WaveShuffleMode to gpu::ShuffleMode.
static gpu::ShuffleMode convertShuffleMode(wave::WaveShuffleMode mode) {
  switch (mode) {
  case wave::WaveShuffleMode::XOR:
    return gpu::ShuffleMode::XOR;
  case wave::WaveShuffleMode::DOWN:
    return gpu::ShuffleMode::DOWN;
  case wave::WaveShuffleMode::UP:
    return gpu::ShuffleMode::UP;
  case wave::WaveShuffleMode::IDX:
    return gpu::ShuffleMode::IDX;
  }
  llvm_unreachable("unknown shuffle mode");
}

class ShuffleOpLoweringPattern : public OpConversionPattern<wave::ShuffleOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wave::ShuffleOp op, wave::ShuffleOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type convertedType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedType)
      return rewriter.notifyMatchFailure(op,
                                         "WaveTensorType conversion failed");

    Value input = adaptor.getValue();
    int32_t offset = op.getOffset();
    int32_t width = op.getWidth();
    wave::WaveShuffleMode mode = op.getMode();

    Value offsetValue =
        arith::ConstantIntOp::create(rewriter, loc, offset, /*width=*/32);
    Value widthValue =
        arith::ConstantIntOp::create(rewriter, loc, width, /*width=*/32);

    gpu::ShuffleMode gpuMode = convertShuffleMode(mode);

    auto shuffleOp = gpu::ShuffleOp::create(rewriter, loc, input, offsetValue,
                                            widthValue, gpuMode);

    rewriter.replaceOp(op, shuffleOp.getShuffleResult());
    return success();
  }
};

} // namespace

void wave::populateWaveMiscellaneousOpsLoweringPatterns(
    WaveTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<CastOpLoweringPattern, ExtractOpLoweringPattern,
               ExtractSliceOpLoweringPattern, IterateOpLoweringPattern,
               RegisterOpLoweringPattern, ShuffleOpLoweringPattern>(
      typeConverter, patterns.getContext());
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

    std::optional<wave::WaveMmaKind> kind = op.getKind();
    if (!kind)
      return rewriter.notifyMatchFailure(
          op, "mma operation without kind attribute");

    auto [M, N, K] =
        wave::WaveMmaKindAttr::getShape(rewriter.getContext(), *kind);

    // TODO: Extend lowering for ops beyond MFMA, e.g. WMMA
    auto mfma = amdgpu::MFMAOp::create(rewriter, loc, acc.getType(),
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

//===----------------------------------------------------------------------===//
// Reduction ops (SumOp, MaxElementOp)
//===----------------------------------------------------------------------===//

namespace {

/// Convert vector::CombiningKind to gpu::AllReduceOperation.
constexpr gpu::AllReduceOperation
combiningKindToAllReduceOp(vector::CombiningKind kind) {
  switch (kind) {
  case vector::CombiningKind::ADD:
    return gpu::AllReduceOperation::ADD;
  case vector::CombiningKind::MAXIMUMF:
    return gpu::AllReduceOperation::MAXIMUMF;
  default:
    llvm_unreachable("unsupported reduction kind");
  }
}

/// Emit a warning if the block reduction setting is inconsistent with the
/// hardware constraint's waves_per_block.
static void warnIfReductionScopeMismatch(Operation *op, bool isBlockReduction) {
  func::FuncOp parentFunc = op->getParentOfType<func::FuncOp>();
  if (!parentFunc)
    return;
  ArrayAttr constraints = parentFunc->getAttrOfType<ArrayAttr>(
      wave::WaveDialect::kWaveConstraintsAttrName);
  if (!constraints)
    return;
  for (Attribute constraint : constraints) {
    auto hwConstraint =
        llvm::dyn_cast<wave::HardwareConstraintAttr>(constraint);
    if (!hwConstraint)
      continue;
    ArrayRef<unsigned> wavesPerBlock = hwConstraint.getWavesPerBlock();
    unsigned totalWaves = 1;
    for (unsigned w : wavesPerBlock)
      totalWaves *= w;
    if (isBlockReduction && totalWaves == 1) {
      op->emitWarning()
          << "block reduction requested but hardware constraint "
             "specifies only one wave per block (waves_per_block = ["
          << wavesPerBlock[0] << ", " << wavesPerBlock[1] << ", "
          << wavesPerBlock[2]
          << "]); consider using wave-level reduction instead";
    } else if (!isBlockReduction && totalWaves > 1) {
      op->emitWarning()
          << "wave-level reduction requested but hardware constraint "
             "specifies multiple waves per block (waves_per_block = ["
          << wavesPerBlock[0] << ", " << wavesPerBlock[1] << ", "
          << wavesPerBlock[2]
          << "]); consider using block reduction to reduce across all waves";
    }
    return;
  }
}

template <typename WaveOp, vector::CombiningKind Kind>
class ReductionOpLoweringPattern : public OpConversionPattern<WaveOp> {
public:
  using OpConversionPattern<WaveOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WaveOp op, typename WaveOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    static_assert(Kind == vector::CombiningKind::ADD ||
                      Kind == vector::CombiningKind::MAXIMUMF,
                  "unsupported reduction kind");
    // Expect PropagateElementsPerThread pass to have run, converting
    // WaveTensorType results to VectorType.
    Location loc = op.getLoc();

    Value input = adaptor.getInput();
    Value init = adaptor.getInit();
    bool isBlockReduction = op.getScope() == wave::WaveReductionScope::Block;

    // Warn if reduction scope is inconsistent with hardware constraints.
    warnIfReductionScopeMismatch(op, isBlockReduction);

    Value initElement = vector::ExtractOp::create(rewriter, loc, init, 0);
    Value threadReduce =
        vector::ReductionOp::create(rewriter, loc, Kind, input, initElement);

    constexpr gpu::AllReduceOperation gpuReduceOp =
        combiningKindToAllReduceOp(Kind);

    Value result;
    if (isBlockReduction) {
      auto opAttr =
          gpu::AllReduceOperationAttr::get(rewriter.getContext(), gpuReduceOp);
      result =
          gpu::AllReduceOp::create(rewriter, loc, threadReduce, opAttr, false);
    } else {
      result = gpu::SubgroupReduceOp::create(rewriter, loc, threadReduce,
                                             gpuReduceOp, false);
    }
    rewriter.replaceOp(op, result);

    return success();
  }
};

} // namespace

void wave::populateWaveReductionOpLoweringPatterns(
    WaveTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns
      .add<ReductionOpLoweringPattern<wave::SumOp, vector::CombiningKind::ADD>,
           ReductionOpLoweringPattern<wave::MaxElementOp,
                                      vector::CombiningKind::MAXIMUMF>>(
          typeConverter, patterns.getContext());
}
