// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace mlir::water {
#define GEN_PASS_DEF_WATERMEMREFDECOMPOSITIONPASS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

namespace {

static Value getValue(OpBuilder &rewriter, Location loc, OpFoldResult in) {
  if (Attribute offsetAttr = dyn_cast<Attribute>(in)) {
    return arith::ConstantIndexOp::create(
        rewriter, loc, cast<IntegerAttr>(offsetAttr).getInt());
  }
  return cast<Value>(in);
}

static SmallVector<Value> getValues(OpBuilder &rewriter, Location loc,
                                    ArrayRef<OpFoldResult> in) {
  SmallVector<Value> result;
  for (OpFoldResult value : in)
    result.push_back(getValue(rewriter, loc, value));
  return result;
}

static SmallVector<Value> flatten(ArrayRef<ValueRange> values) {
  SmallVector<Value> result;
  for (ValueRange value : values)
    llvm::append_range(result, value);

  return result;
}

static std::tuple<Value, ValueRange, ValueRange>
unflattenDescriptor(ValueRange values, unsigned rank) {
  Value buffer = values.front();
  values = values.drop_front();
  ValueRange sizes = values.take_front(rank);
  values = values.drop_front(rank);
  ValueRange strides = values.take_front(rank);
  return {buffer, sizes, strides};
}

/// Type converter for memref decomposition.
/// Converts memref<NxMx...xT> to (memref<?xi8>, sizes..., strides...)
class MemrefDecompositionTypeConverter : public TypeConverter {
public:
  MemrefDecompositionTypeConverter() {
    // Keep all other types unchanged.
    addConversion([](Type type) { return type; });

    // Convert memref types to memref<?xi8> + sizes + strides (1-to-N
    // conversion).
    addConversion(
        [](MemRefType type,
           SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
          MLIRContext *ctx = type.getContext();
          auto indexType = IndexType::get(ctx);
          auto i8Type = IntegerType::get(ctx, 8);

          // Create the byte buffer type.
          auto byteMemrefType = MemRefType::get({ShapedType::kDynamic}, i8Type,
                                                MemRefLayoutAttrInterface{},
                                                type.getMemorySpace());

          results.push_back(byteMemrefType);
          unsigned rank = type.getRank();
          results.resize(1 + rank * 2, indexType);
          return success();
        });

    /// Source materialization to reconstruct memref from components.
    addSourceMaterialization([](OpBuilder &builder, MemRefType resultType,
                                ValueRange inputs, Location loc) -> Value {
      unsigned rank = resultType.getRank();
      if (inputs.size() != 1 + rank * 2)
        return {};

      if (!isa<MemRefType>(inputs.front().getType()))
        return {};

      if (!llvm::all_of(inputs.drop_front(), [](Value value) {
            return isa<IndexType>(value.getType());
          }))
        return {};

      auto bufferType = MemRefType::get({}, resultType.getElementType(),
                                        MemRefLayoutAttrInterface{},
                                        resultType.getMemorySpace());

      auto [buffer, sizes, strides] = unflattenDescriptor(inputs, rank);
      buffer =
          UnrealizedConversionCastOp::create(builder, loc, bufferType, buffer)
              .getResult(0);
      Value offset = arith::ConstantIndexOp::create(builder, loc, 0);
      return memref::ReinterpretCastOp::create(builder, loc, resultType, buffer,
                                               offset, sizes, strides);
    });

    /// Target materialization to decompose memref into components.
    addTargetMaterialization([](OpBuilder &builder, TypeRange resultType,
                                ValueRange inputs,
                                Location loc) -> SmallVector<Value> {
      if (inputs.size() != 1)
        return {};

      Value input = inputs.front();
      auto memrefType = dyn_cast<MemRefType>(input.getType());
      if (!memrefType)
        return {};

      unsigned rank = memrefType.getRank();
      if (resultType.size() != 1 + rank * 2)
        return {};

      if (!isa<MemRefType>(resultType.front()))
        return {};

      if (!llvm::all_of(resultType.drop_front(),
                        [](Type type) { return isa<IndexType>(type); }))
        return {};

      int64_t staticOffset = 0;
      SmallVector<int64_t> staticStrides;
      if (failed(memrefType.getStridesAndOffset(staticStrides, staticOffset)))
        return {};

      unsigned bitwidth = memrefType.getElementType().getIntOrFloatBitWidth();
      AffineExpr sizeExpr = builder.getAffineConstantExpr(bitwidth / 8);
      for (auto i : llvm::seq(rank))
        sizeExpr = sizeExpr * builder.getAffineSymbolExpr(i);

      auto metadata =
          memref::ExtractStridedMetadataOp::create(builder, loc, input);
      Value offset = metadata.getOffset();
      ValueRange sizes = metadata.getSizes();
      ValueRange strides = metadata.getStrides();

      if (staticOffset != ShapedType::kDynamic)
        offset = arith::ConstantIndexOp::create(builder, loc, staticOffset);

      AffineExpr offsetExpr = builder.getAffineConstantExpr(0) * (bitwidth / 8);
      offset =
          getValue(builder, loc,
                   affine::makeComposedFoldedAffineApply(
                       builder, loc, offsetExpr, getAsOpFoldResult(offset)));

      for (auto i : llvm::seq(rank))
        if (staticStrides[i] != ShapedType::kDynamic)
          strides[i] =
              arith::ConstantIndexOp::create(builder, loc, staticStrides[i]);

      Value size =
          getValue(builder, loc,
                   affine::makeComposedFoldedAffineApply(
                       builder, loc, sizeExpr, getAsOpFoldResult(sizes)));

      Type bufferType = resultType.front();
      Value base =
          UnrealizedConversionCastOp::create(builder, loc, bufferType, input)
              .getResult(0);
      base =
          memref::ViewOp::create(builder, loc, bufferType, base, offset, size);

      SmallVector<Value> result;
      result.push_back(base);
      llvm::append_range(result, sizes);
      llvm::append_range(result, strides);
      return result;
    });
  }
};

/// Returns a collapsed memref and the linearized index to access the element
/// at the specified indices.
static Value getFlattenMemref(OpBuilder &rewriter, Location loc, Value source,
                              unsigned typeBit, Type loadType, ValueRange sizes,
                              ValueRange strides, ValueRange indices) {
  auto sourceType = cast<MemRefType>(source.getType());
  OpFoldResult zero = rewriter.getIndexAttr(0);
  OpFoldResult linearizedIndices;
  memref::LinearizedMemRefInfo linearizedInfo;
  std::tie(linearizedInfo, linearizedIndices) =
      memref::getLinearizedMemRefOffsetAndSize(
          rewriter, loc, typeBit, typeBit, zero, getAsOpFoldResult(sizes),
          getAsOpFoldResult(strides), getAsOpFoldResult(indices));

  auto targetMemrefType = MemRefType::get(
      /*shape*/ {}, loadType, MemRefLayoutAttrInterface{},
      sourceType.getMemorySpace());

  AffineExpr mul = rewriter.getAffineSymbolExpr(0) * (typeBit / 8);
  linearizedIndices = affine::makeComposedFoldedAffineApply(rewriter, loc, mul,
                                                            linearizedIndices);

  Value offset = getValue(rewriter, loc, linearizedIndices);
  return memref::ViewOp::create(rewriter, loc, targetMemrefType, source, offset,
                                {});
}

template <typename OpTy>
struct DecomposeLoadOp : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OneToNOpAdaptor =
      typename OpTy::template GenericAdaptor<ArrayRef<ValueRange>>;

  LogicalResult
  matchAndRewrite(OpTy loadOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = loadOp.getLoc();
    auto memrefType = cast<MemRefType>(loadOp.getMemRefType());
    unsigned rank = memrefType.getRank();
    unsigned typeBit = memrefType.getElementType().getIntOrFloatBitWidth();

    if (rank == 0)
      return rewriter.notifyMatchFailure(loadOp, "already 0D memref");

    Type loadType = loadOp.getType();

    ValueRange sourceDecomposed;
    if constexpr (std::is_same_v<OpTy, memref::LoadOp>) {
      sourceDecomposed = adaptor.getMemref();
    } else {
      sourceDecomposed = adaptor.getBase();
    }

    if (sourceDecomposed.size() != 1 + rank * 2)
      return rewriter.notifyMatchFailure(loadOp,
                                         "expected memref to be decomposed");

    auto [buffer, sizes, strides] = unflattenDescriptor(sourceDecomposed, rank);
    SmallVector<Value> indices = flatten(adaptor.getIndices());

    Value viewMemref = getFlattenMemref(rewriter, loc, buffer, typeBit,
                                        loadType, sizes, strides, indices);

    rewriter.replaceOpWithNewOp<memref::LoadOp>(
        loadOp, loadType, viewMemref, /*indices*/ ValueRange{},
        loadOp.getNontemporal(), loadOp.getAlignmentAttr());
    return success();
  }
};

template <typename OpTy>
struct DecomposeStoreOp : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OneToNOpAdaptor =
      typename OpTy::template GenericAdaptor<ArrayRef<ValueRange>>;

  LogicalResult
  matchAndRewrite(OpTy storeOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = storeOp.getLoc();
    auto memrefType = cast<MemRefType>(storeOp.getMemRefType());
    unsigned rank = memrefType.getRank();
    unsigned typeBit = memrefType.getElementType().getIntOrFloatBitWidth();

    if (rank == 0)
      return rewriter.notifyMatchFailure(storeOp, "already 0D memref");

    ValueRange sourceDecomposed;
    if constexpr (std::is_same_v<OpTy, memref::StoreOp>) {
      sourceDecomposed = adaptor.getMemref();
    } else {
      sourceDecomposed = adaptor.getBase();
    }

    if (sourceDecomposed.size() != 1 + rank * 2)
      return rewriter.notifyMatchFailure(storeOp,
                                         "expected memref to be decomposed");

    auto [buffer, sizes, strides] = unflattenDescriptor(sourceDecomposed, rank);
    SmallVector<Value> indices = flatten(adaptor.getIndices());
    Value valueToStore;
    if constexpr (std::is_same_v<OpTy, memref::StoreOp>) {
      valueToStore = llvm::getSingleElement(adaptor.getValue());
    } else {
      valueToStore = llvm::getSingleElement(adaptor.getValueToStore());
    }
    Type storeType = valueToStore.getType();

    Value viewMemref = getFlattenMemref(rewriter, loc, buffer, typeBit,
                                        storeType, sizes, strides, indices);
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        storeOp, valueToStore, viewMemref, /*indices*/ ValueRange{},
        storeOp.getNontemporal(), storeOp.getAlignmentAttr());
    return success();
  }
};

struct DecomposeReinterpretCast
    : public OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern::OpConversionPattern;
  using OneToNOpAdaptor =
      memref::ReinterpretCastOp::GenericAdaptor<ArrayRef<ValueRange>>;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp castOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = castOp.getLoc();
    auto resultType = cast<MemRefType>(castOp.getType());
    unsigned resultRank = resultType.getRank();

    if (resultRank == 0)
      return rewriter.notifyMatchFailure(castOp, "already 0D memref");

    auto sourceType = cast<MemRefType>(castOp.getSource().getType());
    unsigned sourceRank = sourceType.getRank();

    // Get decomposed source.
    ValueRange sourceDecomposed = adaptor.getSource();
    if (sourceDecomposed.size() != 1 + sourceRank * 2)
      return rewriter.notifyMatchFailure(castOp,
                                         "expected source to be decomposed");

    auto [buffer, oldSizes, oldStrides] =
        unflattenDescriptor(sourceDecomposed, sourceRank);

    // Get new offset, sizes, and strides from the operation.
    ArrayRef<OpFoldResult> offsetsRef = castOp.getMixedOffsets();
    Value offset = getValue(rewriter, loc, offsetsRef[0]);
    SmallVector<OpFoldResult> newSizes = getMixedValues(
        castOp.getStaticSizes(), flatten(adaptor.getSizes()), rewriter);
    SmallVector<OpFoldResult> newStrides = getMixedValues(
        castOp.getStaticStrides(), flatten(adaptor.getStrides()), rewriter);

    // Apply offset to buffer using memref.view.
    unsigned typeBit = resultType.getElementType().getIntOrFloatBitWidth();

    // Compute size: (product of sizes) * (element size in bytes).
    AffineExpr sizeExpr = rewriter.getAffineConstantExpr(typeBit / 8);
    for (auto i : llvm::seq(resultRank))
      sizeExpr = sizeExpr * rewriter.getAffineSymbolExpr(i);

    assert(resultRank == newSizes.size() && resultRank == newStrides.size() &&
           "sizes and strides must have the same rank");

    Value size = getValue(rewriter, loc,
                          affine::makeComposedFoldedAffineApply(
                              rewriter, loc, sizeExpr, newSizes));

    // Compute adjusted offset in bytes.
    AffineExpr offsetExpr = rewriter.getAffineSymbolExpr(0) * (typeBit / 8);
    OpFoldResult adjustedOffset = affine::makeComposedFoldedAffineApply(
        rewriter, loc, offsetExpr, getAsOpFoldResult(offset));
    Value adjustedOffsetValue = getValue(rewriter, loc, adjustedOffset);

    Value newBuffer = memref::ViewOp::create(rewriter, loc, buffer.getType(),
                                             buffer, adjustedOffsetValue, size);

    // Build result as decomposed memref (buffer, sizes, strides).
    SmallVector<Value> decomposedResult;
    decomposedResult.push_back(newBuffer);
    llvm::append_range(decomposedResult, getValues(rewriter, loc, newSizes));
    llvm::append_range(decomposedResult, getValues(rewriter, loc, newStrides));

    rewriter.replaceOpWithMultiple(castOp, {decomposedResult});
    return success();
  }
};

template <typename OpTy> static bool isDynamicallyLegalOp(OpTy op) {
  auto memrefType = cast<MemRefType>(op.getMemRefType());
  return memrefType.getRank() == 0;
}

class MemrefDecompositionPass
    : public water::impl::WaterMemrefDecompositionPassBase<
          MemrefDecompositionPass> {
public:
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();

    // Set up type converter.
    MemrefDecompositionTypeConverter typeConverter;

    // Set up conversion target.
    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                           memref::MemRefDialect, vector::VectorDialect>();
    // Mark load/store operations with non-0D memrefs as illegal.
    target.addDynamicallyLegalOp<memref::LoadOp>(
        isDynamicallyLegalOp<memref::LoadOp>);
    target.addDynamicallyLegalOp<memref::StoreOp>(
        isDynamicallyLegalOp<memref::StoreOp>);
    target.addDynamicallyLegalOp<vector::LoadOp>(
        isDynamicallyLegalOp<vector::LoadOp>);
    target.addDynamicallyLegalOp<vector::StoreOp>(
        isDynamicallyLegalOp<vector::StoreOp>);

    // Mark reinterpret_cast with non-0D result as illegal.
    target.addDynamicallyLegalOp<memref::ReinterpretCastOp>(
        [](memref::ReinterpretCastOp op) {
          auto resultType = cast<MemRefType>(op.getType());
          return resultType.getRank() == 0;
        });

    // Add conversion patterns with type converter.
    RewritePatternSet patterns(ctx);
    patterns
        .add<DecomposeLoadOp<memref::LoadOp>, DecomposeStoreOp<memref::StoreOp>,
             DecomposeLoadOp<vector::LoadOp>, DecomposeStoreOp<vector::StoreOp>,
             DecomposeReinterpretCast>(typeConverter, ctx);

    // Apply partial conversion.
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
