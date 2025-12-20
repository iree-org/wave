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

          // Add size types (one index per dimension).
          unsigned rank = type.getRank();
          for (unsigned i = 0; i < rank; ++i) {
            results.push_back(indexType);
          }

          // Add stride types (one index per dimension).
          for (unsigned i = 0; i < rank; ++i) {
            results.push_back(indexType);
          }

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

      unsigned bitwidth = memrefType.getElementType().getIntOrFloatBitWidth();
      AffineExpr sizeExpr = builder.getAffineConstantExpr(bitwidth / 8);
      for (auto i : llvm::seq(rank))
        sizeExpr = sizeExpr * builder.getAffineSymbolExpr(i);

      auto metadata =
          memref::ExtractStridedMetadataOp::create(builder, loc, input);
      Value base = metadata.getBaseBuffer();
      Value offset = metadata.getOffset();
      ValueRange sizes = metadata.getSizes();
      ValueRange strides = metadata.getStrides();

      Value size =
          getValue(builder, loc,
                   affine::makeComposedFoldedAffineApply(
                       builder, loc, sizeExpr, getAsOpFoldResult(sizes)));

      Type bufferType = resultType.front();
      base = UnrealizedConversionCastOp::create(builder, loc, bufferType, base)
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
                              Type loadType, ValueRange sizes,
                              ValueRange strides, ValueRange indices) {
  auto sourceType = cast<MemRefType>(source.getType());
  unsigned typeBit = sourceType.getElementType().getIntOrFloatBitWidth();
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

struct DecomposeMemrefLoadOp : public OpConversionPattern<memref::LoadOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(memref::LoadOp loadOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = loadOp.getLoc();
    auto memrefType = cast<MemRefType>(loadOp.getMemRefType());
    unsigned rank = memrefType.getRank();

    if (rank == 0)
      return rewriter.notifyMatchFailure(loadOp, "already 0D memref");

    Type loadType = loadOp.getType();

    ValueRange sourceDecomposed = adaptor.getMemref();
    if (sourceDecomposed.size() != 1 + rank * 2)
      return rewriter.notifyMatchFailure(loadOp,
                                         "expected memref to be decomposed");

    auto [buffer, sizes, strides] = unflattenDescriptor(sourceDecomposed, rank);
    SmallVector<Value> indices = flatten(adaptor.getIndices());

    Value viewMemref = getFlattenMemref(rewriter, loc, buffer, loadType, sizes,
                                        strides, indices);

    rewriter.replaceOpWithNewOp<memref::LoadOp>(loadOp, loadType, viewMemref);
    return success();
  }
};

struct DecomposeMemrefStoreOp : public OpConversionPattern<memref::StoreOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(memref::StoreOp storeOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = storeOp.getLoc();
    auto memrefType = cast<MemRefType>(storeOp.getMemRefType());
    unsigned rank = memrefType.getRank();

    if (rank == 0)
      return rewriter.notifyMatchFailure(storeOp, "already 0D memref");

    ValueRange sourceDecomposed = adaptor.getMemref();
    if (sourceDecomposed.size() != 1 + rank * 2)
      return rewriter.notifyMatchFailure(storeOp,
                                         "expected memref to be decomposed");

    auto [buffer, sizes, strides] = unflattenDescriptor(sourceDecomposed, rank);
    SmallVector<Value> indices = flatten(adaptor.getIndices());
    Value valueToStore = llvm::getSingleElement(adaptor.getValue());
    Type storeType = valueToStore.getType();

    Value viewMemref = getFlattenMemref(rewriter, loc, buffer, storeType, sizes,
                                        strides, indices);
    rewriter.replaceOpWithNewOp<memref::StoreOp>(storeOp, valueToStore,
                                                 viewMemref);
    return success();
  }
};

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
                           memref::MemRefDialect>();

    // Mark load/store operations with non-0D memrefs as illegal.
    target.addDynamicallyLegalOp<memref::LoadOp>([](memref::LoadOp op) {
      auto memrefType = cast<MemRefType>(op.getMemRefType());
      return memrefType.getRank() == 0;
    });

    target.addDynamicallyLegalOp<memref::StoreOp>([](memref::StoreOp op) {
      auto memrefType = cast<MemRefType>(op.getMemRefType());
      return memrefType.getRank() == 0;
    });

    // Add conversion patterns with type converter.
    RewritePatternSet patterns(ctx);
    patterns.add<DecomposeMemrefLoadOp, DecomposeMemrefStoreOp>(typeConverter,
                                                                ctx);

    // Apply partial conversion.
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
