// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::water {
#define GEN_PASS_DEF_WATERMEMREFDECOMPOSITIONPASS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

using namespace mlir;

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

static MemRefType make0DMemRefType(MemRefType type) {
  return MemRefType::get({}, type.getElementType(), MemRefLayoutAttrInterface{},
                         type.getMemorySpace());
}

static Type getMemrefStructType(OpBuilder &builder, Location loc, Type ptrType,
                                unsigned rank) {
  auto i64 = builder.getIntegerType(64);
  if (rank == 0)
    return LLVM::LLVMStructType::getLiteral(builder.getContext(),
                                            {ptrType, ptrType, i64});

  auto arrayType = LLVM::LLVMArrayType::get(i64, rank);

  return LLVM::LLVMStructType::getLiteral(
      builder.getContext(), {ptrType, ptrType, i64, arrayType, arrayType});
}

static Value toPtr(OpBuilder &builder, Location loc,
                   LLVM::LLVMPointerType ptrType, Value value) {
  auto memrefType = cast<MemRefType>(value.getType());
  auto memrefStructType =
      getMemrefStructType(builder, loc, ptrType, memrefType.getRank());
  value =
      UnrealizedConversionCastOp::create(builder, loc, memrefStructType, value)
          .getResult(0);
  return MemRefDescriptor(value).alignedPtr(builder, loc);
}

static Value fromPtr(OpBuilder &builder, Location loc, MemRefType memrefType,
                     Value value) {
  auto ptrType = cast<LLVM::LLVMPointerType>(value.getType());
  assert(memrefType.getRank() == 0 && "only 0D memrefs supported");

  auto memrefStructType = getMemrefStructType(builder, loc, ptrType, 0);
  auto descriptor = MemRefDescriptor::poison(builder, loc, memrefStructType);
  descriptor.setAllocatedPtr(builder, loc, value);
  descriptor.setAlignedPtr(builder, loc, value);
  descriptor.setConstantOffset(builder, loc, 0);
  return UnrealizedConversionCastOp::create(builder, loc, memrefType,
                                            (Value)descriptor)
      .getResult(0);
}

static Value GEP(OpBuilder &builder, Location loc, Value buffer, Value offset) {
  Type elementType = builder.getIntegerType(8);
  Type i64 = builder.getIntegerType(64);
  auto flags = LLVM::GEPNoWrapFlags::nusw;
  offset = UnrealizedConversionCastOp::create(builder, loc, i64, offset)
               .getResult(0);
  return LLVM::GEPOp::create(builder, loc, buffer.getType(), elementType,
                             buffer, offset, flags);
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
        [this](MemRefType type,
               SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
          // Convert memory space attribute.
          unsigned addressSpace = 0;
          if (Attribute memorySpace = type.getMemorySpace()) {
            std::optional<Attribute> convertedSpace =
                convertTypeAttribute(type, memorySpace);
            if (!convertedSpace)
              return std::nullopt;

            if (!(*convertedSpace)) { // Conversion to default is 0.
              addressSpace = 0;
            } else if (auto explicitSpace =
                           dyn_cast_if_present<IntegerAttr>(*convertedSpace)) {
              addressSpace = explicitSpace.getInt();
            } else {
              return std::nullopt;
            }
          }

          MLIRContext *ctx = type.getContext();
          Type ptrType = LLVM::LLVMPointerType::get(ctx, addressSpace);

          unsigned rank = type.getRank();
          auto indexType = IndexType::get(ctx);

          results.push_back(ptrType);
          results.resize(1 + rank * 2, indexType);
          return success();
        });

    /// Source materialization to reconstruct memref from components.
    addSourceMaterialization([](OpBuilder &builder, MemRefType resultType,
                                ValueRange inputs, Location loc) -> Value {
      unsigned rank = resultType.getRank();
      if (inputs.size() != 1 + rank * 2)
        return {};

      if (!isa<LLVM::LLVMPointerType>(inputs.front().getType()))
        return {};

      if (!llvm::all_of(inputs.drop_front(), [](Value value) {
            return isa<IndexType>(value.getType());
          }))
        return {};

      int64_t staticOffset = 0;
      SmallVector<int64_t> staticStrides;
      if (failed(resultType.getStridesAndOffset(staticStrides, staticOffset)))
        return {};

      auto [buffer, sizes, strides] = unflattenDescriptor(inputs, rank);

      auto memrefType = MemRefType::get({}, resultType.getElementType(),
                                        MemRefLayoutAttrInterface{},
                                        resultType.getMemorySpace());
      buffer = fromPtr(builder, loc, memrefType, buffer);

      OpFoldResult offset = builder.getIndexAttr(0);
      SmallVector<OpFoldResult> mixedSizes;
      SmallVector<OpFoldResult> mixedStrides;
      for (auto i : llvm::seq(rank)) {
        if (ShapedType::isDynamic(staticStrides[i]))
          mixedStrides.push_back(strides[i]);
        else
          mixedStrides.push_back(builder.getIndexAttr(staticStrides[i]));

        int64_t size = resultType.getDimSize(i);
        if (ShapedType::isDynamic(size))
          mixedSizes.push_back(sizes[i]);
        else
          mixedSizes.push_back(builder.getIndexAttr(size));
      }

      return memref::ReinterpretCastOp::create(
          builder, loc, resultType, buffer, offset, mixedSizes, mixedStrides);
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

      if (!isa<LLVM::LLVMPointerType>(resultType.front()))
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

      input = toPtr(builder, loc,
                    cast<LLVM::LLVMPointerType>(resultType.front()), input);
      Value base = GEP(builder, loc, input, offset);

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
                              Type loadType, ValueRange sizes, unsigned typeBit,
                              ValueRange strides, ValueRange indices) {
  OpFoldResult zero = rewriter.getIndexAttr(0);
  OpFoldResult linearizedIndices;
  memref::LinearizedMemRefInfo linearizedInfo;
  std::tie(linearizedInfo, linearizedIndices) =
      memref::getLinearizedMemRefOffsetAndSize(
          rewriter, loc, typeBit, typeBit, zero, getAsOpFoldResult(sizes),
          getAsOpFoldResult(strides), getAsOpFoldResult(indices));

  AffineExpr mul = rewriter.getAffineSymbolExpr(0) * (typeBit / 8);
  linearizedIndices = affine::makeComposedFoldedAffineApply(rewriter, loc, mul,
                                                            linearizedIndices);

  Value offset = getValue(rewriter, loc, linearizedIndices);
  return GEP(rewriter, loc, source, offset);
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

    Value ptr = getFlattenMemref(rewriter, loc, buffer, loadType, sizes,
                                 typeBit, strides, indices);

    unsigned alignment = loadOp.getAlignment().value_or(0);
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(loadOp, loadType, ptr, alignment,
                                              /*volatile_*/ false,
                                              loadOp.getNontemporal());
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

    Value ptr = getFlattenMemref(rewriter, loc, buffer, storeType, sizes,
                                 typeBit, strides, indices);

    unsigned alignment = storeOp.getAlignment().value_or(0);
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(storeOp, valueToStore, ptr,
                                               alignment, /*volatile_*/ false,
                                               storeOp.getNontemporal());
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

    // Compute adjusted offset in bytes.
    AffineExpr offsetExpr = rewriter.getAffineSymbolExpr(0) * (typeBit / 8);
    OpFoldResult adjustedOffset = affine::makeComposedFoldedAffineApply(
        rewriter, loc, offsetExpr, getAsOpFoldResult(offset));
    Value adjustedOffsetValue = getValue(rewriter, loc, adjustedOffset);

    Value newBuffer = GEP(rewriter, loc, buffer, adjustedOffsetValue);

    // Build result as decomposed memref (buffer, sizes, strides).
    SmallVector<Value> decomposedResult;
    decomposedResult.push_back(newBuffer);
    llvm::append_range(decomposedResult, getValues(rewriter, loc, newSizes));
    llvm::append_range(decomposedResult, getValues(rewriter, loc, newStrides));

    rewriter.replaceOpWithMultiple(castOp, {decomposedResult});
    return success();
  }
};

struct DecomposeFatRawBufferCast
    : public OpConversionPattern<amdgpu::FatRawBufferCastOp> {
  using OpConversionPattern::OpConversionPattern;
  using OneToNOpAdaptor =
      amdgpu::FatRawBufferCastOp::GenericAdaptor<ArrayRef<ValueRange>>;

  LogicalResult
  matchAndRewrite(amdgpu::FatRawBufferCastOp castOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = castOp.getLoc();
    auto sourceType = cast<MemRefType>(castOp.getSource().getType());
    auto resultType = cast<MemRefType>(castOp.getType());
    unsigned resultRank = resultType.getRank();

    if (resultRank == 0)
      return rewriter.notifyMatchFailure(castOp, "already 0D memref");

    unsigned sourceRank = sourceType.getRank();

    sourceType = make0DMemRefType(sourceType);
    resultType = make0DMemRefType(resultType);

    auto resultPtrType =
        typeConverter->convertType<LLVM::LLVMPointerType>(resultType);
    if (!resultPtrType)
      return rewriter.notifyMatchFailure(castOp,
                                         "failed to convert result type");

    // Get decomposed source.
    ValueRange sourceDecomposed = adaptor.getSource();
    if (sourceDecomposed.size() != 1 + sourceRank * 2)
      return rewriter.notifyMatchFailure(castOp,
                                         "expected source to be decomposed");

    auto [buffer, sizes, strides] =
        unflattenDescriptor(sourceDecomposed, sourceRank);

    auto sourcePtrType = dyn_cast<LLVM::LLVMPointerType>(buffer.getType());
    if (!sourcePtrType)
      return rewriter.notifyMatchFailure(castOp,
                                         "failed to convert source type");

    buffer = fromPtr(rewriter, loc, sourceType, buffer);

    Value fatBuffer = amdgpu::FatRawBufferCastOp::create(
        rewriter, loc, resultType, buffer, castOp.getValidBytes(),
        castOp.getCacheSwizzleStride(), castOp.getBoundsCheck(),
        castOp.getResetOffset());

    fatBuffer = toPtr(rewriter, loc, cast<LLVM::LLVMPointerType>(resultPtrType),
                      fatBuffer);

    // Build result as decomposed memref (buffer, sizes, strides).
    SmallVector<Value> decomposedResult;
    decomposedResult.push_back(fatBuffer);
    llvm::append_range(decomposedResult, sizes);
    llvm::append_range(decomposedResult, strides);

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

    MemrefDecompositionTypeConverter typeConverter;
    amdgpu::populateCommonGPUTypeAndAttributeConversions(typeConverter);
    populateAMDGPUTypeAndAttributeConversions(typeConverter);

    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                           memref::MemRefDialect, vector::VectorDialect,
                           amdgpu::AMDGPUDialect, LLVM::LLVMDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    target.addDynamicallyLegalOp<memref::LoadOp>(
        isDynamicallyLegalOp<memref::LoadOp>);
    target.addDynamicallyLegalOp<memref::StoreOp>(
        isDynamicallyLegalOp<memref::StoreOp>);
    target.addDynamicallyLegalOp<vector::LoadOp>(
        isDynamicallyLegalOp<vector::LoadOp>);
    target.addDynamicallyLegalOp<vector::StoreOp>(
        isDynamicallyLegalOp<vector::StoreOp>);

    target.addDynamicallyLegalOp<memref::ReinterpretCastOp,
                                 amdgpu::FatRawBufferCastOp>(
        [&](Operation *op) {
          auto resultType = cast<MemRefType>(op->getResult(0).getType());
          return resultType.getRank() == 0;
        });

    RewritePatternSet patterns(ctx);
    patterns
        .add<DecomposeLoadOp<memref::LoadOp>, DecomposeStoreOp<memref::StoreOp>,
             DecomposeLoadOp<vector::LoadOp>, DecomposeStoreOp<vector::StoreOp>,
             DecomposeReinterpretCast, DecomposeFatRawBufferCast>(typeConverter,
                                                                  ctx);

    // Apply partial conversion.
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
