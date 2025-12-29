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
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
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

/// Unflatten a list of values into a (buffer, sizes, strides).
static std::tuple<LogicalResult, Value, SmallVector<OpFoldResult>,
                  SmallVector<OpFoldResult>>
unflattenDescriptor(ValueRange values, MemRefType memrefType) {
  unsigned rank = memrefType.getRank();
  if (values.size() != 1 + rank * 2)
    return {failure(), {}, {}, {}};

  int64_t staticOffset = 0;
  SmallVector<int64_t> staticStrides;
  if (failed(memrefType.getStridesAndOffset(staticStrides, staticOffset)))
    return {failure(), {}, {}, {}};

  Value buffer = values.front();
  values = values.drop_front();
  ValueRange sizes = values.take_front(rank);
  values = values.drop_front(rank);
  ValueRange strides = values.take_front(rank);

  SmallVector<OpFoldResult> mixedSizes;
  SmallVector<OpFoldResult> mixedStrides;
  auto intType = IndexType::get(memrefType.getContext());
  for (auto i : llvm::seq(rank)) {
    int64_t size = memrefType.getDimSize(i);
    if (ShapedType::isDynamic(size))
      mixedSizes.push_back(sizes[i]);
    else
      mixedSizes.push_back(IntegerAttr::get(intType, size));

    if (ShapedType::isDynamic(staticStrides[i]))
      mixedStrides.push_back(strides[i]);
    else
      mixedStrides.push_back(IntegerAttr::get(intType, staticStrides[i]));
  }

  return {success(), buffer, mixedSizes, mixedStrides};
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

/// Extract the pointer from a memref descriptor.
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

/// Create a 0D memref descriptor from a pointer.
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

/// Cast an integer or index value to a different integer or index type.
static Value integerCast(OpBuilder &builder, Location loc, Type dstType,
                         Value value) {
  Type srcType = value.getType();
  assert((isa<IntegerType, IndexType>(srcType)) &&
         "source type must be integer or index");
  assert((isa<IntegerType, IndexType>(dstType)) &&
         "destination type must be integer or index");
  if (srcType == dstType)
    return value;

  // If one of them is index, use index_cast.
  if (isa<IndexType>(srcType) || isa<IndexType>(dstType))
    return arith::IndexCastOp::create(builder, loc, dstType, value);

  // If both are integers, use trunc/ext.
  if (cast<IntegerType>(srcType).getWidth() >
      cast<IntegerType>(dstType).getWidth())
    return arith::TruncIOp::create(builder, loc, dstType, value);
  else
    return arith::ExtSIOp::create(builder, loc, dstType, value);
}

/// Generate a GEP op with the given buffer and byte offset.
static Value createGEP(OpBuilder &builder, Location loc, Value buffer,
                       Value offset) {
  Type elementType = builder.getIntegerType(8);
  Type i64 = builder.getIntegerType(64);
  auto flags = LLVM::GEPNoWrapFlags::nusw;
  offset = integerCast(builder, loc, i64, offset);
  return LLVM::GEPOp::create(builder, loc, buffer.getType(), elementType,
                             buffer, offset, flags);
}

/// Given source pointer, type, sizes, strides, and indices, generate an
/// adjusted pointer.
static Value getFlattenMemref(OpBuilder &rewriter, Location loc, Value source,
                              Type loadType, ArrayRef<OpFoldResult> sizes,
                              unsigned typeBit, ArrayRef<OpFoldResult> strides,
                              ValueRange indices) {
  OpFoldResult zero = rewriter.getIndexAttr(0);
  OpFoldResult linearizedIndices;
  memref::LinearizedMemRefInfo linearizedInfo;
  std::tie(linearizedInfo, linearizedIndices) =
      memref::getLinearizedMemRefOffsetAndSize(rewriter, loc, typeBit, typeBit,
                                               zero, sizes, strides,
                                               getAsOpFoldResult(indices));

  AffineExpr mul = rewriter.getAffineSymbolExpr(0) * (typeBit / 8);
  linearizedIndices = affine::makeComposedFoldedAffineApply(rewriter, loc, mul,
                                                            linearizedIndices);

  Value offset = getValue(rewriter, loc, linearizedIndices);
  return createGEP(rewriter, loc, source, offset);
}

namespace {

class MemrefDecompositionTypeConverter : public TypeConverter {
public:
  MemrefDecompositionTypeConverter() {
    // Keep all other types unchanged.
    addConversion([](Type type) { return type; });

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

          // ptr, sizes, strides
          results.push_back(ptrType);
          results.resize(1 + rank * 2, indexType);
          return success();
        });

    /// Source materialization to reconstruct memref from components.
    addSourceMaterialization([](OpBuilder &builder, MemRefType resultType,
                                ValueRange inputs, Location loc) -> Value {
      auto [valid, buffer, sizes, strides] =
          unflattenDescriptor(inputs, resultType);
      if (failed(valid))
        return {};

      if (!isa<LLVM::LLVMPointerType>(inputs.front().getType()))
        return {};

      if (!llvm::all_of(inputs.drop_front(), [](Value value) {
            return isa<IndexType>(value.getType());
          }))
        return {};

      auto memrefType = MemRefType::get({}, resultType.getElementType(),
                                        MemRefLayoutAttrInterface{},
                                        resultType.getMemorySpace());
      buffer = fromPtr(builder, loc, memrefType, buffer);

      OpFoldResult offset = builder.getIndexAttr(0);
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

      auto metadata =
          memref::ExtractStridedMetadataOp::create(builder, loc, input);
      Value offset = metadata.getOffset();
      ValueRange sizes = metadata.getSizes();
      ValueRange strides = metadata.getStrides();

      if (staticOffset != ShapedType::kDynamic)
        offset = arith::ConstantIndexOp::create(builder, loc, staticOffset);

      AffineExpr offsetExpr = builder.getAffineSymbolExpr(0) * (bitwidth / 8);
      offset =
          getValue(builder, loc,
                   affine::makeComposedFoldedAffineApply(
                       builder, loc, offsetExpr, getAsOpFoldResult(offset)));

      for (auto i : llvm::seq(rank)) {
        if (ShapedType::isStatic(memrefType.getDimSize(i)))
          sizes[i] = arith::ConstantIndexOp::create(builder, loc,
                                                    memrefType.getDimSize(i));

        if (ShapedType::isStatic(staticStrides[i]))
          strides[i] =
              arith::ConstantIndexOp::create(builder, loc, staticStrides[i]);
      }

      input = toPtr(builder, loc,
                    cast<LLVM::LLVMPointerType>(resultType.front()), input);
      Value base = createGEP(builder, loc, input, offset);

      SmallVector<Value> result;
      result.push_back(base);
      llvm::append_range(result, sizes);
      llvm::append_range(result, strides);
      return result;
    });
  }
};

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

    auto [valid, buffer, sizes, strides] =
        unflattenDescriptor(sourceDecomposed, memrefType);
    if (failed(valid))
      return rewriter.notifyMatchFailure(loadOp,
                                         "expected memref to be decomposed");

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

    auto [valid, buffer, sizes, strides] =
        unflattenDescriptor(sourceDecomposed, memrefType);
    if (failed(valid))
      return rewriter.notifyMatchFailure(storeOp,
                                         "expected memref to be decomposed");

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
  using Base::Base;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp castOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = castOp.getLoc();
    auto resultType = cast<MemRefType>(castOp.getType());
    unsigned resultRank = resultType.getRank();

    if (resultRank == 0)
      return rewriter.notifyMatchFailure(castOp, "already 0D memref");

    auto sourceType = cast<MemRefType>(castOp.getSource().getType());

    ValueRange sourceDecomposed = adaptor.getSource();
    auto [valid, buffer, oldSizes, oldStrides] =
        unflattenDescriptor(sourceDecomposed, sourceType);
    if (failed(valid))
      return rewriter.notifyMatchFailure(castOp,
                                         "expected source to be decomposed");

    // Get new offset, sizes, and strides from the operation.
    Value offset = getValue(rewriter, loc, castOp.getMixedOffsets()[0]);
    SmallVector<OpFoldResult> newSizes = getMixedValues(
        castOp.getStaticSizes(), flatten(adaptor.getSizes()), rewriter);
    SmallVector<OpFoldResult> newStrides = getMixedValues(
        castOp.getStaticStrides(), flatten(adaptor.getStrides()), rewriter);

    unsigned typeBit = resultType.getElementType().getIntOrFloatBitWidth();

    assert(resultRank == newSizes.size() && resultRank == newStrides.size() &&
           "sizes and strides must have the same rank");

    // Compute adjusted offset in bytes.
    AffineExpr offsetExpr = rewriter.getAffineSymbolExpr(0) * (typeBit / 8);
    OpFoldResult adjustedOffset = affine::makeComposedFoldedAffineApply(
        rewriter, loc, offsetExpr, getAsOpFoldResult(offset));
    Value adjustedOffsetValue = getValue(rewriter, loc, adjustedOffset);

    Value newBuffer = createGEP(rewriter, loc, buffer, adjustedOffsetValue);

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
  using Base::Base;

  LogicalResult
  matchAndRewrite(amdgpu::FatRawBufferCastOp castOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = castOp.getLoc();
    auto sourceType = cast<MemRefType>(castOp.getSource().getType());
    auto resultType = cast<MemRefType>(castOp.getType());
    unsigned resultRank = resultType.getRank();

    if (resultRank == 0)
      return rewriter.notifyMatchFailure(castOp, "already 0D memref");

    ValueRange sourceDecomposed = adaptor.getSource();

    auto [valid, buffer, sizes, strides] =
        unflattenDescriptor(sourceDecomposed, sourceType);
    if (failed(valid))
      return rewriter.notifyMatchFailure(castOp,
                                         "expected source to be decomposed");

    sourceType = make0DMemRefType(sourceType);
    resultType = make0DMemRefType(resultType);

    auto resultPtrType =
        typeConverter->convertType<LLVM::LLVMPointerType>(resultType);
    if (!resultPtrType)
      return rewriter.notifyMatchFailure(castOp,
                                         "failed to convert result type");

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
    llvm::append_range(decomposedResult, getValues(rewriter, loc, sizes));
    llvm::append_range(decomposedResult, getValues(rewriter, loc, strides));

    rewriter.replaceOpWithMultiple(castOp, {decomposedResult});
    return success();
  }
};

struct DecomposeGatherToLDS
    : public OpConversionPattern<amdgpu::GatherToLDSOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(amdgpu::GatherToLDSOp gatherOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = gatherOp.getLoc();
    auto srcType = cast<MemRefType>(gatherOp.getSrc().getType());
    auto dstType = cast<MemRefType>(gatherOp.getDst().getType());
    unsigned srcRank = srcType.getRank();
    unsigned dstRank = dstType.getRank();

    if (srcRank == 0 && dstRank == 0)
      return rewriter.notifyMatchFailure(gatherOp, "already 0D memrefs");

    // Decompose source memref.
    ValueRange srcDecomposed = adaptor.getSrc();
    auto [srcValid, srcBuffer, srcSizes, srcStrides] =
        unflattenDescriptor(srcDecomposed, srcType);
    if (failed(srcValid))
      return rewriter.notifyMatchFailure(gatherOp,
                                         "expected src to be decomposed");

    // Decompose destination memref.
    ValueRange dstDecomposed = adaptor.getDst();
    auto [dstValid, dstBuffer, dstSizes, dstStrides] =
        unflattenDescriptor(dstDecomposed, dstType);
    if (failed(dstValid))
      return rewriter.notifyMatchFailure(gatherOp,
                                         "expected dst to be decomposed");

    // Get flattened indices.
    SmallVector<Value> srcIndices = flatten(adaptor.getSrcIndices());
    SmallVector<Value> dstIndices = flatten(adaptor.getDstIndices());

    // Compute linearized offsets and apply to buffers.
    unsigned srcTypeBit = srcType.getElementType().getIntOrFloatBitWidth();
    unsigned dstTypeBit = dstType.getElementType().getIntOrFloatBitWidth();

    Type srcElementType = srcType.getElementType();
    Type dstElementType = dstType.getElementType();

    Value srcPtr =
        getFlattenMemref(rewriter, loc, srcBuffer, srcElementType, srcSizes,
                         srcTypeBit, srcStrides, srcIndices);
    Value dstPtr =
        getFlattenMemref(rewriter, loc, dstBuffer, dstElementType, dstSizes,
                         dstTypeBit, dstStrides, dstIndices);

    // Convert to 0D memrefs.
    auto src0DType = make0DMemRefType(srcType);
    auto dst0DType = make0DMemRefType(dstType);

    Value src0D = fromPtr(rewriter, loc, src0DType, srcPtr);
    Value dst0D = fromPtr(rewriter, loc, dst0DType, dstPtr);

    // Create the gather_to_lds operation with 0D memrefs and no indices.
    rewriter.replaceOpWithNewOp<amdgpu::GatherToLDSOp>(
        gatherOp, src0D, ValueRange{}, dst0D, ValueRange{},
        gatherOp.getTransferTypeAttr());

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

    MemrefDecompositionTypeConverter typeConverter;
    amdgpu::populateCommonGPUTypeAndAttributeConversions(typeConverter);
    populateAMDGPUTypeAndAttributeConversions(typeConverter);

    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                           memref::MemRefDialect, vector::VectorDialect,
                           amdgpu::AMDGPUDialect, LLVM::LLVMDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalOp<memref::LoadOp, memref::StoreOp, vector::LoadOp,
                        vector::StoreOp>();

    target.addDynamicallyLegalOp<memref::ReinterpretCastOp,
                                 amdgpu::FatRawBufferCastOp>(
        [&](Operation *op) {
          auto resultType = cast<MemRefType>(op->getResult(0).getType());
          return resultType.getRank() == 0;
        });

    target.addDynamicallyLegalOp<amdgpu::GatherToLDSOp>([&](Operation *op) {
      auto gatherOp = cast<amdgpu::GatherToLDSOp>(op);
      auto srcType = cast<MemRefType>(gatherOp.getSrc().getType());
      auto dstType = cast<MemRefType>(gatherOp.getDst().getType());
      return srcType.getRank() == 0 && dstType.getRank() == 0;
    });

    RewritePatternSet patterns(ctx);
    patterns
        .add<DecomposeLoadOp<memref::LoadOp>, DecomposeStoreOp<memref::StoreOp>,
             DecomposeLoadOp<vector::LoadOp>, DecomposeStoreOp<vector::StoreOp>,
             DecomposeReinterpretCast, DecomposeFatRawBufferCast,
             DecomposeGatherToLDS>(typeConverter, ctx);

    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                         patterns, target);

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
