// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"
#include "water/Dialect/Wave/Transforms/LoweringPatterns.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "wave-lowering"
#define LDBG() llvm::dbgs() << "[" DEBUG_TYPE "] "

using namespace mlir;

namespace {

struct TransferOpCommonAttrs {
  AffineMapAttr perm; // affine_map<(d0..)->(d_vectorizedDim)>
  ArrayAttr inTrue;   // [true,  true,  ...] length = vector rank
  ArrayAttr inFalse;  // [false, false, ...] length = vector rank
};

/// Build the common attrs for vector.transfer_{read,write} when vectorizing a
/// memref dimension.
static TransferOpCommonAttrs
make1DTransferCommonAttrs(MemRefType memrefType, int64_t vectorizedDim,
                          VectorType vecType, PatternRewriter &rewriter) {

  uint64_t memrefRank = memrefType.getRank();
  MLIRContext *ctx = rewriter.getContext();

  assert(vecType.getRank() == 1 && "only 1-D vectors supported here");

  // Permutation map: (d0,..,d{rank-1}) -> (d_{vectorizedDim})
  AffineExpr e = getAffineDimExpr(vectorizedDim, ctx);
  AffineMap map = AffineMap::get(memrefRank, /*numSymbols=*/0, e, ctx);
  AffineMapAttr permAttr = AffineMapAttr::get(map);

  ArrayAttr inTrue = rewriter.getBoolArrayAttr({true});
  ArrayAttr inFalse = rewriter.getBoolArrayAttr({false});

  return TransferOpCommonAttrs{permAttr, inTrue, inFalse};
}

/// Materialize affine.apply for expressions inside a `map` with `symbols`.
/// Each symbol is either a GPU id (thread/block) or a constant from `hyper`.
static FailureOr<SmallVector<Value>>
materializeAffine(Location loc, ArrayRef<wave::WaveSymbolAttr> symbols,
                  AffineMap map, PatternRewriter &rewriter,
                  wave::WaveHyperparameterAttr hyper) {
  // NOTE: This helper assumes 0 dims in `map`. If you add dims, prepend
  // the dim operands before the symbol operands below.
  assert(map.getNumDims() == 0 && "expected 0 dims");

  auto threadId = [&](gpu::Dimension d) -> Value {
    return rewriter.create<gpu::ThreadIdOp>(loc, rewriter.getIndexType(), d);
  };
  auto blockId = [&](gpu::Dimension d) -> Value {
    return rewriter.create<gpu::BlockIdOp>(loc, rewriter.getIndexType(), d);
  };

  SmallVector<Value> baseSymVals;
  baseSymVals.reserve(map.getNumSymbols());
  int64_t numSym = map.getNumSymbols();
  for (int64_t i = 0; i < numSym; ++i) {
    StringRef name = symbols[i].getName();

    Value v;
    if (name == "_T0")
      v = threadId(gpu::Dimension::x);
    else if (name == "_T1")
      v = threadId(gpu::Dimension::y);
    else if (name == "_T2")
      v = threadId(gpu::Dimension::z);
    else if (name == "_WG0")
      v = blockId(gpu::Dimension::x);
    else if (name == "_WG1")
      v = blockId(gpu::Dimension::y);
    else if (name == "_WG2")
      v = blockId(gpu::Dimension::z);
    else if (std::optional<int64_t> value = hyper.getSymbolValue(name)) {
      v = rewriter.create<arith::ConstantIndexOp>(loc, *value);
    } else {
      LLVM_DEBUG(llvm::errs() << "symbol: " << name << "\n");
      assert(false && "unknown symbol, should have been caught by verifiers");
    }
    baseSymVals.push_back(v);
  }
  // In case map contains multiple results, create one apply per result.
  SmallVector<Value> results;
  results.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults()) {
    AffineMap submap =
        AffineMap::get(map.getNumDims(), map.getNumSymbols(), expr);
    SmallVector<Value> symVals = baseSymVals;
    affine::canonicalizeMapAndOperands(&submap, &symVals);

    Value apply = rewriter.create<affine::AffineApplyOp>(loc, submap, symVals);
    results.push_back(apply);
  }

  return results;
}

/// Build per-dimension start indices in the order specified by `orderedSyms`
/// (the Wave tensor’s dim order). For each symbol name in `orderedSyms`, this
/// looks up a WaveIndexMappingAttr in `indexDict`, materializes its `start`
/// map and returns one index-typed Value per dimension.
static FailureOr<SmallVector<Value>>
buildStartIndices(Location loc, DictionaryAttr indexDict,
                  ArrayRef<wave::WaveSymbolAttr> orderedSyms,
                  PatternRewriter &rewriter,
                  wave::WaveHyperparameterAttr hyper) {
  SmallVector<Value> indices;
  indices.reserve(orderedSyms.size());
  for (wave::WaveSymbolAttr symAttr : orderedSyms) {
    StringRef name = symAttr.getName();
    Attribute a = indexDict.get(name);
    assert(a && "index dict missing entry for dimension symbol");
    auto mapAttr = cast<wave::WaveIndexMappingAttr>(a);

    FailureOr<SmallVector<Value>> startFo = materializeAffine(
        loc, mapAttr.getSymbolNames(), mapAttr.getStart(), rewriter, hyper);
    if (failed(startFo))
      return failure();
    SmallVector<Value> start = std::move(*startFo);
    assert(llvm::hasSingleElement(start));
    indices.push_back(start[0]); // Start map has one result.
  }
  return indices;
}

/// Build a per-thread mask:
///
///   mask = AND_d ( id_start_d(elements_per_thread) <
///                  bound_d(elements_per_thread))
///          foreach d in dimensions.
///
/// whenever a bounds dictionary is provided. When it is not provided, return a
/// null mask. If the vectorized dimension cannot be identified, return failure.
static FailureOr<Value>
buildMask(Location loc, wave::WaveReadWriteBoundsAttr boundsDict,
          ArrayRef<wave::WaveSymbolAttr> orderedSyms, PatternRewriter &rewriter,
          DictionaryAttr indexDict, wave::WaveHyperparameterAttr hyper,
          ArrayRef<Value> startIdx, int64_t elementsPerThread) {
  if (!boundsDict)
    return Value();

  const uint64_t rank = startIdx.size();
  IndexType idxType = rewriter.getIndexType();
  VectorType vecIdxType = VectorType::get({elementsPerThread}, idxType);
  IntegerType i1Type = IntegerType::get(rewriter.getContext(), 1);
  VectorType maskType = VectorType::get({elementsPerThread}, i1Type);

  // finalMask is the AND of per-dimension bound checks.
  Value finalMask;
  for (uint64_t d = 0; d < rank; ++d) {
    StringRef name = orderedSyms[d].getName();
    Attribute a = boundsDict.getMapping().get(name);
    assert(a && "bounds dict missing entry for dimension symbol");
    auto boundAttr = cast<wave::WaveExprListAttr>(a);
    // Materialize bounds.
    FailureOr<SmallVector<Value>> boundValsFo = materializeAffine(
        loc, boundAttr.getSymbols(), boundAttr.getMap(), rewriter, hyper);
    if (failed(boundValsFo))
      return failure();
    SmallVector<Value> boundVals = std::move(*boundValsFo);
    Value bound = boundVals[0];

    Value clause;
    if (d == rank - 1) {
      // iota [0..L-1] : vector<index>
      Value iota = rewriter.create<vector::StepOp>(loc, vecIdxType);

      // Lane indices for fastest dim: start + iota.
      Value startFastVec =
          rewriter.create<vector::BroadcastOp>(loc, vecIdxType, startIdx[d]);
      Value laneIdxFast =
          rewriter.create<arith::AddIOp>(loc, startFastVec, iota);

      // lane-wise compare: (start + iota) < bound.
      Value boundVec =
          rewriter.create<vector::BroadcastOp>(loc, vecIdxType, bound);
      clause = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                              laneIdxFast, boundVec);
    } else {
      // scalar compare then broadcast: startIdx[d] < bound.
      Value scalarCmp = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, startIdx[d], bound);
      clause = rewriter.create<vector::BroadcastOp>(loc, maskType, scalarCmp);
    }

    finalMask =
        finalMask
            ? rewriter.create<arith::AndIOp>(loc, finalMask, clause).getResult()
            : clause;
  }

  return finalMask;
}

/// Checks whether the vectorized dimension is the innermost dimension and if
/// its stride is 1. This is required to use vector.load/store or
/// masked_load/store.
static FailureOr<bool> isMinorContiguous(MemRefType memrefType,
                                         int64_t vectorizedDim) {
  int64_t memrefRank = memrefType.getRank();

  // Must be vectorizing the innermost dimension.
  if (vectorizedDim != static_cast<int64_t>(memrefRank - 1))
    return false;

  int64_t offset = 0;
  SmallVector<int64_t> strides;
  if (failed(memrefType.getStridesAndOffset(strides, offset))) {
    LLVM_DEBUG(llvm::dbgs()
               << "Non-strided memref layout; using transfer ops\n \n");
    return failure(); // layout not representable as simple strided
  }

  // Minor (last) stride must be known and equal to 1 to be contiguous.
  if (strides.empty() || ShapedType::isDynamic(strides.back()))
    return false;

  return strides.back() == 1;
}

/// Build a read or a masked read operation based on presence of a mask.
/// - If vectorizing the innermost dimension with unit stride (contiguous), emit
/// `vector.load` or `vector.maskedload`
/// - Otherwise (non-innermost or non-contiguous innermost), emit
/// `vector.transfer_read`.
///`in_bounds` semantics for transfer ops:
/// - if set to `[true]`  we believe the indices passed through are safe (i.e.
/// within bounds); the compiler may
///   elide bounds checks.
/// - if set to [false] : lowering emits bounds check with respect to memref
/// shape.
static Value buildVectorRead(Location loc, PatternRewriter &rewriter, Value mem,
                             ArrayRef<Value> indices, VectorType vecType,
                             Value mask, uint64_t vectorizedDim) {

  Type eltType = vecType.getElementType();
  TypedAttr zeroElement;
  if (auto flt = dyn_cast<FloatType>(eltType))
    zeroElement = rewriter.getFloatAttr(flt, 0.0);
  else if (auto it = dyn_cast<IntegerType>(eltType))
    zeroElement = rewriter.getIntegerAttr(it, 0);
  else
    assert(false && "unsupported element type");

  auto memrefType = cast<MemRefType>(mem.getType());
  FailureOr<bool> isMinorContig = isMinorContiguous(memrefType, vectorizedDim);
  bool usePlainVectorRead = isMinorContig.value_or(false);

  // vector.load or vector.masked_load
  if (usePlainVectorRead) {
    if (!mask)
      return rewriter.create<vector::LoadOp>(loc, vecType, mem, indices);

    // Create a passthrough vector with elements set to zero corresponding to
    // the element type in memory.
    Value passthrough = rewriter.create<arith::ConstantOp>(
        loc, vecType, SplatElementsAttr::get(vecType, zeroElement));
    return rewriter.create<vector::MaskedLoadOp>(loc, vecType, mem, indices,
                                                 mask, passthrough);
  }

  // vector.transfer_read (masked or unmasked)
  TransferOpCommonAttrs common =
      make1DTransferCommonAttrs(memrefType, vectorizedDim, vecType, rewriter);
  // Padding value used by vector.transfer_read to fill lanes that are
  // out-of-bounds or masked-off.
  Value padding = rewriter.create<arith::ConstantOp>(loc, eltType, zeroElement);

  if (!mask) {
    return rewriter.create<vector::TransferReadOp>(
        loc, vecType, mem, indices, /*padding=*/padding,
        /*permutation_map=*/common.perm,
        /*in_bounds=*/common.inFalse);
  }
  return rewriter.create<vector::TransferReadOp>(
      loc, vecType, mem, indices, /*permutation_map=*/common.perm, padding,
      mask,
      /*in_bounds=*/common.inTrue);
}

/// Build a write or a masked write operation based on presence of a mask.
/// - If vectorizing the innermost dimension with unit stride (contiguous), emit
/// `vector.store` or `vector.maskedstore`
/// - Otherwise (non-innermost or non-contiguous innermost), emit
/// `vector.transfer_write`.
/// `in_bounds` semantics for transfer ops:
/// - if set to `[true]` we believe the indices passed through are safe (i.e.
/// within bounds); the compiler may
///   elide bounds checks.
/// - if set to [false] the lowering emits bounds check with respect to memref
/// shape.
static void buildVectorWrite(Location loc, PatternRewriter &rewriter, Value mem,
                             ArrayRef<Value> indices, Value vecValue,
                             Value mask, uint64_t vectorizedDim) {

  auto memrefType = cast<MemRefType>(mem.getType());
  FailureOr<bool> isMinorContig = isMinorContiguous(memrefType, vectorizedDim);
  bool usePlainVectorStore = succeeded(isMinorContig) && *isMinorContig;

  // vector.store or vector.masked_store
  if (usePlainVectorStore) {
    if (mask) {
      rewriter.create<vector::MaskedStoreOp>(loc, mem, indices, mask, vecValue);
    } else {
      rewriter.create<vector::StoreOp>(loc, vecValue, mem, indices);
    }
  }

  // vector.transfer_write (masked or unmasked)
  auto vecType = cast<VectorType>(vecValue.getType());
  TransferOpCommonAttrs common =
      make1DTransferCommonAttrs(memrefType, vectorizedDim, vecType, rewriter);

  if (mask) {
    rewriter.create<vector::TransferWriteOp>(loc, vecValue, mem, indices,
                                             /*permutation_map=*/common.perm,
                                             /*mask=*/mask,
                                             /*in_bounds=*/common.inTrue);

  } else {
    rewriter.create<vector::TransferWriteOp>(loc, vecValue, mem, indices,
                                             /*permutation_map=*/common.perm,
                                             /*in_bounds=*/common.inFalse);
  }
}

/// Describes access info used when lowering Wave ops to vector read/write ops.
/// - startIndices: base element indices into the memref (size == memref rank).
/// - mask: vector<i1> guarding per-lane accesses (length == elementsPerThread).
/// - vectorizedDim: memref dimension spanned by the vector elements
struct MemAccessInfo {
  SmallVector<Value> startIndices;
  Value mask;
  int64_t vectorizedDim;
};

// Common lowering for memory operations such as read/write. Checks whether
// enough information is present on the operation and constructs the starting
// index (populates the trailing vector argument) and eventually the mask.
template <typename OpTy>
static FailureOr<MemAccessInfo> createMemoryIndicesAndMask(
    ConversionPatternRewriter &rewriter, const TypeConverter *typeConverter,
    OpTy op, wave::WaveTensorType memoryType, VectorType vectorType) {

  int64_t elementsPerThread = vectorType.getNumElements();

  ArrayRef<wave::WaveSymbolAttr> orderedSyms = memoryType.getShape();

  wave::WaveReadWriteBoundsAttr boundsDict = op.getBoundsAttr();
  wave::WaveHyperparameterAttr hyper =
      static_cast<const wave::WaveTypeConverter &>(*typeConverter)
          .getHyperparameters();

  // This logic operates after the "expansion" pass that usually unrolls all
  // tensor dimensions but one so the extent along them is 1. The dimension
  // with non-unit extent is considered to be vectorized.

  // Read/Write ops only carry a single index expression: the first (and only)
  // dictionary inside the array attribute.
  ArrayAttr indexArr = op.getIndexAttr();
  if (!indexArr || indexArr.size() != 1)
    return rewriter.notifyMatchFailure(
        op, "'index' must be an array with exactly one dictionary");
  DictionaryAttr indexDict = dyn_cast<DictionaryAttr>(indexArr[0]);
  if (!indexDict)
    return rewriter.notifyMatchFailure(
        op, "cannot lower without 'index' attribute");
  std::optional<int64_t> vectorizedDim =
      wave::getPositionOfVectorizedDim(orderedSyms, indexDict, hyper);

  if (!vectorizedDim.has_value()) {
    return rewriter.notifyMatchFailure(
        op, "failed to identify vectorized dimension");
  }
  FailureOr<SmallVector<Value>> maybeStartIndices =
      buildStartIndices(op->getLoc(), indexDict, orderedSyms, rewriter, hyper);
  if (failed(maybeStartIndices))
    return rewriter.notifyMatchFailure(
        op, "failed to convert start indices to affine");
  SmallVector<Value> startIndices = std::move(*maybeStartIndices);

  FailureOr<Value> mask =
      buildMask(op->getLoc(), boundsDict, orderedSyms, rewriter, indexDict,
                hyper, startIndices, elementsPerThread);
  if (failed(mask))
    return rewriter.notifyMatchFailure(op, "couldn't build the required mask");

  return MemAccessInfo{startIndices, *mask, *vectorizedDim};
}

class ReadOpLoweringPattern : public OpConversionPattern<wave::ReadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wave::ReadOp op, wave::ReadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedType)
      return rewriter.notifyMatchFailure(op,
                                         "WaveTensorType conversion failed");
    auto vectorType = cast<VectorType>(convertedType);
    FailureOr<MemAccessInfo> memInfo = createMemoryIndicesAndMask(
        rewriter, getTypeConverter(), op, op.getMemory().getType(), vectorType);
    if (failed(memInfo))
      return failure();

    Value readOp = buildVectorRead(op.getLoc(), rewriter, adaptor.getMemory(),
                                   memInfo->startIndices, vectorType,
                                   memInfo->mask, memInfo->vectorizedDim);
    rewriter.replaceOp(op, readOp);
    return success();
  }
};

class WriteOpLoweringPattern : public OpConversionPattern<wave::WriteOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wave::WriteOp op, wave::WriteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value vec = adaptor.getValueToStore();
    auto vecType = cast<VectorType>(vec.getType());

    FailureOr<MemAccessInfo> memInfo = createMemoryIndicesAndMask(
        rewriter, getTypeConverter(), op, op.getMemory().getType(), vecType);
    if (failed(memInfo))
      return failure();

    buildVectorWrite(op.getLoc(), rewriter, adaptor.getMemory(),
                     memInfo->startIndices, vec, memInfo->mask,
                     memInfo->vectorizedDim);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void wave::populateWaveReadWriteLoweringPatterns(
    WaveTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<ReadOpLoweringPattern, WriteOpLoweringPattern>(
      typeConverter, patterns.getContext());
}
