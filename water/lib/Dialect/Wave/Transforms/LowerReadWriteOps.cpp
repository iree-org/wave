// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/IR/Attributes.h"
#include "water/Dialect/Wave/IR/IndexExpr.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"
#include "water/Dialect/Wave/Transforms/LoweringPatterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
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
make1DTransferCommonAttrs(int64_t memoryRank, int64_t vectorizedDim,
                          PatternRewriter &rewriter) {

  MLIRContext *ctx = rewriter.getContext();

  // Permutation map: (d0,..,d{rank-1}) -> (d_{vectorizedDim})
  AffineExpr e = getAffineDimExpr(vectorizedDim, ctx);
  AffineMap map = AffineMap::get(memoryRank, /*numSymbols=*/0, e, ctx);
  AffineMapAttr permAttr = AffineMapAttr::get(map);

  ArrayAttr inTrue = rewriter.getBoolArrayAttr({true});
  ArrayAttr inFalse = rewriter.getBoolArrayAttr({false});

  return TransferOpCommonAttrs{permAttr, inTrue, inFalse};
}

} // namespace

namespace {

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

    FailureOr<SmallVector<Value>> startFo = wave::materializeAffine(
        loc, mapAttr.getSymbols(), mapAttr.getStart(), rewriter, hyper);
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
///   mask = AND ( start-index_d[0..elements_per_thread-1] <
///                splat(bound_d, elements_per_thread))
///          foreach d in dimensions.
///
/// whenever a bounds mapping is provided. When it is not provided, return a
/// null mask. elements_per_thread is considered to be 1 for any dimension
/// other than `vectorizedDim` and the given value for `vectorizedDim`.
static FailureOr<Value>
buildMask(Location loc, wave::WaveSymbolMappingAttr boundsMapping,
          ArrayRef<wave::WaveSymbolAttr> orderedSyms, PatternRewriter &rewriter,
          wave::WaveHyperparameterAttr hyper, ArrayRef<Value> startIdx,
          int64_t elementsPerThread, uint64_t vectorizedDim) {
  if (!boundsMapping)
    return Value();

  const uint64_t rank = startIdx.size();
  IndexType idxType = rewriter.getIndexType();
  VectorType vecIdxType = VectorType::get({elementsPerThread}, idxType);
  IntegerType i1Type = IntegerType::get(rewriter.getContext(), 1);
  VectorType maskType = VectorType::get({elementsPerThread}, i1Type);

  // finalMask is the AND of per-dimension bound checks. The bounds dict may
  // be sparse: only dimensions that require masking have an entry. Dimensions
  // without an entry are fully in-bounds and are skipped.
  Value finalMask;
  for (uint64_t d = 0; d < rank; ++d) {
    auto boundAttr =
        boundsMapping.lookup<wave::WaveExprListAttr>(orderedSyms[d]);
    if (!boundAttr)
      continue;
    // Materialize bounds.
    FailureOr<SmallVector<Value>> boundValsFo = wave::materializeAffine(
        loc, boundAttr.getSymbols(), boundAttr.getMap(), rewriter, hyper);
    if (failed(boundValsFo))
      return failure();
    SmallVector<Value> boundVals = std::move(*boundValsFo);
    Value bound = boundVals[0];

    Value clause;
    if (d == vectorizedDim) {
      // iota [0..L-1] : vector<index>
      Value iota = vector::StepOp::create(rewriter, loc, vecIdxType);

      // Lane indices for fastest dim: start + iota.
      Value startFastVec =
          vector::BroadcastOp::create(rewriter, loc, vecIdxType, startIdx[d]);
      Value laneIdxFast =
          arith::AddIOp::create(rewriter, loc, startFastVec, iota);

      // lane-wise compare: (start + iota) < bound.
      Value boundVec =
          vector::BroadcastOp::create(rewriter, loc, vecIdxType, bound);
      clause = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                     laneIdxFast, boundVec);
    } else {
      // scalar compare then broadcast: startIdx[d] < bound.
      Value scalarCmp = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::slt, startIdx[d], bound);
      clause = vector::BroadcastOp::create(rewriter, loc, maskType, scalarCmp);
    }

    finalMask = finalMask
                    ? arith::AndIOp::create(rewriter, loc, finalMask, clause)
                          .getResult()
                    : clause;
  }

  return finalMask;
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
                             Value mask, int64_t vectorizedDim) {

  Type eltType = vecType.getElementType();
  TypedAttr zeroElement;
  if (auto flt = dyn_cast<FloatType>(eltType))
    zeroElement = rewriter.getFloatAttr(flt, 0.0);
  else if (auto it = dyn_cast<IntegerType>(eltType))
    zeroElement = rewriter.getIntegerAttr(it, 0);
  else
    assert(false && "unsupported element type");

  int64_t memoryRank = cast<MemRefType>(mem.getType()).getRank();
  bool usePlainVectorRead = vectorizedDim == memoryRank - 1;

  // vector.load or vector.masked_load
  if (usePlainVectorRead) {
    if (!mask)
      return vector::LoadOp::create(rewriter, loc, vecType, mem, indices);

    // Create a passthrough vector with elements set to zero corresponding to
    // the element type in memory.
    Value passthrough = arith::ConstantOp::create(
        rewriter, loc, vecType, SplatElementsAttr::get(vecType, zeroElement));
    return vector::MaskedLoadOp::create(rewriter, loc, vecType, mem, indices,
                                        mask, passthrough);
  }

  // vector.transfer_read (masked or unmasked)
  TransferOpCommonAttrs common =
      make1DTransferCommonAttrs(memoryRank, vectorizedDim, rewriter);
  // Padding value used by vector.transfer_read to fill lanes that are
  // out-of-bounds or masked-off.
  Value padding =
      arith::ConstantOp::create(rewriter, loc, eltType, zeroElement);

  if (!mask) {
    return vector::TransferReadOp::create(rewriter, loc, vecType, mem, indices,
                                          /*padding=*/padding,
                                          /*permutation_map=*/common.perm,
                                          /*in_bounds=*/common.inFalse);
  }
  return vector::TransferReadOp::create(rewriter, loc, vecType, mem, indices,
                                        /*permutation_map=*/common.perm,
                                        padding, mask,
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
                             Value mask, int64_t vectorizedDim) {

  int64_t memoryRank = cast<MemRefType>(mem.getType()).getRank();
  bool usePlainVectorStore = vectorizedDim == memoryRank - 1;

  // vector.store or vector.masked_store
  if (usePlainVectorStore) {
    if (mask) {
      vector::MaskedStoreOp::create(rewriter, loc, mem, indices, mask,
                                    vecValue);
    } else {
      vector::StoreOp::create(rewriter, loc, vecValue, mem, indices);
    }
    return;
  }

  // vector.transfer_write (masked or unmasked)
  TransferOpCommonAttrs common =
      make1DTransferCommonAttrs(memoryRank, vectorizedDim, rewriter);

  if (mask) {
    vector::TransferWriteOp::create(rewriter, loc, vecValue, mem, indices,
                                    /*permutation_map=*/common.perm,
                                    /*mask=*/mask,
                                    /*in_bounds=*/common.inTrue);

  } else {
    vector::TransferWriteOp::create(rewriter, loc, vecValue, mem, indices,
                                    /*permutation_map=*/common.perm,
                                    /*in_bounds=*/common.inFalse);
  }
}

static DictionaryAttr
transformIndex(DictionaryAttr indexDict,
               ArrayRef<wave::WaveSymbolAttr> orderedSyms,
               wave::WaveExprListAttr mapping,
               SmallVectorImpl<wave::WaveSymbolAttr> &updatedOrderedSyms) {
  assert(updatedOrderedSyms.empty() &&
         "updatedOrderedSyms must be empty and will be populated");
  if (!mapping || mapping.getMap().isIdentity()) {
    updatedOrderedSyms.assign(orderedSyms);
    return indexDict;
  }

  // When we hit this while increasing mapping expressiveness, it would mean
  // that we need ot add the symbol part of the mapping to the new value. We
  // will need to figure out which dimension.
  assert(mapping.getMap().isPermutation() &&
         "NYI: only permutation mappings are currently supported");

  wave::permuteShape(orderedSyms, mapping.getMap(), /*inverse=*/true,
                     updatedOrderedSyms);

  // XXX: step/stride are not permuted similarly to pywave. For step, this
  // appears necessary to avoid losing track of the dimension that is
  // vectorized. For stride, not sure, it may be unused at all at this stage.
  SmallVector<NamedAttribute> newIndexDictEntries;
  for (size_t i = 0, e = updatedOrderedSyms.size(); i < e; ++i) {
    auto updatedIndexMapping = cast<wave::WaveIndexMappingAttr>(
        indexDict.get(orderedSyms[i].getName()));
    auto originalIndexMapping = cast<wave::WaveIndexMappingAttr>(
        indexDict.get(updatedOrderedSyms[i].getName()));

    SmallVector<Attribute> allSymbols;
    wave::aggregateAllSymbols(
        ArrayRef<ArrayRef<Attribute>>{updatedIndexMapping.getSymbols(),
                                      originalIndexMapping.getSymbols()},
        allSymbols);

    AffineMap start =
        wave::alignMapSymbols(updatedIndexMapping.getStart(),
                              updatedIndexMapping.getSymbols(), allSymbols);
    AffineMap step =
        wave::alignMapSymbols(originalIndexMapping.getStep(),
                              originalIndexMapping.getSymbols(), allSymbols);
    AffineMap stride =
        wave::alignMapSymbols(originalIndexMapping.getStride(),
                              originalIndexMapping.getSymbols(), allSymbols);

    newIndexDictEntries.emplace_back(
        updatedOrderedSyms[i].getName(),
        wave::WaveIndexMappingAttr::get(updatedIndexMapping.getContext(),
                                        allSymbols, start, step, stride));
  }
  return DictionaryAttr::get(indexDict.getContext(), newIndexDictEntries);
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
static FailureOr<MemAccessInfo>
createMemoryIndicesAndMask(ConversionPatternRewriter &rewriter,
                           const TypeConverter *typeConverter, OpTy op,
                           Type memoryTypeArg, VectorType vectorType) {
  int64_t elementsPerThread = vectorType.getNumElements();

  wave::WaveSymbolMappingAttr boundsMapping = op.getBoundsAttr();
  wave::WaveHyperparameterAttr hyper =
      static_cast<const wave::WaveTypeConverter &>(*typeConverter)
          .getHyperparameters();

  // This logic operates after the "expansion" pass that usually unrolls all
  // tensor dimensions but one so the extent along them is 1. The dimension
  // with non-unit extent is considered to be vectorized.

  // Read/Write ops only carry a single index expression: the first (and only)
  // dictionary inside the array attribute. The IndexExprsSpecified normal form
  // guarantees this attribute is present.
  ArrayAttr indexArr = op.getIndexAttr();
  assert(indexArr && "IndexExprsSpecified normal form guarantees index attr");
  assert(llvm::hasSingleElement(indexArr.getValue()) &&
         "'index' must be an array with exactly one dictionary");
  DictionaryAttr indexDict = cast<DictionaryAttr>(indexArr[0]);

  // Get ordered symbols for dimension ordering.
  // DictionaryAttr stores entries sorted alphabetically by key, which doesn't
  // match the semantic dimension ordering. We need to get the correct order
  // from either:
  // 1. The ordered_syms attribute (set by ResolveDistributedAllocations for
  //    ops with MemRefType memory operands).
  // 2. The WaveTensorType shape (for ops that still have WaveTensorType).
  SmallVector<wave::WaveSymbolAttr> orderedSymsStorage;
  ArrayRef<wave::WaveSymbolAttr> orderedSyms;

  if (ArrayAttr orderedSymsAttr = op.getOrderedSymsAttr()) {
    orderedSymsStorage =
        llvm::to_vector(orderedSymsAttr.getAsRange<wave::WaveSymbolAttr>());
    orderedSyms = orderedSymsStorage;
  } else if (auto waveTensorType =
                 dyn_cast<wave::WaveTensorType>(memoryTypeArg)) {
    orderedSyms = waveTensorType.getShape();
  } else {
    return rewriter.notifyMatchFailure(
        op, "MemRefType memory operand requires ordered_syms attribute");
  }

  wave::WaveExprListAttr mapping = op.getMappingAttr();
  SmallVector<wave::WaveSymbolAttr> memoryShape;
  DictionaryAttr transformedIndexDict =
      transformIndex(indexDict, orderedSyms, mapping, memoryShape);

  std::optional<int64_t> vectorizedDim = wave::getPositionOfVectorizedDim(
      memoryShape, transformedIndexDict, hyper);

  if (!vectorizedDim.has_value()) {
    return rewriter.notifyMatchFailure(
        op, "failed to identify vectorized dimension");
  }

  FailureOr<SmallVector<Value>> maybeStartIndices = buildStartIndices(
      op->getLoc(), transformedIndexDict, memoryShape, rewriter, hyper);
  if (failed(maybeStartIndices))
    return rewriter.notifyMatchFailure(
        op, "failed to convert start indices to affine");
  SmallVector<Value> startIndices = std::move(*maybeStartIndices);

  // When this assertion hits while increasing mapping expressiveness, the
  // `buildMask` call below may need to be updated to account for the mapping,
  // in particular for any additive component of the mapping.
  // XXX: pywave somehow uses the unmodified index dict to build the mask in
  // some case regardless of the mapping, which looks suspicious. This approach
  // doesn't need the index dict entries but relies on the order in which start
  // indices occur (matches the memory shape, rather than the unmodified index
  // dictionary order that matches the value shape).
  assert((!mapping || mapping.getMap().isPermutation()) &&
         "NYI: only permutation mappings are currently supported");
  FailureOr<Value> mask =
      buildMask(op->getLoc(), boundsMapping, memoryShape, rewriter, hyper,
                startIndices, elementsPerThread);
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
    auto memoryType = cast<MemRefType>(adaptor.getMemory().getType());
    if (!memoryType.getLayout().isIdentity()) {
      return rewriter.notifyMatchFailure(op, "non-identity memory layout");
    }

    // wave.read produces a register-resident value. PropagateElementsPerThread
    // converts these from WaveTensorType<register> to VectorType. The
    // MemoryOnlyTypes normal form (required by LowerWaveToMLIR) verifies that
    // no WaveTensorType<register> remains, ensuring VectorType is used.
    auto vectorType = dyn_cast<VectorType>(op.getResult().getType());
    assert(vectorType && "expected VectorType result; "
                         "PropagateElementsPerThread may not have run");
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
    auto memoryType = cast<MemRefType>(adaptor.getMemory().getType());
    if (!memoryType.getLayout().isIdentity()) {
      return rewriter.notifyMatchFailure(op, "non-identity memory layout");
    }

    // wave.write consumes a register-resident value. PropagateElementsPerThread
    // converts these from WaveTensorType<register> to VectorType. The
    // MemoryOnlyTypes normal form (required by LowerWaveToMLIR) verifies that
    // no WaveTensorType<register> remains, ensuring VectorType is used.
    Value vec = adaptor.getValueToStore();
    auto vecType = dyn_cast<VectorType>(vec.getType());
    assert(vecType && "expected VectorType operand; PropagateElementsPerThread "
                      "may not have run");

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
