// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// MemRef Dialect Handlers
//===----------------------------------------------------------------------===//
//
// This file implements handlers for MemRef dialect operations:
//   - memref.alloc - track LDS allocation size
//   - memref.view - set LDS base offset
//   - memref.reinterpret_cast - propagate LDS offset and update SRD size
//   - memref.subview - pass through source mapping
//   - memref.load - emit ds_read or buffer_load
//   - memref.store - emit ds_write or buffer_store
//   - memref.cast - pass through source mapping
//
// CRITICAL: These handlers include LDS offset tracking via:
//   - ctx.setLDSBaseOffset() in handleMemRefView
//   - ctx.getLDSBaseOffset() / ctx.setLDSBaseOffset() in
//   handleMemRefReinterpretCast
//   - ctx.addLDSSize() in handleMemRefAlloc
//   - ctx.updateSRDBufferSize() in handleMemRefReinterpretCast
//
//===----------------------------------------------------------------------===//

#include "Handlers.h"

#include "waveasm/Dialect/WaveASMOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace waveasm {

//===----------------------------------------------------------------------===//
// Handler Implementations
//===----------------------------------------------------------------------===//

/// Handle memref.alloc - track LDS allocation and assign base offset
///
/// Each LDS memref.alloc is assigned a cumulative byte offset so that
/// operations (vector.load, vector.store, gather_to_lds) can compute the
/// correct LDS address.  Two patterns exist in practice:
///
///   (a) memref.alloc of a raw byte buffer + memref.view with byte offset
///       -> handleMemRefView sets the offset from the view's byteShift
///
///   (b) memref.alloc of a typed, shaped buffer (e.g. memref<256x128xi8, 3>)
///       -> no memref.view follows; we must set the offset here
///
/// Pattern (b) is used by MXFP4 scheduled GEMM kernels that allocate
/// separate data and scale buffers for double-buffering.
LogicalResult handleMemRefAlloc(Operation *op, TranslationContext &ctx) {
  auto allocOp = cast<memref::AllocOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // Check if this is an LDS allocation (workgroup address space)
  auto memrefType = allocOp.getResult().getType();
  if (isLDSMemRef(memrefType)) {
    // Compute the allocation size in bytes
    int64_t numElements = 1;
    for (int64_t dim : memrefType.getShape()) {
      if (dim != ShapedType::kDynamic) {
        numElements *= dim;
      }
    }
    int64_t elementBytes = (memrefType.getElementTypeBitWidth() + 7) / 8;
    int64_t allocSize = numElements * elementBytes;

    // Track the total LDS size for the kernel descriptor
    ctx.addLDSSize(allocSize);

    // For typed shaped LDS allocations (pattern b), set the LDS base offset
    // so that vector.load/store and gather_to_lds handlers can compute the
    // correct LDS address.  Skip 1-D i8 buffers (pattern a) since those use
    // memref.view which sets the offset from the view's byteShift.
    bool isRawByteBuffer =
        (memrefType.getRank() == 1 && memrefType.getElementType().isInteger(8));
    if (!isRawByteBuffer) {
      int64_t baseOffset = ctx.getLDSAllocOffset();
      auto immType = ctx.createImmType(baseOffset);
      auto offsetVal = ConstantOp::create(builder, loc, immType, baseOffset);
      ctx.setLDSBaseOffset(allocOp.getResult(), offsetVal);
      ctx.advanceLDSAllocOffset(allocSize);
    }
  }

  return success();
}

/// Handle memref.view - compute LDS offset
/// CRITICAL: This sets the LDS base offset for the memref result
LogicalResult handleMemRefView(Operation *op, TranslationContext &ctx) {
  auto viewOp = cast<memref::ViewOp>(op);

  // Track the byte offset for LDS addressing
  // The byteShift is the base offset into LDS that this view starts at
  if (auto offset = ctx.getMapper().getMapped(viewOp.getByteShift())) {
    // Store the LDS base offset for this memref so it can be added
    // during vector.load/store operations
    ctx.setLDSBaseOffset(viewOp.getResult(), *offset);
  }

  return success();
}

/// Handle memref.reinterpret_cast - track memref identity
/// CRITICAL: This propagates LDS offset and updates SRD buffer size
LogicalResult handleMemRefReinterpretCast(Operation *op,
                                          TranslationContext &ctx) {
  auto castOp = cast<memref::ReinterpretCastOp>(op);

  // Reinterpret cast doesn't change the underlying buffer
  if (auto src = ctx.getMapper().getMapped(castOp.getSource())) {
    ctx.getMapper().mapValue(castOp.getResult(), *src);
  }

  // Propagate LDS base offset from source to result
  // This is needed when memref.view creates an LDS view with a byte offset,
  // and then memref.reinterpret_cast is applied to reshape it
  if (auto ldsOffset = ctx.getLDSBaseOffset(castOp.getSource())) {
    ctx.setLDSBaseOffset(castOp.getResult(), *ldsOffset);
  }

  // The result type often has more specific shape info than the source.
  // Update the SRD buffer size if the result type is larger.
  if (auto memrefType = dyn_cast<MemRefType>(castOp.getResult().getType())) {
    int64_t bufferSize = computeBufferSizeFromMemRef(memrefType);
    // Update the SRD for the source memref with the more accurate size
    ctx.updateSRDBufferSize(castOp.getSource(), bufferSize);
  }

  return success();
}

/// Handle memref.subview - compute offset
LogicalResult handleMemRefSubView(Operation *op, TranslationContext &ctx) {
  auto subviewOp = cast<memref::SubViewOp>(op);

  // Pass through source (offset computation handled by indices)
  if (auto src = ctx.getMapper().getMapped(subviewOp.getSource())) {
    ctx.getMapper().mapValue(subviewOp.getResult(), *src);
  }
  return success();
}

/// Handle memref.load - emit ds_read or buffer_load
LogicalResult handleMemRefLoad(Operation *op, TranslationContext &ctx) {
  auto loadOp = cast<memref::LoadOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto memrefType = loadOp.getMemRefType();
  auto vregType = ctx.createVRegType();

  if (isLDSMemRef(memrefType)) {
    // LDS load
    Value vaddr;
    if (!loadOp.getIndices().empty()) {
      if (auto mapped = ctx.getMapper().getMapped(loadOp.getIndices()[0])) {
        vaddr = *mapped;
      }
    }
    if (!vaddr) {
      auto immType = ctx.createImmType(0);
      vaddr = ConstantOp::create(builder, loc, immType, 0);
    }

    auto readOp = DS_READ_B32::create(builder, loc, TypeRange{vregType}, vaddr);
    ctx.getMapper().mapValue(loadOp.getResult(), readOp.getResult(0));
  } else {
    // Global load
    auto sregType = ctx.createSRegType(4, 4);
    auto srd = PrecoloredSRegOp::create(builder, loc, sregType, 8, 4);

    Value voffset;
    if (!loadOp.getIndices().empty()) {
      if (auto mapped = ctx.getMapper().getMapped(loadOp.getIndices()[0])) {
        voffset = *mapped;
      }
    }
    if (!voffset) {
      auto immType = ctx.createImmType(0);
      voffset = ConstantOp::create(builder, loc, immType, 0);
    }

    auto loadInstr = BUFFER_LOAD_DWORD::create(
        builder, loc, TypeRange{vregType}, srd, voffset);
    ctx.getMapper().mapValue(loadOp.getResult(), loadInstr.getResult(0));
  }

  return success();
}

/// Handle memref.store - emit ds_write or buffer_store
LogicalResult handleMemRefStore(Operation *op, TranslationContext &ctx) {
  auto storeOp = cast<memref::StoreOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto memrefType = storeOp.getMemRefType();

  auto data = ctx.getMapper().getMapped(storeOp.getValueToStore());
  if (!data) {
    return op->emitError("data value not mapped");
  }

  if (isLDSMemRef(memrefType)) {
    // LDS store
    Value vaddr;
    if (!storeOp.getIndices().empty()) {
      if (auto mapped = ctx.getMapper().getMapped(storeOp.getIndices()[0])) {
        vaddr = *mapped;
      }
    }
    if (!vaddr) {
      auto immType = ctx.createImmType(0);
      vaddr = ConstantOp::create(builder, loc, immType, 0);
    }

    DS_WRITE_B32::create(builder, loc, *data, vaddr);
  } else {
    // Global store
    auto sregType = ctx.createSRegType(4, 4);
    auto srd = PrecoloredSRegOp::create(builder, loc, sregType, 8, 4);

    Value voffset;
    if (!storeOp.getIndices().empty()) {
      if (auto mapped = ctx.getMapper().getMapped(storeOp.getIndices()[0])) {
        voffset = *mapped;
      }
    }
    if (!voffset) {
      auto immType = ctx.createImmType(0);
      voffset = ConstantOp::create(builder, loc, immType, 0);
    }

    BUFFER_STORE_DWORD::create(builder, loc, *data, srd, voffset);
  }

  return success();
}

/// Handle memref.cast - pass through source
LogicalResult handleMemRefCast(Operation *op, TranslationContext &ctx) {
  auto castOp = cast<memref::CastOp>(op);

  if (auto src = ctx.getMapper().getMapped(castOp.getSource())) {
    ctx.getMapper().mapValue(castOp.getResult(), *src);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MemRefAddressComputer Implementation
//===----------------------------------------------------------------------===//

Value MemRefAddressComputer::computeByteOffset(ArrayRef<Value> indices,
                                               MemRefType memrefType,
                                               TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = builder.getUnknownLoc();
  auto vregType = ctx.createVRegType();

  int64_t elemSize = getElementBytes(memrefType.getElementType());

  // Start with zero offset
  auto immZero = ctx.createImmType(0);
  Value offset = ConstantOp::create(builder, loc, immZero, 0);

  // Compute linearized offset: sum of (index[i] * stride[i])
  for (unsigned i = 0; i < indices.size(); ++i) {
    auto idxMapped = ctx.getMapper().getMapped(indices[i]);
    if (!idxMapped)
      continue;

    int64_t stride = getStride(memrefType, i);
    int64_t byteStride = stride * elemSize;

    if (byteStride == 1) {
      offset = V_ADD_U32::create(builder, loc, vregType, offset, *idxMapped);
    } else {
      auto immStride = ctx.createImmType(byteStride);
      auto strideConst =
          ConstantOp::create(builder, loc, immStride, byteStride);
      auto scaled =
          V_MUL_LO_U32::create(builder, loc, vregType, *idxMapped, strideConst);
      offset = V_ADD_U32::create(builder, loc, vregType, offset, scaled);
    }
  }

  return offset;
}

std::optional<int64_t> MemRefAddressComputer::computeConstantOffset(
    ArrayRef<Value> indices, MemRefType memrefType, TranslationContext &ctx) {
  int64_t elemSize = getElementBytes(memrefType.getElementType());
  int64_t totalOffset = 0;

  for (unsigned i = 0; i < indices.size(); ++i) {
    auto idxMapped = ctx.getMapper().getMapped(indices[i]);
    if (!idxMapped)
      return std::nullopt;

    // Check if index is a constant
    if (auto constOp = idxMapped->getDefiningOp<ConstantOp>()) {
      int64_t stride = getStride(memrefType, i);
      totalOffset += constOp.getValue() * stride * elemSize;
    } else {
      return std::nullopt; // Non-constant index
    }
  }

  return totalOffset;
}

int64_t MemRefAddressComputer::getStride(MemRefType memrefType, unsigned dim) {
  auto shape = memrefType.getShape();

  // For row-major layout, stride[i] = product of shape[i+1..n]
  int64_t stride = 1;
  for (unsigned j = dim + 1; j < shape.size(); ++j) {
    if (shape[j] == ShapedType::kDynamic)
      return 1; // Can't compute statically
    stride *= shape[j];
  }
  return stride;
}

//===----------------------------------------------------------------------===//
// VectorSplitter Implementation
//===----------------------------------------------------------------------===//

llvm::SmallVector<std::pair<int64_t, int64_t>>
VectorSplitter::getSplitRanges(int64_t totalBytes) {
  llvm::SmallVector<std::pair<int64_t, int64_t>> ranges;

  int64_t offset = 0;
  while (offset < totalBytes) {
    // Determine chunk size: prefer 16, 8, 4, 2, 1 bytes
    int64_t remaining = totalBytes - offset;
    int64_t chunkSize;

    if (remaining >= 16) {
      chunkSize = 16;
    } else if (remaining >= 8) {
      chunkSize = 8;
    } else if (remaining >= 4) {
      chunkSize = 4;
    } else if (remaining >= 2) {
      chunkSize = 2;
    } else {
      chunkSize = 1;
    }

    ranges.push_back({offset, chunkSize});
    offset += chunkSize;
  }

  return ranges;
}

} // namespace waveasm
