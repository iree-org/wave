// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Shared Helper Functions for Operation Handlers
//===----------------------------------------------------------------------===//
//
// This file implements utility functions declared in Handlers.h that are
// shared across multiple handler files (ArithHandlers, AffineHandlers,
// MemRefHandlers, AMDGPUHandlers, etc.).
//
//===----------------------------------------------------------------------===//

#include "Handlers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace waveasm {

//===----------------------------------------------------------------------===//
// isLDSMemRef
//===----------------------------------------------------------------------===//

bool isLDSMemRef(MemRefType memrefType) {
  auto memSpace = memrefType.getMemorySpace();
  if (!memSpace)
    return false;

  // Check for gpu.address_space<workgroup> attribute
  if (auto gpuSpace = dyn_cast<gpu::AddressSpaceAttr>(memSpace)) {
    return gpuSpace.getValue() == gpu::AddressSpace::Workgroup;
  }
  // Also check for integer address space (3 on AMDGPU)
  if (auto intAttr = dyn_cast<IntegerAttr>(memSpace)) {
    return intAttr.getInt() == 3;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// getElementBytes
//===----------------------------------------------------------------------===//

int64_t getElementBytes(Type type) {
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.getWidth() / 8;
  if (auto intType = dyn_cast<IntegerType>(type))
    return (intType.getWidth() + 7) / 8;
  return 4;
}

//===----------------------------------------------------------------------===//
// computeBufferSizeFromMemRef
//===----------------------------------------------------------------------===//

int64_t computeBufferSizeFromMemRef(MemRefType memrefType) {
  // Compute the maximum byte offset reachable via this memref's layout.
  // For a memref with strides [s0, s1, …, 1] and dims [d0, d1, …, dN]:
  //   max_offset = (d0-1)*s0 + (d1-1)*s1 + … + (dN-1)*1 + 1  (in elements)
  //   num_records = max_offset * elemBytes
  // If any dimension or stride is dynamic, fall back to a large bound.
  int64_t elemBytes = getElementBytes(memrefType.getElementType());

  SmallVector<int64_t, 4> strides;
  int64_t offset;
  if (succeeded(memrefType.getStridesAndOffset(strides, offset)) &&
      memrefType.hasStaticShape()) {
    bool allStaticStrides = llvm::all_of(
        strides, [](int64_t s) { return s != ShapedType::kDynamic; });
    if (allStaticStrides) {
      int64_t maxOffset = 0;
      auto shape = memrefType.getShape();
      for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] > 1)
          maxOffset += (shape[i] - 1) * std::abs(strides[i]);
      }
      return (maxOffset + 1) * elemBytes;
    }
  }
  // Dynamic dims or dynamic strides — use a large-but-bounded value so that
  // hardware OOB checking catches wild offsets (e.g., negative wraparound)
  // and returns zero rather than faulting on unmapped memory.
  return 0x20000000;
}

//===----------------------------------------------------------------------===//
// isPowerOf2 / log2
//===----------------------------------------------------------------------===//

bool isPowerOf2(int64_t val) { return val > 0 && (val & (val - 1)) == 0; }

int64_t log2(int64_t val) {
  int64_t result = 0;
  while ((1LL << result) < val)
    ++result;
  return result;
}

//===----------------------------------------------------------------------===//
// getArithConstantValue
//===----------------------------------------------------------------------===//

std::optional<int64_t> getArithConstantValue(Value val) {
  IntegerAttr attr;
  if (mlir::matchPattern(val, mlir::m_Constant(&attr)))
    return attr.getInt();
  return std::nullopt;
}

} // namespace waveasm
