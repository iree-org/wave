// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::water {
#define GEN_PASS_DEF_WATERNUMBERREGISTERS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

namespace {

/// Check if a memref type is in virtual register space (memspace 128).
static bool isInRegisterSpace(MemRefType memrefType) {
  if (auto memSpace =
          dyn_cast_or_null<IntegerAttr>(memrefType.getMemorySpace()))
    return memSpace.getInt() == 128;
  return false;
}

/// Calculate the number of 32-bit registers needed for a memref type.
static FailureOr<unsigned> getRegisterCount(MemRefType memrefType) {
  // Calculate total size in bytes
  unsigned elementSizeBytes = memrefType.getElementTypeBitWidth() / 8;
  unsigned numElements = 1;
  for (int64_t dim : memrefType.getShape()) {
    if (dim == ShapedType::kDynamic)
      return failure(); // Can't allocate dynamic sizes in registers.

    numElements *= dim;
  }

  unsigned totalBytes = elementSizeBytes * numElements;

  // Each register is 32 bits = 4 bytes
  // Round up to next register boundary.
  return (totalBytes + 3) / 4;
}

/// Assign physical registers to register space allocas.
class WaterNumberRegistersPass
    : public water::impl::WaterNumberRegistersBase<WaterNumberRegistersPass> {
public:
  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext *ctx = &getContext();

    SmallVector<std::pair<unsigned, Operation *>> regCounts;

    Type i32 = IntegerType::get(ctx, 32);
    WalkResult result = func->walk([&](memref::AllocaOp allocaOp) {
      auto memrefType = allocaOp.getType();
      if (!isInRegisterSpace(memrefType))
        return WalkResult::advance();

      auto regCount = getRegisterCount(memrefType);
      if (failed(regCount)) {
        allocaOp->emitError(
            "Cannot allocate dynamic-sized memref in register space");
        return WalkResult::interrupt();
      }

      regCounts.emplace_back(*regCount, allocaOp);
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      return signalPassFailure();

    // Sort by register size to reduce register alignment gaps.
    llvm::stable_sort(regCounts, [](const std::pair<unsigned, Operation *> &a,
                                    const std::pair<unsigned, Operation *> &b) {
      return a.first < b.first;
    });

    // TODO: for now, just assign registers sequentially. In the future,
    // we need a liveness analysis to assign registers.
    unsigned nextRegister = 0;

    for (auto [regCount, op] : regCounts) {
      // Align to regCount boundary.
      nextRegister = ((nextRegister + regCount - 1) / regCount) * regCount;

      // Assign starting register number.
      op->setAttr("water.vgpr_number", IntegerAttr::get(i32, nextRegister));

      // Track how many registers this alloca uses.
      op->setAttr("water.vgpr_count", IntegerAttr::get(i32, regCount));

      // Advance to next available register.
      nextRegister += regCount;
    }

    // Attach metadata to function with total register count.
    func->setAttr("water.total_vgprs", IntegerAttr::get(i32, nextRegister));
  }
};

} // namespace
