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

    // TODO: for now, just assign registers sequentially. In the future,
    // we need a liveness analysis to assign registers.
    unsigned nextRegister = 0;

    WalkResult result = func->walk([&](memref::AllocaOp allocaOp) {
      auto memrefType = allocaOp.getType();
      if (!isInRegisterSpace(memrefType))
        return WalkResult::advance();

      auto regCountOr = getRegisterCount(memrefType);
      if (failed(regCountOr)) {
        allocaOp->emitError(
            "Cannot allocate dynamic-sized memref in register space");
        return WalkResult::interrupt();
      }

      unsigned regCount = *regCountOr;

      // Assign starting register number.
      allocaOp->setAttr(
          "water.register_number",
          IntegerAttr::get(IntegerType::get(ctx, 32), nextRegister));

      // Track how many registers this alloca uses.
      allocaOp->setAttr("water.register_count",
                        IntegerAttr::get(IntegerType::get(ctx, 32), regCount));

      // Advance to next available register.
      nextRegister += regCount;

      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      return signalPassFailure();

    // Attach metadata to function with total register count.
    func->setAttr("water.total_registers",
                  IntegerAttr::get(IntegerType::get(ctx, 32), nextRegister));
  }
};

} // namespace
