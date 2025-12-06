// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::water {
#define GEN_PASS_DEF_WATERLOWERMEMORYOPS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

namespace {

/// Get the AMDGPU global_load instruction suffix based on bit width
static StringRef getGlobalLoadSuffix(unsigned bitWidth) {
  switch (bitWidth) {
  case 32:
    return "b32";
  case 64:
    return "b64";
  case 96:
    return "b96";
  case 128:
    return "b128";
  default:
    return "";
  }
}

/// Get the AMDGPU global_store instruction suffix based on bit width
static StringRef getGlobalStoreSuffix(unsigned bitWidth) {
  return getGlobalLoadSuffix(bitWidth);
}

/// Lower vector.load to LLVM inline assembly (global_load_*)
static void lowerVectorLoad(vector::LoadOp loadOp, IRRewriter &rewriter) {
  auto vectorType = loadOp.getVectorType();
  unsigned bitWidth =
      vectorType.getNumElements() * vectorType.getElementTypeBitWidth();

  StringRef suffix = getGlobalLoadSuffix(bitWidth);
  if (suffix.empty())
    return;

  Location loc = loadOp.getLoc();

  // Build the inline assembly string: "global_load_b64 $0, $1, off"
  std::string asmStr = ("global_load_" + suffix + " $0, $1, off").str();

  // Constraints: "=v" for output (VGPR), "v" for input address (VGPR)
  std::string constraints = "=v,v";

  // Get the base pointer - need to convert memref to pointer
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto i64Type = rewriter.getI64Type();

  rewriter.setInsertionPoint(loadOp);

  // Extract pointer as index, cast to i64, then to ptr
  Value basePtr = memref::ExtractAlignedPointerAsIndexOp::create(
      rewriter, loc, loadOp.getBase());
  basePtr = arith::IndexCastOp::create(rewriter, loc, i64Type, basePtr);
  basePtr = LLVM::IntToPtrOp::create(rewriter, loc, ptrType, basePtr);

  // Create the inline assembly operation
  auto asmOp = LLVM::InlineAsmOp::create(
      rewriter, loc,
      /*resultTypes=*/vectorType,
      /*operands=*/ValueRange{basePtr},
      /*asm_string=*/asmStr,
      /*constraints=*/constraints,
      /*has_side_effects=*/false,
      /*is_align_stack=*/false,
      /*tail_call_kind=*/LLVM::tailcallkind::TailCallKind::None,
      /*asm_dialect=*/LLVM::AsmDialectAttr{},
      /*operand_attrs=*/ArrayAttr{});

  rewriter.replaceOp(loadOp, asmOp.getResult(0));
}

/// Lower vector.store to LLVM inline assembly (global_store_*)
static void lowerVectorStore(vector::StoreOp storeOp, IRRewriter &rewriter) {
  auto vectorType = cast<VectorType>(storeOp.getValueToStore().getType());
  unsigned bitWidth =
      vectorType.getNumElements() * vectorType.getElementTypeBitWidth();

  StringRef suffix = getGlobalStoreSuffix(bitWidth);
  if (suffix.empty())
    return;

  Location loc = storeOp.getLoc();

  // Build the inline assembly string: "global_store_b64 $0, $1, off"
  std::string asmStr = ("global_store_" + suffix + " $0, $1, off").str();

  // Constraints: "v" for address (VGPR), "v" for data (VGPR)
  std::string constraints = "v,v";

  // Get the base pointer
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto i64Type = rewriter.getI64Type();

  rewriter.setInsertionPoint(storeOp);

  // Extract pointer as index, cast to i64, then to ptr
  Value basePtr = memref::ExtractAlignedPointerAsIndexOp::create(
      rewriter, loc, storeOp.getBase());
  basePtr = arith::IndexCastOp::create(rewriter, loc, i64Type, basePtr);
  basePtr = LLVM::IntToPtrOp::create(rewriter, loc, ptrType, basePtr);

  // Create the inline assembly operation (no result for store)
  LLVM::InlineAsmOp::create(
      rewriter, loc,
      /*resultTypes=*/TypeRange{},
      /*operands=*/ValueRange{basePtr, storeOp.getValueToStore()},
      /*asm_string=*/asmStr,
      /*constraints=*/constraints,
      /*has_side_effects=*/true,
      /*is_align_stack=*/false,
      /*tail_call_kind=*/LLVM::tailcallkind::TailCallKind::None,
      /*asm_dialect=*/LLVM::AsmDialectAttr{},
      /*operand_attrs=*/ArrayAttr{});

  rewriter.eraseOp(storeOp);
}

/// Pass that lowers high-level memory operations to LLVM inline assembly
/// for AMDGPU global memory instructions.
class WaterLowerMemoryOpsPass
    : public water::impl::WaterLowerMemoryOpsBase<WaterLowerMemoryOpsPass> {
public:
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());

    getOperation()->walk([&](Operation *op) {
      if (auto loadOp = dyn_cast<vector::LoadOp>(op)) {
        lowerVectorLoad(loadOp, rewriter);
      } else if (auto storeOp = dyn_cast<vector::StoreOp>(op)) {
        lowerVectorStore(storeOp, rewriter);
      }
    });
  }
};

} // namespace
