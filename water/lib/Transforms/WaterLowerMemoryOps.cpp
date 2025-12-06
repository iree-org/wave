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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

/// Pattern to lower vector.load to LLVM inline assembly (global_load_*)
struct VectorLoadToInlineAsmPattern : public OpRewritePattern<vector::LoadOp> {
  using OpRewritePattern<vector::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto vectorType = loadOp.getVectorType();
    unsigned bitWidth =
        vectorType.getNumElements() * vectorType.getElementTypeBitWidth();

    StringRef suffix = getGlobalLoadSuffix(bitWidth);
    if (suffix.empty())
      return failure();

    Location loc = loadOp.getLoc();

    // Build the inline assembly string: "global_load_b64 $0, $1, off"
    std::string asmStr = ("global_load_" + suffix + " $0, $1, off").str();

    // Constraints: "=v" for output (VGPR), "v" for input address (VGPR)
    std::string constraints = "=v,v";

    // Get the base pointer - need to convert memref to pointer
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64Type = rewriter.getI64Type();

    // Extract pointer as index, cast to i64, then to ptr
    Value basePtr = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, loadOp.getBase());
    basePtr = rewriter.create<arith::IndexCastOp>(loc, i64Type, basePtr);
    basePtr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, basePtr);

    // Create the inline assembly operation
    auto asmOp = rewriter.create<LLVM::InlineAsmOp>(
        loc,
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
    return success();
  }
};

/// Pattern to lower vector.store to LLVM inline assembly (global_store_*)
struct VectorStoreToInlineAsmPattern
    : public OpRewritePattern<vector::StoreOp> {
  using OpRewritePattern<vector::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto vectorType = cast<VectorType>(storeOp.getValueToStore().getType());
    unsigned bitWidth =
        vectorType.getNumElements() * vectorType.getElementTypeBitWidth();

    StringRef suffix = getGlobalStoreSuffix(bitWidth);
    if (suffix.empty())
      return failure();

    Location loc = storeOp.getLoc();

    // Build the inline assembly string: "global_store_b64 $0, $1, off"
    std::string asmStr = ("global_store_" + suffix + " $0, $1, off").str();

    // Constraints: "v" for address (VGPR), "v" for data (VGPR)
    std::string constraints = "v,v";

    // Get the base pointer
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64Type = rewriter.getI64Type();

    // Extract pointer as index, cast to i64, then to ptr
    Value basePtr = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, storeOp.getBase());
    basePtr = rewriter.create<arith::IndexCastOp>(loc, i64Type, basePtr);
    basePtr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, basePtr);

    // Create the inline assembly operation (no result for store)
    rewriter.create<LLVM::InlineAsmOp>(
        loc,
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
    return success();
  }
};

/// Pass that lowers high-level memory operations to LLVM inline assembly
/// for AMDGPU global memory instructions.
class WaterLowerMemoryOpsPass
    : public water::impl::WaterLowerMemoryOpsBase<WaterLowerMemoryOpsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Add patterns for lowering vector.load/store to inline assembly
    patterns.add<VectorLoadToInlineAsmPattern, VectorStoreToInlineAsmPattern>(
        context);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
