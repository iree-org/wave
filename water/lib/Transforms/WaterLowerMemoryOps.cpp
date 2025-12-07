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

/// Get the AMDGPU instruction suffix based on bit width
static FailureOr<StringRef> getSizeSuffix(unsigned bitWidth) {
  switch (bitWidth) {
  case 32:
    return StringRef("b32");
  case 64:
    return StringRef("b64");
  case 96:
    return StringRef("b96");
  case 128:
    return StringRef("b128");
  default:
    return failure();
  }
}

/// Create an LLVM inline assembly operation with standard attributes
static LLVM::InlineAsmOp createInlineAsm(IRRewriter &rewriter, Location loc,
                                         TypeRange resultTypes,
                                         ValueRange operands, StringRef asmStr,
                                         StringRef constraints,
                                         bool hasSideEffects) {
  return LLVM::InlineAsmOp::create(
      rewriter, loc, resultTypes, operands, asmStr, constraints, hasSideEffects,
      /*is_align_stack=*/false,
      /*tail_call_kind=*/LLVM::tailcallkind::TailCallKind::None,
      /*asm_dialect=*/LLVM::AsmDialectAttr{},
      /*operand_attrs=*/ArrayAttr{});
}

/// Compute the final address for a memref access with indices
static Value computeMemrefAddress(IRRewriter &rewriter, Location loc,
                                  Value memref, ValueRange indices,
                                  unsigned elementBitWidth) {
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto i64Type = rewriter.getI64Type();

  // Extract strided metadata to get base pointer, offset, sizes, and strides
  auto metadataOp =
      memref::ExtractStridedMetadataOp::create(rewriter, loc, memref);
  Value basePtr = metadataOp.getBaseBuffer();
  Value offset = metadataOp.getOffset();

  // Compute linear index from multidimensional indices
  Value linearIndex = offset;
  for (auto i : llvm::seq<size_t>(0, indices.size())) {
    Value stride = metadataOp.getStrides()[i];
    Value indexTimesStride = arith::MulIOp::create(
        rewriter, loc, indices[i], stride, arith::IntegerOverflowFlags::nsw);
    linearIndex =
        arith::AddIOp::create(rewriter, loc, linearIndex, indexTimesStride,
                              arith::IntegerOverflowFlags::nsw);
  }

  // Convert base pointer to i64
  Value basePtrInt =
      memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, basePtr);
  basePtrInt = arith::IndexCastOp::create(rewriter, loc, i64Type, basePtrInt);

  // Convert linear index to i64 and scale by element size
  unsigned elementBytes = elementBitWidth / 8;
  Value elementSize =
      arith::ConstantIndexOp::create(rewriter, loc, elementBytes);
  Value byteOffset =
      arith::MulIOp::create(rewriter, loc, linearIndex, elementSize,
                            arith::IntegerOverflowFlags::nsw);
  Value byteOffsetI64 =
      arith::IndexCastOp::create(rewriter, loc, i64Type, byteOffset);

  // Add byte offset to base pointer
  Value finalAddr =
      arith::AddIOp::create(rewriter, loc, basePtrInt, byteOffsetI64,
                            arith::IntegerOverflowFlags::nsw);
  return LLVM::IntToPtrOp::create(rewriter, loc, ptrType, finalAddr);
}

/// Lower vector.load to LLVM inline assembly (global_load_*)
static LogicalResult lowerVectorLoad(vector::LoadOp loadOp,
                                     IRRewriter &rewriter) {
  auto vectorType = loadOp.getVectorType();
  unsigned bitWidth =
      vectorType.getNumElements() * vectorType.getElementTypeBitWidth();

  FailureOr<StringRef> suffix = getSizeSuffix(bitWidth);
  if (failed(suffix))
    return loadOp.emitError("unsupported vector load bit width: ") << bitWidth;

  Location loc = loadOp.getLoc();

  // Build the inline assembly string: "global_load_b64 $0, $1, off"
  std::string asmStr = ("global_load_" + *suffix + " $0, $1, off").str();

  // Constraints: "=v" for output (VGPR), "v" for input address (VGPR)
  StringRef constraints = "=v,v";

  rewriter.setInsertionPoint(loadOp);

  // Compute the final address
  Value addr =
      computeMemrefAddress(rewriter, loc, loadOp.getBase(), loadOp.getIndices(),
                           vectorType.getElementTypeBitWidth());

  // Create the inline assembly operation
  auto asmOp = createInlineAsm(rewriter, loc, vectorType, ValueRange{addr},
                               asmStr, constraints, /*hasSideEffects=*/true);

  rewriter.replaceOp(loadOp, asmOp.getResult(0));
  return success();
}

/// Lower vector.store to LLVM inline assembly (global_store_*)
static LogicalResult lowerVectorStore(vector::StoreOp storeOp,
                                      IRRewriter &rewriter) {
  auto vectorType = cast<VectorType>(storeOp.getValueToStore().getType());
  unsigned bitWidth =
      vectorType.getNumElements() * vectorType.getElementTypeBitWidth();

  FailureOr<StringRef> suffix = getSizeSuffix(bitWidth);
  if (failed(suffix))
    return storeOp.emitError("unsupported vector store bit width: ")
           << bitWidth;

  Location loc = storeOp.getLoc();

  // Build the inline assembly string: "global_store_b64 $0, $1, off"
  std::string asmStr = ("global_store_" + *suffix + " $0, $1, off").str();

  // Constraints: "v" for address (VGPR), "v" for data (VGPR)
  StringRef constraints = "v,v";

  rewriter.setInsertionPoint(storeOp);

  // Compute the final address
  Value addr = computeMemrefAddress(rewriter, loc, storeOp.getBase(),
                                    storeOp.getIndices(),
                                    vectorType.getElementTypeBitWidth());

  // Create the inline assembly operation (no result for store)
  createInlineAsm(rewriter, loc, TypeRange{},
                  ValueRange{addr, storeOp.getValueToStore()}, asmStr,
                  constraints, /*hasSideEffects=*/true);

  rewriter.eraseOp(storeOp);
  return success();
}

/// Wrapper functions for operation lowering
static LogicalResult lowerLoadOp(Operation *op, IRRewriter &rewriter) {
  return lowerVectorLoad(cast<vector::LoadOp>(op), rewriter);
}

static LogicalResult lowerStoreOp(Operation *op, IRRewriter &rewriter) {
  return lowerVectorStore(cast<vector::StoreOp>(op), rewriter);
}

/// Operation lowering handler entry
struct OpLoweringHandler {
  TypeID typeID;
  LogicalResult (*lowerFn)(Operation *, IRRewriter &);
};

/// Table of lowering handlers for different operation types
static const OpLoweringHandler loweringHandlers[] = {
    {TypeID::get<vector::LoadOp>(), lowerLoadOp},
    {TypeID::get<vector::StoreOp>(), lowerStoreOp},
};

/// Pass that lowers high-level memory operations to LLVM inline assembly
/// for AMDGPU global memory instructions.
class WaterLowerMemoryOpsPass
    : public water::impl::WaterLowerMemoryOpsBase<WaterLowerMemoryOpsPass> {
public:
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());

    auto walkFn = [&](Operation *op) {
      TypeID opTypeID = op->getName().getTypeID();
      for (const auto &handler : loweringHandlers) {
        if (handler.typeID == opTypeID) {
          if (failed(handler.lowerFn(op, rewriter)))
            return WalkResult::interrupt();
          return WalkResult::advance();
        }
      }
      return WalkResult::advance();
    };

    if (getOperation()->walk(walkFn).wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
