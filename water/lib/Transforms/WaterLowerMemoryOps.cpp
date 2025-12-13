// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
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

/// Get the AMDGPU instruction suffix based on bit width (for loads - unsigned)
static FailureOr<StringRef> getSizeSuffixLoad(unsigned bitWidth) {
  switch (bitWidth) {
  case 8:
    return StringRef("u8");
  case 16:
    return StringRef("u16");
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

/// Get the AMDGPU instruction suffix based on bit width (for stores)
static FailureOr<StringRef> getSizeSuffixStore(unsigned bitWidth) {
  switch (bitWidth) {
  case 8:
    return StringRef("b8");
  case 16:
    return StringRef("b16");
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

/// Detect if chipset is RDNA architecture
static bool isRDNA(StringRef chipset) {
  return chipset.starts_with("gfx11") || chipset.starts_with("gfx12");
}

/// Compute byte offset as iX for a memref access with indices
template <int Bits>
static Value computeMemrefByteOffset(IRRewriter &rewriter, Location loc,
                                     Value memref, ValueRange indices,
                                     unsigned elementBitWidth) {
  // Extract strided metadata to get offset and strides
  auto metadataOp =
      memref::ExtractStridedMetadataOp::create(rewriter, loc, memref);
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

  // Convert linear index to byte offset
  unsigned elementBytes = elementBitWidth / 8;
  Value elementSize =
      arith::ConstantIndexOp::create(rewriter, loc, elementBytes);
  Value byteOffset =
      arith::MulIOp::create(rewriter, loc, linearIndex, elementSize,
                            arith::IntegerOverflowFlags::nsw);

  Type indexType = IntegerType::get(rewriter.getContext(), Bits);
  return arith::IndexCastOp::create(rewriter, loc, indexType, byteOffset);
}

/// Compute the final address for a memref access with indices (for global
/// operations)
static Value computeMemrefAddress(IRRewriter &rewriter, Location loc,
                                  Value memref, ValueRange indices,
                                  unsigned elementBitWidth) {
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto i64Type = rewriter.getI64Type();

  // Extract base pointer
  auto metadataOp =
      memref::ExtractStridedMetadataOp::create(rewriter, loc, memref);
  Value basePtr = metadataOp.getBaseBuffer();

  // Convert base pointer to i64
  Value basePtrInt =
      memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, basePtr);
  basePtrInt = arith::IndexCastOp::create(rewriter, loc, i64Type, basePtrInt);

  // Compute byte offset
  Value byteOffsetI64 = computeMemrefByteOffset<64>(rewriter, loc, memref,
                                                    indices, elementBitWidth);

  // Add byte offset to base pointer
  Value finalAddr =
      arith::AddIOp::create(rewriter, loc, basePtrInt, byteOffsetI64,
                            arith::IntegerOverflowFlags::nsw);
  return LLVM::IntToPtrOp::create(rewriter, loc, ptrType, finalAddr);
}

/// Get buffer instruction suffix based on bit width (for loads - unsigned)
static FailureOr<StringRef> getBufferSuffixLoad(unsigned bitWidth,
                                                bool isRDNAArch) {
  if (isRDNAArch) {
    // RDNA uses b32, b64, etc.
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
  } else {
    // CDNA uses dword, dwordx2, etc.
    switch (bitWidth) {
    case 8:
      return StringRef("ubyte");
    case 16:
      return StringRef("ushort");
    case 32:
      return StringRef("dword");
    case 64:
      return StringRef("dwordx2");
    case 96:
      return StringRef("dwordx3");
    case 128:
      return StringRef("dwordx4");
    default:
      return failure();
    }
  }
}

/// Get buffer instruction suffix based on bit width (for stores)
static FailureOr<StringRef> getBufferSuffixStore(unsigned bitWidth,
                                                 bool isRDNAArch) {
  if (isRDNAArch) {
    // RDNA uses b32, b64, etc.
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
  } else {
    // CDNA uses dword, dwordx2, etc.
    switch (bitWidth) {
    case 8:
      return StringRef("byte");
    case 16:
      return StringRef("short");
    case 32:
      return StringRef("dword");
    case 64:
      return StringRef("dwordx2");
    case 96:
      return StringRef("dwordx3");
    case 128:
      return StringRef("dwordx4");
    default:
      return failure();
    }
  }
}

/// Extract buffer descriptor and base offset from a fat_raw_buffer memref
/// addrspace(7) format: {<4 x i32> rsrc, i32 offset} (160 bits total)
/// Returns: {resource descriptor (i128), base offset (i32)}
static std::pair<Value, Value>
extractBufferDescriptor(IRRewriter &rewriter, Location loc, Value memref) {
  // Create proper memref descriptor struct type: {ptr, ptr, offset,
  // sizes[rank], strides[rank]}
  auto memrefType = cast<MemRefType>(memref.getType());
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 7);
  auto i32Type = rewriter.getI32Type();
  auto i64Type = rewriter.getI64Type();
  auto arrayType = LLVM::LLVMArrayType::get(i64Type, memrefType.getRank());
  Type descriptorFields[] = {ptrType, ptrType, i64Type, arrayType, arrayType};

  auto memrefDescType =
      LLVM::LLVMStructType::getLiteral(rewriter.getContext(), descriptorFields);

  Value memrefDescVal =
      UnrealizedConversionCastOp::create(rewriter, loc, memrefDescType, memref)
          .getResult(0);

  MemRefDescriptor memrefDesc(memrefDescVal);
  Value bufferPtr = memrefDesc.alignedPtr(rewriter, loc);
  Value bufferOffset = memrefDesc.offset(rewriter, loc);
  bufferOffset = arith::TruncIOp::create(rewriter, loc, i32Type, bufferOffset);

  // Convert to i160 to access full buffer descriptor {<4 x i32> rsrc, i32
  // offset}
  auto i160Type = IntegerType::get(rewriter.getContext(), 160);
  Value fullDesc = LLVM::PtrToIntOp::create(rewriter, loc, i160Type, bufferPtr);

  // Extract lower 32 bits for base offset
  Value baseOffset = arith::TruncIOp::create(rewriter, loc, i32Type, fullDesc);

  baseOffset = arith::AddIOp::create(rewriter, loc, baseOffset, bufferOffset,
                                     arith::IntegerOverflowFlags::nsw);

  // Extract upper 128 bits for resource descriptor
  auto c32 = arith::ConstantIntOp::create(rewriter, loc, i160Type, 32);
  Value rsrcBits160 = arith::ShRUIOp::create(rewriter, loc, fullDesc, c32);
  auto i128Type = IntegerType::get(rewriter.getContext(), 128);
  Value rsrcBits =
      arith::TruncIOp::create(rewriter, loc, i128Type, rsrcBits160);

  return {rsrcBits, baseOffset};
}

/// Helper to get memref, result type, and bit width from load operation
template <typename LoadOpTy>
static std::tuple<Value, Type, unsigned> getLoadOpInfo(LoadOpTy loadOp) {
  if constexpr (std::is_same_v<LoadOpTy, vector::LoadOp>) {
    auto vectorType = loadOp.getVectorType();
    unsigned bitWidth =
        vectorType.getNumElements() * vectorType.getElementTypeBitWidth();
    return {loadOp.getBase(), vectorType, bitWidth};
  } else {
    auto elementType = loadOp.getResult().getType();
    unsigned bitWidth = elementType.getIntOrFloatBitWidth();
    return {loadOp.getMemRef(), elementType, bitWidth};
  }
}

/// Helper to get memref, value type, and bit width from store operation
template <typename StoreOpTy>
static std::tuple<Value, Type, unsigned> getStoreOpInfo(StoreOpTy storeOp) {
  if constexpr (std::is_same_v<StoreOpTy, vector::StoreOp>) {
    auto vectorType = cast<VectorType>(storeOp.getValueToStore().getType());
    unsigned bitWidth =
        vectorType.getNumElements() * vectorType.getElementTypeBitWidth();
    return {storeOp.getBase(), vectorType, bitWidth};
  } else {
    auto elementType = storeOp.getValueToStore().getType();
    unsigned bitWidth = elementType.getIntOrFloatBitWidth();
    return {storeOp.getMemRef(), elementType, bitWidth};
  }
}

/// Lower vector/scalar load to AMDGPU buffer load inline assembly
template <typename LoadOpTy>
static LogicalResult lowerLoadBuffer(LoadOpTy loadOp, IRRewriter &rewriter,
                                     bool isRDNAArch) {
  auto [memref, resultType, bitWidth] = getLoadOpInfo(loadOp);

  if (bitWidth < 32)
    return success();

  FailureOr<StringRef> suffix = getBufferSuffixLoad(bitWidth, isRDNAArch);
  if (failed(suffix))
    return loadOp.emitError("unsupported buffer load bit width: ") << bitWidth;

  Location loc = loadOp.getLoc();
  rewriter.setInsertionPoint(loadOp);

  // Build inline assembly: "buffer_load_<suffix> $0, $1, $2, 0 offen"
  std::string asmStr =
      ("buffer_load_" + *suffix + " $0, $1, $2, 0 offen").str();

  // Constraints: "=v" for output (VGPR), "v" for offset (VGPR), "s" for
  // descriptor (SGPR[4])
  StringRef constraints = "=v,v,s";

  // Compute byte offset from indices
  unsigned elementBitWidth =
      std::is_same_v<LoadOpTy, vector::LoadOp>
          ? cast<VectorType>(resultType).getElementTypeBitWidth()
          : bitWidth;
  Value offset = computeMemrefByteOffset<32>(
      rewriter, loc, memref, loadOp.getIndices(), elementBitWidth);

  // Extract buffer descriptor and base offset from memref
  auto [bufferDesc, baseOffset] =
      extractBufferDescriptor(rewriter, loc, memref);

  // Add base offset to computed offset
  Value finalOffset = arith::AddIOp::create(rewriter, loc, offset, baseOffset,
                                            arith::IntegerOverflowFlags::nsw);

  // Create inline assembly operation with result type directly
  auto asmOp = createInlineAsm(rewriter, loc, resultType,
                               ValueRange{finalOffset, bufferDesc}, asmStr,
                               constraints, /*hasSideEffects=*/true);

  rewriter.replaceOp(loadOp, asmOp.getResult(0));
  return success();
}

/// Lower vector/scalar load to LLVM inline assembly (global_load_*)
template <typename LoadOpTy>
static LogicalResult lowerLoadGlobal(LoadOpTy loadOp, IRRewriter &rewriter) {
  auto [memref, resultType, bitWidth] = getLoadOpInfo(loadOp);

  if (bitWidth < 32)
    return success();

  FailureOr<StringRef> suffix = getSizeSuffixLoad(bitWidth);
  if (failed(suffix))
    return loadOp.emitError("unsupported load bit width: ") << bitWidth;

  Location loc = loadOp.getLoc();

  // Build the inline assembly string: "global_load_b64 $0, $1, off"
  std::string asmStr = ("global_load_" + *suffix + " $0, $1, off").str();

  // Constraints: "=v" for output (VGPR), "v" for input address (VGPR)
  StringRef constraints = "=v,v";

  rewriter.setInsertionPoint(loadOp);

  // Compute the final address
  unsigned elementBitWidth =
      std::is_same_v<LoadOpTy, vector::LoadOp>
          ? cast<VectorType>(resultType).getElementTypeBitWidth()
          : bitWidth;
  Value addr = computeMemrefAddress(rewriter, loc, memref, loadOp.getIndices(),
                                    elementBitWidth);

  // Create the inline assembly operation with result type directly
  auto asmOp = createInlineAsm(rewriter, loc, resultType, ValueRange{addr},
                               asmStr, constraints, /*hasSideEffects=*/true);

  rewriter.replaceOp(loadOp, asmOp.getResult(0));
  return success();
}

/// Lower vector/scalar store to AMDGPU buffer store inline assembly
template <typename StoreOpTy>
static LogicalResult lowerStoreBuffer(StoreOpTy storeOp, IRRewriter &rewriter,
                                      bool isRDNAArch) {
  auto [memref, valueType, bitWidth] = getStoreOpInfo(storeOp);

  if (bitWidth < 32)
    return success();

  FailureOr<StringRef> suffix = getBufferSuffixStore(bitWidth, isRDNAArch);
  if (failed(suffix))
    return storeOp.emitError("unsupported buffer store bit width: ")
           << bitWidth;

  Location loc = storeOp.getLoc();
  rewriter.setInsertionPoint(storeOp);

  // Build inline assembly: "buffer_store_<suffix> $0, $1, $2, 0 offen"
  std::string asmStr =
      ("buffer_store_" + *suffix + " $0, $1, $2, 0 offen").str();

  // Constraints: "v" for data (VGPR), "v" for offset (VGPR), "s" for descriptor
  // (SGPR[4])
  StringRef constraints = "v,v,s";

  // Compute byte offset from indices
  unsigned elementBitWidth =
      std::is_same_v<StoreOpTy, vector::StoreOp>
          ? cast<VectorType>(valueType).getElementTypeBitWidth()
          : bitWidth;
  Value offset = computeMemrefByteOffset<32>(
      rewriter, loc, memref, storeOp.getIndices(), elementBitWidth);

  // Extract buffer descriptor and base offset from memref
  auto [bufferDesc, baseOffset] =
      extractBufferDescriptor(rewriter, loc, memref);

  // Add base offset to computed offset
  Value finalOffset = arith::AddIOp::create(rewriter, loc, offset, baseOffset,
                                            arith::IntegerOverflowFlags::nsw);

  Value valueToStore = storeOp.getValueToStore();

  // Create inline assembly operation (no result for store)
  createInlineAsm(rewriter, loc, TypeRange{},
                  ValueRange{valueToStore, finalOffset, bufferDesc}, asmStr,
                  constraints, /*hasSideEffects=*/true);

  rewriter.eraseOp(storeOp);
  return success();
}

/// Lower vector/scalar store to LLVM inline assembly (global_store_*)
template <typename StoreOpTy>
static LogicalResult lowerStoreGlobal(StoreOpTy storeOp, IRRewriter &rewriter) {
  auto [memref, valueType, bitWidth] = getStoreOpInfo(storeOp);

  if (bitWidth < 32)
    return success();

  FailureOr<StringRef> suffix = getSizeSuffixStore(bitWidth);
  if (failed(suffix))
    return storeOp.emitError("unsupported store bit width: ") << bitWidth;

  Location loc = storeOp.getLoc();

  // Build the inline assembly string: "global_store_b64 $0, $1, off"
  std::string asmStr = ("global_store_" + *suffix + " $0, $1, off").str();

  // Constraints: "v" for address (VGPR), "v" for data (VGPR)
  StringRef constraints = "v,v";

  rewriter.setInsertionPoint(storeOp);

  // Compute the final address
  unsigned elementBitWidth =
      std::is_same_v<StoreOpTy, vector::StoreOp>
          ? cast<VectorType>(valueType).getElementTypeBitWidth()
          : bitWidth;
  Value addr = computeMemrefAddress(rewriter, loc, memref, storeOp.getIndices(),
                                    elementBitWidth);

  Value valueToStore = storeOp.getValueToStore();

  // Create the inline assembly operation (no result for store)
  createInlineAsm(rewriter, loc, TypeRange{}, ValueRange{addr, valueToStore},
                  asmStr, constraints,
                  /*hasSideEffects=*/true);

  rewriter.eraseOp(storeOp);
  return success();
}

/// Lower vector/scalar load to AMDGPU DS load inline assembly
template <typename LoadOpTy>
static LogicalResult lowerLoadDS(LoadOpTy loadOp, IRRewriter &rewriter) {
  auto [memref, resultType, bitWidth] = getLoadOpInfo(loadOp);

  if (bitWidth < 32)
    return success();

  FailureOr<StringRef> suffix = getSizeSuffixLoad(bitWidth);
  if (failed(suffix))
    return loadOp.emitError("unsupported DS load bit width: ") << bitWidth;

  Location loc = loadOp.getLoc();
  rewriter.setInsertionPoint(loadOp);

  // Build inline assembly: "ds_read_b32 $0, $1"
  std::string asmStr = ("ds_read_" + *suffix + " $0, $1").str();

  // Constraints: "=v" for output (VGPR), "v" for address (VGPR)
  StringRef constraints = "=v,v";

  // Compute byte offset as i64
  unsigned elementBitWidth =
      std::is_same_v<LoadOpTy, vector::LoadOp>
          ? cast<VectorType>(resultType).getElementTypeBitWidth()
          : bitWidth;
  Value offset = computeMemrefByteOffset<32>(
      rewriter, loc, memref, loadOp.getIndices(), elementBitWidth);

  // Create inline assembly operation (DS operations use 32-bit addresses)
  auto asmOp = createInlineAsm(rewriter, loc, resultType, ValueRange{offset},
                               asmStr, constraints, /*hasSideEffects=*/true);

  rewriter.replaceOp(loadOp, asmOp.getResult(0));
  return success();
}

/// Lower vector/scalar store to AMDGPU DS store inline assembly
template <typename StoreOpTy>
static LogicalResult lowerStoreDS(StoreOpTy storeOp, IRRewriter &rewriter) {
  auto [memref, valueType, bitWidth] = getStoreOpInfo(storeOp);

  if (bitWidth < 32)
    return success();

  FailureOr<StringRef> suffix = getSizeSuffixStore(bitWidth);
  if (failed(suffix))
    return storeOp.emitError("unsupported DS store bit width: ") << bitWidth;

  Location loc = storeOp.getLoc();
  rewriter.setInsertionPoint(storeOp);

  // Build inline assembly: "ds_write_b32 $0, $1"
  std::string asmStr = ("ds_write_" + *suffix + " $0, $1").str();

  // Constraints: "v" for address (VGPR), "v" for data (VGPR)
  StringRef constraints = "v,v";

  // Compute byte offset as i64
  unsigned elementBitWidth =
      std::is_same_v<StoreOpTy, vector::StoreOp>
          ? cast<VectorType>(valueType).getElementTypeBitWidth()
          : bitWidth;
  Value offset = computeMemrefByteOffset<32>(
      rewriter, loc, memref, storeOp.getIndices(), elementBitWidth);

  Value valueToStore = storeOp.getValueToStore();

  // Create inline assembly operation (no result for store, DS uses 32-bit
  // addresses)
  createInlineAsm(rewriter, loc, TypeRange{}, ValueRange{offset, valueToStore},
                  asmStr, constraints,
                  /*hasSideEffects=*/true);

  rewriter.eraseOp(storeOp);
  return success();
}

/// Check if a memref uses AMDGPU fat_raw_buffer address space
static bool usesBufferAddressSpace(Value memref) {
  auto memrefType = cast<MemRefType>(memref.getType());
  auto memorySpace = memrefType.getMemorySpace();

  if (!memorySpace)
    return false;

  // Check for #amdgpu.address_space<fat_raw_buffer> attribute
  if (auto enumAttr = dyn_cast<amdgpu::AddressSpaceAttr>(memorySpace))
    return enumAttr.getValue() == amdgpu::AddressSpace::FatRawBuffer;

  return false;
}

/// Check if a memref uses workgroup (LDS) address space
static bool usesWorkgroupAddressSpace(Value memref) {
  auto memrefType = cast<MemRefType>(memref.getType());
  auto memorySpace = memrefType.getMemorySpace();

  if (!memorySpace)
    return false;

  // Check for #gpu.address_space<workgroup> attribute
  if (auto enumAttr = dyn_cast<gpu::AddressSpaceAttr>(memorySpace))
    return enumAttr.getValue() == gpu::AddressSpace::Workgroup;

  return false;
}

/// Pass that lowers high-level memory operations to AMDGPU memory instructions.
/// Uses buffer operations for memrefs with
/// #amdgpu.address_space<fat_raw_buffer>, DS operations for memrefs with
/// #gpu.address_space<workgroup>, and global operations for all other memrefs.
class WaterLowerMemoryOpsPass
    : public water::impl::WaterLowerMemoryOpsBase<WaterLowerMemoryOpsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());

    // Determine if we're targeting RDNA architecture
    bool isRDNAArch = isRDNA(chipset);

    // Helper to dispatch to the appropriate lowering function based on address
    // space
    auto lowerMemoryOp = [&](Value base, auto lowerBuffer, auto lowerWorkgroup,
                             auto lowerGlobal) -> LogicalResult {
      if (usesBufferAddressSpace(base))
        return lowerBuffer();
      if (usesWorkgroupAddressSpace(base))
        return lowerWorkgroup();
      return lowerGlobal();
    };

    auto walkFn = [&](Operation *op) {
      if (auto loadOp = dyn_cast<vector::LoadOp>(op)) {
        LogicalResult result = lowerMemoryOp(
            loadOp.getBase(),
            [&]() { return lowerLoadBuffer(loadOp, rewriter, isRDNAArch); },
            [&]() { return lowerLoadDS(loadOp, rewriter); },
            [&]() { return lowerLoadGlobal(loadOp, rewriter); });
        if (failed(result))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto storeOp = dyn_cast<vector::StoreOp>(op)) {
        LogicalResult result = lowerMemoryOp(
            storeOp.getBase(),
            [&]() { return lowerStoreBuffer(storeOp, rewriter, isRDNAArch); },
            [&]() { return lowerStoreDS(storeOp, rewriter); },
            [&]() { return lowerStoreGlobal(storeOp, rewriter); });
        if (failed(result))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
        LogicalResult result = lowerMemoryOp(
            loadOp.getMemRef(),
            [&]() { return lowerLoadBuffer(loadOp, rewriter, isRDNAArch); },
            [&]() { return lowerLoadDS(loadOp, rewriter); },
            [&]() { return lowerLoadGlobal(loadOp, rewriter); });
        if (failed(result))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
        LogicalResult result = lowerMemoryOp(
            storeOp.getMemRef(),
            [&]() { return lowerStoreBuffer(storeOp, rewriter, isRDNAArch); },
            [&]() { return lowerStoreDS(storeOp, rewriter); },
            [&]() { return lowerStoreGlobal(storeOp, rewriter); });
        if (failed(result))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      return WalkResult::advance();
    };

    if (getOperation()->walk(walkFn).wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
