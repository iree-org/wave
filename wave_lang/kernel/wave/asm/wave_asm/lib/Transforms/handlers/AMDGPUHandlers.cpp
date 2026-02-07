// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// AMDGPU, ROCDL, and IREE Dialect Handlers
//===----------------------------------------------------------------------===//

#include "Handlers.h"

#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

using namespace mlir;
using namespace waveasm;

namespace waveasm {

//===----------------------------------------------------------------------===//
// AMDGPU Dialect Handlers
//===----------------------------------------------------------------------===//

LogicalResult handleAMDGPULdsBarrier(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  S_WAITCNT_LGKMCNT::create(builder, loc, 0);
  S_BARRIER::create(builder, loc);

  return success();
}

LogicalResult handleAMDGPUMfma(Operation *op, TranslationContext &ctx) {
  auto mfmaOp = cast<amdgpu::MFMAOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  int64_t m = mfmaOp.getM();
  int64_t n = mfmaOp.getN();
  int64_t k = mfmaOp.getK();

  auto srcA = ctx.getMapper().getMapped(mfmaOp.getSourceA());
  auto srcB = ctx.getMapper().getMapped(mfmaOp.getSourceB());
  auto srcC = ctx.getMapper().getMapped(mfmaOp.getDestC());

  if (!srcA || !srcB || !srcC) {
    return op->emitError("MFMA operands not mapped");
  }

  Type srcAType = mfmaOp.getSourceA().getType();
  Type elemType;
  if (auto vecType = dyn_cast<VectorType>(srcAType)) {
    elemType = vecType.getElementType();
  } else {
    elemType = srcAType;
  }

  int64_t accSize = 4;
  if (m == 32 && n == 32) {
    accSize = 16;
  } else if (m == 4 && n == 4) {
    accSize = 4;
  } else if (m == 16 && n == 16 && k == 4) {
    accSize = 16;
  }

  if (elemType.isF64()) {
    if (m == 16 && n == 16)
      accSize = 8;
    else if (m == 4 && n == 4)
      accSize = 2;
  }

  auto vregType = ctx.createVRegType(accSize, 4);
  Value result;

  auto isF16 = [&]() { return elemType.isF16(); };
  auto isBF16 = [&]() { return elemType.isBF16(); };
  auto isI8 = [&]() { return elemType.isInteger(8); };
  auto isF32 = [&]() { return elemType.isF32(); };
  auto isF64 = [&]() { return elemType.isF64(); };

  if (isF16()) {
    if (m == 16 && n == 16 && k == 16) {
      result = V_MFMA_F32_16X16X16_F16::create(builder, loc, vregType, *srcA,
                                               *srcB, *srcC);
    } else if (m == 16 && n == 16 && k == 32) {
      result = V_MFMA_F32_16X16X32_F16::create(builder, loc, vregType, *srcA,
                                               *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 8) {
      result = V_MFMA_F32_32X32X8_F16::create(builder, loc, vregType, *srcA,
                                              *srcB, *srcC);
    } else if (m == 16 && n == 16 && k == 4) {
      auto largeVregType = ctx.createVRegType(16, 4);
      result = V_MFMA_F32_16X16X4_F16::create(builder, loc, largeVregType,
                                              *srcA, *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 4) {
      auto largeVregType = ctx.createVRegType(32, 4);
      result = V_MFMA_F32_32X32X4_F16::create(builder, loc, largeVregType,
                                              *srcA, *srcB, *srcC);
    } else if (m == 4 && n == 4 && k == 4) {
      result = V_MFMA_F32_4X4X4_F16::create(builder, loc, vregType, *srcA,
                                            *srcB, *srcC);
    } else {
      result = V_MFMA_F32_16X16X16_F16::create(builder, loc, vregType, *srcA,
                                               *srcB, *srcC);
    }
  } else if (isBF16()) {
    if (m == 16 && n == 16 && k == 16) {
      result = V_MFMA_F32_16X16X16_BF16::create(builder, loc, vregType, *srcA,
                                                *srcB, *srcC);
    } else if (m == 16 && n == 16 && k == 32) {
      result = V_MFMA_F32_16X16X32_BF16::create(builder, loc, vregType, *srcA,
                                                *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 8) {
      result = V_MFMA_F32_32X32X8_BF16::create(builder, loc, vregType, *srcA,
                                               *srcB, *srcC);
    } else if (m == 16 && n == 16 && k == 4) {
      auto largeVregType = ctx.createVRegType(16, 4);
      result = V_MFMA_F32_16X16X4_BF16::create(builder, loc, largeVregType,
                                               *srcA, *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 4) {
      auto largeVregType = ctx.createVRegType(32, 4);
      result = V_MFMA_F32_32X32X4_BF16::create(builder, loc, largeVregType,
                                               *srcA, *srcB, *srcC);
    } else if (m == 4 && n == 4 && k == 4) {
      result = V_MFMA_F32_4X4X4_BF16::create(builder, loc, vregType, *srcA,
                                             *srcB, *srcC);
    } else {
      result = V_MFMA_F32_16X16X16_BF16::create(builder, loc, vregType, *srcA,
                                                *srcB, *srcC);
    }
  } else if (isI8()) {
    if (m == 16 && n == 16 && k == 16) {
      result = V_MFMA_I32_16X16X16_I8::create(builder, loc, vregType, *srcA,
                                              *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 8) {
      result = V_MFMA_I32_32X32X8_I8::create(builder, loc, vregType, *srcA,
                                             *srcB, *srcC);
    } else if (m == 16 && n == 16 && k == 4) {
      auto largeVregType = ctx.createVRegType(16, 4);
      result = V_MFMA_I32_16X16X4_I8::create(builder, loc, largeVregType, *srcA,
                                             *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 4) {
      auto largeVregType = ctx.createVRegType(32, 4);
      result = V_MFMA_I32_32X32X4_I8::create(builder, loc, largeVregType, *srcA,
                                             *srcB, *srcC);
    } else if (m == 4 && n == 4 && k == 4) {
      result = V_MFMA_I32_4X4X4_I8::create(builder, loc, vregType, *srcA, *srcB,
                                           *srcC);
    } else {
      result = V_MFMA_I32_16X16X16_I8::create(builder, loc, vregType, *srcA,
                                              *srcB, *srcC);
    }
  } else if (isF32()) {
    if (m == 16 && n == 16 && k == 4) {
      result = V_MFMA_F32_16X16X4_F32::create(builder, loc, vregType, *srcA,
                                              *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 2) {
      result = V_MFMA_F32_32X32X2_F32::create(builder, loc, vregType, *srcA,
                                              *srcB, *srcC);
    } else if (m == 4 && n == 4 && k == 1) {
      result = V_MFMA_F32_4X4X1_F32::create(builder, loc, vregType, *srcA,
                                            *srcB, *srcC);
    } else {
      result = V_MFMA_F32_16X16X4_F32::create(builder, loc, vregType, *srcA,
                                              *srcB, *srcC);
    }
  } else if (isF64()) {
    if (m == 16 && n == 16 && k == 4) {
      auto f64VregType = ctx.createVRegType(8, 4);
      result = V_MFMA_F64_16X16X4_F64::create(builder, loc, f64VregType, *srcA,
                                              *srcB, *srcC);
    } else if (m == 4 && n == 4 && k == 4) {
      auto f64VregType = ctx.createVRegType(2, 2);
      result = V_MFMA_F64_4X4X4_F64::create(builder, loc, f64VregType, *srcA,
                                            *srcB, *srcC);
    } else {
      auto f64VregType = ctx.createVRegType(8, 4);
      result = V_MFMA_F64_16X16X4_F64::create(builder, loc, f64VregType, *srcA,
                                              *srcB, *srcC);
    }
  } else {
    result = V_MFMA_F32_16X16X16_F16::create(builder, loc, vregType, *srcA,
                                             *srcB, *srcC);
  }

  ctx.getMapper().mapValue(mfmaOp.getDestD(), result);
  return success();
}

LogicalResult handleFatRawBufferCast(Operation *op, TranslationContext &ctx) {
  if (op->getNumOperands() < 1) {
    return success();
  }

  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto srcMapped = ctx.getMapper().getMapped(op->getOperand(0));
  if (!srcMapped) {
    return success();
  }

  bool hasCacheSwizzle = false;
  int64_t swizzleStride = 0;
  if (op->getNumOperands() >= 3) {
    if (auto swizzleVal = getConstantValue(op->getOperand(2))) {
      swizzleStride = *swizzleVal;
      hasCacheSwizzle = (swizzleStride > 0);
    }
  }

  if (!hasCacheSwizzle) {
    ctx.getMapper().mapValue(op->getResult(0), *srcMapped);
    return success();
  }

  int64_t newSrdBase = ctx.getNextSwizzleSRDIndex();
  int64_t srcSrdBase = -1;

  if (auto defOp = srcMapped->getDefiningOp()) {
    if (defOp->getName().getStringRef() == "waveasm.precolored.sreg") {
      if (auto indexAttr = defOp->getAttrOfType<IntegerAttr>("index")) {
        srcSrdBase = indexAttr.getInt();
      }
    }
  }

  if (srcSrdBase < 0) {
    if (auto psreg = dyn_cast<PSRegType>(srcMapped->getType())) {
      srcSrdBase = psreg.getIndex();
    }
  }

  if (srcSrdBase < 0) {
    if (auto srcSrdIdx = ctx.getSRDIndex(op->getOperand(0))) {
      srcSrdBase = *srcSrdIdx;
    }
  }

  if (srcSrdBase < 0) {
    ctx.getMapper().mapValue(op->getResult(0), *srcMapped);
    return success();
  }

  std::string mov0 = "s_mov_b32 s" + std::to_string(newSrdBase) + ", s" +
                     std::to_string(srcSrdBase);
  RawOp::create(builder, loc, mov0);

  std::string and1 = "s_and_b32 s" + std::to_string(newSrdBase + 1) + ", s" +
                     std::to_string(srcSrdBase + 1) + ", 0xffff";
  RawOp::create(builder, loc, and1);

  std::string or1 = "s_or_b32 s" + std::to_string(newSrdBase + 1) + ", s" +
                    std::to_string(newSrdBase + 1) + ", 0x40400000";
  RawOp::create(builder, loc, or1);

  std::string mov2 =
      "s_mov_b32 s" + std::to_string(newSrdBase + 2) + ", 0x7ffffffd";
  RawOp::create(builder, loc, mov2);

  std::string mov3 =
      "s_mov_b32 s" + std::to_string(newSrdBase + 3) + ", 0x27000";
  RawOp::create(builder, loc, mov3);

  auto srdType = ctx.createSRegType(4, 4);
  auto newSrd = PrecoloredSRegOp::create(builder, loc, srdType, newSrdBase, 4);
  ctx.getMapper().mapValue(op->getResult(0), newSrd);
  ctx.setCacheSwizzleStride(op->getResult(0), swizzleStride);

  return success();
}

/// Handle amdgpu.gather_to_lds - full implementation with dynamic M0
/// computation
LogicalResult handleGatherToLds(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  Value srcBuffer = op->getOperand(0);
  auto srcMapped = ctx.getMapper().getMapped(srcBuffer);
  if (!srcMapped) {
    return op->emitError("source buffer not mapped");
  }

  int64_t elemBytes = 2;
  if (auto transferTypeAttr = op->getAttrOfType<TypeAttr>("transferType")) {
    if (auto vecType = dyn_cast<VectorType>(transferTypeAttr.getValue())) {
      auto elemType = vecType.getElementType();
      elemBytes = elemType.getIntOrFloatBitWidth() / 8;
    }
  }

  // Create voffset for global memory access
  // When the srcIndex is base + constant, we compute the byte-scaled voffset
  // as base_voff + (constant * elemBytes) to enable CSE of base_voff across
  // multiple gathers that share the same base index.
  Value voff;
  if (op->getNumOperands() > 1) {
    Value srcIndex = op->getOperand(1);
    auto srcIndexMapped = ctx.getMapper().getMapped(srcIndex);
    if (srcIndexMapped) {
      auto vregType = ctx.createVRegType();
      if (elemBytes != 1 && llvm::isPowerOf2_64(elemBytes)) {
        int64_t shiftAmt = llvm::Log2_64(elemBytes);

        // Check if srcIndex is V_ADD_U32(base, constant) - if so, distribute
        // the byte scaling: (base + K) * elemBytes = base * elemBytes + K *
        // elemBytes This enables CSE of base*elemBytes across gathers.
        Value idxVal = *srcIndexMapped;
        auto addOp = idxVal.getDefiningOp<V_ADD_U32>();
        Value baseOperand;
        int64_t constVal = 0;
        bool hasConstAdd = false;

        if (addOp) {
          // Check if either operand is a constant
          if (auto constOp = addOp.getSrc1().getDefiningOp<ConstantOp>()) {
            constVal = constOp.getValue();
            baseOperand = addOp.getSrc0();
            hasConstAdd = true;
          } else if (auto constOp =
                         addOp.getSrc0().getDefiningOp<ConstantOp>()) {
            constVal = constOp.getValue();
            baseOperand = addOp.getSrc1();
            hasConstAdd = true;
          }
        }

        if (hasConstAdd) {
          // Distribute: (base + K) << shift = (base << shift) + (K << shift)
          auto shiftImm = ctx.createImmType(shiftAmt);
          auto shiftConst =
              ConstantOp::create(builder, loc, shiftImm, shiftAmt);
          Value baseVoff = V_LSHLREV_B32::create(builder, loc, vregType,
                                                 shiftConst, baseOperand);
          int64_t scaledConst = constVal * elemBytes;
          auto scaledImm = ctx.createImmType(scaledConst);
          auto scaledConstOp =
              ConstantOp::create(builder, loc, scaledImm, scaledConst);
          // NOTE: constant must be src0 (first operand) for VOP2 encoding.
          // src1 must be a VGPR on AMDGCN.
          voff = V_ADD_U32::create(builder, loc, vregType, scaledConstOp,
                                   baseVoff);
        } else {
          // Simple case: just shift
          auto shiftImm = ctx.createImmType(shiftAmt);
          auto shiftConst =
              ConstantOp::create(builder, loc, shiftImm, shiftAmt);
          voff = V_LSHLREV_B32::create(builder, loc, vregType, shiftConst,
                                       *srcIndexMapped);
        }
      } else if (elemBytes != 1) {
        auto vregType = ctx.createVRegType();
        auto scaleImm = ctx.createImmType(elemBytes);
        auto scaleConst = ConstantOp::create(builder, loc, scaleImm, elemBytes);
        voff = V_MUL_LO_U32::create(builder, loc, vregType, scaleConst,
                                    *srcIndexMapped);
      } else {
        auto vregType = ctx.createVRegType();
        voff = V_MOV_B32::create(builder, loc, vregType, *srcIndexMapped);
      }
    }
  }
  if (!voff) {
    auto immType = ctx.createImmType(0);
    auto zeroConst = ConstantOp::create(builder, loc, immType, 0);
    auto vregType = ctx.createVRegType();
    voff = V_MOV_B32::create(builder, loc, vregType, zeroConst);
  }

  // Get LDS info
  int64_t ldsBaseOffsetConst = 0;
  int64_t ldsRowStride = 0;
  bool hasLdsBaseOffset = false;

  if (op->getNumOperands() > 2) {
    mlir::Value ldsMemref = op->getOperand(2);
    if (auto baseOffset = ctx.getLDSBaseOffset(ldsMemref)) {
      if (auto immType = dyn_cast<ImmType>(baseOffset->getType())) {
        ldsBaseOffsetConst = immType.getValue();
        hasLdsBaseOffset = true;
      }
    }
    if (auto memrefType = dyn_cast<MemRefType>(ldsMemref.getType())) {
      auto shape = memrefType.getShape();
      int64_t elementBytes = 2;
      if (auto elemType = memrefType.getElementType()) {
        elementBytes = elemType.getIntOrFloatBitWidth() / 8;
      }
      if (shape.size() >= 2) {
        ldsRowStride = shape[1] * elementBytes;
      }
    }
  }

  int64_t ldsElemBytes = 2;
  if (op->getNumOperands() > 2) {
    if (auto memrefType = dyn_cast<MemRefType>(op->getOperand(2).getType())) {
      if (auto elemType = memrefType.getElementType()) {
        ldsElemBytes = elemType.getIntOrFloatBitWidth() / 8;
      }
    }
  }

  int64_t m0Const = ldsBaseOffsetConst;
  bool canUseImmediateM0 = hasLdsBaseOffset;
  int64_t colIdxConst = 0;
  bool hasConstCol = false;

  if (op->getNumOperands() > 4) {
    mlir::Value ldsColIndex = op->getOperand(4);
    auto ldsColMapped = ctx.getMapper().getMapped(ldsColIndex);
    if (ldsColMapped) {
      if (auto immType = dyn_cast<ImmType>(ldsColMapped->getType())) {
        colIdxConst = immType.getValue();
        hasConstCol = true;
      }
    }
  }

  if (op->getNumOperands() > 3) {
    mlir::Value ldsRowIndex = op->getOperand(3);
    auto ldsRowMapped = ctx.getMapper().getMapped(ldsRowIndex);
    if (ldsRowMapped) {
      if (auto immType = dyn_cast<ImmType>(ldsRowMapped->getType())) {
        int64_t rowIdxConst = immType.getValue();
        if (ldsRowStride > 0) {
          m0Const = ldsBaseOffsetConst + rowIdxConst * ldsRowStride;
          if (hasConstCol) {
            m0Const += colIdxConst * ldsElemBytes;
          }
        } else {
          m0Const = ldsBaseOffsetConst + rowIdxConst;
        }
        canUseImmediateM0 = hasConstCol || (colIdxConst == 0);
      } else {
        canUseImmediateM0 = false;
      }
    }
  }

  // Set M0 for LDS destination
  if (canUseImmediateM0) {
    auto m0Imm = ctx.createImmType(m0Const);
    auto m0ConstVal = ConstantOp::create(builder, loc, m0Imm, m0Const);
    S_MOV_B32_M0::create(builder, loc, m0ConstVal);
  } else {
    // Need to compute m0 dynamically
    Value m0Val;
    if (op->getNumOperands() > 3) {
      mlir::Value ldsRowIndex = op->getOperand(3);
      auto ldsRowMapped = ctx.getMapper().getMapped(ldsRowIndex);
      if (ldsRowMapped) {
        m0Val = *ldsRowMapped;

        // Lambda to convert SGPR to VGPR if needed
        auto convertToVgpr = [&](Value val) -> Value {
          if (!isVGPRType(val.getType())) {
            auto zeroImm = ctx.createImmType(0);
            auto zeroConst = ConstantOp::create(builder, loc, zeroImm, 0);
            Value zeroVgpr = V_MOV_B32::create(builder, loc,
                                               ctx.createVRegType(), zeroConst);
            return V_ADD_U32::create(builder, loc, ctx.createVRegType(),
                                     zeroVgpr, val);
          }
          return val;
        };

        // Multiply row index by row stride if needed
        if (ldsRowStride > 1) {
          auto strideImm = ctx.createImmType(ldsRowStride);
          auto strideConst =
              ConstantOp::create(builder, loc, strideImm, ldsRowStride);
          m0Val = convertToVgpr(m0Val);
          m0Val = V_MUL_LO_U32::create(builder, loc, ctx.createVRegType(),
                                       m0Val, strideConst);
        }

        // Add column offset
        if (op->getNumOperands() > 4) {
          mlir::Value ldsColIndex = op->getOperand(4);
          auto ldsColMapped = ctx.getMapper().getMapped(ldsColIndex);
          if (ldsColMapped) {
            if (auto immType = dyn_cast<ImmType>(ldsColMapped->getType())) {
              int64_t colOffset = immType.getValue() * ldsElemBytes;
              if (colOffset != 0) {
                auto colImm = ctx.createImmType(colOffset);
                auto colConst =
                    ConstantOp::create(builder, loc, colImm, colOffset);
                Value colVgpr = V_MOV_B32::create(
                    builder, loc, ctx.createVRegType(), colConst);
                m0Val = convertToVgpr(m0Val);
                m0Val = V_ADD_U32::create(builder, loc, ctx.createVRegType(),
                                          m0Val, colVgpr);
              }
            } else {
              m0Val = convertToVgpr(m0Val);
              Value colVgpr = convertToVgpr(*ldsColMapped);
              if (ldsElemBytes > 1) {
                auto scaleImm = ctx.createImmType(ldsElemBytes);
                auto scaleConst =
                    ConstantOp::create(builder, loc, scaleImm, ldsElemBytes);
                colVgpr = V_MUL_LO_U32::create(
                    builder, loc, ctx.createVRegType(), colVgpr, scaleConst);
              }
              m0Val = V_ADD_U32::create(builder, loc, ctx.createVRegType(),
                                        m0Val, colVgpr);
            }
          }
        }

        // Add base offset
        if (hasLdsBaseOffset && ldsBaseOffsetConst != 0) {
          auto baseImm = ctx.createImmType(ldsBaseOffsetConst);
          auto baseConst =
              ConstantOp::create(builder, loc, baseImm, ldsBaseOffsetConst);
          Value baseVgpr =
              V_MOV_B32::create(builder, loc, ctx.createVRegType(), baseConst);
          m0Val = convertToVgpr(m0Val);
          m0Val = V_ADD_U32::create(builder, loc, ctx.createVRegType(), m0Val,
                                    baseVgpr);
        }
      }
    }

    if (m0Val) {
      Value m0Src = m0Val;
      // Convert VGPR to SGPR using v_readfirstlane_b32
      if (isVGPRType(m0Src.getType())) {
        auto sregType = ctx.createSRegType();
        m0Src = V_READFIRSTLANE_B32::create(builder, loc, sregType, m0Src);
      }
      S_MOV_B32_M0::create(builder, loc, m0Src);
    } else {
      auto zeroImm = ctx.createImmType(0);
      auto zeroConst = ConstantOp::create(builder, loc, zeroImm, 0);
      S_MOV_B32_M0::create(builder, loc, zeroConst);
    }
  }

  // Determine transfer size
  int64_t transferBytes = 4;
  if (auto transferTypeAttr = op->getAttrOfType<TypeAttr>("transferType")) {
    if (auto vecType = dyn_cast<VectorType>(transferTypeAttr.getValue())) {
      int64_t numElems = vecType.getNumElements();
      auto elemType = vecType.getElementType();
      int64_t elemSize = elemType.getIntOrFloatBitWidth() / 8;
      transferBytes = numElems * elemSize;
    }
  }

  // Emit buffer_load instruction
  auto soffImm = ctx.createImmType(0);
  auto soffConst = ConstantOp::create(builder, loc, soffImm, 0);
  if (transferBytes == 16) {
    BUFFER_LOAD_DWORDX4_LDS::create(builder, loc, voff, *srcMapped, soffConst);
  } else if (transferBytes == 4) {
    BUFFER_LOAD_DWORD_LDS::create(builder, loc, voff, *srcMapped, soffConst);
  } else {
    return op->emitError("unsupported transfer size for gather_to_lds: " +
                         std::to_string(transferBytes) + " bytes");
  }

  // NOTE: We do NOT emit s_waitcnt here. The Ticketing pass handles
  // synchronization by inserting s_waitcnt vmcnt(0) before every s_barrier.
  // Since gather-to-lds is always followed by a barrier before any ds_read,
  // this ensures correctness without redundant per-gather waits.

  return success();
}

LogicalResult handleRawBufferLoad(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  if (op->getNumOperands() < 1) {
    return op->emitError("raw_buffer_load requires at least a memref operand");
  }

  Value memref = op->getOperand(0);
  auto srcMapped = ctx.getMapper().getMapped(memref);
  if (!srcMapped) {
    return op->emitError("buffer descriptor not mapped");
  }

  Value voffset;
  if (op->getNumOperands() > 1) {
    auto voffMapped = ctx.getMapper().getMapped(op->getOperand(1));
    if (voffMapped) {
      voffset = *voffMapped;
    }
  }
  if (!voffset) {
    auto immZero = ctx.createImmType(0);
    voffset = ConstantOp::create(builder, loc, immZero, 0);
  }

  Type resultType = op->getResult(0).getType();
  int64_t numElements = 1;
  if (auto vecType = dyn_cast<VectorType>(resultType)) {
    numElements = vecType.getNumElements();
  }

  Operation *loadInstr;
  if (numElements == 1) {
    auto vregType = ctx.createVRegType(1);
    loadInstr = BUFFER_LOAD_DWORD::create(builder, loc, TypeRange{vregType},
                                          *srcMapped, voffset);
  } else if (numElements == 2) {
    auto vregType = ctx.createVRegType(2, 2);
    loadInstr = BUFFER_LOAD_DWORDX2::create(builder, loc, TypeRange{vregType},
                                            *srcMapped, voffset);
  } else if (numElements == 4) {
    auto vregType = ctx.createVRegType(4, 4);
    loadInstr = BUFFER_LOAD_DWORDX4::create(builder, loc, TypeRange{vregType},
                                            *srcMapped, voffset);
  } else {
    return op->emitError("unsupported buffer load width");
  }

  ctx.getMapper().mapValue(op->getResult(0), loadInstr->getResult(0));
  return success();
}

LogicalResult handleRawBufferStore(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  if (op->getNumOperands() < 2) {
    return op->emitError("raw_buffer_store requires value and memref operands");
  }

  Value storeValue = op->getOperand(0);
  Value memref = op->getOperand(1);

  auto valueMapped = ctx.getMapper().getMapped(storeValue);
  auto srd = ctx.getMapper().getMapped(memref);

  if (!valueMapped) {
    return op->emitError("store value not mapped");
  }
  if (!srd) {
    return op->emitError("buffer descriptor not mapped");
  }

  Value voffset;
  if (op->getNumOperands() > 2) {
    auto voffMapped = ctx.getMapper().getMapped(op->getOperand(2));
    if (voffMapped) {
      voffset = *voffMapped;
    }
  }
  if (!voffset) {
    auto immZero = ctx.createImmType(0);
    voffset = ConstantOp::create(builder, loc, immZero, 0);
  }

  Type valueType = storeValue.getType();
  int64_t numElements = 1;
  if (auto vecType = dyn_cast<VectorType>(valueType)) {
    numElements = vecType.getNumElements();
  }

  if (numElements == 1) {
    BUFFER_STORE_DWORD::create(builder, loc, *valueMapped, *srd, voffset);
  } else if (numElements == 2) {
    BUFFER_STORE_DWORDX2::create(builder, loc, *valueMapped, *srd, voffset);
  } else if (numElements == 4) {
    BUFFER_STORE_DWORDX4::create(builder, loc, *valueMapped, *srd, voffset);
  } else {
    return op->emitError("unsupported buffer store width");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ROCDL Dialect Handlers
//===----------------------------------------------------------------------===//

LogicalResult handleReadFirstLane(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto sregType = ctx.createSRegType();

  if (op->getNumOperands() < 1) {
    return op->emitError("readfirstlane requires an operand");
  }

  auto src = ctx.getMapper().getMapped(op->getOperand(0));
  if (!src) {
    return op->emitError("source operand not mapped");
  }

  auto result = V_READFIRSTLANE_B32::create(builder, loc, sregType, *src);
  ctx.getMapper().mapValue(op->getResult(0), result);

  return success();
}

LogicalResult handleSWaitcnt(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  int64_t vmcnt = -1, lgkmcnt = -1;

  if (auto bitfieldAttr = op->getAttrOfType<IntegerAttr>("bitfield")) {
    int64_t bitfield = bitfieldAttr.getInt();
    int64_t vmcnt_lo = bitfield & 0xF;
    int64_t vmcnt_hi = (bitfield >> 14) & 0x3;
    int64_t vmcnt_full = vmcnt_lo | (vmcnt_hi << 4);
    int64_t lgkmcnt_full = (bitfield >> 8) & 0xF;

    if (vmcnt_full < 63) {
      vmcnt = vmcnt_full;
    }
    if (lgkmcnt_full < 15) {
      lgkmcnt = lgkmcnt_full;
    }
  }

  if (vmcnt < 0 && lgkmcnt < 0) {
    return success();
  }

  if (vmcnt >= 0 && lgkmcnt >= 0) {
    auto vmcntAttr = builder.getI32IntegerAttr(vmcnt);
    auto lgkmcntAttr = builder.getI32IntegerAttr(lgkmcnt);
    S_WAITCNT::create(builder, loc, vmcntAttr, lgkmcntAttr, IntegerAttr());
  } else if (vmcnt >= 0) {
    S_WAITCNT_VMCNT::create(builder, loc, vmcnt);
  } else {
    S_WAITCNT_LGKMCNT::create(builder, loc, lgkmcnt);
  }
  return success();
}

} // namespace waveasm
