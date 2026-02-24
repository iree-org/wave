// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Arith Dialect Handlers
//===----------------------------------------------------------------------===//
//
// This file implements handlers for Arith dialect operations.
// Many handlers use the handleBinaryVALU template to reduce code duplication.
//
//===----------------------------------------------------------------------===//

#include "Handlers.h"

#include "waveasm/Dialect/WaveASMOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "waveasm-arith-handlers"

using namespace mlir;

namespace waveasm {

//===----------------------------------------------------------------------===//
// Constant Handler
//===----------------------------------------------------------------------===//

LogicalResult handleArithConstant(Operation *op, TranslationContext &ctx) {
  auto constOp = cast<arith::ConstantOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto value = constOp.getValue();

  // Handle integer constants
  if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
    int64_t intVal = intAttr.getInt();
    auto immType = ctx.createImmType(intVal);
    auto immOp = ConstantOp::create(builder, loc, immType, intVal);
    ctx.getMapper().mapValue(constOp.getResult(), immOp);
    return success();
  }

  // Handle float constants - use bitCast helper for type-safe conversion
  if (auto floatAttr = dyn_cast<FloatAttr>(value)) {
    double floatVal = floatAttr.getValueAsDouble();
    if (floatAttr.getType().isF32()) {
      float f = static_cast<float>(floatVal);
      int32_t bits = bitCast<int32_t>(f);
      auto immType = ctx.createImmType(bits);
      auto immOp = ConstantOp::create(builder, loc, immType, bits);
      ctx.getMapper().mapValue(constOp.getResult(), immOp);
    }
    return success();
  }

  // Handle dense vector constants (e.g., dense<0.0> for accumulator init)
  if (auto denseAttr = dyn_cast<DenseElementsAttr>(value)) {
    if (denseAttr.isSplat()) {
      auto splatVal = denseAttr.getSplatValue<Attribute>();
      if (auto floatAttr = dyn_cast<FloatAttr>(splatVal)) {
        double floatVal = floatAttr.getValueAsDouble();
        if (floatVal == 0.0) {
          auto immType = ctx.createImmType(0);
          auto immOp = ConstantOp::create(builder, loc, immType, 0);
          ctx.getMapper().mapValue(constOp.getResult(), immOp);
        }
      }
    }
    return success();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Simple Binary Integer Operations (using template)
//===----------------------------------------------------------------------===//

LogicalResult handleArithAddI(Operation *op, TranslationContext &ctx) {
  return handleBinaryVALU<arith::AddIOp, V_ADD_U32>(op, ctx);
}

LogicalResult handleArithSubI(Operation *op, TranslationContext &ctx) {
  return handleBinaryVALU<arith::SubIOp, V_SUB_U32>(op, ctx);
}

LogicalResult handleArithMulI(Operation *op, TranslationContext &ctx) {
  return handleBinaryVALU<arith::MulIOp, V_MUL_LO_U32>(op, ctx);
}

LogicalResult handleArithMinSI(Operation *op, TranslationContext &ctx) {
  return handleBinaryVALU<arith::MinSIOp, V_MIN_I32>(op, ctx);
}

LogicalResult handleArithMaxSI(Operation *op, TranslationContext &ctx) {
  return handleBinaryVALU<arith::MaxSIOp, V_MAX_I32>(op, ctx);
}

LogicalResult handleArithMinUI(Operation *op, TranslationContext &ctx) {
  return handleBinaryVALU<arith::MinUIOp, V_MIN_U32>(op, ctx);
}

LogicalResult handleArithMaxUI(Operation *op, TranslationContext &ctx) {
  return handleBinaryVALU<arith::MaxUIOp, V_MAX_U32>(op, ctx);
}

LogicalResult handleArithAndI(Operation *op, TranslationContext &ctx) {
  return handleBinaryVALU<arith::AndIOp, V_AND_B32>(op, ctx);
}

LogicalResult handleArithOrI(Operation *op, TranslationContext &ctx) {
  return handleBinaryVALU<arith::OrIOp, V_OR_B32>(op, ctx);
}

LogicalResult handleArithXorI(Operation *op, TranslationContext &ctx) {
  return handleBinaryVALU<arith::XOrIOp, V_XOR_B32>(op, ctx);
}

//===----------------------------------------------------------------------===//
// Shift Operations (reversed operand order - using template)
//===----------------------------------------------------------------------===//

LogicalResult handleArithShLI(Operation *op, TranslationContext &ctx) {
  return handleBinaryVALURev<arith::ShLIOp, V_LSHLREV_B32>(op, ctx);
}

LogicalResult handleArithShRUI(Operation *op, TranslationContext &ctx) {
  return handleBinaryVALURev<arith::ShRUIOp, V_LSHRREV_B32>(op, ctx);
}

LogicalResult handleArithShRSI(Operation *op, TranslationContext &ctx) {
  return handleBinaryVALURev<arith::ShRSIOp, V_ASHRREV_I32>(op, ctx);
}

//===----------------------------------------------------------------------===//
// Simple Binary Float Operations (using template)
//===----------------------------------------------------------------------===//

LogicalResult handleArithAddF(Operation *op, TranslationContext &ctx) {
  return handleBinaryVALU<arith::AddFOp, V_ADD_F32>(op, ctx);
}

LogicalResult handleArithSubF(Operation *op, TranslationContext &ctx) {
  return handleBinaryVALU<arith::SubFOp, V_SUB_F32>(op, ctx);
}

LogicalResult handleArithMulF(Operation *op, TranslationContext &ctx) {
  return handleBinaryVALU<arith::MulFOp, V_MUL_F32>(op, ctx);
}

//===----------------------------------------------------------------------===//
// Division and Remainder (special handling for power-of-2 optimization)
//===----------------------------------------------------------------------===//

LogicalResult handleArithDivUI(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto divOp = cast<arith::DivUIOp>(op);
  std::optional<Value> lhs, rhs;
  if (failed(validateBinaryOperands(divOp, ctx, lhs, rhs))) {
    return failure();
  }

  // Check if RHS is a power of 2 constant - use shift instead
  if (auto constOp = rhs->getDefiningOp<ConstantOp>()) {
    int64_t divisor = constOp.getValue();
    if (isPowerOf2(divisor)) {
      int64_t shiftAmt = log2(divisor);
      auto immShift = ctx.createImmType(shiftAmt);
      auto shiftConst = ConstantOp::create(builder, loc, immShift, shiftAmt);
      auto result =
          V_LSHRREV_B32::create(builder, loc, vregType, shiftConst, *lhs);
      ctx.getMapper().mapValue(divOp.getResult(), result);
      return success();
    }
  }

  // General case: non-power-of-2 division requires complex reciprocal sequence
  // Emit an error rather than silently producing incorrect code
  return op->emitError("unsigned integer division by non-power-of-2 is not "
                       "yet implemented; divisor must be a power of 2");
}

LogicalResult handleArithRemUI(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto remOp = cast<arith::RemUIOp>(op);
  std::optional<Value> lhs, rhs;
  if (failed(validateBinaryOperands(remOp, ctx, lhs, rhs))) {
    return failure();
  }

  // Check if RHS is a power of 2 constant - use AND instead
  if (auto constOp = rhs->getDefiningOp<ConstantOp>()) {
    int64_t modulus = constOp.getValue();
    if (isPowerOf2(modulus)) {
      auto immMask = ctx.createImmType(modulus - 1);
      auto maskConst = ConstantOp::create(builder, loc, immMask, modulus - 1);
      auto result = V_AND_B32::create(builder, loc, vregType, *lhs, maskConst);
      ctx.getMapper().mapValue(remOp.getResult(), result);
      return success();
    }
  }

  // General case: non-power-of-2 modulo requires division
  // Emit an error rather than silently producing incorrect code
  return op->emitError("unsigned integer remainder by non-power-of-2 is not "
                       "yet implemented; modulus must be a power of 2");
}

LogicalResult handleArithDivF(Operation *op, TranslationContext &ctx) {
  auto divOp = cast<arith::DivFOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  std::optional<Value> lhs, rhs;
  if (failed(validateBinaryOperands(divOp, ctx, lhs, rhs))) {
    return failure();
  }

  // Use reciprocal + multiply for float division
  auto rcp = V_RCP_F32::create(builder, loc, vregType, *rhs);
  auto div = V_MUL_F32::create(builder, loc, vregType, *lhs, rcp);
  ctx.getMapper().mapValue(divOp.getResult(), div);
  return success();
}

//===----------------------------------------------------------------------===//
// Type Conversion Operations (typically no-ops on GPU)
//===----------------------------------------------------------------------===//

LogicalResult handleArithIndexCast(Operation *op, TranslationContext &ctx) {
  auto castOp = cast<arith::IndexCastOp>(op);
  auto src = ctx.getMapper().getMapped(castOp.getIn());
  if (src) {
    ctx.getMapper().mapValue(castOp.getResult(), *src);
  } else {
    // Source not mapped - this can happen for compile-time constants
    // which don't need runtime representation
    LLVM_DEBUG(llvm::dbgs()
               << "IndexCast source not mapped (may be constant)\n");
  }
  return success();
}

LogicalResult handleArithExtUI(Operation *op, TranslationContext &ctx) {
  auto extOp = cast<arith::ExtUIOp>(op);
  auto src = ctx.getMapper().getMapped(extOp.getIn());
  if (src) {
    ctx.getMapper().mapValue(extOp.getResult(), *src);
  } else {
    LLVM_DEBUG(llvm::dbgs() << "ExtUI source not mapped (may be constant)\n");
  }
  return success();
}

LogicalResult handleArithExtSI(Operation *op, TranslationContext &ctx) {
  auto extOp = cast<arith::ExtSIOp>(op);
  auto src = ctx.getMapper().getMapped(extOp.getIn());
  if (src) {
    ctx.getMapper().mapValue(extOp.getResult(), *src);
  } else {
    LLVM_DEBUG(llvm::dbgs() << "ExtSI source not mapped (may be constant)\n");
  }
  return success();
}

LogicalResult handleArithTruncI(Operation *op, TranslationContext &ctx) {
  auto truncOp = cast<arith::TruncIOp>(op);
  auto src = ctx.getMapper().getMapped(truncOp.getIn());
  if (src) {
    ctx.getMapper().mapValue(truncOp.getResult(), *src);
  } else {
    LLVM_DEBUG(llvm::dbgs() << "TruncI source not mapped (may be constant)\n");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Comparison and Select Operations
//===----------------------------------------------------------------------===//

LogicalResult handleArithCmpI(Operation *op, TranslationContext &ctx) {
  auto cmpOp = cast<arith::CmpIOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  std::optional<Value> lhs, rhs;
  if (failed(validateBinaryOperands(cmpOp, ctx, lhs, rhs))) {
    return failure();
  }

  // Emit comparison based on predicate
  // These operations set VCC implicitly (no SSA result)
  switch (cmpOp.getPredicate()) {
  case arith::CmpIPredicate::eq:
    V_CMP_EQ_U32::create(builder, loc, *lhs, *rhs);
    break;
  case arith::CmpIPredicate::ne:
    V_CMP_NE_U32::create(builder, loc, *lhs, *rhs);
    break;
  case arith::CmpIPredicate::slt:
    V_CMP_LT_I32::create(builder, loc, *lhs, *rhs);
    break;
  case arith::CmpIPredicate::sle:
    V_CMP_LE_I32::create(builder, loc, *lhs, *rhs);
    break;
  case arith::CmpIPredicate::sgt:
    V_CMP_GT_I32::create(builder, loc, *lhs, *rhs);
    break;
  case arith::CmpIPredicate::sge:
    V_CMP_GE_I32::create(builder, loc, *lhs, *rhs);
    break;
  case arith::CmpIPredicate::ult:
    V_CMP_LT_U32::create(builder, loc, *lhs, *rhs);
    break;
  case arith::CmpIPredicate::ule:
    V_CMP_LE_U32::create(builder, loc, *lhs, *rhs);
    break;
  case arith::CmpIPredicate::ugt:
    V_CMP_GT_U32::create(builder, loc, *lhs, *rhs);
    break;
  case arith::CmpIPredicate::uge:
    V_CMP_GE_U32::create(builder, loc, *lhs, *rhs);
    break;
  }

  // Map result to a placeholder (the select handler will use VCC implicitly)
  auto immOne = ctx.createImmType(1);
  auto one = ConstantOp::create(builder, loc, immOne, 1);
  ctx.getMapper().mapValue(cmpOp.getResult(), one);
  return success();
}

LogicalResult handleArithSelect(Operation *op, TranslationContext &ctx) {
  auto selectOp = cast<arith::SelectOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto cond = ctx.getMapper().getMapped(selectOp.getCondition());
  auto trueVal = ctx.getMapper().getMapped(selectOp.getTrueValue());
  auto falseVal = ctx.getMapper().getMapped(selectOp.getFalseValue());

  if (!cond || !trueVal || !falseVal) {
    return op->emitError("operands not mapped");
  }

  auto result =
      V_CNDMASK_B32::create(builder, loc, vregType, *falseVal, *trueVal, *cond);
  ctx.getMapper().mapValue(selectOp.getResult(), result);
  return success();
}

LogicalResult handleArithNegF(Operation *op, TranslationContext &ctx) {
  auto negOp = cast<arith::NegFOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto src = ctx.getMapper().getMapped(negOp.getOperand());
  if (!src) {
    return op->emitError("operand not mapped");
  }

  // XOR with sign bit to negate
  auto immSignBit = ctx.createImmType(0x80000000);
  auto signBit = ConstantOp::create(builder, loc, immSignBit, 0x80000000);
  auto neg = V_XOR_B32::create(builder, loc, vregType, *src, signBit);
  ctx.getMapper().mapValue(negOp.getResult(), neg);
  return success();
}

LogicalResult handleArithCmpF(Operation *op, TranslationContext &ctx) {
  auto cmpOp = cast<arith::CmpFOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  std::optional<Value> lhs, rhs;
  if (failed(validateBinaryOperands(cmpOp, ctx, lhs, rhs))) {
    return failure();
  }

  // Emit comparison based on predicate
  // These operations set VCC implicitly (no SSA result)
  switch (cmpOp.getPredicate()) {
  case arith::CmpFPredicate::OEQ:
  case arith::CmpFPredicate::UEQ:
    V_CMP_EQ_F32::create(builder, loc, *lhs, *rhs);
    break;
  case arith::CmpFPredicate::ONE:
  case arith::CmpFPredicate::UNE:
    V_CMP_NE_F32::create(builder, loc, *lhs, *rhs);
    break;
  case arith::CmpFPredicate::OLT:
  case arith::CmpFPredicate::ULT:
    V_CMP_LT_F32::create(builder, loc, *lhs, *rhs);
    break;
  case arith::CmpFPredicate::OLE:
  case arith::CmpFPredicate::ULE:
    V_CMP_LE_F32::create(builder, loc, *lhs, *rhs);
    break;
  case arith::CmpFPredicate::OGT:
  case arith::CmpFPredicate::UGT:
    V_CMP_GT_F32::create(builder, loc, *lhs, *rhs);
    break;
  case arith::CmpFPredicate::OGE:
  case arith::CmpFPredicate::UGE:
    V_CMP_GE_F32::create(builder, loc, *lhs, *rhs);
    break;
  default:
    V_CMP_EQ_F32::create(builder, loc, *lhs, *rhs);
    break;
  }

  // Map result to a placeholder (the select handler will use VCC implicitly)
  auto immOne = ctx.createImmType(1);
  auto one = ConstantOp::create(builder, loc, immOne, 1);
  ctx.getMapper().mapValue(cmpOp.getResult(), one);
  return success();
}

LogicalResult handleArithTruncF(Operation *op, TranslationContext &ctx) {
  auto truncOp = cast<arith::TruncFOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto src = ctx.getMapper().getMapped(truncOp.getIn());
  if (!src) {
    return op->emitError("truncf source not mapped");
  }

  Type srcElemType = getElementTypeOrSelf(truncOp.getIn().getType());
  Type dstElemType = getElementTypeOrSelf(truncOp.getResult().getType());

  auto srcType = truncOp.getIn().getType();
  auto vecType = dyn_cast<VectorType>(srcType);
  int64_t numElems = vecType ? vecType.getNumElements() : 1;

  if (numElems > 1) {
    ctx.getMapper().mapValue(truncOp.getResult(), *src);
  } else {
    auto vregType = ctx.createVRegType();
    Value result;
    if (srcElemType.isF32() && dstElemType.isBF16()) {
      result = V_CVT_BF16_F32::create(builder, loc, vregType, *src);
    } else if (srcElemType.isF32() && dstElemType.isF16()) {
      result = V_CVT_F16_F32::create(builder, loc, vregType, *src);
    } else {
      result = V_MOV_B32::create(builder, loc, vregType, *src);
    }
    ctx.getMapper().mapValue(truncOp.getResult(), result);
  }
  return success();
}

LogicalResult handleArithExtF(Operation *op, TranslationContext &ctx) {
  auto extOp = cast<arith::ExtFOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto src = ctx.getMapper().getMapped(extOp.getIn());
  if (!src) {
    return op->emitError("extf source not mapped");
  }

  Type srcElemType = getElementTypeOrSelf(extOp.getIn().getType());
  Type dstElemType = getElementTypeOrSelf(extOp.getResult().getType());

  auto vregType = ctx.createVRegType();
  Value result;

  if (srcElemType.isBF16() && dstElemType.isF32()) {
    result = V_CVT_F32_BF16::create(builder, loc, vregType, *src);
  } else if (srcElemType.isF16() && dstElemType.isF32()) {
    result = V_CVT_F32_F16::create(builder, loc, vregType, *src);
  } else {
    result = V_MOV_B32::create(builder, loc, vregType, *src);
  }

  ctx.getMapper().mapValue(extOp.getResult(), result);
  return success();
}

} // namespace waveasm
