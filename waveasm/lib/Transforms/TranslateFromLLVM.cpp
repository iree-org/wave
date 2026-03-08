// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// TranslateFromLLVM: Strict LLVM dialect → WaveASM translation.
//
// Consumes gpu.module { llvm.func @kernel ... } with rocdl intrinsics.
// Fails on any unhandled op — no silent fallthrough.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Passes.h"
#include "waveasm/Transforms/TranslateFromMLIR.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "waveasm-translate-llvm"

using namespace mlir;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMTRANSLATEFROMLLVM
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

namespace waveasm {

//===----------------------------------------------------------------------===//
// LLVM Translation State
//===----------------------------------------------------------------------===//

/// Tracks decomposed buffer pointer info from GEP operations.
/// A GEP on ptr<7> decomposes into (SRD, byte-offset-vgpr).
/// TODO: consider a separate decomposition pass for ptr<7>.
struct BufferPtrInfo {
  Value srd;     // The SRD (4×SGPR) from rocdl.make.buffer.rsrc.
  Value voffset; // Byte offset VGPR.
};

/// State for LLVM→WaveASM translation, layered on top of TranslationContext.
class LLVMTranslationState {
public:
  explicit LLVMTranslationState(TranslationContext &ctx) : ctx(ctx) {}

  TranslationContext &ctx;

  /// Map rocdl.make.buffer.rsrc result → SRD SGPR value from prologue.
  void mapBufferRsrc(Value rsrc, Value srd) { rsrcToSRD[rsrc] = srd; }
  std::optional<Value> lookupSRD(Value rsrc) const {
    auto it = rsrcToSRD.find(rsrc);
    if (it != rsrcToSRD.end())
      return it->second;
    return std::nullopt;
  }

  /// Map GEP result → decomposed (SRD, voffset).
  void mapGEP(Value gep, BufferPtrInfo info) { gepMap[gep] = info; }
  const BufferPtrInfo *lookupGEP(Value gep) const {
    auto it = gepMap.find(gep);
    if (it != gepMap.end())
      return &it->second;
    return nullptr;
  }

  /// Track base-pointer byte offset from bare-pointer GEPs.
  /// These offsets accumulate and get added to voffset when the pointer
  /// is used via make.buffer.rsrc + buffer GEP.
  void setBaseOffset(Value ptr, Value offset) { baseOffsets[ptr] = offset; }
  Value getBaseOffset(Value ptr) const {
    auto it = baseOffsets.find(ptr);
    return it != baseOffsets.end() ? it->second : Value{};
  }

  /// Track LDS base values (from addressof / ptr<3> GEPs).
  void setLDSBase(Value v) { ldsBaseValues.insert(v); }
  bool isLDSBase(Value v) const { return ldsBaseValues.contains(v); }

private:
  DenseMap<Value, Value> rsrcToSRD;
  DenseMap<Value, BufferPtrInfo> gepMap;
  DenseMap<Value, Value> baseOffsets;
  DenseSet<Value> ldsBaseValues;
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Extract workgroup size from llvm.func attributes.
static std::tuple<int64_t, int64_t, int64_t>
getWorkgroupSize(LLVM::LLVMFuncOp func) {
  for (StringRef attrName :
       {"gpu.known_block_size", "rocdl.reqd_work_group_size"}) {
    if (auto attr = func->getAttrOfType<DenseI32ArrayAttr>(attrName)) {
      auto vals = attr.asArrayRef();
      int64_t x = vals.size() > 0 ? vals[0] : 64;
      int64_t y = vals.size() > 1 ? vals[1] : 1;
      int64_t z = vals.size() > 2 ? vals[2] : 1;
      return {x, y, z};
    }
  }
  return {64, 1, 1};
}

/// Create a waveasm.program from an llvm.func kernel.
static ProgramOp createProgramFromLLVMFunc(LLVM::LLVMFuncOp func,
                                           OpBuilder &builder,
                                           StringRef targetId) {
  auto *mlirCtx = builder.getContext();
  auto loc = func.getLoc();

  auto targetAttr =
      TargetAttr::get(mlirCtx, getTargetKindAttr(mlirCtx, targetId), 5);
  auto abiAttr = KernelABIAttr::get(mlirCtx, 0, 0, std::nullopt, std::nullopt,
                                    std::nullopt);

  auto [wgX, wgY, wgZ] = getWorkgroupSize(func);
  SmallVector<Attribute, 3> sizes = {builder.getI64IntegerAttr(wgX),
                                     builder.getI64IntegerAttr(wgY),
                                     builder.getI64IntegerAttr(wgZ)};

  // Mangle the program name to avoid symbol collision with the original
  // llvm.func (which we keep alive for gpu.launch_func verification).
  // Store the original kernel name for assembly emission.
  std::string programName = (func.getName() + "__waveasm").str();
  auto program =
      ProgramOp::create(builder, loc, programName, targetAttr, abiAttr,
                        /*vgprs=*/int64_t{256},
                        /*sgprs=*/int64_t{104},
                        /*workgroup_size=*/builder.getArrayAttr(sizes),
                        /*lds_size=*/IntegerAttr{});

  program->setAttr("kernel_name", builder.getStringAttr(func.getName()));

  if (program.getBody().empty())
    program.getBody().emplaceBlock();

  return program;
}

/// Resolve an LLVM SSA value to its WaveASM counterpart via the mapper.
static Value resolve(Value v, TranslationContext &ctx) {
  if (auto mapped = ctx.getMapper().getMapped(v))
    return *mapped;
  return v;
}

/// Infer the pseudo-op result type from operand types.
/// If any operand is VGPR → VReg; otherwise SReg.
static Type inferResultType(ValueRange operands, TranslationContext &ctx) {
  for (Value v : operands)
    if (isVGPRType(v.getType()))
      return ctx.createVRegType();
  return ctx.createSRegType();
}

//===----------------------------------------------------------------------===//
// Op handlers
//===----------------------------------------------------------------------===//

static LogicalResult handlePoison(LLVM::PoisonOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  // Poison is undefined — materialize as zero.
  auto immTy = ctx.createImmType(0);
  auto imm = ConstantOp::create(builder, loc, immTy, int64_t{0});
  auto vregTy = ctx.createVRegType();
  auto mov = V_MOV_B32::create(builder, loc, vregTy, imm);
  ctx.getMapper().mapValue(op.getResult(), mov);
  return success();
}

static LogicalResult handleConstant(LLVM::ConstantOp op,
                                    LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  auto valAttr = op.getValue();

  // Dense vector constant (e.g. MFMA accumulator init).
  if (auto denseAttr = dyn_cast<DenseElementsAttr>(valAttr)) {
    if (!denseAttr.isSplat())
      return op->emitOpError("non-splat dense constant not yet supported");
    int64_t numElems = denseAttr.getNumElements();
    // Splat of zero → wide VReg initialized with v_mov_b32.
    APFloat splatVal = denseAttr.getSplatValue<APFloat>();
    int64_t rawBits = splatVal.bitcastToAPInt().getZExtValue();
    auto immTy = ctx.createImmType(rawBits);
    auto immOp = ConstantOp::create(builder, loc, immTy, rawBits);
    auto vregTy = ctx.createVRegType(numElems, numElems);
    auto mov = V_MOV_B32::create(builder, loc, vregTy, immOp);
    ctx.getMapper().mapValue(op.getResult(), mov);
    return success();
  }

  int64_t intVal = 0;
  if (auto intAttr = dyn_cast<IntegerAttr>(valAttr))
    intVal = intAttr.getValue().getSExtValue();
  else
    return op->emitOpError("unsupported constant type");

  // Materialize as v_mov_b32 into a VGPR.
  auto immTy = ctx.createImmType(intVal);
  auto immOp = ConstantOp::create(builder, loc, immTy, intVal);
  auto vregTy = ctx.createVRegType();
  auto mov = V_MOV_B32::create(builder, loc, vregTy, immOp);
  ctx.getMapper().mapValue(op.getResult(), mov);
  return success();
}

static LogicalResult handleWorkitemIdX(ROCDL::ThreadIdXOp op,
                                       LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  // rocdl.workitem.id.x → hardware v0 (flat workitem ID).
  ctx.setUsesWorkitemId(true);
  auto vregTy = ctx.createVRegType();
  auto v0 = PrecoloredVRegOp::create(builder, loc, vregTy, /*regIndex=*/0,
                                     /*size=*/1);
  ctx.getMapper().mapValue(op.getResult(), v0);
  return success();
}

// rocdl.workgroup.id.{x,y,z} → system SGPRs (set by hardware dispatch).
template <typename OpTy>
static LogicalResult handleWorkgroupId(OpTy op, LLVMTranslationState &st,
                                       int dimIndex) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  if (dimIndex == 0)
    ctx.setUsesWorkgroupIdX(true);
  else if (dimIndex == 1)
    ctx.setUsesWorkgroupIdY(true);
  else
    ctx.setUsesWorkgroupIdZ(true);

  int64_t sgprIndex = ctx.getWorkgroupIdSgprIndex(dimIndex);
  auto sregType = ctx.createSRegType();
  auto blockId = PrecoloredSRegOp::create(builder, loc, sregType, sgprIndex, 1);
  ctx.getMapper().mapValue(op.getResult(), blockId);
  return success();
}

// Emit arith pseudo-ops for i32↔i64 casts — legalization pass handles width.
static LogicalResult handleSext(LLVM::SExtOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  Value src = resolve(op.getOperand(), ctx);
  Type resTy = isVGPRType(src.getType()) ? (Type)ctx.createVRegType()
                                         : (Type)ctx.createSRegType();
  auto pseudo = Arith_SExtOp::create(builder, op.getLoc(), resTy, src);
  ctx.getMapper().mapValue(op.getResult(), pseudo);
  return success();
}

static LogicalResult handleZext(LLVM::ZExtOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  Value src = resolve(op.getOperand(), ctx);
  Type resTy = isVGPRType(src.getType()) ? (Type)ctx.createVRegType()
                                         : (Type)ctx.createSRegType();
  auto pseudo = Arith_ZExtOp::create(builder, op.getLoc(), resTy, src);
  ctx.getMapper().mapValue(op.getResult(), pseudo);
  return success();
}

static LogicalResult handleTrunc(LLVM::TruncOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  Value src = resolve(op.getOperand(), ctx);
  Type resTy = isVGPRType(src.getType()) ? (Type)ctx.createVRegType()
                                         : (Type)ctx.createSRegType();
  auto pseudo = Arith_TruncOp::create(builder, op.getLoc(), resTy, src);
  ctx.getMapper().mapValue(op.getResult(), pseudo);
  return success();
}

/// Map LLVM ICmpPredicate to WaveASM CmpPredicate.
static CmpPredicate mapLLVMPredicate(LLVM::ICmpPredicate pred) {
  using LP = LLVM::ICmpPredicate;
  switch (pred) {
  case LP::eq:
    return CmpPredicate::eq;
  case LP::ne:
    return CmpPredicate::ne;
  case LP::slt:
    return CmpPredicate::slt;
  case LP::sle:
    return CmpPredicate::sle;
  case LP::sgt:
    return CmpPredicate::sgt;
  case LP::sge:
    return CmpPredicate::sge;
  case LP::ult:
    return CmpPredicate::ult;
  case LP::ule:
    return CmpPredicate::ule;
  case LP::ugt:
    return CmpPredicate::ugt;
  case LP::uge:
    return CmpPredicate::uge;
  }
  llvm_unreachable("unhandled LLVM ICmpPredicate");
}

static LogicalResult handleICmp(LLVM::ICmpOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  Value lhs = resolve(op.getLhs(), ctx);
  Value rhs = resolve(op.getRhs(), ctx);
  auto resTy = inferResultType({lhs, rhs}, ctx);
  auto pred = mapLLVMPredicate(op.getPredicate());
  auto cmp = Arith_CmpOp::create(builder, op.getLoc(), resTy, pred, lhs, rhs);
  ctx.getMapper().mapValue(op.getResult(), cmp);
  return success();
}

static LogicalResult handleSelect(LLVM::SelectOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  Value cond = resolve(op.getCondition(), ctx);
  Value trueVal = resolve(op.getTrueValue(), ctx);
  Value falseVal = resolve(op.getFalseValue(), ctx);
  auto resTy = inferResultType({trueVal, falseVal}, ctx);
  auto sel = Arith_SelectOp::create(builder, op.getLoc(), resTy, cond, trueVal,
                                    falseVal);
  ctx.getMapper().mapValue(op.getResult(), sel);
  return success();
}

static LogicalResult handleAdd(LLVM::AddOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  Value lhs = resolve(op.getLhs(), ctx);
  Value rhs = resolve(op.getRhs(), ctx);
  auto resTy = inferResultType({lhs, rhs}, ctx);
  auto add = Arith_AddOp::create(builder, op.getLoc(), resTy, lhs, rhs);
  ctx.getMapper().mapValue(op.getResult(), add);
  return success();
}

static LogicalResult handleMul(LLVM::MulOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  Value lhs = resolve(op.getLhs(), ctx);
  Value rhs = resolve(op.getRhs(), ctx);
  auto resTy = inferResultType({lhs, rhs}, ctx);
  auto mul = Arith_MulOp::create(builder, op.getLoc(), resTy, lhs, rhs);
  ctx.getMapper().mapValue(op.getResult(), mul);
  return success();
}

/// Try to extract a constant integer from an LLVM SSA value.
static std::optional<int64_t> getConstantInt(Value v) {
  if (auto constOp = v.getDefiningOp<LLVM::ConstantOp>())
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getValue().getSExtValue();
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// LDS global / addressof handlers
//===----------------------------------------------------------------------===//

/// Handle llvm.mlir.addressof @global → LDS base pointer.
/// The LDS size is extracted from the global's type and recorded on the
/// program.
static LogicalResult handleAddressOf(LLVM::AddressOfOp op,
                                     LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  // Look up the global to determine LDS size.
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    module =
        op->getParentOfType<gpu::GPUModuleOp>()->getParentOfType<ModuleOp>();
  auto global = SymbolTable::lookupNearestSymbolFrom<LLVM::GlobalOp>(
      op, op.getGlobalNameAttr());
  if (global) {
    // Compute byte size from the global's type.
    auto globalTy = global.getType();
    int64_t sizeBytes = 0;
    if (auto arrTy = dyn_cast<LLVM::LLVMArrayType>(globalTy))
      sizeBytes = arrTy.getNumElements();
    if (sizeBytes > 0)
      ctx.addLDSSize(sizeBytes);
  }

  // Map to a constant 0 offset in a VGPR — LDS addressing is relative.
  auto &builder = ctx.getBuilder();
  auto immTy = ctx.createImmType(0);
  auto zero = ConstantOp::create(builder, op.getLoc(), immTy, int64_t{0});
  auto vregTy = ctx.createVRegType();
  auto mov = V_MOV_B32::create(builder, op.getLoc(), vregTy, zero);
  ctx.getMapper().mapValue(op.getResult(), mov);
  st.setLDSBase(op.getResult());
  return success();
}

//===----------------------------------------------------------------------===//
// Signed div/rem handlers
//===----------------------------------------------------------------------===//

static LogicalResult handleSDiv(LLVM::SDivOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  Value lhs = resolve(op.getLhs(), ctx);

  // Power-of-2 constant divisor → arithmetic shift right.
  if (auto constVal = getConstantInt(op.getRhs())) {
    int64_t divisor = *constVal;
    if (divisor > 0 && (divisor & (divisor - 1)) == 0) {
      int64_t shiftAmt = llvm::Log2_64(divisor);
      auto immTy = ctx.createImmType(shiftAmt);
      auto shiftConst =
          ConstantOp::create(builder, op.getLoc(), immTy, shiftAmt);
      auto vregTy = ctx.createVRegType();
      auto result =
          V_ASHRREV_I32::create(builder, op.getLoc(), vregTy, shiftConst, lhs);
      ctx.getMapper().mapValue(op.getResult(), result);
      return success();
    }
  }
  return op->emitOpError("signed division by non-power-of-2 not yet supported");
}

static LogicalResult handleSRem(LLVM::SRemOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  Value lhs = resolve(op.getLhs(), ctx);

  // Power-of-2 constant divisor → v_and_b32 with (divisor - 1).
  // This is correct for non-negative dividends (thread IDs, indices).
  if (auto constVal = getConstantInt(op.getRhs())) {
    int64_t divisor = *constVal;
    if (divisor > 0 && (divisor & (divisor - 1)) == 0) {
      int64_t mask = divisor - 1;
      auto immTy = ctx.createImmType(mask);
      auto maskConst = ConstantOp::create(builder, op.getLoc(), immTy, mask);
      auto vregTy = ctx.createVRegType();
      auto result =
          V_AND_B32::create(builder, op.getLoc(), vregTy, lhs, maskConst);
      ctx.getMapper().mapValue(op.getResult(), result);
      return success();
    }
  }
  return op->emitOpError(
      "signed remainder by non-power-of-2 not yet supported");
}

//===----------------------------------------------------------------------===//
// Memory fence / barrier handlers
//===----------------------------------------------------------------------===//

static LogicalResult handleFence(LLVM::FenceOp, LLVMTranslationState &) {
  // Memory fences are handled implicitly by s_barrier and waitcnt insertion.
  return success();
}

template <typename OpTy>
static LogicalResult handleBarrier(OpTy op, LLVMTranslationState &st) {
  S_BARRIER::create(st.ctx.getBuilder(), op.getLoc());
  return success();
}

//===----------------------------------------------------------------------===//
// Vector shuffle handler
//===----------------------------------------------------------------------===//

static LogicalResult handleShuffleVector(LLVM::ShuffleVectorOp op,
                                         LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  Value src = resolve(op.getV1(), ctx);

  // shufflevector with a single index extracts one element.
  auto mask = op.getMask();
  if (mask.size() == 1) {
    int64_t idx = mask[0];
    auto vregTy = ctx.createVRegType();
    auto extract = ExtractOp::create(builder, op.getLoc(), vregTy, src, idx);
    ctx.getMapper().mapValue(op.getResult(), extract);
    return success();
  }

  return op->emitOpError("multi-element shufflevector not yet supported");
}

//===----------------------------------------------------------------------===//
// MFMA handler
//===----------------------------------------------------------------------===//

static LogicalResult handleMFMA_F32_16x16x16_F16(ROCDL::mfma_f32_16x16x16f16 op,
                                                 LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  Value a = resolve(op.getA(), ctx);
  Value b = resolve(op.getB(), ctx);
  Value c = resolve(op.getC(), ctx);

  // MFMA: v_mfma_f32_16x16x16_f16 dst, a, b, c.
  // Result and accumulator are vector<4xf32> → 4 VGPRs.
  auto accTy = ctx.createVRegType(4, 4);
  auto mfma = V_MFMA_F32_16X16X16_F16::create(builder, loc, accTy, a, b, c);
  ctx.getMapper().mapValue(op.getResult(), mfma);
  return success();
}

//===----------------------------------------------------------------------===//
// SCF for/yield handler
//===----------------------------------------------------------------------===//

/// Forward declaration for recursive op translation.
static LogicalResult translateOp(Operation *op, LLVMTranslationState &st);

static LogicalResult handleSCFFor(scf::ForOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  Value lb = resolve(op.getLowerBound(), ctx);
  Value ub = resolve(op.getUpperBound(), ctx);
  Value step = resolve(op.getStep(), ctx);

  // Build init args: [lower_bound, iter_args...].
  // LoopOp's first block arg is the induction variable.
  SmallVector<Value> initArgs;
  initArgs.push_back(lb);
  for (Value arg : op.getInitArgs())
    initArgs.push_back(resolve(arg, ctx));

  // Create the waveasm.loop (do-while semantics).
  // TODO: Guard with if (lb < ub) for loops that may execute 0 times.
  auto loopOp = LoopOp::create(builder, loc, initArgs);
  Block &bodyBlock = loopOp.getBodyBlock();

  // Map the induction variable (block arg 0).
  ctx.getMapper().mapValue(op.getInductionVar(), bodyBlock.getArgument(0));

  // Map iter_args (block args 1..N).
  for (unsigned i = 0; i < op.getInitArgs().size(); ++i)
    ctx.getMapper().mapValue(op.getRegionIterArgs()[i],
                             bodyBlock.getArgument(i + 1));

  // Translate the loop body.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&bodyBlock);
  for (Operation &bodyOp : op.getBody()->without_terminator())
    if (failed(translateOp(&bodyOp, st)))
      return failure();

  // Build loop increment and condition: iv_next = iv + step; cond = iv_next <
  // ub.
  Value inductionVar = bodyBlock.getArgument(0);
  auto ivTy = inductionVar.getType();
  Value nextIV = Arith_AddOp::create(builder, loc, ivTy, inductionVar, step);
  // Condition must be SGPR for waveasm.condition. Convert to scalar if needed.
  auto sregTy = ctx.createSRegType();
  Value scalarIV = nextIV;
  Value scalarUB = ub;
  if (isVGPRType(nextIV.getType()))
    scalarIV = V_READFIRSTLANE_B32::create(builder, loc, sregTy, nextIV);
  if (isVGPRType(ub.getType()))
    scalarUB = V_READFIRSTLANE_B32::create(builder, loc, sregTy, ub);
  Value cond = S_CMP_LT_U32::create(builder, loc, sregTy, scalarIV, scalarUB);

  // Collect iter args from yield.
  auto yieldOp = cast<scf::YieldOp>(op.getBody()->getTerminator());
  SmallVector<Value> condIterArgs;
  condIterArgs.push_back(nextIV);
  for (Value v : yieldOp.getOperands())
    condIterArgs.push_back(resolve(v, ctx));

  ConditionOp::create(builder, loc, cond, condIterArgs);

  // Map loop results. scf.for results are iter_args only (no IV),
  // but waveasm.loop results include the IV at index 0.
  for (unsigned i = 0; i < op.getNumResults(); ++i)
    ctx.getMapper().mapValue(op.getResult(i), loopOp.getResult(i + 1));

  return success();
}

/// Compute buffer load/store size from the LLVM element type.
static int64_t getBufferAccessBytes(Type ty) {
  if (auto vecTy = dyn_cast<VectorType>(ty))
    return vecTy.getNumElements() *
           vecTy.getElementType().getIntOrFloatBitWidth() / 8;
  if (ty.isIntOrFloat())
    return ty.getIntOrFloatBitWidth() / 8;
  return 0;
}

//===----------------------------------------------------------------------===//
// LDS load/store handlers
//===----------------------------------------------------------------------===//

static LogicalResult handleLDSLoad(LLVM::LoadOp op, Value addr,
                                   LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  int64_t numBytes = getBufferAccessBytes(op.getResult().getType());

  Value offset = resolve(addr, ctx);
  auto vregTy = ctx.createVRegType();

  Operation *loadOp = nullptr;
  if (numBytes == 2)
    loadOp = DS_READ_U16::create(builder, loc, TypeRange{vregTy}, offset);
  else if (numBytes == 4)
    loadOp = DS_READ_B32::create(builder, loc, TypeRange{vregTy}, offset);
  else if (numBytes == 8) {
    auto wideTy = ctx.createVRegType(2, 2);
    loadOp = DS_READ_B64::create(builder, loc, TypeRange{wideTy}, offset);
  } else if (numBytes == 16) {
    auto wideTy = ctx.createVRegType(4, 4);
    loadOp = DS_READ_B128::create(builder, loc, TypeRange{wideTy}, offset);
  } else
    return op->emitOpError("unsupported LDS load size: ")
           << numBytes << " bytes";

  ctx.getMapper().mapValue(op.getResult(), loadOp->getResult(0));
  return success();
}

static LogicalResult handleLDSStore(LLVM::StoreOp op, Value addr,
                                    LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  Value data = resolve(op.getValue(), ctx);
  Value offset = resolve(addr, ctx);
  int64_t numBytes = getBufferAccessBytes(op.getValue().getType());

  if (numBytes == 2)
    DS_WRITE_B16::create(builder, loc, data, offset);
  else if (numBytes == 4)
    DS_WRITE_B32::create(builder, loc, data, offset);
  else if (numBytes == 8)
    DS_WRITE_B64::create(builder, loc, data, offset);
  else if (numBytes == 16)
    DS_WRITE_B128::create(builder, loc, data, offset);
  else
    return op->emitOpError("unsupported LDS store size: ")
           << numBytes << " bytes";

  return success();
}

static LogicalResult handleMakeBufferRsrc(ROCDL::MakeBufferRsrcOp op,
                                          LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  // The base pointer was set up as an SRD in the prologue via queueSRDSetup.
  Value basePtr = op.getBase();
  auto srdVal = ctx.getMapper().getMapped(basePtr);
  if (!srdVal)
    return op->emitOpError("SRD not found for base pointer");

  // The prologue used s_mov_b64 to copy the 64-bit pointer into SRD[0:1].
  // This corrupts SRD word 1 bits [31:16] (stride/swizzle) with pointer bits.
  // Also, the prologue hardcodes SRD[3]=0x20000 but make.buffer.rsrc may
  // want different flags. Patch both now that we know the actual values.
  auto srdOp = dyn_cast<PrecoloredSRegOp>(srdVal->getDefiningOp());
  if (srdOp) {
    int64_t srdBase = srdOp.getIndex();

    // Clear stride/swizzle bits in SRD word 1 (keep only base_addr[47:32]).
    std::string andStr = "s_and_b32 s" + std::to_string(srdBase + 1) + ", s" +
                         std::to_string(srdBase + 1) + ", 0xFFFF";
    RawOp::create(builder, loc, andStr);

    // Patch SRD[3] with the actual flags from make.buffer.rsrc.
    auto flags = getConstantInt(op.getFlags());
    if (flags && *flags != 0x20000) {
      std::string movFlags = "s_mov_b32 s" + std::to_string(srdBase + 3) +
                             ", 0x" + llvm::utohexstr(*flags);
      RawOp::create(builder, loc, movFlags);
    }
  }

  st.mapBufferRsrc(op.getResult(), *srdVal);

  // Propagate any base offset from bare-pointer GEPs so buffer GEPs
  // can add it to their voffset.
  Value baseOff = st.getBaseOffset(basePtr);
  if (baseOff)
    st.setBaseOffset(op.getResult(), baseOff);

  return success();
}

/// Compute the byte offset for a GEP as a single WaveASM Value.
/// Handles both constant-attr indices and dynamic Value indices.
/// For all-zero constant indices, returns std::nullopt (offset is 0).
static std::optional<Value> computeGEPOffset(LLVM::GEPOp op,
                                             TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();
  auto indices = op.getIndices();

  // Check if all indices are constant zero → no offset needed.
  bool allZero = true;
  for (auto idx : indices) {
    if (auto val = idx.dyn_cast<Value>()) {
      allZero = false;
      break;
    }
    auto constIdx = cast<IntegerAttr>(idx);
    if (constIdx.getInt() != 0) {
      allZero = false;
      break;
    }
  }
  if (allZero)
    return std::nullopt;

  // Single index — common case.
  if (indices.size() == 1) {
    auto idx = indices[0];
    if (auto val = idx.dyn_cast<Value>())
      return resolve(val, ctx);
    // Constant integer index.
    int64_t constVal = cast<IntegerAttr>(idx).getInt();
    auto immTy = ctx.createImmType(constVal);
    return ConstantOp::create(builder, loc, immTy, constVal)->getResult(0);
  }

  // Multi-index: accumulate. For now, only support constant indices
  // (typical for array GEPs like [0, N]).
  int64_t totalOffset = 0;
  for (auto idx : indices) {
    if (auto val = idx.dyn_cast<Value>())
      return op->emitOpError("multi-index GEP with dynamic indices "
                             "not yet supported"),
             std::nullopt;
    totalOffset += cast<IntegerAttr>(idx).getInt();
  }
  if (totalOffset == 0)
    return std::nullopt;
  auto immTy = ctx.createImmType(totalOffset);
  return ConstantOp::create(builder, loc, immTy, totalOffset)->getResult(0);
}

static LogicalResult handleGEP(LLVM::GEPOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();
  Value base = op.getBase();

  auto baseTy = op.getBase().getType();
  if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(baseTy)) {
    // LDS GEP (ptr<3>): compute byte offset for ds_read/ds_write.
    if (ptrTy.getAddressSpace() == 3) {
      Value baseOff = resolve(base, ctx);
      auto maybeOffset = computeGEPOffset(op, ctx);
      if (!maybeOffset) {
        // All-zero indices — pass through the base.
        ctx.getMapper().mapValue(op.getResult(), baseOff);
      } else {
        auto resTy = inferResultType({baseOff, *maybeOffset}, ctx);
        auto sum =
            Arith_AddOp::create(builder, loc, resTy, baseOff, *maybeOffset);
        ctx.getMapper().mapValue(op.getResult(), sum);
      }
      st.setLDSBase(op.getResult());
      return success();
    }

    if (ptrTy.getAddressSpace() == 0) {
      auto maybeOffset = computeGEPOffset(op, ctx);
      // Forward mapper entry so make.buffer.rsrc can find the SRD.
      auto mapped = ctx.getMapper().getMapped(base);
      if (mapped)
        ctx.getMapper().mapValue(op.getResult(), *mapped);

      if (!maybeOffset) {
        // All-zero GEP — just forward base offset.
        Value prevOffset = st.getBaseOffset(base);
        if (prevOffset)
          st.setBaseOffset(op.getResult(), prevOffset);
        return success();
      }

      Value newOffset = *maybeOffset;
      // Accumulate base offset.
      Value prevOffset = st.getBaseOffset(base);
      if (prevOffset) {
        auto vregTy = ctx.createVRegType();
        newOffset =
            V_ADD_U32::create(builder, loc, vregTy, prevOffset, newOffset);
      }
      st.setBaseOffset(op.getResult(), newOffset);
      return success();
    }
  }

  // Buffer GEP (ptr<7>) — must have single dynamic index.
  auto indices = op.getIndices();
  if (indices.size() != 1)
    return op->emitOpError("buffer GEP must have a single index");
  auto idx = indices[0].dyn_cast<Value>();
  if (!idx)
    return op->emitOpError("buffer GEP with constant index not yet supported");
  Value newOffset = resolve(idx, ctx);

  // Buffer GEP (ptr<7>): decompose into (SRD, voffset).
  auto srd = st.lookupSRD(base);
  if (srd) {
    // Check if the make.buffer.rsrc had a base offset from bare-pointer GEPs.
    Value baseOff = st.getBaseOffset(base);
    if (baseOff) {
      auto vregTy = ctx.createVRegType();
      newOffset = V_ADD_U32::create(builder, loc, vregTy, baseOff, newOffset);
    }
    st.mapGEP(op.getResult(), {*srd, newOffset});
    return success();
  }

  auto *baseGEP = st.lookupGEP(base);
  if (!baseGEP)
    return op->emitOpError("GEP base is not a tracked buffer resource");

  // Chain: add this offset to the base GEP's offset.
  auto vregTy = ctx.createVRegType();
  auto sum =
      V_ADD_U32::create(builder, loc, vregTy, baseGEP->voffset, newOffset);
  st.mapGEP(op.getResult(), {baseGEP->srd, sum});
  return success();
}

static LogicalResult handleLoad(LLVM::LoadOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  // LDS load (ptr<3>).
  if (st.isLDSBase(op.getAddr()))
    return handleLDSLoad(op, op.getAddr(), st);

  auto *ptr = st.lookupGEP(op.getAddr());
  if (!ptr)
    return op->emitOpError("load address not from a tracked GEP");

  int64_t numBytes = getBufferAccessBytes(op.getResult().getType());

  auto soffsetTy = ctx.createImmType(0);
  auto zeroSoffset = ConstantOp::create(builder, loc, soffsetTy, 0);
  auto vregTy = ctx.createVRegType();

  Operation *loadOp = nullptr;
  if (numBytes == 2)
    loadOp = BUFFER_LOAD_USHORT::create(builder, loc, TypeRange{vregTy},
                                        ptr->srd, ptr->voffset, zeroSoffset,
                                        /*instOffset=*/0);
  else if (numBytes == 4)
    loadOp =
        BUFFER_LOAD_DWORD::create(builder, loc, TypeRange{vregTy}, ptr->srd,
                                  ptr->voffset, zeroSoffset, /*instOffset=*/0);
  else if (numBytes == 8) {
    auto wideTy = ctx.createVRegType(2, 2);
    loadOp = BUFFER_LOAD_DWORDX2::create(builder, loc, TypeRange{wideTy},
                                         ptr->srd, ptr->voffset, zeroSoffset,
                                         /*instOffset=*/0);
  } else if (numBytes == 12) {
    auto wideTy = ctx.createVRegType(3, 3);
    loadOp = BUFFER_LOAD_DWORDX3::create(builder, loc, TypeRange{wideTy},
                                         ptr->srd, ptr->voffset, zeroSoffset,
                                         /*instOffset=*/0);
  } else if (numBytes == 16) {
    auto wideTy = ctx.createVRegType(4, 4);
    loadOp = BUFFER_LOAD_DWORDX4::create(builder, loc, TypeRange{wideTy},
                                         ptr->srd, ptr->voffset, zeroSoffset,
                                         /*instOffset=*/0);
  } else
    return op->emitOpError("unsupported load size: ") << numBytes << " bytes";

  ctx.getMapper().mapValue(op.getResult(), loadOp->getResult(0));
  return success();
}

static LogicalResult handleStore(LLVM::StoreOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  // LDS store (ptr<3>).
  if (st.isLDSBase(op.getAddr()))
    return handleLDSStore(op, op.getAddr(), st);

  auto *ptr = st.lookupGEP(op.getAddr());
  if (!ptr)
    return op->emitOpError("store address not from a tracked GEP");

  Value data = resolve(op.getValue(), ctx);
  int64_t numBytes = getBufferAccessBytes(op.getValue().getType());

  if (numBytes == 2)
    BUFFER_STORE_SHORT::create(builder, loc, data, ptr->srd, ptr->voffset,
                               /*instOffset=*/0);
  else if (numBytes == 4)
    BUFFER_STORE_DWORD::create(builder, loc, data, ptr->srd, ptr->voffset,
                               /*instOffset=*/0);
  else if (numBytes == 8)
    BUFFER_STORE_DWORDX2::create(builder, loc, data, ptr->srd, ptr->voffset,
                                 /*instOffset=*/0);
  else if (numBytes == 12)
    BUFFER_STORE_DWORDX3::create(builder, loc, data, ptr->srd, ptr->voffset,
                                 /*instOffset=*/0);
  else if (numBytes == 16)
    BUFFER_STORE_DWORDX4::create(builder, loc, data, ptr->srd, ptr->voffset,
                                 /*instOffset=*/0);
  else
    return op->emitOpError("unsupported store size: ") << numBytes << " bytes";

  return success();
}

//===----------------------------------------------------------------------===//
// Op dispatch
//===----------------------------------------------------------------------===//

static LogicalResult translateOp(Operation *op, LLVMTranslationState &st) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](LLVM::ConstantOp o) { return handleConstant(o, st); })
      .Case([&](LLVM::PoisonOp o) { return handlePoison(o, st); })
      .Case([&](ROCDL::ThreadIdXOp o) { return handleWorkitemIdX(o, st); })
      .Case([&](ROCDL::BlockIdXOp o) { return handleWorkgroupId(o, st, 0); })
      .Case([&](ROCDL::BlockIdYOp o) { return handleWorkgroupId(o, st, 1); })
      .Case([&](ROCDL::BlockIdZOp o) { return handleWorkgroupId(o, st, 2); })
      .Case([&](LLVM::SExtOp o) { return handleSext(o, st); })
      .Case([&](LLVM::ZExtOp o) { return handleZext(o, st); })
      .Case([&](LLVM::TruncOp o) { return handleTrunc(o, st); })
      .Case([&](LLVM::ICmpOp o) { return handleICmp(o, st); })
      .Case([&](LLVM::SelectOp o) { return handleSelect(o, st); })
      .Case([&](LLVM::MulOp o) { return handleMul(o, st); })
      .Case([&](LLVM::AddOp o) { return handleAdd(o, st); })
      .Case([&](ROCDL::MakeBufferRsrcOp o) {
        return handleMakeBufferRsrc(o, st);
      })
      .Case([&](LLVM::GEPOp o) { return handleGEP(o, st); })
      .Case([&](LLVM::LoadOp o) { return handleLoad(o, st); })
      .Case([&](LLVM::StoreOp o) { return handleStore(o, st); })
      .Case([&](LLVM::AddressOfOp o) { return handleAddressOf(o, st); })
      .Case([&](LLVM::SDivOp o) { return handleSDiv(o, st); })
      .Case([&](LLVM::SRemOp o) { return handleSRem(o, st); })
      .Case([&](LLVM::FenceOp o) { return handleFence(o, st); })
      .Case([&](LLVM::ShuffleVectorOp o) { return handleShuffleVector(o, st); })
      .Case([&](ROCDL::BarrierOp o) { return handleBarrier(o, st); })
      .Case([&](ROCDL::SBarrierOp o) { return handleBarrier(o, st); })
      .Case([&](ROCDL::mfma_f32_16x16x16f16 o) {
        return handleMFMA_F32_16x16x16_F16(o, st);
      })
      .Case([&](scf::ForOp o) { return handleSCFFor(o, st); })
      .Case([&](scf::YieldOp) { return success(); }) // Handled inside ForOp.
      .Default([](Operation *op) {
        return op->emitOpError("unhandled op in LLVM->WaveASM translation");
      });
}

//===----------------------------------------------------------------------===//
// Core translation logic
//===----------------------------------------------------------------------===//

static LogicalResult translateLLVMModule(ModuleOp module, StringRef targetId) {
  auto target = getTargetKindAttr(module.getContext(), targetId);
  if (!target)
    return module.emitError() << "unknown target: " << targetId;

  SmallVector<LLVM::LLVMFuncOp> kernels;
  module.walk([&](LLVM::LLVMFuncOp func) {
    if (func->hasAttr("gpu.kernel") || func->hasAttr("rocdl.kernel"))
      kernels.push_back(func);
  });

  if (kernels.empty())
    return module.emitError() << "no llvm.func kernel found in module";

  for (auto func : kernels) {
    OpBuilder builder(module.getContext());
    // Insert program inside the gpu.module that contains this kernel.
    auto *parentOp = func->getParentOp();
    if (auto gpuModule = dyn_cast<gpu::GPUModuleOp>(parentOp))
      builder.setInsertionPointToEnd(gpuModule.getBody());
    else
      builder.setInsertionPointToEnd(module.getBody());

    auto program = createProgramFromLLVMFunc(func, builder, targetId);
    builder.setInsertionPointToStart(&program.getBodyBlock());
    TranslationContext ctx(builder, program, target);
    LLVMTranslationState st(ctx);

    // Map llvm.func arguments: pointers get SRD setup, scalars get mapped
    // to their preloaded SGPR positions directly.
    SmallVector<BlockArgument> scalarArgs;
    for (auto arg : func.getBody().getArguments()) {
      if (isa<LLVM::LLVMPointerType>(arg.getType())) {
        int64_t argIdx = arg.getArgNumber();
        ctx.queueSRDSetup(arg, argIdx, /*bufferSize=*/0x7FFFFFFC);
      } else {
        scalarArgs.push_back(arg);
      }
    }

    ctx.setTotalKernelArgs(func.getNumArguments());
    ctx.emitSRDPrologue();

    // Map scalar (non-pointer) args to their preloaded SGPR positions.
    // On gfx950, arg N is preloaded at s[2+N*2 : 2+N*2+1] (64-bit each).
    for (auto arg : scalarArgs) {
      int64_t argIdx = arg.getArgNumber();
      int64_t preloadBase = 2 + argIdx * 2;
      auto sregTy = ctx.createSRegType(2, 2);
      auto sreg = PrecoloredSRegOp::create(builder, arg.getLoc(), sregTy,
                                           preloadBase, /*size=*/2);
      ctx.getMapper().mapValue(arg, sreg);
    }

    // Enable all workgroup IDs so the SGPR layout is predictable.
    // The real LLVM backend does the same (enables all three).
    ctx.enableAllWorkgroupIds();

    for (Operation &op : func.getBody().front()) {
      if (isa<LLVM::ReturnOp>(op))
        continue;
      if (failed(translateOp(&op, st)))
        return failure();
    }

    S_ENDPGM::create(builder, func.getLoc());

    program->setAttr("num_kernel_args",
                     builder.getI64IntegerAttr(func.getNumArguments()));

    int64_t ldsSize = ctx.getTotalLDSSize();
    if (ldsSize > 0)
      program->setAttr("lds_size", builder.getI64IntegerAttr(ldsSize));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace {

struct WAVEASMTranslateFromLLVMPass
    : impl::WAVEASMTranslateFromLLVMBase<WAVEASMTranslateFromLLVMPass> {
  using WAVEASMTranslateFromLLVMBase::WAVEASMTranslateFromLLVMBase;

  void runOnOperation() override {
    auto module = getOperation();

    bool hasLLVMKernels = false;
    module.walk([&](LLVM::LLVMFuncOp func) {
      if (func->hasAttr("gpu.kernel") || func->hasAttr("rocdl.kernel"))
        hasLLVMKernels = true;
    });

    if (!hasLLVMKernels)
      return;

    if (failed(translateLLVMModule(module, targetArch)))
      return signalPassFailure();
  }
};

} // namespace

} // namespace waveasm
