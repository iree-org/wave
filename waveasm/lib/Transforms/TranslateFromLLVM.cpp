// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// TranslateFromLLVM: Strict LLVM dialect -> WaveASM translation.
//
// Consumes gpu.module { llvm.func @kernel ... } with rocdl intrinsics.
// Fails on any unhandled op — no silent fallthrough.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/AssemblyEmitter.h"
#include "waveasm/Transforms/Passes.h"
#include "waveasm/Transforms/TranslateFromMLIR.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
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

/// State for LLVM->WaveASM translation, layered on top of TranslationContext.
class LLVMTranslationState {
public:
  explicit LLVMTranslationState(TranslationContext &ctx) : ctx(ctx) {}

  TranslationContext &ctx;

  /// Map rocdl.make.buffer.rsrc result -> SRD SGPR value from prologue.
  void mapBufferRsrc(Value rsrc, Value srd) { rsrcToSRD[rsrc] = srd; }
  Value lookupSRD(Value rsrc) const { return rsrcToSRD.lookup(rsrc); }

  /// Map GEP result -> decomposed (SRD, voffset).
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
  Value lookupBaseOffset(Value ptr) const {
    auto it = baseOffsets.find(ptr);
    return it != baseOffsets.end() ? it->second : Value{};
  }

private:
  DenseMap<Value, Value> rsrcToSRD;
  DenseMap<Value, BufferPtrInfo> gepMap;
  DenseMap<Value, Value> baseOffsets;
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

  program->setAttr(kKernelNameAttr, builder.getStringAttr(func.getName()));
  if (program.getBody().empty())
    program.getBody().emplaceBlock();
  return program;
}

/// Resolve an LLVM SSA value to its WaveASM counterpart via the mapper
/// or return the value itself if unmapped.
static Value resolve(Value v, TranslationContext &ctx) {
  if (auto mapped = ctx.getMapper().getMapped(v))
    return *mapped;
  return v;
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

static LogicalResult handleThreadIdX(ROCDL::ThreadIdXOp op,
                                     LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  // rocdl.workitem.id.x -> hardware v0 (flat workitem ID).
  ctx.setUsesWorkitemId(true);
  auto vregTy = ctx.createVRegType();
  auto v0 = PrecoloredVRegOp::create(builder, loc, vregTy, /*regIndex=*/0,
                                     /*size=*/1);
  ctx.getMapper().mapValue(op.getResult(), v0);
  return success();
}

// rocdl.workgroup.id.{x,y,z} -> system SGPRs (set by hardware dispatch).
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

// i32<->i64 casts are identity on a 32-bit GPU.
static LogicalResult handleSext(LLVM::SExtOp op, LLVMTranslationState &st) {
  st.ctx.getMapper().mapValue(op.getResult(), resolve(op.getOperand(), st.ctx));
  return success();
}

static LogicalResult handleZext(LLVM::ZExtOp op, LLVMTranslationState &st) {
  st.ctx.getMapper().mapValue(op.getResult(), resolve(op.getOperand(), st.ctx));
  return success();
}

static LogicalResult handleTrunc(LLVM::TruncOp op, LLVMTranslationState &st) {
  st.ctx.getMapper().mapValue(op.getResult(), resolve(op.getOperand(), st.ctx));
  return success();
}

static LogicalResult handleICmp(LLVM::ICmpOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  Value lhs = resolve(op.getLhs(), ctx);
  Value rhs = resolve(op.getRhs(), ctx);

  // V_CMP_* ops set VCC implicitly (no SSA result).
  using Pred = LLVM::ICmpPredicate;
  switch (op.getPredicate()) {
  case Pred::slt:
    V_CMP_LT_I32::create(builder, loc, lhs, rhs);
    break;
  case Pred::sgt:
    V_CMP_GT_I32::create(builder, loc, lhs, rhs);
    break;
  case Pred::sle:
    V_CMP_LE_I32::create(builder, loc, lhs, rhs);
    break;
  case Pred::sge:
    V_CMP_GE_I32::create(builder, loc, lhs, rhs);
    break;
  case Pred::eq:
    V_CMP_EQ_I32::create(builder, loc, lhs, rhs);
    break;
  case Pred::ne:
    V_CMP_NE_I32::create(builder, loc, lhs, rhs);
    break;
  case Pred::ult:
    V_CMP_LT_U32::create(builder, loc, lhs, rhs);
    break;
  case Pred::ugt:
    V_CMP_GT_U32::create(builder, loc, lhs, rhs);
    break;
  case Pred::ule:
    V_CMP_LE_U32::create(builder, loc, lhs, rhs);
    break;
  case Pred::uge:
    V_CMP_GE_U32::create(builder, loc, lhs, rhs);
    break;
  }

  // Map result to a placeholder — V_CNDMASK reads VCC implicitly.
  auto immOne = ctx.createImmType(1);
  auto placeholder = ConstantOp::create(builder, loc, immOne, 1);
  ctx.getMapper().mapValue(op.getResult(), placeholder);
  return success();
}

static LogicalResult handleSelect(LLVM::SelectOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  Value cond = resolve(op.getCondition(), ctx);
  Value trueVal = resolve(op.getTrueValue(), ctx);
  Value falseVal = resolve(op.getFalseValue(), ctx);

  // v_cndmask_b32: dst = vcc ? src1 : src0 (falseVal=src0, trueVal=src1).
  auto vregTy = ctx.createVRegType();
  auto sel =
      V_CNDMASK_B32::create(builder, loc, vregTy, falseVal, trueVal, cond);
  ctx.getMapper().mapValue(op.getResult(), sel);
  return success();
}

static LogicalResult handleAdd(LLVM::AddOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  Value lhs = resolve(op.getLhs(), ctx);
  Value rhs = resolve(op.getRhs(), ctx);

  auto vregTy = ctx.createVRegType();
  auto add = V_ADD_U32::create(builder, loc, vregTy, lhs, rhs);
  ctx.getMapper().mapValue(op.getResult(), add);
  return success();
}

static LogicalResult handleMul(LLVM::MulOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  Value lhs = resolve(op.getLhs(), ctx);
  Value rhs = resolve(op.getRhs(), ctx);

  auto vregTy = ctx.createVRegType();
  auto mul = V_MUL_LO_U32::create(builder, loc, vregTy, lhs, rhs);
  ctx.getMapper().mapValue(op.getResult(), mul);
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
    auto flags = getConstantIntValue(op.getFlags());
    if (flags && *flags != 0x20000) {
      std::string movFlags = "s_mov_b32 s" + std::to_string(srdBase + 3) +
                             ", 0x" + llvm::utohexstr(*flags);
      RawOp::create(builder, loc, movFlags);
    }
  }

  st.mapBufferRsrc(op.getResult(), *srdVal);

  // Propagate any base offset from bare-pointer GEPs so buffer GEPs
  // can add it to their voffset.
  Value baseOff = st.lookupBaseOffset(basePtr);
  if (baseOff)
    st.setBaseOffset(op.getResult(), baseOff);

  return success();
}

static LogicalResult handleGEP(LLVM::GEPOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();
  Value base = op.getBase();

  // GEP index is a dynamic Value (not a constant attr).
  auto indices = op.getIndices();
  assert(indices.size() == 1 && "expected single GEP index");
  auto idx = indices[0].dyn_cast<Value>();
  if (!idx)
    return op->emitOpError("GEP with constant index attr not yet supported");

  Value newOffset = resolve(idx, ctx);

  // Bare-pointer GEP (!llvm.ptr, not <7>): pointer arithmetic before
  // make.buffer.rsrc. Propagate the mapper entry and accumulate
  // the byte offset so it can be added to voffset at load/store time.
  auto baseTy = op.getBase().getType();
  unsigned addrSpace = 0;
  if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(baseTy))
    addrSpace = ptrTy.getAddressSpace();

  if (addrSpace != 0 && addrSpace != 7)
    return op->emitOpError("unsupported address space ") << addrSpace;

  if (addrSpace == 0) {
    // Forward mapper entry so make.buffer.rsrc can find the SRD.
    std::optional<Value> mapped = ctx.getMapper().getMapped(base);
    if (mapped)
      ctx.getMapper().mapValue(op.getResult(), *mapped);

    // Accumulate base offset.
    Value prevOffset = st.lookupBaseOffset(base);
    if (prevOffset) {
      auto vregTy = ctx.createVRegType();
      newOffset =
          V_ADD_U32::create(builder, loc, vregTy, prevOffset, newOffset);
    }
    st.setBaseOffset(op.getResult(), newOffset);
    return success();
  }

  // Buffer GEP (ptr<7>): decompose into (SRD, voffset).
  auto srd = st.lookupSRD(base);
  if (srd) {
    // Check if the make.buffer.rsrc had a base offset from bare-pointer GEPs.
    Value baseOff = st.lookupBaseOffset(base);
    if (baseOff) {
      auto vregTy = ctx.createVRegType();
      newOffset = V_ADD_U32::create(builder, loc, vregTy, baseOff, newOffset);
    }
    st.mapGEP(op.getResult(), {srd, newOffset});
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

/// Compute buffer load/store size from the LLVM element type.
static int64_t getBufferAccessBytes(Type ty) {
  if (auto vecTy = dyn_cast<VectorType>(ty))
    return vecTy.getNumElements() *
           vecTy.getElementType().getIntOrFloatBitWidth() / 8;
  if (ty.isIntOrFloat())
    return ty.getIntOrFloatBitWidth() / 8;
  return 0;
}

static LogicalResult handleLoad(LLVM::LoadOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

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
      .Case([&](ROCDL::ThreadIdXOp o) { return handleThreadIdX(o, st); })
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
      .Default([](Operation *op) {
        return op->emitOpError("unhandled op in LLVM->WaveASM translation");
      });
}

//===----------------------------------------------------------------------===//
// Core translation logic
//===----------------------------------------------------------------------===//

static LogicalResult translateLLVMModule(Operation *rootOp,
                                         StringRef targetId) {
  auto target = getTargetKindAttr(rootOp->getContext(), targetId);
  if (!target)
    return rootOp->emitError() << "unknown target: " << targetId;

  SmallVector<LLVM::LLVMFuncOp> kernels;
  rootOp->walk([&](LLVM::LLVMFuncOp func) {
    if (func->hasAttr("gpu.kernel") || func->hasAttr("rocdl.kernel"))
      kernels.push_back(func);
  });

  if (kernels.empty())
    return success();

  for (LLVM::LLVMFuncOp func : kernels) {
    OpBuilder builder(rootOp->getContext());
    builder.setInsertionPointAfter(func);

    auto program = createProgramFromLLVMFunc(func, builder, targetId);
    builder.setInsertionPointToStart(&program.getBodyBlock());
    TranslationContext ctx(builder, program, target);
    LLVMTranslationState st(ctx);

    // Map llvm.func arguments — all are !llvm.ptr (bare pointers).
    for (auto arg : func.getBody().getArguments()) {
      int64_t argIdx = arg.getArgNumber();
      ctx.queueSRDSetup(arg, argIdx, /*bufferSize=*/0x7FFFFFFC);
    }

    ctx.emitSRDPrologue();

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
                     builder.getI64IntegerAttr(
                         static_cast<int64_t>(ctx.getNumKernelArgs())));

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
    if (failed(translateLLVMModule(getOperation(), targetArch)))
      return signalPassFailure();
  }
};

} // namespace

} // namespace waveasm
