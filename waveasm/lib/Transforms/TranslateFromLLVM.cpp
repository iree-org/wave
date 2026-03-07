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

#include "waveasm/Transforms/TranslateFromLLVM.h"
#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Passes.h"
#include "waveasm/Transforms/TranslateFromMLIR.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
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

private:
  DenseMap<Value, Value> rsrcToSRD;
  DenseMap<Value, BufferPtrInfo> gepMap;
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

  auto program =
      ProgramOp::create(builder, loc, func.getName(), targetAttr, abiAttr,
                        /*vgprs=*/int64_t{256},
                        /*sgprs=*/int64_t{104},
                        /*workgroup_size=*/builder.getArrayAttr(sizes),
                        /*lds_size=*/IntegerAttr{});

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

//===----------------------------------------------------------------------===//
// Op handlers
//===----------------------------------------------------------------------===//

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

// i32↔i64 casts are identity on a 32-bit GPU.
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

  // The base pointer was set up as an SRD in the prologue via queueSRDSetup.
  Value basePtr = op.getBase();
  auto srdVal = ctx.getMapper().getMapped(basePtr);
  if (!srdVal)
    return op->emitOpError("SRD not found for base pointer");

  st.mapBufferRsrc(op.getResult(), *srdVal);
  return success();
}

static LogicalResult handleGEP(LLVM::GEPOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  Value base = op.getBase();

  // Base must be a make.buffer.rsrc result (ptr<7>).
  auto srd = st.lookupSRD(base);
  if (!srd)
    return op->emitOpError("GEP base is not a tracked buffer resource");

  // GEP index is a dynamic Value (not a constant attr).
  auto indices = op.getIndices();
  assert(indices.size() == 1 && "expected single GEP index");
  auto idx = indices[0].dyn_cast<Value>();
  if (!idx)
    return op->emitOpError("GEP with constant index attr not yet supported");

  Value voffset = resolve(idx, ctx);
  st.mapGEP(op.getResult(), {*srd, voffset});
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
  else
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
      .Case([&](ROCDL::ThreadIdXOp o) { return handleWorkitemIdX(o, st); })
      .Case([&](LLVM::SExtOp o) { return handleSext(o, st); })
      .Case([&](LLVM::ZExtOp o) { return handleZext(o, st); })
      .Case([&](LLVM::TruncOp o) { return handleTrunc(o, st); })
      .Case([&](LLVM::ICmpOp o) { return handleICmp(o, st); })
      .Case([&](LLVM::SelectOp o) { return handleSelect(o, st); })
      .Case([&](LLVM::MulOp o) { return handleMul(o, st); })
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

    // Map llvm.func arguments — all are !llvm.ptr (bare pointers).
    for (auto arg : func.getBody().getArguments()) {
      int64_t argIdx = arg.getArgNumber();
      ctx.queueSRDSetup(arg, argIdx, /*bufferSize=*/0x7FFFFFFC);
    }

    ctx.emitSRDPrologue();

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

    func.erase();
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
