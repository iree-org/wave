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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

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

// AMDGPU SRD (Shader Resource Descriptor) constants.
// SRD is 4 consecutive SGPRs: [base_lo, base_hi|stride, num_records, flags].

/// Mask for SRD word 1 to keep only base_addr[47:32] (lower 16 bits).
static constexpr int64_t kSRDWord1BaseMask = 0xFFFF;
/// Default SRD word 3 flags set by the prologue (OOB_SELECT=2).
static constexpr int64_t kSRDDefaultFlags = 0x20000;
/// Default num_records when buffer size is unknown (max 4-byte-aligned value).
static constexpr int64_t kSRDDefaultNumRecords = 0x7FFFFFFC;

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
  explicit LLVMTranslationState(TranslationContext &ctx,
                                Operation *symbolTableOp)
      : ctx(ctx) {
    assert(symbolTableOp && "expected symbol table operation");
    symbolTableOp->walk([&](LLVM::GlobalOp global) {
      globalsByName[global.getSymName()] = global;
    });
  }

  TranslationContext &ctx;

  /// Map rocdl.make.buffer.rsrc result -> SRD SGPR value from prologue.
  void mapBufferRsrc(Value rsrc, Value srd) { rsrcToSRD[rsrc] = srd; }
  Value lookupSRD(Value rsrc) const { return rsrcToSRD.lookup(rsrc); }

  /// Map GEP result -> decomposed (SRD, voffset).
  void mapGEP(Value gep, BufferPtrInfo info) { gepMap[gep] = info; }
  std::optional<BufferPtrInfo> lookupGEP(Value gep) const {
    auto it = gepMap.find(gep);
    if (it != gepMap.end())
      return it->second;
    return std::nullopt;
  }

  /// Track base-pointer byte offset from bare-pointer GEPs.
  /// These offsets accumulate and get added to voffset when the pointer
  /// is used via make.buffer.rsrc + buffer GEP.
  void setBaseOffset(Value ptr, Value offset) { baseOffsets[ptr] = offset; }
  Value lookupBaseOffset(Value ptr) const { return baseOffsets.lookup(ptr); }

  /// Track the assigned LDS byte offset for each referenced llvm.mlir.global.
  void setLDSGlobalOffset(LLVM::GlobalOp global, int64_t offset) {
    ldsGlobalOffsets[global.getOperation()] = offset;
  }
  std::optional<int64_t> lookupLDSGlobalOffset(LLVM::GlobalOp global) const {
    auto it = ldsGlobalOffsets.find(global.getOperation());
    if (it != ldsGlobalOffsets.end())
      return it->second;
    return std::nullopt;
  }

  LLVM::GlobalOp lookupGlobal(StringRef name) const {
    return globalsByName.lookup(name);
  }

private:
  DenseMap<Value, Value> rsrcToSRD;
  DenseMap<Value, BufferPtrInfo> gepMap;
  DenseMap<Value, Value> baseOffsets;
  DenseMap<Operation *, int64_t> ldsGlobalOffsets;
  llvm::StringMap<LLVM::GlobalOp> globalsByName;
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Extract workgroup size from llvm.func attributes.
/// Returns failure if both gpu.known_block_size and
/// rocdl.reqd_work_group_size are present and disagree.
static FailureOr<std::tuple<int64_t, int64_t, int64_t>>
getWorkgroupSize(LLVM::LLVMFuncOp func) {
  auto gpuAttr = func->getAttrOfType<DenseI32ArrayAttr>("gpu.known_block_size");
  auto rocdlAttr =
      func->getAttrOfType<DenseI32ArrayAttr>("rocdl.reqd_work_group_size");

  if (gpuAttr && rocdlAttr && gpuAttr.asArrayRef() != rocdlAttr.asArrayRef())
    return func->emitOpError("contradicting workgroup size attributes: "
                             "gpu.known_block_size and "
                             "rocdl.reqd_work_group_size disagree");

  DenseI32ArrayAttr attr = gpuAttr ? gpuAttr : rocdlAttr;
  if (!attr)
    return std::tuple<int64_t, int64_t, int64_t>{64, 1, 1};

  auto vals = attr.asArrayRef();
  int64_t x = vals.size() > 0 ? vals[0] : 64;
  int64_t y = vals.size() > 1 ? vals[1] : 1;
  int64_t z = vals.size() > 2 ? vals[2] : 1;
  return std::tuple<int64_t, int64_t, int64_t>{x, y, z};
}

/// Create a waveasm.program from an llvm.func kernel.
static ProgramOp createProgramFromLLVMFunc(LLVM::LLVMFuncOp func,
                                           OpBuilder &builder,
                                           StringRef targetId) {
  auto *mlirCtx = builder.getContext();
  auto loc = func.getLoc();

  // Code object version 5: supports kernel argument preloading.
  auto targetAttr =
      TargetAttr::get(mlirCtx, getTargetKindAttr(mlirCtx, targetId),
                      /*code_object_version=*/5);
  auto abiAttr = KernelABIAttr::get(mlirCtx, /*tid=*/0, /*kernarg=*/0,
                                    /*wg_id_x=*/std::nullopt,
                                    /*wg_id_y=*/std::nullopt,
                                    /*wg_id_z=*/std::nullopt);

  FailureOr<std::tuple<int64_t, int64_t, int64_t>> wgSize =
      getWorkgroupSize(func);
  if (failed(wgSize))
    return {};
  auto [wgX, wgY, wgZ] = *wgSize;
  std::array<Attribute, 3> sizes = {builder.getI64IntegerAttr(wgX),
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

  program->setAttr(WaveASMDialect::getKernelNameAttrName(),
                   builder.getStringAttr(func.getName()));
  assert(!program.getBody().empty() && "ProgramOp builder must create a block");
  return program;
}

/// Look up the WaveASM value that an LLVM value was translated to.
/// Returns failure if the value was never mapped. Never fall back to the
/// original LLVM SSA value: translation erases the LLVM ops after building
/// WaveASM, so reusing the source SSA value here would leave a dangling
/// reference.
static FailureOr<Value> resolve(Value v, TranslationContext &ctx) {
  if (auto mapped = ctx.getMapper().getMapped(v))
    return *mapped;
  // Block arguments (func params) are mapped during prologue setup. Reaching
  // this point means either malformed IR referenced an unmapped SSA value or
  // translation skipped a producer; surface that as a hard failure in the
  // caller instead of guessing through it.
  return failure();
}

/// Truncate an i64 WaveASM value to i32 via an arith.trunc pseudo-op.
/// Returns the value unchanged if the LLVM source type is already <= 32 bits.
static Value truncToI32(Value v, Type llvmType, OpBuilder &builder,
                        Location loc, TranslationContext &ctx) {
  auto intTy = dyn_cast<IntegerType>(llvmType);
  if (!intTy || intTy.getWidth() <= 32)
    return v;
  Type resTy = isVGPRType(v.getType()) ? (Type)ctx.createVRegType()
                                       : (Type)ctx.createSRegType();
  return ArithTruncOp::create(builder, loc, resTy, v);
}

/// Infer the pseudo-op result type from operand types.
/// Register file: VGPR if any operand is VGPR, else SGPR.
/// Width: max operand width (in 32-bit dwords).
static Type inferResultType(ValueRange operands, TranslationContext &ctx) {
  int64_t width = 1;
  for (Value v : operands)
    width = std::max(width, getRegSize(v.getType()));
  for (Value v : operands)
    if (isVGPRType(v.getType()))
      return ctx.createVRegType(width, width);
  return ctx.createSRegType(width, width);
}

/// Return the LLVM pointer address space, or 0 for non-pointer types.
static unsigned getLLVMAddrSpace(Value v) {
  if (LLVM::LLVMPointerType ptrTy =
          dyn_cast<LLVM::LLVMPointerType>(v.getType()))
    return ptrTy.getAddressSpace();
  return 0;
}

/// WaveASM register sizes are tracked in 32-bit dwords.
static int64_t getWaveASMDwordCount(int64_t bitWidth) {
  return llvm::divideCeil(bitWidth, int64_t{32});
}

/// Return the byte size for an LLVM type whose layout is obvious without a
/// DataLayout query.
///
/// We intentionally support only scalars, fixed vectors, and arrays thereof.
/// Nested aggregates such as structs require real DataLayout-based layout
/// reasoning.
static FailureOr<int64_t> getLLVMTypeBytes(Type ty) {
  if (ty.isIntOrFloat())
    return ty.getIntOrFloatBitWidth() / 8;
  if (VectorType vecTy = dyn_cast<VectorType>(ty)) {
    FailureOr<int64_t> elemBytes = getLLVMTypeBytes(vecTy.getElementType());
    if (failed(elemBytes))
      return failure();
    return *elemBytes * vecTy.getNumElements();
  }
  if (LLVM::LLVMArrayType arrTy = dyn_cast<LLVM::LLVMArrayType>(ty)) {
    FailureOr<int64_t> elemBytes = getLLVMTypeBytes(arrTy.getElementType());
    if (failed(elemBytes))
      return failure();
    return *elemBytes * arrTy.getNumElements();
  }
  return failure();
}

/// Return true iff a GEP index is statically known to be zero.
static bool isZeroGEPIndex(llvm::PointerUnion<IntegerAttr, Value> idx) {
  if (Value value = dyn_cast<Value>(idx))
    return isConstantIntValue(value, 0);

  return isConstantIntValue(cast<IntegerAttr>(idx), 0);
}

/// Structural GEPs like [0, 0] are a no-op and can be forwarded even though we
/// do not model nested aggregate layouts yet.
static bool isAllZeroIndexGEP(LLVM::GEPOp op) {
  return llvm::all_of(op.getIndices(), isZeroGEPIndex);
}

/// Compute the non-zero byte offset for a supported single-index GEP.
/// Emits a diagnostic and returns failure for unsupported element types or
/// unmapped dynamic indices.
static FailureOr<Value> computeGEPByteOffset(LLVM::GEPOp op,
                                             TranslationContext &ctx) {
  OpBuilder &builder = ctx.getBuilder();
  Location loc = op.getLoc();
  LLVM::GEPIndicesAdaptor<ValueRange> indices = op.getIndices();

  // Only handle single-index GEPs. Multi-index GEPs walk nested types
  // and require full DataLayout support.
  if (indices.size() != 1)
    return op->emitOpError("GEP byte offset requires a single index");

  FailureOr<int64_t> elemBytes = getLLVMTypeBytes(op.getElemType());
  if (failed(elemBytes))
    return op->emitOpError(
               "unsupported GEP element type for byte offset computation "
               "without DataLayout: ")
           << op.getElemType();

  llvm::PointerUnion<IntegerAttr, Value> idx = indices[0];
  if (Value dynIdx = dyn_cast<Value>(idx)) {
    FailureOr<Value> resolved = resolve(dynIdx, ctx);
    if (failed(resolved))
      return op->emitOpError("unmapped GEP index");
    if (*elemBytes == 1)
      return *resolved;
    // Scale by element size: offset = index * elemBytes.
    ImmType scaleTy = ctx.createImmType(*elemBytes);
    Value scale = ConstantOp::create(builder, loc, scaleTy, *elemBytes);
    Type resTy = inferResultType({*resolved, scale}, ctx);
    return ArithMulOp::create(builder, loc, resTy, *resolved, scale)
        .getResult();
  }

  IntegerAttr constIdxAttr = dyn_cast<IntegerAttr>(idx);
  assert(constIdxAttr && "GEP index must be a Value or IntegerAttr");
  int64_t constIdx = constIdxAttr.getInt();
  int64_t byteOffset = constIdx * *elemBytes;
  ImmType immTy = ctx.createImmType(byteOffset);
  return ConstantOp::create(builder, loc, immTy, byteOffset).getResult();
}

/// DS instructions address LDS by a 32-bit byte offset, not a generic pointer.
/// Valid in-range LDS accesses are bounded by the kernel's per-workgroup LDS
/// allocation, so well-formed LLVM IR may represent the offset as i64 but still
/// fits in the hardware vaddr field after truncation.
static Value materializeLDSVAddr(Value offset, OpBuilder &builder, Location loc,
                                 TranslationContext &ctx) {
  if (getRegSize(offset.getType()) <= 1)
    return offset;
  VRegType vregTy = ctx.createVRegType();
  return ArithTruncOp::create(builder, loc, vregTy, offset);
}

//===----------------------------------------------------------------------===//
// Op handlers
//===----------------------------------------------------------------------===//

static LogicalResult handlePoison(LLVM::PoisonOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  // Poison is undefined -- materialize as zero.  Must be mapped because
  // downstream ops (e.g. GEP, add) may reference the poison result via
  // resolve(), which now requires every LLVM value to have a mapping.
  auto intType = dyn_cast<IntegerType>(op.getResult().getType());
  if (!intType)
    return op->emitOpError("expected integer poison");

  auto immTy = ctx.createImmType(0);
  auto zeroImm = ConstantOp::create(builder, loc, immTy, int64_t{0});
  auto vregTy = ctx.createVRegType();

  if (intType.getWidth() <= 32) {
    Value mov = V_MOV_B32::create(builder, loc, vregTy, zeroImm);
    ctx.getMapper().mapValue(op.getResult(), mov);
    return success();
  }

  if (intType.getWidth() <= 64) {
    Value loMov = V_MOV_B32::create(builder, loc, vregTy, zeroImm);
    Value hiMov = V_MOV_B32::create(builder, loc, vregTy, zeroImm);
    auto wideTy = ctx.createVRegType(2, 2);
    Value packed =
        PackOp::create(builder, loc, wideTy, ValueRange{loMov, hiMov});
    ctx.getMapper().mapValue(op.getResult(), packed);
    return success();
  }

  return op->emitOpError("unsupported poison width (expected i32 or i64)");
}

static LogicalResult handleConstant(LLVM::ConstantOp op,
                                    LLVMTranslationState &st) {
  TranslationContext &ctx = st.ctx;
  OpBuilder &builder = ctx.getBuilder();
  Location loc = op.getLoc();

  Attribute valAttr = op.getValue();

  // Dense vector constant (e.g. MFMA accumulator init).
  if (DenseElementsAttr denseAttr = dyn_cast<DenseElementsAttr>(valAttr)) {
    if (!denseAttr.isSplat())
      return op->emitOpError("non-splat dense constant not yet supported");
    int64_t numElems = denseAttr.getNumElements();
    APFloat splatVal = denseAttr.getSplatValue<APFloat>();
    int64_t rawBits = splatVal.bitcastToAPInt().getZExtValue();
    ImmType immTy = ctx.createImmType(rawBits);
    Value immOp = ConstantOp::create(builder, loc, immTy, rawBits);
    VRegType vregTy = ctx.createVRegType(numElems, numElems);
    Value mov = V_MOV_B32::create(builder, loc, vregTy, immOp);
    ctx.getMapper().mapValue(op.getResult(), mov);
    return success();
  }

  int64_t intVal = 0;
  if (auto intAttr = dyn_cast<IntegerAttr>(valAttr))
    intVal = intAttr.getValue().getSExtValue();
  else
    return op->emitOpError("unsupported constant type");

  IntegerType intType = dyn_cast<IntegerType>(op.getResult().getType());
  if (!intType)
    return op->emitOpError("expected integer constant");

  if (intType.getWidth() <= 32) {
    auto immTy = ctx.createImmType(intVal);
    auto immOp = ConstantOp::create(builder, loc, immTy, intVal);
    auto vregTy = ctx.createVRegType();
    auto mov = V_MOV_B32::create(builder, loc, vregTy, immOp);
    ctx.getMapper().mapValue(op.getResult(), mov);
    return success();
  }

  if (intType.getWidth() <= 64) {
    // Split i64 constant into lo/hi halves and pack into vreg<2>.
    int32_t lo = static_cast<int32_t>(intVal & 0xFFFFFFFF);
    int32_t hi = static_cast<int32_t>(static_cast<uint64_t>(intVal) >> 32);
    auto vregTy = ctx.createVRegType();
    auto loImm = ConstantOp::create(builder, loc, ctx.createImmType(lo), lo);
    auto hiImm = ConstantOp::create(builder, loc, ctx.createImmType(hi), hi);
    Value loMov = V_MOV_B32::create(builder, loc, vregTy, loImm);
    Value hiMov = V_MOV_B32::create(builder, loc, vregTy, hiImm);
    auto wideTy = ctx.createVRegType(2, 2);
    Value packed =
        PackOp::create(builder, loc, wideTy, ValueRange{loMov, hiMov});
    ctx.getMapper().mapValue(op.getResult(), packed);
    return success();
  }

  return op->emitOpError("unsupported constant width (expected i32 or i64)");
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

/// Translate an LLVM cast op to a WaveASM arithmetic pseudo-op.
/// Result width is derived from the LLVM type in 32-bit dwords, so sext
/// i32->i64 produces a 2-wide register and trunc i64->i32 produces a 1-wide
/// one.
template <typename LLVMOp, typename WaveASMOp>
static LogicalResult handleCastOp(LLVMOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  FailureOr<Value> src = resolve(op.getOperand(), ctx);
  if (failed(src))
    return op->emitOpError("unmapped operand in cast");
  int64_t width = 1;
  if (IntegerType intTy = dyn_cast<IntegerType>(op.getResult().getType()))
    width = getWaveASMDwordCount(intTy.getWidth());
  Type resTy = isVGPRType(src->getType())
                   ? (Type)ctx.createVRegType(width, width)
                   : (Type)ctx.createSRegType(width, width);
  Value pseudo = WaveASMOp::create(builder, op.getLoc(), resTy, *src);
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
  FailureOr<Value> lhs = resolve(op.getLhs(), ctx);
  FailureOr<Value> rhs = resolve(op.getRhs(), ctx);
  if (failed(lhs) || failed(rhs))
    return op->emitOpError("unmapped operand in icmp");
  Type resTy = inferResultType({*lhs, *rhs}, ctx);
  auto pred = mapLLVMPredicate(op.getPredicate());
  Value cmp = ArithCmpOp::create(builder, op.getLoc(), resTy, pred, *lhs, *rhs);
  ctx.getMapper().mapValue(op.getResult(), cmp);
  return success();
}

static LogicalResult handleSelect(LLVM::SelectOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  FailureOr<Value> cond = resolve(op.getCondition(), ctx);
  FailureOr<Value> trueVal = resolve(op.getTrueValue(), ctx);
  FailureOr<Value> falseVal = resolve(op.getFalseValue(), ctx);
  if (failed(cond) || failed(trueVal) || failed(falseVal))
    return op->emitOpError("unmapped operand in select");
  Type resTy = inferResultType({*trueVal, *falseVal}, ctx);
  // ODS declaration order: (falseVal, trueVal, condition).
  Value sel = ArithSelectOp::create(builder, op.getLoc(), resTy, *falseVal,
                                    *trueVal, *cond);
  ctx.getMapper().mapValue(op.getResult(), sel);
  return success();
}

/// Translate an LLVM binary op to a WaveASM arithmetic pseudo-op.
/// Width validation is deferred to ArithLegalization.
template <typename LLVMOp, typename WaveASMOp>
static LogicalResult handleBinaryOp(LLVMOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  FailureOr<Value> lhs = resolve(op.getLhs(), ctx);
  FailureOr<Value> rhs = resolve(op.getRhs(), ctx);
  if (failed(lhs) || failed(rhs))
    return op->emitOpError("unmapped operand in binary op");
  Type resTy = inferResultType({*lhs, *rhs}, ctx);
  Value result = WaveASMOp::create(builder, op.getLoc(), resTy, *lhs, *rhs);
  ctx.getMapper().mapValue(op.getResult(), result);
  return success();
}

static LogicalResult handleAShr(LLVM::AShrOp op, LLVMTranslationState &st) {
  TranslationContext &ctx = st.ctx;
  OpBuilder &builder = ctx.getBuilder();
  Location loc = op.getLoc();
  FailureOr<Value> lhs = resolve(op.getLhs(), ctx);
  FailureOr<Value> rhs = resolve(op.getRhs(), ctx);
  if (failed(lhs) || failed(rhs))
    return op->emitOpError("unmapped operand in ashr");
  if (getRegSize(lhs->getType()) != 1 || getRegSize(rhs->getType()) != 1)
    return op->emitOpError(
        "arithmetic shift currently supports only i32 operands");

  Value result;
  if (isSGPRType(lhs->getType())) {
    if (isVGPRType(rhs->getType()))
      return op->emitOpError(
          "SGPR arithmetic shift requires an SGPR or immediate shift amount");
    SRegType sregTy = ctx.createSRegType();
    SCCType sccTy = ctx.createSCCType();
    result =
        S_ASHR_I32::create(builder, loc, sregTy, sccTy, *lhs, *rhs).getDst();
  } else {
    VRegType vregTy = ctx.createVRegType();
    result = V_ASHRREV_I32::create(builder, loc, vregTy, *rhs, *lhs);
  }

  ctx.getMapper().mapValue(op.getResult(), result);
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

  // The prologue seeds SRD[0:1] from the raw 64-bit pointer. That leaves SRD
  // word 1 bits [31:16] (stride/swizzle) populated with pointer bits, and
  // initializes SRD[3] to the default flag value. Rebuild the descriptor here
  // so make.buffer.rsrc controls both fields explicitly.
  bool needsPatch = (srdVal->getDefiningOp() != nullptr);
  if (needsPatch) {
    SRegType sregTy = ctx.createSRegType();

    // Extract individual SRD words from the source.
    Value word0 = ExtractOp::create(builder, loc, sregTy, *srdVal, 0);
    Value word1 = ExtractOp::create(builder, loc, sregTy, *srdVal, 1);
    Value word2 = ExtractOp::create(builder, loc, sregTy, *srdVal, 2);
    Value word3 = ExtractOp::create(builder, loc, sregTy, *srdVal, 3);

    // Clear stride/swizzle bits in SRD word 1 (keep only base_addr[47:32]).
    ImmType maskImm = ctx.createImmType(kSRDWord1BaseMask);
    Value maskVal =
        ConstantOp::create(builder, loc, maskImm, kSRDWord1BaseMask);
    word1 = S_AND_B32::create(builder, loc, sregTy, ctx.createSCCType(), word1,
                              maskVal)
                .getDst();

    // Patch SRD[3] with the requested descriptor flags.
    std::optional<int64_t> flags = getConstantIntValue(op.getFlags());
    if (flags && *flags != kSRDDefaultFlags) {
      ImmType flagsImm = ctx.createImmType(*flags);
      Value flagsVal = ConstantOp::create(builder, loc, flagsImm, *flags);
      word3 = S_MOV_B32::create(builder, loc, sregTy, flagsVal);
    }

    // If bare-pointer GEPs accumulated a byte offset before make.buffer.rsrc,
    // fold it into the SRD base address (64-bit SALU add to SRD[0:1]).
    Value baseOff = st.lookupBaseOffset(basePtr);
    if (baseOff) {
      // Make the offset scalar, preserving width (i32 or i64).
      int64_t width = getRegSize(baseOff.getType());
      SRegType scalarTy = ctx.createSRegType(width, width);
      Value offScalar =
          ArithReadFirstLaneOp::create(builder, loc, scalarTy, baseOff);

      // Split into lo/hi 32-bit halves for the 64-bit SRD base add.
      Value offLo = ArithTruncOp::create(builder, loc, sregTy, offScalar);
      Value offHi;
      if (width > 1)
        offHi = ExtractOp::create(builder, loc, sregTy, offScalar, 1);
      else
        offHi = ConstantOp::create(builder, loc, ctx.createImmType(0), 0);

      // Adjust base: S_ADD_U32 sets SCC, S_ADDC_U32 reads it.
      SCCType sccTy = ctx.createSCCType();
      S_ADD_U32 addLo =
          S_ADD_U32::create(builder, loc, sregTy, sccTy, word0, offLo);
      word0 = addLo.getDst();
      word1 = S_ADDC_U32::create(builder, loc, sregTy, sccTy, addLo.getScc(),
                                 word1, offHi)
                  .getDst();
    }

    // Pack into a 4-wide SGPR SRD.
    SRegType srdType = ctx.createSRegType(4, 4);
    Value newSrd = PackOp::create(builder, loc, srdType,
                                  ValueRange{word0, word1, word2, word3});
    st.mapBufferRsrc(op.getResult(), newSrd);
  } else {
    st.mapBufferRsrc(op.getResult(), *srdVal);
  }

  return success();
}

static LogicalResult handleGEP(LLVM::GEPOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();
  Value base = op.getBase();
  auto indices = op.getIndices();
  bool isSingleIndex = indices.size() == 1;
  bool isAllZeroGEP = isAllZeroIndexGEP(op);

  unsigned addrSpace = getLLVMAddrSpace(op.getBase());

  // LDS GEP (ptr<3>): compute byte offset for ds_read/ds_write.
  // DS instructions only accept a 32-bit vaddr, so truncate wide offsets.
  if (addrSpace == 3) {
    if (!isSingleIndex && !isAllZeroGEP)
      return op->emitOpError(
          "LDS GEP must have a single index or all-zero structural indices");
    FailureOr<Value> baseOff = resolve(base, ctx);
    if (failed(baseOff))
      return op->emitOpError("unmapped LDS GEP base");
    *baseOff = materializeLDSVAddr(*baseOff, builder, loc, ctx);
    if (isAllZeroGEP) {
      ctx.getMapper().mapValue(op.getResult(), *baseOff);
      return success();
    }
    FailureOr<Value> maybeOffset = computeGEPByteOffset(op, ctx);
    if (failed(maybeOffset))
      return failure();
    Value off = materializeLDSVAddr(*maybeOffset, builder, loc, ctx);
    VRegType vregTy = ctx.createVRegType();
    Value sum = ArithAddOp::create(builder, loc, vregTy, *baseOff, off);
    ctx.getMapper().mapValue(op.getResult(), sum);
    return success();
  }

  if (addrSpace != 0 && addrSpace != 7)
    return op->emitOpError("unsupported address space ") << addrSpace;

  // Bare-pointer GEP (!llvm.ptr, not <7>): 64-bit pointer arithmetic before
  // make.buffer.rsrc. Propagate the mapper entry and accumulate the byte
  // offset so it can be folded into the SRD base later.
  if (addrSpace == 0) {
    if (!isSingleIndex && !isAllZeroGEP)
      return op->emitOpError("bare-pointer GEP must have a single index or "
                             "all-zero structural indices");
    // Forward mapper entry so make.buffer.rsrc can find the SRD.
    if (std::optional<Value> mapped = ctx.getMapper().getMapped(base))
      ctx.getMapper().mapValue(op.getResult(), *mapped);

    if (isAllZeroGEP) {
      Value prevOffset = st.lookupBaseOffset(base);
      if (prevOffset)
        st.setBaseOffset(op.getResult(), prevOffset);
      return success();
    }

    FailureOr<Value> maybeOffset = computeGEPByteOffset(op, ctx);
    if (failed(maybeOffset))
      return failure();
    Value newOffset = *maybeOffset;
    Value prevOffset = st.lookupBaseOffset(base);
    if (prevOffset) {
      Type resTy = inferResultType({prevOffset, newOffset}, ctx);
      newOffset =
          ArithAddOp::create(builder, loc, resTy, prevOffset, newOffset);
    }
    st.setBaseOffset(op.getResult(), newOffset);
    return success();
  }

  // Buffer GEP (ptr<7>): single dynamic index.
  if (indices.size() != 1)
    return op->emitOpError("buffer GEP must have a single index");
  Value idx = indices[0].template dyn_cast<Value>();
  if (!idx)
    return op->emitOpError("buffer GEP with constant index not yet supported");

  FailureOr<Value> resolved = resolve(idx, ctx);
  if (failed(resolved))
    return op->emitOpError("unmapped GEP index");
  Value newOffset = truncToI32(*resolved, idx.getType(), builder, loc, ctx);

  // Check gepMap first -- covers both chained buffer GEPs and
  // make.buffer.rsrc entries seeded with a bare-pointer base offset.
  if (std::optional<BufferPtrInfo> baseGEP = st.lookupGEP(base)) {
    auto vregTy = ctx.createVRegType();
    Value sum =
        V_ADD_U32::create(builder, loc, vregTy, baseGEP->voffset, newOffset);
    st.mapGEP(op.getResult(), {baseGEP->srd, sum});
    return success();
  }

  // First buffer GEP directly on make.buffer.rsrc (no bare-pointer offset).
  Value srd = st.lookupSRD(base);
  if (srd) {
    st.mapGEP(op.getResult(), {srd, newOffset});
    return success();
  }

  return op->emitOpError("GEP base is not a tracked buffer resource");
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

/// Forward declaration for LDS handlers.
static LogicalResult handleLDSLoad(LLVM::LoadOp op, Value addr,
                                   LLVMTranslationState &st);
static LogicalResult handleLDSStore(LLVM::StoreOp op, Value addr,
                                    LLVMTranslationState &st);

static LogicalResult handleLoad(LLVM::LoadOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  // LDS load (ptr<3>).
  if (getLLVMAddrSpace(op.getAddr()) == 3)
    return handleLDSLoad(op, op.getAddr(), st);

  std::optional<BufferPtrInfo> ptr = st.lookupGEP(op.getAddr());
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
  if (getLLVMAddrSpace(op.getAddr()) == 3)
    return handleLDSStore(op, op.getAddr(), st);

  std::optional<BufferPtrInfo> ptr = st.lookupGEP(op.getAddr());
  if (!ptr)
    return op->emitOpError("store address not from a tracked GEP");

  FailureOr<Value> data = resolve(op.getValue(), ctx);
  if (failed(data))
    return op->emitOpError("unmapped store value");
  int64_t numBytes = getBufferAccessBytes(op.getValue().getType());

  if (numBytes == 2)
    BUFFER_STORE_SHORT::create(builder, loc, *data, ptr->srd, ptr->voffset,
                               /*instOffset=*/0);
  else if (numBytes == 4)
    BUFFER_STORE_DWORD::create(builder, loc, *data, ptr->srd, ptr->voffset,
                               /*instOffset=*/0);
  else if (numBytes == 8)
    BUFFER_STORE_DWORDX2::create(builder, loc, *data, ptr->srd, ptr->voffset,
                                 /*instOffset=*/0);
  else if (numBytes == 12)
    BUFFER_STORE_DWORDX3::create(builder, loc, *data, ptr->srd, ptr->voffset,
                                 /*instOffset=*/0);
  else if (numBytes == 16)
    BUFFER_STORE_DWORDX4::create(builder, loc, *data, ptr->srd, ptr->voffset,
                                 /*instOffset=*/0);
  else
    return op->emitOpError("unsupported store size: ") << numBytes << " bytes";

  return success();
}

//===----------------------------------------------------------------------===//
// LDS global / addressof handlers
//===----------------------------------------------------------------------===//

static LogicalResult handleAddressOf(LLVM::AddressOfOp op,
                                     LLVMTranslationState &st) {
  TranslationContext &ctx = st.ctx;
  LLVM::GlobalOp global = st.lookupGlobal(op.getGlobalName());
  if (!global)
    return op->emitOpError("referenced global not found");

  IntegerAttr addrSpaceAttr = global->getAttrOfType<IntegerAttr>("addr_space");
  if (!addrSpaceAttr || addrSpaceAttr.getInt() != 3)
    return op->emitOpError(
        "llvm.mlir.addressof currently supports only LDS globals in "
        "addrspace(3)");

  FailureOr<int64_t> sizeBytes = getLLVMTypeBytes(global.getType());
  if (failed(sizeBytes))
    return op->emitOpError(
               "unsupported LDS global type for byte size computation "
               "without DataLayout: ")
           << global.getType();

  int64_t baseOffset = 0;
  if (std::optional<int64_t> existingOffset =
          st.lookupLDSGlobalOffset(global)) {
    baseOffset = *existingOffset;
  } else {
    baseOffset = ctx.getLDSAllocOffset();
    assert(baseOffset >= 0 &&
           static_cast<uint64_t>(baseOffset) <=
               std::numeric_limits<uint32_t>::max() &&
           "LDS base offset must fit in the 32-bit DS vaddr field");
    st.setLDSGlobalOffset(global, baseOffset);
    ctx.addLDSSize(*sizeBytes);
    ctx.advanceLDSAllocOffset(*sizeBytes);
  }

  // Map to the byte offset assigned to this LDS global.
  OpBuilder &builder = ctx.getBuilder();
  ImmType immTy = ctx.createImmType(baseOffset);
  Value baseImm = ConstantOp::create(builder, op.getLoc(), immTy, baseOffset);
  VRegType vregTy = ctx.createVRegType();
  Value mov = V_MOV_B32::create(builder, op.getLoc(), vregTy, baseImm);
  ctx.getMapper().mapValue(op.getResult(), mov);
  return success();
}

//===----------------------------------------------------------------------===//
// LDS load/store handlers
//===----------------------------------------------------------------------===//

static LogicalResult handleLDSLoad(LLVM::LoadOp op, Value addr,
                                   LLVMTranslationState &st) {
  TranslationContext &ctx = st.ctx;
  OpBuilder &builder = ctx.getBuilder();
  Location loc = op.getLoc();

  int64_t numBytes = getBufferAccessBytes(op.getResult().getType());
  FailureOr<Value> offset = resolve(addr, ctx);
  if (failed(offset))
    return op->emitOpError("unmapped LDS address");
  *offset = materializeLDSVAddr(*offset, builder, loc, ctx);
  VRegType vregTy = ctx.createVRegType();

  Operation *loadOp = nullptr;
  if (numBytes == 2)
    loadOp = DS_READ_U16::create(builder, loc, TypeRange{vregTy}, *offset);
  else if (numBytes == 4)
    loadOp = DS_READ_B32::create(builder, loc, TypeRange{vregTy}, *offset);
  else if (numBytes == 8) {
    VRegType wideTy = ctx.createVRegType(2, 2);
    loadOp = DS_READ_B64::create(builder, loc, TypeRange{wideTy}, *offset);
  } else if (numBytes == 16) {
    VRegType wideTy = ctx.createVRegType(4, 4);
    loadOp = DS_READ_B128::create(builder, loc, TypeRange{wideTy}, *offset);
  } else {
    return op->emitOpError("unsupported LDS load size: ")
           << numBytes << " bytes";
  }

  ctx.getMapper().mapValue(op.getResult(), loadOp->getResult(0));
  return success();
}

static LogicalResult handleLDSStore(LLVM::StoreOp op, Value addr,
                                    LLVMTranslationState &st) {
  TranslationContext &ctx = st.ctx;
  OpBuilder &builder = ctx.getBuilder();
  Location loc = op.getLoc();

  FailureOr<Value> data = resolve(op.getValue(), ctx);
  FailureOr<Value> offset = resolve(addr, ctx);
  if (failed(data) || failed(offset))
    return op->emitOpError("unmapped LDS operand");
  *offset = materializeLDSVAddr(*offset, builder, loc, ctx);
  int64_t numBytes = getBufferAccessBytes(op.getValue().getType());

  if (numBytes == 2)
    DS_WRITE_B16::create(builder, loc, *data, *offset);
  else if (numBytes == 4)
    DS_WRITE_B32::create(builder, loc, *data, *offset);
  else if (numBytes == 8)
    DS_WRITE_B64::create(builder, loc, *data, *offset);
  else if (numBytes == 16)
    DS_WRITE_B128::create(builder, loc, *data, *offset);
  else
    return op->emitOpError("unsupported LDS store size: ")
           << numBytes << " bytes";

  return success();
}

//===----------------------------------------------------------------------===//
// Signed div/rem handlers
//===----------------------------------------------------------------------===//

static LogicalResult handleSDiv(LLVM::SDivOp op, LLVMTranslationState &st) {
  return op->emitOpError(
      "llvm.sdiv must be legalized before LLVM->WaveASM translation");
}

static LogicalResult handleSRem(LLVM::SRemOp op, LLVMTranslationState &st) {
  return op->emitOpError(
      "llvm.srem must be legalized before LLVM->WaveASM translation");
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
  TranslationContext &ctx = st.ctx;
  OpBuilder &builder = ctx.getBuilder();
  FailureOr<Value> src = resolve(op.getV1(), ctx);
  if (failed(src))
    return op->emitOpError("unmapped operand in shufflevector");

  // shufflevector with a single index extracts one element.
  llvm::ArrayRef<int32_t> mask = op.getMask();
  if (mask.size() == 1) {
    int64_t idx = mask[0];
    VRegType vregTy = ctx.createVRegType();
    Value extract = ExtractOp::create(builder, op.getLoc(), vregTy, *src, idx);
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
  TranslationContext &ctx = st.ctx;
  OpBuilder &builder = ctx.getBuilder();
  Location loc = op.getLoc();

  FailureOr<Value> a = resolve(op.getA(), ctx);
  FailureOr<Value> b = resolve(op.getB(), ctx);
  FailureOr<Value> c = resolve(op.getC(), ctx);
  if (failed(a) || failed(b) || failed(c))
    return op->emitOpError("unmapped operand in MFMA");

  // Result and accumulator are vector<4xf32> -> 4 VGPRs.
  VRegType accTy = ctx.createVRegType(4, 4);
  Value mfma = V_MFMA_F32_16X16X16_F16::create(builder, loc, accTy, *a, *b, *c);
  ctx.getMapper().mapValue(op.getResult(), mfma);
  return success();
}

//===----------------------------------------------------------------------===//
// SCF for/yield handler
//===----------------------------------------------------------------------===//

static LogicalResult translateOp(Operation *op, LLVMTranslationState &st);

static FailureOr<Value> materializeI32SReg(Value v, StringRef name,
                                           scf::ForOp op,
                                           TranslationContext &ctx,
                                           OpBuilder &builder, Location loc) {
  if (getRegSize(v.getType()) != 1) {
    op->emitOpError(name) << " must lower to i32";
    return failure();
  }
  SRegType sregTy = ctx.createSRegType();
  if (isSGPRType(v.getType()))
    return v;
  if (isImmType(v.getType()))
    return S_MOV_B32::create(builder, loc, sregTy, v).getResult();
  if (isVGPRType(v.getType()))
    return V_READFIRSTLANE_B32::create(builder, loc, sregTy, v).getResult();
  op->emitOpError(name) << " must lower to an SGPR or VGPR i32";
  return failure();
}

struct SCFForLoopBounds {
  Value initialIV;
  Value upperBound;
  Value step;
};

/// Resolve and scalarize scf.for loop control in the preheader.
///
/// scf.for evaluates lb/ub/step once before the first iteration. Capture those
/// values outside the lowered waveasm.loop body so the upper bound is not
/// implicitly rematerialized on the backedge.
static FailureOr<SCFForLoopBounds>
resolveSCFForLoopBounds(scf::ForOp op, LLVMTranslationState &st) {
  TranslationContext &ctx = st.ctx;
  OpBuilder &builder = ctx.getBuilder();
  Location loc = op.getLoc();

  FailureOr<Value> lowerBound = resolve(op.getLowerBound(), ctx);
  FailureOr<Value> upperBound = resolve(op.getUpperBound(), ctx);
  FailureOr<Value> step = resolve(op.getStep(), ctx);
  if (failed(lowerBound) || failed(upperBound) || failed(step)) {
    op->emitOpError("unmapped operand in scf.for");
    return failure();
  }

  // TODO: Preserve full scf.for zero-trip semantics. waveasm.loop is do-while
  // and this lowering currently assumes the trip count is known positive.
  FailureOr<Value> initialIV = materializeI32SReg(
      *lowerBound, "scf.for lower bound", op, ctx, builder, loc);
  FailureOr<Value> loopUpperBound = materializeI32SReg(
      *upperBound, "scf.for upper bound", op, ctx, builder, loc);
  FailureOr<Value> loopStep =
      materializeI32SReg(*step, "scf.for step", op, ctx, builder, loc);
  if (failed(initialIV) || failed(loopUpperBound) || failed(loopStep))
    return failure();

  return SCFForLoopBounds{
      /*initialIV=*/*initialIV,
      /*upperBound=*/*loopUpperBound,
      /*step=*/*loopStep,
  };
}

static LogicalResult handleSCFFor(scf::ForOp op, LLVMTranslationState &st) {
  TranslationContext &ctx = st.ctx;
  OpBuilder &builder = ctx.getBuilder();
  Location loc = op.getLoc();

  FailureOr<SCFForLoopBounds> loopBounds = resolveSCFForLoopBounds(op, st);
  if (failed(loopBounds))
    return failure();

  // Build init args: [lower_bound, iter_args...].
  SmallVector<Value> initArgs;
  initArgs.push_back(loopBounds->initialIV);
  for (Value arg : op.getInitArgs()) {
    FailureOr<Value> resolved = resolve(arg, ctx);
    if (failed(resolved))
      return op->emitOpError("unmapped init arg in scf.for");
    initArgs.push_back(*resolved);
  }

  // Create the waveasm.loop (do-while semantics).
  LoopOp loopOp = LoopOp::create(builder, loc, initArgs);
  Block &bodyBlock = loopOp.getBodyBlock();

  // Map the induction variable (block arg 0).
  ctx.getMapper().mapValue(op.getInductionVar(), bodyBlock.getArgument(0));

  // Map iter_args (block args 1..N).
  for (auto i : llvm::seq(op.getInitArgs().size()))
    ctx.getMapper().mapValue(op.getRegionIterArgs()[i],
                             bodyBlock.getArgument(i + 1));

  // Translate the loop body.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&bodyBlock);
  for (Operation &bodyOp : op.getBody()->without_terminator())
    if (failed(translateOp(&bodyOp, st)))
      return failure();

  // Build loop increment and condition.
  Value inductionVar = bodyBlock.getArgument(0);
  SRegType sregTy = ctx.createSRegType();
  SCCType sccTy = ctx.createSCCType();
  Value nextIV = S_ADD_U32::create(builder, loc, sregTy, sccTy, inductionVar,
                                   loopBounds->step)
                     .getDst();
  Value cond =
      S_CMP_LT_U32::create(builder, loc, sccTy, nextIV, loopBounds->upperBound);

  // Collect iter args from yield.
  scf::YieldOp yieldOp = cast<scf::YieldOp>(op.getBody()->getTerminator());
  SmallVector<Value> condIterArgs;
  condIterArgs.push_back(nextIV);
  for (Value v : yieldOp.getOperands()) {
    FailureOr<Value> resolved = resolve(v, ctx);
    if (failed(resolved))
      return op->emitOpError("unmapped yield operand in scf.for");
    condIterArgs.push_back(*resolved);
  }

  ConditionOp::create(builder, loc, cond, condIterArgs);

  // Map loop results. scf.for results are iter_args only (no IV),
  // but waveasm.loop results include the IV at index 0.
  for (auto i : llvm::seq(op.getNumResults()))
    ctx.getMapper().mapValue(op.getResult(i), loopOp.getResult(i + 1));

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
      .Case([&](LLVM::SExtOp o) {
        return handleCastOp<LLVM::SExtOp, ArithSExtOp>(o, st);
      })
      .Case([&](LLVM::ZExtOp o) {
        return handleCastOp<LLVM::ZExtOp, ArithZExtOp>(o, st);
      })
      .Case([&](LLVM::TruncOp o) {
        return handleCastOp<LLVM::TruncOp, ArithTruncOp>(o, st);
      })
      .Case([&](LLVM::ICmpOp o) { return handleICmp(o, st); })
      .Case([&](LLVM::SelectOp o) { return handleSelect(o, st); })
      .Case([&](LLVM::MulOp o) {
        return handleBinaryOp<LLVM::MulOp, ArithMulOp>(o, st);
      })
      .Case([&](LLVM::AddOp o) {
        return handleBinaryOp<LLVM::AddOp, ArithAddOp>(o, st);
      })
      .Case([&](LLVM::AShrOp o) { return handleAShr(o, st); })
      .Case([&](LLVM::OrOp o) {
        return handleBinaryOp<LLVM::OrOp, ArithOrOp>(o, st);
      })
      .Case([&](LLVM::AndOp o) {
        return handleBinaryOp<LLVM::AndOp, ArithAndOp>(o, st);
      })
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
      .Case([&](scf::YieldOp) { return success(); })
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

  if (!isa<GFX950TargetAttr>(target))
    return rootOp->emitError()
           << "LLVM->WaveASM translation only supports gfx950, got "
           << targetId;

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

    ProgramOp program = createProgramFromLLVMFunc(func, builder, targetId);
    if (!program)
      return failure();
    builder.setInsertionPointToStart(&program.getBodyBlock());
    TranslationContext ctx(builder, program, target);
    Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(func);
    assert(symbolTableOp && "expected nearest symbol table for llvm.func");
    LLVMTranslationState st(ctx, symbolTableOp);

    // Map llvm.func arguments: pointers get SRD setup, scalars get mapped
    // to their preloaded SGPR positions directly.
    SmallVector<BlockArgument> scalarArgs;
    for (auto arg : func.getBody().getArguments()) {
      if (isa<LLVM::LLVMPointerType>(arg.getType())) {
        int64_t argIdx = arg.getArgNumber();
        ctx.queueSRDSetup(arg, argIdx, /*bufferSize=*/kSRDDefaultNumRecords);
      } else {
        scalarArgs.push_back(arg);
        ctx.queueScalarArgLoad(arg, arg.getArgNumber());
      }
    }

    ctx.emitSRDPrologue();

    // Map scalar (non-pointer) args to their SGPR positions.
    // gfx950 hardware preloads arg N into s[2+N*2 : 2+N*2+1] (64-bit each).
    // Assumes all scalar args fit in the preload window (no overflow).
    for (auto arg : scalarArgs) {
      int64_t argIdx = arg.getArgNumber();
      int64_t preloadBase = 2 + argIdx * 2;
      auto sregTy = ctx.createSRegType(2, 2);
      auto sreg = PrecoloredSRegOp::create(builder, arg.getLoc(), sregTy,
                                           preloadBase, /*size=*/2);
      ctx.getMapper().mapValue(arg, sreg);
    }

    // Keep all workgroup IDs enabled so the SGPR layout is stable across
    // translated kernels. LLVM can disable y/z selectively via
    // amdgpu-no-workgroup-id-{y,z}, but the fixed WaveASM prologue layout here
    // assumes all three slots exist.
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
    if (failed(translateLLVMModule(getOperation(), targetArch)))
      return signalPassFailure();
  }
};

} // namespace

} // namespace waveasm
