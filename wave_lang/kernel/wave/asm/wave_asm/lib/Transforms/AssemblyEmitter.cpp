// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Kernel Generator - AMDGCN Assembly Code Generation
//
// Core code generation from WaveASM IR to AMDGCN assembly text.
// Handles value resolution, TypeSwitch-based op dispatch, and the main
// generate() loop. InstructionFormatter, MetadataEmitter, and literal
// materialization logic live in separate files.
//===----------------------------------------------------------------------===//

#include "waveasm/Transforms/AssemblyEmitter.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Target/AMDGCN/RegisterInfo.h"
#include "waveasm/Transforms/RegAlloc.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>

using namespace mlir;

namespace waveasm {

//===----------------------------------------------------------------------===//
// Kernel Generator Implementation (Pure SSA)
//===----------------------------------------------------------------------===//

KernelGenerator::KernelGenerator(ProgramOp program,
                                 const PhysicalMapping &mapping,
                                 TargetAttrInterface target)
    : program(program), mapping(mapping), target(target) {}

std::string KernelGenerator::resolveValue(Value value) {
  Type ty = value.getType();

  if (auto pvreg = dyn_cast<PVRegType>(ty)) {
    return formatVGPRRange(pvreg.getIndex(), pvreg.getSize());
  }
  if (auto psreg = dyn_cast<PSRegType>(ty)) {
    return formatSGPRRange(psreg.getIndex(), psreg.getSize());
  }

  if (isVirtualRegType(ty)) {
    int64_t physIdx = mapping.getPhysReg(value);
    if (physIdx >= 0) {
      int64_t size = getRegSize(ty);
      if (isVGPRType(ty)) {
        return formatVGPRRange(physIdx, size);
      } else {
        return formatSGPRRange(physIdx, size);
      }
    }
    return "%<ssa>";
  }

  if (isa<ImmType>(ty)) {
    if (auto defOp = value.getDefiningOp()) {
      if (auto constOp = dyn_cast<ConstantOp>(defOp)) {
        int64_t val = static_cast<int64_t>(constOp.getValue());
        return std::to_string(val);
      }
    }
    return "0";
  }

  return "<unknown>";
}

std::pair<bool, int64_t> KernelGenerator::getLiteralValue(Value value) {
  Type ty = value.getType();
  if (isa<ImmType>(ty)) {
    if (auto defOp = value.getDefiningOp()) {
      if (auto constOp = dyn_cast<ConstantOp>(defOp)) {
        int64_t val = static_cast<int64_t>(constOp.getValue());
        return {true, val};
      }
    }
  }
  return {false, 0};
}

//===----------------------------------------------------------------------===//
// TypeSwitch-based Operation Code Generation
//===----------------------------------------------------------------------===//

std::string KernelGenerator::emitBufferLoadLDS(Operation *op,
                                               llvm::StringRef mnemonic) {
  std::string result = "  " + mnemonic.str();
  if (op->getNumOperands() >= 3) {
    std::string voffset = resolveValue(op->getOperand(0));
    std::string srd = resolveValue(op->getOperand(1));
    std::string soffset = resolveValue(op->getOperand(2));
    result += " " + voffset + ", " + srd + ", " + soffset + " offen";
    if (auto instOffsetAttr = op->getAttrOfType<IntegerAttr>("instOffset")) {
      int64_t offset = instOffsetAttr.getInt();
      if (offset > 0) {
        result += " offset:" + std::to_string(offset);
      }
    }
    result += " lds";
  }
  return result;
}

std::string KernelGenerator::emitBufferLoad(Operation *op,
                                            llvm::StringRef mnemonic) {
  std::string result = "  " + mnemonic.str();
  std::string vdata;
  for (Value res : op->getResults()) {
    vdata = resolveValue(res);
  }
  if (op->getNumOperands() >= 2) {
    std::string voffset = resolveValue(op->getOperand(1));
    std::string srd = resolveValue(op->getOperand(0));
    result += " " + vdata + ", " + voffset + ", " + srd + ", 0 offen";
    if (auto instOffsetAttr = op->getAttrOfType<IntegerAttr>("instOffset")) {
      int64_t offset = instOffsetAttr.getInt();
      if (offset > 0) {
        result += " offset:" + std::to_string(offset);
      }
    }
  }
  return result;
}

std::string KernelGenerator::emitBufferStore(Operation *op,
                                             llvm::StringRef mnemonic) {
  std::string result = "  " + mnemonic.str();
  if (op->getNumOperands() >= 3) {
    std::string vdata = resolveValue(op->getOperand(0));
    std::string voffset = resolveValue(op->getOperand(2));
    std::string srd = resolveValue(op->getOperand(1));
    result += " " + vdata + ", " + voffset + ", " + srd + ", 0 offen";
    if (auto instOffsetAttr = op->getAttrOfType<IntegerAttr>("instOffset")) {
      int64_t offset = instOffsetAttr.getInt();
      if (offset > 0) {
        result += " offset:" + std::to_string(offset);
      }
    }
  }
  return result;
}

std::string KernelGenerator::emitGlobalLoad(Operation *op,
                                            llvm::StringRef mnemonic) {
  std::string result = "  " + mnemonic.str();
  std::string vdata;
  for (Value res : op->getResults()) {
    vdata = resolveValue(res);
  }
  if (op->getNumOperands() >= 2) {
    std::string vaddr = resolveValue(op->getOperand(1));
    std::string saddr = resolveValue(op->getOperand(0));
    result += " " + vdata + ", " + vaddr + ", " + saddr;
  } else if (op->getNumOperands() >= 1) {
    std::string vaddr = resolveValue(op->getOperand(0));
    result += " " + vdata + ", " + vaddr + ", off";
  }
  return result;
}

std::string KernelGenerator::emitGlobalStore(Operation *op,
                                             llvm::StringRef mnemonic) {
  std::string result = "  " + mnemonic.str();
  if (op->getNumOperands() >= 3) {
    std::string vdata = resolveValue(op->getOperand(0));
    std::string vaddr = resolveValue(op->getOperand(2));
    std::string saddr = resolveValue(op->getOperand(1));
    result += " " + vaddr + ", " + vdata + ", " + saddr;
  }
  return result;
}

std::string KernelGenerator::emitLDSRead(Operation *op,
                                         llvm::StringRef mnemonic) {
  std::string result = "  " + mnemonic.str();
  std::string vdst;
  for (Value res : op->getResults()) {
    vdst = resolveValue(res);
  }
  if (op->getNumOperands() >= 1) {
    std::string vaddr = resolveValue(op->getOperand(0));
    result += " " + vdst + ", " + vaddr;
  }
  if (auto offsetAttr = op->getAttrOfType<IntegerAttr>("offset")) {
    int64_t offset = offsetAttr.getInt();
    if (offset != 0) {
      result += " offset:" + std::to_string(offset);
    }
  }
  return result;
}

std::string KernelGenerator::emitLDSWrite(Operation *op,
                                          llvm::StringRef mnemonic) {
  std::string result = "  " + mnemonic.str();
  if (op->getNumOperands() >= 2) {
    std::string vaddr = resolveValue(op->getOperand(1));
    std::string vdata = resolveValue(op->getOperand(0));
    result += " " + vaddr + ", " + vdata;
  }
  if (auto offsetAttr = op->getAttrOfType<IntegerAttr>("offset")) {
    int64_t offset = offsetAttr.getInt();
    if (offset != 0) {
      result += " offset:" + std::to_string(offset);
    }
  }
  return result;
}

std::string KernelGenerator::resolveScalarValue(Value value) {
  Type ty = value.getType();
  if (auto pvreg = dyn_cast<PVRegType>(ty)) {
    if (pvreg.getSize() > 1) {
      return formatVGPRRange(pvreg.getIndex(), 1);
    }
  }
  if (auto psreg = dyn_cast<PSRegType>(ty)) {
    if (psreg.getSize() > 1) {
      return formatSGPRRange(psreg.getIndex(), 1);
    }
  }
  return resolveValue(value);
}

std::string KernelGenerator::emitDefaultFormat(Operation *op,
                                               llvm::StringRef mnemonic) {
  llvm::SmallVector<std::string> operands;

  bool isScalarOp = false;
  if (op->getNumResults() == 1) {
    Type resTy = op->getResult(0).getType();
    if (auto pvreg = dyn_cast<PVRegType>(resTy)) {
      isScalarOp = (pvreg.getSize() == 1);
    } else if (auto psreg = dyn_cast<PSRegType>(resTy)) {
      isScalarOp = (psreg.getSize() == 1);
    } else if (auto vreg = dyn_cast<VRegType>(resTy)) {
      isScalarOp = (vreg.getSize() == 1);
    } else if (auto sreg = dyn_cast<SRegType>(resTy)) {
      isScalarOp = (sreg.getSize() == 1);
    }
  }

  for (Value result : op->getResults()) {
    operands.push_back(resolveValue(result));
  }
  for (Value operand : op->getOperands()) {
    if (isScalarOp) {
      operands.push_back(resolveScalarValue(operand));
    } else {
      operands.push_back(resolveValue(operand));
    }
  }
  return formatter.format(mnemonic, operands);
}

std::optional<std::string>
KernelGenerator::emitScaledMFMA(Operation *scaledOp, llvm::StringRef mnemonic) {
  std::string line = emitDefaultFormat(scaledOp, mnemonic);
  int32_t cbsz = 4;
  int32_t blgp = 4;
  if (auto cbszAttr = scaledOp->getAttrOfType<IntegerAttr>("cbsz"))
    cbsz = cbszAttr.getInt();
  if (auto blgpAttr = scaledOp->getAttrOfType<IntegerAttr>("blgp"))
    blgp = blgpAttr.getInt();
  line += " cbsz:" + std::to_string(cbsz) + " blgp:" + std::to_string(blgp);
  return line;
}

std::optional<std::string> KernelGenerator::generateOp(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<std::string>>(op)
      .Case<ProgramOp, LabelOp, CommentOp, RawOp, PrecoloredVRegOp,
            PrecoloredSRegOp, ConstantOp, PackOp, ExtractOp>(
          [](auto) { return std::nullopt; })

      .Case<S_WAITCNT>([&](S_WAITCNT waitcntOp) {
        std::optional<int64_t> vmcnt, lgkmcnt, expcnt;
        if (auto vmAttr = waitcntOp.getVmcntAttr())
          vmcnt = vmAttr.getInt();
        if (auto lgkmAttr = waitcntOp.getLgkmcntAttr())
          lgkmcnt = lgkmAttr.getInt();
        if (auto expAttr = waitcntOp.getExpcntAttr())
          expcnt = expAttr.getInt();
        if (!vmcnt && !lgkmcnt && !expcnt) {
          vmcnt = 0;
          lgkmcnt = 0;
        }
        return formatter.formatWaitcnt(vmcnt, lgkmcnt, expcnt);
      })
      .Case<S_WAITCNT_VMCNT>([&](S_WAITCNT_VMCNT waitcntOp) {
        return formatter.formatWaitcnt(waitcntOp.getCount(), std::nullopt,
                                       std::nullopt);
      })
      .Case<S_WAITCNT_LGKMCNT>([&](S_WAITCNT_LGKMCNT waitcntOp) {
        return formatter.formatWaitcnt(std::nullopt, waitcntOp.getCount(),
                                       std::nullopt);
      })

      .Case<BUFFER_LOAD_DWORD>([&](auto loadOp) {
        return emitBufferLoad(loadOp, "buffer_load_dword");
      })
      .Case<BUFFER_LOAD_DWORDX2>([&](auto loadOp) {
        return emitBufferLoad(loadOp, "buffer_load_dwordx2");
      })
      .Case<BUFFER_LOAD_DWORDX3>([&](auto loadOp) {
        return emitBufferLoad(loadOp, "buffer_load_dwordx3");
      })
      .Case<BUFFER_LOAD_DWORDX4>([&](auto loadOp) {
        return emitBufferLoad(loadOp, "buffer_load_dwordx4");
      })

      .Case<BUFFER_LOAD_UBYTE>([&](auto loadOp) {
        return emitBufferLoad(loadOp, "buffer_load_ubyte");
      })
      .Case<BUFFER_LOAD_USHORT>([&](auto loadOp) {
        return emitBufferLoad(loadOp, "buffer_load_ushort");
      })

      // Buffer load to LDS (gather-to-LDS) -- shared helper avoids duplication
      .Case<BUFFER_LOAD_DWORD_LDS>([&](auto loadOp) {
        return emitBufferLoadLDS(loadOp, "buffer_load_dword");
      })
      .Case<BUFFER_LOAD_DWORDX4_LDS>([&](auto loadOp) {
        return emitBufferLoadLDS(loadOp, "buffer_load_dwordx4");
      })

      .Case<BUFFER_STORE_DWORD>([&](auto storeOp) {
        return emitBufferStore(storeOp, "buffer_store_dword");
      })
      .Case<BUFFER_STORE_DWORDX2>([&](auto storeOp) {
        return emitBufferStore(storeOp, "buffer_store_dwordx2");
      })
      .Case<BUFFER_STORE_DWORDX3>([&](auto storeOp) {
        return emitBufferStore(storeOp, "buffer_store_dwordx3");
      })
      .Case<BUFFER_STORE_DWORDX4>([&](auto storeOp) {
        return emitBufferStore(storeOp, "buffer_store_dwordx4");
      })

      .Case<GLOBAL_LOAD_DWORD>([&](auto loadOp) {
        return emitGlobalLoad(loadOp, "global_load_dword");
      })
      .Case<GLOBAL_LOAD_DWORDX2>([&](auto loadOp) {
        return emitGlobalLoad(loadOp, "global_load_dwordx2");
      })
      .Case<GLOBAL_LOAD_DWORDX3>([&](auto loadOp) {
        return emitGlobalLoad(loadOp, "global_load_dwordx3");
      })
      .Case<GLOBAL_LOAD_DWORDX4>([&](auto loadOp) {
        return emitGlobalLoad(loadOp, "global_load_dwordx4");
      })

      .Case<GLOBAL_STORE_DWORD>([&](auto storeOp) {
        return emitGlobalStore(storeOp, "global_store_dword");
      })
      .Case<GLOBAL_STORE_DWORDX2>([&](auto storeOp) {
        return emitGlobalStore(storeOp, "global_store_dwordx2");
      })
      .Case<GLOBAL_STORE_DWORDX3>([&](auto storeOp) {
        return emitGlobalStore(storeOp, "global_store_dwordx3");
      })
      .Case<GLOBAL_STORE_DWORDX4>([&](auto storeOp) {
        return emitGlobalStore(storeOp, "global_store_dwordx4");
      })

      .Case<DS_READ_B32>(
          [&](auto readOp) { return emitLDSRead(readOp, "ds_read_b32"); })
      .Case<DS_READ_B64>(
          [&](auto readOp) { return emitLDSRead(readOp, "ds_read_b64"); })
      .Case<DS_READ_B128>(
          [&](auto readOp) { return emitLDSRead(readOp, "ds_read_b128"); })
      .Case<DS_READ_U8>(
          [&](auto readOp) { return emitLDSRead(readOp, "ds_read_u8"); })
      .Case<DS_READ_U16>(
          [&](auto readOp) { return emitLDSRead(readOp, "ds_read_u16"); })

      .Case<DS_WRITE_B32>(
          [&](auto writeOp) { return emitLDSWrite(writeOp, "ds_write_b32"); })
      .Case<DS_WRITE_B64>(
          [&](auto writeOp) { return emitLDSWrite(writeOp, "ds_write_b64"); })
      .Case<DS_WRITE_B128>(
          [&](auto writeOp) { return emitLDSWrite(writeOp, "ds_write_b128"); })
      .Case<DS_WRITE_B8>(
          [&](auto writeOp) { return emitLDSWrite(writeOp, "ds_write_b8"); })
      .Case<DS_WRITE_B16>(
          [&](auto writeOp) { return emitLDSWrite(writeOp, "ds_write_b16"); })

      .Case<S_MOV_B32_M0>([&](S_MOV_B32_M0 movOp) {
        std::string result = "  s_mov_b32 m0";
        if (movOp->getNumOperands() >= 1) {
          result += ", " + resolveValue(movOp->getOperand(0));
        }
        return result;
      })

      .Case<S_BRANCH>([&](S_BRANCH branchOp) {
        return std::string("  s_branch ") +
               branchOp.getTarget().getRootReference().str();
      })
      .Case<S_CBRANCH_SCC0>([&](S_CBRANCH_SCC0 branchOp) {
        return std::string("  s_cbranch_scc0 ") +
               branchOp.getTarget().getRootReference().str();
      })
      .Case<S_CBRANCH_SCC1>([&](S_CBRANCH_SCC1 branchOp) {
        return std::string("  s_cbranch_scc1 ") +
               branchOp.getTarget().getRootReference().str();
      })

      .Case<S_NOP>([&](S_NOP nopOp) {
        return std::string("  s_nop ") + std::to_string(nopOp.getCount());
      })

      .Case<S_SETPRIO>([&](S_SETPRIO prioOp) {
        return std::string("  s_setprio ") + std::to_string(prioOp.getCount());
      })

      .Case<S_BARRIER>([](auto) { return std::string("  s_barrier"); })
      .Case<S_ENDPGM>([](auto) { return std::string("  s_endpgm"); })

      .Case<V_MOV_B32>([&](V_MOV_B32 movOp) -> std::optional<std::string> {
        Value result = movOp.getDst();
        int64_t size = getRegSize(result.getType());
        if (size > 1) {
          int64_t baseIdx = mapping.getPhysReg(result);
          if (baseIdx < 0) {
            if (auto pvreg = dyn_cast<PVRegType>(result.getType())) {
              baseIdx = pvreg.getIndex();
            }
          }
          if (baseIdx >= 0) {
            std::string src = resolveValue(movOp.getSrc());
            std::string lines;
            for (int64_t i = 0; i < size; ++i) {
              if (i > 0)
                lines += "\n";
              lines +=
                  "  v_mov_b32 v" + std::to_string(baseIdx + i) + ", " + src;
            }
            return lines;
          }
        }
        return emitDefaultFormat(movOp, "v_mov_b32");
      })

      .Case<V_MFMA_SCALE_F32_16X16X128_F8F6F4>(
          [&](auto scaledOp) -> std::optional<std::string> {
            return emitScaledMFMA(scaledOp,
                                  "v_mfma_scale_f32_16x16x128_f8f6f4");
          })
      .Case<V_MFMA_SCALE_F32_32X32X64_F8F6F4>(
          [&](auto scaledOp) -> std::optional<std::string> {
            return emitScaledMFMA(scaledOp, "v_mfma_scale_f32_32x32x64_f8f6f4");
          })

      .Case<LoopOp>([&](LoopOp loopOp) -> std::optional<std::string> {
        std::string labelName = "L_loop_" + std::to_string(loopLabelCounter++);

        std::string buf;
        llvm::raw_string_ostream os(buf);
        os << labelName << ":\n";

        Block &body = loopOp.getBodyBlock();
        for (Operation &bodyOp : body) {
          if (auto condOp = dyn_cast<ConditionOp>(&bodyOp)) {
            {
              unsigned numArgs = body.getNumArguments();
              unsigned numIter = condOp.getIterArgs().size();
              assert(numIter == numArgs &&
                     "ConditionOp iter_args count must match body block "
                     "argument count for correct register rotation");

              struct CopyInfo {
                int64_t dst;
                int64_t src;
                bool isSGPR;
              };
              SmallVector<CopyInfo> pendingCopies;

              auto getPhysRegInfo = [&](Value val) -> std::pair<int64_t, bool> {
                Type ty = val.getType();
                if (auto psreg = dyn_cast<PSRegType>(ty))
                  return {psreg.getIndex(), true};
                if (auto pvreg = dyn_cast<PVRegType>(ty))
                  return {pvreg.getIndex(), false};
                if (isVirtualRegType(ty))
                  return {mapping.getPhysReg(val), isSGPRType(ty)};
                return {-1, false};
              };

              for (unsigned i = 0; i < numIter; ++i) {
                auto [srcPhys, isSGPR] =
                    getPhysRegInfo(condOp.getIterArgs()[i]);
                auto [dstPhys, dstIsSGPR] = getPhysRegInfo(body.getArgument(i));

                if (srcPhys >= 0 && dstPhys >= 0 && srcPhys != dstPhys) {
                  pendingCopies.push_back({dstPhys, srcPhys, isSGPR});
                }
              }

              SmallVector<bool> handled(pendingCopies.size(), false);

              for (size_t i = 0; i < pendingCopies.size(); ++i) {
                if (handled[i])
                  continue;
                for (size_t j = i + 1; j < pendingCopies.size(); ++j) {
                  if (handled[j])
                    continue;
                  if (pendingCopies[i].dst == pendingCopies[j].src &&
                      pendingCopies[j].dst == pendingCopies[i].src) {
                    if (pendingCopies[i].isSGPR && pendingCopies[j].isSGPR) {
                      int64_t regA = pendingCopies[i].dst;
                      int64_t regB = pendingCopies[j].dst;
                      int64_t tmp = peakSGPRs;
                      peakSGPRs = std::max(peakSGPRs, tmp + 1);
                      os << "  s_mov_b32 s" << tmp << ", s" << regA << "\n";
                      os << "  s_mov_b32 s" << regA << ", s" << regB << "\n";
                      os << "  s_mov_b32 s" << regB << ", s" << tmp << "\n";
                      handled[i] = true;
                      handled[j] = true;
                      break;
                    }
                    assert(!((!pendingCopies[i].isSGPR) &&
                             (!pendingCopies[j].isSGPR)) &&
                           "VGPR swap cycles in iter_args are not supported; "
                           "extend swap emission to handle VGPRs");
                  }
                }
              }

              for (size_t i = 0; i < pendingCopies.size(); ++i) {
                if (handled[i])
                  continue;
                const auto &copy = pendingCopies[i];
                if (copy.isSGPR) {
                  os << "  s_mov_b32 s" << copy.dst << ", s" << copy.src
                     << "\n";
                } else {
                  os << "  v_mov_b32 v" << copy.dst << ", v" << copy.src
                     << "\n";
                }
              }
            }

            os << "  s_cbranch_scc1 " << labelName;
            break;
          }

          auto instrLines = generateOpWithLiteralHandling(&bodyOp);
          for (const auto &line : instrLines) {
            os << line << "\n";
          }
        }

        return os.str();
      })
      .Case<IfOp>([&](IfOp ifOp) -> std::optional<std::string> {
        int labelId = loopLabelCounter++;
        std::string elseLabel = "L_if_else_" + std::to_string(labelId);
        std::string endLabel = "L_if_end_" + std::to_string(labelId);

        std::string buf;
        llvm::raw_string_ostream os(buf);

        if (ifOp.hasElse()) {
          os << "  s_cbranch_scc0 " << elseLabel << "\n";
        } else {
          os << "  s_cbranch_scc0 " << endLabel << "\n";
        }

        for (Operation &thenOp : ifOp.getThenBlock()) {
          if (isa<YieldOp>(&thenOp))
            continue;
          auto instrLines = generateOpWithLiteralHandling(&thenOp);
          for (const auto &line : instrLines) {
            os << line << "\n";
          }
        }

        if (ifOp.hasElse()) {
          os << "  s_branch " << endLabel << "\n";
          os << elseLabel << ":\n";
          for (Operation &elseOp : *ifOp.getElseBlock()) {
            if (isa<YieldOp>(&elseOp))
              continue;
            auto instrLines = generateOpWithLiteralHandling(&elseOp);
            for (const auto &line : instrLines) {
              os << line << "\n";
            }
          }
        }

        os << endLabel << ":";
        return os.str();
      })
      .Case<ConditionOp>([&](ConditionOp) -> std::optional<std::string> {
        return std::nullopt;
      })
      .Case<YieldOp>(
          [&](YieldOp) -> std::optional<std::string> { return std::nullopt; })
      .Case<S_CMP_LT_U32, S_CMP_EQ_U32, S_CMP_LE_U32, S_CMP_GT_U32,
            S_CMP_GE_U32, S_CMP_LT_I32, S_CMP_EQ_I32, S_CMP_LE_I32,
            S_CMP_GT_I32, S_CMP_GE_I32>(
          [&](auto cmpOp) -> std::optional<std::string> {
            llvm::StringRef opName = cmpOp->getName().getStringRef();
            llvm::StringRef mnemonic = opName;
            if (opName.starts_with("waveasm.")) {
              mnemonic = opName.drop_front(8);
            }
            llvm::SmallVector<std::string> operands;
            for (Value operand : cmpOp->getOperands()) {
              operands.push_back(resolveValue(operand));
            }
            return formatter.format(mnemonic, operands);
          })

      .Default([&](Operation *defaultOp) -> std::optional<std::string> {
        llvm::StringRef opName = defaultOp->getName().getStringRef();
        llvm::StringRef mnemonic = opName;
        if (opName.starts_with("waveasm.")) {
          mnemonic = opName.drop_front(8);
        }

        if (mnemonic.starts_with("v_cmp_")) {
          std::string mnem64 = (mnemonic + "_e64").str();
          llvm::SmallVector<std::string> operands;
          operands.push_back("vcc");
          for (Value operand : defaultOp->getOperands()) {
            operands.push_back(resolveValue(operand));
          }
          return formatter.format(mnem64, operands);
        }

        return emitDefaultFormat(defaultOp, mnemonic);
      });
}

std::string KernelGenerator::generateLabel(LabelOp labelOp) {
  return formatter.formatLabel(labelOp.getName());
}

std::string KernelGenerator::generateComment(CommentOp commentOp) {
  return formatter.formatComment(commentOp.getText());
}

std::string KernelGenerator::generateRaw(RawOp rawOp) {
  return formatter.formatRaw(rawOp.getText());
}

llvm::SmallVector<std::string> KernelGenerator::generate() {
  llvm::SmallVector<std::string> lines;

  MetadataEmitter metaEmitter(program, target);
  auto prologue = metaEmitter.emitPrologue();
  lines.append(prologue.begin(), prologue.end());

  peakVGPRs = 0;
  peakSGPRs = 0;
  program.walk([&](Operation *preOp) {
    for (Value result : preOp->getResults()) {
      Type ty = result.getType();
      if (auto pvreg = dyn_cast<PVRegType>(ty)) {
        peakVGPRs = std::max(peakVGPRs, pvreg.getIndex() + pvreg.getSize());
      } else if (auto psreg = dyn_cast<PSRegType>(ty)) {
        peakSGPRs = std::max(peakSGPRs, psreg.getIndex() + psreg.getSize());
      } else if (isVirtualRegType(ty)) {
        int64_t size = getRegSize(ty);
        int64_t physIdx = mapping.getPhysReg(result);
        if (physIdx >= 0) {
          if (isVGPRType(ty))
            peakVGPRs = std::max(peakVGPRs, physIdx + size);
          else if (isSGPRType(ty))
            peakSGPRs = std::max(peakSGPRs, physIdx + size);
        }
      }
    }
    for (Value operand : preOp->getOperands()) {
      Type ty = operand.getType();
      if (auto pvreg = dyn_cast<PVRegType>(ty))
        peakVGPRs = std::max(peakVGPRs, pvreg.getIndex() + pvreg.getSize());
      else if (auto psreg = dyn_cast<PSRegType>(ty))
        peakSGPRs = std::max(peakSGPRs, psreg.getIndex() + psreg.getSize());
    }
  });
  peakVGPRs = std::max(peakVGPRs, int64_t(1));
  peakSGPRs = std::max(peakSGPRs, int64_t(2));

  for (Operation &op : program.getBodyBlock()) {
    if (auto labelOp = dyn_cast<LabelOp>(op)) {
      lines.push_back(generateLabel(labelOp));
      continue;
    }
    if (auto commentOp = dyn_cast<CommentOp>(op)) {
      lines.push_back(generateComment(commentOp));
      continue;
    }
    if (auto rawOp = dyn_cast<RawOp>(op)) {
      lines.push_back(generateRaw(rawOp));
      continue;
    }

    // Generate instruction (with literal handling for VALU ops)
    auto instrLines = generateOpWithLiteralHandling(&op);
    lines.append(instrLines.begin(), instrLines.end());
  }

  int64_t ldsSize = program.getLdsSize().value_or(0);
  auto epilogue = metaEmitter.emitEpilogue(peakVGPRs, peakSGPRs, ldsSize);
  lines.append(epilogue.begin(), epilogue.end());

  return lines;
}

//===----------------------------------------------------------------------===//
// Assembly Output Functions
//===----------------------------------------------------------------------===//

LogicalResult writeAssembly(ProgramOp program, const PhysicalMapping &mapping,
                            llvm::StringRef outputPath) {
  std::error_code ec;
  llvm::raw_fd_ostream os(outputPath, ec);
  if (ec) {
    return program.emitError() << "Failed to open output file: " << outputPath;
  }

  return writeAssembly(program, mapping, os);
}

LogicalResult writeAssembly(ProgramOp program, const PhysicalMapping &mapping,
                            llvm::raw_ostream &os) {
  auto targetAttr = program.getTarget();
  if (!targetAttr) {
    return program.emitError() << "target attribute not specified";
  }

  KernelGenerator generator(program, mapping, targetAttr.getTargetKind());
  auto lines = generator.generate();

  for (const auto &line : lines) {
    os << line << "\n";
  }

  return success();
}

} // namespace waveasm
