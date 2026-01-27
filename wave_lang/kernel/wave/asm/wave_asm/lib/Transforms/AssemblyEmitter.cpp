// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Transforms/AssemblyEmitter.h"
#include "waveasm/Transforms/RegAlloc.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Target/AMDGCN/AMDGCNTarget.h"
#include "waveasm/Target/AMDGCN/RegisterInfo.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>

using namespace mlir;

namespace waveasm {

//===----------------------------------------------------------------------===//
// Instruction Formatter Implementation
//===----------------------------------------------------------------------===//

std::string InstructionFormatter::format(llvm::StringRef name,
                                          llvm::ArrayRef<std::string> operands) {
  std::string result = "  " + name.str();

  if (!operands.empty()) {
    result += " ";
    for (size_t i = 0; i < operands.size(); ++i) {
      if (i > 0)
        result += ", ";
      result += operands[i];
    }
  }

  return result;
}

std::string InstructionFormatter::formatLabel(llvm::StringRef label) {
  return label.str() + ":";
}

std::string InstructionFormatter::formatComment(llvm::StringRef text) {
  return "  ; " + text.str();
}

std::string InstructionFormatter::formatWaitcnt(std::optional<int64_t> vmcnt,
                                                 std::optional<int64_t> lgkmcnt,
                                                 std::optional<int64_t> expcnt) {
  std::string result = "  s_waitcnt";
  llvm::SmallVector<std::string> counts;

  if (vmcnt.has_value()) {
    counts.push_back("vmcnt(" + std::to_string(*vmcnt) + ")");
  }
  if (lgkmcnt.has_value()) {
    counts.push_back("lgkmcnt(" + std::to_string(*lgkmcnt) + ")");
  }
  if (expcnt.has_value()) {
    counts.push_back("expcnt(" + std::to_string(*expcnt) + ")");
  }

  if (!counts.empty()) {
    result += " ";
    for (size_t i = 0; i < counts.size(); ++i) {
      if (i > 0)
        result += " & ";
      result += counts[i];
    }
  }

  return result;
}

std::string InstructionFormatter::formatBarrier() {
  return "  s_barrier";
}

std::string InstructionFormatter::formatEndpgm() {
  return "  s_endpgm";
}

std::string InstructionFormatter::formatRaw(llvm::StringRef text) {
  return "  " + text.str();
}

//===----------------------------------------------------------------------===//
// Metadata Emitter Implementation
//===----------------------------------------------------------------------===//

MetadataEmitter::MetadataEmitter(ProgramOp program, const AMDGCNTarget &target)
    : program(program), target(target) {}

llvm::SmallVector<std::string> MetadataEmitter::emitPrologue() {
  llvm::SmallVector<std::string> lines;

  // Target directive
  lines.push_back(target.getTargetDirective());
  lines.push_back("");

  // Text section
  lines.push_back(".text");
  lines.push_back("");

  // Kernel symbol
  std::string symName = program.getSymName().str();
  lines.push_back(".protected " + symName);
  lines.push_back(".globl " + symName);
  lines.push_back(".p2align 8");
  lines.push_back(".type " + symName + ",@function");
  lines.push_back(symName + ":");

  return lines;
}

llvm::SmallVector<std::string>
MetadataEmitter::emitEpilogue(int64_t peakVGPRs, int64_t peakSGPRs,
                               int64_t ldsSize) {
  llvm::SmallVector<std::string> lines;

  // Kernel descriptor
  auto descriptor = emitKernelDescriptor(peakVGPRs, peakSGPRs, ldsSize);
  lines.append(descriptor.begin(), descriptor.end());

  lines.push_back("");

  // Metadata YAML
  auto metadata = emitMetadataYAML(peakVGPRs, peakSGPRs, ldsSize);
  lines.append(metadata.begin(), metadata.end());

  return lines;
}

/// Scan program operations to detect system register usage
static void scanSystemRegisterUsage(ProgramOp program,
                                     bool &usesWorkgroupIdX,
                                     bool &usesWorkgroupIdY,
                                     bool &usesWorkgroupIdZ,
                                     bool &usesWorkitemId) {
  usesWorkgroupIdX = false;
  usesWorkgroupIdY = false;
  usesWorkgroupIdZ = false;
  usesWorkitemId = false;

  // Compute user SGPR count to determine where workgroup IDs are located
  // User SGPRs = 2 (kernarg ptr) + preloaded args (on gfx950+)
  auto targetAttr = program.getTarget();
  llvm::StringRef targetId = targetAttr ? targetAttr.getId() : "";
  bool isGfx950 = targetId.starts_with("gfx95");

  int64_t numArgs = 2;  // Default to 2 pointers
  if (auto numArgsAttr = program->getAttrOfType<IntegerAttr>("num_kernel_args")) {
    numArgs = numArgsAttr.getInt();
  }

  int64_t userSgprCount = 2;  // Base: kernarg ptr
  if (isGfx950) {
    userSgprCount = 2 + numArgs * 2;  // kernarg ptr + preloaded args
  }

  // Workgroup IDs are at system SGPR positions (after user SGPRs)
  int64_t wgIdXIndex = userSgprCount;
  int64_t wgIdYIndex = userSgprCount + 1;
  int64_t wgIdZIndex = userSgprCount + 2;

  // Scan for system register usage by looking at precolored SGPRs.
  // Workgroup IDs are passed in single SGPRs (size=1) at indices after user SGPRs.
  // SRDs use 4-SGPR quads (size=4), so we filter those out.
  //
  // Note: workitem_id is NOT set when v_mbcnt is used, because wave kernels
  // compute thread IDs from the exec mask, not from hardware-provided v0.
  program.walk([&](Operation *op) {
    if (auto precolored = dyn_cast<PrecoloredSRegOp>(op)) {
      int64_t idx = precolored.getIndex();
      int64_t size = precolored.getSize();
      // Only single SGPRs (size=1) are workgroup IDs, not SRD quads (size=4)
      if (size == 1) {
        if (idx == wgIdXIndex) usesWorkgroupIdX = true;
        else if (idx == wgIdYIndex) usesWorkgroupIdY = true;
        else if (idx == wgIdZIndex) usesWorkgroupIdZ = true;
      }
    }
  });
}

llvm::SmallVector<std::string>
MetadataEmitter::emitKernelDescriptor(int64_t peakVGPRs, int64_t peakSGPRs,
                                       int64_t ldsSize) {
  llvm::SmallVector<std::string> lines;
  std::string symName = program.getSymName().str();

  // Scan for system register usage
  bool usesWorkgroupIdX, usesWorkgroupIdY, usesWorkgroupIdZ, usesWorkitemId;
  scanSystemRegisterUsage(program, usesWorkgroupIdX, usesWorkgroupIdY,
                          usesWorkgroupIdZ, usesWorkitemId);

  lines.push_back("");
  lines.push_back(".section .rodata,#alloc");
  lines.push_back(".p2align 6");
  lines.push_back(".amdhsa_kernel " + symName);

  // Compute LDS block size (granularity of 128 bytes)
  int64_t ldsBlocks = (ldsSize + 127) / 128;
  lines.push_back("  .amdhsa_group_segment_fixed_size " +
                  std::to_string(ldsBlocks * 128));
  lines.push_back("  .amdhsa_private_segment_fixed_size 0");

  // User SGPR settings - match Python backend format
  // For gfx95* with preloading: user_sgpr_count = 2 (kernarg ptr) + 4 (preload) = 6
  // For other targets: user_sgpr_count = 2 (just kernarg ptr)
  auto targetAttr = program.getTarget();
  llvm::StringRef targetId = targetAttr ? targetAttr.getId() : "";

  // Determine preload length based on actual kernel args
  int64_t preloadLength = program.getKernargPreloadLength();
  bool usePreloading = targetId.starts_with("gfx95");

  // If no explicit preload length but we're on gfx95*, use actual kernel arg count
  if (usePreloading && preloadLength == 0) {
    int64_t numArgs = 2; // Default to 2 pointers
    if (auto numArgsAttr = program->getAttrOfType<IntegerAttr>("num_kernel_args")) {
      numArgs = numArgsAttr.getInt();
    }
    preloadLength = numArgs * 2;  // 2 SGPRs per pointer
  }

  int64_t userSgprCount = 2;  // Base: kernarg segment pointer
  if (usePreloading && preloadLength > 0) {
    userSgprCount = 2 + preloadLength;  // kernarg ptr + preloaded args
  }

  lines.push_back("  .amdhsa_user_sgpr_count " + std::to_string(userSgprCount));
  lines.push_back("  .amdhsa_user_sgpr_dispatch_ptr 0");
  lines.push_back("  .amdhsa_user_sgpr_queue_ptr 0");
  lines.push_back("  .amdhsa_user_sgpr_kernarg_segment_ptr 1");
  lines.push_back("  .amdhsa_user_sgpr_dispatch_id 0");

  // Kernel argument preloading (gfx95*)
  // Preloads kernel args directly into SGPRs, saving ~100 cycles at kernel start
  if (usePreloading && preloadLength > 0) {
    lines.push_back("  .amdhsa_user_sgpr_kernarg_preload_length " +
                    std::to_string(preloadLength));
    lines.push_back("  .amdhsa_user_sgpr_kernarg_preload_offset 0");
  }

  lines.push_back("  .amdhsa_user_sgpr_private_segment_size 0");

  // Stack and private segment settings
  lines.push_back("  .amdhsa_uses_dynamic_stack 0");
  lines.push_back("  .amdhsa_enable_private_segment 0");

  // Compute VGPR and SGPR counts
  // gfx940/gfx942/gfx950: VGPRs allocated in groups of 8
  // other GFX9: VGPRs allocated in groups of 4
  int64_t vgprGranularity = 4;
  if (targetId.contains("gfx940") || targetId.contains("gfx942") ||
      targetId.contains("gfx950")) {
    vgprGranularity = 8;
  }
  int64_t nextFreeVGPR = ((peakVGPRs + vgprGranularity - 1) / vgprGranularity) *
                          vgprGranularity;

  int64_t sgprGranularity = 8;
  int64_t nextFreeSGPR = ((peakSGPRs + sgprGranularity - 1) / sgprGranularity) *
                          sgprGranularity;
  // Cap to max for target (targetAttr/targetId already declared above)
  if (targetId.starts_with("gfx9")) {
    nextFreeSGPR = std::min(nextFreeSGPR, int64_t(102));
  }

  // Accumulator offset (required for gfx9 targets) - must come before next_free_vgpr
  if (targetId.starts_with("gfx9")) {
    // accum_offset must be in range [4, 256] and multiple of 4
    int64_t accumOffset = std::max(int64_t(4), ((nextFreeVGPR + 3) / 4) * 4);
    lines.push_back("  .amdhsa_accum_offset " + std::to_string(accumOffset));
  }

  // Register counts
  lines.push_back("  .amdhsa_next_free_vgpr " + std::to_string(nextFreeVGPR));
  lines.push_back("  .amdhsa_next_free_sgpr " + std::to_string(nextFreeSGPR));

  // System SGPR settings - enable workgroup IDs based on kernel usage
  lines.push_back("  .amdhsa_system_sgpr_workgroup_id_x " +
                  std::to_string(usesWorkgroupIdX ? 1 : 0));
  lines.push_back("  .amdhsa_system_sgpr_workgroup_id_y " +
                  std::to_string(usesWorkgroupIdY ? 1 : 0));
  lines.push_back("  .amdhsa_system_sgpr_workgroup_id_z " +
                  std::to_string(usesWorkgroupIdZ ? 1 : 0));

  // System VGPR settings - workitem ID dimension
  // Policy (matching Python abi.py):
  // - Single-wave workgroup (wg_y==1 and wg_z==1): request 0 (lane ID via mbcnt)
  // - Multi-wave workgroup (wg_y>1 or wg_z>1): request 1 (flat workitem ID in v0)
  int64_t systemVgprWorkitemId = 0;
  auto workgroupSize = program.getWorkgroupSize();
  if (workgroupSize.has_value() && workgroupSize->size() >= 2) {
    int64_t wgY = 1, wgZ = 1;
    if (auto intAttr = dyn_cast<IntegerAttr>((*workgroupSize)[1])) {
      wgY = intAttr.getInt();
    }
    if (workgroupSize->size() >= 3) {
      if (auto intAttr = dyn_cast<IntegerAttr>((*workgroupSize)[2])) {
        wgZ = intAttr.getInt();
      }
    }
    if (wgY > 1 || wgZ > 1) {
      systemVgprWorkitemId = 1;
    }
  }
  lines.push_back("  .amdhsa_system_vgpr_workitem_id " +
                  std::to_string(systemVgprWorkitemId));

  // Denorm mode (matching Python)
  lines.push_back("  .amdhsa_float_denorm_mode_32 3");
  lines.push_back("  .amdhsa_float_denorm_mode_16_64 3");

  lines.push_back(".end_amdhsa_kernel");

  return lines;
}

llvm::SmallVector<std::string>
MetadataEmitter::emitMetadataYAML(int64_t peakVGPRs, int64_t peakSGPRs,
                                   int64_t ldsSize) {
  llvm::SmallVector<std::string> lines;
  std::string symName = program.getSymName().str();

  lines.push_back(".amdgpu_metadata");
  lines.push_back("---");
  lines.push_back("amdhsa.version:");
  lines.push_back("  - 1");
  lines.push_back("  - 2");
  lines.push_back("amdhsa.kernels:");
  lines.push_back("  - .name: " + symName);
  lines.push_back("    .symbol: " + symName + ".kd");

  // Generate .args based on kernel arguments (each 8-byte pointer)
  // First, try to get the actual number of kernel args from the attribute
  int64_t numArgs = 2; // Default fallback
  if (auto numArgsAttr = program->getAttrOfType<IntegerAttr>("num_kernel_args")) {
    numArgs = numArgsAttr.getInt();
  } else {
    // Legacy: compute from kernarg preload length (gfx950+)
    int64_t kernargPreload = program.getKernargPreloadLength();
    if (kernargPreload > 0) {
      numArgs = kernargPreload / 2; // 2 SGPRs per pointer
    }
  }
  int64_t kernargSize = numArgs * 8; // 8 bytes per pointer arg
  lines.push_back("    .args:");
  for (int64_t i = 0; i < numArgs; ++i) {
    lines.push_back("      - .name:       arg" + std::to_string(i) + "_ptr");
    lines.push_back("        .offset:     " + std::to_string(i * 8));
    lines.push_back("        .size:       8");
    lines.push_back("        .value_kind: global_buffer");
    lines.push_back("        .value_type: 'i8*'");
  }

  lines.push_back("    .kernarg_segment_size: " + std::to_string(kernargSize));
  lines.push_back("    .group_segment_fixed_size: " + std::to_string(ldsSize));
  lines.push_back("    .private_segment_fixed_size: 0");
  lines.push_back("    .kernarg_segment_align: 8");

  // Wave size
  WaveSize waveSize = target.getDefaultWaveSize();
  int64_t waveSizeVal = static_cast<int64_t>(waveSize);
  lines.push_back("    .wavefront_size: " + std::to_string(waveSizeVal));

  // VGPR/SGPR counts
  lines.push_back("    .sgpr_count: " + std::to_string(peakSGPRs));
  lines.push_back("    .vgpr_count: " + std::to_string(peakVGPRs));

  // Max flat workgroup size
  auto workgroupSize = program.getWorkgroupSize();
  int64_t maxFlatSize = 256; // Default
  if (workgroupSize.has_value()) {
    maxFlatSize = 1;
    for (Attribute attr : *workgroupSize) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        maxFlatSize *= intAttr.getInt();
      }
    }
  }
  lines.push_back("    .max_flat_workgroup_size: " +
                  std::to_string(maxFlatSize));

  lines.push_back("...");
  lines.push_back(".end_amdgpu_metadata");

  return lines;
}

//===----------------------------------------------------------------------===//
// Kernel Generator Implementation (Pure SSA)
//===----------------------------------------------------------------------===//

KernelGenerator::KernelGenerator(ProgramOp program,
                                  const PhysicalMapping &mapping,
                                  const AMDGCNTarget &target)
    : program(program), mapping(mapping), target(target) {}

// Check if a literal value is within the inline constant range
// AMDGCN inline constants: -16 to 64 (inclusive), plus special values
static bool isInlineConstant(int64_t val) {
  // Standard inline constant range
  if (val >= -16 && val <= 64)
    return true;
  // Special float values encoded as integers (0.5, 1.0, 2.0, 4.0, etc.)
  // For simplicity, we just check the common integer range
  return false;
}

std::string KernelGenerator::resolveValue(Value value) {
  Type ty = value.getType();

  // Handle physical registers (already assigned)
  if (auto pvreg = dyn_cast<PVRegType>(ty)) {
    return formatVGPRRange(pvreg.getIndex(), pvreg.getSize());
  }
  if (auto psreg = dyn_cast<PSRegType>(ty)) {
    return formatSGPRRange(psreg.getIndex(), psreg.getSize());
  }

  // Handle virtual registers (look up in mapping)
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
    // Fallback: use SSA value notation (for debugging)
    return "%<ssa>";
  }

  // Handle immediates
  if (isa<ImmType>(ty)) {
    // For immediates, we need to get the value from the defining op
    if (auto defOp = value.getDefiningOp()) {
      if (auto constOp = dyn_cast<ConstantOp>(defOp)) {
        // Cast to signed int64_t to handle negative values correctly
        int64_t val = static_cast<int64_t>(constOp.getValue());
        return std::to_string(val);
      }
    }
    return "0";
  }

  return "<unknown>";
}

// Check if an operand value is a literal outside inline constant range
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

std::optional<std::string> KernelGenerator::generateOp(Operation *op) {
  // Get the operation name and extract the instruction mnemonic
  llvm::StringRef opName = op->getName().getStringRef();

  // Skip non-instruction ops
  if (opName == "waveasm.program" || opName == "waveasm.label" ||
      opName == "waveasm.comment" || opName == "waveasm.raw" ||
      opName == "waveasm.precolored.vreg" || opName == "waveasm.precolored.sreg" ||
      opName == "waveasm.constant" || opName == "waveasm.pack" ||
      opName == "waveasm.extract") {
    return std::nullopt;
  }

  // Extract instruction mnemonic from op name (e.g., "waveasm.v_add_u32" -> "v_add_u32")
  llvm::StringRef mnemonic = opName;
  if (opName.starts_with("waveasm.")) {
    mnemonic = opName.drop_front(8);  // "waveasm." is 8 characters
  }

  // Handle S_WAITCNT specially - uses attributes not operands
  if (auto waitcntOp = dyn_cast<S_WAITCNT>(op)) {
    std::optional<int64_t> vmcnt, lgkmcnt, expcnt;
    if (auto vmAttr = waitcntOp.getVmcntAttr())
      vmcnt = vmAttr.getInt();
    if (auto lgkmAttr = waitcntOp.getLgkmcntAttr())
      lgkmcnt = lgkmAttr.getInt();
    if (auto expAttr = waitcntOp.getExpcntAttr())
      expcnt = expAttr.getInt();
    // If no counts specified, default to vmcnt(0) lgkmcnt(0)
    if (!vmcnt && !lgkmcnt && !expcnt) {
      vmcnt = 0;
      lgkmcnt = 0;
    }
    return formatter.formatWaitcnt(vmcnt, lgkmcnt, expcnt);
  }

  // Handle S_WAITCNT_VMCNT specially
  if (auto waitcntOp = dyn_cast<S_WAITCNT_VMCNT>(op)) {
    return formatter.formatWaitcnt(waitcntOp.getCount(), std::nullopt, std::nullopt);
  }

  // Handle S_WAITCNT_LGKMCNT specially
  if (auto waitcntOp = dyn_cast<S_WAITCNT_LGKMCNT>(op)) {
    return formatter.formatWaitcnt(std::nullopt, waitcntOp.getCount(), std::nullopt);
  }

  // Handle buffer_load instructions (not LDS variant)
  // Format: buffer_load_dwordx4 vdata, voffset, srd, soffset offen [offset:N]
  // The "offen" modifier is required when using a VGPR for the offset
  if (mnemonic.starts_with("buffer_load") && !mnemonic.contains("lds")) {
    std::string result = "  " + mnemonic.str();
    // Result (vdata) first
    std::string vdata;
    for (Value res : op->getResults()) {
      vdata = resolveValue(res);
    }
    // Op order is (srd, voffset) but assembly order is (vdata, voffset, srd, soffset offen)
    if (op->getNumOperands() >= 2) {
      std::string voffset = resolveValue(op->getOperand(1));
      std::string srd = resolveValue(op->getOperand(0));
      result += " " + vdata + ", " + voffset + ", " + srd + ", 0 offen";
      // Check for instruction offset attribute
      if (auto instOffsetAttr = op->getAttrOfType<IntegerAttr>("instOffset")) {
        int64_t offset = instOffsetAttr.getInt();
        if (offset > 0) {
          result += " offset:" + std::to_string(offset);
        }
      }
    }
    return result;
  }

  // Handle buffer_load_*_lds instructions (gather-to-LDS)
  // Format: buffer_load_dword voffset, srd, soffset offen lds
  // Note: mnemonic has _lds suffix, need to emit without _lds but with lds modifier
  if (mnemonic.starts_with("buffer_load") && mnemonic.contains("lds")) {
    // Remove _lds suffix from mnemonic
    std::string baseMnemonic = mnemonic.str();
    size_t ldsPos = baseMnemonic.find("_lds");
    if (ldsPos != std::string::npos) {
      baseMnemonic = baseMnemonic.substr(0, ldsPos);
    }
    std::string result = "  " + baseMnemonic;
    // VMEMToLDSLoadOp: (voffset, srd, soffset)
    if (op->getNumOperands() >= 3) {
      std::string voffset = resolveValue(op->getOperand(0));
      std::string srd = resolveValue(op->getOperand(1));
      std::string soffset = resolveValue(op->getOperand(2));
      result += " " + voffset + ", " + srd + ", " + soffset + " offen lds";
    } else if (op->getNumOperands() >= 2) {
      std::string voffset = resolveValue(op->getOperand(0));
      std::string srd = resolveValue(op->getOperand(1));
      result += " " + voffset + ", " + srd + ", 0 offen lds";
    }
    return result;
  }

  // Handle buffer_store instructions
  // Format: buffer_store_dwordx4 vdata, voffset, srd, soffset offen [offset:N]
  if (mnemonic.starts_with("buffer_store")) {
    std::string result = "  " + mnemonic.str();
    // Op order is (data, srd, voffset) but assembly order is (vdata, voffset, srd, soffset offen)
    if (op->getNumOperands() >= 3) {
      std::string vdata = resolveValue(op->getOperand(0));
      std::string voffset = resolveValue(op->getOperand(2));
      std::string srd = resolveValue(op->getOperand(1));
      result += " " + vdata + ", " + voffset + ", " + srd + ", 0 offen";
      // Check for instruction offset attribute
      if (auto instOffsetAttr = op->getAttrOfType<IntegerAttr>("instOffset")) {
        int64_t offset = instOffsetAttr.getInt();
        if (offset > 0) {
          result += " offset:" + std::to_string(offset);
        }
      }
    }
    return result;
  }

  // Handle global_load instructions
  // Format: global_load_dwordx4 vdata, vaddr, off
  // vaddr is a VGPR pair for 64-bit address, off is scalar offset (usually "off" for no offset)
  if (mnemonic.starts_with("global_load")) {
    std::string result = "  " + mnemonic.str();
    std::string vdata;
    for (Value res : op->getResults()) {
      vdata = resolveValue(res);
    }
    // Op order is (saddr, voffset) - saddr is SGPR pair, voffset is VGPR
    if (op->getNumOperands() >= 2) {
      std::string vaddr = resolveValue(op->getOperand(1));
      std::string saddr = resolveValue(op->getOperand(0));
      // Use saddr as scalar offset, vaddr as vector address
      result += " " + vdata + ", " + vaddr + ", " + saddr;
    } else if (op->getNumOperands() >= 1) {
      std::string vaddr = resolveValue(op->getOperand(0));
      result += " " + vdata + ", " + vaddr + ", off";
    }
    return result;
  }

  // Handle global_store instructions
  // Format: global_store_dwordx4 vaddr, vdata, off
  if (mnemonic.starts_with("global_store")) {
    std::string result = "  " + mnemonic.str();
    // Op order is (data, saddr, voffset)
    if (op->getNumOperands() >= 3) {
      std::string vdata = resolveValue(op->getOperand(0));
      std::string vaddr = resolveValue(op->getOperand(2));
      std::string saddr = resolveValue(op->getOperand(1));
      result += " " + vaddr + ", " + vdata + ", " + saddr;
    }
    return result;
  }

  // Handle flat_load instructions
  // Format: flat_load_dwordx4 vdata, vaddr
  if (mnemonic.starts_with("flat_load")) {
    std::string result = "  " + mnemonic.str();
    std::string vdata;
    for (Value res : op->getResults()) {
      vdata = resolveValue(res);
    }
    if (op->getNumOperands() >= 1) {
      std::string vaddr = resolveValue(op->getOperand(0));
      result += " " + vdata + ", " + vaddr;
    }
    return result;
  }

  // Handle flat_store instructions
  // Format: flat_store_dwordx4 vaddr, vdata
  if (mnemonic.starts_with("flat_store")) {
    std::string result = "  " + mnemonic.str();
    if (op->getNumOperands() >= 2) {
      std::string vdata = resolveValue(op->getOperand(0));
      std::string vaddr = resolveValue(op->getOperand(1));
      result += " " + vaddr + ", " + vdata;
    }
    return result;
  }

  // Handle LDS read operations
  // Format: ds_read_b32 vdst, vaddr [offset:N]
  if (mnemonic.starts_with("ds_read")) {
    std::string result = "  " + mnemonic.str();
    std::string vdst;
    for (Value res : op->getResults()) {
      vdst = resolveValue(res);
    }
    if (op->getNumOperands() >= 1) {
      std::string vaddr = resolveValue(op->getOperand(0));
      result += " " + vdst + ", " + vaddr;
    }
    // Add offset:N if present
    if (auto offsetAttr = op->getAttrOfType<IntegerAttr>("offset")) {
      int64_t offset = offsetAttr.getInt();
      if (offset != 0) {
        result += " offset:" + std::to_string(offset);
      }
    }
    return result;
  }

  // Handle LDS write operations
  // Format: ds_write_b32 vaddr, vdata [offset:N]
  if (mnemonic.starts_with("ds_write")) {
    std::string result = "  " + mnemonic.str();
    if (op->getNumOperands() >= 2) {
      std::string vaddr = resolveValue(op->getOperand(1));
      std::string vdata = resolveValue(op->getOperand(0));
      result += " " + vaddr + ", " + vdata;
    }
    return result;
  }

  // Handle s_mov_b32_m0 - move to M0 special register
  // Format: s_mov_b32 m0, value
  if (mnemonic == "s_mov_b32_m0") {
    std::string result = "  s_mov_b32 m0";
    if (op->getNumOperands() >= 1) {
      result += ", " + resolveValue(op->getOperand(0));
    }
    return result;
  }

  // Handle conditional branches (s_cbranch_scc0, s_cbranch_scc1, etc.)
  // These take a SymbolRefAttr target, not a regular operand
  if (mnemonic.starts_with("s_cbranch_")) {
    std::string result = "  " + mnemonic.str();
    if (auto targetAttr = op->getAttrOfType<SymbolRefAttr>("target")) {
      result += " " + targetAttr.getRootReference().str();
    }
    return result;
  }

  // Handle s_nop - takes a count attribute
  if (mnemonic == "s_nop") {
    std::string result = "  s_nop";
    if (auto countAttr = op->getAttrOfType<IntegerAttr>("count")) {
      result += " " + std::to_string(countAttr.getInt());
    } else {
      result += " 0";  // Default to 0 wait states
    }
    return result;
  }

  // Handle v_mov_b32 with multi-element results (used for accumulator initialization)
  // v_mov_b32 only operates on single 32-bit registers, so we need to emit
  // multiple instructions for multi-element results
  if (mnemonic == "v_mov_b32" && op->getNumResults() == 1) {
    Value result = op->getResult(0);
    int64_t size = getRegSize(result.getType());
    if (size > 1) {
      // Multi-element result - emit multiple v_mov_b32 instructions
      int64_t baseIdx = mapping.getPhysReg(result);
      if (baseIdx < 0) {
        // Try PVRegType
        if (auto pvreg = dyn_cast<PVRegType>(result.getType())) {
          baseIdx = pvreg.getIndex();
        }
      }
      if (baseIdx >= 0 && op->getNumOperands() >= 1) {
        std::string src = resolveValue(op->getOperand(0));
        std::string lines;
        for (int64_t i = 0; i < size; ++i) {
          if (i > 0) lines += "\n";
          lines += "  v_mov_b32 v" + std::to_string(baseIdx + i) + ", " + src;
        }
        return lines;
      }
    }
  }

  // Collect operands (results first, then operands)
  llvm::SmallVector<std::string> operands;

  // Results come first in assembly output
  for (Value result : op->getResults()) {
    operands.push_back(resolveValue(result));
  }

  // Then input operands
  for (Value operand : op->getOperands()) {
    operands.push_back(resolveValue(operand));
  }

  return formatter.format(mnemonic, operands);
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

// Check if an instruction is a VOP3 that doesn't support literal operands
static bool isVOP3NoLiteral(llvm::StringRef mnemonic) {
  // VOP3 instructions that don't support literal operands
  // This includes most v_* integer instructions when not in VOP2/VOP1 form
  return mnemonic.starts_with("v_mul_lo") ||
         mnemonic.starts_with("v_mul_hi") ||
         mnemonic.starts_with("v_mad") ||
         mnemonic.starts_with("v_fma") ||
         mnemonic.starts_with("v_lshl") ||
         mnemonic.starts_with("v_lshr") ||
         mnemonic.starts_with("v_ashr") ||
         mnemonic.starts_with("v_bfe") ||
         mnemonic.starts_with("v_bfi") ||
         mnemonic.starts_with("v_alignbit") ||
         mnemonic.starts_with("v_alignbyte");
}

llvm::SmallVector<std::string>
KernelGenerator::generateOpWithLiteralHandling(Operation *op) {
  llvm::SmallVector<std::string> lines;

  // Get the operation name and extract the instruction mnemonic
  llvm::StringRef opName = op->getName().getStringRef();
  llvm::StringRef mnemonic = opName;
  if (opName.starts_with("waveasm.")) {
    mnemonic = opName.drop_front(8);
  }

  // Check if this is a VOP3 that doesn't support literals
  if (!isVOP3NoLiteral(mnemonic)) {
    // Use normal generation
    if (auto line = generateOp(op)) {
      lines.push_back(*line);
    }
    return lines;
  }

  // Check operands for literals outside inline range
  bool needsLiteralLoad = false;
  int64_t literalValue = 0;
  int literalOperandIdx = -1;

  for (int i = 0; i < static_cast<int>(op->getNumOperands()); ++i) {
    auto [isLiteral, val] = getLiteralValue(op->getOperand(i));
    if (isLiteral && !isInlineConstant(val)) {
      needsLiteralLoad = true;
      literalValue = val;
      literalOperandIdx = i;
      break; // Only handle first non-inline literal
    }
  }

  if (!needsLiteralLoad) {
    // All literals are inline constants, use normal generation
    if (auto line = generateOp(op)) {
      lines.push_back(*line);
    }
    return lines;
  }

  // Emit v_mov_b32 to load the literal into scratch VGPR
  // This matches the Python backend approach in kernel_expr_emitter.py
  std::string scratchReg = formatVGPRRange(kScratchVGPR, 1);
  lines.push_back("  v_mov_b32 " + scratchReg + ", " +
                  std::to_string(literalValue));

  // Now generate the instruction with scratch VGPR instead of literal
  llvm::SmallVector<std::string> operands;

  // Results come first
  for (Value result : op->getResults()) {
    operands.push_back(resolveValue(result));
  }

  // Then input operands, replacing the literal with scratch register
  for (int i = 0; i < static_cast<int>(op->getNumOperands()); ++i) {
    if (i == literalOperandIdx) {
      operands.push_back(scratchReg);
    } else {
      operands.push_back(resolveValue(op->getOperand(i)));
    }
  }

  lines.push_back(formatter.format(mnemonic, operands));

  // Track that we used the scratch VGPR (update peak if needed)
  peakVGPRs = std::max(peakVGPRs, kScratchVGPR + 1);

  return lines;
}

llvm::SmallVector<std::string> KernelGenerator::generate() {
  llvm::SmallVector<std::string> lines;

  // Emit prologue
  MetadataEmitter metaEmitter(program, target);
  auto prologue = metaEmitter.emitPrologue();
  lines.append(prologue.begin(), prologue.end());

  // Calculate peak register usage from mapping
  peakVGPRs = 0;
  peakSGPRs = 0;

  // Generate code for each operation
  for (Operation &op : program.getBodyBlock()) {
    // Handle special ops first
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

    // Generate instruction (with literal handling for VOP3 ops)
    auto instrLines = generateOpWithLiteralHandling(&op);
    lines.append(instrLines.begin(), instrLines.end());

    // Track register usage for VGPR/SGPR accounting
    for (Value result : op.getResults()) {
      Type ty = result.getType();
      int64_t size = getRegSize(ty);
      int64_t physIdx = mapping.getPhysReg(result);

      if (physIdx >= 0) {
        if (isVGPRType(ty)) {
          peakVGPRs = std::max(peakVGPRs, physIdx + size);
        } else if (isSGPRType(ty)) {
          peakSGPRs = std::max(peakSGPRs, physIdx + size);
        }
      }

      // Handle already-physical registers
      if (auto pvreg = dyn_cast<PVRegType>(ty)) {
        peakVGPRs = std::max(peakVGPRs, pvreg.getIndex() + pvreg.getSize());
      } else if (auto psreg = dyn_cast<PSRegType>(ty)) {
        peakSGPRs = std::max(peakSGPRs, psreg.getIndex() + psreg.getSize());
      }
    }
  }

  // Ensure minimums
  peakVGPRs = std::max(peakVGPRs, int64_t(1));
  peakSGPRs = std::max(peakSGPRs, int64_t(2)); // Kernarg pointer

  // Emit epilogue
  int64_t ldsSize = program.getLdsSize().value_or(0);
  auto epilogue = metaEmitter.emitEpilogue(peakVGPRs, peakSGPRs, ldsSize);
  lines.append(epilogue.begin(), epilogue.end());

  return lines;
}

//===----------------------------------------------------------------------===//
// Assembly Output Functions
//===----------------------------------------------------------------------===//

LogicalResult writeAssembly(ProgramOp program,
                             const PhysicalMapping &mapping,
                             llvm::StringRef outputPath) {
  std::error_code ec;
  llvm::raw_fd_ostream os(outputPath, ec);
  if (ec) {
    return program.emitError() << "Failed to open output file: " << outputPath;
  }

  return writeAssembly(program, mapping, os);
}

LogicalResult writeAssembly(ProgramOp program,
                             const PhysicalMapping &mapping,
                             llvm::raw_ostream &os) {
  // Get target
  auto targetAttr = program.getTarget();
  auto target = getAMDGCNTarget(targetAttr.getId());
  if (!target) {
    return program.emitError() << "Unknown target: " << targetAttr.getId();
  }

  // Generate assembly
  KernelGenerator generator(program, mapping, *target);
  auto lines = generator.generate();

  // Write to stream
  for (const auto &line : lines) {
    os << line << "\n";
  }

  return success();
}

} // namespace waveasm
