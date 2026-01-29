// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_TRANSFORMS_ASSEMBLYEMITTER_H
#define WaveASM_TRANSFORMS_ASSEMBLYEMITTER_H

#include "waveasm/Transforms/RegAlloc.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Target/AMDGCN/AMDGCNTarget.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

namespace waveasm {

//===----------------------------------------------------------------------===//
// Instruction Formatter
//===----------------------------------------------------------------------===//

/// Formats instructions for assembly output
class InstructionFormatter {
public:
  InstructionFormatter() = default;

  /// Format an instruction with operands
  std::string format(llvm::StringRef name, llvm::ArrayRef<std::string> operands);

  /// Format a label
  std::string formatLabel(llvm::StringRef label);

  /// Format a comment
  std::string formatComment(llvm::StringRef text);

  /// Format s_waitcnt instruction
  std::string formatWaitcnt(std::optional<int64_t> vmcnt,
                            std::optional<int64_t> lgkmcnt,
                            std::optional<int64_t> expcnt = std::nullopt);

  /// Format s_barrier instruction
  std::string formatBarrier();

  /// Format s_endpgm instruction
  std::string formatEndpgm();

  /// Format a raw assembly line (passthrough)
  std::string formatRaw(llvm::StringRef text);
};

//===----------------------------------------------------------------------===//
// Kernel Metadata Emitter
//===----------------------------------------------------------------------===//

/// Emits kernel metadata for AMDGPU
class MetadataEmitter {
public:
  MetadataEmitter(ProgramOp program, TargetAttrInterface target);

  /// Emit assembly prologue (target directive, .text, kernel directive)
  llvm::SmallVector<std::string> emitPrologue();

  /// Emit assembly epilogue (kernel symbol info, metadata YAML)
  llvm::SmallVector<std::string> emitEpilogue(int64_t peakVGPRs,
                                               int64_t peakSGPRs,
                                               int64_t ldsSize);

private:
  /// Emit .amdhsa_kernel directive and its fields
  llvm::SmallVector<std::string> emitKernelDescriptor(int64_t peakVGPRs,
                                                       int64_t peakSGPRs,
                                                       int64_t ldsSize);

  /// Emit .amdgpu_metadata section
  llvm::SmallVector<std::string> emitMetadataYAML(int64_t peakVGPRs,
                                                   int64_t peakSGPRs,
                                                   int64_t ldsSize);

  ProgramOp program;
  TargetAttrInterface target;
};

//===----------------------------------------------------------------------===//
// Kernel Generator (Pure SSA)
//===----------------------------------------------------------------------===//

/// Generates assembly code for a kernel with pure SSA ops
class KernelGenerator {
public:
  KernelGenerator(ProgramOp program, const PhysicalMapping &mapping,
                  TargetAttrInterface target);

  /// Generate complete assembly output
  llvm::SmallVector<std::string> generate();

  /// Get peak VGPR usage from allocation
  int64_t getPeakVGPRs() const { return peakVGPRs; }

  /// Get peak SGPR usage from allocation
  int64_t getPeakSGPRs() const { return peakSGPRs; }

private:
  /// Resolve an SSA Value to its physical register string
  std::string resolveValue(mlir::Value value);

  /// Get the literal value if the operand is an immediate constant
  /// Returns (isLiteral, value) pair
  std::pair<bool, int64_t> getLiteralValue(mlir::Value value);

  /// Generate code for a single operation
  /// Returns (instruction_lines, needs_literal_load)
  std::optional<std::string> generateOp(mlir::Operation *op);

  /// Generate instructions with literal handling
  /// For VOP3 ops that don't support literals, emit s_mov_b32 first
  llvm::SmallVector<std::string> generateOpWithLiteralHandling(mlir::Operation *op);

  /// Generate code for a label op
  std::string generateLabel(LabelOp labelOp);

  /// Generate code for a comment op
  std::string generateComment(CommentOp commentOp);

  /// Generate code for a raw op
  std::string generateRaw(RawOp rawOp);

  ProgramOp program;
  const PhysicalMapping &mapping;
  TargetAttrInterface target;
  InstructionFormatter formatter;

  int64_t peakVGPRs = 0;
  int64_t peakSGPRs = 0;

  /// Scratch VGPR for loading non-inline literals
  /// We use a lower VGPR (v15) to avoid excessive VGPR allocation.
  /// The Python backend uses v_mov_b32 to materialize large constants
  /// during IR generation, which is better.
  /// TODO: Handle this properly during MLIR translation instead.
  static constexpr int64_t kScratchVGPR = 15;
};

//===----------------------------------------------------------------------===//
// Assembly Output
//===----------------------------------------------------------------------===//

/// Write assembly to a file
mlir::LogicalResult writeAssembly(ProgramOp program,
                                   const PhysicalMapping &mapping,
                                   llvm::StringRef outputPath);

/// Write assembly to a stream
mlir::LogicalResult writeAssembly(ProgramOp program,
                                   const PhysicalMapping &mapping,
                                   llvm::raw_ostream &os);

} // namespace waveasm

#endif // WaveASM_TRANSFORMS_ASSEMBLYEMITTER_H
