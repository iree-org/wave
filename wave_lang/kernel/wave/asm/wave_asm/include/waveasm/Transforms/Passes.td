// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_TRANSFORMS_PASSES
#define WaveASM_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// MLIR to WAVEASM Translation Pass
//===----------------------------------------------------------------------===//

def WAVEASMTranslateFromMLIR : Pass<"waveasm-translate-from-mlir", "mlir::ModuleOp"> {
  let summary = "Translate standard MLIR dialects to WAVEASM dialect";
  let description = [{
    Translates MLIR modules containing standard dialects (arith, vector,
    memref, gpu, amdgpu, stream, scf, affine) to the WAVEASM kernel dialect.

    Input dialects supported:
    - stream.executable, stream.binding (IREE)
    - func.func (builtin)
    - gpu.thread_id, gpu.block_id
    - vector.load, vector.store, vector.extract_strided_slice
    - memref.alloc, memref.view, memref.reinterpret_cast
    - amdgpu.mfma, amdgpu.lds_barrier
    - arith.constant, arith.addi, arith.muli, arith.index_cast
    - affine.apply
    - scf.for

    The pass:
    1. Extracts kernel functions from stream.executable
    2. Analyzes memory access patterns and buffer bindings
    3. Emits WAVEASM instructions with virtual registers
    4. Creates an waveasm.program with appropriate ABI configuration
  }];

  let options = [
    Option<"target", "target", "std::string", "\"gfx942\"",
           "Target GPU architecture">,
    Option<"codeObjectVersion", "code-object-version", "int64_t", "5",
           "Code object version">
  ];

  let dependentDialects = [
    "::waveasm::WaveASMDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::affine::AffineDialect"
  ];
}

//===----------------------------------------------------------------------===//
// SSA Validation Pass
//===----------------------------------------------------------------------===//

def WAVEASMValidateSSA : Pass<"waveasm-validate-ssa"> {
  let summary = "Validate that the program is in SSA form";
  let description = [{
    Checks that each virtual register is defined exactly once.

    Exceptions:
    - Loop control registers (SGPRs used for loop counters)
    - Accumulator registers (VGPRs used for MFMA read-modify-write)

    These exceptions must be registered on the ProgramOp before validation.
  }];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Liveness Analysis Pass
//===----------------------------------------------------------------------===//

def WAVEASMLiveness : Pass<"waveasm-liveness"> {
  let summary = "Compute liveness information for virtual registers";
  let description = [{
    Runs CFG-based backward dataflow analysis to compute live ranges
    for all virtual registers. The analysis:

    1. Builds a control flow graph from labels and branches
    2. Computes local use/def sets for each basic block
    3. Iterates to fixed point using backward dataflow equations:
       - live_out[B] = union of live_in[S] for all successors S
       - live_in[B] = use[B] union (live_out[B] - def[B])
    4. Computes live ranges from definition to last use
    5. Extends ranges based on CFG liveness for loops

    The results are stored as an analysis and can be queried by
    subsequent passes (e.g., register allocation).
  }];

  let options = [
    Option<"useCFG", "use-cfg", "bool", "true",
           "Use CFG-based analysis (required for loops)">
  ];

  let statistics = [
    Statistic<"numVRegs", "Number of virtual VGPRs", "count">,
    Statistic<"numSRegs", "Number of virtual SGPRs", "count">,
    Statistic<"maxVRegPressure", "Peak VGPR pressure", "registers">,
    Statistic<"maxSRegPressure", "Peak SGPR pressure", "registers">
  ];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Linear Scan Register Allocation Pass
//===----------------------------------------------------------------------===//

def WAVEASMLinearScan : Pass<"waveasm-linear-scan"> {
  let summary = "Linear scan register allocation";
  let description = [{
    Allocates physical registers to virtual registers using the
    linear scan algorithm:

    1. Sort live ranges by start point
    2. For each range (in order):
       a. Expire ranges that ended before this point
       b. Allocate physical register respecting constraints
       c. Add to active set

    Features:
    - Respects ABI precoloring (v0, s[0:1], etc.)
    - Handles alignment constraints (for register pairs/quads)
    - Handles size constraints (for MFMA accumulators)
    - Fails compilation if allocation exceeds limits (no spilling)

    After allocation, virtual register types are replaced with
    physical register types in-place.
  }];

  let options = [
    Option<"maxVGPRs", "max-vgprs", "int64_t", "256",
           "Maximum VGPRs available">,
    Option<"maxSGPRs", "max-sgprs", "int64_t", "104",
           "Maximum SGPRs available">
  ];

  let statistics = [
    Statistic<"peakVGPRs", "Peak VGPRs used", "registers">,
    Statistic<"peakSGPRs", "Peak SGPRs used", "registers">,
    Statistic<"rangesAllocated", "Total ranges allocated", "count">,
    Statistic<"rangesExpired", "Total ranges expired", "count">
  ];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Hazard Mitigation Pass
//===----------------------------------------------------------------------===//

def WAVEASMHazardMitigation : Pass<"waveasm-hazard-mitigation"> {
  let summary = "Insert s_nop instructions to mitigate hardware hazards";
  let description = [{
    Handles hardware-specific hazards that require NOP insertion.

    Currently handles:
    - VALU → v_readfirstlane hazard (gfx940+): When a VALU instruction
      writes a VGPR and the next instruction is v_readfirstlane reading
      that VGPR, an s_nop 0 must be inserted between them.

    The pass scans the instruction stream and inserts s_nop instructions
    where needed to ensure correct execution on the target hardware.
  }];

  let options = [
    Option<"targetArch", "target", "std::string", "\"gfx942\"",
           "Target GPU architecture for hazard rules">
  ];

  let statistics = [
    Statistic<"numNopsInserted", "s_nop instructions inserted", "count">
  ];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Waitcnt Insertion Pass
//===----------------------------------------------------------------------===//

def WAVEASMInsertWaitcnt : Pass<"waveasm-insert-waitcnt"> {
  let summary = "Insert s_waitcnt instructions for memory operation synchronization";
  let description = [{
    Implements a ticket-based tracking system for VMEM and LGKM memory
    operations to insert optimal s_waitcnt instructions.

    The pass tracks two types of memory operations:
    - VMEM: buffer_load, buffer_store, global_load, global_store
    - LGKM: ds_read, ds_write, s_load

    For each memory operation, a "ticket" is assigned. When a value
    produced by a memory operation is used, the pass computes the
    minimum wait threshold needed and inserts s_waitcnt if required.

    Wait coalescing: The pass tracks previously emitted waits and only
    emits a new wait if it's stricter than the previous one, avoiding
    redundant waits and maximizing latency hiding.

    Example:
    - If last_vmem_ticket=5 and we need ticket 3 to complete:
    - threshold = max(0, 5-3) = 2
    - Emit: s_waitcnt vmcnt(2)
    - This waits for ticket 3 while allowing tickets 4,5 to remain in-flight
  }];

  let options = [
    Option<"insertAfterLoads", "insert-after-loads", "bool", "false",
           "Insert waitcnt immediately after loads (conservative)">
  ];

  let statistics = [
    Statistic<"numVmemOps", "VMEM operations tracked", "count">,
    Statistic<"numLgkmOps", "LGKM operations tracked", "count">,
    Statistic<"numWaitcntInserted", "Waitcnt instructions inserted", "count">
  ];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Assembly Emission Pass
//===----------------------------------------------------------------------===//

def WAVEASMEmitAssembly : Pass<"waveasm-emit-assembly"> {
  let summary = "Emit AMDGCN assembly text";
  let description = [{
    Converts the allocated WAVEASM program to AMDGCN assembly text.
    Outputs a complete .s file including:

    - Prologue: .amdgcn_target, .amdhsa_kernel, kernel descriptor
    - Body: Instructions with physical registers
    - Epilogue: .amdgpu_metadata YAML section

    The pass requires all instructions to be in physical form
    (after register allocation).
  }];

  let options = [
    Option<"outputPath", "output", "std::string", "",
           "Output file path (empty for stdout)">,
    Option<"emitMetadata", "emit-metadata", "bool", "true",
           "Emit .amdgpu_metadata section">
  ];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

#endif // WaveASM_TRANSFORMS_PASSES
