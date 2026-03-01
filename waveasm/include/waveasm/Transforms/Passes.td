// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_TRANSFORMS_PASSES
#define WaveASM_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

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
           "Maximum SGPRs available">,
    Option<"maxAGPRs", "max-agprs", "int64_t", "256",
           "Maximum AGPRs available">
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
           "Insert waitcnt immediately after loads (conservative)">,
    Option<"ticketedWaitcnt", "ticketed-waitcnt", "bool", "true",
           "Ticket-based waitcnt/barrier insertion (set to false to disable)">
  ];

  let statistics = [
    Statistic<"numVmemOps", "VMEM operations tracked", "count">,
    Statistic<"numLgkmOps", "LGKM operations tracked", "count">,
    Statistic<"numWaitcntInserted", "Waitcnt instructions inserted", "count">
  ];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Peephole Optimization Pass
//===----------------------------------------------------------------------===//

def WAVEASMPeephole : Pass<"waveasm-peephole"> {
  let summary = "Local pattern-based peephole optimizations for WAVEASM IR";
  let description = [{
    Instruction fusion, strength reduction, constant folding, and redundant
    move elimination.  All transformations are local rewrite patterns applied
    via the greedy pattern rewrite driver.
  }];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// M0 Redundancy Elimination Pass
//===----------------------------------------------------------------------===//

def WAVEASMM0RedundancyElim : Pass<"waveasm-m0-redundancy-elim"> {
  let summary = "Eliminate redundant M0 register writes";
  let description = [{
    Tracks the last s_mov_b32_m0 source value within each basic block and
    erases writes that set M0 to the value it already holds.  Conservatively
    resets tracking across region-bearing ops (loops, ifs) whose bodies may
    clobber M0.
  }];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Scoped CSE Pass
//===----------------------------------------------------------------------===//

def WAVEASMScopedCSE : Pass<"waveasm-scoped-cse"> {
  let summary = "Scoped Common Subexpression Elimination for WAVEASM IR";
  let description = [{
    Implements scoped CSE respecting structured control flow by maintaining
    a scope stack.  Only operations within the same or dominating scope can
    be deduplicated.
  }];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Buffer Load Strength Reduction Pass
//===----------------------------------------------------------------------===//

def WAVEASMBufferLoadStrengthReduction
    : Pass<"waveasm-buffer-load-strength-reduction"> {
  let summary = "Replace per-iteration buffer_load address computation with "
                "soffset-based incremental addressing";
  let description = [{
    Replaces per-iteration buffer_load voffset recomputation with precomputed
    voffsets and SGPR soffset bumping.  Symbolically differentiates the address
    chain w.r.t. the induction variable to compute a constant stride, then
    groups candidates by (SRD, stride) to share soffsets.
  }];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Loop Address Promotion Pass
//===----------------------------------------------------------------------===//

def WAVEASMLoopAddressPromotion : Pass<"waveasm-loop-address-promotion"> {
  let summary = "Promote loop-carried LDS read address computation";
  let description = [{
    Promotes loop-carried LDS read address computation from per-iteration
    V_ADD_U32 operations to precomputed rotating VGPR iter_args.  Eliminates
    VALU address computation from the loop body.
  }];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Scale Pack Elimination Pass
//===----------------------------------------------------------------------===//

def WAVEASMScalePackElimination : Pass<"waveasm-scale-pack-elimination"> {
  let summary = "Eliminate BFE/LSHL_OR round-trips for B-scale iter_args";
  let description = [{
    Eliminates redundant BFE->iter_arg->LSHL_OR round-trips for B-scale values
    in scaled MFMA loops.  Carries the dword directly as a loop iter_arg instead
    of four separate byte extractions.
  }];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Memory Offset Optimization Pass
//===----------------------------------------------------------------------===//

def WAVEASMMemoryOffsetOpt : Pass<"waveasm-memory-offset-opt"> {
  let summary = "Fold constant address components into memory instruction "
                "offset fields";
  let description = [{
    Folds constant address components into memory instruction offset fields,
    reducing VALU instruction count.  Handles constant adds, constant through
    V_MOV_B32, and algebraic distribution through shifts.
  }];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

#endif // WaveASM_TRANSFORMS_PASSES
