# Cross-Class Register Spilling in WaveASM

- **Author:** Ivan Butygin
- **Status:** Draft
- **Created:** 2026-03-28

## Table of Contents

- [Problem Statement](#problem-statement)
- [Background](#background)
- [Design](#design)
- [Implementation Plan](#implementation-plan)
- [Alternatives Considered](#alternatives-considered)
- [Open Questions](#open-questions)

## Problem Statement

The WaveASM linear-scan register allocator treats VGPR, SGPR, and AGPR pools
as independent.  When any pool is exhausted the compilation fails with a hard
error.  There is no spilling of any kind.

The 256x192x256 block MXFP4 GEMM with dynamic M/N/K allocates ~261 VGPRs,
5 over the gfx950 hardware limit of 256.  The kernel simultaneously leaves
~64 AGPRs unused (only 192 of 256 consumed by MFMA accumulators).

A post-regalloc Python hack rewrites assembly text to shuttle excess VGPRs
through AGPRs via `v_accvgpr_read/write`.  It works but is fragile: no
dataflow analysis, regex-based operand classification, hardcoded scratch
registers, and `s_waitcnt vmcnt(0)` sledgehammers that destroy latency
hiding.

This document proposes a proper cross-class spilling mechanism inside the
linear-scan allocator where liveness, tied values, and hazard information
are available.

## Background

### AMDGCN Register File (gfx9)

| Class | Count | Typical use                    | Move cost (cycles) |
|-------|-------|--------------------------------|--------------------|
| SGPR  | 102   | Scalar values, SRDs, loop IVs  | --                 |
| VGPR  | 256   | Per-lane vector values          | --                 |
| AGPR  | 256   | MFMA accumulators               | 4 (read/write)     |

Cross-class move instructions:

- **SGPR -> VGPR**: `v_mov_b32 vD, sS` (1 VALU cycle).
- **VGPR -> SGPR**: `v_readfirstlane_b32 sD, vS` (1 VALU cycle, lane 0).
- **VGPR -> AGPR**: `v_accvgpr_write_b32 aD, vS` (4 cycles).
- **AGPR -> VGPR**: `v_accvgpr_read_b32 vD, aS` (4 cycles, +1 s_nop on
  gfx950 before consumer).

There is no direct SGPR <-> AGPR path; a VGPR scratch is required as
intermediate.

### LLVM Precedent

LLVM's AMDGPU backend implements SGPR -> VGPR spilling
(`SILowerSGPRSpills`).  When SGPRs are exhausted, values are parked in
dedicated VGPR "spill lanes."  Reload uses `v_readfirstlane_b32`.

LLVM does **not** implement VGPR -> AGPR spilling.  When VGPRs overflow,
values spill to scratch memory via `buffer_store/load` on the private
segment.  This costs ~100+ cycles per access and consumes VMEM bandwidth.

The proposed cross-class approach avoids scratch memory entirely for moderate
overflow (tens of registers) at a cost of 4-8 VALU cycles per spill/reload.

### Current Allocator Architecture

Relevant source files (all under `waveasm/`):

```
include/waveasm/Transforms/RegAlloc.h      -- RegPool, PhysicalMapping, allocator class
lib/Transforms/LinearScanRegAlloc.cpp       -- core linear scan
lib/Transforms/LinearScanPass.cpp           -- pass wrapper, precoloring
lib/Transforms/Liveness.cpp                 -- live range computation
```

Key properties:

1. Three independent `RegPool` instances (VGPR, SGPR, AGPR).
2. Live ranges sorted by `(start, end)` and processed in start-point order.
3. Two separate tie mechanisms: `TiedValueClasses` in Liveness for loop
   iter_args (block_arg + init_arg + iter_arg + result), and a separate
   `tiedPairs` map in LinearScanPass for MFMA accumulator -> result ties.
4. Bottom-up allocation only (`allocSingle`, `allocRange`).  No top-down
   or bidirectional heuristics.
5. `allocateRegClass()` receives a single `RegPool &pool` -- it has no
   access to other register classes when allocation fails.
6. Hard error on pool exhaustion -- no eviction, no spilling.
7. No rematerialization.  No post-regalloc compaction.
8. v15 reserved for literal materialization (`kScratchVGPR` in
   AssemblyEmitter.h).  v14 is not currently reserved.
9. Hazard mitigation pass handles VALU -> `v_readfirstlane` and
   Trans -> VALU hazards only.  No `v_accvgpr_read_b32` RAW hazard
   handling exists yet.

## Design

### Spill Cascade

When `tryAllocate()` returns no register for a given class, the allocator
attempts cross-class eviction before failing:

```
SGPR overflow  -->  park in spare VGPR     (v_readfirstlane to reload)
VGPR overflow  -->  park in spare AGPR     (v_accvgpr_write/read)
AGPR overflow  -->  park in spare VGPR     (v_accvgpr_read/write)
all exhausted  -->  hard error (future: scratch memory)
```

The cascade is one-level deep: a VGPR spilled to AGPR does not trigger
further cascading.  This keeps the design simple and covers the practical
cases (VGPR overflow with spare AGPRs being the dominant scenario).

### Eviction Strategy

When the incoming range cannot be allocated, we evict an already-allocated
**victim** range to the alternate class.  This frees the victim's physical
register for the incoming range.

Victim selection criteria (in priority order):

1. **Fewest use sites** in the hot region (loop body).  Each use site
   requires a reload instruction, so fewer uses means lower spill cost.
2. **Not part of a tied equivalence class.**  Spilling a tied value
   requires spilling the entire class (loop init_arg + block_arg +
   iter_arg + result all share one physical register).  Avoid this
   unless there is no untied candidate.
3. **Longest remaining range.**  Among equal-cost candidates, evicting
   the longest range frees the register for the most time, reducing
   future pressure.

The victim must have `size == 1`.  Multi-register spills (size 2/4/8)
require contiguous ranges in the target class and complicate reload
sequences.  For the initial implementation we restrict to single-register
eviction.  This covers the common case (individual scalar values that
happen to sit in VGPRs, or single excess VGPRs from address computation).

### Spill and Reload Insertion

The spill/reload is a **range split**: the victim's original live range is
shortened to just its def point, and a new "spill range" in the alternate
class covers the rest.  At each use site a reload is inserted.

#### VGPR -> AGPR

At the victim's def point, **after** the defining op:

```asm
v_accvgpr_write_b32 aX, vY       ; spill to AGPR
```

If the def is an async load (`buffer_load`, `global_load`), the spill
must wait for the load to complete.  Rather than inserting a blanket
`s_waitcnt vmcnt(0)`, the spill point is deferred to just before the
first use, and the existing waitcnt analysis pass handles the dependency.
(See [Interaction with WaitCnt Pass](#waitcnt-pass) below.)

Before each use site:

```asm
v_accvgpr_read_b32  vSCR, aX     ; reload to scratch VGPR
; (hazard mitigation pass inserts s_nop if needed)
<original op with vY replaced by vSCR>
```

The scratch register `vSCR` is a dedicated per-spill-site temporary
allocated from a small reserved pool (currently v14, v15).  After the
use, `vSCR` is dead and available for the next reload.

#### SGPR -> VGPR

At the victim's def point:

```asm
v_mov_b32 vX, sY                 ; broadcast scalar to VGPR lane
```

Before each use site:

```asm
v_readfirstlane_b32 sSCR, vX     ; extract lane 0 back to SGPR
<original op with sY replaced by sSCR>
```

#### AGPR -> VGPR

Same as VGPR -> AGPR but reversed:

```asm
v_accvgpr_read_b32 vX, aY        ; at def
v_accvgpr_write_b32 aSCR, vX     ; reload before use
```

This direction is uncommon (MFMA-heavy kernels rarely overflow AGPRs
while having spare VGPRs) but is included for completeness.

### Scratch Register Management

The current allocator reserves v15 for literal materialization
(`kScratchVGPR` in AssemblyEmitter.h, reserved at LinearScanPass.cpp:159).
v14 is not currently reserved.

Proposed approach:

- Reserve v14 as a second scratch VGPR for spill reloads
  (`reservedVGPRs.insert(14)` in LinearScanPass.cpp).  This gives a
  **scratch pool** of 2 VGPRs (v14, v15).
- Each reload site needs exactly 1 scratch register for the duration
  of one instruction.  With 2 scratches available, two independent
  reloads can be in flight (e.g., an instruction reading two different
  spilled values).
- If more than 2 simultaneous reloads are needed at one instruction,
  the allocator must serialize them (reload one, use it, then reload
  the next).  This is handled by the reload insertion logic.
- Scratch registers for SGPR reloads (`sSCR`) are drawn from the
  SGPR pool.  One dedicated SGPR scratch (e.g., s0, since kernel args
  are already loaded) suffices for single-register spills.

### Interaction with Existing Passes

#### Liveness

No changes to the liveness pass itself.  `LiveRange` carries a `regClass`
field and results are separated into `vregRanges`, `sregRanges`,
`aregRanges` (Liveness.cpp:662-669).  The `usePoints` map
(Liveness.h:177) stores per-value use-site indices, which the eviction
heuristic needs for "fewest use sites" ranking.

The spill/reload insertion creates new SSA values with their own def/use
points; the liveness data is recomputed if needed (or the spill is done
in a fixup walk after initial allocation).

#### Tied Values

Two separate tie mechanisms exist:

- **Loop ties** (`TiedValueClasses` in Liveness): group block_arg +
  init_arg + iter_arg + loop_result into equivalence classes.
- **MFMA ties** (`tiedPairs` in LinearScanPass): map MFMA result ->
  accumulator operand, added via `allocator.addTiedOperand()`.

Neither kind is a candidate for cross-class spilling in the initial
implementation.  Spilling a loop-tied value requires spill/reload at
every back-edge.  Spilling an MFMA-tied value means the accumulator
lives in an AGPR but the result needs a VGPR, which defeats the tie.

If the *only* way to satisfy pressure is to spill a tied value, the
allocator falls back to the hard error.  A future extension could handle
this by breaking the tie and inserting explicit copies on back-edges, but
this is out of scope.

#### WaitCnt Pass (Ticketing.cpp)

The existing `--waveasm-insert-waitcnt` pass tracks outstanding VMEM and
LGKM operations and inserts `s_waitcnt` based on operand dependencies.
If a `v_accvgpr_write` consumes a VGPR defined by a `buffer_load`, the
pass must ensure the load has completed.

The spill insertion should emit the `v_accvgpr_write` as a normal
WaveASM op.  The waitcnt pass then sees its operand dependency and
inserts the appropriate `s_waitcnt vmcnt(N)` with a precise count
rather than the blanket `vmcnt(0)` used by the Python hack.

**Caveat**: the waitcnt pass currently tracks memory ops and their
*direct* consumers.  Verify that a `v_accvgpr_write_b32` reading a
`buffer_load` result is recognized as a consumer that needs a wait.

#### Hazard Mitigation

The `--waveasm-hazard-mitigation` pass currently handles only two
hazard patterns (HazardMitigation.cpp):

1. VALU -> `v_readfirstlane_b32` RAW hazard.
2. Transcendental -> non-Trans VALU forwarding hazard.

It does **not** handle `v_accvgpr_read_b32` RAW hazards on GFX950.
This must be added as part of Phase 1: after a `v_accvgpr_read_b32`
writes a VGPR, the next consumer needs an `s_nop 0` (or scheduling
gap) to avoid silent data corruption.

#### ScopedCSE

The `--mlir-cse` pass runs before regalloc.  Spill/reload ops are
inserted after CSE, so there is no risk of the CSE pass merging a
spill reload with an unrelated `v_accvgpr_read`.

## Implementation Plan

### Phase 1: VGPR -> AGPR Spilling (MVP)

This covers the immediate need (256x192 GEMM, 5 excess VGPRs, 64 spare
AGPRs).

1. **RegAlloc.h**: Add `SpillRecord` struct and expose spill results:

   ```cpp
   struct SpillRecord {
     Value victim;          // original SSA value
     int64_t sourcePhysReg; // freed VGPR index
     int64_t targetPhysReg; // allocated AGPR index
     RegClass sourceClass;  // VGPR
     RegClass targetClass;  // AGPR
   };
   ```

   The allocator's return type must be extended to include
   `SmallVector<SpillRecord>` alongside `PhysicalMapping` and
   `AllocationStats`.

2. **LinearScanRegAlloc.cpp**: `allocateRegClass()` currently receives
   a single `RegPool &pool`.  Change signature to also accept the
   alternate-class pool (AGPR pool when allocating VGPRs).  When
   `tryAllocate()` fails:

   a. Scan active list for eviction candidates (untied, size==1,
      fewest uses via `LivenessInfo::usePoints`).
   b. Allocate an AGPR from the alternate pool for the victim.
   c. Free the victim's VGPR.
   d. Re-attempt `tryAllocate()` for the incoming range.
   e. Record eviction in `SmallVector<SpillRecord>`.

3. **LinearScanPass.cpp**: Reserve v14 as spill scratch
   (`reservedVGPRs.insert(14)`) alongside the existing v15 reservation.
   After `allocator.allocate()` succeeds, process `SpillRecord` list
   (insertion point: after mapping adjustments at ~line 290, before IR
   type transformation at ~line 292):

   a. For each spill, walk the program and insert
      `v_accvgpr_write_b32` after the victim's def.
   b. Before each use of the victim, insert
      `v_accvgpr_read_b32 vSCR, aX`.
   c. Rewrite the use to read from `vSCR`.
   d. Update the op's result type from `PVRegType` to `PARegType`
      for the spilled value (or keep as VGPR and let the inserted
      ops handle it -- TBD based on what is cleaner for downstream
      passes).

4. **HazardMitigation.cpp**: Add `v_accvgpr_read_b32` RAW hazard
   detection for GFX950.  When a `v_accvgpr_read_b32` writes a VGPR
   and the next instruction consumes it, insert `s_nop 0`.  This is
   required for correctness -- without it, MFMA scale operands can be
   silently corrupted.

5. **Tests**: Add LIT tests in `waveasm/test/Transforms/`:

   - `cross-class-spill-vgpr-to-agpr.mlir`: Verify spill/reload
     insertion with max-vgprs=4 on a program needing 5.
   - `cross-class-spill-no-tied.mlir`: Verify tied values are not
     spilled.
   - `cross-class-spill-async-load.mlir`: Verify correct ordering
     when the spilled value comes from a buffer_load.

### Phase 2: SGPR -> VGPR Spilling

Same pattern, different move instructions.  Lower priority since SGPR
overflow is less common in current kernels (SALU promotion already
keeps most scalars in SGPRs).

1. When SGPR allocation fails, find an untied SGPR victim.
2. Allocate a VGPR for it.
3. Insert `v_mov_b32 vX, sY` at def, `v_readfirstlane_b32 sSCR, vX`
   at uses.

### Phase 3: AGPR -> VGPR Spilling

Reverse of Phase 1.  Lowest priority -- AGPR overflow with spare VGPRs
is rare in practice.

### Phase 4 (Future): Scratch Memory Spill

When cross-class spilling is insufficient (all three classes at capacity),
the last resort is scratch memory.  This requires:

- A scratch SRD (already available in the kernel descriptor).
- `buffer_store_dword` at spill, `buffer_load_dword` at reload.
- Stack frame offset management.
- Integration with the waitcnt pass for VMEM tracking.

Out of scope for this design.  The cross-class approach should cover
practical cases for the foreseeable kernel sizes.

## Alternatives Considered

1. **Post-regalloc assembly text rewriting (current hack).**
   Works but unsound: no dataflow analysis, regex operand classification,
   blanket waitcnts, hardcoded scratch registers.  See Problem Statement.

2. **Reduce register pressure at the IR level.**
   More aggressive rematerialization, shorter live ranges, or kernel
   restructuring.  This is complementary (and we should do it), but
   cannot always close the gap.  The 256x192 kernel is already heavily
   optimized and still needs 261 VGPRs.

3. **Scratch memory spilling only (like LLVM).**
   Sound but expensive.  4 cycles for an AGPR move vs 100+ cycles for
   scratch buffer_load, plus VMEM bandwidth contention with data loads.
   Cross-class spilling is strictly better when spare registers exist in
   the target class.

4. **Occupancy reduction.**
   Accept fewer waves per CU by using more registers.  Not applicable
   here: gfx950 has a hard limit of 256 VGPRs regardless of occupancy.

5. **Second-chance allocation (evict + re-spill).**
   When evicting a victim, allow the victim to itself be evicted further
   (multi-level cascade).  Adds complexity for marginal benefit.
   Single-level suffices for tens of excess registers.

## Open Questions

1. **Spill at def vs spill at last-use-before-gap?**  Spilling at def is
   simpler but keeps the AGPR occupied for the entire range.  Spilling
   at the last use before a pressure spike is optimal but requires
   pressure curve analysis.  Start with spill-at-def.

2. **Cost model for eviction.**  Use count alone, or weight by loop depth?
   A value used once inside a loop body that executes 128 times is
   effectively 128 uses.  Start with raw use count; refine if profiling
   shows poor spill placement.

3. **Multiple simultaneous spills.**  If 5 VGPRs must spill, we need 5
   spare AGPRs and potentially 2+ reloads in a single instruction.  The
   scratch pool of 2 VGPRs limits us to 2 simultaneous reloads.  Should
   we grow the scratch pool dynamically, or serialize reloads?  Start
   with serialization (insert multiple read + nop sequences).

4. **Interaction with loop-address-promotion.**  The promotion pass adds
   VGPRs to eliminate VALU from the loop body.  If those promoted VGPRs
   are then spilled to AGPRs, the VALU savings are partially offset by
   spill/reload VALU.  Should we teach the promotion pass about the
   AGPR budget?  Deferred -- the fallback path already disables
   promotion when VGPRs exceed the limit.
