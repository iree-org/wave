# CDNA4 Instruction Scheduling Guide

Reference for the Conductor LLM scheduling loop. Based on the AMD Instinct
CDNA4 ISA Reference Guide (gfx950).

## Architecture Overview

- Wave size: 64 threads.
- Register files: 256 arch VGPRs (V0-V255) + 256 AccVGPRs/AGPRs (A0-A255) = 512 total per wave.
- SGPRs: 16-102 per wave (VCC occupies SGPR 106-107).
- LDS: 160 KiB per CU, 64 banks x 32-bit, 1280-byte allocation granularity.
- Allocation granularity: VGPRs in groups of 8, SGPRs in groups of 16.
- Issue model: one instruction per cycle per wavefront (latency hiding via interleaving wavefronts).

## Instruction Latencies

| Instruction class | Latency (cycles) | Counter |
|---|---|---|
| Global load (buffer_load, global_load) | ~100 | vmcnt |
| LDS read (ds_read) | ~20 | lgkmcnt |
| LDS write (ds_write) | ~20 | lgkmcnt |
| MFMA F16/BF16 16x16x16 | 16 | — |
| MFMA F16/BF16 32x32x8 | 32 | — |
| MFMA F32 16x16x4 | 32 | — |
| MFMA F32 32x32x2 | 64 | — |
| MFMA F8/F4 16x16x128 | 16 (FP4/FP6) or 32 (FP8) | — |
| MFMA F8/F4 32x32x64 | 32 (FP4/FP6) or 64 (FP8) | — |
| scaled_mfma (MXFP4) | Same as F8/F4 above | — |
| MFMA F64 16x16x4 | 64 | — |
| VALU (non-transcendental) | 1 | — |
| VALU transcendental (exp, log, rcp, rsq, sqrt, sin, cos) | 2 | — |
| SALU | 1 | — |
| Scalar memory read | variable | lgkmcnt |

## Waitcnt Counters

| Counter | Bits | Max outstanding | Tracked operations |
|---|---|---|---|
| vmcnt | 6 | 63 | Global/buffer loads and stores |
| lgkmcnt | 4 | 15 | LDS ops, scalar memory, sendmsg |

- VMEM reads/writes return in issue order.
- Scalar memory reads can return **out of order** — only `lgkmcnt(0)` is safe for SMEM.
- FLAT instructions increment both vmcnt and lgkmcnt — only `s_waitcnt 0` is safe after FLAT.
- `s_endpgm` implicitly executes `s_waitcnt 0`.

## MFMA Dependency Rules

### Accumulator chains (SrcC = previous vDst, same opcode, same register range)

**Zero software NOPs required.** The hardware provides 2 implicit wait cycles.
This is the intended use pattern for matrix accumulation loops — chain MFMAs
back-to-back on the same accumulator with no stall.

### Cross-dependencies (reading MFMA output as SrcA/SrcB or in VALU)

These require the full output latency to elapse:

| Scenario | Wait (NOPs) |
|---|---|
| XDL write → SrcA/B of any MFMA | 5 / 8 / 12 / 20 |
| XDL write → VALU read/write (RAW+WAW) | 5 / 8 / 12 / 20 |
| XDL write → VMEM/LDS/FLAT overlap | 5 / 8 / 12 / 20 |
| SGEMM write → SrcA/B of any MFMA | 4 / 6 / 10 / 18 |
| DGEMM 16x16x4 write → SrcA/B or VALU | 19 |

The multiple values correspond to different output register overlap depths.

### Cross-type forwarding

| Scenario | Wait (NOPs) |
|---|---|
| SGEMM write → XDL SrcC (overlapping) | **0** (XDL reads SrcC 2x faster) |
| XDL write → SGEMM SrcC (overlapping) | 3 |
| v_cmpx (writes EXEC) → any V_MFMA | **4** (no exec forwarding to matrix core) |

## Software Hazards (Table 11)

The hardware does **not** detect these. The compiler must insert s_nop or
independent instructions.

| First instruction | Second instruction | NOPs required |
|---|---|---|
| VALU writes SGPR | VMEM reads that SGPR | **5** |
| VALU sets VCC or EXEC | VALU reads EXECZ/VCCZ as data | **5** |
| VALU writes EXEC | VALU DPP op | **5** |
| VALU writes SGPR/VCC | v_readlane/v_writelane lane select | **4** |
| VALU writes VCC | v_div_fmas | **4** |
| v_cmpx writes EXEC | v_readlane/v_readfirstlane/v_writelane | **4** |
| VALU writes SGPR/VCC | VALU reads SGPR as constant | **2** |
| v_cmpx writes EXEC | VALU reads EXEC as constant | **2** |
| S_SETREG MODE.vskip | Any vector op | **2** |
| Trans op (exp, log, rcp, ...) | Non-trans VALU consuming result | **1** |
| VALU writes VGPR | VALU DPP reads that VGPR | **1** |
| SALU writes M0 | LDS add-TID / buffer_store_LDS | **1** |
| Mixed VCC alias access | VALU reads VCC as constant | **1** |
| FLAT/BUFFER_STORE x3/x4 | Write VGPRs holding writedata | **1** |

Note: `s_nop N` inserts N+1 idle cycles. So `s_nop 4` = 5 NOPs.

## Occupancy and Register Pressure

Waves per SIMD as a function of VGPR usage (512-slot pool, 8-register granularity):

| VGPRs used | Waves/SIMD | Waves/CU (4 SIMDs) |
|---|---|---|
| ≤64 | 8 | 32 |
| 65-72 | 7 | 28 |
| 73-80 | 6 | 24 |
| 81-96 | 5 | 20 |
| 97-128 | 4 | 16 |
| 129-168 | 3 | 12 |
| 169-256 | 2 | 8 |

AGPRs use an independent pool with the same formula. Final occupancy is
`min(vgpr_waves, agpr_waves, sgpr_waves, lds_waves)`.

**Key breakpoints:** Dropping below 128, 96, 80, or 64 VGPRs each adds one
wave per SIMD. For MFMA-dominant kernels, 2-4 waves/SIMD is often optimal.

## Scheduling Strategy

### 1. Issue global loads early

Global loads have ~100 cycle latency. Move them as far above their consumers
as possible. Every MFMA (16-64 cycles) or LDS op placed between the load and
its `s_waitcnt vmcnt` hides latency for free.

### 2. Interleave LDS reads with MFMA

LDS reads take ~20 cycles. A single MFMA 16x16x16 (16 cycles) nearly covers
one LDS read. Interleaving `ds_read → mfma → ds_read → mfma` hides LDS
latency much better than `ds_read, ds_read, ..., mfma, mfma, ...`.

### 3. Fill MFMA pipeline bubbles

Between two MFMAs with an accumulator dependency (B.SrcC = A.vDst), the
hardware stalls until A completes. Fill this gap with:
- Independent LDS reads/writes (for the next iteration).
- Independent VALU (address computation, data formatting).
- Independent global loads (prefetch for next tile).

### 4. Minimize live ranges

Moving a producer closer to its consumer (or deferring a def until just
before use) reduces the number of simultaneously live VGPRs, potentially
lowering peak register pressure and enabling higher occupancy.

### 5. Double-buffering pattern

The ideal software-pipelined loop body:
```
barrier
ds_read current tile from LDS
global_load next tile (prefetch)
mfma on current tile (hides global_load latency)
s_waitcnt vmcnt(0)
barrier
ds_write next tile to LDS
```

### 6. Metric priorities (lexicographic)

1. **peak_vgpr** — lower = higher occupancy.
2. **s_waitcnt** — fewer waits = better latency hiding.
3. **s_nop** — fewer NOPs = fewer pipeline hazards.
4. **total_instructions** — fewer = less instruction cache pressure.

## LDS Details

- 160 KiB per CU, 64 banks, each 32-bit wide.
- Best case latency (no bank conflicts): 2 cycles.
- Worst case (64-way bank conflict): 64 cycles.
- A wavefront (64 threads) is dispatched over 4 sub-cycles (16 threads each).
- DS_READ2/DS_WRITE2 (64-bit extended) double per-op bandwidth.

## Barrier Semantics

- `s_barrier` synchronizes all wavefronts in a workgroup.
- It does NOT implicitly wait for memory — issue `s_waitcnt` before `s_barrier`
  if protecting memory operations.
- Treat barriers as hard scheduling boundaries. Never move ops across barriers.
