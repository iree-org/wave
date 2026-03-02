# Per-Workgroup SRD Base Adjustment for >4GB Output Buffers

## Problem

For large GEMM shapes (e.g., `M=32768, N=57344, K=16384`), the C output matrix `memref<32768x57344xf32>` is ~7GB. This caused two failures:

1. **Assembly error**: `s_mov_b32 srd[2], 0x1C0000000` — the SRD `num_records` field is 32 bits, but the computed buffer size (7,516,192,768 bytes) exceeds 2^32.
2. **Address overflow**: even if `num_records` were clamped, the store voffset `row * 229376 + col * 4` overflows a 32-bit VGPR for workgroups targeting the upper portion of the output matrix.

## Fix

Split the store byte offset into a **workgroup base** (folded into the SRD base address via 64-bit SALU) and a **thread offset** (small, used as `voffset`). This matches AITER's per-workgroup SRD pattern.

### Layer 1: MLIR codegen (Python)

**File**: `wave_lang/kernel/compiler/wave_codegen/read_write.py`

The existing `_linearize_memref` function already separates workgroup offsets (from `block_id * tile_size`) into the memref base pointer and returns thread-only offsets for indexing. It was previously gated on `buffer_ops_enabled`.

Change: for global writes without `buffer_ops`, also call `_linearize_memref` (skipping `_cast_buffer_and_encode_stride`). This produces a `memref.reinterpret_cast` with a dynamic per-workgroup element offset and 1D thread-only store indices:

```mlir
%wg_offset = arith.addi(arith.muli(%block_id_x_times_128, 57344), %block_id_y_times_256)
%tile_mem = memref.reinterpret_cast(%c_raw) offset: [%wg_offset], sizes: [536870910], strides: [1]
vector.store(%val, %tile_mem, [%thread_offset])
```

Thread offsets stay within ~28MB (the 128x256 tile), fitting comfortably in 32 bits.

### Layer 2: C++ backend

Three changes in the WaveASM C++ backend:

**1. Clamp buffer size** (`TranslateFromMLIR.cpp`): In `emitSRDPrologue`, clamp `pending.bufferSize` to `0xFFFFFFFF` before emitting `s_mov_b32` for `num_records`. This is a safety net — the original full-sized `reinterpret_cast` still exists in the MLIR but is unused by stores after linearization.

**2. Track pending SRD adjustments** (`MemRefHandlers.cpp`): In `handleMemRefReinterpretCast`, detect dynamic offsets (from `_linearize_memref`) and store the element offset Value, source SRD index, and element byte width in a `PendingSRDBaseAdjust` map. The actual SALU ops are deferred to the store handler to survive DCE.

**3. Emit SRD adjustment inline** (`TranslateFromMLIR.cpp`): In `handleVectorStore`, when a pending adjustment exists for the store target, emit:

```asm
s_mov_b64 s[N:N+1], s[src:src+1]     ; copy source SRD base
v_readfirstlane_b32 s[N+3], vOffset   ; element offset → SGPR
s_mul_hi_u32 s[N+2], s[N+3], 4       ; byte offset high (for >4GB)
s_mul_i32 s[N+3], s[N+3], 4          ; byte offset low
s_add_u32 s[N], s[N], s[N+3]         ; base_lo += byteOffLo (sets SCC)
s_addc_u32 s[N+1], s[N+1], s[N+2]   ; base_hi += byteOffHi + carry
s_mov_b32 s[N+2], 0x7FFFFFF8         ; num_records (tile-sized)
s_mov_b32 s[N+3], 0x20000            ; stride descriptor
```

The adjustment uses `PSRegType` (precolored physical SGPRs) for all intermediates, with `s[N+2]` and `s[N+3]` serving as temporaries before being overwritten by `num_records` and `stride`. After the first store emits the adjustment, subsequent stores reuse the adjusted SRD via `setSRDIndex`.

### Layer 3: Dialect changes

**File**: `WaveASMOps.td`

- Added `S_ADDC_U32` (carry-dependent add, reads SCC from preceding `s_add_u32`).
- Made `S_ADD_U32` and `S_ADDC_U32` non-`Pure`. These ops set SCC as a side effect; removing `Pure` prevents the canonicalizer from DCE'ing the SRD adjustment chain (whose PSRegType results have no explicit SSA users — they communicate through physical register aliasing with the later `PrecoloredSRegOp`).

## Files modified

| File | Change |
|------|--------|
| `wave_codegen/read_write.py` | Call `_linearize_memref` for global writes without `buffer_ops` |
| `TranslateFromMLIR.cpp` | Clamp `bufferSize` in `emitSRDPrologue`; emit SRD adjustment in `handleVectorStore` |
| `TranslateFromMLIR.h` | Add `PendingSRDBaseAdjust` struct and tracking methods |
| `handlers/MemRefHandlers.cpp` | Detect dynamic offset in `handleMemRefReinterpretCast`, track for deferred emission |
| `WaveASMOps.td` | Add `S_ADDC_U32`; make `S_ADD_U32`/`S_ADDC_U32` non-Pure |
