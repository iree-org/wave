WaveASM is an optional C++ backend in the Wave compiler stack that replaces IREE's GPU codegen. It translates MLIR into AMDGCN assembly for AMD GPUs (gfx942/CDNA3, gfx950/CDNA3.5, gfx1250/RDNA4) and produces `.hsaco` binaries via its own `waveasm.*` MLIR dialect, linear-scan register allocator, and assembly emitter.

## Building

```bash
# First build
WAVE_BUILD_WAVEASM=1 pip install -e ".[dev]"

# Iterating on C++ changes (same pattern as Water)
ninja -C waveasm/build
pip install -e ".[dev]"   # re-links extension, skips CMake
```

Set `WAVE_WAVEASM_DIR=waveasm/build` after first build to avoid full rebuilds on pip install. CLI tool: `waveasm-translate`.

## Testing

```bash
ninja -C waveasm/build check-waveasm      # lit regression tests
ninja -C waveasm/build check-waveasm-all  # + GPU functional tests (requires hardware)
lit test/Transforms/<test>.mlir -vv       # single test
```

## Architecture

### Compilation Pipeline

```
Input MLIR (gpu, arith, vector, memref, scf, amdgpu dialects)
    ↓  TranslateFromMLIR  [lib/Transforms/TranslateFromMLIR.cpp]
WaveASM IR (virtual registers, pseudo-ops)
    ↓  ScopedCSE, Peephole, BufferLoadStrengthReduction
    ↓  ArithLegalization
Concrete SALU/VALU machine ops
    ↓  Liveness → LinearScanRegAlloc → VGPRCompaction
Physical register assignments
    ↓  Ticketing, HazardMitigation
    ↓  AssemblyEmitter → clang++
.hsaco GPU binary
```

### Dialect

Types (`WaveASMTypes.td`): virtual (`!waveasm.vreg/sreg/areg`) and physical (`!waveasm.pvreg/psreg/pareg`) register types, plus `!waveasm.imm` and `!waveasm.scc`. The two-phase virtual→physical split is intentional — optimization passes run on virtual SSA, allocation happens once at the end.

~300 machine ops in `WaveASMOps.td`: VALU, SALU, MFMA, memory (global/LDS/SMEM), control flow, and utility ops. Pseudo-ops (`waveasm.arith.*`) exist for cases where the concrete instruction depends on register class — ArithLegalization resolves them.

### Adding New Dialect Support

`TranslateFromMLIR` uses a handler registry. To translate a new upstream op, add a handler to the appropriate file in `lib/Transforms/handlers/` and register it in the `TranslationContext`. The `TranslationContext` also manages the SRD (Shader Resource Descriptor) table and expression cache — use it rather than tracking state locally in handlers.

### Non-Obvious Constraints

- **No spilling**: `LinearScanRegAlloc` aborts if register pressure exceeds hardware limits. If you see allocation failures, the kernel uses too many live values simultaneously.
- **Tied operands**: MFMA accumulator input and output must share the same physical registers. This is expressed via `TiedClass` equivalence classes in `Liveness` — new MFMA variants must declare their ties correctly.
- **SCC liveness**: `!waveasm.scc` is an implicit 1-bit condition code, not a normal SSA value. The SCC verifier enforces that SCC is consumed before the next instruction that overwrites it. SCC spill/reload uses `s_cselect_b32` / `s_cmp_ne`.
- **Ticketing**: `s_waitcnt` insertion is demand-driven via ticket tracking, not conservative. Passes that add new memory ops must ensure they participate in the ticket system.
