Water is an optional MLIR layer in the Wave compiler stack that replaces IREE's middle-end lowering. It defines the `wave.*` and `normalform.*` dialects, transformation passes, and Python bindings (`water_mlir` package).

## Building

```bash
# First build — builds LLVM from source, takes a while
WAVE_WATER_DIR=water/build pip install -e ".[dev]"

# Iterating on C++ changes
ninja -C water/build          # rebuild changed targets only
pip install -e ".[dev]"       # re-links Python extension (fast, skips CMake)
```

`WAVE_WATER_DIR` tells the Wave build system to use an existing build directory instead of rebuilding from scratch. Without it, the full LLVM + Water CMake build runs on every `pip install`.

LLVM is pinned at `water/llvm-sha.txt`. CLI tool: `water-opt` (analogous to `mlir-opt`).

## Formatting

C++ code is formatted with `clang-format`. Run via pre-commit or directly:

```bash
clang-format -i <file>          # format a single file in-place
pre-commit run clang-format     # format all staged files
```

## Testing

```bash
ninja -C water/build check-water        # all lit tests
lit test/Dialect/Wave/<test>.mlir -vv   # single test
```

Tests use lit + FileCheck. `.mlir` files use `// CHECK` comments. Negative tests are named `*-invalid.mlir`.

## Architecture

### Dialects

**`wave.*`** — primary dialect. `wave.tensor` has symbolic shapes (unknown until inferred by passes) and an address space (`Global`, `Shared`, `Register`). Each op carries a `WaveIndexMappingAttr` encoding element distribution across device/workgroup/workitem/register dimensions as `(offset, count, step)` triples.

**`normalform.*`** — `normalform.module` wraps IR and enforces declared invariants. Passes declare pre/post-conditions as normal form attributes, enabling composable pass ordering without new IR constructs.

### Pass Pipeline

`water-middle-end-lowering` runs these in order (`include/water/Dialect/Wave/Transforms/Passes.td`):

| Pass | Purpose |
|---|---|
| `water-wave-detect-normal-forms` | Detect satisfied invariants |
| `water-wave-infer-types` | Shape inference via dataflow |
| `water-wave-infer-index-exprs` | Forward/backward index expression propagation |
| `water-wave-propagate-elements-per-thread` | Replace register tensors with vector types |
| `water-wave-resolve-distributed-allocations` | Map distributed shapes to concrete memref layouts |
| `lower-wave-to-mlir` | Lower to arith/math/vector/memref dialects |
| `lower-normalform-module` | Remove the normalform wrapper |

Generic passes include SLP vectorization, bounds-checking assertions, alloc-to-alloca, and GPU module serialization (ROCDL).

### Python Bindings

Package `water_mlir` (prefixed to avoid IREE conflicts):
- `water_mlir.dialects.wave` — auto-generated op bindings from `WaveOps.td`
- `water_mlir.sympy_to_affine_converter` — converts SymPy expressions to MLIR affine expressions
- C++ extension via nanobind (`WaterExtensionNanobind.cpp`)

### Key Design Principles

- **Lazy type inference**: `wave.tensor` shapes start unknown — don't assume they're set at construction.
- **Elements-per-thread (EPT)**: tracked separately from types; required before register tensors can be lowered to vector types. A pass that changes element counts must update EPT.
- **`water_mlir` prefix**: the Python package is prefixed to avoid conflicts with IREE's MLIR bindings. Import as `from water_mlir.dialects import wave`, not `mlir.dialects.wave`.
- **subprocess isolation**: the Wave-side `mlir_converter` runs Water in a subprocess specifically to avoid MLIR library symbol clashes with IREE.
