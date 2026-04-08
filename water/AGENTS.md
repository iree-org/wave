Water is an optional MLIR layer in the Wave compiler stack that replaces IREE's middle-end lowering. It defines the `wave.*` and `normalform.*` dialects, transformation passes, and Python bindings (`water_mlir` package).

## Building

Water must be built with CMake first. `pip install` alone does not build Water — `WAVE_WATER_DIR` is required to point Wave at an existing Water build.

LLVM is pinned at `water/llvm-sha.txt`. CLI tool: `water-opt` (analogous to `mlir-opt`).

### Step 1: Build Water with CMake

Requires a pre-built LLVM/MLIR. Set `$BUILD_DIR` to your LLVM build or install tree.

```bash
# Configure
cmake -G Ninja \
      -B water/build \
      water/ \
      -DMLIR_DIR=$BUILD_DIR/lib/cmake/mlir \
      -DBUILD_SHARED_LIBS=ON \
      -DPython3_EXECUTABLE="$(which python)" \
      -DWATER_ENABLE_PYTHON=ON

# Optional: faster builds with clang + ccache + lld
cmake -B water/build \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DLLVM_USE_LINKER=lld

# Build
cmake --build water/build
```

### Step 2: Install Wave with Water bindings

```bash
WAVE_WATER_DIR=water/build pip install -e ".[dev]"
```

`WAVE_WATER_DIR` tells Wave where to find the Water build. Without it, Water is not included.

### Iterating on C++ changes

```bash
ninja -C water/build          # rebuild changed C++ targets and Python bindings
```

## Formatting

C++ code is formatted with `git clang-format` which formats only the lines changed relative to a commit (default: `HEAD`)
```bash
git clang-format                # format staged changes
git clang-format HEAD~1         # also include most recent commit
git clang-format main           # format everything touched on your branch
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
