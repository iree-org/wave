Wave is a Python DSL for high-performance ML kernel development targeting AMD GPUs (ROCm). The default compilation path is pure Python using IREE for codegen. Water and WaveASM are optional C++ extensions that replace parts of the IREE path.

## Commands

### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-iree-pinned.txt
pip install -r pytorch-cpu-requirements.txt  # CPU-only dev/testing
pip install -e ".[dev]"
pre-commit install && pre-commit install --hook-type commit-msg
```

### Testing
```bash
pytest -n 4 --capture=tee-sys -vv ./tests/unittests/   # unit tests
pytest -s tests/unittests/test_file.py::test_name -v   # single test
lit lit_tests/ -vv                                     # MLIR LIT tests
pytest -s tests/ --run-e2e                             # GPU tests (requires hardware)
```

### Linting
```bash
mypy                        # type check wave_lang
pre-commit run --all-files  # Black, Ruff, clang-format
```

### Gotchas
- **Always set `WAVE_CACHE_ON=0`** when testing code changes — stale cache entries hide the effect of edits: `WAVE_CACHE_ON=0 pytest ...`
- DCO sign-off required on commits: `git commit -s`
- Dump MLIR for debugging: `pytest --dump-mlir-files-path=/tmp/mlir tests/`

## Architecture

### Compilation Flow

```
Wave Python DSL
    ↓  graph transformation passes  [wave_lang/kernel/wave/codegen/]
Transformed FX graph
    ↓  WaveEmitter  [compiler/wave_codegen/emitter.py]
stream.executable MLIR
    ↓  iree.compiler.compile_str()  [wave/utils/compile_utils.py]
VMFB (IREE bytecode module)
    ↓  iree.runtime.VmModule
GPU kernel execution
```

Entry point: `wave_compile()` in `wave_lang/kernel/wave/compile.py`.

### Runtimes

**IREE runtime (default):** Loads VMFB into IREE's VM. Handles GPU command buffers, queue submission, benchmarking, multi-device.

**Wave runtime (`options.wave_runtime=True`):** Launches HSACO kernels directly via HIP API. Supports dynamic strides and custom grid layout. Typically paired with WaveASM. Entry point: `invoke_with_wave_runtime()` in `wave_lang/kernel/wave/utils/run_utils.py`.

### Key Source Locations

- `wave_lang/kernel/wave/compile.py` — pipeline orchestration, backend/runtime selection
- `wave_lang/kernel/wave/codegen/` — graph transformation passes (scheduling, barriers, index analysis)
- `wave_lang/kernel/compiler/wave_codegen/emitter.py` — lowers FX graph to MLIR
- `wave_lang/kernel/wave/water.py` — Water/WaveASM lowering pipeline entry points
- `wave_lang/kernel/wave/mlir_converter/` — Wave FX ↔ Water MLIR conversion; runs in a subprocess to avoid MLIR library conflicts (Water backend only)

### Optional Extensions

Water and WaveASM intercept MLIR before IREE and produce HSACO directly. Enable via env vars:

| Variable | Purpose |
|---|---|
| `WAVE_BUILD_WATER=1` | Build Water from source |
| `WAVE_BUILD_WAVEASM=1` | Build WaveASM from source |
| `WAVE_WATER_DIR=water/build` | Use existing Water build (fast) |
| `WAVE_WAVEASM_DIR=waveasm/build` | Use existing WaveASM build (fast) |

When both active: stream.executable MLIR → `water-opt` → `waveasm-translate` → `water-opt` → ExecutionEngine.
