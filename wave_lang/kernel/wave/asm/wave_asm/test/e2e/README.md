# WaveASM End-to-End Tests

This directory contains end-to-end tests that validate the C++ WaveASM backend against the Python implementation and actual GPU execution.

## Test Structure

```
test/e2e/
├── waveasm_e2e.py           # Test utilities and compiler wrapper
├── run_cpp_backend_e2e.py   # Standalone E2E test (run directly)
├── conftest.py              # Pytest configuration
├── test_copy_kernel.py      # Copy kernel tests (pytest)
├── test_asm_backend_e2e.py  # Comprehensive test suite (mirrors asm_backend_test.py)
└── README.md                # This file
```

## Test Coverage

The `test_asm_backend_e2e.py` file provides comprehensive coverage mirroring the Python `tests/kernel/asm_backend_test.py`:

| Test | Description |
|------|-------------|
| `test_copy_kernel_cpp_backend` | Simple copy kernel (16x16) |
| `test_mma_kernel_cpp_backend` | Single-tile MMA (16x16x16, 16x16x32) |
| `test_mma_multi_workgroup_single_wave_cpp_backend` | Multi-workgroup MMA (32x32 to 256x256) |
| `test_mma_multi_wave_cpp_backend` | Multi-wave MMA (4-16 waves per workgroup) |
| `test_gemm_cpp_backend` | Full GEMM with K-loop (single/multi-wave, g2s) |
| `test_compare_backends_copy_kernel` | C++ vs Python backend comparison |

## Quick Start: Comprehensive E2E Tests

Run the full test suite (mirrors asm_backend_test.py):

```bash
# SSH into mi350-4 and start docker
ssh mi350-4
docker exec -it c2b37c0e9709 bash

# Set up paths
cd /path/to/wave-asm
export WAVEASM_TRANSLATE=$(pwd)/build/tools/waveasm-translate/waveasm-translate

# Run all comprehensive e2e tests
pytest test/e2e/test_asm_backend_e2e.py -v

# Run specific test categories
pytest test/e2e/test_asm_backend_e2e.py -v -k "copy"      # Copy kernel only
pytest test/e2e/test_asm_backend_e2e.py -v -k "mma"       # All MMA tests
pytest test/e2e/test_asm_backend_e2e.py -v -k "gemm"      # GEMM tests
pytest test/e2e/test_asm_backend_e2e.py -v -k "multi"     # Multi-wave/workgroup
pytest test/e2e/test_asm_backend_e2e.py -v -k "compare"   # Backend comparison
```

## Quick Start: Standalone E2E Test

The fastest way to test the C++ backend on mi350-4:

```bash
# SSH into mi350-4 and start docker
ssh mi350-4
docker exec -it c2b37c0e9709 bash

# Set up paths
cd /path/to/wave-asm
export WAVEASM_TRANSLATE=$(pwd)/build/tools/waveasm-translate/waveasm-translate

# Run the standalone E2E test
python test/e2e/run_cpp_backend_e2e.py

# Compare C++ vs Python backend output
python test/e2e/run_cpp_backend_e2e.py --compare

# Keep temp files for debugging
python test/e2e/run_cpp_backend_e2e.py --keep-temp
```

The standalone test will:
1. Define a copy kernel using wave_lang
2. Capture MLIR IR before ASM codegen
3. Compile to assembly using C++ waveasm-translate
4. Assemble to .hsaco binary using amdclang++
5. Execute on GPU using wave_runtime
6. Validate output matches PyTorch

## Test Categories

### 1. Translation Tests (No GPU Required)
These tests verify MLIR → ASM translation using the C++ backend:

```bash
# Run translation tests only
pytest test/e2e/ -v -k "translation"

# Or run the test directly
python test/e2e/test_copy_kernel.py
```

### 2. Compilation Comparison Tests
These tests compare C++ and Python backend outputs:

```bash
pytest test/e2e/test_copy_kernel.py::TestWaveKernelCompilation -v
```

### 3. GPU Execution Tests
These tests run the full pipeline and validate on GPU:

```bash
# Enable GPU tests
RUN_GPU_TESTS=1 pytest test/e2e/ -v --run-e2e
```

## Requirements

### For Translation Tests
- C++ WaveASM built (`waveasm-translate` in PATH or `build/` directory)
- wave_lang Python package

### For GPU Tests
- ROCm installed (amdclang++ available)
- AMD GPU (gfx942, gfx950, etc.)
- wave_runtime Python package
- PyTorch with ROCm support

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WAVEASM_TRANSLATE` | Path to waveasm-translate | Auto-detected |
| `WAVE_DEFAULT_ARCH` | Target architecture | Auto-detected or gfx942 |
| `ROCM_PATH` | ROCm installation path | /opt/rocm |
| `RUN_GPU_TESTS` | Enable GPU execution tests | 0 |

## Running on MI350

SSH into the test machine and run in Docker:

```bash
ssh mi350-4
docker exec -it c2b37c0e9709 bash

# In docker
cd /path/to/wave-asm
export WAVEASM_TRANSLATE=/path/to/build/tools/waveasm-translate/waveasm-translate

# Run translation tests
pytest test/e2e/ -v -k "translation"

# Run full e2e tests
RUN_GPU_TESTS=1 pytest test/e2e/ -v --run-e2e
```

## Adding New Tests

To add a new kernel test:

1. Create a new test file `test_<kernel_name>.py`
2. Define the kernel using wave_lang decorators
3. Use `capture_wave_mlir()` to get MLIR
4. Use `WaveASMCompiler` to compile with C++ backend
5. Optionally compare with Python backend
6. For GPU tests, use `run_with_wave_runtime()` to execute

Example:

```python
from waveasm_e2e import WaveASMCompiler, capture_wave_mlir

def test_my_kernel():
    # Define kernel
    @tkw.wave(constraints)
    def my_kernel(a, b):
        ...

    # Capture MLIR
    mlir_text = capture_wave_mlir(options, my_kernel)

    # Compile with C++ backend
    compiler = WaveASMCompiler(target="gfx942")
    success, asm, _ = compiler.compile_mlir_to_asm(mlir_text)

    assert success, f"Compilation failed: {asm}"
    assert "s_endpgm" in asm  # Basic sanity check
```

## Debugging

Keep temporary files for inspection:

```bash
pytest test/e2e/ -v --keep-temp
```

Files will be saved to `/tmp/waveasm_e2e_*/`:
- `input.mlir` - Input MLIR
- `output.s` - Generated assembly
- `kernel.o` - Object file
- `kernel.hsaco` - GPU binary
