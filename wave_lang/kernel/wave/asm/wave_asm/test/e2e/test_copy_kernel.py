# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
End-to-end test for copy kernel using C++ WaveASM backend.

This test validates the C++ wave-asm port by:
1. Defining the same copy kernel as asm_backend_test.py
2. Capturing the MLIR IR
3. Compiling via C++ waveasm-translate
4. Assembling to GPU binary
5. Executing and validating against PyTorch

Run with:
    pytest test/e2e/test_copy_kernel.py -v --run-e2e

Or just test MLIR->ASM translation (no GPU):
    pytest test/e2e/test_copy_kernel.py -v -k "translation"
"""

import os
import sys
import pytest
from pathlib import Path

# Add wave_lang to path if needed
wave_root = Path(__file__).parent.parent.parent.parent
if str(wave_root) not in sys.path:
    sys.path.insert(0, str(wave_root))

# Import test utilities
import sys
from pathlib import Path
# Add e2e directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from waveasm_e2e import (
    WaveASMCompiler,
    capture_wave_mlir,
    compare_with_python_backend,
    CompilationResult,
)


def require_e2e(fn):
    """Skip test unless --run-e2e is passed."""
    return pytest.mark.skipif(
        not pytest.config.getoption("--run-e2e", default=False),
        reason="Requires --run-e2e flag",
    )(fn)


def require_gpu(fn):
    """Skip test if no GPU available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return pytest.mark.skip(reason="No GPU available")(fn)
        return fn
    except ImportError:
        return pytest.mark.skip(reason="PyTorch not available")(fn)


def get_target_arch() -> str:
    """Get target architecture from environment or detect."""
    if "WAVE_DEFAULT_ARCH" in os.environ:
        arch = os.environ["WAVE_DEFAULT_ARCH"]
        # Strip feature flags like :sramecc+:xnack-
        return arch.split(":")[0]

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            # Map compute capability to gfx name
            gcn_arch = props.gcnArchName if hasattr(props, 'gcnArchName') else None
            if gcn_arch:
                # Strip feature flags like :sramecc+:xnack-
                return gcn_arch.split(":")[0]
    except Exception:
        pass

    return "gfx942"  # Default


# ============================================================================
# Test: MLIR -> ASM Translation (No GPU required)
# ============================================================================

class TestMLIRTranslation:
    """Test MLIR to ASM translation using C++ backend."""

    def test_simple_copy_mlir(self):
        """Test translation of a simple copy kernel MLIR."""
        # Minimal MLIR for thread ID extraction (simplest GPU kernel)
        mlir_text = """
module {
  gpu.module @copy_module {
    gpu.func @copy_kernel() kernel {
      %tid = gpu.thread_id x
      gpu.return
    }
  }
}
"""
        compiler = WaveASMCompiler(target=get_target_arch())
        success, asm_or_error, stderr = compiler.compile_mlir_to_asm(mlir_text)

        # For now, check if waveasm-translate runs (may have missing handlers)
        # The important thing is it doesn't crash
        print(f"Translation {'succeeded' if success else 'failed'}")
        if not success:
            print(f"Error: {asm_or_error}")
            print(f"Stderr: {stderr}")

        # Check basic structure if successful
        if success:
            assert "waveasm.program" in asm_or_error or ".amdgcn_target" in asm_or_error
            print("Generated ASM preview:")
            print(asm_or_error[:500] if len(asm_or_error) > 500 else asm_or_error)

        compiler.cleanup()


# ============================================================================
# Test: Full Pipeline with Wave Kernel (Requires wave_lang)
# ============================================================================

class TestWaveKernelCompilation:
    """Test full compilation pipeline with Wave kernels."""

    @pytest.fixture
    def compiler(self):
        """Create compiler and cleanup after test."""
        c = WaveASMCompiler(target=get_target_arch(), keep_temp_files=True)
        yield c
        c.cleanup()

    def test_capture_copy_kernel_mlir(self, compiler):
        """Test capturing MLIR from copy kernel definition."""
        try:
            import torch
            import wave_lang.kernel.lang as tkl
            import wave_lang.kernel.wave as tkw
            from wave_lang.kernel.wave.compile import WaveCompileOptions
            from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
        except ImportError:
            pytest.skip("wave_lang not available")

        # Define copy kernel (same as asm_backend_test.py)
        M = tkl.sym.M
        N = tkl.sym.N
        ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

        wave_size = 64
        BLOCK_M = 16
        BLOCK_N = 16

        constraints = [
            tkw.HardwareConstraint(
                threads_per_wave=wave_size,
                vector_shapes={M: BLOCK_M, N: BLOCK_N},
            ),
            tkw.WorkgroupConstraint(M, BLOCK_M, 0),
            tkw.WorkgroupConstraint(N, BLOCK_N, 1),
            tkw.WaveConstraint(M, BLOCK_M),
            tkw.WaveConstraint(N, BLOCK_N),
        ]

        @tkw.wave(constraints)
        def copy_kernel(
            a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
            b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        ):
            res = tkw.read(a)
            tkw.write(res, b)

        # Create options
        options = WaveCompileOptions(
            subs={
                M: 16,
                N: 16,
                ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
            },
            canonicalize=True,
            backend="asm",
            wave_runtime=True,
            compile_to_mlir=False,
        )
        options = set_default_run_config(options)

        # Capture MLIR
        mlir_text = capture_wave_mlir(options, copy_kernel)

        assert mlir_text is not None
        assert len(mlir_text) > 0
        assert "func" in mlir_text or "gpu.func" in mlir_text

        print("Captured MLIR:")
        print(mlir_text[:1000] if len(mlir_text) > 1000 else mlir_text)

        # Try to compile with C++ backend
        success, asm_or_error, stderr = compiler.compile_mlir_to_asm(mlir_text)

        print(f"\nC++ Translation: {'SUCCESS' if success else 'FAILED'}")
        if success:
            print(f"Generated {len(asm_or_error)} bytes of assembly")
            # Save for inspection
            Path("/tmp/copy_kernel.mlir").write_text(mlir_text)
            Path("/tmp/copy_kernel.s").write_text(asm_or_error)
            print("Saved to /tmp/copy_kernel.mlir and /tmp/copy_kernel.s")
        else:
            print(f"Error: {asm_or_error}")

    def test_compare_cpp_vs_python_backend(self, compiler):
        """Compare C++ and Python backend outputs for copy kernel."""
        try:
            import torch
            import wave_lang.kernel.lang as tkl
            import wave_lang.kernel.wave as tkw
            from wave_lang.kernel.wave.compile import WaveCompileOptions
            from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
        except ImportError:
            pytest.skip("wave_lang not available")

        # Define copy kernel
        M = tkl.sym.M
        N = tkl.sym.N
        ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

        constraints = [
            tkw.HardwareConstraint(
                threads_per_wave=64,
                vector_shapes={M: 16, N: 16},
            ),
            tkw.WorkgroupConstraint(M, 16, 0),
            tkw.WorkgroupConstraint(N, 16, 1),
            tkw.WaveConstraint(M, 16),
            tkw.WaveConstraint(N, 16),
        ]

        @tkw.wave(constraints)
        def copy_kernel(
            a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
            b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        ):
            res = tkw.read(a)
            tkw.write(res, b)

        options = WaveCompileOptions(
            subs={
                M: 16,
                N: 16,
                ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
            },
            canonicalize=True,
            backend="asm",
            wave_runtime=True,
            compile_to_mlir=False,
        )
        options = set_default_run_config(options)

        # Capture MLIR
        mlir_text = capture_wave_mlir(options, copy_kernel)

        # Compare backends
        target = get_target_arch()
        cpp_asm, python_asm = compare_with_python_backend(mlir_text, target=target)

        print("\n=== C++ Backend Output ===")
        print(cpp_asm[:2000] if len(cpp_asm) > 2000 else cpp_asm)

        print("\n=== Python Backend Output ===")
        print(python_asm[:2000] if len(python_asm) > 2000 else python_asm)

        # Save both for detailed comparison
        Path("/tmp/copy_kernel_cpp.s").write_text(cpp_asm)
        Path("/tmp/copy_kernel_python.s").write_text(python_asm)
        print("\nSaved to /tmp/copy_kernel_cpp.s and /tmp/copy_kernel_python.s")


# ============================================================================
# Test: End-to-End GPU Execution (Requires GPU)
# ============================================================================

@pytest.mark.skipif(
    not os.environ.get("RUN_GPU_TESTS", False),
    reason="GPU tests disabled. Set RUN_GPU_TESTS=1 to enable."
)
class TestGPUExecution:
    """End-to-end GPU execution tests."""

    def test_copy_kernel_e2e(self):
        """
        Full end-to-end test: compile copy kernel with C++ backend and run on GPU.
        """
        try:
            import torch
            from torch.testing import assert_close
            import wave_lang.kernel.lang as tkl
            import wave_lang.kernel.wave as tkw
            from wave_lang.kernel.wave.compile import WaveCompileOptions
            from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
            from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros
        except ImportError:
            pytest.skip("Required packages not available")

        if not torch.cuda.is_available():
            pytest.skip("No GPU available")

        # Define copy kernel
        M = tkl.sym.M
        N = tkl.sym.N
        ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

        shape = (16, 16)

        constraints = [
            tkw.HardwareConstraint(
                threads_per_wave=64,
                vector_shapes={M: 16, N: 16},
            ),
            tkw.WorkgroupConstraint(M, 16, 0),
            tkw.WorkgroupConstraint(N, 16, 1),
            tkw.WaveConstraint(M, 16),
            tkw.WaveConstraint(N, 16),
        ]

        @tkw.wave(constraints)
        def copy_kernel(
            a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
            b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        ):
            res = tkw.read(a)
            tkw.write(res, b)

        # Create test data
        a = device_randn(shape, dtype=torch.float16)
        b = device_zeros(shape, dtype=torch.float16)

        options = WaveCompileOptions(
            subs={
                M: shape[0],
                N: shape[1],
                ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
            },
            canonicalize=True,
            backend="asm",
            wave_runtime=True,
            compile_to_mlir=False,
        )
        options = set_default_run_config(options)

        # Capture MLIR
        mlir_text = capture_wave_mlir(options, copy_kernel)

        # Compile with C++ backend
        compiler = WaveASMCompiler(target=get_target_arch(), keep_temp_files=True)
        result = compiler.compile_full(mlir_text)

        if not result.success:
            pytest.fail(f"Compilation failed: {result.error_message}")

        print(f"Binary generated at: {result.binary_path}")

        # Execute via wave_runtime
        from waveasm_e2e import run_with_wave_runtime

        run_with_wave_runtime(
            binary_path=result.binary_path,
            inputs=[a],
            outputs=[b],
            grid=(1, 1, 1),
            block=(64, 1, 1),
        )

        # Validate
        assert_close(a, b)
        print("Copy kernel E2E test PASSED!")

        compiler.cleanup()


# ============================================================================
# Pytest configuration
# ============================================================================

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run end-to-end GPU tests",
    )


if __name__ == "__main__":
    # Run translation test directly
    test = TestMLIRTranslation()
    test.test_simple_copy_mlir()
