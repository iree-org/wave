#!/usr/bin/env python3
# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Compare assembly output from Python and C++ WaveASM backends.

This script:
1. Defines a simple copy kernel using wave_lang
2. Captures the MLIR IR
3. Compiles with both Python and C++ backends
4. Prints side-by-side comparison

Run with:
    python test/e2e/compare_backends.py
"""

import os
import sys
from pathlib import Path

# Add wave_lang to path
wave_root = Path(__file__).parent.parent.parent.parent
if str(wave_root) not in sys.path:
    sys.path.insert(0, str(wave_root))

# Add e2e directory for local imports
sys.path.insert(0, str(Path(__file__).parent))

from waveasm_e2e import WaveASMCompiler, capture_wave_mlir


def get_target_arch() -> str:
    """Get target architecture from environment or detect."""
    if "WAVE_DEFAULT_ARCH" in os.environ:
        arch = os.environ["WAVE_DEFAULT_ARCH"]
        return arch.split(":")[0]

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            if hasattr(props, 'gcnArchName'):
                return props.gcnArchName.split(":")[0]
    except Exception:
        pass

    return "gfx942"


def capture_copy_kernel_mlir(target: str) -> str:
    """Define a copy kernel and capture its MLIR."""
    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw
    from wave_lang.kernel.wave.compile import (
        WaveCompileOptions,
        _trace_launchable_and_get_kernel_signature,
    )
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    from wave_lang.kernel._support.indexing import IndexingContext

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

    # Must use IndexingContext for wave kernel tracing
    with IndexingContext() as idxc:
        idxc.set_subs(options.subs)

        # Initialize constraints
        copy_kernel.initialize_wave_constraints()
        copy_kernel.initialize_symbolic_constraints()
        copy_kernel.initialize_workgroup_constraints()

        # Trace to get MLIR
        # Returns: mb, graph, exe, kernel_sig, entrypoint, options, debug_info, debug_handlers, device_layout
        result = _trace_launchable_and_get_kernel_signature(
            copy_kernel, options
        )
        mb = result[0]

        # Get MLIR text
        mlir_text = mb.module_op.get_asm(
            enable_debug_info=False,
            use_local_scope=options.use_local_scope,
        )

    return mlir_text


def compile_with_python_backend(mlir_text: str, target: str) -> str:
    """Compile MLIR using Python backend."""
    try:
        from wave_lang.kernel.wave.asm.kernel_module_compiler import KernelModuleCompiler
        compiler = KernelModuleCompiler(targetid=target, codeobj="5")
        return compiler.compile_mlir_string(mlir_text)
    except Exception as e:
        return f"Python backend error: {e}"


def extract_func_from_stream(mlir_text: str) -> str:
    """
    Extract func.func from stream.executable wrapper.

    The wave kernel MLIR is wrapped in IREE stream dialect which
    our C++ backend doesn't support. This extracts the inner func.func
    and rewrites stream.binding to memref types.
    """
    import re

    # Find the func.func inside builtin.module
    # Pattern: func.func @name(...) attributes {...} { ... }
    func_match = re.search(
        r'(func\.func @\w+\([^)]*\)[^{]*\{.*?return\s*\})',
        mlir_text,
        re.DOTALL
    )

    if not func_match:
        return mlir_text

    func_body = func_match.group(1)

    # Replace !stream.binding with memref<f16>
    func_body = func_body.replace('!stream.binding', 'memref<f16>')

    # Remove stream.binding.subspan operations - they map to the argument directly
    # Replace: %0 = stream.binding.subspan %arg0[%c0] : memref<f16> -> memref<f16>
    # With nothing (the reinterpret_cast will use %arg0 directly)
    lines = func_body.split('\n')
    new_lines = []
    binding_map = {}  # map result SSA to source arg

    for line in lines:
        if 'stream.binding.subspan' in line:
            # Extract mapping: %0 = stream.binding.subspan %arg0[%c0] : ...
            match = re.match(r'\s*(%\w+)\s*=\s*stream\.binding\.subspan\s+(%arg\d+)', line)
            if match:
                binding_map[match.group(1)] = match.group(2)
            continue
        new_lines.append(line)

    func_body = '\n'.join(new_lines)

    # Replace binding results with arg references
    for result, arg in binding_map.items():
        func_body = func_body.replace(result, arg)

    # Remove iree_codegen attributes that our parser doesn't understand
    func_body = re.sub(r'attributes\s*\{[^}]*translation_info[^}]*\}', '', func_body)

    # Extract any affine_map definitions (#map = affine_map<...>)
    # The map can contain nested parens, so match until end of line
    map_defs = re.findall(r'^(#\w+\s*=\s*affine_map<.+>)\s*$', mlir_text, re.MULTILINE)
    map_section = '\n'.join(map_defs) if map_defs else ''

    # Wrap in a simple module
    simplified_mlir = f'''{map_section}
module {{
  {func_body}
}}
'''
    return simplified_mlir


def compile_with_cpp_backend(mlir_text: str, target: str) -> str:
    """Compile MLIR using C++ backend via waveasm-translate directly."""
    import subprocess
    import tempfile

    # Extract func.func from stream wrapper
    simplified_mlir = extract_func_from_stream(mlir_text)

    # Save simplified MLIR for debugging
    Path("/tmp/copy_kernel_simplified.mlir").write_text(simplified_mlir)

    # Find waveasm-translate
    script_dir = Path(__file__).parent
    waveasm_translate = script_dir.parent.parent / "build" / "tools" / "waveasm-translate" / "waveasm-translate"

    if not waveasm_translate.exists():
        return f"C++ backend error: waveasm-translate not found at {waveasm_translate}"

    # Write MLIR to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(simplified_mlir)
        mlir_file = f.name

    try:
        # Run waveasm-translate with full pipeline
        cmd = [
            str(waveasm_translate),
            f"--target={target}",
            "--waveasm-scoped-cse",
            "--waveasm-linear-scan",
            "--waveasm-insert-waitcnt",
            "--waveasm-hazard-mitigation",
            "--emit-assembly",
            mlir_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            return f"C++ backend error: {result.stderr}"

        return result.stdout

    except subprocess.TimeoutExpired:
        return "C++ backend error: Compilation timed out"
    except Exception as e:
        return f"C++ backend error: {e}"
    finally:
        Path(mlir_file).unlink(missing_ok=True)


def compare_assembly(python_asm: str, cpp_asm: str):
    """Print comparison of assembly outputs."""
    print("=" * 80)
    print("PYTHON BACKEND OUTPUT")
    print("=" * 80)
    print(python_asm)

    print("\n" + "=" * 80)
    print("C++ BACKEND OUTPUT")
    print("=" * 80)
    print(cpp_asm)

    # Basic comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    python_lines = [l.strip() for l in python_asm.split('\n') if l.strip() and not l.strip().startswith(';')]
    cpp_lines = [l.strip() for l in cpp_asm.split('\n') if l.strip() and not l.strip().startswith(';')]

    print(f"Python backend: {len(python_lines)} non-empty lines")
    print(f"C++ backend: {len(cpp_lines)} non-empty lines")

    # Find instruction differences
    python_instrs = [l for l in python_lines if not l.startswith('.') and not l.endswith(':')]
    cpp_instrs = [l for l in cpp_lines if not l.startswith('.') and not l.endswith(':')]

    print(f"\nPython instructions: {len(python_instrs)}")
    print(f"C++ instructions: {len(cpp_instrs)}")

    # Check for key instructions
    key_patterns = ['buffer_load', 'buffer_store', 'global_load', 'global_store',
                    's_endpgm', 's_waitcnt', 'v_mov', 's_mov']

    print("\nKey instruction comparison:")
    for pattern in key_patterns:
        py_count = sum(1 for l in python_instrs if pattern in l.lower())
        cpp_count = sum(1 for l in cpp_instrs if pattern in l.lower())
        if py_count > 0 or cpp_count > 0:
            match = "✓" if py_count == cpp_count else "✗"
            print(f"  {pattern}: Python={py_count}, C++={cpp_count} {match}")


def main():
    target = get_target_arch()
    print(f"Target architecture: {target}")

    # Define kernel and capture MLIR
    print("\nCapturing MLIR from copy kernel...")
    mlir_text = capture_copy_kernel_mlir(target)

    # Save MLIR for inspection
    mlir_path = Path("/tmp/copy_kernel_compare.mlir")
    mlir_path.write_text(mlir_text)
    print(f"MLIR saved to: {mlir_path}")

    print("\n" + "=" * 80)
    print("INPUT MLIR (first 50 lines)")
    print("=" * 80)
    mlir_lines = mlir_text.split('\n')[:50]
    print('\n'.join(mlir_lines))
    if len(mlir_text.split('\n')) > 50:
        print(f"... ({len(mlir_text.split(chr(10)))} total lines)")

    # Compile with both backends
    print("\n\nCompiling with Python backend...")
    python_asm = compile_with_python_backend(mlir_text, target)

    print("Compiling with C++ backend...")
    cpp_asm = compile_with_cpp_backend(mlir_text, target)

    # Save outputs
    Path("/tmp/copy_kernel_python.s").write_text(python_asm)
    Path("/tmp/copy_kernel_cpp.s").write_text(cpp_asm)
    print("Saved to /tmp/copy_kernel_python.s and /tmp/copy_kernel_cpp.s")

    # Compare
    compare_assembly(python_asm, cpp_asm)


if __name__ == "__main__":
    main()
