#!/usr/bin/env python3
# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Compare Python and C++ WaveASM backend outputs for GEMM kernel without g2s.

Run with:
    python test/e2e/compare_gemm.py
"""

import os
import sys
import re
import subprocess
import tempfile
from pathlib import Path

# Add wave_lang to path
wave_root = Path(__file__).parent.parent.parent.parent
if str(wave_root) not in sys.path:
    sys.path.insert(0, str(wave_root))


def get_target_arch() -> str:
    """Get target architecture from environment or detect."""
    if "WAVE_DEFAULT_ARCH" in os.environ:
        arch = os.environ["WAVE_DEFAULT_ARCH"]
        return arch.split(":")[0]
    return "gfx950"


def get_waveasm_translate_path() -> Path:
    """Get path to waveasm-translate executable."""
    script_dir = Path(__file__).parent
    default_path = script_dir.parent.parent / "build" / "tools" / "waveasm-translate" / "waveasm-translate"
    if default_path.exists():
        return default_path
    raise FileNotFoundError(f"waveasm-translate not found at {default_path}")


def capture_mma_kernel_mlir(target: str, use_g2s: bool = False) -> tuple[str, str, tuple[int, int, int]]:
    """Define a multi-workgroup multi-wave MMA kernel and capture its MLIR.

    Config: 128x128 problem with 64x64 blocks = 2x2 workgroups
            64x64 block with 16x16 wave tiles = 4x4 waves per WG (16 waves total)

    Returns:
        tuple of (mlir_text, kernel_name, workgroup_size)
    """
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
    K = tkl.sym.K
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    SHARED_ADDRESS_SPACE = tkl.AddressSpace.SHARED_MEMORY.value
    GLOBAL_ADDRESS_SPACE = tkl.AddressSpace.GLOBAL_MEMORY.value

    # Multi-WG multi-wave config:
    # 64x64 block with 16x16 wave tiles = 4x4 waves per WG (16 waves)
    BLOCK_M = 64
    BLOCK_N = 64
    WAVE_M = 16
    WAVE_N = 16
    wave_size = 64

    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, WAVE_M),
        tkw.WaveConstraint(N, WAVE_N),
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        ),
    ]

    @tkw.wave(constraints)
    def mma_multi_wg_wave(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        """MMA kernel with multi-workgroup multi-wave support."""
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    # 128x128x16 problem with 64x64 blocks = 2x2 workgroups, 16 waves per WG
    options = WaveCompileOptions(
        subs={
            M: 128, N: 128, K: 16,
            LOAD_ELEMS_PER_THREAD: 4, STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
        use_global_to_shared=use_g2s,
    )
    options = set_default_run_config(options)

    with IndexingContext() as idxc:
        idxc.set_subs(options.subs)

        mma_multi_wg_wave.initialize_wave_constraints()
        mma_multi_wg_wave.initialize_symbolic_constraints()
        mma_multi_wg_wave.initialize_workgroup_constraints()

        result = _trace_launchable_and_get_kernel_signature(mma_multi_wg_wave, options)
        mb = result[0]

        mlir_text = mb.module_op.get_asm(
            enable_debug_info=False,
            use_local_scope=options.use_local_scope,
        )

        # Extract workgroup size from the launchable's hardware constraints
        hw_constraint = mma_multi_wg_wave.hardware_constraints[0]
        threads_per_block = hw_constraint.threads_per_block
        # Normalize to 3D tuple
        if len(threads_per_block) == 2:
            wg_size = (threads_per_block[0], threads_per_block[1], 1)
        elif len(threads_per_block) == 1:
            wg_size = (threads_per_block[0], 1, 1)
        else:
            wg_size = tuple(threads_per_block[:3])

    return mlir_text, "mma_multi_wg_wave", wg_size


def capture_gemm_kernel_mlir(target: str, use_g2s: bool = False) -> tuple[str, str, tuple[int, int, int]]:
    """Define a GEMM kernel with K-loop and capture its MLIR.

    Config: 64x64x128 problem with 32x32 blocks, BLOCK_K=64
            4 waves per workgroup (2x2 in M/N dimensions)
            K-loop with 2 iterations (128/64=2)

    Returns:
        tuple of (mlir_text, kernel_name, workgroup_size)
    """
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
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    SHARED_ADDRESS_SPACE = tkl.AddressSpace.SHARED_MEMORY.value
    GLOBAL_ADDRESS_SPACE = tkl.AddressSpace.GLOBAL_MEMORY.value

    wave_size = 64

    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M // 2),  # 2 waves in M dimension
        tkw.WaveConstraint(N, BLOCK_N // 2),  # 2 waves in N dimension
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        ),
    ]

    @tkw.wave(constraints)
    def gemm_k_loop(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        """GEMM kernel with K-loop."""
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    # 64x64x128 problem with 32x32 blocks, BLOCK_K=64 -> 2 K-loop iterations
    options = WaveCompileOptions(
        subs={
            M: 64, N: 64, K: 128,
            BLOCK_M: 32, BLOCK_N: 32, BLOCK_K: 64,
            LOAD_ELEMS_PER_THREAD: 4, STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
        use_global_to_shared=use_g2s,
    )
    options = set_default_run_config(options)

    with IndexingContext() as idxc:
        idxc.set_subs(options.subs)

        gemm_k_loop.initialize_wave_constraints()
        gemm_k_loop.initialize_symbolic_constraints()
        gemm_k_loop.initialize_workgroup_constraints()

        result = _trace_launchable_and_get_kernel_signature(gemm_k_loop, options)
        mb = result[0]

        mlir_text = mb.module_op.get_asm(
            enable_debug_info=False,
            use_local_scope=options.use_local_scope,
        )

        # Extract workgroup size from the launchable's hardware constraints
        hw_constraint = gemm_k_loop.hardware_constraints[0]
        threads_per_block = hw_constraint.threads_per_block
        # Normalize to 3D tuple
        if len(threads_per_block) == 2:
            wg_size = (threads_per_block[0], threads_per_block[1], 1)
        elif len(threads_per_block) == 1:
            wg_size = (threads_per_block[0], 1, 1)
        else:
            wg_size = tuple(threads_per_block[:3])

    return mlir_text, "gemm_k_loop", wg_size


def compile_with_python_backend(mlir_text: str, target: str) -> str:
    """Compile MLIR using Python backend."""
    try:
        from wave_lang.kernel.wave.asm.kernel_module_compiler import KernelModuleCompiler
        compiler = KernelModuleCompiler(targetid=target, codeobj="5")
        return compiler.compile_mlir_string(mlir_text)
    except Exception as e:
        import traceback
        return f"Python backend error: {e}\n{traceback.format_exc()}"


def extract_func_from_stream(mlir_text: str) -> str:
    """Extract func.func from stream.executable wrapper."""
    # First pass: extract arg -> memref type mappings from stream.binding.subspan
    arg_types = {}
    for match in re.finditer(
        r'stream\.binding\.subspan\s+(%arg\d+)\[[^\]]*\]\s*:\s*!stream\.binding\s*->\s*(memref<[^>]+>)',
        mlir_text
    ):
        arg_name = match.group(1)
        memref_type = match.group(2)
        arg_types[arg_name] = memref_type

    # Find the func.func inside builtin.module
    func_match = re.search(
        r'(func\.func @\w+\([^)]*\)[^{]*\{.*?return\s*\})',
        mlir_text,
        re.DOTALL
    )

    if not func_match:
        return mlir_text

    func_body = func_match.group(1)

    # Replace function signature's !stream.binding with proper types
    def replace_arg_type(m):
        arg_name = m.group(1)
        if arg_name in arg_types:
            return f"{arg_name}: {arg_types[arg_name]}"
        return m.group(0)

    func_body = re.sub(r'(%arg\d+):\s*!stream\.binding', replace_arg_type, func_body)

    # Also handle any remaining !stream.binding
    func_body = func_body.replace('!stream.binding', 'memref<f16>')

    # Remove stream.binding.subspan operations and create mapping
    lines = func_body.split('\n')
    new_lines = []
    binding_map = {}

    for line in lines:
        if 'stream.binding.subspan' in line:
            match = re.match(r'\s*(%\w+)\s*=\s*stream\.binding\.subspan\s+(%arg\d+)', line)
            if match:
                binding_map[match.group(1)] = match.group(2)
            continue
        new_lines.append(line)

    func_body = '\n'.join(new_lines)

    # Replace binding results with arg references
    for result, arg in binding_map.items():
        func_body = func_body.replace(result, arg)

    # Remove iree_codegen attributes
    func_body = re.sub(r'attributes\s*\{[^}]*translation_info[^}]*\}', '', func_body)

    # Extract affine_map definitions (use .+ to match to end of line for nested >)
    map_defs = re.findall(r'^(#\w+\s*=\s*affine_map<.+>)\s*$', mlir_text, re.MULTILINE)
    map_section = '\n'.join(map_defs) if map_defs else ''

    return f'''{map_section}
module {{
  {func_body}
}}
'''


def compile_with_cpp_backend(mlir_text: str, target: str = "gfx942",
                              wg_size: tuple[int, int, int] = None) -> str:
    """Compile MLIR using C++ backend via waveasm-translate."""
    waveasm_translate = get_waveasm_translate_path()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(mlir_text)
        mlir_file = f.name

    try:
        cmd = [
            str(waveasm_translate),
            mlir_file,
            f"--target={target}",
            "--waveasm-scoped-cse",
            "--waveasm-peephole",
            "--waveasm-linear-scan",
            "--waveasm-insert-waitcnt",
            "--waveasm-hazard-mitigation",
            "--emit-assembly",
        ]
        # Add workgroup size if specified
        if wg_size:
            cmd.extend([
                f"--workgroup-size-x={wg_size[0]}",
                f"--workgroup-size-y={wg_size[1]}",
                f"--workgroup-size-z={wg_size[2]}",
            ])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            return f"C++ backend error: {result.stderr}"

        return result.stdout

    except subprocess.TimeoutExpired:
        return "C++ backend error: Compilation timed out"
    except Exception as e:
        return f"C++ backend error: {e}"
    finally:
        Path(mlir_file).unlink(missing_ok=True)


def is_label_line(line: str) -> bool:
    """Check if line is a label (e.g., 'foo:', 'L_loop:') - not a register range like s[4:7]."""
    stripped = line.strip()
    # Labels end with : and the part before : is a valid label name (alphanumeric/_)
    if stripped.endswith(':'):
        label_name = stripped[:-1]
        return label_name.replace('_', '').replace('.', '').isalnum()
    # Also check for label at start of line like "label: instr"
    if ':' in stripped:
        before_colon = stripped.split(':')[0]
        # If before_colon is a simple identifier, it's a label
        if before_colon.replace('_', '').replace('.', '').isalnum() and not before_colon.startswith(('s[', 'v[')):
            return True
    return False


def count_instructions(asm: str) -> int:
    """Count actual instructions in assembly."""
    count = 0
    in_kernel = False
    for line in asm.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        # Skip directives and comments
        if stripped.startswith('.') or stripped.startswith('#') or stripped.startswith('//') or stripped.startswith(';'):
            continue
        # Detect kernel entry (a label line)
        if is_label_line(stripped):
            in_kernel = True
            continue
        if in_kernel:
            # Count instruction lines
            if stripped and not stripped.startswith('.'):
                count += 1
    return count


def extract_instruction_counts(asm: str) -> dict:
    """Extract counts of specific instruction types."""
    counts = {}
    in_kernel = False
    for line in asm.split('\n'):
        stripped = line.strip()
        if not stripped or stripped.startswith('.') or stripped.startswith('#') or stripped.startswith('//') or stripped.startswith(';'):
            continue
        if is_label_line(stripped):
            in_kernel = True
            continue
        if not in_kernel:
            continue
        # Get instruction mnemonic
        parts = stripped.split()
        if parts:
            mnemonic = parts[0]
            counts[mnemonic] = counts.get(mnemonic, 0) + 1
    return counts


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare Python and C++ WaveASM backends")
    parser.add_argument("--g2s", action="store_true", help="Enable global_to_shared")
    parser.add_argument("--gemm", action="store_true", help="Use GEMM with K-loop instead of simple MMA")
    parser.add_argument("--mma", action="store_true", help="Use simple MMA kernel (default)")
    args = parser.parse_args()

    use_g2s = args.g2s
    use_gemm = args.gemm

    print("="*80)
    if use_gemm:
        print("GEMM with K-loop (64x64x128, 32x32 blocks, BLOCK_K=64)")
        print("4 waves per WG (2x2 in M/N), 2 K-loop iterations")
    else:
        print("Multi-WG Multi-Wave MMA (2x2 WGs, 4x4 waves per WG = 16 waves)")
    print("="*80)

    target = get_target_arch()
    print(f"\nTarget: {target}")
    print(f"use_global_to_shared: {use_g2s}")

    print("\nCapturing MLIR...")
    if use_gemm:
        mlir_text, kernel_name, wg_size = capture_gemm_kernel_mlir(target, use_g2s=use_g2s)
    else:
        mlir_text, kernel_name, wg_size = capture_mma_kernel_mlir(target, use_g2s=use_g2s)
    Path('/tmp/gemm_full.mlir').write_text(mlir_text)
    print(f"Kernel: {kernel_name}")
    print(f"Workgroup size: {wg_size} (total threads: {wg_size[0] * wg_size[1] * wg_size[2]})")

    print("Extracting kernel MLIR...")
    try:
        simplified_mlir = extract_func_from_stream(mlir_text)
        Path('/tmp/gemm_simplified.mlir').write_text(simplified_mlir)
    except Exception as e:
        print(f"Error extracting MLIR: {e}")
        return 1

    print("Compiling with Python backend...")
    python_asm = compile_with_python_backend(mlir_text, target)
    Path('/tmp/gemm_python.s').write_text(python_asm)

    if "Python backend error" in python_asm:
        print(f"\n{python_asm}")
        return 1

    print("Compiling with C++ backend...")
    cpp_asm = compile_with_cpp_backend(simplified_mlir, target, wg_size)
    Path('/tmp/gemm_cpp.s').write_text(cpp_asm)

    if "C++ backend error" in cpp_asm:
        print(f"\n{cpp_asm}")
        return 1

    # Compare
    python_count = count_instructions(python_asm)
    cpp_count = count_instructions(cpp_asm)

    print(f"\nPython backend: {python_count} instructions")
    print(f"C++ backend: {cpp_count} instructions")

    print("\n" + "="*80)
    print("INSTRUCTION BREAKDOWN")
    print("="*80)

    python_counts = extract_instruction_counts(python_asm)
    cpp_counts = extract_instruction_counts(cpp_asm)

    all_mnemonics = sorted(set(python_counts.keys()) | set(cpp_counts.keys()))

    print(f"\n{'Instruction':<40} {'Python':>10} {'C++':>10} {'Match':>10}")
    print("-"*70)
    for mnemonic in all_mnemonics:
        py = python_counts.get(mnemonic, 0)
        cpp = cpp_counts.get(mnemonic, 0)
        match = "Y" if py == cpp else "X"
        print(f"{mnemonic:<40} {py:>10} {cpp:>10} {match:>10}")

    print("\n" + "="*80)
    print("KEY METRICS")
    print("="*80)
    key_patterns = ['v_mfma', 'buffer_load', 'buffer_store', 'ds_read', 'ds_write', 's_barrier', 's_waitcnt', 's_endpgm']
    for pattern in key_patterns:
        py = sum(v for k, v in python_counts.items() if pattern in k)
        cpp = sum(v for k, v in cpp_counts.items() if pattern in k)
        match = "Y" if py == cpp else "X"
        print(f"{pattern:<40} {py:>10} {cpp:>10} {match:>10}")

    print(f"\nAssembly saved to:")
    print(f"  Python: /tmp/gemm_python.s")
    print(f"  C++:    /tmp/gemm_cpp.s")
    print(f"  MLIR:   /tmp/gemm_simplified.mlir")

    return 0


if __name__ == "__main__":
    sys.exit(main())
