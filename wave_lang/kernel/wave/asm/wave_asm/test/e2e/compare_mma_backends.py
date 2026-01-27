#!/usr/bin/env python3
"""
Compare MMA assembly output from Python and C++ WaveASM backends.

This script:
1. Defines an MMA kernel matching asm.py test cases
2. Captures the MLIR IR
3. Compiles with both Python and C++ backends
4. Prints side-by-side comparison with detailed diff
"""

import os
import sys
import re
from pathlib import Path

# Add wave_lang to path
wave_root = Path(__file__).parent.parent.parent.parent
if str(wave_root) not in sys.path:
    sys.path.insert(0, str(wave_root))

sys.path.insert(0, str(Path(__file__).parent))


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


def capture_mma_kernel_mlir(target: str) -> tuple[str, str]:
    """Define an MMA kernel (matching asm.py test_mma) and capture its MLIR."""
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
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    SHARED_ADDRESS_SPACE = tkl.AddressSpace.SHARED_MEMORY.value
    GLOBAL_ADDRESS_SPACE = tkl.AddressSpace.GLOBAL_MEMORY.value

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        )
    ]

    @tkw.wave(constraints)
    def mma(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            M: 16,
            N: 16,
            K: 16,
            BLOCK_M: 16,
            BLOCK_N: 16,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
    )
    options = set_default_run_config(options)

    with IndexingContext() as idxc:
        idxc.set_subs(options.subs)

        mma.initialize_wave_constraints()
        mma.initialize_symbolic_constraints()
        mma.initialize_workgroup_constraints()

        result = _trace_launchable_and_get_kernel_signature(mma, options)
        mb = result[0]

        mlir_text = mb.module_op.get_asm(
            enable_debug_info=False,
            use_local_scope=options.use_local_scope,
        )

    return mlir_text, "mma"


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
    # Pattern: %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<f16>
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
    # Pattern: %arg0: !stream.binding, %arg1: !stream.binding, ...
    def replace_arg_type(m):
        arg_name = m.group(1)
        if arg_name in arg_types:
            return f"{arg_name}: {arg_types[arg_name]}"
        return m.group(0)  # Keep original if not found

    func_body = re.sub(r'(%arg\d+):\s*!stream\.binding', replace_arg_type, func_body)

    # Also handle any remaining !stream.binding (shouldn't happen but just in case)
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

    # Extract affine_map definitions
    map_defs = re.findall(r'^(#\w+\s*=\s*affine_map<.+>)\s*$', mlir_text, re.MULTILINE)
    map_section = '\n'.join(map_defs) if map_defs else ''

    simplified_mlir = f'''{map_section}
module {{
  {func_body}
}}
'''
    return simplified_mlir


def compile_with_cpp_backend(mlir_text: str, target: str) -> str:
    """Compile MLIR using C++ backend via waveasm-translate."""
    import subprocess
    import tempfile

    simplified_mlir = extract_func_from_stream(mlir_text)
    Path("/tmp/mma_kernel_simplified.mlir").write_text(simplified_mlir)

    script_dir = Path(__file__).parent
    waveasm_translate = script_dir.parent.parent / "build" / "tools" / "waveasm-translate" / "waveasm-translate"

    if not waveasm_translate.exists():
        return f"C++ backend error: waveasm-translate not found at {waveasm_translate}"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(simplified_mlir)
        mlir_file = f.name

    try:
        cmd = [
            str(waveasm_translate),
            f"--target={target}",
            "--waveasm-scoped-cse",
            "--waveasm-peephole",
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


def extract_instructions(asm: str) -> list[str]:
    """Extract instruction lines from assembly."""
    lines = []
    in_kernel = False
    for line in asm.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        # Skip comments and directives
        if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith(';'):
            continue
        # Look for kernel entry
        if stripped.endswith(':') and not stripped.startswith('.'):
            in_kernel = True
            lines.append(stripped)
            continue
        if in_kernel:
            # Skip directives
            if stripped.startswith('.'):
                in_kernel = False
                continue
            lines.append(stripped)
            if 's_endpgm' in stripped:
                break
    return lines


def compare_instructions(python_instrs: list[str], cpp_instrs: list[str]):
    """Compare instruction sequences."""
    print("\n" + "=" * 80)
    print("INSTRUCTION SEQUENCE COMPARISON")
    print("=" * 80)

    # Count instruction types
    def count_instrs(instrs):
        counts = {}
        for instr in instrs:
            parts = instr.split()
            if parts:
                mnemonic = parts[0].rstrip(':')
                counts[mnemonic] = counts.get(mnemonic, 0) + 1
        return counts

    py_counts = count_instrs(python_instrs)
    cpp_counts = count_instrs(cpp_instrs)

    all_mnemonics = sorted(set(py_counts.keys()) | set(cpp_counts.keys()))

    print("\nInstruction counts:")
    print(f"{'Instruction':<35} {'Python':>10} {'C++':>10} {'Match':>10}")
    print("-" * 65)

    differences = []
    for mnemonic in all_mnemonics:
        py_c = py_counts.get(mnemonic, 0)
        cpp_c = cpp_counts.get(mnemonic, 0)
        match = "✓" if py_c == cpp_c else "✗"
        print(f"{mnemonic:<35} {py_c:>10} {cpp_c:>10} {match:>10}")
        if py_c != cpp_c:
            differences.append((mnemonic, py_c, cpp_c))

    if differences:
        print("\n" + "=" * 80)
        print("DIFFERENCES FOUND")
        print("=" * 80)
        for mnemonic, py_c, cpp_c in differences:
            print(f"  {mnemonic}: Python has {py_c}, C++ has {cpp_c}")

    # Check for key MMA-related instructions
    key_patterns = [
        'v_mfma_f32_16x16x16_f16',
        'buffer_load',
        'buffer_store',
        'ds_write',
        'ds_read',
        's_barrier',
        's_waitcnt',
        's_endpgm',
    ]

    print("\n" + "=" * 80)
    print("KEY MMA INSTRUCTION COMPARISON")
    print("=" * 80)

    for pattern in key_patterns:
        py_count = sum(1 for i in python_instrs if pattern in i)
        cpp_count = sum(1 for i in cpp_instrs if pattern in i)
        match = "✓" if py_count == cpp_count else "✗"
        print(f"  {pattern:<35}: Python={py_count:>3}, C++={cpp_count:>3} {match}")


def print_side_by_side(python_instrs: list[str], cpp_instrs: list[str], max_lines: int = 50):
    """Print instructions side by side."""
    print("\n" + "=" * 100)
    print("SIDE-BY-SIDE INSTRUCTION COMPARISON (first {} lines)".format(max_lines))
    print("=" * 100)

    max_len = max(len(python_instrs), len(cpp_instrs))

    print(f"{'Python Backend':<50} | {'C++ Backend':<50}")
    print("-" * 101)

    for i in range(min(max_len, max_lines)):
        py_line = python_instrs[i] if i < len(python_instrs) else ""
        cpp_line = cpp_instrs[i] if i < len(cpp_instrs) else ""

        # Truncate long lines
        py_line = py_line[:48] if len(py_line) > 48 else py_line
        cpp_line = cpp_line[:48] if len(cpp_line) > 48 else cpp_line

        # Highlight differences
        if py_line and cpp_line:
            py_mnemonic = py_line.split()[0] if py_line.split() else ""
            cpp_mnemonic = cpp_line.split()[0] if cpp_line.split() else ""
            marker = "  " if py_mnemonic == cpp_mnemonic else "!!"
        else:
            marker = "!!"
        print(f"{py_line:<48} {marker}| {cpp_line:<48}")


def main():
    target = get_target_arch()
    print(f"Target architecture: {target}")

    print("\nCapturing MLIR from MMA kernel...")
    mlir_text, kernel_name = capture_mma_kernel_mlir(target)

    mlir_path = Path("/tmp/mma_kernel_compare.mlir")
    mlir_path.write_text(mlir_text)
    print(f"MLIR saved to: {mlir_path}")

    print("\n" + "=" * 80)
    print("INPUT MLIR (first 30 lines)")
    print("=" * 80)
    mlir_lines = mlir_text.split('\n')[:30]
    print('\n'.join(mlir_lines))

    print("\n\nCompiling with Python backend...")
    python_asm = compile_with_python_backend(mlir_text, target)

    print("Compiling with C++ backend...")
    cpp_asm = compile_with_cpp_backend(mlir_text, target)

    # Save outputs
    Path("/tmp/mma_kernel_python.s").write_text(python_asm)
    Path("/tmp/mma_kernel_cpp.s").write_text(cpp_asm)
    print("Saved to /tmp/mma_kernel_python.s and /tmp/mma_kernel_cpp.s")

    # Check for errors
    if "error" in python_asm.lower():
        print("\n" + "=" * 80)
        print("PYTHON BACKEND ERROR")
        print("=" * 80)
        print(python_asm)
        return

    if "error" in cpp_asm.lower():
        print("\n" + "=" * 80)
        print("C++ BACKEND ERROR")
        print("=" * 80)
        print(cpp_asm)
        return

    # Extract and compare instructions
    python_instrs = extract_instructions(python_asm)
    cpp_instrs = extract_instructions(cpp_asm)

    print(f"\nPython backend: {len(python_instrs)} instructions")
    print(f"C++ backend: {len(cpp_instrs)} instructions")

    compare_instructions(python_instrs, cpp_instrs)
    print_side_by_side(python_instrs, cpp_instrs)

    # Print full assembly
    print("\n" + "=" * 80)
    print("FULL PYTHON ASSEMBLY")
    print("=" * 80)
    print(python_asm)

    print("\n" + "=" * 80)
    print("FULL C++ ASSEMBLY")
    print("=" * 80)
    print(cpp_asm)


if __name__ == "__main__":
    main()
