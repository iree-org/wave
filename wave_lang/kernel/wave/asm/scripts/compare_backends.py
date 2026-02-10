#!/usr/bin/env python3
"""
Compare LLVM, ASM, and C++ backend assembly for GEMM kernels.

This script produces side-by-side assembly artifacts for analysis of
performance differences between the three backends.

Usage:
  python compare_backends.py                           # Default failing config
  python compare_backends.py --passing                 # Working config
  python compare_backends.py --benchmark gemm-asm-benchmark  # Load from benchmark_configs.json
  python compare_backends.py --benchmark gemm-asm-benchmark --output-dir ./asm_comparison
  python compare_backends.py --benchmark gemm-asm-benchmark --cpp  # Include C++ backend

Output:
  - llvm_asm.s / asm_asm.s / cpp_asm.s: Raw assembly from each backend
  - llvm_disasm.s / asm_disasm.s / cpp_disasm.s: Disassembly from HSACO binaries
  - comparison_report.txt: Instruction counts, metrics, and analysis

Requirements:
  - llvm-objdump (from ROCm or LLVM install)
  - ROCm runtime for GPU execution
  - waveasm-translate (for C++ backend)
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.utils.run_utils import get_default_arch
from wave_lang.kernel.wave.perf.benchmark_asm_backend import (
    create_gemm_kernel,
    create_compile_options,
    create_gemm_schedule,
)
from wave_lang.kernel.wave.asm.utils import extract_func_from_stream_mlir


# =============================================================================
# C++ Backend Support
# =============================================================================


def get_waveasm_translate_path() -> Path:
    """Get path to waveasm-translate executable."""
    if "WAVEASM_TRANSLATE" in os.environ:
        return Path(os.environ["WAVEASM_TRANSLATE"])

    # Default: look in wave-asm build directory
    script_dir = Path(__file__).parent
    default_path = (
        script_dir.parent.parent.parent
        / "wave_asm"
        / "build"
        / "tools"
        / "waveasm-translate"
        / "waveasm-translate"
    )

    if default_path.exists():
        return default_path

    # Try alternative path
    alt_path = (
        script_dir.parent.parent.parent.parent.parent
        / "kernel"
        / "wave"
        / "asm"
        / "wave_asm"
        / "build"
        / "tools"
        / "waveasm-translate"
        / "waveasm-translate"
    )
    if alt_path.exists():
        return alt_path

    raise FileNotFoundError(
        "waveasm-translate not found. Set WAVEASM_TRANSLATE environment variable "
        "or build wave-asm project."
    )


def get_amdclang_path() -> str:
    """Get path to amdclang++ for assembly compilation."""
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    amdclang = os.path.join(rocm_path, "bin", "amdclang++")

    if os.path.exists(amdclang):
        return amdclang

    try:
        result = subprocess.run(["which", "amdclang++"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    raise FileNotFoundError(
        "amdclang++ not found. Ensure ROCm is installed and in PATH."
    )


# =============================================================================
# Configuration Presets
# =============================================================================

# Configuration presets (legacy)
CONFIGS = {
    # Failing config: non-square blocks (64x32) with BLOCK_K=64
    "failing": {
        "m": 128,
        "n": 64,
        "k": 64,
        "block_m": 64,
        "block_n": 32,
        "block_k": 64,
        "wave_m": 16,
        "wave_n": 16,
    },
    # Passing config: square blocks (32x32) with BLOCK_K=64
    "passing": {
        "m": 64,
        "n": 64,
        "k": 128,
        "block_m": 32,
        "block_n": 32,
        "block_k": 64,
        "wave_m": 16,
        "wave_n": 16,
    },
}

# Path to benchmark configs JSON (relative to asm/scripts/ directory)
BENCHMARK_CONFIGS_PATH = (
    Path(__file__).parent.parent.parent / "perf/benchmark_configs.json"
)


@dataclass
class BackendResult:
    """Results from compiling with a specific backend."""

    backend: str
    raw_asm: str
    hsaco_path: Optional[str] = None
    disasm: Optional[str] = None
    vgpr_count: int = 0
    sgpr_count: int = 0
    lds_size: int = 0


@dataclass
class InstructionMetrics:
    """Instruction counts by category."""

    total: int = 0
    salu: int = 0  # Scalar ALU
    valu: int = 0  # Vector ALU
    vmem: int = 0  # Vector memory (global)
    smem: int = 0  # Scalar memory
    ds: int = 0  # LDS/GDS operations
    mfma: int = 0  # Matrix operations
    branch: int = 0  # Control flow
    wait: int = 0  # Wait/sync instructions
    barrier: int = 0  # Barriers
    nop: int = 0  # NOPs (hazard mitigation)
    other: int = 0
    # Specific patterns of interest
    buffer_load_lds: int = 0
    ds_read: int = 0
    ds_write: int = 0
    m0_setup: int = 0
    readfirstlane: int = 0


def load_benchmark_config(name: str) -> dict:
    """Load a config from benchmark_configs.json by name."""
    if not BENCHMARK_CONFIGS_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark config file not found: {BENCHMARK_CONFIGS_PATH}"
        )

    with open(BENCHMARK_CONFIGS_PATH, "r") as f:
        data = json.load(f)

    for shape in data.get("gemm_shapes", []):
        if shape.get("name") == name:
            # Convert to internal format
            return {
                "m": shape["M"],
                "n": shape["N"],
                "k": shape["K"],
                "block_m": shape["BLOCK_M"],
                "block_n": shape["BLOCK_N"],
                "block_k": shape["BLOCK_K"],
                "wave_m": shape["WAVE_M"],
                "wave_n": shape["WAVE_N"],
            }

    available = [s.get("name") for s in data.get("gemm_shapes", [])]
    raise ValueError(f"Config '{name}' not found. Available: {available}")


def compile_backend(
    kernel,
    symbols,
    config: dict,
    backend: str,
    use_global_to_shared: bool = True,
    output_dir: Optional[Path] = None,
    use_schedule: bool = False,
) -> BackendResult:
    """Compile with specified backend and extract artifacts."""
    options = create_compile_options(
        symbols,
        config,
        backend,
        use_global_to_shared=use_global_to_shared,
        use_schedule=use_schedule,
        dump_intermediates=str(output_dir) if output_dir else None,
    )

    if use_schedule:
        schedule = create_gemm_schedule()
        result = wave_compile(options, kernel, schedule)
    else:
        result = wave_compile(options, kernel)

    # Extract raw assembly
    raw_asm = ""
    hsaco_path = None
    disasm = None
    rocmasm_content = None

    if backend == "asm":
        raw_asm = getattr(result, "asm", "") or ""
        hsaco_path = getattr(result, "gpu_binary_path", None)
    else:
        # LLVM backend - try to get disassembly from HSACO
        hsaco_path = getattr(result, "gpu_binary_path", None)

        # Look for .rocmasm file - check multiple locations:
        # 1. The output directory (intermediates)
        # 2. The cache directory where HSACO is stored
        search_dirs = []
        if output_dir:
            search_dirs.append(Path(output_dir))
        if hsaco_path:
            # Check the cache directory where HSACO is stored
            cache_dir = Path(hsaco_path).parent
            search_dirs.append(cache_dir)

        for search_dir in search_dirs:
            if search_dir.exists():
                rocmasm_files = list(search_dir.glob("*.rocmasm"))
                if rocmasm_files:
                    try:
                        rocmasm_content = rocmasm_files[0].read_text()
                        break
                    except Exception as e:
                        print(f"  Warning: Could not read {rocmasm_files[0]}: {e}")

    # Disassemble HSACO if available
    if hsaco_path and Path(hsaco_path).exists():
        try:
            disasm = subprocess.check_output(
                ["/opt/rocm/llvm/bin/llvm-objdump", "-d", hsaco_path],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"  Warning: Could not disassemble {hsaco_path}: {e}")

    # For LLVM backend, use disasm as raw_asm if we don't have raw_asm
    if backend == "llvm" and not raw_asm and disasm:
        raw_asm = disasm

    # Extract register/LDS metadata
    # For ASM backend: use raw_asm (has HSA metadata)
    # For LLVM backend: prefer rocmasm file (has YAML metadata), then HSACO metadata, fallback to disasm
    if backend == "asm" and raw_asm:
        vgpr_count, sgpr_count, lds_size = extract_resource_usage(raw_asm)
    elif backend == "llvm" and rocmasm_content:
        vgpr_count, sgpr_count, lds_size = extract_resource_usage(rocmasm_content)
    elif backend == "llvm" and hsaco_path:
        # Try extracting from HSACO metadata using llvm-readelf
        vgpr_count, sgpr_count, lds_size = extract_resource_usage_from_hsaco(hsaco_path)
    else:
        vgpr_count, sgpr_count, lds_size = extract_resource_usage(disasm or raw_asm)

    # For LLVM backend, prefer rocmasm content for raw_asm (used for instruction counting)
    # because disasm includes padding NOPs that aren't real instructions
    if backend == "llvm" and rocmasm_content:
        raw_asm = rocmasm_content

    return BackendResult(
        backend=backend,
        raw_asm=raw_asm,
        hsaco_path=hsaco_path,
        disasm=disasm,
        vgpr_count=vgpr_count,
        sgpr_count=sgpr_count,
        lds_size=lds_size,
    )


def extract_resource_usage(asm_text: str) -> Tuple[int, int, int]:
    """Extract VGPR, SGPR, and LDS usage from assembly/disassembly.

    Handles multiple formats:
    1. AMD HSA directives: .amdhsa_next_free_vgpr N
    2. YAML metadata (from .rocmasm): .vgpr_count:     27
    3. Generic patterns: vgpr_count = N
    """
    vgpr = 0
    sgpr = 0
    lds = 0

    if not asm_text:
        return vgpr, sgpr, lds

    # AMD HSA kernel metadata patterns
    # .amdhsa_next_free_vgpr N
    vgpr_match = re.search(r"\.amdhsa_next_free_vgpr\s+(\d+)", asm_text)
    if vgpr_match:
        vgpr = int(vgpr_match.group(1))

    # Fallback: YAML format .vgpr_count:  N (with optional spaces)
    if not vgpr:
        vgpr_match = re.search(r"\.vgpr_count:\s+(\d+)", asm_text)
        if vgpr_match:
            vgpr = int(vgpr_match.group(1))

    # Fallback: vgpr_count = N
    if not vgpr:
        vgpr_match = re.search(r"vgpr_count\s*[=:]\s*(\d+)", asm_text, re.IGNORECASE)
        if vgpr_match:
            vgpr = int(vgpr_match.group(1))

    # .amdhsa_next_free_sgpr N
    sgpr_match = re.search(r"\.amdhsa_next_free_sgpr\s+(\d+)", asm_text)
    if sgpr_match:
        sgpr = int(sgpr_match.group(1))

    # Fallback: YAML format .sgpr_count:  N
    if not sgpr:
        sgpr_match = re.search(r"\.sgpr_count:\s+(\d+)", asm_text)
        if sgpr_match:
            sgpr = int(sgpr_match.group(1))

    # Fallback: sgpr_count = N
    if not sgpr:
        sgpr_match = re.search(r"sgpr_count\s*[=:]\s*(\d+)", asm_text, re.IGNORECASE)
        if sgpr_match:
            sgpr = int(sgpr_match.group(1))

    # LDS: .amdhsa_group_segment_fixed_size N
    lds_match = re.search(r"\.amdhsa_group_segment_fixed_size\s+(\d+)", asm_text)
    if lds_match:
        lds = int(lds_match.group(1))

    # Fallback: YAML format .group_segment_fixed_size: N
    if not lds:
        lds_match = re.search(r"\.group_segment_fixed_size:\s+(\d+)", asm_text)
        if lds_match:
            lds = int(lds_match.group(1))

    # Fallback: lds_size = N
    if not lds:
        lds_match = re.search(r"lds_size\s*[=:]\s*(\d+)", asm_text, re.IGNORECASE)
        if lds_match:
            lds = int(lds_match.group(1))

    return vgpr, sgpr, lds


def extract_resource_usage_from_hsaco(hsaco_path: str) -> Tuple[int, int, int]:
    """Extract VGPR, SGPR, and LDS usage from HSACO binary using llvm-readelf."""
    vgpr = 0
    sgpr = 0
    lds = 0

    if not hsaco_path or not Path(hsaco_path).exists():
        return vgpr, sgpr, lds

    try:
        result = subprocess.run(
            ["/opt/rocm/llvm/bin/llvm-readelf", "--notes", hsaco_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            output = result.stdout
            # Parse YAML metadata from llvm-readelf output
            vgpr_match = re.search(r"\.vgpr_count:\s+(\d+)", output)
            if vgpr_match:
                vgpr = int(vgpr_match.group(1))

            sgpr_match = re.search(r"\.sgpr_count:\s+(\d+)", output)
            if sgpr_match:
                sgpr = int(sgpr_match.group(1))

            lds_match = re.search(r"\.group_segment_fixed_size:\s+(\d+)", output)
            if lds_match:
                lds = int(lds_match.group(1))
    except Exception:
        pass

    return vgpr, sgpr, lds


def compile_cpp_backend(
    mlir_text: str,
    target: str = "gfx942",
    output_dir: Optional[Path] = None,
) -> BackendResult:
    """Compile MLIR using C++ waveasm-translate backend."""
    try:
        waveasm_translate = get_waveasm_translate_path()
        amdclang = get_amdclang_path()
    except FileNotFoundError as e:
        return BackendResult(
            backend="cpp",
            raw_asm=f"C++ backend error: {e}",
            hsaco_path=None,
            disasm=None,
        )

    # Extract func from stream wrapper
    try:
        simplified_mlir = extract_func_from_stream_mlir(mlir_text)
    except Exception as e:
        return BackendResult(
            backend="cpp",
            raw_asm=f"C++ backend error: Failed to extract func: {e}",
            hsaco_path=None,
            disasm=None,
        )

    with tempfile.TemporaryDirectory(prefix="waveasm_cpp_") as temp_dir:
        temp_path = Path(temp_dir)
        mlir_file = temp_path / "input.mlir"
        asm_file = temp_path / "output.s"
        obj_file = temp_path / "kernel.o"
        hsaco_file = temp_path / "kernel.hsaco"

        mlir_file.write_text(simplified_mlir)

        # Run waveasm-translate
        cmd = [
            str(waveasm_translate),
            f"--target={target}",
            "--mlir-cse",  # Pre-translation MLIR CSE for redundant index elimination
            "--waveasm-scoped-cse",
            "--waveasm-peephole",
            "--waveasm-memory-offset-opt",  # Fold constant addresses into offset fields
            "--waveasm-linear-scan",
            "--waveasm-insert-waitcnt",
            "--waveasm-hazard-mitigation",
            "--emit-assembly",
            str(mlir_file),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                return BackendResult(
                    backend="cpp",
                    raw_asm=f"C++ backend error: {result.stderr}",
                    hsaco_path=None,
                    disasm=None,
                )
            raw_asm = result.stdout
        except subprocess.TimeoutExpired:
            return BackendResult(
                backend="cpp",
                raw_asm="C++ backend error: Compilation timed out",
                hsaco_path=None,
                disasm=None,
            )

        # Save raw assembly
        asm_file.write_text(raw_asm)

        # Assemble to HSACO
        compile_cmd = [
            amdclang,
            "-x",
            "assembler",
            "-target",
            "amdgcn-amd-amdhsa",
            "-mcode-object-version=5",
            f"-mcpu={target}",
            "-mwavefrontsize64",
            "-c",
            str(asm_file),
            "-o",
            str(obj_file),
        ]

        try:
            result = subprocess.run(
                compile_cmd, capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                # Still return the assembly even if linking fails
                vgpr, sgpr, lds = extract_resource_usage(raw_asm)
                return BackendResult(
                    backend="cpp",
                    raw_asm=raw_asm,
                    hsaco_path=None,
                    disasm=None,
                    vgpr_count=vgpr,
                    sgpr_count=sgpr,
                    lds_size=lds,
                )

            # Link to HSACO
            link_cmd = [
                amdclang,
                "-target",
                "amdgcn-amd-amdhsa",
                "-Xlinker",
                "--build-id=sha1",
                "-o",
                str(hsaco_file),
                str(obj_file),
            ]
            result = subprocess.run(
                link_cmd, capture_output=True, text=True, timeout=60
            )

            hsaco_path_str = None
            disasm = None
            if result.returncode == 0 and hsaco_file.exists():
                # Copy HSACO to output dir if specified
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    final_hsaco = output_dir / "cpp_kernel.hsaco"
                    import shutil

                    shutil.copy(hsaco_file, final_hsaco)
                    hsaco_path_str = str(final_hsaco)

                # Disassemble
                try:
                    disasm = subprocess.check_output(
                        ["/opt/rocm/llvm/bin/llvm-objdump", "-d", str(hsaco_file)],
                        text=True,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    pass

        except Exception as e:
            vgpr, sgpr, lds = extract_resource_usage(raw_asm)
            return BackendResult(
                backend="cpp",
                raw_asm=raw_asm,
                hsaco_path=None,
                disasm=None,
                vgpr_count=vgpr,
                sgpr_count=sgpr,
                lds_size=lds,
            )

        vgpr, sgpr, lds = extract_resource_usage(raw_asm)
        return BackendResult(
            backend="cpp",
            raw_asm=raw_asm,
            hsaco_path=hsaco_path_str,
            disasm=disasm,
            vgpr_count=vgpr,
            sgpr_count=sgpr,
            lds_size=lds,
        )


def classify_instruction(line: str) -> str:
    """Classify an instruction line into a category."""
    line_lower = line.lower().strip()

    # Skip non-instruction lines
    if not line_lower or line_lower.startswith(".") or line_lower.startswith(";"):
        return "skip"
    if ":" in line_lower and not any(
        op in line_lower for op in ["v_", "s_", "ds_", "buffer_", "global_"]
    ):
        return "skip"  # Label

    # MFMA
    if "v_mfma" in line_lower or "mfma" in line_lower:
        return "mfma"

    # Wait/sync
    if "s_waitcnt" in line_lower:
        return "wait"
    if "s_barrier" in line_lower:
        return "barrier"

    # NOP
    if "s_nop" in line_lower:
        return "nop"

    # Branch/control
    if any(
        x in line_lower
        for x in ["s_cbranch", "s_branch", "s_setpc", "s_call", "s_endpgm"]
    ):
        return "branch"

    # LDS operations
    if (
        line_lower.startswith("ds_")
        or "ds_read" in line_lower
        or "ds_write" in line_lower
    ):
        return "ds"

    # Vector memory
    if any(
        x in line_lower
        for x in [
            "buffer_load",
            "buffer_store",
            "global_load",
            "global_store",
            "flat_load",
            "flat_store",
        ]
    ):
        return "vmem"

    # Scalar memory
    if any(x in line_lower for x in ["s_load", "s_store", "s_buffer_load"]):
        return "smem"

    # Vector ALU - check if first word starts with v_
    if line_lower.split() and line_lower.split()[0].startswith("v_"):
        return "valu"

    # Scalar ALU - check if first word starts with s_
    if line_lower.split() and line_lower.split()[0].startswith("s_"):
        return "salu"

    return "other"


def compute_metrics(asm_text: str) -> InstructionMetrics:
    """Compute instruction metrics from assembly text."""
    metrics = InstructionMetrics()

    if not asm_text:
        return metrics

    for line in asm_text.split("\n"):
        line_lower = line.lower().strip()
        category = classify_instruction(line)

        if category == "skip":
            continue

        metrics.total += 1

        if category == "salu":
            metrics.salu += 1
        elif category == "valu":
            metrics.valu += 1
        elif category == "vmem":
            metrics.vmem += 1
        elif category == "smem":
            metrics.smem += 1
        elif category == "ds":
            metrics.ds += 1
        elif category == "mfma":
            metrics.mfma += 1
        elif category == "branch":
            metrics.branch += 1
        elif category == "wait":
            metrics.wait += 1
        elif category == "barrier":
            metrics.barrier += 1
        elif category == "nop":
            metrics.nop += 1
        else:
            metrics.other += 1

        # Specific patterns
        if "buffer_load" in line_lower and "lds" in line_lower:
            metrics.buffer_load_lds += 1
        if "ds_read" in line_lower:
            metrics.ds_read += 1
        if "ds_write" in line_lower:
            metrics.ds_write += 1
        if "s_mov_b32 m0" in line_lower or "s_mov_b32\tm0" in line_lower:
            metrics.m0_setup += 1
        if "v_readfirstlane" in line_lower:
            metrics.readfirstlane += 1

    return metrics


def format_metrics_comparison(
    llvm_metrics: InstructionMetrics,
    asm_metrics: InstructionMetrics,
    cpp_metrics: Optional[InstructionMetrics] = None,
) -> str:
    """Format a comparison table of metrics."""
    lines = []
    lines.append("=" * 90)
    lines.append("Instruction Metrics Comparison")
    lines.append("=" * 90)

    if cpp_metrics:
        lines.append(
            f"{'Category':<25} {'LLVM':>10} {'ASM':>10} {'C++':>10} {'ASM/LLVM':>10} {'C++/LLVM':>10}"
        )
        lines.append("-" * 90)

        def row(name, llvm_val, asm_val, cpp_val):
            asm_ratio = f"{asm_val/llvm_val:.2f}x" if llvm_val > 0 else "N/A"
            cpp_ratio = f"{cpp_val/llvm_val:.2f}x" if llvm_val > 0 else "N/A"
            return f"{name:<25} {llvm_val:>10} {asm_val:>10} {cpp_val:>10} {asm_ratio:>10} {cpp_ratio:>10}"

        lines.append(
            row(
                "Total Instructions",
                llvm_metrics.total,
                asm_metrics.total,
                cpp_metrics.total,
            )
        )
        lines.append("-" * 90)
        lines.append(
            row(
                "Scalar ALU (SALU)",
                llvm_metrics.salu,
                asm_metrics.salu,
                cpp_metrics.salu,
            )
        )
        lines.append(
            row(
                "Vector ALU (VALU)",
                llvm_metrics.valu,
                asm_metrics.valu,
                cpp_metrics.valu,
            )
        )
        lines.append(
            row(
                "Vector Memory (VMEM)",
                llvm_metrics.vmem,
                asm_metrics.vmem,
                cpp_metrics.vmem,
            )
        )
        lines.append(
            row(
                "Scalar Memory (SMEM)",
                llvm_metrics.smem,
                asm_metrics.smem,
                cpp_metrics.smem,
            )
        )
        lines.append(
            row("LDS/GDS (DS)", llvm_metrics.ds, asm_metrics.ds, cpp_metrics.ds)
        )
        lines.append(
            row(
                "Matrix Ops (MFMA)",
                llvm_metrics.mfma,
                asm_metrics.mfma,
                cpp_metrics.mfma,
            )
        )
        lines.append(
            row(
                "Branch/Control",
                llvm_metrics.branch,
                asm_metrics.branch,
                cpp_metrics.branch,
            )
        )
        lines.append(
            row(
                "Waits (s_waitcnt)",
                llvm_metrics.wait,
                asm_metrics.wait,
                cpp_metrics.wait,
            )
        )
        lines.append(
            row(
                "Barriers (s_barrier)",
                llvm_metrics.barrier,
                asm_metrics.barrier,
                cpp_metrics.barrier,
            )
        )
        lines.append(
            row("NOPs (s_nop)", llvm_metrics.nop, asm_metrics.nop, cpp_metrics.nop)
        )
        lines.append(
            row("Other", llvm_metrics.other, asm_metrics.other, cpp_metrics.other)
        )
        lines.append("-" * 90)
        lines.append("Specific Patterns:")
        lines.append(
            row(
                "  buffer_load...lds",
                llvm_metrics.buffer_load_lds,
                asm_metrics.buffer_load_lds,
                cpp_metrics.buffer_load_lds,
            )
        )
        lines.append(
            row(
                "  ds_read*",
                llvm_metrics.ds_read,
                asm_metrics.ds_read,
                cpp_metrics.ds_read,
            )
        )
        lines.append(
            row(
                "  ds_write*",
                llvm_metrics.ds_write,
                asm_metrics.ds_write,
                cpp_metrics.ds_write,
            )
        )
        lines.append(
            row(
                "  M0 setup (s_mov m0)",
                llvm_metrics.m0_setup,
                asm_metrics.m0_setup,
                cpp_metrics.m0_setup,
            )
        )
        lines.append(
            row(
                "  v_readfirstlane",
                llvm_metrics.readfirstlane,
                asm_metrics.readfirstlane,
                cpp_metrics.readfirstlane,
            )
        )
    else:
        lines.append(
            f"{'Category':<25} {'LLVM':>10} {'ASM':>10} {'Diff':>10} {'Ratio':>10}"
        )
        lines.append("-" * 70)

        def row(name, llvm_val, asm_val):
            diff = asm_val - llvm_val
            ratio = f"{asm_val/llvm_val:.2f}x" if llvm_val > 0 else "N/A"
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            return f"{name:<25} {llvm_val:>10} {asm_val:>10} {diff_str:>10} {ratio:>10}"

        lines.append(row("Total Instructions", llvm_metrics.total, asm_metrics.total))
        lines.append("-" * 70)
        lines.append(row("Scalar ALU (SALU)", llvm_metrics.salu, asm_metrics.salu))
        lines.append(row("Vector ALU (VALU)", llvm_metrics.valu, asm_metrics.valu))
        lines.append(row("Vector Memory (VMEM)", llvm_metrics.vmem, asm_metrics.vmem))
        lines.append(row("Scalar Memory (SMEM)", llvm_metrics.smem, asm_metrics.smem))
        lines.append(row("LDS/GDS (DS)", llvm_metrics.ds, asm_metrics.ds))
        lines.append(row("Matrix Ops (MFMA)", llvm_metrics.mfma, asm_metrics.mfma))
        lines.append(row("Branch/Control", llvm_metrics.branch, asm_metrics.branch))
        lines.append(row("Waits (s_waitcnt)", llvm_metrics.wait, asm_metrics.wait))
        lines.append(
            row("Barriers (s_barrier)", llvm_metrics.barrier, asm_metrics.barrier)
        )
        lines.append(row("NOPs (s_nop)", llvm_metrics.nop, asm_metrics.nop))
        lines.append(row("Other", llvm_metrics.other, asm_metrics.other))
        lines.append("-" * 70)
        lines.append("Specific Patterns:")
        lines.append(
            row(
                "  buffer_load...lds",
                llvm_metrics.buffer_load_lds,
                asm_metrics.buffer_load_lds,
            )
        )
        lines.append(row("  ds_read*", llvm_metrics.ds_read, asm_metrics.ds_read))
        lines.append(row("  ds_write*", llvm_metrics.ds_write, asm_metrics.ds_write))
        lines.append(
            row("  M0 setup (s_mov m0)", llvm_metrics.m0_setup, asm_metrics.m0_setup)
        )
        lines.append(
            row(
                "  v_readfirstlane",
                llvm_metrics.readfirstlane,
                asm_metrics.readfirstlane,
            )
        )
    lines.append("=" * 90)

    return "\n".join(lines)


def format_resource_comparison(
    llvm_result: BackendResult,
    asm_result: BackendResult,
    cpp_result: Optional[BackendResult] = None,
) -> str:
    """Format resource usage comparison."""
    lines = []
    lines.append("=" * 70)
    lines.append("Resource Usage Comparison")
    lines.append("=" * 70)

    if cpp_result:
        lines.append(f"{'Resource':<25} {'LLVM':>10} {'ASM':>10} {'C++':>10}")
        lines.append("-" * 70)

        def row(name, llvm_val, asm_val, cpp_val):
            return f"{name:<25} {llvm_val:>10} {asm_val:>10} {cpp_val:>10}"

        lines.append(
            row(
                "VGPRs",
                llvm_result.vgpr_count,
                asm_result.vgpr_count,
                cpp_result.vgpr_count,
            )
        )
        lines.append(
            row(
                "SGPRs",
                llvm_result.sgpr_count,
                asm_result.sgpr_count,
                cpp_result.sgpr_count,
            )
        )
        lines.append(
            row(
                "LDS (bytes)",
                llvm_result.lds_size,
                asm_result.lds_size,
                cpp_result.lds_size,
            )
        )
    else:
        lines.append(f"{'Resource':<25} {'LLVM':>10} {'ASM':>10} {'Diff':>10}")
        lines.append("-" * 70)

        def row(name, llvm_val, asm_val):
            diff = asm_val - llvm_val
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            return f"{name:<25} {llvm_val:>10} {asm_val:>10} {diff_str:>10}"

        lines.append(row("VGPRs", llvm_result.vgpr_count, asm_result.vgpr_count))
        lines.append(row("SGPRs", llvm_result.sgpr_count, asm_result.sgpr_count))
        lines.append(row("LDS (bytes)", llvm_result.lds_size, asm_result.lds_size))
    lines.append("=" * 70)

    return "\n".join(lines)


def find_context_around_pattern(
    asm_text: str,
    pattern: str,
    context_before: int = 10,
    context_after: int = 3,
    max_occurrences: int = 3,
) -> str:
    """Find lines matching pattern with surrounding context."""
    lines = asm_text.split("\n")
    results = []
    found = 0

    for i, line in enumerate(lines):
        if pattern.lower() in line.lower():
            if found >= max_occurrences:
                results.append(f"  ... and more occurrences ...")
                break
            found += 1
            results.append(f"\n--- Occurrence {found} (line {i}) ---")
            start = max(0, i - context_before)
            end = min(len(lines), i + context_after + 1)
            for j in range(start, end):
                marker = ">>>" if j == i else "   "
                results.append(f"{marker} {j:4d}: {lines[j]}")

    return "\n".join(results) if results else "  (no occurrences found)"


def generate_report(
    config: dict,
    config_name: str,
    llvm_result: BackendResult,
    asm_result: BackendResult,
    llvm_metrics: InstructionMetrics,
    asm_metrics: InstructionMetrics,
    use_global_to_shared: bool,
    use_schedule: bool = False,
    cpp_result: Optional[BackendResult] = None,
    cpp_metrics: Optional[InstructionMetrics] = None,
) -> str:
    """Generate a comprehensive comparison report."""
    lines = []

    # Header
    lines.append("=" * 80)
    title = (
        "LLVM vs ASM"
        + (" vs C++" if cpp_metrics else "")
        + " Backend Assembly Comparison Report"
    )
    lines.append(title)
    lines.append("=" * 80)
    lines.append("")

    # Configuration
    lines.append("Configuration:")
    lines.append(f"  Name: {config_name}")
    lines.append(f"  Shape: M={config['m']}, N={config['n']}, K={config['k']}")
    lines.append(
        f"  Blocks: BLOCK_M={config['block_m']}, BLOCK_N={config['block_n']}, BLOCK_K={config['block_k']}"
    )
    lines.append(f"  Waves: WAVE_M={config['wave_m']}, WAVE_N={config['wave_n']}")
    waves_m = config["block_m"] // config["wave_m"]
    waves_n = config["block_n"] // config["wave_n"]
    lines.append(f"  Waves per WG: {waves_m}x{waves_n} = {waves_m * waves_n}")
    lines.append(f"  global_to_shared: {use_global_to_shared}")
    lines.append(f"  LLVM schedule: False (always best config)")
    lines.append(f"  ASM schedule: {use_schedule}")
    if cpp_metrics:
        lines.append(f"  C++ backend: included")
    lines.append(f"  Architecture: {get_default_arch()}")
    lines.append("")

    # Metrics comparison
    lines.append(format_metrics_comparison(llvm_metrics, asm_metrics, cpp_metrics))
    lines.append("")

    # Resource comparison
    lines.append(format_resource_comparison(llvm_result, asm_result, cpp_result))
    lines.append("")

    # Analysis summary
    lines.append("=" * 70)
    lines.append("Preliminary Analysis")
    lines.append("=" * 70)

    # Compute derived metrics for ASM vs LLVM
    overhead_ratio = (
        asm_metrics.total / llvm_metrics.total if llvm_metrics.total > 0 else 0
    )
    extra_waits = asm_metrics.wait - llvm_metrics.wait
    extra_barriers = asm_metrics.barrier - llvm_metrics.barrier
    extra_nops = asm_metrics.nop - llvm_metrics.nop
    extra_salu = asm_metrics.salu - llvm_metrics.salu
    extra_valu = asm_metrics.valu - llvm_metrics.valu
    extra_m0 = asm_metrics.m0_setup - llvm_metrics.m0_setup
    extra_readfirstlane = asm_metrics.readfirstlane - llvm_metrics.readfirstlane
    extra_vgpr = asm_result.vgpr_count - llvm_result.vgpr_count

    lines.append("ASM vs LLVM:")
    lines.append(
        f"  Instruction overhead: {overhead_ratio:.2f}x ({asm_metrics.total - llvm_metrics.total:+d} instructions)"
    )

    # Identify likely causes for ASM
    causes = []
    if extra_waits > 0:
        causes.append(
            f"  - Extra waits: +{extra_waits} s_waitcnt (conservative scheduling?)"
        )
    if extra_barriers > 0:
        causes.append(f"  - Extra barriers: +{extra_barriers} s_barrier")
    if extra_nops > 0:
        causes.append(
            f"  - Extra NOPs: +{extra_nops} s_nop (hazard mitigation on gfx950?)"
        )
    if extra_salu > 10:
        causes.append(
            f"  - Extra SALU: +{extra_salu} (addressing overhead / missed CSE?)"
        )
    if extra_valu > 10:
        causes.append(f"  - Extra VALU: +{extra_valu} (index computation overhead?)")
    if extra_m0 > 0:
        causes.append(
            f"  - Extra M0 setup: +{extra_m0} (per-operation M0 vs. precomputed?)"
        )
    if extra_readfirstlane > 0:
        causes.append(
            f"  - Extra readfirstlane: +{extra_readfirstlane} (scalar extraction overhead?)"
        )
    if extra_vgpr > 0:
        causes.append(f"  - Higher VGPR usage: +{extra_vgpr} (may reduce occupancy)")

    if causes:
        lines.append("  Likely causes of overhead:")
        lines.extend(causes)
    else:
        lines.append("  No obvious overhead sources identified from static analysis.")

    # Add C++ analysis if available
    if cpp_metrics:
        lines.append("")
        lines.append("C++ vs LLVM:")
        cpp_overhead_ratio = (
            cpp_metrics.total / llvm_metrics.total if llvm_metrics.total > 0 else 0
        )
        cpp_extra_waits = cpp_metrics.wait - llvm_metrics.wait
        cpp_extra_nops = cpp_metrics.nop - llvm_metrics.nop
        cpp_extra_vgpr = (
            cpp_result.vgpr_count - llvm_result.vgpr_count if cpp_result else 0
        )

        lines.append(
            f"  Instruction overhead: {cpp_overhead_ratio:.2f}x ({cpp_metrics.total - llvm_metrics.total:+d} instructions)"
        )

        cpp_causes = []
        if cpp_extra_waits > 0:
            cpp_causes.append(f"  - Extra waits: +{cpp_extra_waits} s_waitcnt")
        if cpp_extra_nops > 0:
            cpp_causes.append(f"  - Extra NOPs: +{cpp_extra_nops} s_nop")
        if cpp_extra_vgpr > 0:
            cpp_causes.append(f"  - Higher VGPR usage: +{cpp_extra_vgpr}")

        if cpp_causes:
            lines.append("  Notable differences:")
            lines.extend(cpp_causes)

        # Compare C++ to ASM (Python)
        lines.append("")
        lines.append("C++ vs ASM (Python):")
        cpp_vs_asm_ratio = (
            cpp_metrics.total / asm_metrics.total if asm_metrics.total > 0 else 0
        )
        lines.append(
            f"  Instruction ratio: {cpp_vs_asm_ratio:.2f}x ({cpp_metrics.total - asm_metrics.total:+d} instructions)"
        )

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    # Disable cache to ensure fresh compilation for each run
    # (moved from module level to avoid side effects at import time)
    os.environ["WAVE_CACHE_ON"] = "0"

    parser = argparse.ArgumentParser(
        description="Compare LLVM and ASM backend assembly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_backends.py --benchmark gemm-asm-benchmark
  python compare_backends.py --benchmark gemm-asm-benchmark --use-schedule
  python compare_backends.py --passing --output-dir ./analysis
  python compare_backends.py --no-g2s  # Compare without gather-to-shared
""",
    )
    parser.add_argument(
        "--passing", action="store_true", help="Use passing preset config"
    )
    parser.add_argument(
        "--failing", action="store_true", help="Use failing preset config (default)"
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        type=str,
        help="Load config from benchmark_configs.json by name",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=".",
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--no-g2s", action="store_true", help="Disable global_to_shared for comparison"
    )
    parser.add_argument(
        "--use-schedule",
        action="store_true",
        help="Use manual scheduling with pipelining and instruction reordering",
    )
    parser.add_argument(
        "--show-context", action="store_true", help="Show context around key patterns"
    )
    parser.add_argument(
        "--cpp",
        action="store_true",
        help="Include C++ waveasm-translate backend in comparison",
    )
    args = parser.parse_args()

    # Determine configuration
    if args.benchmark:
        config = load_benchmark_config(args.benchmark)
        config_name = args.benchmark
    elif args.passing:
        config = CONFIGS["passing"]
        config_name = "passing"
    else:
        config = CONFIGS["failing"]
        config_name = "failing"

    use_global_to_shared = not args.no_g2s
    use_schedule = args.use_schedule
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    include_cpp = args.cpp

    # Print configuration
    print("=" * 80)
    title = (
        f"LLVM vs ASM"
        + (" vs C++" if include_cpp else "")
        + f" Backend Comparison - Config: {config_name}"
    )
    print(title)
    print("=" * 80)
    print(f"Shape: M={config['m']}, N={config['n']}, K={config['k']}")
    print(
        f"Blocks: BLOCK_M={config['block_m']}, BLOCK_N={config['block_n']}, BLOCK_K={config['block_k']}"
    )
    print(f"Waves: WAVE_M={config['wave_m']}, WAVE_N={config['wave_n']}")
    waves_m = config["block_m"] // config["wave_m"]
    waves_n = config["block_n"] // config["wave_n"]
    print(f"Waves per WG: {waves_m}x{waves_n} = {waves_m * waves_n}")
    print(f"global_to_shared: {use_global_to_shared}")
    print(f"LLVM schedule: False (always best config)")
    print(f"ASM schedule: {use_schedule}")
    print(f"Output directory: {output_dir}")

    # Compile LLVM backend (always without schedule for best performance)
    print("\n>>> Compiling LLVM backend (no schedule)...")
    kernel_llvm, symbols_llvm = create_gemm_kernel(
        config["block_m"], config["block_n"], config["wave_m"], config["wave_n"]
    )
    llvm_result = compile_backend(
        kernel_llvm,
        symbols_llvm,
        config,
        "llvm",
        use_global_to_shared=use_global_to_shared,
        output_dir=output_dir / "llvm_intermediates",
        use_schedule=False,  # LLVM always runs without schedule for fair comparison
    )
    print(f"  Raw ASM: {len(llvm_result.raw_asm)} chars")
    print(f"  HSACO: {llvm_result.hsaco_path or 'N/A'}")
    print(f"  Disasm: {len(llvm_result.disasm or '')} chars")
    print(
        f"  Resources: VGPR={llvm_result.vgpr_count}, SGPR={llvm_result.sgpr_count}, LDS={llvm_result.lds_size}"
    )

    # Compile ASM backend (with schedule if requested)
    sched_note = " (with schedule)" if use_schedule else ""
    print(f"\n>>> Compiling ASM backend{sched_note}...")
    kernel_asm, symbols_asm = create_gemm_kernel(
        config["block_m"], config["block_n"], config["wave_m"], config["wave_n"]
    )
    asm_result = compile_backend(
        kernel_asm,
        symbols_asm,
        config,
        "asm",
        use_global_to_shared=use_global_to_shared,
        output_dir=output_dir / "asm_intermediates",
        use_schedule=use_schedule,  # Only ASM uses schedule when requested
    )
    print(f"  Raw ASM: {len(asm_result.raw_asm)} chars")
    print(f"  HSACO: {asm_result.hsaco_path or 'N/A'}")
    print(f"  Disasm: {len(asm_result.disasm or '')} chars")
    print(
        f"  Resources: VGPR={asm_result.vgpr_count}, SGPR={asm_result.sgpr_count}, LDS={asm_result.lds_size}"
    )

    # Compile C++ backend if requested
    cpp_result = None
    if include_cpp:
        print("\n>>> Compiling C++ waveasm-translate backend...")
        from wave_lang.kernel.wave.compile import (
            _trace_launchable_and_get_kernel_signature,
        )
        from wave_lang.kernel._support.indexing import IndexingContext

        # Create kernel and capture MLIR using the internal trace function
        kernel_mlir, symbols_mlir = create_gemm_kernel(
            config["block_m"], config["block_n"], config["wave_m"], config["wave_n"]
        )

        cpp_intermediates = output_dir / "cpp_intermediates"
        cpp_intermediates.mkdir(parents=True, exist_ok=True)

        mlir_options = create_compile_options(
            symbols_mlir,
            config,
            "asm",  # Use ASM backend config to get the right MLIR
            use_global_to_shared=use_global_to_shared,
            use_schedule=False,
        )

        # Capture MLIR using the trace function
        mlir_text = None
        try:
            with IndexingContext() as idxc:
                idxc.set_subs(mlir_options.subs)
                kernel_mlir.initialize_wave_constraints()
                kernel_mlir.initialize_symbolic_constraints()
                kernel_mlir.initialize_workgroup_constraints()

                result = _trace_launchable_and_get_kernel_signature(
                    kernel_mlir, mlir_options
                )
                mb = result[0]
                mlir_text = mb.module_op.get_asm(enable_debug_info=False)
        except Exception as e:
            print(f"  Error capturing MLIR: {e}")

        if mlir_text:
            # Save MLIR for reference
            mlir_path = cpp_intermediates / "input.mlir"
            mlir_path.write_text(mlir_text)

            cpp_result = compile_cpp_backend(
                mlir_text,
                target=get_default_arch(),
                output_dir=cpp_intermediates,
            )
            if cpp_result.raw_asm.startswith("C++ backend error"):
                print(f"  Error: {cpp_result.raw_asm}")
            else:
                print(f"  Raw ASM: {len(cpp_result.raw_asm)} chars")
                print(f"  HSACO: {cpp_result.hsaco_path or 'N/A'}")
                print(f"  Disasm: {len(cpp_result.disasm or '')} chars")
                print(
                    f"  Resources: VGPR={cpp_result.vgpr_count}, SGPR={cpp_result.sgpr_count}, LDS={cpp_result.lds_size}"
                )
        else:
            print("  Error: Failed to generate MLIR from wave_compile")

    # Save raw assembly
    llvm_asm_path = output_dir / "llvm_asm.s"
    asm_asm_path = output_dir / "asm_asm.s"
    with open(llvm_asm_path, "w") as f:
        f.write(llvm_result.raw_asm or llvm_result.disasm or "")
    with open(asm_asm_path, "w") as f:
        f.write(asm_result.raw_asm)
    saved_files = [str(llvm_asm_path), str(asm_asm_path)]

    if cpp_result and not cpp_result.raw_asm.startswith("C++ backend error"):
        cpp_asm_path = output_dir / "cpp_asm.s"
        with open(cpp_asm_path, "w") as f:
            f.write(cpp_result.raw_asm)
        saved_files.append(str(cpp_asm_path))

    print(f"\nSaved: {', '.join(saved_files)}")

    # Save disassembly if different
    if llvm_result.disasm:
        llvm_disasm_path = output_dir / "llvm_disasm.s"
        with open(llvm_disasm_path, "w") as f:
            f.write(llvm_result.disasm)
        print(f"Saved: {llvm_disasm_path}")
    if asm_result.disasm:
        asm_disasm_path = output_dir / "asm_disasm.s"
        with open(asm_disasm_path, "w") as f:
            f.write(asm_result.disasm)
        print(f"Saved: {asm_disasm_path}")

    # Compute metrics
    print("\n>>> Computing metrics...")
    # For LLVM: prefer raw_asm (rocmasm source) over disasm because disasm includes padding NOPs
    # For ASM: use raw_asm which is the actual assembly
    llvm_text = llvm_result.raw_asm or llvm_result.disasm or ""
    asm_text = asm_result.raw_asm  # ASM backend's raw_asm is the actual assembly

    llvm_metrics = compute_metrics(llvm_text)
    asm_metrics = compute_metrics(asm_text)

    cpp_metrics = None
    if cpp_result and not cpp_result.raw_asm.startswith("C++ backend error"):
        cpp_text = cpp_result.raw_asm
        cpp_metrics = compute_metrics(cpp_text)

    # Print metrics comparison
    print("\n" + format_metrics_comparison(llvm_metrics, asm_metrics, cpp_metrics))
    print(
        "\n"
        + format_resource_comparison(
            llvm_result, asm_result, cpp_result if cpp_metrics else None
        )
    )

    # Generate and save report
    report = generate_report(
        config,
        config_name,
        llvm_result,
        asm_result,
        llvm_metrics,
        asm_metrics,
        use_global_to_shared,
        use_schedule,
        cpp_result=cpp_result if cpp_metrics else None,
        cpp_metrics=cpp_metrics,
    )
    report_path = output_dir / "comparison_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nSaved report: {report_path}")

    # Show context if requested
    if args.show_context:
        print("\n" + "=" * 80)
        print("Context: LLVM around first buffer_load...lds")
        print("=" * 80)
        print(
            find_context_around_pattern(
                llvm_text, "buffer_load", context_before=15, max_occurrences=2
            )
        )

        print("\n" + "=" * 80)
        print("Context: ASM around first buffer_load...lds")
        print("=" * 80)
        print(
            find_context_around_pattern(
                asm_text, "buffer_load", context_before=15, max_occurrences=2
            )
        )

    print("\n>>> Done!")


if __name__ == "__main__":
    main()
