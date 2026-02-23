#!/usr/bin/env python3
"""MXFP4 4-Wave Parity Benchmark: aiter vs LLVM vs C++ ASM Backend.

Compares performance (TFLOPS), assembly quality (instruction counts),
and loop structure between the hand-tuned aiter kernel and the wave
compiler backends.  Serves as the single source of truth for tracking
parity between Wave C++ ASM and aiter codegen.

Usage:
    cd /dockerx/wave
    WAVEASM_TRANSLATE=... python bench_mxfp4_comparison.py
    WAVEASM_TRANSLATE=... python bench_mxfp4_comparison.py --output-dir /tmp/mxfp4_bench
    WAVEASM_TRANSLATE=... python bench_mxfp4_comparison.py --check-correctness
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

os.environ["WAVE_CACHE_ON"] = "0"

# ============================================================
# Parity Configuration (Phase 2 -- single source of truth)
# ============================================================


@dataclass
class ParityConfig:
    """Canonical configuration bundle for parity comparisons.

    Every comparison / benchmark run should use the *same* ParityConfig so
    that performance deltas are attributable to codegen, not config drift.
    """

    M: int = 1024
    N: int = 1024
    K: int = 8192
    BLOCK_M: int = 256
    BLOCK_N: int = 256
    BLOCK_K: int = 256
    num_waves: int = 4
    # Wave tile sizes (derived: WAVE_M = BLOCK_M / sqrt(num_waves), etc.)
    wave_m: int = 128
    wave_n: int = 128
    # Schedule variant
    schedule_name: str = "aiter_style"
    use_stagger: bool = False
    # Datatype / preshuffle
    dtype: str = "mxfp4"
    preshuffle: bool = True
    # Architecture
    arch: str = "gfx950"

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.M, self.N, self.K)

    @property
    def block_shape(self) -> Tuple[int, int, int]:
        return (self.BLOCK_M, self.BLOCK_N, self.BLOCK_K)

    @property
    def waves_per_wg(self) -> int:
        return (self.BLOCK_M // self.wave_m) * (self.BLOCK_N // self.wave_n)

    @property
    def flops(self) -> int:
        return 2 * self.M * self.N * self.K

    def to_manifest(self) -> dict:
        """Serialisable manifest for reproducibility."""
        return {
            "shape": list(self.shape),
            "block_shape": list(self.block_shape),
            "num_waves": self.num_waves,
            "wave_m": self.wave_m,
            "wave_n": self.wave_n,
            "waves_per_wg": self.waves_per_wg,
            "schedule": self.schedule_name,
            "use_stagger": self.use_stagger,
            "dtype": self.dtype,
            "preshuffle": self.preshuffle,
            "arch": self.arch,
        }


# ============================================================
# Instruction metrics (reused from compare_backends.py logic)
# ============================================================


@dataclass
class InstructionMetrics:
    total: int = 0
    salu: int = 0
    valu: int = 0
    vmem: int = 0
    smem: int = 0
    ds: int = 0
    mfma: int = 0
    branch: int = 0
    wait: int = 0
    barrier: int = 0
    nop: int = 0
    other: int = 0
    buffer_load_lds: int = 0
    ds_read: int = 0
    ds_write: int = 0
    accvgpr_read: int = 0
    accvgpr_write: int = 0
    buffer_store: int = 0
    global_store: int = 0
    scratch_spill: int = 0
    m0_setup: int = 0
    readfirstlane: int = 0


@dataclass
class LoopStructure:
    """Per-loop structural analysis of assembly."""

    loop_count: int = 0
    loop_body_instructions: int = 0
    prologue_instructions: int = 0
    epilogue_instructions: int = 0
    # Per-loop breakdown
    loop_mfma: int = 0
    loop_vmem: int = 0
    loop_ds: int = 0
    loop_salu: int = 0
    loop_valu: int = 0
    loop_wait: int = 0
    loop_barrier: int = 0
    loop_nop: int = 0
    # Derived
    mfma_density: float = 0.0  # mfma / loop_body_instructions
    prefetch_loads_before_first_mfma: int = 0
    waitcnt_per_barrier: float = 0.0  # wait / barrier in loop


def classify_instruction(line: str) -> Optional[str]:
    """Classify an assembly instruction line into a category."""
    stripped = line.strip()
    if not stripped or stripped.startswith(";") or stripped.startswith("//"):
        return None
    if stripped.startswith(".") or stripped.endswith(":"):
        return None

    parts = stripped.split()
    if not parts:
        return None
    mnemonic = parts[0]

    if mnemonic.startswith("s_"):
        if "barrier" in mnemonic:
            return "barrier"
        if "waitcnt" in mnemonic:
            return "wait"
        if "nop" in mnemonic:
            return "nop"
        if "branch" in mnemonic or "jump" in mnemonic:
            return "branch"
        if "load" in mnemonic:
            return "smem"
        if "endpgm" in mnemonic or "setprio" in mnemonic:
            return "branch"
        return "salu"
    if mnemonic.startswith("v_"):
        if "mfma" in mnemonic:
            return "mfma"
        return "valu"
    if mnemonic.startswith("ds_"):
        return "ds"
    if (
        mnemonic.startswith("buffer_")
        or mnemonic.startswith("global_")
        or mnemonic.startswith("flat_")
    ):
        return "vmem"
    if mnemonic.startswith("scratch_"):
        return "vmem"
    return None


def compute_metrics(asm_text: str) -> InstructionMetrics:
    """Compute instruction metrics from assembly text."""
    m = InstructionMetrics()
    if not asm_text:
        return m

    for line in asm_text.splitlines():
        cat = classify_instruction(line)
        if cat is None:
            continue
        m.total += 1
        stripped = line.strip()

        if cat == "salu":
            m.salu += 1
        elif cat == "valu":
            m.valu += 1
            if "v_accvgpr_read" in stripped:
                m.accvgpr_read += 1
            if "v_accvgpr_write" in stripped:
                m.accvgpr_write += 1
            if "v_readfirstlane" in stripped:
                m.readfirstlane += 1
        elif cat == "vmem":
            m.vmem += 1
            if "buffer_load" in stripped and "lds" in stripped:
                m.buffer_load_lds += 1
            if "buffer_store" in stripped:
                m.buffer_store += 1
            if "global_store" in stripped:
                m.global_store += 1
            if "scratch_" in stripped:
                m.scratch_spill += 1
        elif cat == "smem":
            m.smem += 1
        elif cat == "ds":
            m.ds += 1
            if "ds_read" in stripped:
                m.ds_read += 1
            if "ds_write" in stripped:
                m.ds_write += 1
        elif cat == "mfma":
            m.mfma += 1
        elif cat == "branch":
            m.branch += 1
        elif cat == "wait":
            m.wait += 1
        elif cat == "barrier":
            m.barrier += 1
        elif cat == "nop":
            m.nop += 1
        else:
            m.other += 1

        if "s_mov_b32 m0" in stripped or "s_mov_b32\tm0" in stripped:
            m.m0_setup += 1

    return m


def analyze_loop_structure(asm_text: str) -> LoopStructure:
    """Analyse assembly for loop boundaries and per-loop instruction mix.

    Heuristic: the *main* loop is identified by a backward branch
    (``s_cbranch_scc1`` / ``s_cbranch_scc0`` targeting a label that
    appeared earlier, or any ``s_cbranch`` whose target label appears
    before the branch in the listing).  In disassembly output from
    ``llvm-objdump``, loop back-edges show up as branches to lower
    addresses (detected by hex-address comparison).

    Returns a ``LoopStructure`` summarising the dominant (largest) loop
    body and surrounding prologue / epilogue.
    """
    ls = LoopStructure()
    if not asm_text:
        return ls

    lines = asm_text.splitlines()

    # Detect whether this is llvm-objdump disassembly (hex addr lines) or
    # raw assembly (.rocmasm / C++ ASM with text labels).
    # llvm-objdump lines look like:  "  2c00: 8601FF01 ..."  or have
    # "<label>:" function headers.
    is_disasm = any(re.match(r"^\s+[0-9a-f]+:", l) for l in lines[:200])

    # ------------------------------------------------------------------
    # Unified label / branch detection
    #
    # Handles three formats:
    #   1. Raw asm labels:   "BB0_1:"  or  ".LBB0_1:"
    #   2. Objdump labels:   "000000000000355c <label_0257>:"
    #   3. Hex-address-only: no labels, branches use literal addresses
    #
    # We build a label_positions map (label_name -> line index), then
    # scan for backward branches (s_cbranch_* or s_branch targeting an
    # earlier label).
    # ------------------------------------------------------------------

    label_positions: Dict[str, int] = {}
    instruction_lines: List[Tuple[int, str, str]] = []  # (line_idx, cat, text)

    # Label patterns
    # "BB0_1:" or ".LBB0_1:" (raw asm)
    raw_label_re = re.compile(r"^(\.?[A-Za-z_]\w*):")
    # "00000000002c00 <label_0257>:" (llvm-objdump function/label headers)
    objdump_label_re = re.compile(r"^[0-9a-f]+\s+<(\w+)>:")

    for idx, raw_line in enumerate(lines):
        stripped = raw_line.strip()

        # Try to detect a label definition
        m_raw = raw_label_re.match(stripped)
        if m_raw:
            label_positions[m_raw.group(1)] = idx
            continue
        m_obj = objdump_label_re.match(stripped)
        if m_obj:
            label_positions[m_obj.group(1)] = idx
            continue

        # For disassembly, extract the instruction text after hex bytes
        if is_disasm:
            # Lines look like: "\ts_load_dwordx2 ... // HEX"
            # or              "  2c00:\t8601FF01\ts_and_b32 ..."
            parts = stripped.split("\t")
            if len(parts) >= 2:
                inst_text = parts[-1].split("//")[0].strip()
                if not inst_text:
                    # Try second-to-last (some formats put hex in middle)
                    for p in reversed(parts):
                        t = p.split("//")[0].strip()
                        if t and not re.match(r"^[0-9a-fA-F]+$", t.replace(" ", "")):
                            inst_text = t
                            break
            else:
                inst_text = stripped
        else:
            inst_text = stripped

        cat = classify_instruction(inst_text)
        if cat is not None:
            instruction_lines.append((idx, cat, inst_text))

    # Find backward branches (loop back-edges)
    # Match both s_cbranch_* and s_branch (unconditional) targeting an
    # earlier label.
    loop_ranges: List[Tuple[int, int]] = []  # (header_line, branch_line)
    for inst_idx, (line_idx, cat, text) in enumerate(instruction_lines):
        if cat == "branch" and (
            "cbranch" in text.lower() or "s_branch " in text.lower()
        ):
            # Strip trailing // comments before extracting the target
            text_no_comment = text.split("//")[0].strip()
            parts = text_no_comment.split()
            if len(parts) >= 2:
                target = parts[-1].strip()
                if target in label_positions and label_positions[target] < line_idx:
                    loop_ranges.append((label_positions[target], line_idx))

    if not loop_ranges:
        # No loops found -- everything is prologue
        ls.prologue_instructions = len(instruction_lines)
        return ls

    # Use the largest loop body
    best = max(loop_ranges, key=lambda r: r[1] - r[0])
    header, back_edge = best
    ls.loop_count = len(loop_ranges)

    # Count instructions in prologue / loop / epilogue
    for inst_idx, (line_idx, cat, text) in enumerate(instruction_lines):
        if line_idx < header:
            ls.prologue_instructions += 1
        elif line_idx <= back_edge:
            ls.loop_body_instructions += 1
            _accumulate_loop_cat(ls, cat, text)
        else:
            ls.epilogue_instructions += 1

    # Derived metrics
    if ls.loop_body_instructions > 0:
        ls.mfma_density = ls.loop_mfma / ls.loop_body_instructions
    if ls.loop_barrier > 0:
        ls.waitcnt_per_barrier = ls.loop_wait / ls.loop_barrier

    # Prefetch depth: count VMEM/DS loads before the first MFMA in loop
    if loop_ranges:
        ls.prefetch_loads_before_first_mfma = _count_prefetch_loads(
            asm_text, loop_header_line=best[0]
        )
    else:
        ls.prefetch_loads_before_first_mfma = 0

    return ls


def _accumulate_loop_cat(ls: LoopStructure, cat: str, text: str):
    """Accumulate a classified instruction into the loop structure."""
    if cat == "mfma":
        ls.loop_mfma += 1
    elif cat == "vmem":
        ls.loop_vmem += 1
    elif cat == "ds":
        ls.loop_ds += 1
    elif cat == "salu":
        ls.loop_salu += 1
    elif cat == "valu":
        ls.loop_valu += 1
    elif cat == "wait":
        ls.loop_wait += 1
    elif cat == "barrier":
        ls.loop_barrier += 1
    elif cat == "nop":
        ls.loop_nop += 1


def _count_prefetch_loads(asm_text: str, loop_header_line: int = 0) -> int:
    """Count load instructions before the first MFMA inside the main loop.

    If loop_header_line is provided, scans from that line to the first MFMA.
    Otherwise falls back to scanning from the first label.
    """
    lines = asm_text.splitlines()
    in_loop = False
    count = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if idx >= loop_header_line and loop_header_line > 0:
            in_loop = True
        elif not in_loop:
            # Detect any label as potential loop header
            if stripped.endswith(":") or re.match(r"^[0-9a-f]+\s+<\w+>:", stripped):
                in_loop = True
            continue
        if in_loop:
            cat = classify_instruction(line)
            if cat is None:
                continue
            if cat == "mfma":
                break
            if cat in ("vmem", "ds"):
                count += 1
    return count


@dataclass
class LoopSignature:
    """Detailed instruction-level signature of a loop body for diffing."""

    # Ordered list of instruction categories in the loop body
    category_sequence: List[str] = field(default_factory=list)
    # Waitcnt values encountered, ordered: [(position_in_loop, vmcnt, lgkmcnt)]
    waitcnt_sequence: List[Tuple[int, Optional[int], Optional[int]]] = field(
        default_factory=list
    )
    # Number of NOPs adjacent to each barrier (within 3 instructions)
    barrier_adjacent_nops: List[int] = field(default_factory=list)
    # MFMA-to-load interleave ratio: loads interspersed among MFMAs / total loads
    interleave_ratio: float = 0.0
    # Runs: consecutive instructions of same type [(category, count)]
    runs: List[Tuple[str, int]] = field(default_factory=list)
    # Max consecutive MFMAs without an intervening load
    max_consecutive_mfma: int = 0
    # Max consecutive loads (vmem+ds) without an intervening MFMA
    max_consecutive_loads: int = 0


def extract_loop_signature(asm_text: str) -> LoopSignature:
    """Extract a detailed loop-body signature for comparison.

    Uses the same loop-detection heuristic as analyze_loop_structure,
    then builds a fine-grained instruction sequence for diffing.
    """
    sig = LoopSignature()
    if not asm_text:
        return sig

    lines = asm_text.splitlines()

    # Detect format
    is_disasm = any(re.match(r"^\s+[0-9a-f]+:", l) for l in lines[:200])

    label_positions: Dict[str, int] = {}
    instruction_lines: List[Tuple[int, str, str]] = []

    raw_label_re = re.compile(r"^(\.?[A-Za-z_]\w*):")
    objdump_label_re = re.compile(r"^[0-9a-f]+\s+<(\w+)>:")

    for idx, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        m_raw = raw_label_re.match(stripped)
        if m_raw:
            label_positions[m_raw.group(1)] = idx
            continue
        m_obj = objdump_label_re.match(stripped)
        if m_obj:
            label_positions[m_obj.group(1)] = idx
            continue

        if is_disasm:
            parts = stripped.split("\t")
            if len(parts) >= 2:
                inst_text = parts[-1].split("//")[0].strip()
                if not inst_text:
                    for p in reversed(parts):
                        t = p.split("//")[0].strip()
                        if t and not re.match(r"^[0-9a-fA-F]+$", t.replace(" ", "")):
                            inst_text = t
                            break
            else:
                inst_text = stripped
        else:
            inst_text = stripped

        cat = classify_instruction(inst_text)
        if cat is not None:
            instruction_lines.append((idx, cat, inst_text))

    # Find largest loop
    loop_ranges: List[Tuple[int, int]] = []
    for inst_idx, (line_idx, cat, text) in enumerate(instruction_lines):
        if cat == "branch" and (
            "cbranch" in text.lower() or "s_branch " in text.lower()
        ):
            text_no_comment = text.split("//")[0].strip()
            parts = text_no_comment.split()
            if len(parts) >= 2:
                target = parts[-1].strip()
                if target in label_positions and label_positions[target] < line_idx:
                    loop_ranges.append((label_positions[target], line_idx))

    if not loop_ranges:
        return sig

    best = max(loop_ranges, key=lambda r: r[1] - r[0])
    header, back_edge = best

    # Extract loop body instructions
    loop_cats: List[str] = []
    loop_texts: List[str] = []
    for line_idx, cat, text in instruction_lines:
        if header <= line_idx <= back_edge:
            loop_cats.append(cat)
            loop_texts.append(text)

    sig.category_sequence = loop_cats

    # Extract waitcnt values
    waitcnt_re = re.compile(r"vmcnt\((\d+)\)")
    lgkmcnt_re = re.compile(r"lgkmcnt\((\d+)\)")
    for pos, (cat, text) in enumerate(zip(loop_cats, loop_texts)):
        if cat == "wait":
            vm = waitcnt_re.search(text)
            lg = lgkmcnt_re.search(text)
            vmcnt_val = int(vm.group(1)) if vm else None
            lgkmcnt_val = int(lg.group(1)) if lg else None
            sig.waitcnt_sequence.append((pos, vmcnt_val, lgkmcnt_val))

    # Barrier-adjacent NOP count (within 3 instructions after each barrier)
    for pos, cat in enumerate(loop_cats):
        if cat == "barrier":
            nop_count = 0
            for j in range(1, 4):
                if pos + j < len(loop_cats) and loop_cats[pos + j] == "nop":
                    nop_count += 1
                else:
                    break
            sig.barrier_adjacent_nops.append(nop_count)

    # Compute runs of consecutive same-category instructions
    if loop_cats:
        runs = []
        cur_cat = loop_cats[0]
        cur_count = 1
        for cat in loop_cats[1:]:
            if cat == cur_cat:
                cur_count += 1
            else:
                runs.append((cur_cat, cur_count))
                cur_cat = cat
                cur_count = 1
        runs.append((cur_cat, cur_count))
        sig.runs = runs

    # Max consecutive MFMAs and loads
    max_mfma_run = 0
    max_load_run = 0
    cur_mfma = 0
    cur_load = 0
    for cat in loop_cats:
        if cat == "mfma":
            cur_mfma += 1
            cur_load = 0
            max_mfma_run = max(max_mfma_run, cur_mfma)
        elif cat in ("vmem", "ds"):
            cur_load += 1
            cur_mfma = 0
            max_load_run = max(max_load_run, cur_load)
        else:
            cur_mfma = 0
            cur_load = 0
    sig.max_consecutive_mfma = max_mfma_run
    sig.max_consecutive_loads = max_load_run

    # Interleave ratio: count loads that appear between MFMAs
    loads_between_mfma = 0
    total_loads = sum(1 for c in loop_cats if c in ("vmem", "ds"))
    in_mfma_region = False
    for cat in loop_cats:
        if cat == "mfma":
            in_mfma_region = True
        elif cat in ("vmem", "ds"):
            if in_mfma_region:
                loads_between_mfma += 1
        elif cat not in ("nop", "salu", "wait", "barrier"):
            in_mfma_region = False
    sig.interleave_ratio = loads_between_mfma / total_loads if total_loads > 0 else 0.0

    return sig


def format_loop_signature_diff(
    name_a: str,
    sig_a: LoopSignature,
    name_b: str,
    sig_b: LoopSignature,
) -> str:
    """Format a human-readable diff between two loop signatures."""
    lines = []
    w = 80
    lines.append("=" * w)
    lines.append(f"Loop Body Signature Diff: {name_a} vs {name_b}")
    lines.append("=" * w)

    lines.append(f"{'Metric':<40s} {name_a:>18s} {name_b:>18s}")
    lines.append("-" * w)

    lines.append(
        f"{'Loop body length':<40s} {len(sig_a.category_sequence):>18d} {len(sig_b.category_sequence):>18d}"
    )
    lines.append(
        f"{'Waitcnt count':<40s} {len(sig_a.waitcnt_sequence):>18d} {len(sig_b.waitcnt_sequence):>18d}"
    )
    lines.append(
        f"{'Max consecutive MFMAs':<40s} {sig_a.max_consecutive_mfma:>18d} {sig_b.max_consecutive_mfma:>18d}"
    )
    lines.append(
        f"{'Max consecutive loads':<40s} {sig_a.max_consecutive_loads:>18d} {sig_b.max_consecutive_loads:>18d}"
    )
    lines.append(
        f"{'Load interleave ratio':<40s} {sig_a.interleave_ratio:>18.2%} {sig_b.interleave_ratio:>18.2%}"
    )
    lines.append(
        f"{'Barrier count':<40s} {sig_a.barrier_adjacent_nops.__len__():>18d} {sig_b.barrier_adjacent_nops.__len__():>18d}"
    )

    # Barrier-adjacent NOPs
    if sig_a.barrier_adjacent_nops or sig_b.barrier_adjacent_nops:
        avg_a = (
            sum(sig_a.barrier_adjacent_nops) / len(sig_a.barrier_adjacent_nops)
            if sig_a.barrier_adjacent_nops
            else 0
        )
        avg_b = (
            sum(sig_b.barrier_adjacent_nops) / len(sig_b.barrier_adjacent_nops)
            if sig_b.barrier_adjacent_nops
            else 0
        )
        lines.append(f"{'Avg NOPs after barrier':<40s} {avg_a:>18.1f} {avg_b:>18.1f}")

    lines.append("-" * w)

    # Waitcnt sequence comparison
    lines.append("Waitcnt sequences:")
    max_show = max(len(sig_a.waitcnt_sequence), len(sig_b.waitcnt_sequence))
    for i in range(min(max_show, 20)):
        a_str = ""
        b_str = ""
        if i < len(sig_a.waitcnt_sequence):
            pos, vm, lg = sig_a.waitcnt_sequence[i]
            parts = []
            if vm is not None:
                parts.append(f"vmcnt({vm})")
            if lg is not None:
                parts.append(f"lgkmcnt({lg})")
            a_str = f"@{pos}: {' '.join(parts)}"
        if i < len(sig_b.waitcnt_sequence):
            pos, vm, lg = sig_b.waitcnt_sequence[i]
            parts = []
            if vm is not None:
                parts.append(f"vmcnt({vm})")
            if lg is not None:
                parts.append(f"lgkmcnt({lg})")
            b_str = f"@{pos}: {' '.join(parts)}"
        match = " ==" if a_str == b_str and a_str else ""
        lines.append(f"  [{i:2d}] {a_str:<35s} {b_str:<35s}{match}")

    if max_show > 20:
        lines.append(f"  ... ({max_show - 20} more)")

    # Run-length summary (top 10 longest runs per side)
    lines.append("-" * w)
    lines.append("Top instruction runs (category, length):")
    top_a = sorted(sig_a.runs, key=lambda r: -r[1])[:10]
    top_b = sorted(sig_b.runs, key=lambda r: -r[1])[:10]
    for i in range(max(len(top_a), len(top_b))):
        a_str = f"{top_a[i][0]}x{top_a[i][1]}" if i < len(top_a) else ""
        b_str = f"{top_b[i][0]}x{top_b[i][1]}" if i < len(top_b) else ""
        lines.append(f"  {a_str:<35s} {b_str:<35s}")

    lines.append("=" * w)
    return "\n".join(lines)


def extract_resource_usage(asm_text: str) -> dict:
    """Extract VGPRs, SGPRs, LDS from assembly metadata."""
    resources = {"vgprs": 0, "sgprs": 0, "lds": 0, "agprs": 0}
    for line in asm_text.splitlines():
        if "amdhsa_next_free_vgpr" in line or ".vgpr_count" in line:
            nums = re.findall(r"\d+", line)
            if nums:
                resources["vgprs"] = max(resources["vgprs"], int(nums[-1]))
        if "amdhsa_next_free_sgpr" in line or ".sgpr_count" in line:
            nums = re.findall(r"\d+", line)
            if nums:
                resources["sgprs"] = max(resources["sgprs"], int(nums[-1]))
        if "group_segment_fixed_size" in line:
            nums = re.findall(r"\d+", line)
            if nums:
                resources["lds"] = max(resources["lds"], int(nums[-1]))
        if ".agpr_count" in line:
            nums = re.findall(r"\d+", line)
            if nums:
                resources["agprs"] = max(resources["agprs"], int(nums[-1]))
        if "amdhsa_accum_offset" in line:
            nums = re.findall(r"\d+", line)
            if nums:
                offset = int(nums[-1])
                if resources["vgprs"] > offset:
                    resources["agprs"] = resources["vgprs"] - offset
    return resources


def disassemble_co(co_path: str) -> str:
    """Disassemble a .co file using llvm-objdump."""
    objdump = "/opt/rocm/llvm/bin/llvm-objdump"
    if not os.path.exists(objdump):
        objdump = "llvm-objdump"
    cmd = [objdump, "-d", "--mcpu=gfx950", co_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.stdout if result.returncode == 0 else ""


def extract_co_resources(co_path: str) -> dict:
    """Extract resource usage from a .co file's ELF notes via llvm-readelf."""
    resources = {"vgprs": 0, "sgprs": 0, "lds": 0, "agprs": 0}
    readelf = "/opt/rocm/llvm/bin/llvm-readelf"
    if not os.path.exists(readelf):
        readelf = "llvm-readelf"
    cmd = [readelf, "--notes", co_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        return resources
    for line in result.stdout.splitlines():
        if ".vgpr_count:" in line:
            nums = re.findall(r"\d+", line)
            if nums:
                resources["vgprs"] = int(nums[-1])
        if ".sgpr_count:" in line:
            nums = re.findall(r"\d+", line)
            if nums:
                resources["sgprs"] = int(nums[-1])
        if ".group_segment_fixed_size:" in line:
            nums = re.findall(r"\d+", line)
            if nums:
                resources["lds"] = int(nums[-1])
        if ".wavefront_size:" in line:
            pass  # informational only
    # On gfx9, vgpr_count includes AGPRs. Infer AGPR count from accvgpr_write
    # in the disassembly (done separately). For now, if vgpr_count > 256,
    # the upper half are AGPRs.
    if resources["vgprs"] > 256:
        resources["agprs"] = resources["vgprs"] - 256
    return resources


def format_comparison(backends: dict, pcfg: Optional[ParityConfig] = None) -> str:
    """Format a multi-backend comparison table with loop structure analysis."""
    names = list(backends.keys())
    lines = []

    # Header
    w = 90 + 14 * max(0, len(names) - 2)
    lines.append("=" * w)
    lines.append("Instruction Metrics Comparison")
    lines.append("=" * w)
    hdr = f"{'Category':<30s}"
    for n in names:
        hdr += f" {n:>14s}"
    lines.append(hdr)
    lines.append("-" * w)

    metrics_list = [backends[n]["metrics"] for n in names]
    resources_list = [backends[n]["resources"] for n in names]

    def row(label, getter):
        r = f"{label:<30s}"
        for m in metrics_list:
            r += f" {getter(m):>14d}"
        return r

    lines.append(row("Total Instructions", lambda m: m.total))
    lines.append("-" * w)
    lines.append(row("Scalar ALU (SALU)", lambda m: m.salu))
    lines.append(row("Vector ALU (VALU)", lambda m: m.valu))
    lines.append(row("Vector Memory (VMEM)", lambda m: m.vmem))
    lines.append(row("Scalar Memory (SMEM)", lambda m: m.smem))
    lines.append(row("LDS/GDS (DS)", lambda m: m.ds))
    lines.append(row("Matrix Ops (MFMA)", lambda m: m.mfma))
    lines.append(row("Branch/Control", lambda m: m.branch))
    lines.append(row("Waits (s_waitcnt)", lambda m: m.wait))
    lines.append(row("Barriers (s_barrier)", lambda m: m.barrier))
    lines.append(row("NOPs (s_nop)", lambda m: m.nop))
    lines.append(row("Other", lambda m: m.other))
    lines.append("-" * w)
    lines.append(row("  buffer_load...lds", lambda m: m.buffer_load_lds))
    lines.append(row("  ds_read*", lambda m: m.ds_read))
    lines.append(row("  ds_write*", lambda m: m.ds_write))
    lines.append(row("  buffer_store*", lambda m: m.buffer_store))
    lines.append(row("  global_store*", lambda m: m.global_store))
    lines.append(row("  accvgpr_read", lambda m: m.accvgpr_read))
    lines.append(row("  accvgpr_write", lambda m: m.accvgpr_write))
    lines.append(row("  scratch spills", lambda m: m.scratch_spill))
    lines.append(row("  M0 setup", lambda m: m.m0_setup))
    lines.append(row("  v_readfirstlane", lambda m: m.readfirstlane))
    lines.append("=" * w)

    # ---- Resource Usage ----
    lines.append("")
    lines.append("=" * w)
    lines.append("Resource Usage")
    lines.append("=" * w)
    rhdr = f"{'Resource':<30s}"
    for n in names:
        rhdr += f" {n:>14s}"
    lines.append(rhdr)
    lines.append("-" * w)
    for key in ["vgprs", "sgprs", "agprs", "lds"]:
        r = f"{key.upper():<30s}"
        for res in resources_list:
            r += f" {res.get(key, 0):>14d}"
        lines.append(r)
    lines.append("=" * w)

    # ---- Loop Structure Analysis ----
    loop_list = [backends[n].get("loop_structure") for n in names]
    if any(ls is not None and ls.loop_body_instructions > 0 for ls in loop_list):
        lines.append("")
        lines.append("=" * w)
        lines.append("Loop Structure Analysis")
        lines.append("=" * w)
        lhdr = f"{'Metric':<30s}"
        for n in names:
            lhdr += f" {n:>14s}"
        lines.append(lhdr)
        lines.append("-" * w)

        def lrow(label, getter, fmt="d"):
            r = f"{label:<30s}"
            for ls in loop_list:
                if ls is None:
                    r += f" {'N/A':>14s}"
                else:
                    val = getter(ls)
                    if fmt == "d":
                        r += f" {int(val):>14d}"
                    elif fmt == ".2f":
                        r += f" {val:>14.2f}"
                    elif fmt == ".1%":
                        r += f" {val:>13.1%} "
            return r

        lines.append(lrow("Loop count", lambda ls: ls.loop_count))
        lines.append(
            lrow("Loop body instructions", lambda ls: ls.loop_body_instructions)
        )
        lines.append(lrow("Prologue instructions", lambda ls: ls.prologue_instructions))
        lines.append(lrow("Epilogue instructions", lambda ls: ls.epilogue_instructions))
        lines.append("-" * w)
        lines.append(lrow("  Loop MFMA", lambda ls: ls.loop_mfma))
        lines.append(lrow("  Loop VMEM", lambda ls: ls.loop_vmem))
        lines.append(lrow("  Loop DS", lambda ls: ls.loop_ds))
        lines.append(lrow("  Loop SALU", lambda ls: ls.loop_salu))
        lines.append(lrow("  Loop VALU", lambda ls: ls.loop_valu))
        lines.append(lrow("  Loop waits", lambda ls: ls.loop_wait))
        lines.append(lrow("  Loop barriers", lambda ls: ls.loop_barrier))
        lines.append(lrow("  Loop NOPs", lambda ls: ls.loop_nop))
        lines.append("-" * w)
        lines.append(lrow("MFMA density", lambda ls: ls.mfma_density, ".2f"))
        lines.append(
            lrow(
                "Prefetch loads before MFMA",
                lambda ls: ls.prefetch_loads_before_first_mfma,
            )
        )
        lines.append(
            lrow("waitcnt/barrier ratio", lambda ls: ls.waitcnt_per_barrier, ".2f")
        )
        lines.append("=" * w)

    # ---- Normalised per-wave metrics (requires ParityConfig) ----
    if pcfg is not None:
        wpg = pcfg.waves_per_wg
        lines.append("")
        lines.append("=" * w)
        lines.append(f"Normalised Metrics (waves_per_wg={wpg})")
        lines.append("=" * w)
        nhdr = f"{'Metric':<30s}"
        for n in names:
            nhdr += f" {n:>14s}"
        lines.append(nhdr)
        lines.append("-" * w)

        def nrow(label, getter, fmt=".1f"):
            r = f"{label:<30s}"
            for m in metrics_list:
                val = getter(m) / wpg if wpg > 0 else 0
                r += f" {val:>14{fmt}}"
            return r

        lines.append(nrow("Instructions / wave", lambda m: m.total))
        lines.append(nrow("MFMA / wave", lambda m: m.mfma))
        lines.append(nrow("VMEM / wave", lambda m: m.vmem))
        lines.append(nrow("DS / wave", lambda m: m.ds))
        lines.append(nrow("SALU / wave", lambda m: m.salu))
        lines.append(nrow("VALU / wave", lambda m: m.valu))
        lines.append("=" * w)

    return "\n".join(lines)


# ============================================================
# Regression gates (Phase 4)
# ============================================================


@dataclass
class RegressionResult:
    """Result of a regression gate check."""

    passed: bool = True
    regressions: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)


def check_regression_gates(
    baseline: dict,
    candidate: dict,
    *,
    max_vgpr_regression: int = 4,
    max_instruction_overhead: float = 1.05,
    min_mfma_density_ratio: float = 0.95,
) -> RegressionResult:
    """Compare a candidate backend result against a baseline.

    Returns RegressionResult with pass/fail and detail messages.
    The baseline is typically aiter (the reference) or a previous C++ run.
    """
    result = RegressionResult()
    b_res = baseline.get("resources", {})
    c_res = candidate.get("resources", {})
    b_met = baseline.get("metrics", InstructionMetrics())
    c_met = candidate.get("metrics", InstructionMetrics())
    b_loop = baseline.get("loop_structure", LoopStructure())
    c_loop = candidate.get("loop_structure", LoopStructure())

    # VGPR regression
    vgpr_delta = c_res.get("vgprs", 0) - b_res.get("vgprs", 0)
    if vgpr_delta > max_vgpr_regression:
        result.passed = False
        result.regressions.append(
            f"VGPR regression: +{vgpr_delta} (limit: +{max_vgpr_regression})"
        )
    elif vgpr_delta < 0:
        result.improvements.append(f"VGPR improvement: {vgpr_delta}")

    # Instruction count overhead
    if b_met.total > 0:
        ratio = c_met.total / b_met.total
        if ratio > max_instruction_overhead:
            result.passed = False
            result.regressions.append(
                f"Instruction overhead: {ratio:.2f}x (limit: {max_instruction_overhead:.2f}x)"
            )
        elif ratio < 1.0:
            result.improvements.append(f"Instruction reduction: {ratio:.2f}x")

    # MFMA density
    if b_loop and c_loop and b_loop.mfma_density > 0:
        density_ratio = c_loop.mfma_density / b_loop.mfma_density
        if density_ratio < min_mfma_density_ratio:
            result.passed = False
            result.regressions.append(
                f"MFMA density drop: {density_ratio:.2f}x (limit: {min_mfma_density_ratio:.2f}x)"
            )
        elif density_ratio > 1.0:
            result.improvements.append(f"MFMA density gain: {density_ratio:.2f}x")

    # TFLOPS regression (if present)
    b_tflops = baseline.get("tflops", 0)
    c_tflops = candidate.get("tflops", 0)
    if b_tflops > 0 and c_tflops > 0:
        perf_ratio = c_tflops / b_tflops
        if perf_ratio < 0.90:
            result.passed = False
            result.regressions.append(
                f"TFLOPS regression: {perf_ratio:.2f}x of baseline ({c_tflops:.2f} vs {b_tflops:.2f})"
            )
        elif perf_ratio > 1.0:
            result.improvements.append(
                f"TFLOPS improvement: {perf_ratio:.2f}x ({c_tflops:.2f} vs {b_tflops:.2f})"
            )

    return result


def generate_mxfp4_inputs_and_reference(shape):
    """Generate MXFP4 inputs and compute reference output for correctness checks.

    Uses the same input generation as test_asm_backend_e2e.py: properly packed
    MXFP4 data via generate_gemm_afp4wfp4_inputs and torchScaledGemmMXFP4 as
    the software reference.

    Returns:
        (x, w, x_scales, w_scales, torch_ref) - all on CPU
    """
    from wave_lang.kernel.wave.utils.mxfp_utils import (
        generate_gemm_afp4wfp4_inputs,
        torchScaledGemmMXFP4,
    )

    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_ref = torchScaledGemmMXFP4(x, w, x_scales, w_scales)
    return x, w, x_scales, w_scales, torch_ref


def check_correctness(name, output_gpu, torch_ref):
    """Check numerical correctness of a backend's output against reference.

    Uses torch.testing.assert_close with the same tolerance as the e2e test
    (default tolerances, which are strict for f32 MXFP4 GEMM).
    """
    output_cpu = output_gpu.cpu()
    ref_cpu = torch_ref.cpu() if torch_ref.is_cuda else torch_ref
    try:
        torch.testing.assert_close(ref_cpu, output_cpu, check_dtype=False)
        print(f"  Correctness: PASS")
        return True
    except AssertionError as e:
        max_err = (output_cpu - ref_cpu).abs().max().item()
        mean_err = (output_cpu - ref_cpu).abs().mean().item()
        print(f"  Correctness: FAIL (max_err={max_err:.6f}, mean_err={mean_err:.6f})")
        return False


# ============================================================
# ATT Thread Trace Analysis
# ============================================================


@dataclass
class ATTInstructionEntry:
    """One row from an ATT stats_ui CSV."""

    vaddr: int = 0
    instruction: str = ""
    hitcount: int = 0
    latency: int = 0
    stall: int = 0
    idle: int = 0
    source: str = ""


@dataclass
class ATTStallRegion:
    """A contiguous region of instructions with high stall/idle cycles."""

    start_idx: int = 0
    end_idx: int = 0
    total_stall: int = 0
    total_idle: int = 0
    total_latency: int = 0
    dominant_instruction: str = ""
    category: str = ""  # "waitcnt", "barrier", "ds_read", "vmem", etc.
    instructions: List[str] = field(default_factory=list)


@dataclass
class ATTAnalysis:
    """Summary of an ATT trace for one kernel."""

    backend_name: str = ""
    total_instructions: int = 0
    total_latency: int = 0
    total_stall: int = 0
    total_idle: int = 0
    # Per-category stall breakdowns
    stall_by_category: Dict[str, int] = field(default_factory=dict)
    idle_by_category: Dict[str, int] = field(default_factory=dict)
    latency_by_category: Dict[str, int] = field(default_factory=dict)
    # Top stall instructions
    top_stall_instructions: List[Tuple[int, str]] = field(default_factory=list)
    top_idle_instructions: List[Tuple[int, str]] = field(default_factory=list)
    # Stall regions (clusters of stalls)
    stall_regions: List[ATTStallRegion] = field(default_factory=list)
    # Efficiency metrics
    mfma_latency: int = 0
    mfma_stall: int = 0
    mfma_count: int = 0
    useful_compute_pct: float = 0.0  # mfma_latency / total_latency


def _classify_att_instruction(inst: str) -> str:
    """Classify an ATT instruction into a category for aggregation."""
    inst_lower = inst.strip().strip('"').lower()
    if inst_lower.startswith(";") or not inst_lower:
        return "comment"
    if "waitcnt" in inst_lower:
        return "waitcnt"
    if "s_barrier" in inst_lower:
        return "barrier"
    if "s_nop" in inst_lower:
        return "nop"
    if "mfma" in inst_lower:
        return "mfma"
    if "ds_read" in inst_lower or "ds_write" in inst_lower:
        return "ds"
    if "buffer_load" in inst_lower:
        if "lds" in inst_lower:
            return "buffer_load_lds"
        return "buffer_load"
    if "buffer_store" in inst_lower or "global_store" in inst_lower:
        return "store"
    if "accvgpr_read" in inst_lower:
        return "accvgpr_read"
    if "accvgpr_write" in inst_lower:
        return "accvgpr_write"
    if inst_lower.startswith("v_"):
        return "valu"
    if inst_lower.startswith("s_"):
        return "salu"
    return "other"


def parse_att_csv(csv_path: Path) -> List[ATTInstructionEntry]:
    """Parse an ATT stats_ui CSV into instruction entries."""
    import csv

    entries = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inst = row.get("Instruction", "").strip().strip('"')
            if not inst or inst.startswith(";"):
                continue
            entries.append(
                ATTInstructionEntry(
                    vaddr=int(row.get("Vaddr", 0)),
                    instruction=inst,
                    hitcount=int(row.get("Hitcount", 0)),
                    latency=int(row.get("Latency", 0)),
                    stall=int(row.get("Stall", 0)),
                    idle=int(row.get("Idle", 0)),
                    source=row.get("Source", "").strip().strip('"'),
                )
            )
    return entries


def analyze_att_trace(
    backend_name: str,
    csv_path: Path,
    top_n: int = 15,
) -> ATTAnalysis:
    """Analyze an ATT trace CSV and produce a stall analysis summary.

    Columns: CodeObj, Vaddr, Instruction, Hitcount, Latency, Stall, Idle, Source

    - Hitcount: how many times this instruction was sampled
    - Latency: total cycles spent on this instruction across all hits
    - Stall: cycles the wavefront was stalled (waiting on dependency)
    - Idle: cycles the CU was idle (no wavefronts scheduled -- pipeline bubble)
    """
    analysis = ATTAnalysis(backend_name=backend_name)
    entries = parse_att_csv(csv_path)
    if not entries:
        return analysis

    analysis.total_instructions = len(entries)

    for e in entries:
        cat = _classify_att_instruction(e.instruction)
        analysis.total_latency += e.latency
        analysis.total_stall += e.stall
        analysis.total_idle += e.idle

        # Per-category aggregation
        analysis.stall_by_category[cat] = (
            analysis.stall_by_category.get(cat, 0) + e.stall
        )
        analysis.idle_by_category[cat] = analysis.idle_by_category.get(cat, 0) + e.idle
        analysis.latency_by_category[cat] = (
            analysis.latency_by_category.get(cat, 0) + e.latency
        )

        # MFMA stats
        if cat == "mfma":
            analysis.mfma_latency += e.latency
            analysis.mfma_stall += e.stall
            analysis.mfma_count += 1

    # Useful compute percentage
    if analysis.total_latency > 0:
        analysis.useful_compute_pct = analysis.mfma_latency / analysis.total_latency

    # Top stall instructions (sorted by stall cycles)
    stall_sorted = sorted(entries, key=lambda e: e.stall, reverse=True)
    analysis.top_stall_instructions = [
        (e.stall, e.instruction) for e in stall_sorted[:top_n] if e.stall > 0
    ]

    # Top idle instructions (sorted by idle cycles)
    idle_sorted = sorted(entries, key=lambda e: e.idle, reverse=True)
    analysis.top_idle_instructions = [
        (e.idle, e.instruction) for e in idle_sorted[:top_n] if e.idle > 0
    ]

    # Find stall regions: contiguous sequences of instructions with high
    # stall or idle. A "region" starts when stall+idle > threshold and
    # ends when it drops below.
    threshold = max(100, analysis.total_stall // (len(entries) or 1))
    current_region: Optional[ATTStallRegion] = None
    for idx, e in enumerate(entries):
        cost = e.stall + e.idle
        if cost > threshold:
            if current_region is None:
                current_region = ATTStallRegion(start_idx=idx)
            current_region.end_idx = idx
            current_region.total_stall += e.stall
            current_region.total_idle += e.idle
            current_region.total_latency += e.latency
            current_region.instructions.append(e.instruction)
            if e.stall > current_region.total_stall // 2:
                current_region.dominant_instruction = e.instruction
                current_region.category = _classify_att_instruction(e.instruction)
        else:
            if current_region is not None and (
                current_region.total_stall + current_region.total_idle > threshold * 3
            ):
                if not current_region.dominant_instruction:
                    # Pick the highest-stall instruction in the region
                    max_e = max(
                        entries[current_region.start_idx : current_region.end_idx + 1],
                        key=lambda x: x.stall + x.idle,
                    )
                    current_region.dominant_instruction = max_e.instruction
                    current_region.category = _classify_att_instruction(
                        max_e.instruction
                    )
                analysis.stall_regions.append(current_region)
            current_region = None

    # Flush last region
    if current_region is not None and (
        current_region.total_stall + current_region.total_idle > threshold * 3
    ):
        if not current_region.dominant_instruction and current_region.instructions:
            current_region.dominant_instruction = current_region.instructions[0]
            current_region.category = _classify_att_instruction(
                current_region.instructions[0]
            )
        analysis.stall_regions.append(current_region)

    # Sort regions by total cost (stall + idle)
    analysis.stall_regions.sort(
        key=lambda r: r.total_stall + r.total_idle, reverse=True
    )

    return analysis


def format_att_analysis(analyses: List[ATTAnalysis]) -> str:
    """Format a side-by-side ATT stall analysis comparison."""
    names = [a.backend_name for a in analyses]
    lines = []
    w = 90 + 18 * max(0, len(names) - 2)

    lines.append("=" * w)
    lines.append("ATT Thread Trace Stall Analysis")
    lines.append("=" * w)

    # --- Overview ---
    hdr = f"{'Metric':<35s}"
    for n in names:
        hdr += f" {n:>18s}"
    lines.append(hdr)
    lines.append("-" * w)

    def row(label, getter, fmt="d"):
        r = f"{label:<35s}"
        for a in analyses:
            val = getter(a)
            if fmt == "d":
                r += f" {int(val):>18,}"
            elif fmt == ".1f":
                r += f" {val:>18.1f}"
            elif fmt == ".1%":
                r += f" {val:>17.1%} "
        return r

    lines.append(row("Total instructions traced", lambda a: a.total_instructions))
    lines.append(row("Total latency (cycles)", lambda a: a.total_latency))
    lines.append(row("Total stall (cycles)", lambda a: a.total_stall))
    lines.append(row("Total idle (cycles)", lambda a: a.total_idle))
    lines.append(row("MFMA count", lambda a: a.mfma_count))
    lines.append(row("MFMA latency (cycles)", lambda a: a.mfma_latency))
    lines.append(row("MFMA stall (cycles)", lambda a: a.mfma_stall))
    lines.append(
        row("Useful compute (MFMA/total)", lambda a: a.useful_compute_pct, ".1%")
    )
    lines.append("")

    # --- Stall breakdown by category ---
    lines.append("-" * w)
    lines.append("Stall Cycles by Instruction Category")
    lines.append("-" * w)
    all_cats = sorted(set().union(*(a.stall_by_category.keys() for a in analyses)))
    for cat in all_cats:
        r = f"  {cat:<33s}"
        for a in analyses:
            val = a.stall_by_category.get(cat, 0)
            total = a.total_stall or 1
            pct = val / total * 100
            r += f" {val:>12,} ({pct:4.1f}%)"
        lines.append(r)
    lines.append("")

    # --- Idle breakdown by category ---
    lines.append("-" * w)
    lines.append("Idle Cycles by Instruction Category")
    lines.append("-" * w)
    all_idle_cats = sorted(set().union(*(a.idle_by_category.keys() for a in analyses)))
    for cat in all_idle_cats:
        r = f"  {cat:<33s}"
        for a in analyses:
            val = a.idle_by_category.get(cat, 0)
            total = a.total_idle or 1
            pct = val / total * 100
            r += f" {val:>12,} ({pct:4.1f}%)"
        lines.append(r)
    lines.append("")

    # --- Top stall instructions (per backend) ---
    for a in analyses:
        lines.append("-" * w)
        lines.append(f"Top Stall Instructions: {a.backend_name}")
        lines.append("-" * w)
        lines.append(f"  {'Stall':>10s}  {'Instruction'}")
        for stall_val, inst in a.top_stall_instructions[:10]:
            # Truncate long instruction text
            inst_short = inst[:70] + "..." if len(inst) > 70 else inst
            lines.append(f"  {stall_val:>10,}  {inst_short}")
        lines.append("")

    # --- Top idle instructions (per backend) ---
    for a in analyses:
        lines.append("-" * w)
        lines.append(f"Top Idle Instructions: {a.backend_name}")
        lines.append("-" * w)
        lines.append(f"  {'Idle':>10s}  {'Instruction'}")
        for idle_val, inst in a.top_idle_instructions[:10]:
            inst_short = inst[:70] + "..." if len(inst) > 70 else inst
            lines.append(f"  {idle_val:>10,}  {inst_short}")
        lines.append("")

    # --- Stall regions ---
    for a in analyses:
        if a.stall_regions:
            lines.append("-" * w)
            lines.append(
                f"High-Cost Regions: {a.backend_name} "
                f"(contiguous instruction sequences with high stall+idle)"
            )
            lines.append("-" * w)
            for i, region in enumerate(a.stall_regions[:8]):
                span = region.end_idx - region.start_idx + 1
                lines.append(
                    f"  Region {i+1}: instructions [{region.start_idx}:{region.end_idx}] "
                    f"({span} instructions)"
                )
                lines.append(
                    f"    Stall={region.total_stall:,}  Idle={region.total_idle:,}  "
                    f"Latency={region.total_latency:,}"
                )
                lines.append(f"    Dominant: {region.dominant_instruction[:80]}")
                lines.append(f"    Category: {region.category}")
            lines.append("")

    lines.append("=" * w)
    return "\n".join(lines)


def find_best_att_csv(trace_dir: Path) -> Optional[Path]:
    """Find the largest (most data) stats_ui CSV in a trace directory."""
    csvs = list(trace_dir.glob("stats_ui_output_*.csv"))
    if not csvs:
        return None
    # Pick the largest file (most instruction data)
    return max(csvs, key=lambda f: f.stat().st_size)


# ============================================================
# ATT (Advanced Thread Trace) capture via rocprofv3
# ============================================================


def _find_rocprofv3(explicit_path: Optional[str] = None) -> str:
    """Locate the rocprofv3 binary."""
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path
    import shutil

    p = shutil.which("rocprofv3")
    if p:
        return p
    default = "/opt/rocm/bin/rocprofv3"
    if os.path.isfile(default):
        return default
    raise FileNotFoundError(
        "rocprofv3 not found. Install ROCm or pass --rocprofv3 /path/to/rocprofv3"
    )


def _write_att_json(
    path: Path,
    kernel_include_regex: str,
    kernel_iteration: str = "[1]",
    att_buffer_size: str = "0x60000000",
) -> None:
    """Write an ATT job config for rocprofv3."""
    cfg = {
        "jobs": [
            {
                "kernel_include_regex": kernel_include_regex,
                "kernel_exclude_regex": "",
                "kernel_iteration_range": kernel_iteration,
                "advanced_thread_trace": True,
                "att_parse": "trace",
                "att_target_cu": 0,
                "att_shader_engine_mask": "0xF",
                "att_simd_select": "0xF",
                "att_buffer_size": att_buffer_size,
            }
        ]
    }
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)


def _write_trace_script_aiter(path: Path, M: int, N: int, K: int) -> None:
    """Write a standalone Python script that runs the aiter kernel once."""
    path.write_text(
        f"""\
#!/usr/bin/env python3
\"\"\"ATT trace runner for aiter MXFP4 kernel.\"\"\"
import sys, os
os.environ["WAVE_CACHE_ON"] = "0"
# Ensure aiter is importable (installed at /dockerx/aiter)
for p in ["/dockerx/aiter", "/dockerx/wave"]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight

M, N, K = {M}, {N}, {K}
dtype = dtypes.bf16
x_fp = torch.randn((M, K), dtype=dtype, device="cuda:0")
w_fp = torch.randn((N, K), dtype=dtype, device="cuda:0")
quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
x_q, x_scales = quant_func(x_fp, shuffle=True)
w_q, w_scales = quant_func(w_fp, shuffle=True)
w_q = shuffle_weight(w_q)
del x_fp, w_fp
torch.cuda.empty_cache()

# Warmup
for _ in range(2):
    out = aiter.gemm_a4w4(x_q, w_q, x_scales, w_scales, bpreshuffle=True)
torch.cuda.synchronize()

# Traced dispatch
out = aiter.gemm_a4w4(x_q, w_q, x_scales, w_scales, bpreshuffle=True)
torch.cuda.synchronize()
print("aiter trace dispatch done")
"""
    )


def _write_trace_script_llvm(
    path: Path,
    M: int,
    N: int,
    K: int,
    block_shape: Tuple[int, int, int],
    num_waves: int,
) -> None:
    """Write a standalone Python script that compiles + runs LLVM kernel once."""
    bm, bn, bk = block_shape
    path.write_text(
        f"""\
#!/usr/bin/env python3
\"\"\"ATT trace runner for LLVM MXFP4 kernel.\"\"\"
import sys, os
os.environ["WAVE_CACHE_ON"] = "0"
for p in ["/dockerx/wave", "/dockerx/aiter"]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from wave_lang.kernel.wave.templates import get_tagged_mxfp4_gemm, get_tagged_mxfp4_gemm_preshuffle_b
from wave_lang.kernel.wave.schedules import get_mxfp4_asymmetric_schedule
from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.mxfp_utils import generate_gemm_afp4wfp4_inputs, b_preshuffle, e8m0_shuffle

M, N, K = {M}, {N}, {K}
shape = (M, N, K)
block_shape = ({bm}, {bn}, {bk})

gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(shape, block_shape, wave_shape=(1, 4)) if {num_waves} <= 4 else get_tagged_mxfp4_gemm(shape, block_shape, wave_shape=(4, 2))
schedule = get_mxfp4_asymmetric_schedule()
options = set_default_run_config(options)

print("Compiling LLVM...")
compiled = wave_compile(options, gemm, schedule)

x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
x, w = x.cuda(), w.cuda()
x_scales, w_scales = x_scales.cuda(), w_scales.cuda()
c = torch.zeros(M, N, dtype=torch.float32, device="cuda:0")
from wave_lang.kernel.wave.utils.mxfp_utils import b_preshuffle, e8m0_shuffle
w_ps = b_preshuffle(w.T.contiguous()).contiguous().cuda()
ws_ps = e8m0_shuffle(w_scales).contiguous().cuda()

# Warmup
for _ in range(2):
    c.zero_()
    compiled(x, x_scales, w_ps, ws_ps, c)
torch.cuda.synchronize()

# Traced dispatch
c.zero_()
from wave_lang.kernel.wave.utils.mxfp_utils import b_preshuffle, e8m0_shuffle
w_ps = b_preshuffle(w.T.contiguous()).contiguous().cuda()
ws_ps = e8m0_shuffle(w_scales).contiguous().cuda()
compiled(x, x_scales, w_ps, ws_ps, c)
torch.cuda.synchronize()
print("LLVM trace dispatch done")

# Explicit cleanup to help rocprofv3 teardown
del c, x, w, x_scales, w_scales, compiled
torch.cuda.empty_cache()
import gc; gc.collect()
"""
    )


def _write_trace_script_cpp(
    path: Path,
    M: int,
    N: int,
    K: int,
    block_shape: Tuple[int, int, int],
    num_waves: int,
) -> None:
    """Write a standalone Python script that compiles + runs C++ ASM kernel once."""
    bm, bn, bk = block_shape
    path.write_text(
        f"""\
#!/usr/bin/env python3
\"\"\"ATT trace runner for C++ ASM MXFP4 kernel.\"\"\"
import sys, os
os.environ["WAVE_CACHE_ON"] = "0"
for p in ["/dockerx/wave", "/dockerx/aiter",
          "/dockerx/wave/wave_lang/kernel/wave/asm/scripts",
          "/dockerx/wave/wave_lang/kernel/wave/asm/wave_asm/test/e2e"]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from wave_lang.kernel.wave.templates import get_tagged_mxfp4_gemm, get_tagged_mxfp4_gemm_preshuffle_b
from wave_lang.kernel.wave.schedules import get_mxfp4_asymmetric_schedule
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.mxfp_utils import generate_gemm_afp4wfp4_inputs, b_preshuffle, e8m0_shuffle
from compare_backends import get_default_arch
from test_asm_backend_e2e import capture_wave_kernel_info
from waveasm_e2e import WaveASMCompiler, run_with_wave_runtime

M, N, K = {M}, {N}, {K}
shape = (M, N, K)
block_shape = ({bm}, {bn}, {bk})

gemm, opts = get_tagged_mxfp4_gemm_preshuffle_b(shape, block_shape, wave_shape=(1, 4)) if {num_waves} <= 4 else get_tagged_mxfp4_gemm(shape, block_shape, wave_shape=(4, 2))
schedule = get_mxfp4_asymmetric_schedule()
opts = set_default_run_config(opts)
opts.backend = "asm"
opts.wave_runtime = True
opts.compile_to_mlir = False

print("Capturing MLIR...")
kernel_info = capture_wave_kernel_info(opts, gemm, schedule=schedule)

print("Compiling C++ backend...")
compiler = WaveASMCompiler(target=get_default_arch(), keep_temp_files=True)
cpp_result = compiler.compile_full(kernel_info.mlir_text, kernel_info.workgroup_size)
assert cpp_result.success, f"C++ compile failed: {{cpp_result.error_message[:300]}}"

x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
x, w = x.cuda(), w.cuda()
x_scales, w_scales = x_scales.cuda(), w_scales.cuda()
c = torch.zeros(M, N, dtype=torch.float32, device="cuda:0")
kernel_name = cpp_result.get_kernel_name() or kernel_info.kernel_name

# Warmup
for _ in range(2):
    c.zero_()
    run_with_wave_runtime(
        binary_path=cpp_result.binary_path,
        inputs=[x, x_scales, b_preshuffle(w.T.contiguous()).contiguous().cuda(), e8m0_shuffle(w_scales).contiguous().cuda()],
        outputs=[c],
        grid=kernel_info.grid_size,
        block=kernel_info.workgroup_size,
        shared_memory_bytes=kernel_info.lds_size,
        func_name=kernel_name,
    )
torch.cuda.synchronize()

# Traced dispatch
c.zero_()
run_with_wave_runtime(
    binary_path=cpp_result.binary_path,
    inputs=[x, x_scales, b_preshuffle(w.T.contiguous()).contiguous().cuda(), e8m0_shuffle(w_scales).contiguous().cuda()],
    outputs=[c],
    grid=kernel_info.grid_size,
    block=kernel_info.workgroup_size,
    shared_memory_bytes=kernel_info.lds_size,
    func_name=kernel_name,
)
torch.cuda.synchronize()
print("C++ ASM trace dispatch done")

# Explicit cleanup to help rocprofv3 teardown
del c, x, w, x_scales, w_scales
torch.cuda.empty_cache()
import gc; gc.collect()
"""
    )


def capture_att_trace(
    backend_name: str,
    kernel_regex: str,
    trace_script: Path,
    trace_output_dir: Path,
    rocprofv3_path: str,
    att_library_path: str,
    working_dir: str = ".",
) -> bool:
    """Run a trace script under rocprofv3 --att and capture the thread trace.

    Returns True on success, False on failure.
    """
    # Clean and create trace output directory (avoid stale files)
    import shutil

    if trace_output_dir.exists():
        shutil.rmtree(trace_output_dir)
    trace_output_dir.mkdir(parents=True, exist_ok=True)

    # Write ATT config
    att_json = trace_output_dir / "att.json"
    _write_att_json(att_json, kernel_include_regex=kernel_regex)

    cmd = [
        rocprofv3_path,
        "--att",
        "--att-activity",
        "10",
        "--att-library-path",
        att_library_path,
        "-i",
        str(att_json),
        "-d",
        str(trace_output_dir),
        "--",
        sys.executable,
        str(trace_script),
    ]

    print(f"  Running: {' '.join(cmd)}")
    env = os.environ.copy()
    env["WAVE_CACHE_ON"] = "0"
    env.setdefault("ROCPROFILER_LOG_LEVEL", "error")
    # Force all waves onto the target CU (att_target_cu=0) so the hardware
    # trace buffer is guaranteed to capture data.  Without this, kernels
    # that don't launch enough waves may never schedule on the target CU,
    # producing an empty trace CSV.
    # Syntax: GPU_ID:CU_ID  (GPU 0, CU 0 only)
    env["HSA_CU_MASK"] = "0:0"
    # Ensure Python can find wave_lang, aiter, and test helpers
    extra_paths = [
        working_dir,
        os.path.join(working_dir, "wave_lang/kernel/wave/asm/scripts"),
        os.path.join(working_dir, "wave_lang/kernel/wave/asm/wave_asm/test/e2e"),
    ]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(extra_paths + ([existing] if existing else []))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=working_dir,
            env=env,
        )

        # Check for captured trace files (may exist even if process crashed
        # during teardown -- common with PyTorch + rocprofv3 ATT)
        trace_csvs = sorted(trace_output_dir.glob("stats_ui_output_*.csv"))
        non_empty = [f for f in trace_csvs if f.stat().st_size > 200]

        if proc.returncode != 0 and not non_empty:
            print(f"  ATT trace FAILED (exit {proc.returncode})")
            if proc.stderr:
                err_lines = proc.stderr.strip().splitlines()
                for line in err_lines[-10:]:
                    print(f"    {line}")
            return False

        if proc.returncode != 0:
            print(
                f"  ATT trace: process exited {proc.returncode} "
                f"(teardown crash) but trace data was captured"
            )

        # List captured trace files
        trace_files = sorted(trace_output_dir.glob("*"))
        print(f"  ATT trace captured: {len(trace_files)} files in {trace_output_dir}")
        for tf in trace_files[:10]:
            size = tf.stat().st_size if tf.is_file() else 0
            print(f"    {tf.name} ({size:,} bytes)")
        if len(trace_files) > 10:
            print(f"    ... and {len(trace_files) - 10} more")

        if non_empty:
            print(f"  Trace CSVs with data: {len(non_empty)}")
            for f in non_empty[:5]:
                print(f"    {f.name} ({f.stat().st_size:,} bytes)")
        else:
            # Check for raw .att files (hardware trace captured but
            # decoder produced empty CSV -- common on gfx950)
            raw_att_files = list(trace_output_dir.rglob("*.att"))
            if raw_att_files:
                total_raw = sum(f.stat().st_size for f in raw_att_files)
                print(
                    f"  Raw ATT data captured: {len(raw_att_files)} .att files "
                    f"({total_raw:,} bytes total)"
                )
                print(
                    f"  NOTE: The ATT decoder produced empty CSVs but raw "
                    f"trace data is available in the output directory."
                )
                print(
                    f"  You can decode these traces with the UI tool or "
                    f"re-run the ATT decoder manually."
                )
                return True
            else:
                print(
                    f"  WARNING: No trace data captured. "
                    f"The kernel regex '{kernel_regex}' may not match "
                    f"the actual kernel symbol name."
                )
        return bool(non_empty) or bool(list(trace_output_dir.rglob("*.att")))

    except subprocess.TimeoutExpired:
        print(f"  ATT trace TIMEOUT (>600s)")
        return False
    except Exception as e:
        print(f"  ATT trace ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="MXFP4 parity benchmark: aiter vs wave"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/mxfp4_bench",
        help="Directory to dump assembly files",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--shape", type=str, default="1024,1024,8192", help="M,N,K shape"
    )
    parser.add_argument(
        "--check-correctness",
        action="store_true",
        help="Check numerical correctness against torch reference",
    )
    parser.add_argument(
        "--check-regression",
        action="store_true",
        help="Check C++ ASM against aiter regression gates",
    )
    parser.add_argument(
        "--capture-trace",
        action="store_true",
        help="Capture ATT (Advanced Thread Trace) via rocprofv3 "
        "for each backend kernel",
    )
    parser.add_argument(
        "--att-library-path",
        type=str,
        default="/dockerx/rocprof-trace-decoder-therock-7.10/"
        "releases/linux_glibc_2_28_x86_64/",
        help="Path to ATT decoder shared library directory",
    )
    parser.add_argument(
        "--rocprofv3",
        type=str,
        default=None,
        help="Path to rocprofv3 binary (auto-detected if omitted)",
    )
    args = parser.parse_args()

    M, N, K = [int(x) for x in args.shape.split(",")]
    # Build parity config from args
    pcfg = ParityConfig(M=M, N=N, K=K)
    shape = pcfg.shape
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save parity manifest (Phase 2)
    manifest = pcfg.to_manifest()
    manifest_path = out_dir / "parity_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"MXFP4 GEMM Parity Benchmark: {M}x{N}x{K}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iters}")
    print(f"Correctness check: {args.check_correctness}")
    print(f"Regression gates: {args.check_regression}")
    print(f"Parity manifest: {manifest_path}")
    print(f"Output: {out_dir}")
    print("=" * 70)

    # Generate shared MXFP4 inputs for correctness checks and preshuffle
    torch_ref = None
    mxfp4_inputs = None
    w_preshuffled = None
    w_scales_preshuffled = None
    if args.check_correctness:
        print("\n>>> Generating MXFP4 inputs and torch reference...")
        x, w, x_scales, w_scales, torch_ref = generate_mxfp4_inputs_and_reference(shape)
        mxfp4_inputs = (x, w, x_scales, w_scales)
        from wave_lang.kernel.wave.utils.mxfp_utils import b_preshuffle, e8m0_shuffle

        w_preshuffled = b_preshuffle(w.T.contiguous()).contiguous()
        w_scales_preshuffled = e8m0_shuffle(w_scales).contiguous()
        print(
            f"  Reference output: {torch_ref.shape}, "
            f"range=[{torch_ref.min().item():.4f}, {torch_ref.max().item():.4f}]"
        )

    backends = {}

    # ============================================================
    # 1. aiter
    # ============================================================
    print("\n>>> aiter (preshuffle BF16 output)...")
    try:
        import aiter
        from aiter import dtypes
        from aiter.ops.shuffle import shuffle_weight

        dtype = dtypes.bf16
        x_fp = torch.randn((M, K), dtype=dtype, device="cuda:0")
        w_fp = torch.randn((N, K), dtype=dtype, device="cuda:0")
        quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
        x_q, x_scales = quant_func(x_fp, shuffle=True)
        w_q, w_scales = quant_func(w_fp, shuffle=True)
        w_q = shuffle_weight(w_q)
        del x_fp, w_fp
        torch.cuda.empty_cache()

        for _ in range(args.warmup):
            out = aiter.gemm_a4w4(x_q, w_q, x_scales, w_scales, bpreshuffle=True)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            out = aiter.gemm_a4w4(x_q, w_q, x_scales, w_scales, bpreshuffle=True)
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000 / args.iters
        tflops = 2 * M * N * K / us / 1e6
        print(f"  {us:.1f} us, {tflops:.2f} TFLOPS")

        # Disassemble aiter .co
        # Pick the 128x128 BpreShuffle kernel (closest to 4-wave config)
        aiter_co = "/dockerx/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_128x128.co"
        aiter_asm = disassemble_co(aiter_co)
        # Extract resources from ELF notes (disassembly doesn't have metadata)
        aiter_resources = extract_co_resources(aiter_co)
        if aiter_asm:
            asm_path = out_dir / "aiter_128x128.s"
            asm_path.write_text(aiter_asm)
            print(
                f"  Assembly dumped: {asm_path} ({len(aiter_asm.splitlines())} lines)"
            )
            print(
                f"  Resources: VGPRs={aiter_resources['vgprs']}, "
                f"SGPRs={aiter_resources['sgprs']}, "
                f"AGPRs={aiter_resources['agprs']}, "
                f"LDS={aiter_resources['lds']}"
            )
            backends["aiter"] = {
                "us": us,
                "tflops": tflops,
                "metrics": compute_metrics(aiter_asm),
                "resources": aiter_resources,
                "loop_structure": analyze_loop_structure(aiter_asm),
                "asm_path": str(asm_path),
            }
        else:
            backends["aiter"] = {
                "us": us,
                "tflops": tflops,
                "metrics": InstructionMetrics(),
                "resources": aiter_resources,
                "loop_structure": LoopStructure(),
                "asm_path": None,
            }
        del x_q, w_q, x_scales, w_scales, out
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR: {e}")

    # ============================================================
    # 2. LLVM Backend (4-wave)
    # ============================================================
    print("\n>>> LLVM backend (4-wave, f32 output)...")
    try:
        from wave_lang.kernel.wave.templates import (
            get_tagged_mxfp4_gemm,
            get_tagged_mxfp4_gemm_preshuffle_b,
        )
        from wave_lang.kernel.wave.schedules import get_mxfp4_asymmetric_schedule
        from wave_lang.kernel.wave.compile import wave_compile
        from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
        from wave_lang.kernel.wave.utils.mxfp_utils import (
            generate_gemm_afp4wfp4_inputs,
            b_preshuffle,
            e8m0_shuffle,
        )

        _ws = (1, 4) if pcfg.num_waves <= 4 else (4, 2)
        if pcfg.num_waves <= 4:
            gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
                shape, pcfg.block_shape, wave_shape=_ws
            )
        else:
            gemm, options = get_tagged_mxfp4_gemm(
                shape, pcfg.block_shape, wave_shape=_ws
            )
        schedule = get_mxfp4_asymmetric_schedule()
        options = set_default_run_config(options)

        # Enable dumping intermediates to capture assembly
        llvm_dump_dir = out_dir / "llvm_intermediates"
        llvm_dump_dir.mkdir(parents=True, exist_ok=True)
        options.dump_intermediates = str(llvm_dump_dir)

        print("  Compiling...")
        compiled = wave_compile(options, gemm, schedule)

        # Capture LLVM assembly from the compiled result or dump directory
        llvm_asm = ""
        llvm_resources = {}
        llvm_loop = LoopStructure()

        # Try to get raw assembly from .rocmasm file in intermediates
        rocmasm_files = list(llvm_dump_dir.glob("*.rocmasm"))
        if rocmasm_files:
            llvm_asm = rocmasm_files[0].read_text()
            print(
                f"  Got LLVM assembly from .rocmasm ({len(llvm_asm.splitlines())} lines)"
            )
        else:
            # Fallback: disassemble the HSACO binary
            hsaco_path = getattr(compiled, "gpu_binary_path", None)
            if hsaco_path and Path(hsaco_path).exists():
                llvm_asm = disassemble_co(hsaco_path)
                print(
                    f"  Got LLVM assembly from HSACO disassembly ({len(llvm_asm.splitlines())} lines)"
                )

        if llvm_asm:
            llvm_asm_path = out_dir / "llvm_4wave.s"
            llvm_asm_path.write_text(llvm_asm)
            llvm_resources = extract_resource_usage(llvm_asm)
            # Also try ELF notes for more accurate data
            hsaco_path = getattr(compiled, "gpu_binary_path", None)
            if hsaco_path and Path(hsaco_path).exists():
                elf_res = extract_co_resources(hsaco_path)
                if elf_res.get("vgprs", 0) > 0:
                    llvm_resources = elf_res
            llvm_loop = analyze_loop_structure(llvm_asm)
            print(
                f"  VGPRs={llvm_resources.get('vgprs', 0)}, "
                f"SGPRs={llvm_resources.get('sgprs', 0)}, "
                f"AGPRs={llvm_resources.get('agprs', 0)}"
            )

        # Use shared inputs for correctness, or generate fresh for benchmark
        if mxfp4_inputs is not None:
            x, w, x_scales, w_scales = mxfp4_inputs
            x, w = x.cuda(), w.cuda()
            x_scales, w_scales = x_scales.cuda(), w_scales.cuda()
        else:
            x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
            x, w = x.cuda(), w.cuda()
            x_scales, w_scales = x_scales.cuda(), w_scales.cuda()
        c = torch.zeros(M, N, dtype=torch.float32, device="cuda:0")

        if w_preshuffled is None:
            from wave_lang.kernel.wave.utils.mxfp_utils import (
                b_preshuffle,
                e8m0_shuffle,
            )

            w_preshuffled = b_preshuffle(w.T.contiguous()).contiguous()
            w_scales_preshuffled = e8m0_shuffle(w_scales).contiguous()

        for _ in range(args.warmup):
            c.zero_()
            compiled(x, x_scales, w_preshuffled.cuda(), w_scales_preshuffled.cuda(), c)
        torch.cuda.synchronize()

        # Correctness check (after warmup, before timed region)
        if args.check_correctness and torch_ref is not None:
            c.zero_()
            compiled(x, x_scales, w_preshuffled.cuda(), w_scales_preshuffled.cuda(), c)
            torch.cuda.synchronize()
            check_correctness("LLVM 4-wave", c, torch_ref)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            c.zero_()
            compiled(x, x_scales, w_preshuffled.cuda(), w_scales_preshuffled.cuda(), c)
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000 / args.iters
        tflops = 2 * M * N * K / us / 1e6
        print(f"  {us:.1f} us, {tflops:.2f} TFLOPS")

        backends["LLVM 4-wave"] = {
            "us": us,
            "tflops": tflops,
            "metrics": compute_metrics(llvm_asm) if llvm_asm else InstructionMetrics(),
            "resources": llvm_resources,
            "loop_structure": llvm_loop,
            "asm_path": str(out_dir / "llvm_4wave.s") if llvm_asm else None,
        }
        del c
        torch.cuda.empty_cache()
    except Exception as e:
        import traceback

        traceback.print_exc()

    # ============================================================
    # 3. C++ ASM Backend (4-wave)
    # ============================================================
    print("\n>>> C++ ASM backend (4-wave, via waveasm-translate)...")
    try:
        from wave_lang.kernel.wave.templates import (
            get_tagged_mxfp4_gemm,
            get_tagged_mxfp4_gemm_preshuffle_b,
        )
        from wave_lang.kernel.wave.schedules import get_mxfp4_asymmetric_schedule
        from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
        from wave_lang.kernel.wave.utils.mxfp_utils import (
            generate_gemm_afp4wfp4_inputs,
            b_preshuffle,
            e8m0_shuffle,
        )

        sys.path.insert(0, "wave_lang/kernel/wave/asm/scripts")
        from compare_backends import get_default_arch

        sys.path.insert(0, "wave_lang/kernel/wave/asm/wave_asm/test/e2e")
        from test_asm_backend_e2e import capture_wave_kernel_info
        from waveasm_e2e import WaveASMCompiler, run_with_wave_runtime

        _ws2 = (1, 4) if pcfg.num_waves <= 4 else (4, 2)
        if pcfg.num_waves <= 4:
            gemm2, opts2 = get_tagged_mxfp4_gemm_preshuffle_b(
                shape, pcfg.block_shape, wave_shape=_ws2
            )
        else:
            gemm2, opts2 = get_tagged_mxfp4_gemm(
                shape, pcfg.block_shape, wave_shape=_ws2
            )
        schedule2 = get_mxfp4_asymmetric_schedule()
        opts2 = set_default_run_config(opts2)
        opts2.backend = "asm"
        opts2.wave_runtime = True
        opts2.compile_to_mlir = False

        print("  Capturing MLIR...")
        kernel_info = capture_wave_kernel_info(opts2, gemm2, schedule=schedule2)

        print("  Compiling with C++ backend...")
        compiler = WaveASMCompiler(target=get_default_arch(), keep_temp_files=True)
        cpp_result = compiler.compile_full(
            kernel_info.mlir_text, kernel_info.workgroup_size
        )

        # Always save generated assembly (even if assembler rejects it)
        cpp_asm = cpp_result.asm_text or ""
        if cpp_asm:
            cpp_asm_path = out_dir / "cpp_4wave.s"
            cpp_asm_path.write_text(cpp_asm)
            print(
                f"  Assembly dumped: {cpp_asm_path} ({len(cpp_asm.splitlines())} lines)"
            )

        if not cpp_result.success:
            print(f"  C++ compile failed: {cpp_result.error_message[:200]}")
        else:
            cpp_asm = cpp_result.asm_text
            cpp_asm_path = out_dir / "cpp_4wave.s"
            cpp_asm_path.write_text(cpp_asm)
            print(
                f"  Assembly dumped: {cpp_asm_path} ({len(cpp_asm.splitlines())} lines)"
            )

            cpp_metrics = compute_metrics(cpp_asm)
            cpp_resources = extract_resource_usage(cpp_asm)
            print(
                f"  VGPRs={cpp_resources.get('vgprs', 0)}, "
                f"SGPRs={cpp_resources.get('sgprs', 0)}, "
                f"AGPRs={cpp_resources.get('agprs', 0)}"
            )

            # Run on GPU - use shared inputs for correctness if available
            if mxfp4_inputs is not None:
                x, w, x_scales, w_scales = mxfp4_inputs
                x, w = x.cuda(), w.cuda()
                x_scales, w_scales = x_scales.cuda(), w_scales.cuda()
            else:
                x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
                x, w = x.cuda(), w.cuda()
                x_scales, w_scales = x_scales.cuda(), w_scales.cuda()
            c = torch.zeros(M, N, dtype=torch.float32, device="cuda:0")

            kernel_name = cpp_result.get_kernel_name() or kernel_info.kernel_name

            for _ in range(args.warmup):
                c.zero_()
                run_with_wave_runtime(
                    binary_path=cpp_result.binary_path,
                    inputs=[
                        x,
                        x_scales,
                        w_preshuffled.cuda(),
                        w_scales_preshuffled.cuda(),
                    ],
                    outputs=[c],
                    grid=kernel_info.grid_size,
                    block=kernel_info.workgroup_size,
                    shared_memory_bytes=kernel_info.lds_size,
                    func_name=kernel_name,
                )
            torch.cuda.synchronize()

            # Correctness check (after warmup, before timed region)
            if args.check_correctness and torch_ref is not None:
                c.zero_()
                run_with_wave_runtime(
                    binary_path=cpp_result.binary_path,
                    inputs=[
                        x,
                        x_scales,
                        w_preshuffled.cuda(),
                        w_scales_preshuffled.cuda(),
                    ],
                    outputs=[c],
                    grid=kernel_info.grid_size,
                    block=kernel_info.workgroup_size,
                    shared_memory_bytes=kernel_info.lds_size,
                    func_name=kernel_name,
                )
                torch.cuda.synchronize()
                check_correctness("C++ ASM 4-wave", c, torch_ref)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                c.zero_()
                run_with_wave_runtime(
                    binary_path=cpp_result.binary_path,
                    inputs=[
                        x,
                        x_scales,
                        w_preshuffled.cuda(),
                        w_scales_preshuffled.cuda(),
                    ],
                    outputs=[c],
                    grid=kernel_info.grid_size,
                    block=kernel_info.workgroup_size,
                    shared_memory_bytes=kernel_info.lds_size,
                    func_name=kernel_name,
                )
            end.record()
            torch.cuda.synchronize()
            us = start.elapsed_time(end) * 1000 / args.iters
            tflops = 2 * M * N * K / us / 1e6
            print(f"  {us:.1f} us, {tflops:.2f} TFLOPS")

            cpp_loop = analyze_loop_structure(cpp_asm)
            backends["C++ ASM 4-wave"] = {
                "us": us,
                "tflops": tflops,
                "metrics": cpp_metrics,
                "resources": cpp_resources,
                "loop_structure": cpp_loop,
                "asm_path": str(cpp_asm_path),
            }
            del x, w, x_scales, w_scales, c
            torch.cuda.empty_cache()
    except Exception as e:
        import traceback

        traceback.print_exc()

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    aiter_tflops = backends.get("aiter", {}).get("tflops", 0)
    header = f"{'Backend':<25s} {'Time (us)':>12s} {'TFLOPS':>12s} {'vs aiter':>12s}"
    print(header)
    print("-" * 70)
    for name, data in backends.items():
        us_val = data["us"]
        tf_val = data["tflops"]
        ratio = tf_val / aiter_tflops if aiter_tflops > 0 else 0
        print(f"{name:<25s} {us_val:12.1f} {tf_val:12.2f} {ratio:11.2f}x")
    print("=" * 70)

    # Assembly comparison (only backends with metrics)
    asm_backends = {n: d for n, d in backends.items() if d["metrics"].total > 0}
    if len(asm_backends) >= 2:
        print("\n" + format_comparison(asm_backends, pcfg=pcfg))

    # Regression gate checks (Phase 4)
    if args.check_regression and "aiter" in backends and "C++ ASM 4-wave" in backends:
        print("\n" + "=" * 70)
        print("Regression Gate: C++ ASM vs aiter")
        print("=" * 70)
        rg = check_regression_gates(backends["aiter"], backends["C++ ASM 4-wave"])
        if rg.improvements:
            for imp in rg.improvements:
                print(f"  [+] {imp}")
        if rg.regressions:
            for reg in rg.regressions:
                print(f"  [-] {reg}")
        print(f"  Result: {'PASS' if rg.passed else 'FAIL'}")
        print("=" * 70)

    # Loop signature diff: C++ ASM vs aiter reference kernel
    aiter_ref_s = Path(
        "/dockerx/aiter_kernels/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.s"
    )
    if aiter_ref_s.exists() and "C++ ASM 4-wave" in backends:
        cpp_asm_path = backends["C++ ASM 4-wave"].get("asm_path")
        cpp_asm_text = (
            Path(cpp_asm_path).read_text()
            if cpp_asm_path and Path(cpp_asm_path).exists()
            else ""
        )
        aiter_ref_text = aiter_ref_s.read_text()
        if cpp_asm_text and aiter_ref_text:
            sig_aiter = extract_loop_signature(aiter_ref_text)
            sig_cpp = extract_loop_signature(cpp_asm_text)
            sig_diff = format_loop_signature_diff(
                "aiter-ref", sig_aiter, "C++ ASM", sig_cpp
            )
            print("\n" + sig_diff)
            sig_path = out_dir / "loop_signature_diff.txt"
            sig_path.write_text(sig_diff)
            print(f"  Loop signature diff saved: {sig_path}")

    # ============================================================
    # ATT Thread Trace capture (optional)
    # ============================================================
    if args.capture_trace:
        print("\n" + "=" * 70)
        print("Capturing ATT Thread Traces")
        print("=" * 70)

        try:
            rocprofv3_path = _find_rocprofv3(args.rocprofv3)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            rocprofv3_path = None

        if rocprofv3_path:
            att_lib = args.att_library_path
            print(f"  rocprofv3: {rocprofv3_path}")
            print(f"  ATT library: {att_lib}")

            # Determine working directory (repo root for wave imports)
            # The script must be run from the wave repo root.
            wave_root = str(Path(__file__).resolve().parent)

            # --- aiter trace ---
            if "aiter" in backends:
                print("\n>>> Capturing aiter ATT trace...")
                aiter_trace_dir = out_dir / "att_traces" / "aiter"
                aiter_script = out_dir / "att_traces" / "trace_aiter.py"
                aiter_script.parent.mkdir(parents=True, exist_ok=True)
                _write_trace_script_aiter(aiter_script, M, N, K)
                # aiter kernel name contains "f4gemm" and "BpreShuffle"
                capture_att_trace(
                    backend_name="aiter",
                    kernel_regex="f4gemm.*BpreShuffle",
                    trace_script=aiter_script,
                    trace_output_dir=aiter_trace_dir,
                    rocprofv3_path=rocprofv3_path,
                    att_library_path=att_lib,
                    working_dir=wave_root,
                )

            # --- LLVM trace ---
            if "LLVM 4-wave" in backends:
                print("\n>>> Capturing LLVM ATT trace...")
                llvm_trace_dir = out_dir / "att_traces" / "llvm"
                llvm_script = out_dir / "att_traces" / "trace_llvm.py"
                llvm_script.parent.mkdir(parents=True, exist_ok=True)
                _write_trace_script_llvm(
                    llvm_script, M, N, K, pcfg.block_shape, pcfg.num_waves
                )
                # LLVM kernel name: "gemm" (from Wave/IREE runtime)
                capture_att_trace(
                    backend_name="LLVM",
                    kernel_regex="gemm",
                    trace_script=llvm_script,
                    trace_output_dir=llvm_trace_dir,
                    rocprofv3_path=rocprofv3_path,
                    att_library_path=att_lib,
                    working_dir=wave_root,
                )

            # --- C++ ASM trace ---
            if "C++ ASM 4-wave" in backends:
                print("\n>>> Capturing C++ ASM ATT trace...")
                cpp_trace_dir = out_dir / "att_traces" / "cpp_asm"
                cpp_script = out_dir / "att_traces" / "trace_cpp_asm.py"
                cpp_script.parent.mkdir(parents=True, exist_ok=True)
                _write_trace_script_cpp(
                    cpp_script, M, N, K, pcfg.block_shape, pcfg.num_waves
                )
                # C++ ASM kernel name: "gemm" (from wave_runtime)
                capture_att_trace(
                    backend_name="C++ ASM",
                    kernel_regex="gemm",
                    trace_script=cpp_script,
                    trace_output_dir=cpp_trace_dir,
                    rocprofv3_path=rocprofv3_path,
                    att_library_path=att_lib,
                    working_dir=wave_root,
                )

            # ---- Analyze ATT traces ----
            print("\n" + "=" * 70)
            print("ATT Stall Analysis")
            print("=" * 70)
            att_analyses = []

            for label, tdir in [
                ("aiter", out_dir / "att_traces" / "aiter"),
                ("C++ ASM", out_dir / "att_traces" / "cpp_asm"),
            ]:
                csv_path = find_best_att_csv(tdir)
                if csv_path and csv_path.stat().st_size > 200:
                    print(f"\n  Analyzing {label}: {csv_path.name}")
                    analysis = analyze_att_trace(label, csv_path)
                    att_analyses.append(analysis)
                else:
                    print(f"\n  {label}: No trace data available")

            if att_analyses:
                comparison = format_att_analysis(att_analyses)
                print("\n" + comparison)

                # Save ATT analysis to file
                att_report = out_dir / "att_stall_analysis.txt"
                att_report.write_text(comparison)
                print(f"\n  ATT analysis saved: {att_report}")

    # Save report
    report_path = out_dir / "benchmark_report.txt"
    with open(report_path, "w") as f:
        f.write(f"MXFP4 GEMM Parity Benchmark: {M}x{N}x{K}\n")
        f.write(f"Parity config: {json.dumps(manifest, indent=2)}\n\n")
        f.write(header + "\n" + "-" * 70 + "\n")
        for name, data in backends.items():
            us_val = data["us"]
            tf_val = data["tflops"]
            ratio = tf_val / aiter_tflops if aiter_tflops > 0 else 0
            f.write(f"{name:<25s} {us_val:12.1f} {tf_val:12.2f} {ratio:11.2f}x\n")
        if len(asm_backends) >= 2:
            f.write("\n" + format_comparison(asm_backends, pcfg=pcfg) + "\n")
        # Include regression gate results in report
        if (
            args.check_regression
            and "aiter" in backends
            and "C++ ASM 4-wave" in backends
        ):
            rg = check_regression_gates(backends["aiter"], backends["C++ ASM 4-wave"])
            f.write("\n" + "=" * 70 + "\n")
            f.write("Regression Gate: C++ ASM vs aiter\n")
            f.write("=" * 70 + "\n")
            for imp in rg.improvements:
                f.write(f"  [+] {imp}\n")
            for reg in rg.regressions:
                f.write(f"  [-] {reg}\n")
            f.write(f"  Result: {'PASS' if rg.passed else 'FAIL'}\n")
        # Include loop signature diff if available
        sig_path = out_dir / "loop_signature_diff.txt"
        if sig_path.exists():
            f.write("\n" + sig_path.read_text() + "\n")
        # Include ATT analysis if available
        att_report = out_dir / "att_stall_analysis.txt"
        if att_report.exists():
            f.write("\n" + att_report.read_text() + "\n")
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
