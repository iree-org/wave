"""
This script is based on CMS logic to calculate the loop efficiency of a kernel.

per_wave_eff = (n_mfma * mfma_cycles) * 100 / single_loop_cycles

Usage:
    python get_loop_efficiency.py <att_dir> [--inst mfma|wmma] [-v]

Examples:
    python get_loop_efficiency.py myultrafast_kernel --inst wmma -v
    python get_loop_efficiency.py my_att_traces --inst mfma
    python get_loop_efficiency.py my_att_traces
"""

import os
import re
import json

from glob import glob
from dataclasses import dataclass

ARCH_INSTRUCTION_CYCLES = {
    "gfx950": {
        # Reference: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
        # Section 7.1.2 List of Dense MFMA instructions
        # V_MFMA_F32_{*}_BF16
        "v_mfma_f32_32x32x4_2b_bf16": 64,
        "v_mfma_f32_16x16x4_4b_bf16": 32,
        "v_mfma_f32_4x4x4_16b_bf16": 8,
        "v_mfma_f32_32x32x8_bf16": 32,
        "v_mfma_f32_16x16x16_bf16": 16,
        # further down in the same table
        "v_mfma_f32_16x16x32_bf16": 16,
        "v_mfma_f32_32x32x16_bf16": 32,
        # V_MFMA_SCALE_F32 (MXFP)
        "v_mfma_scale_f32_16x16x128_f8f6f4": 16,
        "v_mfma_scale_f32_32x32x64_f8f6f4": 32,
        # V_WMMA_F32 (experimental)
        # TODO: check on the issue rate of wmma instructions
        "v_wmma_f32_16x16x32_f16": 8,
    }
}


@dataclass
class KernelStats:
    """
    Class to store the wave kernel statistics.

    Attributes:
        preloop_cycles: Number of cycles in the preloop.
        mainloop_cycles: Number of cycles in the mainloop.
        postloop_cycles: Number of cycles in the postloop.
        mainloop_iterations: Number of iterations in the mainloop.
        mainloopmfma_instructions: Number of MFMA instructions in the mainloop.
        mfma_instruction_cycles: Number of cycles for the MFMA instruction.
        total_cycles: Total number of cycles in the kernel.
        single_loop_cycles: Number of cycles in a single iteration of the mainloop.
        efficiency: Efficiency of the kernel (ratio of theoretical MFMA instruction cycles to actual main loop cycles).
    """

    preloop_cycles: int = -1
    mainloop_cycles: int = -1
    postloop_cycles: int = -1

    mainloop_iterations: int = -1
    mainloopmfma_instructions: int = -1

    @property
    def mfma_instruction_cycles(self):
        return self._mfma_instruction_cycles

    @property
    def total_cycles(self):
        assert (
            self.preloop_cycles >= 0
            and self.mainloop_cycles >= 0
            and self.postloop_cycles >= 0
        ), "Preloop, mainloop, and postloop cycles should be greater than 0"
        return self.preloop_cycles + self.mainloop_cycles + self.postloop_cycles

    @property
    def single_loop_cycles(self):
        assert (
            self.mainloop_iterations > 0 and self.mainloop_cycles >= 0
        ), "Mainloop iterations should be greater than 0 and mainloop cycles should be greater than 0"
        return int(self.mainloop_cycles / self.mainloop_iterations)

    @property
    def efficiency(self):
        assert (
            self.mfma_instruction_cycles > 0
        ), "MFMA instruction cycles should be greater than 0"
        assert (
            self.mainloopmfma_instructions > 0 and self.mainloop_cycles > 0
        ), "Mainloop MFMA instructions and mainloop cycles should be greater than 0"
        return (
            self.mainloopmfma_instructions
            * self.mfma_instruction_cycles
            / self.single_loop_cycles
        ) * 100

    def __init__(self, arch: str, mfma_instruction: str):
        if arch.lower() not in ARCH_INSTRUCTION_CYCLES:
            raise ValueError(
                f"Arch {arch} is not in ARCH_INSTRUCTION_CYCLES - pls update it in the code"
            )
        self._arch = arch.lower()

        if mfma_instruction.lower() not in ARCH_INSTRUCTION_CYCLES[self._arch]:
            raise ValueError(
                f"MFMA instruction {mfma_instruction} is not in the list of MFMA instructions for arch {arch} - pls update ARCH_INSTRUCTION_CYCLES in this file."
            )
        self._mfma_instruction = mfma_instruction.lower()

        self._mfma_instruction_cycles = ARCH_INSTRUCTION_CYCLES[self._arch][
            self._mfma_instruction
        ]

    def pretty_print(self, verbose: bool = False, per_wave_stats=None):
        print("========== Kernel Stats ==========")
        if verbose:
            print(
                f"Arch:            {self._arch}, {self._mfma_instruction}: {self._mfma_instruction_cycles} cycles"
            )
            print(
                f"Preloop cycles:  {self.preloop_cycles}, {self.preloop_cycles/self.total_cycles*100:.0f}%"
            )
            print(
                f"Mainloop cycles: {self.mainloop_cycles}, {self.mainloop_cycles/self.total_cycles*100:.0f}%"
            )
            print(
                f"Postloop cycles: {self.postloop_cycles}, {self.postloop_cycles/self.total_cycles*100:.0f}%"
            )
            print(f"Total cycles:    {self.total_cycles}")
            print(
                f"MFMA instructions in the mainloop: {self.mainloopmfma_instructions}"
            )
            print(f"MFMA instruction cycles: {self._mfma_instruction_cycles}")
            print(f"Mainloop iterations:     {self.mainloop_iterations}")

        print(f"Single iteration cycles: {self.single_loop_cycles}")
        print(f"Efficiency:              {self.efficiency:.1f}%")

        if per_wave_stats and len(per_wave_stats) > 1:
            import re as _re

            mfma_cyc = self._mfma_instruction_cycles
            n_mfma = self.mainloopmfma_instructions
            theoretical_per_wave = n_mfma * mfma_cyc

            # Per-wave table
            print(f"\n{'=' * 60}")
            print(f"Per-wave breakdown ({len(per_wave_stats)} waves)")
            print(f"{'=' * 60}")
            print(
                f"{'Wave':<28s} {'Prelog':>7s} {'Loop':>7s} {'Epilog':>7s} {'Total':>7s} {'Iter':>5s} {'Eff':>6s}"
            )
            print("-" * 60)
            for ws in per_wave_stats:
                sl = ws["single_loop"]
                eff = (theoretical_per_wave / sl * 100) if sl > 0 else 0
                print(
                    f"{ws['name']:<28s} {ws['preloop']:>7d} {ws['mainloop']:>7d} {ws['postloop']:>7d} {ws['total']:>7d} {sl:>5d} {eff:>5.1f}%"
                )

            # Group by SIMD: waves with same se*_sm* co-execute and share the MFMA unit
            simd_groups = {}
            for ws in per_wave_stats:
                m = _re.match(r"(se\d+_sm\d+)", ws["name"])
                simd_key = m.group(1) if m else ws["name"]
                simd_groups.setdefault(simd_key, []).append(ws)

            print(f"\n{'=' * 60}")
            print(f"Per-SIMD efficiency (co-executing waves share MFMA unit)")
            print(f"{'=' * 60}")
            print(f"{'SIMD':<16s} {'Waves':>5s} {'Avg Iter':>8s} {'SIMD Eff':>9s}")
            print("-" * 60)
            all_simd_eff = []
            for simd_key in sorted(simd_groups):
                waves = simd_groups[simd_key]
                n_co = len(waves)
                avg_sl = sum(ws["single_loop"] for ws in waves) // n_co
                simd_eff = (
                    (n_co * theoretical_per_wave / avg_sl * 100) if avg_sl > 0 else 0
                )
                all_simd_eff.append(simd_eff)
                sat = " (saturated)" if simd_eff > 100 else ""
                print(
                    f"{simd_key:<16s} {n_co:>5d} {avg_sl:>8d} {min(simd_eff,100):>8.1f}%{sat}"
                )

            n_simds = len(all_simd_eff)
            avg_simd_eff = sum(all_simd_eff) / n_simds if n_simds > 0 else 0
            print("-" * 60)
            sat = " (saturated)" if avg_simd_eff > 100 else ""
            print(
                f"{'AVERAGE':<16s} {'':>5s} {'':>8s} {min(avg_simd_eff,100):>8.1f}%{sat}"
            )


class AttParser:
    def __init__(self, att_dir: str, verbose: bool = False, inst_type: str = "mfma"):
        """
        Initialize the AttParser object and parse the ATT trace files.

        Args:
            att_dir: Path to the ATT analysis directory.
            verbose: Whether to print verbose output.
            inst_type: Instruction type to look for: "mfma" or "wmma".

        TODO: wmma is WIP!
        """
        self.att_dir = att_dir
        self.verbose = verbose
        self.inst_prefix = f"v_{inst_type}_"
        self.start_cycle = 0
        self.end_cycle = 0

        self.mainloop_starts_at_cycle = 0
        self.mainloop_ends_at_cycle = 0
        self.total_loops = 0
        self.totalmfma_instructions = 0

        self.dispatch_csv = self.get_dispatch_csv()
        if self.verbose:
            print("AttParser: Dispatch CSV Path:", self.dispatch_csv)

        self.wave_jsons = self.get_wave_jsons(self.dispatch_csv)
        self.cu_json = self.wave_jsons[0]
        if self.verbose:
            print(f"AttParser: {len(self.wave_jsons)} wave JSON(s) found")
            for wj in self.wave_jsons:
                print(f"  {os.path.basename(wj)}")

        self.code_json = os.path.join(os.path.dirname(self.cu_json), "code.json")
        if self.verbose:
            print("AttParser: Code JSON Path:", self.code_json)

        self.mfma_instruction = ""
        if not (arch := find_gfx_arch(att_dir)):
            print(
                f"Warning: Couldn't resolve gfx architecture revision from ATT traces. Assuming gfx950"
            )
            arch = "gfx950"
        self.arch = arch

        self.per_wave_stats = []
        self._parse()

    def get_kernel_stats(self) -> KernelStats:
        stats = KernelStats(self.arch, self.mfma_instruction)
        stats.preloop_cycles = self.preloop_cycles
        stats.mainloop_cycles = self.mainloop_cycles
        stats.postloop_cycles = self.postloop_cycles
        stats.mainloop_iterations = self.total_loops
        stats.mainloopmfma_instructions = self.totalmfma_instructions

        return stats

    def get_dispatch_csv(self):
        """Find the largest stats*.csv file in att_dir (recursively)."""
        csv_files = glob(os.path.join(self.att_dir, "**", "stats*.csv"), recursive=True)
        if not csv_files:
            raise Exception(f"No stats*.csv file found in {self.att_dir}")
        # pick the largest one
        best = max(csv_files, key=os.path.getsize)
        # return path relative to att_dir
        return os.path.relpath(best, self.att_dir)

    def get_wave_jsons(self, dispatch_csv):
        """Return all se*_sm*_sl*_wv*.json files, searching near the dispatch CSV."""
        csv_abs = os.path.join(self.att_dir, dispatch_csv)
        csv_dir = os.path.dirname(csv_abs)

        # Try 1: subdirectory derived from CSV name (rocprofv3 layout)
        dispatch = os.path.basename(dispatch_csv).replace(".csv", "")
        cu_dir = os.path.join(csv_dir, dispatch.replace("stats_", ""))
        wave_files = sorted(glob(os.path.join(cu_dir, "se*_sm*_sl*_wv*.json")))
        if wave_files:
            return wave_files

        # Try 2: same directory as the CSV
        wave_files = sorted(glob(os.path.join(csv_dir, "se*_sm*_sl*_wv*.json")))
        if wave_files:
            return wave_files

        raise Exception(f"No wave JSON files found near {csv_abs}")

    def get_cu_utilization_json(self, dispatch_csv):
        """Return first wave JSON (for backward compat)."""
        return self.get_wave_jsons(dispatch_csv)[0]

    @property
    def preloop_cycles(self):
        return self.mainloop_starts_at_cycle - self.start_cycle

    @property
    def postloop_cycles(self):
        return self.end_cycle - self.mainloop_ends_at_cycle

    @property
    def mainloop_cycles(self):
        return self.mainloop_ends_at_cycle - self.mainloop_starts_at_cycle

    # Core logic to parse a single wave's ATT trace and extract the loop timing information.
    def _parse_single_wave(self, json_path):
        """
        Parse one wave's ATT trace to find the main loop and measure its cycle cost.

        Steps:
          1. Read instructions from the wave JSON. Each entry is (cycle, _, _, _, idx).
          2. Find the main loop by looking for the most common backward jump (idx going back by >20).
          3. Record when the loop starts/ends (in cycles) and how many iterations it runs.
          4. Split the wave's lifetime into preloop, mainloop, and postloop cycle regions.

        Returns a dict with cycle counts and loop info for this wave.
        """
        from collections import Counter

        with open(os.path.normpath(json_path), "r") as f:
            cu_data = json.load(f)

        instructions = cu_data["wave"]["instructions"]
        start_cycle = instructions[0][0]
        end_cycle = instructions[-1][0]

        # find loop boundaries by finding repeated backward jumps
        backward_jumps = []
        idx_prev = -1
        for cycle, _, _, _, idx in instructions:
            if idx_prev - idx > 20:
                backward_jumps.append((idx, idx_prev))
            idx_prev = idx

        jump_counts = Counter(backward_jumps)
        if not jump_counts:
            raise Exception(f"No backward jumps found in {json_path}")
        (idx_start, idx_end), _ = jump_counts.most_common(1)[0]

        # count cycles and loops
        start_cycles, end_cycles = [], []
        loops = 0
        loop_start_cycle = 0
        loop_end_cycle = 0
        for cycle, _, _, _, idx in instructions:
            if idx == idx_start:
                start_cycles.append(cycle)
                if loop_start_cycle == 0:
                    loop_start_cycle = cycle
            if idx == idx_end:
                end_cycles.append(cycle)
                loops += 1
                loop_end_cycle = cycle

        preloop = loop_start_cycle - start_cycle
        mainloop = loop_end_cycle - loop_start_cycle
        postloop = end_cycle - loop_end_cycle

        return {
            "name": os.path.basename(json_path).replace(".json", ""),
            "start_cycle": start_cycle,
            "end_cycle": end_cycle,
            "loop_start": loop_start_cycle,
            "loop_end": loop_end_cycle,
            "loops": loops,
            "idx_start": idx_start,
            "idx_end": idx_end,
            "preloop": preloop,
            "mainloop": mainloop,
            "postloop": postloop,
            "total": preloop + mainloop + postloop,
            "single_loop": int(mainloop / loops) if loops > 0 else 0,
        }

    def _parse(self):
        # Parse all waves
        for wj in self.wave_jsons:
            try:
                ws = self._parse_single_wave(wj)
                self.per_wave_stats.append(ws)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: skipping {os.path.basename(wj)}: {e}")

        if not self.per_wave_stats:
            raise Exception("No waves could be parsed")

        # Use first wave for backward-compat fields
        w0 = self.per_wave_stats[0]
        self.start_cycle = w0["start_cycle"]
        self.end_cycle = w0["end_cycle"]
        self.mainloop_starts_at_cycle = w0["loop_start"]
        self.mainloop_ends_at_cycle = w0["loop_end"]
        self.total_loops = w0["loops"]

        if self.verbose:
            print(
                f"Mainloop detected at: start idx {w0['idx_start']}, end idx {w0['idx_end']}"
            )

        # Count MFMA instructions in the loop (from code.json, same for all waves)
        idx_start = w0["idx_start"]
        idx_end = w0["idx_end"]
        with open(self.code_json, "r") as f:
            code_data = json.load(f)["code"]

        idx = -1
        for inst, _, idx, asm_src, _, addr, _, _, _, _ in code_data:
            idx += 1
            if idx < idx_start:
                continue
            if idx > idx_end:
                break
            if inst.startswith(self.inst_prefix):
                mfma_instruction = inst.split()[0]
                if not self.mfma_instruction:
                    self.mfma_instruction = mfma_instruction
                if mfma_instruction == self.mfma_instruction:
                    self.totalmfma_instructions += 1

        if not self.mfma_instruction:
            raise Exception(f"No {self.inst_prefix}* instruction detected in the loop")


def find_gfx_arch(att_path: str):
    """
    Search for .out files under att_path and find a string matching gfx<DDD> where DDD is a 3 or 4 digit number.
    Stops at the first match found.
    Returns the matched architecture string (e.g. 'gfx1030') or None if not found.
    """
    gfx_pattern = re.compile(rb"gfx\d{3,4}")
    for out_path in glob(os.path.join(att_path, "**", "*.out"), recursive=True):
        try:
            with open(out_path, "rb") as f:
                data = f.read()
                match = gfx_pattern.search(data)
                if match:
                    # Convert bytes to str if needed
                    return match.group(0).decode("ascii")
        except Exception as e:
            pass  # Skip unreadable files
    return None


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(
        description="Parse ATT trace files to extract kernel statistics and MFMA instruction usage.",
        epilog="""\
examples:
  python get_loop_efficiency.py myultrafast_kernel --inst wmma -v
  python get_loop_efficiency.py my_att_traces --inst mfma
  python get_loop_efficiency.py my_att_traces""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    argparser.add_argument(
        "att_dir", type=str, help="Path to the ATT analysis directory."
    )
    argparser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output."
    )
    argparser.add_argument(
        "--inst",
        type=str,
        default="mfma",
        choices=["mfma", "wmma"],
        help="Instruction type to look for: mfma (default) or wmma (experimental).",
    )

    args = argparser.parse_args()
    parser = AttParser(args.att_dir, args.verbose, inst_type=args.inst)
    parser.get_kernel_stats().pretty_print(
        verbose=args.verbose, per_wave_stats=parser.per_wave_stats
    )
