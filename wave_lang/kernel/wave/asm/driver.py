# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Driver for MLIR to AMDGCN assembly emitter.

This module implements a single-path architecture where all instructions
(including kernarg loading, s_endpgm) go through the Kernel IR path:

    ┌─────────────────────────────────────────────────────────────┐
    │  MetadataEmitter.emit_prologue()                            │
    │  - .amdgcn_target, .amdhsa_kernel, etc.                     │
    └─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  KernelCompilationContext + KernelGenerator                 │
    │  - emit_kernargs() -> s_load_dwordx2                        │
    │  - MLIR walker -> kernel body                               │
    │  - finalize() adds s_endpgm                                 │
    └─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  MetadataEmitter.emit_epilogue()                            │
    │  - .amdgpu_metadata                                         │
    └─────────────────────────────────────────────────────────────┘
"""

import argparse
import sys
from typing import Iterable, List, Tuple

from wave_lang.support.ir_imports import (
    func_d,
    Context,
    Module,
    Operation,
)

from .mlir_walker import IRWalker
from .asm_emitter import AsmEmitter
from .kernel_pipeline import KernelCompilationContext
from .metadata_emitter import MetadataEmitter, KernelMetadata, create_metadata
from .mlir_analysis import (
    walk_ops_recursively,
    detect_needed_workgroup_ids,
    extract_translation_info,
    should_skip_function,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="-", help="Input MLIR")
    ap.add_argument("--out", dest="out", default="-", help="Output .s")
    ap.add_argument("--targetid", default="gfx942", help="Target for .amdgcn_target")
    ap.add_argument(
        "--codeobj",
        choices=["4", "5"],
        default="5",
        help="Code object version for metadata",
    )
    args = ap.parse_args()

    mlir_text = sys.stdin.read() if args.inp == "-" else open(args.inp, "r").read()

    with Context() as ctx:
        module = Module.parse(mlir_text)

        all_lines: List[str] = []
        all_kernels: List[str] = []

        for function_operation in walk_ops_recursively(module.operation):
            if isinstance(function_operation, func_d.FuncOp):
                # Extract basic info directly from MLIR function
                kernel_name = function_operation.sym_name.value
                if should_skip_function(function_operation):
                    continue
                num_args = len(function_operation.entry_block.arguments)

                ti = extract_translation_info(function_operation)
                wg_size = ti.wg_size
                subgroup_size = ti.subgroup_size

                # Detect which workgroup IDs are needed
                needs_wgid_x, needs_wgid_y, needs_wgid_z = detect_needed_workgroup_ids(function_operation)

                # Create metadata for prologue/epilogue
                metadata = create_metadata(
                    name=kernel_name,
                    targetid=args.targetid,
                    codeobj=args.codeobj,
                    wg_size=wg_size,
                    subgroup_size=subgroup_size,
                    needs_wgid=(needs_wgid_x, needs_wgid_y, needs_wgid_z),
                    num_args=num_args,
                )
                
                # Emit prologue (assembler directives)
                meta_emitter = MetadataEmitter(metadata)
                prologue_lines = meta_emitter.emit_prologue()

                # Create kernel compilation context
                kernel_ctx = KernelCompilationContext(
                    use_flat_tid=True,
                    use_workgroup_ids=(needs_wgid_x, needs_wgid_y, needs_wgid_z),
                )
                
                # Emit kernarg loading at the start of kernel IR
                kernel_ctx.emit_kernargs(num_args)
                
                # Walk MLIR and emit instructions via kernel IR
                # Note: We still need a minimal emitter for some legacy operations
                # during migration - this will be removed once fully migrated
                emitter = AsmEmitter(targetid=args.targetid, codeobj=args.codeobj)
                emitter.needs_wgid_x = needs_wgid_x
                emitter.needs_wgid_y = needs_wgid_y
                emitter.needs_wgid_z = needs_wgid_z
                
                walker = IRWalker(emitter, kernel_ctx=kernel_ctx)
                kernel_info = walker.interpret_func(function_operation)
                
                # Finalize kernel IR (adds s_endpgm, runs allocation, renders)
                body_lines, stats = kernel_ctx.finalize()
                
                # Update metadata with actual resource usage
                metadata.vgprs_used = stats.peak_vgprs
                metadata.sgprs_used = stats.peak_sgprs
                metadata.agprs_used = getattr(stats, 'peak_agprs', 0)
                metadata.lds_size_bytes = kernel_info.lds_size_bytes
                
                # Patch prologue with actual resource values
                patched_prologue = MetadataEmitter.patch_resource_usage(
                    prologue_lines,
                    stats.peak_vgprs,
                    stats.peak_sgprs,
                    getattr(stats, 'peak_agprs', 0),
                    kernel_info.lds_size_bytes,
                    args.targetid,
                )
                
                # Emit epilogue (YAML metadata)
                epilogue_lines = meta_emitter.emit_epilogue()

                # Combine all lines
                all_lines.extend(patched_prologue)
                all_lines.extend(body_lines)
                all_lines.extend(epilogue_lines)

                all_kernels.append(kernel_info.name)

        # Output
        if args.out == "-":
            print("\n".join(all_lines))
        else:
            with open(args.out, "w") as f:
                f.write("\n".join(all_lines))


if __name__ == "__main__":
    main()
