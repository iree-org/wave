# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .kernel_compilation_context import KernelCompilationContext


@dataclass
class KernelModuleCompiler:
    """
    Module-level kernel compiler that generates complete .s assembly files.

    This class handles the full compilation pipeline from MLIR to assembly:
    1. Parse MLIR and extract kernel metadata
    2. Create KernelCompilationContext
    3. Walk MLIR operations and emit to kernel IR
    4. Run register allocation
    5. Generate complete assembly (prologue + body + epilogue + metadata)

    Uses MetadataEmitter for prologue/epilogue generation (single source of truth).

    Usage:
        compiler = KernelModuleCompiler(targetid="gfx942", codeobj="5")
        asm = compiler.compile_mlir_string(mlir_text)
    """

    targetid: str = "gfx942"
    codeobj: str = "5"

    def compile_mlir_string(self, mlir_text: str) -> str:
        """
        Compile MLIR text to complete AMDGCN assembly.

        Args:
            mlir_text: MLIR module text

        Returns:
            Complete assembly text ready for assembler
        """
        from wave_lang.support.ir_imports import Context, Module, func_d
        from .mlir_walker import IRWalker
        from .metadata_emitter import MetadataEmitter, create_metadata
        from .mlir_analysis import (
            walk_ops_recursively,
            detect_needed_workgroup_ids,
            extract_translation_info,
            should_skip_function,
        )

        all_lines: List[str] = []

        with Context() as ctx:
            ctx.allow_unregistered_dialects = True
            module = Module.parse(mlir_text)

            for fn in walk_ops_recursively(module.operation):
                if not isinstance(fn, func_d.FuncOp):
                    continue

                kernel_name = fn.sym_name.value

                # Skip non-kernel functions (async wrappers, benchmark scaffolding)
                if should_skip_function(fn):
                    continue

                num_args = len(list(fn.entry_block.arguments))

                # Extract kernel metadata
                ti = extract_translation_info(fn)
                wg_size, subgroup_size = ti.wg_size, ti.subgroup_size

                # Detect workgroup ID needs
                needs_wgid_x, needs_wgid_y, needs_wgid_z = detect_needed_workgroup_ids(
                    fn
                )

                # Create metadata for prologue/epilogue (via MetadataEmitter)
                metadata = create_metadata(
                    name=kernel_name,
                    targetid=self.targetid,
                    codeobj=self.codeobj,
                    wg_size=wg_size,
                    subgroup_size=subgroup_size,
                    needs_wgid=(needs_wgid_x, needs_wgid_y, needs_wgid_z),
                    num_args=num_args,
                )

                # Emit prologue (assembler directives)
                meta_emitter = MetadataEmitter(metadata)
                prologue_lines = meta_emitter.emit_prologue()

                # Create kernel context with proper thread ID bounds
                num_waves = max(
                    1, wg_size[0] * wg_size[1] * wg_size[2] // subgroup_size
                )
                kernel_ctx = KernelCompilationContext(
                    use_flat_tid=(num_waves > 1),
                    use_workgroup_ids=(needs_wgid_x, needs_wgid_y, needs_wgid_z),
                    tid_ub_x=wg_size[0],
                    tid_ub_y=wg_size[1],
                    tid_ub_z=wg_size[2] if len(wg_size) > 2 else 1,
                    subgroup_size=subgroup_size,
                    wg_size=wg_size,
                )

                # Emit kernarg loading at the start of kernel IR
                kernel_ctx.emit_kernargs(num_args)

                # Walk MLIR and emit to kernel IR
                walker = IRWalker(kernel_ctx)
                kernel_info = walker.interpret_func(fn)

                # Finalize kernel IR (adds s_endpgm, runs allocation, renders)
                body_lines, stats = kernel_ctx.finalize()

                # Get LDS size from kernel_info
                lds_size_bytes = getattr(kernel_info, "lds_size_bytes", 0)

                # Patch prologue with actual resource values
                patched_prologue = MetadataEmitter.patch_resource_usage(
                    prologue_lines,
                    stats.peak_vgprs,
                    stats.peak_sgprs,
                    getattr(stats, "peak_agprs", 0),
                    lds_size_bytes,
                    self.targetid,
                )

                # Emit epilogue (YAML metadata)
                metadata.vgprs_used = stats.peak_vgprs
                metadata.sgprs_used = stats.peak_sgprs
                metadata.agprs_used = getattr(stats, "peak_agprs", 0)
                metadata.lds_size_bytes = lds_size_bytes
                epilogue_lines = meta_emitter.emit_epilogue()

                # Combine all lines: prologue + body + epilogue
                all_lines.extend(patched_prologue)
                all_lines.extend(body_lines)
                all_lines.extend(epilogue_lines)

        return "\n".join(all_lines)

    def _extract_kernel_metadata(self, fn) -> Tuple[Tuple[int, int, int], int]:
        """Deprecated: use mlir_analysis.extract_translation_info instead."""
        from .mlir_analysis import extract_translation_info

        ti = extract_translation_info(fn)
        return ti.wg_size, ti.subgroup_size
