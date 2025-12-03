# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Driver for MLIR to AMDGCN assembly emitter.
"""

import argparse
import sys
from typing import Iterable, List

from wave_lang.support.ir_imports import (
    func_d,
    Context,
    Module,
    Operation,
)

from .mlir_walker import IRWalker
from .asm_emitter import AsmEmitter


def walk_ops_recursively(operation: Operation) -> Iterable[Operation]:
    for region in operation.regions:
        for block in region.blocks:
            for inner_operation in block.operations:
                yield inner_operation
                yield from walk_ops_recursively(inner_operation)


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

        all_kernels: List[str] = []

        for function_operation in walk_ops_recursively(module.operation):
            if isinstance(function_operation, func_d.FuncOp):
                # Extract basic info directly from MLIR function
                kernel_name = function_operation.sym_name.value
                num_args = len(function_operation.entry_block.arguments)

                # Extract workgroup size from function attributes
                from .utils import parse_wg_and_subgroup
                from wave_lang.support.ir_imports import OpAttributeMap

                wg_size = None
                function_attributes = (
                    dict(function_operation.attributes)
                    if isinstance(function_operation.attributes, OpAttributeMap)
                    else {}
                )
                translation_info = function_attributes.get("translation_info")
                if translation_info is not None:
                    workgroup_size_tuple, _ = parse_wg_and_subgroup(translation_info)
                    if workgroup_size_tuple:
                        wg_size = workgroup_size_tuple

                # Workgroup size is required for code generation
                assert (
                    wg_size is not None
                ), "translation_info with workgroup_size must be present in MLIR function attributes"

                # Create emitter and detect which workgroup IDs are needed
                emitter = AsmEmitter(targetid=args.targetid, codeobj=args.codeobj)
                needs_wgid_x, needs_wgid_y, needs_wgid_z = (
                    AsmEmitter._detect_needed_workgroup_ids(function_operation)
                )
                emitter.needs_wgid_x = needs_wgid_x
                emitter.needs_wgid_y = needs_wgid_y
                emitter.needs_wgid_z = needs_wgid_z

                # Emit kernel preamble and kernargs
                emitter.emit_prologue(kernel_name, wg_size)
                emitter.emit_kernargs(num_args)

                # Walk MLIR and emit instructions
                walker = IRWalker(emitter)
                kernel_info = walker.interpret_func(function_operation)

                emitter.emit_epilogue(
                    kernel_info.name,
                    kernel_info.wg_size[0],
                    kernel_info.subgroup_size,
                    len(kernel_info.arg_ssa_order),
                    kernel_info.lds_size_bytes,
                )

                all_kernels.append(kernel_info.name)

        # Output
        if args.out == "-":
            print("\n".join(emitter.lines))
        else:
            with open(args.out, "w") as f:
                f.write("\n".join(emitter.lines))


if __name__ == "__main__":
    main()
