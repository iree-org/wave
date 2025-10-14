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

                # Emit kernel preamble and kernargs
                emitter = AsmEmitter(targetid=args.targetid, codeobj=args.codeobj)
                emitter.emit_prologue(kernel_name)
                emitter.emit_kernargs(num_args)

                # Do full traversal with emitter to emit instructions
                walker = IRWalker(emitter)
                kernel_info = walker.interpret_func(function_operation)

                emitter.emit_epilogue(
                    kernel_info.name,
                    kernel_info.wg_size[0],
                    kernel_info.subgroup_size,
                    len(kernel_info.arg_ssa_order),
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
