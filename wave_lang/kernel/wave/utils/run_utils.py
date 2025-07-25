# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import os
from itertools import chain
from typing import Any, Callable

import torch

from wave_lang.kernel.lang import IndexSymbol

from ..compile_options import WaveCompileOptions


@functools.lru_cache
def compute_grid(kernel_dynamic_dims: tuple[int], grid_fn: Callable):
    return [int(x) for x in grid_fn(list(kernel_dynamic_dims))]


def write_file(name, mode, data):
    with open(name, mode) as file:
        file.write(data)


def print_bench_result(result, filename):
    import json

    res = json.dumps(result, sort_keys=True, indent=4)
    if filename is not None:
        write_file(filename, "w", res)
    else:
        print(res)


def get_benchmark_flags(options: WaveCompileOptions):
    benchmark_flags = {}
    benchmark_flags["batch_size"] = options.benchmark_batch_size

    if options.benchmark_repetitions is not None:
        benchmark_flags["benchmark_repetitions"] = int(options.benchmark_repetitions)
    return benchmark_flags


def invoke_with_wave_runtime(
    options: WaveCompileOptions,
    kernel_inputs: list[torch.Tensor],
    kernel_outputs: list[torch.Tensor],
    scalar_args: list[int | float],
    bound_scalar_symbols: dict[IndexSymbol, int],
    dynamic_symbols: list[int],
    gpu_func: Any,
):
    """
    Invokes the kernel with the wave runtime.
    """
    import wave_runtime

    num_inputs = len(kernel_inputs)
    dynamic_dims = tuple(
        scalar_args[v - num_inputs] for v in bound_scalar_symbols.values()
    ) + tuple(dynamic_symbols)
    # Update the grid size as this may vary depending
    # on the dynamic symbols.
    import math

    options.kernel_launch_info.grid.__globals__.setdefault("math", math)
    grid = compute_grid(dynamic_dims, options.kernel_launch_info.grid)

    stream = torch.cuda.current_stream().cuda_stream

    # Populate all the information required to launch the kernel.
    kernel_launch_info = wave_runtime.KernelLaunchInfo(
        stream,
        gpu_func,
        options.kernel_launch_info.shared_memory_bytes,
        grid[0],
        grid[1],
        grid[2],
        options.kernel_launch_info.blocks[0],
        options.kernel_launch_info.blocks[1],
        options.kernel_launch_info.blocks[2],
    )

    # Ensure that the tensors are contiguous.
    kern_args = []
    for arg_tensor in chain(kernel_inputs, kernel_outputs):
        if not arg_tensor.is_contiguous():
            arg_tensor = arg_tensor.contiguous()
        kern_args.append(arg_tensor.data_ptr())

    kernel_args = wave_runtime.Int64Vector(kern_args)
    dyn_dims = wave_runtime.Int64Vector(dynamic_dims[len(bound_scalar_symbols) :])
    # Launch the kernel.
    wave_runtime.launch(kernel_launch_info, kernel_args, dyn_dims, scalar_args)


def get_default_arch() -> str:
    """Return default ROCM architecture"""

    default_arch = os.environ.get("WAVE_DEFAULT_ARCH", None)
    if default_arch:
        return default_arch

    if not torch.cuda.is_available():
        return "cpu"

    device = torch.device("cuda")
    gcnArch = torch.cuda.get_device_properties(device).gcnArchName
    if "gfx" not in gcnArch:
        return "cpu"

    # The gcnArchName comes back like gfx90a:sramecc+:xnack.
    colon_pos = gcnArch.find(":")
    return gcnArch[0:colon_pos]


# Whether to dump the generated MLIR module.
dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))

# Dump the generated MLIR module to a file.
dump_generated_mlir_file = os.environ.get("WAVE_DUMP_MLIR_FILE", None)

# Whether to use scheduling group barriers (needs LLVM fix).
enable_scheduling_barriers = os.environ.get("WAVE_USE_SCHED_BARRIERS", None)

# Whether to verify each individual kernels when there are multiple ones.
check_individual_kernels = os.environ.get("WAVE_CHECK_INDIV_KERNS", None)


def set_default_run_config(options: WaveCompileOptions) -> WaveCompileOptions:
    """Return default config for running."""

    if dump_generated_mlir:
        options.print_mlir = True

    if dump_generated_mlir_file is not None and not options.print_mlir_file:
        # Do not override file if `print_mlir_file` is already set.
        options.print_mlir_file = str(dump_generated_mlir_file)

    if enable_scheduling_barriers is not None:
        options.use_scheduling_barriers = bool(int(enable_scheduling_barriers))

    options.backend = "rocm"
    options.target = get_default_arch()
    return options
