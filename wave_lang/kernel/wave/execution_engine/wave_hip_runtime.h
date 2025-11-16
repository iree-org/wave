// Copyright 2025 The IREE Authors
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

/// Opaque kernel handle type
typedef void *WaveKernelHandle;

/// Load a GPU kernel from a binary file
///
/// Args:
///   binary_path: Path to the kernel binary (.hsaco file)
///   kernel_name: Name of the kernel function to load
///
/// Returns:
///   Opaque kernel handle, or nullptr on failure
WaveKernelHandle wave_load_kernel(const char *binary_path,
                                  const char *kernel_name);

/// Launch a GPU kernel
///
/// Args:
///   handle: Kernel handle from wave_load_kernel
///   stream: HIP stream pointer
///   grid_x, grid_y, grid_z: Grid dimensions
///   block_x, block_y, block_z: Block dimensions
///   args: Pointer to array of kernel argument pointers
///   num_args: Number of kernel arguments
///
/// Returns:
///   0 on success, non-zero on failure
int wave_launch_kernel(WaveKernelHandle handle, void *stream, uint32_t grid_x,
                       uint32_t grid_y, uint32_t grid_z, uint32_t block_x,
                       uint32_t block_y, uint32_t block_z, void **args,
                       size_t num_args);

/// Unload a GPU kernel
///
/// Args:
///   handle: Kernel handle from wave_load_kernel
void wave_unload_kernel(WaveKernelHandle handle);
}
