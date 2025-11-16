// Copyright 2025 The IREE Authors
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "wave_hip_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// TODO: Include HIP headers when implementing
// #include <hip/hip_runtime.h>

extern "C" WaveKernelHandle wave_load_kernel(const char *binary_path,
                                             const char *kernel_name) {
  // TODO: Implement kernel loading
  // 1. Load binary file
  // 2. Use hipModuleLoadData or similar
  // 3. Get kernel function handle
  // 4. Return opaque handle

  fprintf(stderr,
          "wave_load_kernel: stub implementation (binary=%s, kernel=%s)\n",
          binary_path, kernel_name);
  return nullptr;
}

extern "C" int wave_launch_kernel(WaveKernelHandle handle, void *stream,
                                  uint32_t grid_x, uint32_t grid_y,
                                  uint32_t grid_z, uint32_t block_x,
                                  uint32_t block_y, uint32_t block_z,
                                  void **args, size_t num_args) {
  // TODO: Implement kernel launch
  // 1. Validate handle
  // 2. Set up launch parameters
  // 3. Use hipModuleLaunchKernel or similar
  // 4. Return status

  fprintf(stderr,
          "wave_launch_kernel: stub implementation (grid=[%u,%u,%u], "
          "block=[%u,%u,%u], args=%zu)\n",
          grid_x, grid_y, grid_z, block_x, block_y, block_z, num_args);
  return -1; // Not implemented
}

extern "C" void wave_unload_kernel(WaveKernelHandle handle) {
  // TODO: Implement kernel unloading
  // 1. Free module resources
  // 2. Clean up handle

  fprintf(stderr, "wave_unload_kernel: stub implementation\n");
}
