// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstddef>

extern "C" {
void load_functions();

void *wave_load_kernel(void *stream, void **cached_kernel_handle,
                       const void *binary_pointer, size_t binary_size,
                       const char *kernel_name);

void wave_launch_kernel(void *stream, void *function, int shared_memory_bytes,
                        int grid_x, int grid_y, int grid_z, int block_x,
                        int block_y, int block_z, void **args, int num_args);
}
