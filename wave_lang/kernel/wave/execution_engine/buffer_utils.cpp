// Copyright 2025 The IREE Authors
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "buffer_utils.h"
#include <nanobind/nanobind.h>
#include <stdexcept>

namespace nb = nanobind;

extern "C" void _mlir_ciface_wave_get_buffer(MemRef1Di8 *ret,
                                             PyObject *obj_ptr) {
  // Wrap PyObject* in nanobind::object for safe access
  nb::object obj = nb::borrow(obj_ptr);

  // Call tensor.data_ptr() to get the data pointer
  nb::object data_ptr_result = obj.attr("data_ptr")();
  void *data_ptr =
      reinterpret_cast<void *>(nb::cast<uintptr_t>(data_ptr_result));

  // Get tensor.numel() for the number of elements
  int64_t numel = nb::cast<int64_t>(obj.attr("numel")());

  // Create and return memref descriptor
  MemRef1Di8 descriptor;
  descriptor.basePtr = static_cast<uint8_t *>(data_ptr);
  descriptor.data = static_cast<uint8_t *>(data_ptr);
  descriptor.offset = 0;
  descriptor.sizes[0] = numel;
  descriptor.strides[0] = 1;

  *ret = descriptor;
}
