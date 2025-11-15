// Copyright 2025 The IREE Authors
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "buffer_utils.h"
#include <Python.h>
#include <cstring>
#include <stdexcept>

// PyTorch C API definitions
// We use weak symbols so the code compiles even if PyTorch is not available
// The symbols will be resolved at runtime when PyTorch is loaded

extern "C" {
// PyTorch Tensor C API functions (from torch/csrc/Module.h)
void *__attribute__((weak)) THPVariable_Unpack(PyObject *obj);
void *__attribute__((weak)) at_tensor_data_ptr(void *tensor);
int64_t __attribute__((weak)) at_tensor_numel(void *tensor);
int64_t __attribute__((weak)) at_tensor_element_size(void *tensor);
}

/// Helper to check if PyTorch symbols are available
static bool isPyTorchAvailable() {
  return THPVariable_Unpack != nullptr && at_tensor_data_ptr != nullptr &&
         at_tensor_numel != nullptr && at_tensor_element_size != nullptr;
}

extern "C" MemRef1Di8 wave_get_buffer(PyObject *obj) {
  if (!obj) {
    throw std::runtime_error("wave_get_buffer: NULL PyObject");
  }

  // Check if PyTorch is available
  if (!isPyTorchAvailable()) {
    throw std::runtime_error(
        "wave_get_buffer: PyTorch C API symbols not found. "
        "Make sure PyTorch is loaded before calling this function.");
  }

  // Extract the ATen tensor from the PyTorch Python object
  void *tensor = THPVariable_Unpack(obj);
  if (!tensor) {
    throw std::runtime_error(
        "wave_get_buffer: Failed to unpack PyTorch tensor. "
        "Object is not a valid torch.Tensor.");
  }

  // Get the data pointer
  void *data_ptr = at_tensor_data_ptr(tensor);
  if (!data_ptr) {
    throw std::runtime_error("wave_get_buffer: Tensor has NULL data pointer");
  }

  // Calculate total size in bytes
  int64_t numel = at_tensor_numel(tensor);
  int64_t element_size = at_tensor_element_size(tensor);
  int64_t total_bytes = numel * element_size;

  // Create and return memref descriptor
  MemRef1Di8 descriptor;
  descriptor.basePtr = static_cast<uint8_t *>(data_ptr);
  descriptor.data = static_cast<uint8_t *>(data_ptr);
  descriptor.offset = 0;
  descriptor.sizes[0] = total_bytes;
  descriptor.strides[0] = 1;

  return descriptor;
}
