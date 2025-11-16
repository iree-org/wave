// Copyright 2025 The IREE Authors
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "buffer_utils.h"
#include <Python.h>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <string>

// PyTorch Tensor C API function pointers
typedef void *(*THPVariable_Unpack_t)(PyObject *);
typedef void *(*at_tensor_data_ptr_t)(void *);
typedef int64_t (*at_tensor_numel_t)(void *);
typedef int64_t (*at_tensor_element_size_t)(void *);

// Global function pointers (initialized on first use)
static THPVariable_Unpack_t THPVariable_Unpack_ptr = nullptr;
static at_tensor_data_ptr_t at_tensor_data_ptr_ptr = nullptr;
static at_tensor_numel_t at_tensor_numel_ptr = nullptr;

// Helper to get symbol address
static void *get_symbol_address(void *handle, const char *symbol_name) {
  return dlsym(handle, symbol_name);
}

// Macro to load a function pointer and check for errors
#define GET_FUNC(handle, name)                                                 \
  do {                                                                         \
    name =                                                                     \
        reinterpret_cast<decltype(name)>(get_symbol_address(handle, #name));   \
    if (!name) {                                                               \
      throw std::runtime_error("Failed to load PyTorch symbol: " +             \
                               std::string(#name));                            \
    }                                                                          \
  } while (0)

/// Initialize PyTorch C API function pointers using dlsym
static void initPyTorchSymbols() {
  if (THPVariable_Unpack_ptr != nullptr) {
    return; // Already initialized
  }

  // Use RTLD_DEFAULT to search all loaded libraries
  void *handle = RTLD_DEFAULT;

  GET_FUNC(handle, THPVariable_Unpack_ptr);
  GET_FUNC(handle, at_tensor_data_ptr_ptr);
  GET_FUNC(handle, at_tensor_numel_ptr);
}

extern "C" MemRef1Di8 wave_get_buffer(PyObject *obj) {
  // Initialize PyTorch symbols on first use
  initPyTorchSymbols();

  // Extract the ATen tensor from the PyTorch Python object
  void *tensor = THPVariable_Unpack_ptr(obj);
  if (!tensor) {
    throw std::runtime_error(
        "wave_get_buffer: Failed to unpack PyTorch tensor. "
        "Object is not a valid torch.Tensor.");
  }

  // Get the data pointer
  void *data_ptr = at_tensor_data_ptr_ptr(tensor);
  if (!data_ptr) {
    throw std::runtime_error("wave_get_buffer: Tensor has NULL data pointer");
  }

  // Calculate total size in bytes
  int64_t numel = at_tensor_numel_ptr(tensor);

  // Create and return memref descriptor
  MemRef1Di8 descriptor;
  descriptor.basePtr = static_cast<uint8_t *>(data_ptr);
  descriptor.data = static_cast<uint8_t *>(data_ptr);
  descriptor.offset = 0;
  descriptor.sizes[0] = numel;
  descriptor.strides[0] = 1;

  return descriptor;
}
