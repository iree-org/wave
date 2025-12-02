// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "buffer_utils.h"
#include <Python.h>

#include <memory>
#include <stdexcept>

namespace {
/// RAII wrapper for acquiring and releasing the GIL.
struct GILState {
  GILState() : gstate(PyGILState_Ensure()) {}
  ~GILState() { PyGILState_Release(gstate); }
  PyGILState_STATE gstate;
};

/// Deleter for std::unique_ptr<PyObject>.
struct PyDeleter {
  void operator()(PyObject *obj) const { Py_DECREF(obj); }
};

using PyObjectPtr = std::unique_ptr<PyObject, PyDeleter>;
} // namespace

extern "C" void _mlir_ciface_wave_get_buffer(MemRef1Di8 *ret,
                                             PyObject *obj_ptr) {
  GILState gil_state;

  // Get tensor.data_ptr() method and call it
  PyObjectPtr data_ptr_method(PyObject_GetAttrString(obj_ptr, "data_ptr"));
  if (!data_ptr_method) {
    PyErr_Clear();
    throw std::runtime_error(
        "wave_get_buffer: Object does not have 'data_ptr' attribute");
  }

  PyObjectPtr data_ptr_result(PyObject_CallNoArgs(data_ptr_method.get()));
  if (!data_ptr_result) {
    PyErr_Clear();
    throw std::runtime_error("wave_get_buffer: Failed to call data_ptr()");
  }

  // Convert Python int to pointer
  void *data_ptr = PyLong_AsVoidPtr(data_ptr_result.get());
  if (!data_ptr && PyErr_Occurred()) {
    PyErr_Clear();
    throw std::runtime_error(
        "wave_get_buffer: data_ptr() did not return a valid pointer");
  }

  // Fill in the memref descriptor
  ret->basePtr = static_cast<uint8_t *>(data_ptr);
  ret->data = static_cast<uint8_t *>(data_ptr);
  ret->offset = 0;
  // Actual size doesn't matter we will cast it to 0D memref immediately.
  ret->sizes[0] = -1;
  ret->strides[0] = 1;
}

extern "C" int64_t _mlir_ciface_wave_get_int64(PyObject *obj_ptr) {
  GILState gil_state;

  int64_t value = PyLong_AsLongLong(obj_ptr);
  if (PyErr_Occurred()) {
    PyErr_Clear();
    throw std::runtime_error("wave_get_int64: Failed to convert to int64");
  }

  return value;
}

extern "C" double _mlir_ciface_wave_get_float64(PyObject *obj_ptr) {
  GILState gil_state;

  double value = PyFloat_AsDouble(obj_ptr);
  if (PyErr_Occurred()) {
    PyErr_Clear();
    throw std::runtime_error("wave_get_float64: Failed to convert to double");
  }

  return value;
}

extern "C" int64_t _mlir_ciface_wave_get_dim(PyObject *obj_ptr,
                                             int32_t dim_idx) {
  GILState gil_state;

  // Get tensor.size() method
  PyObjectPtr size_method(PyObject_GetAttrString(obj_ptr, "size"));
  if (!size_method) {
    PyErr_Clear();
    throw std::runtime_error(
        "wave_get_dim: Object does not have 'size' attribute");
  }

  // Call tensor.size(dim_idx)
  PyObjectPtr dim_arg(PyLong_FromLong(dim_idx));
  if (!dim_arg) {
    PyErr_Clear();
    throw std::runtime_error(
        "wave_get_dim: Failed to create dimension argument");
  }

  PyObjectPtr size_result(
      PyObject_CallOneArg(size_method.get(), dim_arg.get()));
  if (!size_result) {
    PyErr_Clear();
    throw std::runtime_error("wave_get_dim: Failed to call size()");
  }

  // Convert result to int64_t
  int64_t dim_size = PyLong_AsLongLong(size_result.get());
  if (PyErr_Occurred()) {
    PyErr_Clear();
    throw std::runtime_error("wave_get_dim: size() returned invalid value");
  }

  return dim_size;
}
