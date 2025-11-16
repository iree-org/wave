// Copyright 2025 The IREE Authors
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "buffer_utils.h"
#include <Python.h>

#include <memory>

namespace {
struct GILState {
  GILState() : gstate(PyGILState_Ensure()) {}
  ~GILState() { PyGILState_Release(gstate); }
  PyGILState_STATE gstate;
};

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
    return;
  }

  PyObjectPtr data_ptr_result(PyObject_CallNoArgs(data_ptr_method.get()));

  if (!data_ptr_result) {
    PyErr_Clear();
    return;
  }

  // Convert Python int to pointer
  void *data_ptr = PyLong_AsVoidPtr(data_ptr_result.get());

  if (!data_ptr && PyErr_Occurred()) {
    PyErr_Clear();
    return;
  }

  // Get tensor.numel() method and call it
  PyObjectPtr numel_method(PyObject_GetAttrString(obj_ptr, "numel"));
  if (!numel_method) {
    PyErr_Clear();
    return;
  }

  PyObjectPtr numel_result(PyObject_CallNoArgs(numel_method.get()));

  if (!numel_result) {
    PyErr_Clear();
    return;
  }

  int64_t numel = PyLong_AsLongLong(numel_result.get());

  if (PyErr_Occurred()) {
    PyErr_Clear();
    return;
  }

  // Fill in the memref descriptor
  ret->basePtr = static_cast<uint8_t *>(data_ptr);
  ret->data = static_cast<uint8_t *>(data_ptr);
  ret->offset = 0;
  ret->sizes[0] = numel;
  ret->strides[0] = 1;
}
