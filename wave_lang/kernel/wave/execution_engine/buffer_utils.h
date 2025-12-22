// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <Python.h>
#include <cstdint>

extern "C" {

/// Extract a raw buffer pointer from a PyObject (PyTorch tensor).
void *_mlir_ciface_wave_get_buffer(PyObject *obj);

/// Extract an int64_t value from a PyObject.
/// Throws std::runtime_error if conversion fails.
int64_t _mlir_ciface_wave_get_int64(PyObject *obj);

/// Extract a double value from a PyObject.
/// Throws std::runtime_error if conversion fails.
double _mlir_ciface_wave_get_float64(PyObject *obj);

/// Extract the size of a specific dimension from a PyObject (PyTorch tensor).
///
/// Args:
///   obj: PyObject* pointing to a PyTorch tensor
///   dim_idx: Dimension index to query (0-based)
///
/// Returns:
///   Size of the specified dimension as int64_t
///
/// Throws:
///   std::runtime_error if the object doesn't have a size() method or
///   if the dimension index is invalid
int64_t _mlir_ciface_wave_get_dim(PyObject *obj, int32_t dim_idx);
}
