// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <Python.h>
#include <cstdint>

/// StridedMemRefType is the descriptor structure used by MLIR for memrefs.
/// This matches the ABI used by MLIR's memref lowering.
template <typename T, int N> struct StridedMemRefType {
  T *basePtr;         // Pointer to the allocated buffer
  T *data;            // Aligned data pointer
  int64_t offset;     // Offset in elements
  int64_t sizes[N];   // Size of each dimension
  int64_t strides[N]; // Stride of each dimension in elements
};

/// Rank-1 memref descriptor for memref<?xi8>, we need to pass this to
/// memref.view op which only accepts 1D i8 memrefs.
using MemRef1Di8 = StridedMemRefType<uint8_t, 1>;

extern "C" {

/// Extract a raw buffer pointer from a PyObject (PyTorch tensor).
/// Returns a rank-1 memref descriptor: memref<?xi8>
///
/// The returned descriptor has:
/// - basePtr: pointer to the raw data
/// - data: same as basePtr (no alignment offset)
/// - offset: 0
/// - sizes[0]: -1
/// - strides[0]: 1
///
/// This function assumes the PyObject is a PyTorch tensor and uses
/// the PyTorch C API to extract the data pointer and size.
void _mlir_ciface_wave_get_buffer(MemRef1Di8 *ret, PyObject *obj);

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
