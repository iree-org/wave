// Copyright 2025 The IREE Authors
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

/// Rank-1 memref descriptor for memref<?xi8>
using MemRef1Di8 = StridedMemRefType<uint8_t, 1>;

extern "C" {

/// Extract a raw buffer pointer from a PyObject (PyTorch tensor).
/// Returns a rank-1 memref descriptor: memref<?xi8>
///
/// The returned descriptor has:
/// - basePtr: pointer to the raw data
/// - data: same as basePtr (no alignment offset)
/// - offset: 0
/// - sizes[0]: number of elements
/// - strides[0]: 1
///
/// This function assumes the PyObject is a PyTorch tensor and uses
/// the PyTorch C API to extract the data pointer and size.
void _mlir_ciface_wave_get_buffer(MemRef1Di8 *ret, PyObject *obj);

/// Extract an int64_t value from a PyObject.
/// Throws std::runtime_error if conversion fails.
int64_t wave_get_int64(PyObject *obj);

/// Extract a double value from a PyObject.
/// Throws std::runtime_error if conversion fails.
double wave_get_float64(PyObject *obj);
}
