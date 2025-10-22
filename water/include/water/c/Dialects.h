// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_C_DIALECTS_H
#define WATER_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Wave, wave);

//===---------------------------------------------------------------------===//
// WaveSymbolAttr
//===---------------------------------------------------------------------===//

/// Checks whether the given MLIR attribute is a WaveSymbolAttr.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAWaveSymbolAttr(MlirAttribute attr);

/// Creates a new WaveSymbolAttr with the given symbol name.
MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveSymbolAttrGet(MlirContext mlirCtx, MlirStringRef symbolName);

/// Returns the typeID of a WaveSymbolAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWaveSymbolAttrGetTypeID();

//===---------------------------------------------------------------------===//
// WaveIndexMappingAttr
//===---------------------------------------------------------------------===//

/// Checks whether the given MLIR attribute is a WaveIndexMappingAttr.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsAWaveIndexMappingAttr(MlirAttribute attr);

/// Creates a new WaveIndexMappingAttr with the given start, step and stride
/// maps that are interpreted as accepting the symbols provided in the
/// `symbolNames` list. The list must have as many entries as maps have symbols,
/// and all maps must have the same number of symbols and zero dimensions. The
/// list is expected to only contain WaveSymbolAttr instances.
MLIR_CAPI_EXPORTED MlirAttribute mlirWaveIndexMappingAttrGet(
    MlirContext mlirCtx, MlirAttribute *symbolNames, MlirAffineMap start,
    MlirAffineMap step, MlirAffineMap stride);

/// Returns the typeID of a WaveIndexMappingAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWaveIndexMappingAttrGetTypeID();

//===---------------------------------------------------------------------===//
// WaveHyperparameterAttr
//===---------------------------------------------------------------------===//

/// Checks whether the given MLIR attribute is a WaveHyperparameterAttr.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsAWaveHyperparameterAttr(MlirAttribute attr);

/// Creates a new WaveHyperparameterAttr with the given mapping from symbol
/// names to their concrete integer values.
MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveHyperparameterAttrGet(MlirAttribute mapping);

/// Returns the typeID of a WaveHyperparameterAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWaveHyperparameterAttrGetTypeID();

//===---------------------------------------------------------------------===//
// WaveAddressSpaceAttr
//===---------------------------------------------------------------------===//

/// Checks whether the given MLIR attribute is a WaveAddressSpaceAttr.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsAWaveAddressSpaceAttr(MlirAttribute attr);

/// Creates a new WaveAddressSpaceAttr with the given value.
MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveAddressSpaceAttrGet(MlirContext mlirCtx, uint32_t value);

// Get the value from a WaveAddressSpaceAttr.
MLIR_CAPI_EXPORTED uint32_t
mlirWaveAddressSpaceAttrGetValue(MlirAttribute attr);

/// Returns the typeID of a WaveAddressSpaceAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWaveAddressSpaceAttrGetTypeID();

//===---------------------------------------------------------------------===//
// WaveExprListAttr
//===---------------------------------------------------------------------===//

/// Checks whether the given MLIR attribute is a WaveExprListAttr.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAWaveExprListAttr(MlirAttribute attr);

/// Creates a new WaveExprListAttr with the given map that is
/// interpreted as accepting the symbols provided in the
/// `symbolNames` list. The list must have as many entries as maps have symbols,
/// and all maps must have the same number of symbols and zero dimensions. The
/// list is expected to only contain WaveSymbolAttr instances.
MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveExprListAttrGet(MlirAttribute *symbolNames, MlirAffineMap map);

/// Returns the typeID of a WaveExprListAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWaveExprListAttrGetTypeID();

//===---------------------------------------------------------------------===//
// WaveReadWriteBoundsAttr
//===---------------------------------------------------------------------===//

/// Checks whether the given MLIR attribute is a WaveReadWriteBoundsAttr.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsAWaveReadWriteBoundsAttr(MlirAttribute attr);

/// Creates a new WaveReadWriteBoundsAttr with the given mapping from symbolic
/// dimensions to their bound expressions.
MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveReadWriteBoundsAttrGet(MlirAttribute mapping);

/// Returns the typeID of a WaveReadWriteBoundsAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWaveReadWriteBoundsAttrGetTypeID();

#ifdef __cplusplus
}
#endif

#endif // WATER_C_DIALECTS_H
