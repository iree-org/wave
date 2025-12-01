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
// Wave Dialect Constants
//===---------------------------------------------------------------------===//

/// The attribute name for wave constraints.
MLIR_CAPI_EXPORTED extern const char *const mlirWaveDialectConstraintsAttrName;

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
// WaveIndexSymbolAttr
//===---------------------------------------------------------------------===//

enum WaveIndexSymbol {
  WaveIndexSymbol_DEVICE_DIM_0 = 0,
  WaveIndexSymbol_DEVICE_DIM_1 = 1,
  WaveIndexSymbol_DEVICE_DIM_2 = 2,
  WaveIndexSymbol_WORKGROUP_0 = 3,
  WaveIndexSymbol_WORKGROUP_1 = 4,
  WaveIndexSymbol_WORKGROUP_2 = 5,
  WaveIndexSymbol_THREAD_0 = 6,
  WaveIndexSymbol_THREAD_1 = 7,
  WaveIndexSymbol_THREAD_2 = 8,
  WaveIndexSymbol_GPR_NUMBER = 9,
};

/// Checks whether the given MLIR attribute is a WaveIndexSymbolAttr.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAWaveIndexSymbolAttr(MlirAttribute attr);

/// Creates a new WaveIndexSymbolAttr with the given value.
MLIR_CAPI_EXPORTED MlirAttribute mlirWaveIndexSymbolAttrGet(MlirContext mlirCtx,
                                                            uint32_t value);

/// Get the value from a WaveIndexSymbolAttr.
MLIR_CAPI_EXPORTED uint32_t mlirWaveIndexSymbolAttrGetValue(MlirAttribute attr);

/// Returns the typeID of a WaveIndexSymbolAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWaveIndexSymbolAttrGetTypeID();

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
// WaveWorkgroupDimAttr
//===---------------------------------------------------------------------===//

enum WaveWorkgroupDim {
  WaveWorkgroupDimX = 0,
  WaveWorkgroupDimY = 1,
  WaveWorkgroupDimZ = 2,
};

/// Checks whether the given MLIR attribute is a WaveWorkgroupDimAttr.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsAWaveWorkgroupDimAttr(MlirAttribute attr);

/// Creates a new WaveWorkgroupDimAttr with the given value.
MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveWorkgroupDimAttrGet(MlirContext mlirCtx, uint32_t value);

/// Get the value from a WaveWorkgroupDimAttr.
MLIR_CAPI_EXPORTED uint32_t
mlirWaveWorkgroupDimAttrGetValue(MlirAttribute attr);

/// Returns the typeID of a WaveWorkgroupDimAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWaveWorkgroupDimAttrGetTypeID();

//===---------------------------------------------------------------------===//
// WaveAddressSpaceAttr
//===---------------------------------------------------------------------===//

enum WaveAddressSpace {
  WaveAddressSpaceUnspecified = 0,
  WaveAddressSpaceGlobal = 1,
  WaveAddressSpaceShared = 2,
  WaveAddressSpaceRegister = 3,
};

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
// WaveMmaKindAttr
//===---------------------------------------------------------------------===//

enum WaveMmaKind {
  WaveMmaKind_F32_16x16x16_F16 = 4128,
  WaveMmaKind_F32_32x32x8_F16 = 4129,
  WaveMmaKind_F32_16x16x32_K8_F16 = 4130,
  WaveMmaKind_F32_32x32x16_K8_F16 = 4131,
  WaveMmaKind_I32_16x16x16_I8 = 4288,
  WaveMmaKind_I32_32x32x8_I8 = 4289,
  WaveMmaKind_F32_16x16x32_F8 = 4656,
  WaveMmaKind_F32_32x32x16_F8 = 4657,
  WaveMmaKind_F32_16x16x32_K4_F8 = 4658,
  WaveMmaKind_F32_32x32x16_K4_F8 = 4659,
  WaveMmaKind_I32_16x16x32_I8 = 4800,
  WaveMmaKind_I32_32x32x16_I8 = 4801,
  WaveMmaKind_F32_16x16x128_F8F6F4 = 4928,
  WaveMmaKind_F32_32x32x64_F8F6F4 = 4929,
  WaveMmaKind_F32_16x16x32_F16 = 4896,
  WaveMmaKind_F32_32x32x16_F16 = 4897,
  WaveMmaKind_F32_16x16x32_BF16 = 4898,
  WaveMmaKind_F32_32x32x16_BF16 = 4899,
};

/// Checks whether the given MLIR attribute is a WaveMmaKindAttr.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAWaveMmaKindAttr(MlirAttribute attr);

/// Creates a new WaveMmaKindAttr with the given value.
MLIR_CAPI_EXPORTED MlirAttribute mlirWaveMmaKindAttrGet(MlirContext mlirCtx,
                                                        uint32_t value);

/// Get the value from a WaveMmaKindAttr.
MLIR_CAPI_EXPORTED uint32_t mlirWaveMmaKindAttrGetValue(MlirAttribute attr);

/// Returns the typeID of a WaveMmaKindAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWaveMmaKindAttrGetTypeID();

//===---------------------------------------------------------------------===//
// WaveNormalFormAttr
//===---------------------------------------------------------------------===//

/// Normal forms, this must remain consistent with WaveAttrs.td.
enum WaveNormalForm {
  WaveNormalFormNone = 0,
  WaveNormalFormFunctionBoundarySpecified = 1,
  WaveNormalFormOpTypesSpecified = 2,
  WaveNormalFormIndexExprsSpecified = 4,
  WaveNormalFormMemoryOnlyTypes = 8,

  WaveNormalFormAllTypesSPecified =
      WaveNormalFormFunctionBoundarySpecified | WaveNormalFormOpTypesSpecified
};

/// Checks whether the given MLIR attribute is a WaveNormalFormAttr.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAWaveNormalFormAttr(MlirAttribute attr);

/// Creates a new WaveNormalFormAttr with the given mapping attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirWaveNormalFormAttrGet(MlirContext ctx,
                                                           uint32_t value);

/// Get the value from a WaveNormalFormAttr.
MLIR_CAPI_EXPORTED uint32_t mlirWaveNormalFormAttrGetValue(MlirAttribute attr);

/// Returns the typeID of a WaveNormalFormAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWaveNormalFormAttrGetTypeID();

//===---------------------------------------------------------------------===//
// WaveExprListAttr
//===---------------------------------------------------------------------===//

/// Checks whether the given MLIR attribute is a WaveExprListAttr.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAWaveExprListAttr(MlirAttribute attr);

/// Creates a new WaveExprListAttr with the given map that is
/// interpreted as accepting the symbols provided in the
/// `symbolNames` list. The list must have as many entries as maps have symbols,
/// and all maps must have the same number of symbols and zero dimensions. The
/// list is expected to only contain WaveSymbolAttr or WaveIndexSymbolAttr
/// instances.
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

//===---------------------------------------------------------------------===//
// HardwareConstraintAttr
//===---------------------------------------------------------------------===//

/// Checks whether the given MLIR attribute is a HardwareConstraintAttr.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsAHardwareConstraintAttr(MlirAttribute attr);

/// Creates a new HardwareConstraintAttr
MLIR_CAPI_EXPORTED MlirAttribute mlirHardwareConstraintAttrGet(
    MlirContext mlirCtx, unsigned threadsPerWave, size_t wavesPerBlockSize,
    unsigned *wavesPerBlock, MlirAttribute mmaType, MlirAttribute vectorShapes,
    unsigned maxBitsPerLoad);

/// Returns the typeID of a HardwareConstraintAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWHardwareConstraintAttrGetTypeID();

//===---------------------------------------------------------------------===//
// DeviceConstraintAttr
//===---------------------------------------------------------------------===//

/// Checks whether the given MLIR attribute is a DeviceConstraintAttr.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsADeviceConstraintAttr(MlirAttribute attr);

/// Creates a new DeviceConstraintAttr
MLIR_CAPI_EXPORTED MlirAttribute
mlirDeviceConstraintAttrGet(MlirContext mlirCtx, MlirAttribute dim,
                            MlirAttribute tileSize, unsigned deviceDim);

/// Returns the typeID of a DeviceConstraintAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirDeviceConstraintAttrGetTypeID();

//===---------------------------------------------------------------------===//
// WorkgroupConstraintAttr
//===---------------------------------------------------------------------===//

/// Checks whether the given MLIR attribute is a WorkgroupConstraintAttr.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsAWorkgroupConstraintAttr(MlirAttribute attr);

/// Creates a new WorkgroupConstraintAttr
MLIR_CAPI_EXPORTED MlirAttribute mlirWorkgroupConstraintAttrGet(
    MlirContext mlirCtx, MlirAttribute dim, MlirAttribute tileSize,
    MlirAttribute workgroupDim, bool primary);

/// Returns the typeID of a WorkgroupConstraintAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWorkgroupConstraintAttrGetTypeID();

//===---------------------------------------------------------------------===//
// WaveConstraintAttr
//===---------------------------------------------------------------------===//

/// Checks whether the given MLIR attribute is a WaveConstraintAttr.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAWaveConstraintAttr(MlirAttribute attr);

/// Creates a new WaveConstraintAttr
MLIR_CAPI_EXPORTED MlirAttribute mlirWaveConstraintAttrGet(
    MlirContext mlirCtx, MlirAttribute dim, MlirAttribute tileSize);

/// Returns the typeID of a WaveConstraintAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWaveConstraintAttrGetTypeID();

//===---------------------------------------------------------------------===//
// TilingConstraintAttr
//===---------------------------------------------------------------------===//

/// Checks whether the given MLIR attribute is a TilingConstraintAttr.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsATilingConstraintAttr(MlirAttribute attr);

/// Creates a new TilingConstraintAttr
MLIR_CAPI_EXPORTED MlirAttribute mlirTilingConstraintAttrGet(
    MlirContext mlirCtx, MlirAttribute dim, MlirAttribute tileSize);

/// Returns the typeID of a TilingConstraintAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirTilingConstraintAttrGetTypeID();

#ifdef __cplusplus
}
#endif

#endif // WATER_C_DIALECTS_H
