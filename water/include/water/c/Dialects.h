// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_C_DIALECTS_H
#define WATER_C_DIALECTS_H

#include "mlir-c/AffineMap.h"
#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Wave, wave);

/// Register the Wave dialect passes.
MLIR_CAPI_EXPORTED void mlirWaveDialectRegisterPasses();

//===---------------------------------------------------------------------===//
// Wave Dialect Constants
//===---------------------------------------------------------------------===//

/// The attribute name for wave constraints.
MLIR_CAPI_EXPORTED extern const char *const mlirWaveDialectConstraintsAttrName;

//===---------------------------------------------------------------------===//
// WaveTensorType
//===---------------------------------------------------------------------===//

/// Checks whether the given MLIR type is a WaveTensorType.
MLIR_CAPI_EXPORTED bool mlirTypeIsAWaveTensorType(MlirType type);

/// Returns the typeID of a WaveTensorType.
MLIR_CAPI_EXPORTED MlirTypeID mlirWaveTensorTypeGetTypeID();

/// Returns true if the WaveTensorType is fully specified.
MLIR_CAPI_EXPORTED bool mlirWaveTensorTypeGetFullySpecified(MlirType type);

/// Returns the number of symbols in the WaveTensorType shape.
MLIR_CAPI_EXPORTED intptr_t mlirWaveTensorTypeGetShapeSize(MlirType type);

/// Returns the symbol at the given index in the WaveTensorType shape.
MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveTensorTypeGetShapeSymbol(MlirType type, intptr_t index);

/// Returns the element type of the WaveTensorType.
MLIR_CAPI_EXPORTED MlirType mlirWaveTensorTypeGetElementType(MlirType type);

/// Returns the address space attribute of the WaveTensorType.
MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveTensorTypeGetAddressSpace(MlirType type);

/// Creates a WaveTensorType from shape symbols and element/address space.
MLIR_CAPI_EXPORTED MlirType mlirWaveTensorTypeGet(
    MlirContext mlirCtx, MlirAttribute *shapeSymbols, intptr_t numShapeSymbols,
    bool fullySpecified, MlirType elementType, MlirAttribute addressSpace);

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

/// Gets the name of a WaveSymbolAttr.
MLIR_CAPI_EXPORTED MlirStringRef mlirWaveSymbolAttrGetName(MlirAttribute attr);

//===---------------------------------------------------------------------===//
// WaveIterSymbolAttr
//===---------------------------------------------------------------------===//

/// Checks whether the given MLIR attribute is a WaveIterSymbolAttr.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAWaveIterSymbolAttr(MlirAttribute attr);

/// Creates a new WaveIterSymbolAttr with the given induction variable name.
MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveIterSymbolAttrGet(MlirContext mlirCtx, MlirStringRef symbolName);

/// Returns the typeID of a WaveIterSymbolAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWaveIterSymbolAttrGetTypeID();

/// Gets the induction variable name.
MLIR_CAPI_EXPORTED MlirStringRef
mlirWaveIterSymbolAttrGetName(MlirAttribute attr);

//===---------------------------------------------------------------------===//
// WaveOperandAttr
//===---------------------------------------------------------------------===//

/// Checks whether the given MLIR attribute is a WaveOperandAttr.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAWaveOperandAttr(MlirAttribute attr);

/// Creates a new WaveOperandAttr with the given operand number.
MLIR_CAPI_EXPORTED MlirAttribute mlirWaveOperandAttrGet(MlirContext mlirCtx,
                                                        unsigned operandNumber);

/// Returns the typeID of a WaveOperandAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWaveOperandAttrGetTypeID();

/// Gets the operand number.
MLIR_CAPI_EXPORTED unsigned
mlirWaveOperandAttrGetOperandNumber(MlirAttribute attr);

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

/// Get the start from a WaveIndexMappingAttr.
MLIR_CAPI_EXPORTED MlirAffineMap
mlirWaveIndexMappingAttrGetStart(MlirAttribute attr);

/// Get the step from a WaveIndexMappingAttr.
MLIR_CAPI_EXPORTED MlirAffineMap
mlirWaveIndexMappingAttrGetStep(MlirAttribute attr);

/// Get the stride from a WaveIndexMappingAttr.
MLIR_CAPI_EXPORTED MlirAffineMap
mlirWaveIndexMappingAttrGetStride(MlirAttribute attr);

/// Get the number of (input) symbols.
MLIR_CAPI_EXPORTED intptr_t
mlirWaveIndexMappingAttrGetNumSymbols(MlirAttribute attr);

/// Get the (input) symbol at the given index.
MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveIndexMappingAttrGetSymbol(MlirAttribute attr, intptr_t index);

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

/// Gets the underlying dictionary mapping from a WaveHyperparameterAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveHyperparameterAttrGetMapping(MlirAttribute attr);

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
// WaveReductionScopeAttr
//===---------------------------------------------------------------------===//

enum WaveReductionScope {
  WaveReductionScopeBlock = 0,
  WaveReductionScopeWarp = 1,
};

/// Checks whether the given MLIR attribute is a WaveReductionScopeAttr.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsAWaveReductionScopeAttr(MlirAttribute attr);

/// Creates a new WaveReductionScopeAttr with the given value.
MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveReductionScopeAttrGet(MlirContext mlirCtx, uint32_t value);

/// Get the value from a WaveReductionScopeAttr.
MLIR_CAPI_EXPORTED uint32_t
mlirWaveReductionScopeAttrGetValue(MlirAttribute attr);

/// Returns the typeID of a WaveReductionScopeAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWaveReductionScopeAttrGetTypeID();

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
// WaveShuffleModeAttr
//===---------------------------------------------------------------------===//

enum WaveShuffleMode {
  WaveShuffleModeXOR = 0,
  WaveShuffleModeDOWN = 1,
  WaveShuffleModeUP = 2,
  WaveShuffleModeIDX = 3,
};

/// Checks whether the given MLIR attribute is a WaveShuffleModeAttr.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAWaveShuffleModeAttr(MlirAttribute attr);

/// Creates a new WaveShuffleModeAttr with the given value.
MLIR_CAPI_EXPORTED MlirAttribute mlirWaveShuffleModeAttrGet(MlirContext mlirCtx,
                                                            uint32_t value);

/// Get the value from a WaveShuffleModeAttr.
MLIR_CAPI_EXPORTED uint32_t mlirWaveShuffleModeAttrGetValue(MlirAttribute attr);

/// Returns the typeID of a WaveShuffleModeAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWaveShuffleModeAttrGetTypeID();

//===---------------------------------------------------------------------===//
// WaveApplyExprCombinatorAttr
//===---------------------------------------------------------------------===//

enum WaveApplyExprCombinator {
  WaveApplyExprCombinatorGreater = 0,
  WaveApplyExprCombinatorLess = 1,
  WaveApplyExprCombinatorEqual = 2,
  WaveApplyExprCombinatorNotEqual = 3,
  WaveApplyExprCombinatorGreaterOrEqual = 4,
  WaveApplyExprCombinatorLessOrEqual = 5,
  WaveApplyExprCombinatorMaximum = 6,
  WaveApplyExprCombinatorMinimum = 7,
};

/// Checks whether the given MLIR attribute is a WaveApplyExprCombinatorAttr.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsAWaveApplyExprCombinatorAttr(MlirAttribute attr);

/// Creates a new WaveApplyExprCombinatorAttr with the given value.
MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveApplyExprCombinatorAttrGet(MlirContext mlirCtx, uint32_t value);

/// Get the value from a WaveApplyExprCombinatorAttr.
MLIR_CAPI_EXPORTED uint32_t
mlirWaveApplyExprCombinatorAttrGetValue(MlirAttribute attr);

/// Returns the typeID of a WaveApplyExprCombinatorAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWaveApplyExprCombinatorAttrGetTypeID();

//===---------------------------------------------------------------------===//
// WaveMmaKindAttr
//===---------------------------------------------------------------------===//

// TODO: instead of hardcoding and running the risk of divergence, make this
// generated from tablegen. Consider doing the same for Python.
enum WaveMmaKind {
  WaveMmaKind_F32_16x16x16_F16 = 0x1020,
  WaveMmaKind_F32_32x32x8_F16 = 0x1021,
  WaveMmaKind_F32_16x16x32_K8_F16 = 0x1022,
  WaveMmaKind_F32_32x32x16_K8_F16 = 0x1023,
  WaveMmaKind_I32_16x16x16_I8 = 0x10C0,
  WaveMmaKind_I32_32x32x8_I8 = 0x10C1,

  WaveMmaKind_F32_16x16x32_F8 = 0x1230,
  WaveMmaKind_F32_32x32x16_F8 = 0x1231,
  WaveMmaKind_F32_16x16x32_K4_F8 = 0x1232,
  WaveMmaKind_F32_32x32x16_K4_F8 = 0x1233,
  WaveMmaKind_I32_16x16x32_I8 = 0x12C0,
  WaveMmaKind_I32_32x32x16_I8 = 0x12C1,

  WaveMmaKind_F32_16x16x128_F8F6F4 = 0x1340,
  WaveMmaKind_F32_32x32x64_F8F6F4 = 0x1341,
  WaveMmaKind_F32_16x16x32_F16 = 0x1320,
  WaveMmaKind_F32_32x32x16_F16 = 0x1321,
  WaveMmaKind_F32_16x16x32_BF16 = 0x1322,
  WaveMmaKind_F32_32x32x16_BF16 = 0x1323,
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

/// Get the affine map from a WaveExprListAttr.
MLIR_CAPI_EXPORTED MlirAffineMap mlirWaveExprListAttrGetMap(MlirAttribute attr);

/// Get the number of symbols in a WaveExprListAttr.
MLIR_CAPI_EXPORTED intptr_t
mlirWaveExprListAttrGetNumSymbols(MlirAttribute attr);

/// Get the symbol at index from a WaveExprListAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveExprListAttrGetSymbol(MlirAttribute attr, intptr_t index);

//===---------------------------------------------------------------------===//
// WaveSymbolMappingAttr
//===---------------------------------------------------------------------===//

/// Checks whether the given MLIR attribute is a WaveSymbolMappingAttr.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsAWaveSymbolMappingAttr(MlirAttribute attr);

/// Creates a new WaveSymbolMappingAttr from parallel arrays of keys (symbol
/// attrs) and values (expr list attrs).
MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveSymbolMappingAttrGet(MlirContext ctx, intptr_t numEntries,
                             MlirAttribute *keys, MlirAttribute *values);

/// Returns the number of entries in a WaveSymbolMappingAttr.
MLIR_CAPI_EXPORTED intptr_t
mlirWaveSymbolMappingAttrGetNumEntries(MlirAttribute attr);

/// Returns the key at the given index.
MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveSymbolMappingAttrGetKey(MlirAttribute attr, intptr_t index);

/// Returns the value at the given index.
MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveSymbolMappingAttrGetValue(MlirAttribute attr, intptr_t index);

/// Returns the value for the given key or null if the key is not present.
MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveSymbolMappingAttrLookup(MlirAttribute attr, MlirAttribute key);

/// Returns the typeID of a WaveSymbolMappingAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWaveSymbolMappingAttrGetTypeID();

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

/// Field getters for HardwareConstraintAttr.
MLIR_CAPI_EXPORTED unsigned
mlirHardwareConstraintAttrGetThreadsPerWave(MlirAttribute attr);
MLIR_CAPI_EXPORTED intptr_t
mlirHardwareConstraintAttrGetNumWavesPerBlock(MlirAttribute attr);
MLIR_CAPI_EXPORTED unsigned
mlirHardwareConstraintAttrGetWavesPerBlockElem(MlirAttribute attr, intptr_t i);
MLIR_CAPI_EXPORTED MlirAttribute
mlirHardwareConstraintAttrGetMmaType(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirHardwareConstraintAttrGetVectorShapes(MlirAttribute attr);
MLIR_CAPI_EXPORTED unsigned
mlirHardwareConstraintAttrGetMaxBitsPerLoad(MlirAttribute attr);

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

/// Field getters.
MLIR_CAPI_EXPORTED MlirAttribute
mlirDeviceConstraintAttrGetDim(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirDeviceConstraintAttrGetTileSize(MlirAttribute attr);

MLIR_CAPI_EXPORTED unsigned
mlirDeviceConstraintAttrGetDeviceDim(MlirAttribute attr);

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

/// Field getters.
MLIR_CAPI_EXPORTED MlirAttribute
mlirWorkgroupConstraintAttrGetDim(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirWorkgroupConstraintAttrGetTileSize(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirWorkgroupConstraintAttrGetWorkgroupDim(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool
mlirWorkgroupConstraintAttrGetPrimary(MlirAttribute attr);

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

/// Field getters.
MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveConstraintAttrGetDim(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveConstraintAttrGetTileSize(MlirAttribute attr);

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

/// Field getters.
MLIR_CAPI_EXPORTED MlirAttribute
mlirTilingConstraintAttrGetDim(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirTilingConstraintAttrGetTileSize(MlirAttribute attr);

//===---------------------------------------------------------------------===//
// Wave Operations
//===---------------------------------------------------------------------===//

/// Makes a wave.iterate operation's region isolated from above.
/// This transforms the region to not capture values from outer scopes,
/// instead passing them explicitly as operands.
MLIR_CAPI_EXPORTED void mlirWaveIterateOpMakeIsolated(MlirOperation op);

/// Makes a wave.iterate operation's region non-isolated from above.
/// This allows the region to capture values from outer scopes implicitly.
MLIR_CAPI_EXPORTED void mlirWaveIterateOpMakeNonIsolated(MlirOperation op);

#ifdef __cplusplus
}
#endif

#endif // WATER_C_DIALECTS_H
