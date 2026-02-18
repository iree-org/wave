// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir-c/AffineMap.h"
#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/TypeID.h"

#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"
#include "water/Dialect/Wave/Transforms/Passes.h"
#include "water/c/Dialects.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Wave, wave, ::wave::WaveDialect)

//===---------------------------------------------------------------------===//
// Wave Dialect Passes
//===---------------------------------------------------------------------===//

void mlirWaveDialectRegisterPasses() { wave::registerPasses(); }

//===---------------------------------------------------------------------===//
// Wave Dialect Constants
//===---------------------------------------------------------------------===//

const char *const mlirWaveDialectConstraintsAttrName =
    wave::WaveDialect::kWaveConstraintsAttrName.data();

//===---------------------------------------------------------------------===//
// WaveTensorType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAWaveTensorType(MlirType type) {
  return llvm::isa<wave::WaveTensorType>(unwrap(type));
}

MlirTypeID mlirWaveTensorTypeGetTypeID() {
  return wrap(TypeID::get<wave::WaveTensorType>());
}

bool mlirWaveTensorTypeGetFullySpecified(MlirType type) {
  return llvm::cast<wave::WaveTensorType>(unwrap(type)).getFullySpecified();
}

intptr_t mlirWaveTensorTypeGetShapeSize(MlirType type) {
  return llvm::cast<wave::WaveTensorType>(unwrap(type)).getShape().size();
}

MlirAttribute mlirWaveTensorTypeGetShapeSymbol(MlirType type, intptr_t index) {
  auto tensorType = llvm::cast<wave::WaveTensorType>(unwrap(type));
  return wrap(tensorType.getShape()[index]);
}

MlirType mlirWaveTensorTypeGetElementType(MlirType type) {
  return wrap(llvm::cast<wave::WaveTensorType>(unwrap(type)).getElementType());
}

MlirAttribute mlirWaveTensorTypeGetAddressSpace(MlirType type) {
  return wrap(llvm::cast<wave::WaveTensorType>(unwrap(type)).getAddressSpace());
}

MlirType mlirWaveTensorTypeGet(MlirContext mlirCtx, MlirAttribute *shapeSymbols,
                               intptr_t numShapeSymbols, bool fullySpecified,
                               MlirType elementType,
                               MlirAttribute addressSpace) {
  MLIRContext *ctx = unwrap(mlirCtx);
  assert((numShapeSymbols == 0 || shapeSymbols) &&
         "expected non-null shapeSymbols when numShapeSymbols > 0");
  llvm::SmallVector<Attribute> shapeAttrs;
  shapeAttrs.reserve(numShapeSymbols);
  (void)unwrapList(numShapeSymbols, shapeSymbols, shapeAttrs);
  assert(llvm::all_of(shapeAttrs, llvm::IsaPred<wave::WaveSymbolAttr>) &&
         "expected shapeSymbols to contain only WaveSymbolAttr values");
  assert(llvm::isa<wave::WaveAddressSpaceAttr>(unwrap(addressSpace)) &&
         "expected addressSpace to be a WaveAddressSpaceAttr");
  SmallVector<wave::WaveSymbolAttr> shape =
      llvm::map_to_vector(shapeAttrs, llvm::CastTo<wave::WaveSymbolAttr>);
  auto addrAttr = llvm::cast<wave::WaveAddressSpaceAttr>(unwrap(addressSpace));
  return wrap(wave::WaveTensorType::get(ctx, shape, fullySpecified,
                                        unwrap(elementType), addrAttr));
}

//===---------------------------------------------------------------------===//
// WaveSymbolAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveSymbolAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveSymbolAttr>(unwrap(attr));
}

MlirAttribute mlirWaveSymbolAttrGet(MlirContext mlirCtx,
                                    MlirStringRef symbolNameStrRef) {
  MLIRContext *ctx = unwrap(mlirCtx);
  llvm::StringRef symbolName = unwrap(symbolNameStrRef);
  return wrap(wave::WaveSymbolAttr::get(ctx, symbolName));
}

MlirTypeID mlirWaveSymbolAttrGetTypeID() {
  return wrap(TypeID::get<wave::WaveSymbolAttr>());
}

MlirStringRef mlirWaveSymbolAttrGetName(MlirAttribute attr) {
  return wrap(llvm::cast<wave::WaveSymbolAttr>(unwrap(attr)).getName());
}

//===---------------------------------------------------------------------===//
// WaveIterSymbolAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveIterSymbolAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveIterSymbolAttr>(unwrap(attr));
}

MlirAttribute mlirWaveIterSymbolAttrGet(MlirContext mlirCtx,
                                        MlirStringRef symbolNameStrRef) {
  MLIRContext *ctx = unwrap(mlirCtx);
  llvm::StringRef symbolName = unwrap(symbolNameStrRef);
  return wrap(wave::WaveIterSymbolAttr::get(ctx, symbolName));
}

MlirTypeID mlirWaveIterSymbolAttrGetTypeID() {
  return wrap(TypeID::get<wave::WaveIterSymbolAttr>());
}

MlirStringRef mlirWaveIterSymbolAttrGetName(MlirAttribute attr) {
  return wrap(llvm::cast<wave::WaveIterSymbolAttr>(unwrap(attr)).getName());
}

//===---------------------------------------------------------------------===//
// WaveOperandAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveOperandAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveOperandAttr>(unwrap(attr));
}

MlirAttribute mlirWaveOperandAttrGet(MlirContext mlirCtx,
                                     unsigned operandNumber) {
  MLIRContext *ctx = unwrap(mlirCtx);
  return wrap(wave::WaveOperandAttr::get(ctx, operandNumber));
}

MlirTypeID mlirWaveOperandAttrGetTypeID() {
  return wrap(TypeID::get<wave::WaveOperandAttr>());
}

unsigned mlirWaveOperandAttrGetOperandNumber(MlirAttribute attr) {
  return llvm::cast<wave::WaveOperandAttr>(unwrap(attr)).getOperandNumber();
}

//===---------------------------------------------------------------------===//
// WaveIndexSymbolAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveIndexSymbolAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveIndexSymbolAttr>(unwrap(attr));
}

MlirAttribute mlirWaveIndexSymbolAttrGet(MlirContext mlirCtx, uint32_t value) {
  return wrap(wave::WaveIndexSymbolAttr::get(
      unwrap(mlirCtx), static_cast<wave::WaveIndexSymbol>(value)));
}

uint32_t mlirWaveIndexSymbolAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<wave::WaveIndexSymbolAttr>(unwrap(attr)).getValue());
}

MlirTypeID mlirWaveIndexSymbolAttrGetTypeID() {
  return wrap(TypeID::get<wave::WaveIndexSymbolAttr>());
}

//===---------------------------------------------------------------------===//
// WaveIndexMappingAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveIndexMappingAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveIndexMappingAttr>(unwrap(attr));
}

MlirAttribute mlirWaveIndexMappingAttrGet(MlirContext mlirCtx,
                                          MlirAttribute *symbolNames,
                                          MlirAffineMap start,
                                          MlirAffineMap step,
                                          MlirAffineMap stride) {
  MLIRContext *ctx = unwrap(mlirCtx);

  // Convert C array of MlirAttribute to vector of WaveSymbolAttr.
  unsigned numSymbols = mlirAffineMapGetNumSymbols(start);
  assert(mlirAffineMapGetNumSymbols(step) == numSymbols &&
         "expected start and step to have the same number of dimensions");
  assert(mlirAffineMapGetNumSymbols(stride) == numSymbols &&
         "expected start and stride to have the same number of dimensions");
  llvm::SmallVector<Attribute> symbolAttrs = llvm::map_to_vector(
      llvm::make_range(symbolNames, symbolNames + numSymbols),
      [](MlirAttribute attr) { return unwrap(attr); });

  assert(llvm::all_of(
             symbolAttrs,
             llvm::IsaPred<wave::WaveSymbolAttr, wave::WaveIndexSymbolAttr,
                           wave::WaveIterSymbolAttr>) &&
         "expected mapping to contain only WaveSymbolAttr or "
         "WaveIndexSymbolAttr attributes");

  return wrap(wave::WaveIndexMappingAttr::get(ctx, symbolAttrs, unwrap(start),
                                              unwrap(step), unwrap(stride)));
}

MlirTypeID mlirWaveIndexMappingAttrGetTypeID() {
  return wrap(TypeID::get<wave::WaveIndexMappingAttr>());
}

MlirAffineMap mlirWaveIndexMappingAttrGetStart(MlirAttribute attr) {
  return wrap(llvm::cast<wave::WaveIndexMappingAttr>(unwrap(attr)).getStart());
}

MlirAffineMap mlirWaveIndexMappingAttrGetStep(MlirAttribute attr) {
  return wrap(llvm::cast<wave::WaveIndexMappingAttr>(unwrap(attr)).getStep());
}

MlirAffineMap mlirWaveIndexMappingAttrGetStride(MlirAttribute attr) {
  return wrap(llvm::cast<wave::WaveIndexMappingAttr>(unwrap(attr)).getStride());
}

intptr_t mlirWaveIndexMappingAttrGetNumSymbols(MlirAttribute attr) {
  return llvm::cast<wave::WaveIndexMappingAttr>(unwrap(attr))
      .getSymbols()
      .size();
}

MlirAttribute mlirWaveIndexMappingAttrGetSymbol(MlirAttribute attr,
                                                intptr_t index) {
  return wrap(
      llvm::cast<wave::WaveIndexMappingAttr>(unwrap(attr)).getSymbols()[index]);
}

//===---------------------------------------------------------------------===//
// WaveHyperparameterAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveHyperparameterAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveHyperparameterAttr>(unwrap(attr));
}

MlirAttribute mlirWaveHyperparameterAttrGet(MlirAttribute mapping) {
  auto dictAttr = llvm::cast<DictionaryAttr>(unwrap(mapping));

  MLIRContext *ctx = dictAttr.getContext();

  assert(llvm::all_of(dictAttr,
                      [](const NamedAttribute &namedAttr) {
                        return llvm::isa<IntegerAttr>(namedAttr.getValue());
                      }) &&
         "expected mapping to contain only integer values");

  return wrap(wave::WaveHyperparameterAttr::get(ctx, dictAttr));
}

MlirTypeID mlirWaveHyperparameterAttrGetTypeID() {
  return wrap(TypeID::get<wave::WaveHyperparameterAttr>());
}

MlirAttribute mlirWaveHyperparameterAttrGetMapping(MlirAttribute attr) {
  auto mapping =
      llvm::cast<wave::WaveHyperparameterAttr>(unwrap(attr)).getMapping();
  return wrap(mapping);
}

//===---------------------------------------------------------------------===//
// WaveWorkgroupDimAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveWorkgroupDimAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveWorkgroupDimAttr>(unwrap(attr));
}

MlirAttribute mlirWaveWorkgroupDimAttrGet(MlirContext mlirCtx, uint32_t value) {
  return wrap(wave::WaveWorkgroupDimAttr::get(
      unwrap(mlirCtx), static_cast<wave::WaveWorkgroupDim>(value)));
}

uint32_t mlirWaveWorkgroupDimAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<wave::WaveWorkgroupDimAttr>(unwrap(attr)).getValue());
}

MlirTypeID mlirWaveWorkgroupDimAttrGetTypeID() {
  return wrap(TypeID::get<wave::WaveWorkgroupDimAttr>());
}

//===---------------------------------------------------------------------===//
// WaveReductionScopeAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveReductionScopeAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveReductionScopeAttr>(unwrap(attr));
}

MlirAttribute mlirWaveReductionScopeAttrGet(MlirContext mlirCtx,
                                            uint32_t value) {
  return wrap(wave::WaveReductionScopeAttr::get(
      unwrap(mlirCtx), static_cast<wave::WaveReductionScope>(value)));
}

uint32_t mlirWaveReductionScopeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<wave::WaveReductionScopeAttr>(unwrap(attr)).getValue());
}

MlirTypeID mlirWaveReductionScopeAttrGetTypeID() {
  return wrap(TypeID::get<wave::WaveReductionScopeAttr>());
}

//===---------------------------------------------------------------------===//
// WaveAddressSpaceAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveAddressSpaceAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveAddressSpaceAttr>(unwrap(attr));
}

MlirAttribute mlirWaveAddressSpaceAttrGet(MlirContext mlirCtx, uint32_t value) {
  return wrap(wave::WaveAddressSpaceAttr::get(
      unwrap(mlirCtx), static_cast<wave::WaveAddressSpace>(value)));
}

uint32_t mlirWaveAddressSpaceAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<wave::WaveAddressSpaceAttr>(unwrap(attr)).getValue());
}

MlirTypeID mlirWaveAddressSpaceAttrGetTypeID() {
  return wrap(TypeID::get<wave::WaveAddressSpaceAttr>());
}

//===---------------------------------------------------------------------===//
// WaveShuffleModeAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveShuffleModeAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveShuffleModeAttr>(unwrap(attr));
}

MlirAttribute mlirWaveShuffleModeAttrGet(MlirContext mlirCtx, uint32_t value) {
  return wrap(wave::WaveShuffleModeAttr::get(
      unwrap(mlirCtx), static_cast<wave::WaveShuffleMode>(value)));
}

uint32_t mlirWaveShuffleModeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<wave::WaveShuffleModeAttr>(unwrap(attr)).getValue());
}

MlirTypeID mlirWaveShuffleModeAttrGetTypeID() {
  return wrap(TypeID::get<wave::WaveShuffleModeAttr>());
}

//===---------------------------------------------------------------------===//
// WaveApplyExprCombinatorAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveApplyExprCombinatorAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveApplyExprCombinatorAttr>(unwrap(attr));
}

MlirAttribute mlirWaveApplyExprCombinatorAttrGet(MlirContext mlirCtx,
                                                 uint32_t value) {
  return wrap(wave::WaveApplyExprCombinatorAttr::get(
      unwrap(mlirCtx), static_cast<wave::WaveApplyExprCombinator>(value)));
}

uint32_t mlirWaveApplyExprCombinatorAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<wave::WaveApplyExprCombinatorAttr>(unwrap(attr)).getValue());
}

MlirTypeID mlirWaveApplyExprCombinatorAttrGetTypeID() {
  return wrap(TypeID::get<wave::WaveApplyExprCombinatorAttr>());
}

//===---------------------------------------------------------------------===//
// WaveMmaKindAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveMmaKindAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveMmaKindAttr>(unwrap(attr));
}

MlirAttribute mlirWaveMmaKindAttrGet(MlirContext mlirCtx, uint32_t value) {
  return wrap(wave::WaveMmaKindAttr::get(
      unwrap(mlirCtx), static_cast<wave::WaveMmaKind>(value)));
}

uint32_t mlirWaveMmaKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<wave::WaveMmaKindAttr>(unwrap(attr)).getValue());
}

MlirTypeID mlirWaveMmaKindAttrGetTypeID() {
  return wrap(TypeID::get<wave::WaveMmaKindAttr>());
}

//===---------------------------------------------------------------------===//
// WaveExprListAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveExprListAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveExprListAttr>(unwrap(attr));
}

MlirAttribute mlirWaveExprListAttrGet(MlirAttribute *symbolNames,
                                      MlirAffineMap map) {
  MLIRContext *ctx = unwrap(map).getContext();

  unsigned numSymbols = mlirAffineMapGetNumSymbols(map);
  llvm::SmallVector<Attribute> symbolAttrs = llvm::map_to_vector(
      llvm::make_range(symbolNames, symbolNames + numSymbols),
      [](MlirAttribute attr) { return unwrap(attr); });

  assert(
      llvm::all_of(
          symbolAttrs,
          llvm::IsaPred<wave::WaveSymbolAttr, wave::WaveIndexSymbolAttr,
                        wave::WaveIterSymbolAttr, wave::WaveOperandAttr>) &&
      "expected mapping to contain only WaveSymbolAttr, "
      "WaveIndexSymbolAttr, WaveIterSymbolAttr or WaveOperandAttr attributes");

  return wrap(wave::WaveExprListAttr::get(ctx, symbolAttrs, unwrap(map)));
}

MlirTypeID mlirWaveExprListAttrGetTypeID() {
  return wrap(TypeID::get<wave::WaveExprListAttr>());
}

MlirAffineMap mlirWaveExprListAttrGetMap(MlirAttribute attr) {
  return wrap(llvm::cast<wave::WaveExprListAttr>(unwrap(attr)).getMap());
}

intptr_t mlirWaveExprListAttrGetNumSymbols(MlirAttribute attr) {
  return llvm::cast<wave::WaveExprListAttr>(unwrap(attr)).getSymbols().size();
}

MlirAttribute mlirWaveExprListAttrGetSymbol(MlirAttribute attr,
                                            intptr_t index) {
  return wrap(
      llvm::cast<wave::WaveExprListAttr>(unwrap(attr)).getSymbols()[index]);
}
//===---------------------------------------------------------------------===//
// WaveSymbolMappingAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveSymbolMappingAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveSymbolMappingAttr>(unwrap(attr));
}

MlirAttribute mlirWaveSymbolMappingAttrGet(MlirContext ctx, intptr_t numEntries,
                                           MlirAttribute *keys,
                                           MlirAttribute *values) {
  SmallVector<wave::WaveSymbolAttr> keyAttrs;
  SmallVector<Attribute> valueAttrs;
  keyAttrs.reserve(numEntries);
  valueAttrs.reserve(numEntries);
  for (intptr_t i = 0; i < numEntries; ++i) {
    keyAttrs.push_back(llvm::cast<wave::WaveSymbolAttr>(unwrap(keys[i])));
    valueAttrs.push_back(unwrap(values[i]));
  }
  return wrap(
      wave::WaveSymbolMappingAttr::get(unwrap(ctx), keyAttrs, valueAttrs));
}

intptr_t mlirWaveSymbolMappingAttrGetNumEntries(MlirAttribute attr) {
  return llvm::cast<wave::WaveSymbolMappingAttr>(unwrap(attr)).getNumEntries();
}

MlirAttribute mlirWaveSymbolMappingAttrGetKey(MlirAttribute attr,
                                              intptr_t index) {
  return wrap(
      llvm::cast<wave::WaveSymbolMappingAttr>(unwrap(attr)).getKeys()[index]);
}

MlirAttribute mlirWaveSymbolMappingAttrGetValue(MlirAttribute attr,
                                                intptr_t index) {
  return wrap(
      llvm::cast<wave::WaveSymbolMappingAttr>(unwrap(attr)).getValues()[index]);
}

MlirAttribute mlirWaveSymbolMappingAttrLookup(MlirAttribute attr,
                                              MlirAttribute key) {
  auto keyAttr = llvm::dyn_cast<wave::WaveSymbolAttr>(unwrap(key));
  if (!keyAttr)
    return MlirAttribute();
  return wrap(
      llvm::cast<wave::WaveSymbolMappingAttr>(unwrap(attr)).lookup(keyAttr));
}

MlirTypeID mlirWaveSymbolMappingAttrGetTypeID() {
  return wrap(TypeID::get<wave::WaveSymbolMappingAttr>());
}

//===---------------------------------------------------------------------===//
// HardwareConstraintAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAHardwareConstraintAttr(MlirAttribute attr) {
  return llvm::isa<wave::HardwareConstraintAttr>(unwrap(attr));
}

MlirAttribute
mlirHardwareConstraintAttrGet(MlirContext mlirCtx, unsigned threadsPerWave,
                              size_t wavesPerBlockSize, unsigned *wavesPerBlock,
                              MlirAttribute mmaType, MlirAttribute vectorShapes,
                              unsigned maxBitsPerLoad) {
  MLIRContext *ctx = unwrap(mlirCtx);
  auto mmaTypeAttr =
      llvm::cast_if_present<wave::WaveMmaKindAttr>(unwrap(mmaType));
  auto vectorShapesAttr =
      llvm::cast_if_present<DictionaryAttr>(unwrap(vectorShapes));

  return wrap(wave::HardwareConstraintAttr::get(
      ctx, threadsPerWave, llvm::ArrayRef(wavesPerBlock, wavesPerBlockSize),
      mmaTypeAttr, vectorShapesAttr, maxBitsPerLoad));
}

MlirTypeID mlirWHardwareConstraintAttrGetTypeID() {
  return wrap(TypeID::get<wave::HardwareConstraintAttr>());
}

unsigned mlirHardwareConstraintAttrGetThreadsPerWave(MlirAttribute attr) {
  return llvm::cast<wave::HardwareConstraintAttr>(unwrap(attr))
      .getThreadsPerWave();
}
intptr_t mlirHardwareConstraintAttrGetNumWavesPerBlock(MlirAttribute attr) {
  return llvm::cast<wave::HardwareConstraintAttr>(unwrap(attr))
      .getWavesPerBlock()
      .size();
}
unsigned mlirHardwareConstraintAttrGetWavesPerBlockElem(MlirAttribute attr,
                                                        intptr_t i) {
  return llvm::cast<wave::HardwareConstraintAttr>(unwrap(attr))
      .getWavesPerBlock()[i];
}
MlirAttribute mlirHardwareConstraintAttrGetMmaType(MlirAttribute attr) {
  return wrap(
      llvm::cast<wave::HardwareConstraintAttr>(unwrap(attr)).getMmaType());
}
MlirAttribute mlirHardwareConstraintAttrGetVectorShapes(MlirAttribute attr) {
  return wrap(llvm::dyn_cast<wave::HardwareConstraintAttr>(unwrap(attr))
                  .getVectorShapes());
}
unsigned mlirHardwareConstraintAttrGetMaxBitsPerLoad(MlirAttribute attr) {
  return llvm::cast<wave::HardwareConstraintAttr>(unwrap(attr))
      .getMaxBitsPerLoad();
}

//===---------------------------------------------------------------------===//
// DeviceConstraintAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsADeviceConstraintAttr(MlirAttribute attr) {
  return llvm::isa<wave::DeviceConstraintAttr>(unwrap(attr));
}

MlirAttribute mlirDeviceConstraintAttrGet(MlirContext mlirCtx,
                                          MlirAttribute dim,
                                          MlirAttribute tileSize,
                                          unsigned deviceDim) {
  MLIRContext *ctx = unwrap(mlirCtx);
  auto dimAttr = llvm::cast<wave::WaveSymbolAttr>(unwrap(dim));
  auto tileSizeAttr = llvm::cast<wave::WaveExprListAttr>(unwrap(tileSize));

  return wrap(
      wave::DeviceConstraintAttr::get(ctx, dimAttr, tileSizeAttr, deviceDim));
}

MlirTypeID mlirDeviceConstraintAttrGetTypeID() {
  return wrap(TypeID::get<wave::DeviceConstraintAttr>());
}

MlirAttribute mlirDeviceConstraintAttrGetDim(MlirAttribute attr) {
  return wrap(llvm::cast<wave::DeviceConstraintAttr>(unwrap(attr)).getDim());
}

MlirAttribute mlirDeviceConstraintAttrGetTileSize(MlirAttribute attr) {
  return wrap(
      llvm::cast<wave::DeviceConstraintAttr>(unwrap(attr)).getTileSize());
}

unsigned mlirDeviceConstraintAttrGetDeviceDim(MlirAttribute attr) {
  return llvm::cast<wave::DeviceConstraintAttr>(unwrap(attr)).getDeviceDim();
}

//===---------------------------------------------------------------------===//
// WorkgroupConstraintAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWorkgroupConstraintAttr(MlirAttribute attr) {
  return llvm::isa<wave::WorkgroupConstraintAttr>(unwrap(attr));
}

MlirAttribute mlirWorkgroupConstraintAttrGet(MlirContext mlirCtx,
                                             MlirAttribute dim,
                                             MlirAttribute tileSize,
                                             MlirAttribute workgroupDim,
                                             bool primary) {
  MLIRContext *ctx = unwrap(mlirCtx);
  auto dimAttr = llvm::cast<wave::WaveSymbolAttr>(unwrap(dim));
  auto tileSizeAttr = llvm::cast<wave::WaveExprListAttr>(unwrap(tileSize));
  auto workgroupDimAttr =
      llvm::cast<wave::WaveWorkgroupDimAttr>(unwrap(workgroupDim));

  return wrap(wave::WorkgroupConstraintAttr::get(ctx, dimAttr, tileSizeAttr,
                                                 workgroupDimAttr, primary));
}

MlirTypeID mlirWorkgroupConstraintAttrGetTypeID() {
  return wrap(TypeID::get<wave::WorkgroupConstraintAttr>());
}

MlirAttribute mlirWorkgroupConstraintAttrGetDim(MlirAttribute attr) {
  return wrap(llvm::cast<wave::WorkgroupConstraintAttr>(unwrap(attr)).getDim());
}

MlirAttribute mlirWorkgroupConstraintAttrGetTileSize(MlirAttribute attr) {
  return wrap(
      llvm::cast<wave::WorkgroupConstraintAttr>(unwrap(attr)).getTileSize());
}

MlirAttribute mlirWorkgroupConstraintAttrGetWorkgroupDim(MlirAttribute attr) {
  return wrap(llvm::cast<wave::WorkgroupConstraintAttr>(unwrap(attr))
                  .getWorkgroupDim());
}

bool mlirWorkgroupConstraintAttrGetPrimary(MlirAttribute attr) {
  return llvm::cast<wave::WorkgroupConstraintAttr>(unwrap(attr)).getPrimary();
}

//===---------------------------------------------------------------------===//
// WaveConstraintAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveConstraintAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveConstraintAttr>(unwrap(attr));
}

MlirAttribute mlirWaveConstraintAttrGet(MlirContext mlirCtx, MlirAttribute dim,
                                        MlirAttribute tileSize) {
  MLIRContext *ctx = unwrap(mlirCtx);
  auto dimAttr = llvm::cast<wave::WaveSymbolAttr>(unwrap(dim));
  auto tileSizeAttr = llvm::cast<wave::WaveExprListAttr>(unwrap(tileSize));
  return wrap(wave::WaveConstraintAttr::get(ctx, dimAttr, tileSizeAttr));
}

MlirTypeID mlirWaveConstraintAttrGetTypeID() {
  return wrap(TypeID::get<wave::WaveConstraintAttr>());
}

MlirAttribute mlirWaveConstraintAttrGetDim(MlirAttribute attr) {
  return wrap(llvm::cast<wave::WaveConstraintAttr>(unwrap(attr)).getDim());
}

MlirAttribute mlirWaveConstraintAttrGetTileSize(MlirAttribute attr) {
  return wrap(llvm::cast<wave::WaveConstraintAttr>(unwrap(attr)).getTileSize());
}

//===---------------------------------------------------------------------===//
// TilingConstraintAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsATilingConstraintAttr(MlirAttribute attr) {
  return llvm::isa<wave::TilingConstraintAttr>(unwrap(attr));
}

MlirAttribute mlirTilingConstraintAttrGet(MlirContext mlirCtx,
                                          MlirAttribute dim,
                                          MlirAttribute tileSize) {
  MLIRContext *ctx = unwrap(mlirCtx);
  auto dimAttr = llvm::cast<wave::WaveSymbolAttr>(unwrap(dim));
  auto tileSizeAttr = llvm::cast<wave::WaveExprListAttr>(unwrap(tileSize));

  return wrap(wave::TilingConstraintAttr::get(ctx, dimAttr, tileSizeAttr));
}

MlirTypeID mlirTilingConstraintAttrGetTypeID() {
  return wrap(TypeID::get<wave::TilingConstraintAttr>());
}

MlirAttribute mlirTilingConstraintAttrGetDim(MlirAttribute attr) {
  return wrap(llvm::cast<wave::TilingConstraintAttr>(unwrap(attr)).getDim());
}

MlirAttribute mlirTilingConstraintAttrGetTileSize(MlirAttribute attr) {
  return wrap(
      llvm::cast<wave::TilingConstraintAttr>(unwrap(attr)).getTileSize());
}

//===---------------------------------------------------------------------===//
// WaveNormalFormAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveNormalFormAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveNormalFormAttr>(unwrap(attr));
}

MlirAttribute mlirWaveNormalFormAttrGet(MlirContext mlirCtx, uint32_t value) {
  return wrap(wave::WaveNormalFormAttr::get(
      unwrap(mlirCtx), static_cast<wave::WaveNormalForm>(value)));
}

uint32_t mlirWaveNormalFormAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<wave::WaveNormalFormAttr>(unwrap(attr)).getValue());
}

MlirTypeID mlirWaveNormalFormAttrGetTypeID() {
  return wrap(TypeID::get<wave::WaveNormalFormAttr>());
}

//===---------------------------------------------------------------------===//
// Wave Operations
//===---------------------------------------------------------------------===//

void mlirWaveIterateOpMakeIsolated(MlirOperation op) {
  Operation *operation = unwrap(op);
  if (auto iterateOp = dyn_cast<wave::IterateOp>(operation)) {
    IRRewriter rewriter(operation->getContext());
    iterateOp.makeIsolated(rewriter);
  }
}

void mlirWaveIterateOpMakeNonIsolated(MlirOperation op) {
  Operation *operation = unwrap(op);
  if (auto iterateOp = dyn_cast<wave::IterateOp>(operation)) {
    IRRewriter rewriter(operation->getContext());
    iterateOp.makeNonIsolated(rewriter);
  }
}
