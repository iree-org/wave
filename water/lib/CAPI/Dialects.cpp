// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir-c/AffineMap.h"
#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Support/TypeID.h"

#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/c/Dialects.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Wave, wave, ::wave::WaveDialect)

//===---------------------------------------------------------------------===//
// WaveSymbolAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveSymbolAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveSymbolAttr>(unwrap(attr));
}

MlirAttribute mlirWaveSymbolAttrGet(MlirContext mlirCtx,
                                    MlirStringRef symbolNameStrRef) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  llvm::StringRef symbolName = unwrap(symbolNameStrRef);
  return wrap(wave::WaveSymbolAttr::get(ctx, symbolName));
}

MlirTypeID mlirWaveSymbolAttrGetTypeID() {
  return wrap(mlir::TypeID::get<wave::WaveSymbolAttr>());
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
  mlir::MLIRContext *ctx = unwrap(mlirCtx);

  // Convert C array of MlirAttribute to vector of WaveSymbolAttr.
  unsigned numSymbols = mlirAffineMapGetNumSymbols(start);
  assert(mlirAffineMapGetNumSymbols(step) == numSymbols &&
         "expected start and step to have the same number of dimensions");
  assert(mlirAffineMapGetNumSymbols(stride) == numSymbols &&
         "expected start and stride to have the same number of dimensions");
  llvm::SmallVector<wave::WaveSymbolAttr> symbolAttrs = llvm::map_to_vector(
      llvm::make_range(symbolNames, symbolNames + numSymbols),
      [](MlirAttribute attr) {
        return llvm::cast<wave::WaveSymbolAttr>(unwrap(attr));
      });

  return wrap(wave::WaveIndexMappingAttr::get(ctx, symbolAttrs, unwrap(start),
                                              unwrap(step), unwrap(stride)));
}

MlirTypeID mlirWaveIndexMappingAttrGetTypeID() {
  return wrap(mlir::TypeID::get<wave::WaveIndexMappingAttr>());
}

//===---------------------------------------------------------------------===//
// WaveHyperparameterAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveHyperparameterAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveHyperparameterAttr>(unwrap(attr));
}

MlirAttribute mlirWaveHyperparameterAttrGet(MlirAttribute mapping) {
  auto dictAttr = llvm::cast<mlir::DictionaryAttr>(unwrap(mapping));

  mlir::MLIRContext *ctx = dictAttr.getContext();

  assert(llvm::all_of(dictAttr,
                      [](const mlir::NamedAttribute &namedAttr) {
                        return llvm::isa<mlir::IntegerAttr>(
                            namedAttr.getValue());
                      }) &&
         "expected mapping to contain only integer values");

  return wrap(wave::WaveHyperparameterAttr::get(ctx, dictAttr));
}

MlirTypeID mlirWaveHyperparameterAttrGetTypeID() {
  return wrap(mlir::TypeID::get<wave::WaveHyperparameterAttr>());
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
  return wrap(mlir::TypeID::get<wave::WaveWorkgroupDimAttr>());
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
  return wrap(mlir::TypeID::get<wave::WaveAddressSpaceAttr>());
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
  return wrap(mlir::TypeID::get<wave::WaveMmaKindAttr>());
}

//===---------------------------------------------------------------------===//
// WaveExprListAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveExprListAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveExprListAttr>(unwrap(attr));
}

MlirAttribute mlirWaveExprListAttrGet(MlirAttribute *symbolNames,
                                      MlirAffineMap map) {
  mlir::MLIRContext *ctx = unwrap(map).getContext();

  unsigned numSymbols = mlirAffineMapGetNumSymbols(map);
  llvm::SmallVector<wave::WaveSymbolAttr> symbolAttrs = llvm::map_to_vector(
      llvm::make_range(symbolNames, symbolNames + numSymbols),
      [](MlirAttribute attr) {
        return llvm::cast<wave::WaveSymbolAttr>(unwrap(attr));
      });

  return wrap(wave::WaveExprListAttr::get(ctx, symbolAttrs, unwrap(map)));
}

MlirTypeID mlirWaveExprListAttrGetTypeID() {
  return wrap(mlir::TypeID::get<wave::WaveExprListAttr>());
}

//===---------------------------------------------------------------------===//
// WaveReadWriteBoundsAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveReadWriteBoundsAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveReadWriteBoundsAttr>(unwrap(attr));
}

MlirAttribute mlirWaveReadWriteBoundsAttrGet(MlirAttribute mapping) {
  auto dictAttr = llvm::cast<mlir::DictionaryAttr>(unwrap(mapping));

  mlir::MLIRContext *ctx = dictAttr.getContext();

  assert(llvm::all_of(dictAttr,
                      [](const mlir::NamedAttribute &namedAttr) {
                        return llvm::isa<wave::WaveExprListAttr>(
                            namedAttr.getValue());
                      }) &&
         "expected mapping to contain only WaveExprListAttr values");

  return wrap(wave::WaveReadWriteBoundsAttr::get(ctx, dictAttr));
}

MlirTypeID mlirWaveReadWriteBoundsAttrGetTypeID() {
  return wrap(mlir::TypeID::get<wave::WaveReadWriteBoundsAttr>());
}

//===---------------------------------------------------------------------===//
// HardwareConstraintAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAHardwareConstraintAttr(MlirAttribute attr) {
  return llvm::isa<wave::HardwareConstraintAttr>(unwrap(attr));
}

MlirAttribute
mlirHardwareConstraintAttrGet(MlirContext mlirCtx, unsigned threadsPerWave,
                              unsigned *wavesPerBlock, size_t wavesPerBlockSize,
                              MlirAttribute mmaType, MlirAttribute vectorShapes,
                              unsigned maxBitsPerLoad) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  auto mmaTypeAttr =
      llvm::cast_if_present<wave::WaveMmaKindAttr>(unwrap(mmaType));
  auto vectorShapesAttr =
      llvm::cast_if_present<mlir::DictionaryAttr>(unwrap(vectorShapes));

  return wrap(wave::HardwareConstraintAttr::get(
      ctx, threadsPerWave, llvm::ArrayRef(wavesPerBlock, wavesPerBlockSize),
      mmaTypeAttr, vectorShapesAttr, maxBitsPerLoad));
}

MlirTypeID mlirWHardwareConstraintAttrGetTypeID() {
  return wrap(mlir::TypeID::get<wave::HardwareConstraintAttr>());
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
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  auto dimAttr = llvm::cast<wave::WaveSymbolAttr>(unwrap(dim));
  auto tileSizeAttr = llvm::cast<wave::WaveExprListAttr>(unwrap(tileSize));

  return wrap(
      wave::DeviceConstraintAttr::get(ctx, dimAttr, tileSizeAttr, deviceDim));
}

MlirTypeID mlirDeviceConstraintAttrGetTypeID() {
  return wrap(mlir::TypeID::get<wave::DeviceConstraintAttr>());
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
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  auto dimAttr = llvm::cast<wave::WaveSymbolAttr>(unwrap(dim));
  auto tileSizeAttr = llvm::cast<wave::WaveExprListAttr>(unwrap(tileSize));
  auto workgroupDimAttr =
      llvm::cast<wave::WaveWorkgroupDimAttr>(unwrap(workgroupDim));

  return wrap(wave::WorkgroupConstraintAttr::get(ctx, dimAttr, tileSizeAttr,
                                                 workgroupDimAttr, primary));
}

MlirTypeID mlirWorkgroupConstraintAttrGetTypeID() {
  return wrap(mlir::TypeID::get<wave::WorkgroupConstraintAttr>());
}

//===---------------------------------------------------------------------===//
// WaveConstraintAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveConstraintAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveConstraintAttr>(unwrap(attr));
}

MlirAttribute mlirWaveConstraintAttrGet(MlirContext mlirCtx, MlirAttribute dim,
                                        MlirAttribute tileSize,
                                        MlirAttribute wgConstraint) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  auto dimAttr = llvm::cast<wave::WaveSymbolAttr>(unwrap(dim));
  auto tileSizeAttr = llvm::cast<wave::WaveExprListAttr>(unwrap(tileSize));
  auto wgConstraintAttr = llvm::cast_if_present<wave::WorkgroupConstraintAttr>(
      unwrap(wgConstraint));

  return wrap(wave::WaveConstraintAttr::get(ctx, dimAttr, tileSizeAttr,
                                            wgConstraintAttr));
}

MlirTypeID mlirWaveConstraintAttrGetTypeID() {
  return wrap(mlir::TypeID::get<wave::WaveConstraintAttr>());
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
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  auto dimAttr = llvm::cast<wave::WaveSymbolAttr>(unwrap(dim));
  auto tileSizeAttr = llvm::cast<wave::WaveExprListAttr>(unwrap(tileSize));

  return wrap(wave::TilingConstraintAttr::get(ctx, dimAttr, tileSizeAttr));
}

MlirTypeID mlirTilingConstraintAttrGetTypeID() {
  return wrap(mlir::TypeID::get<wave::TilingConstraintAttr>());
}
