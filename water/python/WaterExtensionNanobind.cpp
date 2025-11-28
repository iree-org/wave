// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir-c/AffineMap.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "water/c/Dialects.h"

#include "nanobind/nanobind.h"

#include "mlir/CAPI/Support.h"

namespace nb = nanobind;

NB_MODULE(_waterDialects, m) {
  auto d = m.def_submodule("wave");
  d.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle h = mlirGetDialectHandle__wave__();
        mlirDialectHandleRegisterDialect(h, context);
        if (load)
          mlirDialectHandleLoadDialect(h, context);
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);

  // Export dialect constants
  d.attr("WAVE_CONSTRAINTS_ATTR_NAME") = mlirWaveDialectConstraintsAttrName;

  //===---------------------------------------------------------------------===//
  // WaveSymbolAttr
  //===---------------------------------------------------------------------===//

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      d, "WaveSymbolAttr", mlirAttributeIsAWaveSymbolAttr,
      mlirWaveSymbolAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, const std::string &symbolName,
             // MlirContext should always come last to allow for being
             // automatically deduced from context.
             MlirContext context) {
            MlirStringRef symbolNameStrRef =
                mlirStringRefCreate(symbolName.data(), symbolName.size());
            return cls(mlirWaveSymbolAttrGet(context, symbolNameStrRef));
          },
          nb::arg("cls"), nb::arg("symbolName"),
          nb::arg("context") = nb::none(),
          "Gets a wave.WaveSymbolAttr from parameters.");

  //===---------------------------------------------------------------------===//
  // WaveIterSymbolAttr
  //===---------------------------------------------------------------------===//

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      d, "WaveIterSymbolAttr", mlirAttributeIsAWaveIterSymbolAttr,
      mlirWaveIterSymbolAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, const std::string &symbolName,
             MlirContext context) {
            MlirStringRef symbolNameStrRef =
                mlirStringRefCreate(symbolName.data(), symbolName.size());
            return cls(mlirWaveIterSymbolAttrGet(context, symbolNameStrRef));
          },
          nb::arg("cls"), nb::arg("symbolName"),
          nb::arg("context") = nb::none(),
          "Gets a wave.WaveIterSymbolAttr from parameters.")
      .def_property_readonly("name", [](MlirAttribute self) {
        return mlirWaveIterSymbolAttrGetName(self);
      });

  //===---------------------------------------------------------------------===//
  // WaveIndexSymbolAttr
  //===---------------------------------------------------------------------===//

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      d, "WaveIndexSymbolAttr", mlirAttributeIsAWaveIndexSymbolAttr,
      mlirWaveIndexSymbolAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, WaveIndexSymbol value,
             MlirContext context) {
            return cls(mlirWaveIndexSymbolAttrGet(
                context, static_cast<uint32_t>(value)));
          },
          nb::arg("cls"), nb::arg("value"), nb::arg("context") = nb::none(),
          "Gets a wave.WaveIndexSymbolAttr from an index symbol enum value.")
      .def_property_readonly(
          "value",
          [](MlirAttribute self) {
            return static_cast<WaveIndexSymbol>(
                mlirWaveIndexSymbolAttrGetValue(self));
          },
          "Index symbol enum value.");

  nb::enum_<WaveIndexSymbol>(d, "WaveIndexSymbol")
      .value("DEVICE_DIM_0", WaveIndexSymbol_DEVICE_DIM_0)
      .value("DEVICE_DIM_1", WaveIndexSymbol_DEVICE_DIM_1)
      .value("DEVICE_DIM_2", WaveIndexSymbol_DEVICE_DIM_2)
      .value("WORKGROUP_0", WaveIndexSymbol_WORKGROUP_0)
      .value("WORKGROUP_1", WaveIndexSymbol_WORKGROUP_1)
      .value("WORKGROUP_2", WaveIndexSymbol_WORKGROUP_2)
      .value("THREAD_0", WaveIndexSymbol_THREAD_0)
      .value("THREAD_1", WaveIndexSymbol_THREAD_1)
      .value("THREAD_2", WaveIndexSymbol_THREAD_2)
      .value("GPR_NUMBER", WaveIndexSymbol_GPR_NUMBER);

  //===---------------------------------------------------------------------===//
  // WaveIndexMappingAttr
  //===---------------------------------------------------------------------===//

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      d, "WaveIndexMappingAttr", mlirAttributeIsAWaveIndexMappingAttr,
      mlirWaveIndexMappingAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, std::vector<MlirAttribute> &symbols,
             MlirAffineMap start, MlirAffineMap step, MlirAffineMap stride,
             // MlirContext should always come last to allow for being
             // automatically deduced from context.
             MlirContext context) {
            intptr_t numSymbols = symbols.size();
            intptr_t numResults = mlirAffineMapGetNumResults(start);
            for (MlirAffineMap map : {start, step, stride}) {
              if (numSymbols != mlirAffineMapGetNumSymbols(map)) {
                throw nb::value_error("Expected symbols, start, step and "
                                      "stride to be co-indexed.");
              }
              if (mlirAffineMapGetNumDims(map) != 0) {
                throw nb::value_error("Maps should not involve dimensions.");
              }
              if (numResults != mlirAffineMapGetNumResults(map)) {
                throw nb::value_error(
                    "Maps should have the same number of results.");
              }
            }
            return cls(mlirWaveIndexMappingAttrGet(context, symbols.data(),
                                                   start, step, stride));
          },
          nb::arg("cls"), nb::arg("symbols"), nb::arg("start"), nb::arg("step"),
          nb::arg("stride"), nb::arg("context") = nb::none(),
          "Gets a wave.WaveIndexMappingAttr from a list of symbol attributes.");

  //===---------------------------------------------------------------------===//
  // WaveHyperparameterAttr
  //===---------------------------------------------------------------------===//

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      d, "WaveHyperparameterAttr", mlirAttributeIsAWaveHyperparameterAttr,
      mlirWaveHyperparameterAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, const nb::dict &symbolDict,
             // MlirContext should always come last to allow for being
             // automatically deduced from context.
             MlirContext context) {
            std::vector<MlirNamedAttribute> namedAttrs;
            namedAttrs.reserve(symbolDict.size());

            for (auto item : symbolDict) {
              // Get the key (symbol name)
              nb::handle key_handle = item.first;
              if (!nb::isinstance<nb::str>(key_handle)) {
                throw nb::type_error("Symbol dictionary key must be a string");
              }
              std::string symbolName = nb::cast<std::string>(key_handle);

              // Get the value (resolved value)
              nb::handle value_handle = item.second;
              if (!nb::isinstance<nb::int_>(value_handle)) {
                throw nb::type_error(
                    "Symbol dictionary value must be an integer");
              }
              int64_t resolvedValue;
              try {
                resolvedValue = nb::cast<int64_t>(value_handle);
              } catch (const nb::cast_error &e) {
                throw nb::value_error("Value is too large for int64_t");
              }

              namedAttrs.push_back(mlirNamedAttributeGet(
                  mlirIdentifierGet(context,
                                    mlirStringRefCreate(symbolName.data(),
                                                        symbolName.size())),
                  mlirIntegerAttrGet(mlirIntegerTypeGet(context, 64),
                                     resolvedValue)));
            }

            return cls(mlirWaveHyperparameterAttrGet(mlirDictionaryAttrGet(
                context, namedAttrs.size(), namedAttrs.data())));
          },
          nb::arg("cls"), nb::arg("symbol_dict"),
          nb::arg("context") = nb::none(),
          "Gets a wave.WaveHyperparameterAttr from parameters.");

  //===---------------------------------------------------------------------===//
  // WaveNormalFormAttr
  //===---------------------------------------------------------------------===//

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      d, "WaveNormalFormAttr", mlirAttributeIsAWaveNormalFormAttr,
      mlirWaveNormalFormAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, WaveNormalForm value, MlirContext context) {
            return cls(mlirWaveNormalFormAttrGet(context,
                                                 static_cast<uint32_t>(value)));
          },
          nb::arg("cls"), nb::arg("value"), nb::arg("context") = nb::none(),
          "Gets a wave.WaveNormalFormAttr from a normal form enum value.")
      .def_property_readonly(
          "value",
          [](MlirAttribute self) {
            return static_cast<WaveNormalForm>(
                mlirWaveNormalFormAttrGetValue(self));
          },
          "Returns the normal form enum value.");

  nb::enum_<WaveNormalForm>(d, "WaveNormalForm")
      .value("None_", WaveNormalFormNone)
      .value("FunctionBoundarySpecified",
             WaveNormalFormFunctionBoundarySpecified)
      .value("OpTypesSpecified", WaveNormalFormOpTypesSpecified)
      .value("IndexExprsSpecified", WaveNormalFormIndexExprsSpecified)
      .value("MemoryOnlyTypes", WaveNormalFormMemoryOnlyTypes)
      .value("AllTypesSpecified", WaveNormalFormAllTypesSPecified);

  //===---------------------------------------------------------------------===//
  // WaveWorkgroupDimAttr
  //===---------------------------------------------------------------------===//

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      d, "WaveWorkgroupDimAttr", mlirAttributeIsAWaveWorkgroupDimAttr,
      mlirWaveWorkgroupDimAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, WaveWorkgroupDim value,
             MlirContext context) {
            return cls(mlirWaveWorkgroupDimAttrGet(
                context, static_cast<uint32_t>(value)));
          },
          nb::arg("cls"), nb::arg("value"), nb::arg("context") = nb::none(),
          "Gets a wave.WaveWorkgroupDimAttr from a workgroup dim enum value.")
      .def_property_readonly(
          "value",
          [](MlirAttribute self) {
            return static_cast<WaveWorkgroupDim>(
                mlirWaveWorkgroupDimAttrGetValue(self));
          },
          "Workgroup dim enum value.");

  nb::enum_<WaveWorkgroupDim>(d, "WaveWorkgroupDim")
      .value("X", WaveWorkgroupDimX)
      .value("Y", WaveWorkgroupDimY)
      .value("Z", WaveWorkgroupDimZ);

  //===---------------------------------------------------------------------===//
  // WaveAddressSpaceAttr
  //===---------------------------------------------------------------------===//

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      d, "WaveAddressSpaceAttr", mlirAttributeIsAWaveAddressSpaceAttr,
      mlirWaveAddressSpaceAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, WaveAddressSpace value,
             MlirContext context) {
            return cls(mlirWaveAddressSpaceAttrGet(
                context, static_cast<uint32_t>(value)));
          },
          nb::arg("cls"), nb::arg("value"), nb::arg("context") = nb::none(),
          "Gets a wave.WaveAddressSpaceAttr from an address space enum value.")
      .def_property_readonly(
          "value",
          [](MlirAttribute self) {
            return static_cast<WaveAddressSpace>(
                mlirWaveAddressSpaceAttrGetValue(self));
          },
          "Enum value.");

  nb::enum_<WaveAddressSpace>(d, "WaveAddressSpace")
      .value("Unspecified", WaveAddressSpaceUnspecified)
      .value("Global", WaveAddressSpaceGlobal)
      .value("Shared", WaveAddressSpaceShared)
      .value("Register", WaveAddressSpaceRegister);

  //===---------------------------------------------------------------------===//
  // WaveMmaKindAttr
  //===---------------------------------------------------------------------===//

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      d, "WaveMmaKindAttr", mlirAttributeIsAWaveMmaKindAttr,
      mlirWaveMmaKindAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, WaveMmaKind value, MlirContext context) {
            return cls(
                mlirWaveMmaKindAttrGet(context, static_cast<uint32_t>(value)));
          },
          nb::arg("cls"), nb::arg("value"), nb::arg("context") = nb::none(),
          "Gets a wave.WaveMmaKindAttr from an MMA kind enum value.")
      .def_property_readonly(
          "value",
          [](MlirAttribute self) {
            return static_cast<WaveMmaKind>(mlirWaveMmaKindAttrGetValue(self));
          },
          "MMA kind enum value.");

  nb::enum_<WaveMmaKind>(d, "WaveMmaKind")
      // CDNA1
      .value("F32_16x16x16_F16", WaveMmaKind_F32_16x16x16_F16)
      .value("F32_32x32x8_F16", WaveMmaKind_F32_32x32x8_F16)
      .value("F32_16x16x32_K8_F16", WaveMmaKind_F32_16x16x32_K8_F16)
      .value("F32_32x32x16_K8_F16", WaveMmaKind_F32_32x32x16_K8_F16)
      .value("I32_16x16x16_I8", WaveMmaKind_I32_16x16x16_I8)
      .value("I32_32x32x8_I8", WaveMmaKind_I32_32x32x8_I8)
      // CDNA3
      .value("F32_16x16x32_F8", WaveMmaKind_F32_16x16x32_F8)
      .value("F32_32x32x16_F8", WaveMmaKind_F32_32x32x16_F8)
      .value("F32_16x16x32_K4_F8", WaveMmaKind_F32_16x16x32_K4_F8)
      .value("F32_32x32x16_K4_F8", WaveMmaKind_F32_32x32x16_K4_F8)
      .value("I32_16x16x32_I8", WaveMmaKind_I32_16x16x32_I8)
      .value("I32_32x32x16_I8", WaveMmaKind_I32_32x32x16_I8)
      // CDNA4
      .value("F32_16x16x128_F8F6F4", WaveMmaKind_F32_16x16x128_F8F6F4)
      .value("F32_32x32x64_F8F6F4", WaveMmaKind_F32_32x32x64_F8F6F4)
      .value("F32_32x32x16_BF16", WaveMmaKind_F32_32x32x16_BF16)
      .value("F32_16x16x32_BF16", WaveMmaKind_F32_16x16x32_BF16)
      .value("F32_32x32x16_F16", WaveMmaKind_F32_32x32x16_F16)
      .value("F32_16x16x32_F16", WaveMmaKind_F32_16x16x32_F16);

  //===---------------------------------------------------------------------===//
  // WaveExprListAttr
  //===---------------------------------------------------------------------===//

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      d, "WaveExprListAttr", mlirAttributeIsAWaveExprListAttr,
      mlirWaveExprListAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, std::vector<MlirAttribute> &symbols,
             MlirAffineMap map) {
            for (MlirAttribute attr : symbols) {
              if (!mlirAttributeIsAWaveSymbolAttr(attr) &&
                  !mlirAttributeIsAWaveIndexSymbolAttr(attr)) {
                throw nb::type_error(
                    "symbols must contain only WaveSymbolAttr or "
                    "WaveIndexSymbolAttr attributes");
              }
            }

            intptr_t numSymbols = symbols.size();
            if (numSymbols != mlirAffineMapGetNumSymbols(map)) {
              throw nb::value_error("Expected symbols to have as many "
                                    "entries as map have symbols.");
            }
            if (mlirAffineMapGetNumDims(map) != 0) {
              throw nb::value_error("Maps should not involve dimensions.");
            }
            return cls(mlirWaveExprListAttrGet(symbols.data(), map));
          },
          nb::arg("cls"), nb::arg("symbols"), nb::arg("map"),
          "Gets a wave.WaveExprListAttr from a list of symbol attributes.");

  //===---------------------------------------------------------------------===//
  // WaveReadWriteBoundsAttr
  //===---------------------------------------------------------------------===//

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      d, "WaveReadWriteBoundsAttr", mlirAttributeIsAWaveReadWriteBoundsAttr,
      mlirWaveReadWriteBoundsAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, const nb::dict &symDimDict,
             // MlirContext should always come last to allow for being
             // automatically deduced from context.
             MlirContext context) {
            std::vector<MlirNamedAttribute> namedAttrs;
            namedAttrs.reserve(symDimDict.size());

            for (auto [key, value] : symDimDict) {
              // Get the key (symbolic dimension)
              nb::handle key_handle = key;
              if (!nb::isinstance<nb::str>(key_handle)) {
                throw nb::type_error(
                    "Symbolic dimension dictionary key must be a string");
              }
              std::string symbolicDim = nb::cast<std::string>(key_handle);

              // Get the value (bound expression)
              MlirAttribute attr;
              try {
                attr = nb::cast<MlirAttribute>(value);
              } catch (const nb::cast_error &e) {
                throw nb::type_error(
                    "Symbolic dimension dictionary value must be an attribute");
              }
              if (!mlirAttributeIsAWaveExprListAttr(attr)) {
                throw nb::type_error("Symbolic dimension dictionary value must "
                                     "be a WaveExprListAttr");
              }

              namedAttrs.push_back(mlirNamedAttributeGet(
                  mlirIdentifierGet(context,
                                    mlirStringRefCreate(symbolicDim.data(),
                                                        symbolicDim.size())),
                  attr));
            }

            return cls(mlirWaveReadWriteBoundsAttrGet(mlirDictionaryAttrGet(
                context, namedAttrs.size(), namedAttrs.data())));
          },
          nb::arg("cls"), nb::arg("sym_dim_dict"),
          nb::arg("context") = nb::none(),
          "Gets a wave.WaveReadWriteBoundsAttr from parameters.");

  //===---------------------------------------------------------------------===//
  // HardwareConstraintAttr
  //===---------------------------------------------------------------------===//

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      d, "HardwareConstraintAttr", mlirAttributeIsAHardwareConstraintAttr,
      mlirWHardwareConstraintAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, unsigned threadsPerWave,
             const std::optional<std::vector<unsigned>> &wavesPerBlock,
             std::optional<MlirAttribute> mmaType,
             std::optional<MlirAttribute> vectorShapes, unsigned maxBitsPerLoad,
             MlirContext context) {
            unsigned *wavesPerBlockPtr = nullptr;
            size_t wavesPerBlockSize = 0;
            if (wavesPerBlock.has_value()) {
              wavesPerBlockPtr = const_cast<unsigned *>(wavesPerBlock->data());
              wavesPerBlockSize = wavesPerBlock->size();
            }

            return cls(mlirHardwareConstraintAttrGet(
                context, threadsPerWave, wavesPerBlockSize, wavesPerBlockPtr,
                mmaType.value_or(MlirAttribute()),
                vectorShapes.value_or(MlirAttribute()), maxBitsPerLoad));
          },
          nb::arg("cls"), nb::arg("threads_per_wave"),
          nb::arg("waves_per_block") = nb::none(),
          nb::arg("mma_type") = nb::none(),
          nb::arg("vector_shapes") = nb::none(),
          nb::arg("max_bits_per_load") = 128, nb::arg("context") = nb::none(),
          "Gets a wave.HardwareConstraintAttr from parameters.");

  //===---------------------------------------------------------------------===//
  // DeviceConstraintAttr
  //===---------------------------------------------------------------------===//

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      d, "DeviceConstraintAttr", mlirAttributeIsADeviceConstraintAttr,
      mlirDeviceConstraintAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, const std::string &dim,
             MlirAttribute tileSize, unsigned deviceDim, MlirContext context) {
            MlirStringRef dimStrRef =
                mlirStringRefCreate(dim.c_str(), dim.size());
            MlirAttribute dimAttr = mlirWaveSymbolAttrGet(context, dimStrRef);
            return cls(mlirDeviceConstraintAttrGet(context, dimAttr, tileSize,
                                                   deviceDim));
          },
          nb::arg("cls"), nb::arg("dim"), nb::arg("tile_size"),
          nb::arg("device_dim"), nb::arg("context") = nb::none(),
          "Gets a wave.DeviceConstraintAttr from parameters.");

  //===---------------------------------------------------------------------===//
  // WorkgroupConstraintAttr
  //===---------------------------------------------------------------------===//

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      d, "WorkgroupConstraintAttr", mlirAttributeIsAWorkgroupConstraintAttr,
      mlirWorkgroupConstraintAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, const std::string &dim,
             MlirAttribute tileSize, MlirAttribute workgroupDim, bool primary,
             MlirContext context) {
            MlirStringRef dimStrRef =
                mlirStringRefCreate(dim.c_str(), dim.size());
            MlirAttribute dimAttr = mlirWaveSymbolAttrGet(context, dimStrRef);
            return cls(mlirWorkgroupConstraintAttrGet(
                context, dimAttr, tileSize, workgroupDim, primary));
          },
          nb::arg("cls"), nb::arg("dim"), nb::arg("tile_size"),
          nb::arg("workgroup_dim"), nb::arg("primary") = true,
          nb::arg("context") = nb::none(),
          "Gets a wave.WorkgroupConstraintAttr from parameters.");

  //===---------------------------------------------------------------------===//
  // WaveConstraintAttr
  //===---------------------------------------------------------------------===//

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      d, "WaveConstraintAttr", mlirAttributeIsAWaveConstraintAttr,
      mlirWaveConstraintAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, const std::string &dim,
             MlirAttribute tileSize, MlirContext context) {
            MlirStringRef dimStrRef =
                mlirStringRefCreate(dim.c_str(), dim.size());
            MlirAttribute dimAttr = mlirWaveSymbolAttrGet(context, dimStrRef);
            return cls(mlirWaveConstraintAttrGet(context, dimAttr, tileSize));
          },
          nb::arg("cls"), nb::arg("dim"), nb::arg("tile_size"),
          nb::arg("context") = nb::none(),
          "Gets a wave.WaveConstraintAttr from parameters.");

  //===---------------------------------------------------------------------===//
  // TilingConstraintAttr
  //===---------------------------------------------------------------------===//

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      d, "TilingConstraintAttr", mlirAttributeIsATilingConstraintAttr,
      mlirTilingConstraintAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, const std::string &dim,
             MlirAttribute tileSize, MlirContext context) {
            MlirStringRef dimStrRef =
                mlirStringRefCreate(dim.c_str(), dim.size());
            MlirAttribute dimAttr = mlirWaveSymbolAttrGet(context, dimStrRef);
            return cls(mlirTilingConstraintAttrGet(context, dimAttr, tileSize));
          },
          nb::arg("cls"), nb::arg("dim"), nb::arg("tile_size"),
          nb::arg("context") = nb::none(),
          "Gets a wave.TilingConstraintAttr from parameters.");
}
