// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir-c/AffineMap.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
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
  // WaveIndexMappingAttr
  //===---------------------------------------------------------------------===//

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      d, "WaveIndexMappingAttr", mlirAttributeIsAWaveIndexMappingAttr,
      mlirWaveIndexMappingAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, const std::vector<std::string> &symbolNames,
             MlirAffineMap start, MlirAffineMap step, MlirAffineMap stride,
             // MlirContext should always come last to allow for being
             // automatically deduced from context.
             MlirContext context) {
            std::vector<MlirAttribute> symbolAttrs;
            symbolAttrs.reserve(symbolNames.size());

            for (const std::string &symbolName : symbolNames) {
              MlirStringRef symbolNameStrRef =
                  mlirStringRefCreate(symbolName.data(), symbolName.size());
              MlirAttribute symbolAttr =
                  mlirWaveSymbolAttrGet(context, symbolNameStrRef);
              symbolAttrs.push_back(symbolAttr);
            }

            intptr_t numSymbols = symbolAttrs.size();
            intptr_t numResults = mlirAffineMapGetNumResults(start);
            for (MlirAffineMap map : {start, step, stride}) {
              if (numSymbols != mlirAffineMapGetNumSymbols(map)) {
                throw nb::value_error("Expected symbol_names, start, step and "
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
            return cls(mlirWaveIndexMappingAttrGet(context, symbolAttrs.data(),
                                                   start, step, stride));
          },
          nb::arg("cls"), nb::arg("symbol_names"), nb::arg("start"),
          nb::arg("step"), nb::arg("stride"), nb::arg("context") = nb::none(),
          "Gets a wave.WaveIndexMappingAttr from parameters.");

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
}
