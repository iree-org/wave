// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir-c/AffineMap.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "water/c/Dialects.h"

#include "nanobind/nanobind.h"

#include "mlir/CAPI/Support.h"

namespace nb = nanobind;

//===---------------------------------------------------------------------===//
// WaveSymbolAttr
//===---------------------------------------------------------------------===//

struct PyWaveSymbolAttr
    : mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute<
          PyWaveSymbolAttr> {
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAWaveSymbolAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirWaveSymbolAttrGetTypeID;
  static constexpr const char *pyClassName = "WaveSymbolAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &symbolName,
           mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
               context) {
          MlirStringRef symbolNameStrRef =
              mlirStringRefCreate(symbolName.data(), symbolName.size());
          return PyWaveSymbolAttr(
              context->getRef(),
              mlirWaveSymbolAttrGet(context->get(), symbolNameStrRef));
        },
        nb::arg("symbolName"), nb::arg("context") = nb::none(),
        "Gets a wave.WaveSymbolAttr from parameters.");
    c.def_prop_ro("name", [](MlirAttribute self) {
      return mlirWaveSymbolAttrGetName(self);
    });
  }
};

//===---------------------------------------------------------------------===//
// WaveIterSymbolAttr
//===---------------------------------------------------------------------===//

struct PyWaveIterSymbolAttr
    : mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute<
          PyWaveIterSymbolAttr> {
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsAWaveIterSymbolAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirWaveIterSymbolAttrGetTypeID;
  static constexpr const char *pyClassName = "WaveIterSymbolAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &symbolName,
           mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
               context) {
          MlirStringRef symbolNameStrRef =
              mlirStringRefCreate(symbolName.data(), symbolName.size());
          return PyWaveIterSymbolAttr(
              context->getRef(),
              mlirWaveIterSymbolAttrGet(context->get(), symbolNameStrRef));
        },
        nb::arg("symbolName"), nb::arg("context") = nb::none(),
        "Gets a wave.WaveIterSymbolAttr from parameters.");
    c.def_prop_ro("name", [](MlirAttribute self) {
      return mlirWaveIterSymbolAttrGetName(self);
    });
  }
};

//===---------------------------------------------------------------------===//
// WaveIndexSymbolAttr
//===---------------------------------------------------------------------===//

struct PyWaveIndexSymbolAttr
    : mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute<
          PyWaveIndexSymbolAttr> {
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsAWaveIndexSymbolAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirWaveIndexSymbolAttrGetTypeID;
  static constexpr const char *pyClassName = "WaveIndexSymbolAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](WaveIndexSymbol value,
           mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
               context) {
          return PyWaveIndexSymbolAttr(
              context->getRef(),
              mlirWaveIndexSymbolAttrGet(context->get(),
                                         static_cast<uint32_t>(value)));
        },
        nb::arg("value"), nb::arg("context") = nb::none(),
        "Gets a wave.WaveIndexSymbolAttr from an index symbol enum value.");
    c.def_prop_ro(
        "value",
        [](MlirAttribute self) {
          return static_cast<WaveIndexSymbol>(
              mlirWaveIndexSymbolAttrGetValue(self));
        },
        "Index symbol enum value.");
  }
};

//===---------------------------------------------------------------------===//
// WaveIndexMappingAttr
//===---------------------------------------------------------------------===//

struct PyWaveIndexMappingAttr
    : mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute<
          PyWaveIndexMappingAttr> {
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsAWaveIndexMappingAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirWaveIndexMappingAttrGetTypeID;
  static constexpr const char *pyClassName = "WaveIndexMappingAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<MlirAttribute> &symbols, MlirAffineMap start,
           MlirAffineMap step, MlirAffineMap stride,
           mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
               context) {
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
          return PyWaveIndexMappingAttr(
              context->getRef(),
              mlirWaveIndexMappingAttrGet(context->get(), symbols.data(), start,
                                          step, stride));
        },
        nb::arg("symbols"), nb::arg("start"), nb::arg("step"),
        nb::arg("stride"), nb::arg("context") = nb::none(),
        "Gets a wave.WaveIndexMappingAttr from a list of symbol attributes.");
    c.def_prop_ro("start", [](MlirAttribute self) {
      return mlirWaveIndexMappingAttrGetStart(self);
    });
    c.def_prop_ro("step", [](MlirAttribute self) {
      return mlirWaveIndexMappingAttrGetStep(self);
    });
    c.def_prop_ro("stride", [](MlirAttribute self) {
      return mlirWaveIndexMappingAttrGetStride(self);
    });
    c.def_prop_ro("symbols", [](MlirAttribute self) {
      std::vector<MlirAttribute> symbols;
      intptr_t numSymbols = mlirWaveIndexMappingAttrGetNumSymbols(self);
      symbols.reserve(numSymbols);
      for (intptr_t i = 0; i < numSymbols; i++) {
        symbols.push_back(mlirWaveIndexMappingAttrGetSymbol(self, i));
      }
      return symbols;
    });
  }
};

//===---------------------------------------------------------------------===//
// WaveHyperparameterAttr
//===---------------------------------------------------------------------===//

struct PyWaveHyperparameterAttr
    : mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute<
          PyWaveHyperparameterAttr> {
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsAWaveHyperparameterAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirWaveHyperparameterAttrGetTypeID;
  static constexpr const char *pyClassName = "WaveHyperparameterAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const nb::dict &symbolDict,
           mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
               context) {
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
                mlirIdentifierGet(
                    context->get(),
                    mlirStringRefCreate(symbolName.data(), symbolName.size())),
                mlirIntegerAttrGet(mlirIntegerTypeGet(context->get(), 64),
                                   resolvedValue)));
          }

          return PyWaveHyperparameterAttr(
              context->getRef(),
              mlirWaveHyperparameterAttrGet(mlirDictionaryAttrGet(
                  context->get(), namedAttrs.size(), namedAttrs.data())));
        },
        nb::arg("symbol_dict"), nb::arg("context") = nb::none(),
        "Gets a wave.WaveHyperparameterAttr from parameters.");
  }
};

//===---------------------------------------------------------------------===//
// WaveNormalFormAttr
//===---------------------------------------------------------------------===//

struct PyWaveNormalFormAttr
    : mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute<
          PyWaveNormalFormAttr> {
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsAWaveNormalFormAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirWaveNormalFormAttrGetTypeID;
  static constexpr const char *pyClassName = "WaveNormalFormAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](WaveNormalForm value,
           mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
               context) {
          return PyWaveNormalFormAttr(
              context->getRef(),
              mlirWaveNormalFormAttrGet(context->get(),
                                        static_cast<uint32_t>(value)));
        },
        nb::arg("value"), nb::arg("context") = nb::none(),
        "Gets a wave.WaveNormalFormAttr from a normal form enum value.");
    c.def_prop_ro(
        "value",
        [](MlirAttribute self) {
          return static_cast<WaveNormalForm>(
              mlirWaveNormalFormAttrGetValue(self));
        },
        "Returns the normal form enum value.");
  }
};

//===---------------------------------------------------------------------===//
// WaveWorkgroupDimAttr
//===---------------------------------------------------------------------===//

struct PyWaveWorkgroupDimAttr
    : mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute<
          PyWaveWorkgroupDimAttr> {
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsAWaveWorkgroupDimAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirWaveWorkgroupDimAttrGetTypeID;
  static constexpr const char *pyClassName = "WaveWorkgroupDimAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](WaveWorkgroupDim value,
           mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
               context) {
          return PyWaveWorkgroupDimAttr(
              context->getRef(),
              mlirWaveWorkgroupDimAttrGet(context->get(),
                                          static_cast<uint32_t>(value)));
        },
        nb::arg("value"), nb::arg("context") = nb::none(),
        "Gets a wave.WaveWorkgroupDimAttr from a workgroup dim enum value.");
    c.def_prop_ro(
        "value",
        [](MlirAttribute self) {
          return static_cast<WaveWorkgroupDim>(
              mlirWaveWorkgroupDimAttrGetValue(self));
        },
        "Workgroup dim enum value.");
  }
};

//===---------------------------------------------------------------------===//
// WaveAddressSpaceAttr
//===---------------------------------------------------------------------===//

struct PyWaveAddressSpaceAttr
    : mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute<
          PyWaveAddressSpaceAttr> {
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsAWaveAddressSpaceAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirWaveAddressSpaceAttrGetTypeID;
  static constexpr const char *pyClassName = "WaveAddressSpaceAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](WaveAddressSpace value,
           mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
               context) {
          return PyWaveAddressSpaceAttr(
              context->getRef(),
              mlirWaveAddressSpaceAttrGet(context->get(),
                                          static_cast<uint32_t>(value)));
        },
        nb::arg("value"), nb::arg("context") = nb::none(),
        "Gets a wave.WaveAddressSpaceAttr from an address space enum value.");
    c.def_prop_ro(
        "value",
        [](MlirAttribute self) {
          return static_cast<WaveAddressSpace>(
              mlirWaveAddressSpaceAttrGetValue(self));
        },
        "Enum value.");
  }
};

//===---------------------------------------------------------------------===//
// WaveMmaKindAttr
//===---------------------------------------------------------------------===//

struct PyWaveMmaKindAttr
    : mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute<
          PyWaveMmaKindAttr> {
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAWaveMmaKindAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirWaveMmaKindAttrGetTypeID;
  static constexpr const char *pyClassName = "WaveMmaKindAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](WaveMmaKind value,
           mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
               context) {
          return PyWaveMmaKindAttr(
              context->getRef(),
              mlirWaveMmaKindAttrGet(context->get(),
                                     static_cast<uint32_t>(value)));
        },
        nb::arg("value"), nb::arg("context") = nb::none(),
        "Gets a wave.WaveMmaKindAttr from an MMA kind enum value.");
    c.def_prop_ro(
        "value",
        [](MlirAttribute self) {
          return static_cast<WaveMmaKind>(mlirWaveMmaKindAttrGetValue(self));
        },
        "MMA kind enum value.");
  }
};

//===---------------------------------------------------------------------===//
// WaveExprListAttr
//===---------------------------------------------------------------------===//

struct PyWaveExprListAttr
    : mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute<
          PyWaveExprListAttr> {
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAWaveExprListAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirWaveExprListAttrGetTypeID;
  static constexpr const char *pyClassName = "WaveExprListAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<MlirAttribute> &symbols, MlirAffineMap map) {
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
          MlirContext context = mlirAffineMapGetContext(map);
          return PyWaveExprListAttr(
              mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyMlirContext::
                  forContext(context),
              mlirWaveExprListAttrGet(symbols.data(), map));
        },
        nb::arg("symbols"), nb::arg("map"),
        "Gets a wave.WaveExprListAttr from a list of symbol attributes.");
  }
};

//===---------------------------------------------------------------------===//
// WaveReadWriteBoundsAttr
//===---------------------------------------------------------------------===//

struct PyWaveReadWriteBoundsAttr
    : mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute<
          PyWaveReadWriteBoundsAttr> {
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsAWaveReadWriteBoundsAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirWaveReadWriteBoundsAttrGetTypeID;
  static constexpr const char *pyClassName = "WaveReadWriteBoundsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const nb::dict &symDimDict,
           mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
               context) {
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
                mlirIdentifierGet(context->get(),
                                  mlirStringRefCreate(symbolicDim.data(),
                                                      symbolicDim.size())),
                attr));
          }

          return PyWaveReadWriteBoundsAttr(
              context->getRef(),
              mlirWaveReadWriteBoundsAttrGet(mlirDictionaryAttrGet(
                  context->get(), namedAttrs.size(), namedAttrs.data())));
        },
        nb::arg("sym_dim_dict"), nb::arg("context") = nb::none(),
        "Gets a wave.WaveReadWriteBoundsAttr from parameters.");
  }
};

//===---------------------------------------------------------------------===//
// HardwareConstraintAttr
//===---------------------------------------------------------------------===//

struct PyHardwareConstraintAttr
    : mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute<
          PyHardwareConstraintAttr> {
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsAHardwareConstraintAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirWHardwareConstraintAttrGetTypeID;
  static constexpr const char *pyClassName = "HardwareConstraintAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](unsigned threadsPerWave,
           const std::optional<std::vector<unsigned>> &wavesPerBlock,
           std::optional<MlirAttribute> mmaType,
           std::optional<MlirAttribute> vectorShapes, unsigned maxBitsPerLoad,
           mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
               context) {
          unsigned *wavesPerBlockPtr = nullptr;
          size_t wavesPerBlockSize = 0;
          if (wavesPerBlock.has_value()) {
            wavesPerBlockPtr = const_cast<unsigned *>(wavesPerBlock->data());
            wavesPerBlockSize = wavesPerBlock->size();
          }

          return PyHardwareConstraintAttr(
              context->getRef(),
              mlirHardwareConstraintAttrGet(
                  context->get(), threadsPerWave, wavesPerBlockSize,
                  wavesPerBlockPtr, mmaType.value_or(MlirAttribute()),
                  vectorShapes.value_or(MlirAttribute()), maxBitsPerLoad));
        },
        nb::arg("threads_per_wave"), nb::arg("waves_per_block") = nb::none(),
        nb::arg("mma_type") = nb::none(), nb::arg("vector_shapes") = nb::none(),
        nb::arg("max_bits_per_load") = 128, nb::arg("context") = nb::none(),
        "Gets a wave.HardwareConstraintAttr from parameters.");
  }
};

//===---------------------------------------------------------------------===//
// DeviceConstraintAttr
//===---------------------------------------------------------------------===//

struct PyDeviceConstraintAttr
    : mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute<
          PyDeviceConstraintAttr> {
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsADeviceConstraintAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirDeviceConstraintAttrGetTypeID;
  static constexpr const char *pyClassName = "DeviceConstraintAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &dim, MlirAttribute tileSize, unsigned deviceDim,
           mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
               context) {
          MlirStringRef dimStrRef =
              mlirStringRefCreate(dim.c_str(), dim.size());
          MlirAttribute dimAttr =
              mlirWaveSymbolAttrGet(context->get(), dimStrRef);
          return PyDeviceConstraintAttr(
              context->getRef(),
              mlirDeviceConstraintAttrGet(context->get(), dimAttr, tileSize,
                                          deviceDim));
        },
        nb::arg("dim"), nb::arg("tile_size"), nb::arg("device_dim"),
        nb::arg("context") = nb::none(),
        "Gets a wave.DeviceConstraintAttr from parameters.");
  }
};

//===---------------------------------------------------------------------===//
// WorkgroupConstraintAttr
//===---------------------------------------------------------------------===//

struct PyWorkgroupConstraintAttr
    : mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute<
          PyWorkgroupConstraintAttr> {
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsAWorkgroupConstraintAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirWorkgroupConstraintAttrGetTypeID;
  static constexpr const char *pyClassName = "WorkgroupConstraintAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &dim, MlirAttribute tileSize,
           MlirAttribute workgroupDim, bool primary,
           mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
               context) {
          MlirStringRef dimStrRef =
              mlirStringRefCreate(dim.c_str(), dim.size());
          MlirAttribute dimAttr =
              mlirWaveSymbolAttrGet(context->get(), dimStrRef);
          return PyWorkgroupConstraintAttr(
              context->getRef(),
              mlirWorkgroupConstraintAttrGet(context->get(), dimAttr, tileSize,
                                             workgroupDim, primary));
        },
        nb::arg("dim"), nb::arg("tile_size"), nb::arg("workgroup_dim"),
        nb::arg("primary") = true, nb::arg("context") = nb::none(),
        "Gets a wave.WorkgroupConstraintAttr from parameters.");
  }
};

//===---------------------------------------------------------------------===//
// WaveConstraintAttr
//===---------------------------------------------------------------------===//

struct PyWaveConstraintAttr
    : mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute<
          PyWaveConstraintAttr> {
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsAWaveConstraintAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirWaveConstraintAttrGetTypeID;
  static constexpr const char *pyClassName = "WaveConstraintAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &dim, MlirAttribute tileSize,
           mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
               context) {
          MlirStringRef dimStrRef =
              mlirStringRefCreate(dim.c_str(), dim.size());
          MlirAttribute dimAttr =
              mlirWaveSymbolAttrGet(context->get(), dimStrRef);
          return PyWaveConstraintAttr(
              context->getRef(),
              mlirWaveConstraintAttrGet(context->get(), dimAttr, tileSize));
        },
        nb::arg("dim"), nb::arg("tile_size"), nb::arg("context") = nb::none(),
        "Gets a wave.WaveConstraintAttr from parameters.");
  }
};

//===---------------------------------------------------------------------===//
// TilingConstraintAttr
//===---------------------------------------------------------------------===//

struct PyTilingConstraintAttr
    : mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute<
          PyTilingConstraintAttr> {
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsATilingConstraintAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTilingConstraintAttrGetTypeID;
  static constexpr const char *pyClassName = "TilingConstraintAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &dim, MlirAttribute tileSize,
           mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
               context) {
          MlirStringRef dimStrRef =
              mlirStringRefCreate(dim.c_str(), dim.size());
          MlirAttribute dimAttr =
              mlirWaveSymbolAttrGet(context->get(), dimStrRef);
          return PyTilingConstraintAttr(
              context->getRef(),
              mlirTilingConstraintAttrGet(context->get(), dimAttr, tileSize));
        },
        nb::arg("dim"), nb::arg("tile_size"), nb::arg("context") = nb::none(),
        "Gets a wave.TilingConstraintAttr from parameters.");
  }
};

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
  d.def(
      "register_passes", []() { mlirWaveDialectRegisterPasses(); },
      "Registers the wave dialect passes.");

  // Export dialect constants
  d.attr("WAVE_CONSTRAINTS_ATTR_NAME") = mlirWaveDialectConstraintsAttrName;

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

  nb::enum_<WaveNormalForm>(d, "WaveNormalForm")
      .value("None_", WaveNormalFormNone)
      .value("FunctionBoundarySpecified",
             WaveNormalFormFunctionBoundarySpecified)
      .value("OpTypesSpecified", WaveNormalFormOpTypesSpecified)
      .value("IndexExprsSpecified", WaveNormalFormIndexExprsSpecified)
      .value("MemoryOnlyTypes", WaveNormalFormMemoryOnlyTypes)
      .value("AllTypesSpecified", WaveNormalFormAllTypesSPecified);

  nb::enum_<WaveWorkgroupDim>(d, "WaveWorkgroupDim")
      .value("X", WaveWorkgroupDimX)
      .value("Y", WaveWorkgroupDimY)
      .value("Z", WaveWorkgroupDimZ);

  nb::enum_<WaveAddressSpace>(d, "WaveAddressSpace")
      .value("Unspecified", WaveAddressSpaceUnspecified)
      .value("Global", WaveAddressSpaceGlobal)
      .value("Shared", WaveAddressSpaceShared)
      .value("Register", WaveAddressSpaceRegister);

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

  PyWaveSymbolAttr::bind(d);
  PyWaveIterSymbolAttr::bind(d);
  PyWaveIndexSymbolAttr::bind(d);
  PyWaveIndexMappingAttr::bind(d);
  PyWaveHyperparameterAttr::bind(d);
  PyWaveNormalFormAttr::bind(d);
  PyWaveWorkgroupDimAttr::bind(d);
  PyWaveAddressSpaceAttr::bind(d);
  PyWaveMmaKindAttr::bind(d);
  PyWaveExprListAttr::bind(d);
  PyWaveReadWriteBoundsAttr::bind(d);
  PyHardwareConstraintAttr::bind(d);
  PyDeviceConstraintAttr::bind(d);
  PyWorkgroupConstraintAttr::bind(d);
  PyWaveConstraintAttr::bind(d);
  PyTilingConstraintAttr::bind(d);
}
