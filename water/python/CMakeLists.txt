option(WATER_ENABLE_PYTHON "Enable building Python bindings" OFF)

if (WATER_ENABLE_PYTHON)
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
  include(AddMLIRPython)

  # Avoid clashing with IREE Python bindings
  set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "water_mlir")
  add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=water_mlir.")

  declare_mlir_python_sources(WaterPythonSources)

  declare_mlir_python_sources(WaterPythonSources.Root
    ADD_TO_PARENT WaterPythonSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/water_mlir"
    SOURCES
      __init__.py
  )

  declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT WaterPythonSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/water_mlir"
    TD_FILE dialects/WaveOps.td
    SOURCES
      dialects/wave.py
    DIALECT_NAME wave)

  declare_mlir_python_extension(WaterPythonSources.NanobindExtension
    MODULE_NAME _waterDialects
    ADD_TO_PARENT WaterPythonSources
    SOURCES
      WaterExtensionNanobind.cpp
    EMBED_CAPI_LINK_LIBS
      WaterCAPI
    PYTHON_BINDINGS_LIBRARY nanobind
  )

  add_mlir_python_common_capi_library(WaterPythonCAPI
    INSTALL_COMPONENT WaterPythonModules
    INSTALL_DESTINATION water_mlir/_mlir_libs
    OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_packages/water_mlir/_mlir_libs"
    RELATIVE_INSTALL_ROOT ".."
    DECLARED_SOURCES
      WaterPythonSources
      MLIRPythonExtension.RegisterEverything
      MLIRPythonSources.Core
  )

  add_mlir_python_modules(WaterPythonModules
    ROOT_PREFIX "${CMAKE_BINARY_DIR}/python_packages/water_mlir"
    INSTALL_PREFIX "water_mlir"
    DECLARED_SOURCES
      WaterPythonSources
      MLIRPythonExtension.RegisterEverything
      MLIRPythonSources
    COMMON_CAPI_LINK_LIBS
      WaterPythonCAPI
  )
endif()
