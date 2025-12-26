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
      sympy_to_affine_converter.py
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
    PRIVATE_LINK_LIBS
      LLVMSupport
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
      MLIRPythonSources.Dialects.amdgpu
      MLIRPythonSources.Dialects.transform
      MLIRPythonSources.Dialects.transform.interpreter
      MLIRPythonSources.Dialects.transform.extras
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

  if (UNIX)
    target_compile_options(WaterPythonCAPI
      PRIVATE "-fvisibility=hidden"
    )
    target_link_options(WaterPythonCAPI
      PRIVATE "-fvisibility=hidden"
    )
    if (APPLE)
      # See https://github.com/numba/numba-mlir/blob/fd9609d81dab96b0658392886ebc470ea2fa4eeb/numba_mlir/numba_mlir/mlir_compiler/CMakeLists.txt#L71
      message(SEND_ERROR "symbol management not implemented on Darwin")
    else()
      target_link_options(WaterPythonCAPI
        PRIVATE
        "-Wl,--no-undefined"
        "-Wl,-z,defs"
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/capi.lds"
      )
      set_target_properties(WaterPythonCAPI
        PROPERTIES
	LINK_DEPENDS
	"${CMAKE_CURRENT_SOURCE_DIR}/capi.lds"
      )
    endif()
  endif()
endif()
