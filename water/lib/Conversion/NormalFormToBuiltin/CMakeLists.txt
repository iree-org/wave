add_mlir_conversion_library(MLIRNormalFormToBuiltin
  NormalFormToBuiltin.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/NormalFormToBuiltin

  DEPENDS
  MLIRWaterConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRNormalFormDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  )
