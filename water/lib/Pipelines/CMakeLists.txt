add_mlir_library(MLIRWaterPipelines
  Pipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/water

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRNormalFormDialect
  MLIRNormalFormTransforms
  MLIRPass
  MLIRTransforms
  MLIRWaveTransforms
)
