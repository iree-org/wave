add_mlir_library(MLIRWaterAnalysis
  InUseForSpeculation.cpp
  UniformityAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/water

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRGPUDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
)
