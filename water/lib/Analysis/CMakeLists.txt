add_mlir_library(MLIRWaterAnalysis
  InUseForSpeculation.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/water

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
)
