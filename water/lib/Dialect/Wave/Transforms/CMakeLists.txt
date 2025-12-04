add_mlir_dialect_library(MLIRWaveTransforms
  InferTypes.cpp
  LoweringPatterns.cpp
  LowerReadWriteOps.cpp
  LowerWaveToMLIR.cpp
  TypeConverter.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/water/

  DEPENDS
  MLIRWaterWavePassesIncGen

  LINK_LIBS PUBLIC
  MLIRAMDGPUDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  MLIRVectorDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRWaveDialect
)
