add_mlir_dialect_library(MLIRWaterTransforms
  AccessCheckers.cpp
  CheckStaticAssertions.cpp
  GPUToGPURuntime.cpp
  SLPVectorizer.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/water

  DEPENDS
  MLIRWaterPassesIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRWaterAnalysis
)
