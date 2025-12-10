add_mlir_dialect_library(MLIRWaterTransforms
  AccessCheckers.cpp
  AllocToAlloca.cpp
  AssembleISA.cpp
  CheckStaticAssertions.cpp
  GPUModuleToBinary.cpp
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
  MLIRROCDLTarget
  MLIRRewrite
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRWaterAnalysis
)
