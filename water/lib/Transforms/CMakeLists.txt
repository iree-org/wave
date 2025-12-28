add_mlir_dialect_library(MLIRWaterTransforms
  AccessCheckers.cpp
  AllocToAlloca.cpp
  AssembleISA.cpp
  CheckStaticAssertions.cpp
  DropTransformOps.cpp
  GPUModuleToBinary.cpp
  GPUToGPURuntime.cpp
  MemrefDecomposition.cpp
  SLPVectorizer.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/water

  DEPENDS
  MLIRWaterPassesIncGen

  LINK_LIBS PUBLIC
  MLIRAMDGPUDialect
  MLIRAMDGPUToROCDL
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
  MLIRSCFTransforms
  MLIRTransformDialect
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRWaterAnalysis
)
