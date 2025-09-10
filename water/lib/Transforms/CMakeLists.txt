add_mlir_dialect_library(MLIRWaterTransforms
  AccessCheckers.cpp
  CheckStaticAssertions.cpp
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
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRWaterAnalysis
)
