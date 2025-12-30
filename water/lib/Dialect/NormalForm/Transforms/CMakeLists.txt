add_mlir_dialect_library(MLIRNormalFormTransforms
  LowerNormalFormModule.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/water/

  DEPENDS
  MLIRWaterNormalFormPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRNormalFormDialect
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
)
