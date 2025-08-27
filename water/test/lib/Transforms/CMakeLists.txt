set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRWaterTestTransformsIncGen)

add_mlir_library(MLIRWaterTestTransforms
  Passes.cpp
  TestInUseAnalysis.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  MLIRWaterTestTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRWaterAnalysis
)
target_include_directories(MLIRWaterTestTransforms
  PRIVATE
  ${PROJECT_SOURCE_DIR}/test/lib
  ${PROJECT_BINARY_DIR}/test/lib
)
