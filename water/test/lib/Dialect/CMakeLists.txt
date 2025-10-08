set(LLVM_TARGET_DEFINITIONS WaterTestDialect.td)
mlir_tablegen(WaterTestDialect.h.inc -gen-dialect-decls -dialect=water_test)
mlir_tablegen(WaterTestDialect.cpp.inc -gen-dialect-defs -dialect=water_test)
mlir_tablegen(WaterTestDialectOps.h.inc -gen-op-decls)
mlir_tablegen(WaterTestDialectOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRWaterTestDialectIncGen)

add_mlir_library(MLIRWaterTestDialect
  WaterTestDialect.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  MLIRWaterTestDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRWaveDialect
)

target_include_directories(MLIRWaterTestDialect
  PRIVATE
  ${PROJECT_BINARY_DIR}/test/lib/Dialect
)
