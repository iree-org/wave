set(LLVM_TARGET_DEFINITIONS WaterTestDialect.td)
mlir_tablegen(WaterTestDialect.h.inc -gen-dialect-decls -dialect=water_test)
mlir_tablegen(WaterTestDialect.cpp.inc -gen-dialect-defs -dialect=water_test)
mlir_tablegen(WaterTestDialectOps.h.inc -gen-op-decls)
mlir_tablegen(WaterTestDialectOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRWaterTestDialectIncGen)

# Separate tablegen target for test normal form attribute to avoid
# pulling in Wave dialect enums.
set(LLVM_TARGET_DEFINITIONS TestNormalFormAttr.td)
mlir_tablegen(TestNormalFormAttr.h.inc -gen-attrdef-decls -attrdefs-dialect=water_test)
mlir_tablegen(TestNormalFormAttr.cpp.inc -gen-attrdef-defs -attrdefs-dialect=water_test)
add_public_tablegen_target(MLIRTestNormalFormAttrIncGen)

add_mlir_dialect_library(MLIRWaterTestDialect
  WaterTestDialect.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  MLIRWaterTestDialectIncGen
  MLIRTestNormalFormAttrIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRNormalFormDialect
  MLIRWaveDialect
)

target_include_directories(MLIRWaterTestDialect
  PRIVATE
  ${PROJECT_BINARY_DIR}/test/lib/Dialect
)
