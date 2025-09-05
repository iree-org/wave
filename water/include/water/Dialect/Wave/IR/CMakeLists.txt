add_custom_target(MLIRWaveIncGen)

set(LLVM_TARGET_DEFINITIONS WaveAttrs.td)
mlir_tablegen(WaveEnums.h.inc -gen-enum-decls)
mlir_tablegen(WaveEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(WaveAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(WaveAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRWaveAttrsIncGen)
add_dependencies(MLIRWaveIncGen MLIRWaveAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS WaveDialect.td)
mlir_tablegen(WaveDialect.h.inc -gen-dialect-decls -dialect=wave)
mlir_tablegen(WaveDialect.cpp.inc -gen-dialect-defs -dialect=wave)
add_public_tablegen_target(MLIRWaveDialectIncGen)
add_dependencies(MLIRWaveIncGen MLIRWaveTypesIncGen)

set(LLVM_TARGET_DEFINITIONS WaveOps.td)
mlir_tablegen(WaveOps.h.inc -gen-op-decls)
mlir_tablegen(WaveOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRWaveOpsIncGen)
add_dependencies(MLIRWaveIncGen MLIRWaveOpsIncGen)

set(LLVM_TARGET_DEFINITIONS WaveTypes.td)
mlir_tablegen(WaveTypes.h.inc -gen-typedef-decls -typedefs-dialect=wave)
mlir_tablegen(WaveTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=wave)
add_public_tablegen_target(MLIRWaveTypesIncGen)
add_dependencies(MLIRWaveIncGen MLIRWaveTypesIncGen)

add_dependencies(mlir-headers MLIRWaveIncGen)

add_mlir_doc(Wave Wave Dialects/ -gen-dialect-doc -dialect wave)
