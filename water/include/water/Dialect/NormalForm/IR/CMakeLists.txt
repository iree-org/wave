add_custom_target(MLIRNormalFormIncGen)

set(LLVM_TARGET_DEFINITIONS NormalFormDialect.td)
mlir_tablegen(NormalFormDialect.h.inc -gen-dialect-decls -dialect=normalform)
mlir_tablegen(NormalFormDialect.cpp.inc -gen-dialect-defs -dialect=normalform)
add_public_tablegen_target(MLIRNormalFormDialectIncGen)

set(LLVM_TARGET_DEFINITIONS NormalFormOps.td)
mlir_tablegen(NormalFormOps.h.inc -gen-op-decls)
mlir_tablegen(NormalFormOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRNormalFormOpsIncGen)
add_dependencies(MLIRNormalFormIncGen MLIRNormalFormOpsIncGen)

set(LLVM_TARGET_DEFINITIONS NormalFormInterfaces.td)
mlir_tablegen(NormalFormAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(NormalFormAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRNormalFormInterfacesIncGen)
add_dependencies(MLIRNormalFormIncGen MLIRNormalFormInterfacesIncGen)
add_mlir_doc(NormalFormInterfaces NormalFormAttrInterfaces Dialects/ -gen-attr-interface-docs)

add_dependencies(mlir-headers MLIRNormalFormIncGen)

add_mlir_doc(NormalForm NormalForm Dialects/ -gen-dialect-doc -dialect normalform)
