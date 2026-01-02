set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name WaterConversion)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix WaterConversion)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix WaterConversion)
add_mlir_generic_tablegen_target(MLIRWaterConversionPassIncGen)

add_mlir_doc(Passes WaterConversionPasses ./ -gen-pass-doc)
