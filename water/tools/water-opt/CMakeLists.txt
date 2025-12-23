get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIRGPUDialect
        MLIRGPUToLLVMIRTranslation
        MLIRLLVMToLLVMIRTranslation
        MLIRLinalgTransformOps
        MLIRMemRefTransformOps
        MLIROptLib
        MLIRROCDLDialect
        MLIRTransformDialectTransforms
        MLIRWaterTransforms
        MLIRWaveTransforms

        MLIRWaterTestTransforms
        MLIRWaterTestDialect
)

add_llvm_executable(water-opt
        water-opt.cpp
        WaterOptMain.cpp
)

llvm_update_compile_flags(water-opt)
target_link_libraries(water-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(water-opt)

install(TARGETS water-opt)
