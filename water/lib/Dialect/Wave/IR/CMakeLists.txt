add_mlir_dialect_library(MLIRWaveDialect
  IndexExpr.cpp
  IndexExprInference.cpp
  WaveAttrs.cpp
  WaveDialect.cpp
  WaveInterfaces.cpp
  WaveOps.cpp
  WaveTypes.cpp
  WaveUtils.cpp

  DEPENDS
  MLIRWaveIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRIR
  MLIRNormalFormDialect
)

# Install the Wave dialect library so Python can find it at runtime.
install(TARGETS MLIRWaveDialect
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
