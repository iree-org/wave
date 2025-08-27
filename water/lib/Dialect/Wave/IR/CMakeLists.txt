add_mlir_dialect_library(MLIRWaveDialect
  WaveAttrs.cpp
  WaveDialect.cpp
  WaveOps.cpp
  WaveTypes.cpp

  DEPENDS
  MLIRWaveIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
