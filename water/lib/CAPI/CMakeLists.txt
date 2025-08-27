add_mlir_public_c_api_library(WaterCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
    MLIRWaveDialect
)
