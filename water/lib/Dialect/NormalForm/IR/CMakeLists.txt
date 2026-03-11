add_mlir_dialect_library(MLIRNormalFormDialect
  NormalFormDialect.cpp
  NormalFormInterfaces.cpp
  NormalFormOps.cpp

  DEPENDS
  MLIRNormalFormIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

# Install the Wave dialect library so Python can find it at runtime.
install(TARGETS MLIRNormalFormDialect
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
