add_subdirectory(water-opt)


# Find lld
find_program(LLD_BINARY NAMES lld
  HINTS ${LLVM_TOOLS_BINARY_DIR}
  DOC "Path to lld binary"
)

# `water-gpu-module-to-binary` and upstream `gpu-module-to-binary` passes need
# ld.lld and they are very opinionated about search paths, so install ld.lld
# into llvm/bin
if(LLD_BINARY)
  message(STATUS "Found lld: ${LLD_BINARY}")
  # ld.lld is usually a symlink but we only need it and don't care about the
  # original so just copy and rename it.
  install(PROGRAMS "${LLD_BINARY}" DESTINATION llvm/bin RENAME ld.lld)
else()
  message(FATAL_ERROR "lld binary not found")
endif()
