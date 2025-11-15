# Linker version script for wave_execution_engine
# Hide all symbols except the Python module entry point and runtime helpers
{
  global:
    # Python module initialization function
    PyInit_wave_execution_engine;
    # Runtime helper functions for JIT-compiled code
    wave_get_buffer;
  local:
    # Hide all other symbols
    *;
};
