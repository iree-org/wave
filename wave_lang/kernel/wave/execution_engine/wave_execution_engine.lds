# Linker version script for wave_execution_engine
# Hide all symbols except the Python module entry point
{
  global:
    # Python module initialization function
    PyInit_wave_execution_engine;
  local:
    # Hide all other symbols
    *;
};
