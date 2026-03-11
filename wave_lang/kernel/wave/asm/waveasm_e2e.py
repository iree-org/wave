# Backward-compatible re-export. The module has moved to waveasm/waveasm_e2e.py.
from waveasm.waveasm_e2e import *  # noqa: F401,F403
from waveasm.waveasm_e2e import (  # noqa: F401
    WaveASMCompiler,
    capture_wave_kernel_info,
    capture_wave_mlir,
    run_with_wave_runtime,
)
