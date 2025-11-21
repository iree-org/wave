from dataclasses import dataclass, field
from typing import Any, Optional

from .._support.indexing import IndexSymbol
from ...support.location_config import LocationCaptureConfig
from ..compiler.kernel_codegen import KernelBufferUsage
from .scheduling.schedule_enums import SchedulingType
from .utils.classes import KernelLaunchInfo


def _get_location_capture_config():
    """Wrapper to avoid circular import with debugging module."""
    from ...support.debugging import get_location_capture_config

    return get_location_capture_config()


@dataclass
class WaveCompileOptions:
    """
    Options for compiling the wave kernel.
    """

    # === General options ===
    postprocess: Optional[str] = None
    canonicalize: bool = True
    func_name: str = "isolated_benchmark"

    # === Symbol mappings ===
    subs: dict[str | IndexSymbol, Any] = field(default_factory=list)
    dynamic_symbols: list[str] = field(default_factory=list)

    # === Scheduling options ===
    schedule: bool = SchedulingType.NONE
    use_scheduling_barriers: bool = False
    # None for buffer count to be computed automatically
    multi_buffer_count: Optional[int] = None

    # === Runtime options ===
    kernel_launch_info: KernelLaunchInfo = field(default_factory=KernelLaunchInfo)
    kernel_usages: tuple[KernelBufferUsage] = None

    # === Backend options ===
    backend: str = "llvm"  # "llvm" or "asm"
    device: str = "hip"
    target: str = "gfx942"
    codeobj: str = "5"  # Code object version ("4" or "5")
    iree_preprocessing_pass_pipeline: str = None
    num_devices: int = 1
    use_water_pipeline: bool = False  # Use Water lowering pipeline with host wrapper

    # === Benchmark options ===
    run_bench: bool = False
    benchmark_batch_size: int = None
    benchmark_repetitions: int = None
    benchmark_results_file: str = None
    capture_trace: bool = False
    bench_with_constant_weights: bool = False
    profile_python_wrapper: bool = False
    profile_python_cprofile: bool = True  # If False, use timeit.timeit()
    profile_python_warmup: int = 1
    profile_python_repetitions: int = 1000

    # === Cache options ===
    kernel_hash: str = None

    # === Debug options ===
    create_vmfb_file: str = None
    override_mlir: str = None
    dump_binaries: str = None
    dump_intermediates: str = False
    compile_to_mlir: bool = False
    location_capture_config: LocationCaptureConfig = field(
        default_factory=_get_location_capture_config
    )
    use_local_scope: bool = False
    use_water_leak_check: bool | str = False  # If string, check the given IR instead.
    enforce_locations: bool = True
    drop_debug_info_before_mlir: bool = True

    # === Performance options ===
    optimization_level: bool = True
    denorm_fp_math_f32: str = None
    waves_per_eu: int = None
    wave_runtime: bool = False
    iree_launch_async: bool = True
    use_buffer_ops: bool = False
    use_fast_math: bool = False
    use_global_to_shared: bool = False
    linearize_shared_access: bool = False
    scalarize_packed_math: bool = False

    # === Compiler options ===
    minimize_shared_allocs: bool = True
    reorder_allocs: bool = True
    override_schedule: Optional[str] = None
    dump_schedule: Optional[str] = None
    use_bound_check: bool = False
    specialize: bool = False

    # Cluster barrier signal/wait delay in number of loop iterations
    # None - no barriers inside the loop
    # 0 - signal and wait on same iteration
    # 1 - one iteration apart, 2 - two, etc
    cluster_barrier_delay: Optional[int] = None

    # === Print options ===
    mlir_print_ir_after_all: bool = False
    print_ir_after: list[str] = field(default_factory=list)
    print_ir_before: list[str] = field(default_factory=list)
    profile_pass: list[str] = field(default_factory=list)
    print_trace_begin: bool = False
    print_grid: bool = False
    print_signature: bool = False
    print_mlir: bool = False
    print_mlir_file: Optional[str] = None
    print_pass_times: bool = False

    # === ASM backend options ===
    compile_to_asm: bool = (
        False  # Compile to AMDGCN assembly (for lit tests, no amdclang++)
    )
