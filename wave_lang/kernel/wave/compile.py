import glob
from itertools import chain
from typing import Any, Optional, Callable, Sequence

import torch

from wave_lang.kernel.lang import IndexSymbol

from .._support.indexing import IndexingContext
from .._support.location_config import LocationCaptureLevel
from ..compiler import host_codegen, kernel_codegen
from .cache import (
    get_cache_base_dir,
    get_cache_manager,
    get_temp_binary_dir,
    is_cache_enabled,
)
from .compile_options import WaveCompileOptions
from .utils.compile_utils import compile_to_vmfb
from .utils.general_utils import wave_dtype_to_torch
from .utils.run_utils import (
    write_file,
    print_bench_result,
    invoke_with_wave_runtime,
    get_benchmark_flags,
)
from .water import water_leak_in_bounds_check
from wave_lang.runtime.launch import Launchable
from .profiling import benchmark_module
from .debug_log_hoist import DebugArgInfo
import iree.runtime as rt


class WaveKernel:
    """
    Represents a wave kernel that can be invoked by the user.
    """

    def __init__(
        self,
        options: WaveCompileOptions,
        executable: Any,
        asm: str,
        gpu_binary_path: Optional[str],
        bound_scalar_symbols: dict[IndexSymbol, int],
        symbols_args_map: dict[IndexSymbol, tuple[int, int]],
        trace: Optional["CapturedTrace"] = None,
        debug_outputs: Optional[Sequence[DebugArgInfo]] = None,
        debug_handlers: Optional[Sequence[Any]] = None,
    ):
        self.options = options
        self.executable = executable
        self.asm = asm
        self.trace = trace
        if gpu_binary_path:
            import wave_runtime

            # It is safe to call this multiple times.
            wave_runtime.load_hip_functions()

            self.gpu_binary, self.gpu_func = wave_runtime.load_binary(
                gpu_binary_path, options.kernel_launch_info.func_name
            )
        else:
            self.gpu_func = None
        self.bound_scalar_symbols = bound_scalar_symbols
        self.symbols_args_map = symbols_args_map
        self.debug_outputs = debug_outputs
        self.debug_handlers = debug_handlers

        if not options.wave_runtime:
            # Disable async dispatch for benchmarking.
            is_async = options.iree_launch_async and not options.run_bench

            # 'launchable' decides if function is async or not based on name.
            self.func_name = options.func_name + ("$async" if is_async else "")

            def loader(device):
                vm_instance = device.vm_instance
                return rt.VmModule.copy_buffer(vm_instance, self.executable)

            self.launchable = Launchable.from_vm_module(
                loader,
                entry_point=self.func_name,
            )

    def get_trace(self) -> Optional["CapturedTrace"]:
        """Returns the trace used to generate this kernel.
        If this is a cached kernel, the trace is not available.
        """
        return self.trace

    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        """
        Invokes the wave kernel with the given arguments.
        Returns the assembly code of the compiled kernel.
        """

        # Segregate args into kernel tensor and scalars.
        scalar_args = []
        kernel_inputs, kernel_outputs = [], []

        # Partition arguments into kernel inputs and outputs.
        # ToDo: we should expose the `usage` as a property in binding desc
        #       so that we can reduce the code and use `zip``.
        usage_idx = 0
        for arg in args:
            if not isinstance(arg, torch.Tensor):
                scalar_args.append(arg)
                continue
            usage = self.options.kernel_usages[usage_idx]
            usage_idx += 1
            if usage == kernel_codegen.KernelBufferUsage.INPUT:
                kernel_inputs.append(arg)
            if usage == kernel_codegen.KernelBufferUsage.OUTPUT:
                kernel_outputs.append(arg)

        def get_dynamic_dimension_actual(sym):
            arg_idx, dim = self.symbols_args_map[sym]
            return args[arg_idx].shape[dim]

        debug_args = []
        debug_logs = kwargs.get("debug_logs", {})
        if self.debug_outputs:
            for info_dict in self.debug_outputs:
                shape = [
                    self.options.subs.get(symdim, None)
                    or get_dynamic_dimension_actual(symdim)
                    for symdim in info_dict["symbolic_shape"]
                ]
                memory = torch.zeros(
                    shape, dtype=wave_dtype_to_torch(info_dict["dtype"]), device="cuda"
                )
                log_info = {
                    "value": memory,
                    "symbolic_shape": info_dict["symbolic_shape"],
                }
                debug_args.append(memory)
                debug_logs[info_dict["symbol_name"]] = log_info
        kernel_outputs = kernel_outputs + debug_args

        dynamic_symbols = []
        for sym in self.options.dynamic_symbols:
            dynamic_symbols.append(get_dynamic_dimension_actual(sym))

        if self.options.wave_runtime:
            invoke_with_wave_runtime(
                self.options,
                kernel_inputs,
                kernel_outputs,
                scalar_args,
                self.bound_scalar_symbols,
                dynamic_symbols,
                self.gpu_func,
            )
        else:
            tensors = [t.data for t in chain(kernel_inputs, kernel_outputs)]
            self.launchable(*tensors, *scalar_args, *dynamic_symbols)

            if self.options.run_bench:
                benchmark_flags = get_benchmark_flags(self.options)
                benchmark_results = benchmark_module(
                    self.options,
                    [t.data for t in kernel_inputs],
                    [t.data for t in kernel_outputs],
                    dynamic_symbols,
                    self.executable,
                    self.func_name,
                    **benchmark_flags,
                )
                print_bench_result(
                    benchmark_results, self.options.benchmark_results_file
                )

        if self.debug_outputs:
            for info_dict, (label, debug_log) in zip(
                self.debug_outputs, debug_logs.items()
            ):
                if printer := info_dict.get("printer", None):
                    printer(label, debug_log["value"])
            for handler in self.debug_handlers or []:
                handler(debug_logs)

        return self.asm


def invoke_with_profile(options: WaveCompileOptions, invoke: Callable, *args, **kwargs):

    # Warmup
    for _ in range(options.profile_python_warmup):
        invoke(*args, **kwargs)

    repetitions = options.profile_python_repetitions
    if options.profile_python_cprofile:
        import cProfile

        with cProfile.Profile() as pr:
            for _ in range(repetitions):
                res = invoke(*args, **kwargs)

        pr.print_stats(sort="cumulative")
        return res
    else:
        import timeit

        time = timeit.timeit(
            lambda: invoke(*args, **kwargs),
            number=repetitions,
        )
        print(f"Time: {time:.3f}s, {time / repetitions:.6f}s per iteration")
        return invoke(*args, **kwargs)


class WaveKernelWithProfile(WaveKernel):

    def __call__(self, *args, **kwargs):
        return invoke_with_profile(self.options, self.invoke, *args, **kwargs)


def wave_compile(options: WaveCompileOptions, kernel: "LaunchableWave") -> WaveKernel:
    """
    Compiles the wave kernel to an executable.
    """
    validate_options(options)

    cls = WaveKernelWithProfile if options.profile_python_wrapper else WaveKernel

    # Check if this kernel has been compiled before, if the cache is enabled.
    cache_manager = None
    binary_path = None

    def get_binary_path():
        if is_cache_enabled():
            return (
                str(get_cache_base_dir() / options.kernel_hash / options.kernel_hash)
                + ".hsaco"
            )
        else:
            return glob.glob(str(get_temp_binary_dir() / "*.hsaco"))[0]

    # Create an indexing context and populate substitutions.
    with IndexingContext() as idxc:
        idxc.set_subs(options.subs)

        # Since constraints are used to lookup the compiled kernel in the cache,
        # we initialize/update the constraints _before_ the cache lookup.
        kernel.initialize_wave_constraints()
        kernel.initialize_symbolic_constraints()
        kernel.initialize_workgroup_constraints()

        bound_scalar_symbols = kernel.bound_scalar_symbols
        symbols_args_map = kernel.symbols_args_map
        if is_cache_enabled():
            cache_manager = get_cache_manager()
            options.kernel_hash = cache_manager.get_hash(
                kernel.constraints,
                kernel._f,
                options,
            )
            cached_kernel = cache_manager.load_kernel(options.kernel_hash)
            if cached_kernel:
                options.kernel_usages = cached_kernel.kernel_sig
                options.kernel_launch_info = cached_kernel.kernel_launch_info
                if options.wave_runtime:
                    binary_path = get_binary_path()

                if options.print_mlir:
                    print(cached_kernel.asm)

                return cls(
                    options,
                    cached_kernel.vmfb,
                    cached_kernel.asm,
                    binary_path,
                    bound_scalar_symbols,
                    symbols_args_map,
                    None,
                    None,
                )

        # For the wave runtime, we need the hsaco binary. So we turn on
        # dumping of binaries and store in wave runtime directory. If we
        # are caching, this will be moved to the appropriate directory.
        if options.wave_runtime:
            options.dump_binaries = get_temp_binary_dir()

        # Recompile kernel from scratch if not found in cache.
        (
            mb,
            graph,
            exe,
            kernel_sig,
            entrypoint_name,
            options,
            debug_arg_info,
            debug_handlers,
        ) = kernel._trace_and_get_kernel_signature(options)
        options.kernel_sig = kernel_sig

        # Get the trace from the kernel. Since the trace contains complex objects
        # that are not easily serializable, we don't cache the trace. So this trace
        # is not available for cached kernels. The primary use case for the trace is
        # is for tuning where each kernel is different from the others and so we
        # don't want to cache the kernel in that case.
        trace = kernel._trace()

        # Disable async dispatch for benchmarking.
        is_async = options.iree_launch_async and not options.run_bench
        host_codegen.isolated_test_call(
            mb,
            exe,
            kernel_sig,
            entrypoint_name,
            options.func_name,
            options.dynamic_symbols,
            location_capture_config=options.location_capture_config,
            async_dispatch=is_async,
        )
        asm = mb.module_op.get_asm(
            enable_debug_info=options.location_capture_config.level
            != LocationCaptureLevel.NONE,
            use_local_scope=options.use_local_scope,
        )

        if options.print_mlir:
            if options.print_mlir_file:
                write_file(options.print_mlir_file, "w", asm)
            else:
                print(asm)

        if options.use_water_leak_check:
            water_leak_in_bounds_check(
                mb.module_op,
                (
                    options.use_water_leak_check
                    if isinstance(options.use_water_leak_check, str)
                    else ""
                ),
            )

        if options.override_mlir:
            asm = options.override_mlir

        if options.compile_to_mlir:
            return cls(
                options,
                None,
                asm,
                None,
                bound_scalar_symbols,
                symbols_args_map,
                debug_arg_info,
                debug_handlers,
            )

        compiled_wave_vmfb = compile_to_vmfb(asm, options)
        if options.create_vmfb_file:
            write_file(options.create_vmfb_file, "wb", compiled_wave_vmfb)

        kernel_usages = [
            binding.kernel_buffer_type.usage
            for binding in kernel_sig.kernel_buffer_bindings
        ]
        options.kernel_usages = kernel_usages

        if is_cache_enabled() and not debug_arg_info:
            cache_manager.store_kernel(
                compiled_wave_vmfb,
                asm,
                options,
            )

    if options.wave_runtime:
        binary_path = get_binary_path()

    return cls(
        options,
        compiled_wave_vmfb,
        asm,
        binary_path,
        bound_scalar_symbols,
        symbols_args_map,
        trace,
        debug_arg_info,
        debug_handlers,
    )


def validate_options(options: WaveCompileOptions):
    if options.wave_runtime and options.run_bench:
        raise ValueError("Banchmarking is not supported in wave_runtime yet")
