import glob
from itertools import chain
from typing import Any, Optional, Callable, Sequence

import torch

from wave_lang.kernel.lang import IndexSymbol
from wave_lang.support.ir_imports import Module, stream_d

from .._support.indexing import IndexingContext
from ...support.location_config import LocationCaptureLevel
from ..compiler import host_codegen, kernel_codegen
from .cache import (
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
from wave_lang.runtime.multi_device_launch import MultiDeviceLaunchable
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
        compiled_graph: Optional["CapturedTrace"] = None,
        debug_outputs: Optional[Sequence[DebugArgInfo]] = None,
        debug_handlers: Optional[Sequence[Any]] = None,
    ):
        self.options = options
        self.executable = executable
        self.asm = asm
        self.trace = trace
        self.compiled_graph = compiled_graph
        self.gpu_binary_path = gpu_binary_path
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

        # If we are compiling to MLIR, we don't have an executable
        if options.compile_to_mlir:
            return

        # If we are using the wave runtime, we don't need to create a launchable
        if options.wave_runtime:
            return

        # Disable async dispatch for benchmarking.
        is_async = options.iree_launch_async and not options.run_bench

        # 'launchable' decides if function is async or not based on name.
        self.func_name = options.func_name + ("$async" if is_async else "")

        if self.options.num_devices > 1:
            device_list = [f"hip://{d}" for d in range(options.num_devices)]
            self.launchable = MultiDeviceLaunchable.from_vmfb(
                self.executable,
                devices=device_list,
                function_name=self.func_name,
                runtime_flags=["--hip_use_streams=true"],
            )
        else:

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

    def get_compiled_graph(self) -> Optional["CapturedTrace"]:
        """Returns the graph produced after running all compilation passes.
        Currently only needed when compile to MLIR.
        """
        return self.compiled_graph

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

        debug_args = []
        debug_logs = kwargs.get("debug_logs", {})
        debug_extra_dimensions = {}

        def get_dynamic_dimension_actual(sym):
            if sym in debug_extra_dimensions:
                return debug_extra_dimensions[sym]
            arg_idx, dim = self.symbols_args_map[sym]
            return args[arg_idx].shape[dim]

        # If there are debug_log uses with extra iteration dimensions, we need
        # to collect their sizes so that we can allocate the appropriate Torch
        # tensors.
        if self.debug_outputs:
            for info_dict in self.debug_outputs:
                extra_iter_dims = info_dict.get("extra_iteration_dimensions", None)
                if extra_iter_dims:
                    for dim_symbol, _, size in extra_iter_dims:
                        debug_extra_dimensions[dim_symbol] = size

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
                    "iteration_dimensions": [
                        dim
                        for dim, _, _ in info_dict.get("extra_iteration_dimensions", [])
                    ],
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
            self.launchable(*tensors, *scalar_args)

            if self.options.run_bench:
                benchmark_flags = get_benchmark_flags(self.options)
                benchmark_results = benchmark_module(
                    self.options,
                    [t.data for t in kernel_inputs],
                    [t.data for t in kernel_outputs],
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


def wave_compile(
    options: WaveCompileOptions,
    kernel: "LaunchableWave",
    schedule: Optional["WaveSchedule"] = None,
) -> WaveKernel:
    """
    Compiles the wave kernel to an executable.
    """
    validate_options(options)

    cls = WaveKernelWithProfile if options.profile_python_wrapper else WaveKernel

    # Check if this kernel has been compiled before, if the cache is enabled.
    cache_manager = None
    binary_path = None

    def get_binary_path():
        if is_cache_enabled() and cache_manager is not None:
            # For cached kernels, return the cache path directly
            # Use cache_manager.base_dir instead of get_cache_base_dir() because:
            # - get_cache_base_dir() always returns the default cache directory (~/.wave)
            # - cache_manager.base_dir respects reset_cache_manager() calls in tests
            # - This ensures tests use their temporary cache directory instead of the global one
            return (
                str(cache_manager.base_dir / options.kernel_hash / options.kernel_hash)
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
        # Do not lookup cached kernel if user wants to run overriding MLIR.
        if is_cache_enabled() and not options.override_mlir:
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
            device_layout,
        ) = kernel._trace_and_get_kernel_signature(options, schedule)

        ireefy_overriding_module = False
        if options.override_mlir:
            overriding_module_op = Module.parse(
                options.override_mlir, context=mb.module_op.context
            )
            # If there is no stream.executable op at the top-level of the
            # module, we need to iree-fy the module by invoking
            # `compile_to_mlir()` with our existing module.
            if not any(
                isinstance(op, stream_d.ExecutableOp)
                for op in overriding_module_op.operation.regions[0].blocks[0]
            ):
                (
                    mb,
                    _,
                    exe,
                    kernel_sig,
                    entrypoint_name,
                ) = kernel.compile_to_mlir(
                    trace=graph,
                    context=None,
                    module_op=overriding_module_op,
                    options=options,
                )
                ireefy_overriding_module = True

        options.kernel_sig = kernel_sig

        # calculate the number of devices based on the device layout
        for dim in device_layout.dims:
            options.num_devices *= dim

        # Get the trace from the kernel. Since the trace contains complex objects
        # that are not easily serializable, we don't cache the trace. So this trace
        # is not available for cached kernels. The primary use case for the trace is
        # is for tuning where each kernel is different from the others and so we
        # don't want to cache the kernel in that case.
        trace = kernel._trace()

        if not options.use_water_pipeline:
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
                device_layout=device_layout,
                device_constraints=kernel.device_constraints,
            )
        mb.module_op.verify()
        asm = mb.module_op.get_asm(
            enable_debug_info=options.location_capture_config.level
            != LocationCaptureLevel.NONE
            and not options.drop_debug_info_before_mlir,
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

        # Only override here if the overriding module has not been iree-fied by
        # `compile_to_mlir()`. Otherwise the asm would have been available in
        # `mb.module_op.get_asm()`.
        if options.override_mlir and not ireefy_overriding_module:
            asm = options.override_mlir

        # Handle ASM and LLVM backends in a clear, single-pass flow
        compiled_wave_vmfb = None

        if options.compile_to_asm or options.backend == "asm":
            # ASM flow: generate AMDGCN assembly; optionally build a binary
            asm = _generate_asm_code(mb, options)
            if options.backend == "asm" and not options.compile_to_asm:
                _compile_asm_to_binary(asm, options)
        elif not options.compile_to_mlir:
            # LLVM flow: only compile to VMFB when not in MLIR-only mode
            compiled_wave_vmfb = compile_to_vmfb(asm, options)

        # Early return for MLIR-only flows (no VMFB, no ASM binary)
        if options.compile_to_mlir or options.compile_to_asm:
            return cls(
                options,
                None,
                asm,
                None,
                bound_scalar_symbols,
                symbols_args_map,
                trace,
                graph,
                debug_arg_info,
                debug_handlers,
            )
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

    # Set binary path based on backend
    if options.wave_runtime and not options.compile_to_asm:
        # Both ASM and LLVM backends should use get_binary_path() for consistency
        # Skip binary loading for compile_to_asm
        binary_path = get_binary_path()
    else:
        binary_path = None

    return cls(
        options,
        compiled_wave_vmfb,
        asm,
        binary_path,
        bound_scalar_symbols,
        symbols_args_map,
        trace,
        graph,
        debug_arg_info,
        debug_handlers,
    )


def _generate_asm_code(mb, options):
    """Generate AMDGCN assembly from MLIR module."""
    from .asm.asm_emitter import AsmEmitter

    # Convert module_op to MLIR string
    mlir_asm = mb.module_op.get_asm(
        enable_debug_info=options.location_capture_config.level
        != LocationCaptureLevel.NONE
        and not options.drop_debug_info_before_mlir,
        use_local_scope=options.use_local_scope,
    )

    # Generate AMDGCN assembly directly from MLIR string
    return AsmEmitter.from_mlir_string(
        mlir_asm, targetid=options.target, codeobj=options.codeobj
    )


def _compile_asm_to_binary(asm_code, options):
    """Compile AMDGCN assembly to binary using amdclang++."""
    import tempfile
    import os
    import subprocess

    # Create temporary file for assembly output
    with tempfile.NamedTemporaryFile(mode="w", suffix=".s", delete=False) as asm_file:
        asm_file.write(asm_code)
        asm_output = asm_file.name

    try:
        # Generate code object using amdclang++
        kernel_name = options.func_name
        obj_file = os.path.join(get_temp_binary_dir(), f"{kernel_name}.o")
        hsaco_file = os.path.join(get_temp_binary_dir(), f"{kernel_name}.hsaco")

        # Step 1: Compile assembly to object file
        compile_cmd = [
            "amdclang++",
            "-x",
            "assembler",
            "-target",
            "amdgcn-amd-amdhsa",
            f"-mcode-object-version={options.codeobj}",
            f"-mcpu={options.target}",
            "-mwavefrontsize64",
            "-c",
            asm_output,
            "-o",
            obj_file,
        ]

        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Assembly compilation failed: {result.stderr}")

        # Step 2: Link object file to hsaco file
        link_cmd = [
            "amdclang++",
            "-target",
            "amdgcn-amd-amdhsa",
            "-Xlinker",
            "--build-id=sha1",
            "-o",
            hsaco_file,
            obj_file,
        ]

        result = subprocess.run(link_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Hsaco linking failed: {result.stderr}")

    finally:
        # Clean up temporary files
        try:
            os.unlink(asm_output)
        except OSError:
            pass


def validate_options(options: WaveCompileOptions):
    if options.wave_runtime and options.run_bench:
        raise ValueError("Banchmarking is not supported in wave_runtime yet")

    if options.backend not in ["llvm", "asm"]:
        raise ValueError(
            f"Invalid backend '{options.backend}'. Must be 'llvm' or 'asm'"
        )

    if options.backend == "asm" and not options.wave_runtime:
        raise ValueError("ASM backend requires wave_runtime=True")
