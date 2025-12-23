import glob
from itertools import chain
from typing import Any, Optional, Callable, Sequence

import sympy
from sympy.utilities.lambdify import lambdastr
import torch
import ctypes
from ctypes import py_object

from ...support.ir_imports import (
    Module,
    stream_d,
    Context,
    Operation,
    Location,
    InsertionPoint,
    builtin_d,
    Block,
    BlockArgument,
    Context,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    Location,
    MemRefType,
    Module,
    Operation,
    arith_d,
    builtin_d,
    stream_d,
)

# Passes
from .analysis.index_sequence_analysis import (
    set_node_indices,
    set_post_expansion_indices,
)
from .analysis.partition_strided_operators import (
    partition_gather_like_ops,
    partition_ops_with_gpr_offsets,
    partition_strided_operators,
)
from .barriers import add_shared_memory_barriers
from .cluster_barriers import add_cluster_barriers
from .construct_index_mapping import construct_index_mapping
from .debug_log_hoist import (
    debug_log_hoist,
    debug_log_write_replace,
    DebugArgInfo,
)
from .decompose_dot_mma import decompose_dot_mma
from .decompose_reduce_ops import decompose_reduce_ops
from .decompose_scan_ops import decompose_scan_ops
from .decompose_topk_ops import decompose_topk_ops
from .decompose_vmma_ops import decompose_vmma_ops
from .expansion.expansion import add_get_results, expand_graph
from .fuse_tensor_loads import fuse_tensor_loads
from .gather_to_shared import gather_to_shared, gather_to_shared_swizzling
from .generate_bound_checks import generate_bound_checks
from .generate_bounds_exprs import generate_bounds_exprs
from .global_to_shared_gathers import global_to_shared_gathers
from .hardware_transpose import mark_hardware_transpose_candidates
from .hoisting import hoist_loop_invariant_ops
from .in_thread_transpose import in_thread_transpose
from .location_check_pass import location_check_pass
from .memory_analysis.minimize_shared_allocs import minimize_shared_allocs
from .minimize_global_loads import minimize_global_loads
from .multicast import multicast
from .promotion import compute_shared_memory_usage, promote_placeholders
from .schedule_reordering import schedule_reordering
from .scheduling.schedule import schedule_graph
from .scheduling.schedule_enums import SchedulingType
from .shared_memory_indexing import apply_shared_memory_indexing_corrections
from .specialize import specialize_kernel
from .tensor_load_to_shared import tensor_load_to_shared
from .type_inference import infer_types
from .wave_schedule import WaveSchedule
from .workgroup_reordering import reorder_workgroups

# Utilities.
from .utils.compile_utils import canonicalize_module, apply_transform, compile_to_vmfb
from .utils.general_utils import (
    get_hardware_constraint,
    partial,
    remove_files_with_extension,
    remove_files_with_extension,
    wave_dtype_to_torch,
)
from .utils.graph_utils import (
    initialize_iter_args,
    remove_chained_extractslice,
    remove_chained_getresult,
)
from .utils.run_utils import (
    write_file,
    print_bench_result,
    invoke_with_wave_runtime,
    get_benchmark_flags,
)
from .utils.print_utils import print_trace, try_apply_pass

# Support.
from .._support.indexing import IndexingContext, safe_subs
from ...support.location_config import LocationCaptureLevel
from .._support.tracing import CapturedTrace


from ..compiler import host_codegen, kernel_codegen, builder, dispatch_codegen
from ..compiler.wave_codegen import WaveEmitter
from .compile_options import WaveCompileOptions
from .cache import (
    get_cache_manager,
    get_temp_binary_dir,
    is_cache_enabled,
)

from .water import water_leak_in_bounds_check, water_lowering_pipeline
from wave_lang.runtime.launch import Launchable
from wave_lang.runtime.multi_device_launch import MultiDeviceLaunchable
from wave_lang.kernel.lang.global_symbols import *
from .profiling import benchmark_module
from .debug_log_hoist import DebugArgInfo

from ..lang import IndexSymbol
from ..lang.grid import Grid
from .iree_utils import warn_iree_is_too_old

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

    def invoke(self, *args, **kwargs) -> None:
        """
        Invokes the wave kernel with the given arguments.
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


class WaveKernelExecutionEngine:
    def __init__(
        self,
        options: WaveCompileOptions,
        module: Module | bytes | str,
        mlir_asm: str,
        *,
        create_execution_engine: bool = True,
    ):
        self.options = options
        self.asm = mlir_asm

        self._engine = None
        self._module_handle = None
        self._host_func_ptr = None

        # Serialize MLIR module to text if needed
        # TODO: investigate why bytecode deserialization is not working
        if isinstance(module, (bytes, str)):
            # Assume it's already MLIR text
            optimized_mlir = module.decode() if isinstance(module, bytes) else module
        else:
            # Serialize the MLIR module to text
            optimized_mlir = str(module)

        # Get the execution engine instance and load the module
        from wave_lang.kernel.wave.execution_engine import get_execution_engine

        if not create_execution_engine:
            return
        self._engine = get_execution_engine()
        self._module_handle = self._engine.load_module_from_text(optimized_mlir)

        # Look up the host wrapper function
        func_name = self.options.func_name
        try:
            self._host_func_ptr = self._engine.lookup(self._module_handle, func_name)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to lookup function '{func_name}' in loaded module. "
                f"Make sure the module was compiled with emit_host_func. Error: {e}"
            )

        # Create ctypes function type
        # The host wrapper signature is: void func(void* stream, PyObject* arg0, PyObject* arg1, ...)

        num_kernel_args = len(self.options.kernel_usages)
        arg_types = [ctypes.c_void_p] + [
            py_object
        ] * num_kernel_args  # +1 for stream pointer
        func_type = ctypes.CFUNCTYPE(None, *arg_types)
        self._cfunc = func_type(self._host_func_ptr)

    def __call__(self, *args):
        return self.invoke(*args)

    def invoke(self, *args) -> None:
        """
        Invokes the wave kernel with the given arguments using the ExecutionEngine.
        """
        assert (
            self._engine is not None
        ), "Cannot invoke kernel without creating an execution engine. Revise the constructor call."

        # Get the current stream
        stream_ptr = torch.cuda.current_stream().cuda_stream

        # Call the JIT-compiled host wrapper function
        # Signature: void func(void* stream, PyObject* arg0, PyObject* arg1, ...)
        self._cfunc(stream_ptr, *(py_object(arg) for arg in args))


def _build_initial_pass_pipeline(
    launchable: "LaunchableWave",
    trace: CapturedTrace,
    options: WaveCompileOptions,
    debug_arg_info: list[DebugArgInfo],
    debug_handlers: list[Any],
    print_ir_before: Sequence[str] = [],
    print_ir_after: Sequence[str] = [],
):
    idxc = IndexingContext.current()

    def finalize_indices():
        idxc.finalize()

    def substitute_vector_shapes():
        launchable.hardware_constraints[0].subs_vector_shapes(idxc.subs)

    return [
        partial(debug_log_hoist, trace, debug_handlers),
        partial(initialize_iter_args, trace),
        partial(launchable.create_induction_vars, trace),
        partial(launchable.initialize_reductions, trace),
        finalize_indices,
        substitute_vector_shapes,
        partial(add_get_results, trace),
        partial(infer_types, trace, launchable.constraints),
        partial(construct_index_mapping, trace, launchable.constraints),
        partial(
            debug_log_write_replace,
            trace,
            launchable.constraints,
            options,
            debug_arg_info,
        ),
        partial(
            promote_placeholders,
            trace,
            launchable.constraints,
            options.reorder_allocs,
        ),
        partial(
            set_node_indices,
            trace,
            launchable.constraints,
            print_ir_before,
            print_ir_after,
        ),
        partial(reorder_workgroups, trace, launchable.reordering_constraints),
        partial(expand_graph, trace, launchable.constraints),
        partial(set_post_expansion_indices, trace, launchable.constraints),
        partial(remove_chained_getresult, trace),
    ]


def _rewrite_module_for_iree_stream_abi(
    module_op: Module,
    dispatch_entrypoint: dispatch_codegen.DispatchEntrypoint,
    exe: dispatch_codegen.StreamExecutable,
) -> None:
    """
    Update an existing MLIR module that has been wrapped with IREE stream executable
    to be compatible with stream bindings arguments.
    """

    with exe._loc, InsertionPoint.at_block_begin(dispatch_entrypoint.entry_block):
        target_block = dispatch_entrypoint.entry_block
        source_func_op = module_op.operation.regions[0].blocks[0].operations[0]
        source_block = source_func_op.regions[0].blocks[0]

        target_args = list(target_block.arguments)

        def convert_memref_to_stream_binding(
            target_block: Block,
            old_arg: BlockArgument,
            new_arg: BlockArgument,
            index: int,
        ) -> stream_d.BindingSubspanOp:
            """Convert a memref argument to stream.binding + subspan extraction."""
            # Create zero constant
            result_type = IndexType.get()
            zero_value = arith_d.constant(result_type, IntegerAttr.get(result_type, 0))

            # Create subspan operation
            subspan_op = stream_d.binding_subspan(
                old_arg.type,  # The original memref type
                new_arg,  # The stream.binding argument
                byte_offset=zero_value,
                # dynamic_dims=dispatch_entrypoint.get_dynamic_dims(binding),
                dynamic_dims=[],  # TODO: get dynamic dims
            )

            return subspan_op

        # Create argument mapping
        arg_mapping = {}
        for i, old_arg in enumerate(source_block.arguments):
            if i < len(target_args) and isinstance(old_arg.type, MemRefType):
                new_subspan = convert_memref_to_stream_binding(
                    target_block, old_arg, target_args[i], i
                )
                arg_mapping[old_arg] = new_subspan
            else:
                # Map scalar arguments to their corresponding arguments directly.
                arg_mapping[old_arg] = target_args[i]

        # Move operations
        ops_to_move = list(source_block)
        for op in ops_to_move:
            op.detach_from_parent()
            target_block.append(op)

        # Replace all uses of old arguments with new subspan results
        for old_arg, new_value in arg_mapping.items():
            old_arg.replace_all_uses_with(new_value)


def compile_launchable_to_mlir(
    launchable: "LaunchableWave",
    trace: CapturedTrace,
    context: Context,
    module_op: Optional[Module] = None,
    options: WaveCompileOptions = None,
):
    entrypoint_name = launchable.name
    root_graph = trace.get_root_graph()

    # pass device constraint to kernel signature
    # so that we can set the dimensions of the tensors per device
    kernel_sig = kernel_codegen.KernelSignature(launchable.device_constraints)
    kernel_sig.add_from_graph_placeholders(root_graph)
    kernel_sig.add_from_dynamic_symbols(options.dynamic_symbols)
    kernel_sig.add_grid(launchable.grid_type)
    kernel_sig.determine_input_output_buffers(root_graph)
    if options.print_signature:
        print(kernel_sig)

    mb = builder.ModuleBuilder(context=context, module_op=None)
    exe = dispatch_codegen.StreamExecutable(mb, name=entrypoint_name)
    workgroup_size = launchable.hardware_constraints[0].threads_per_block
    subgroup_size = launchable.hardware_constraints[0].threads_per_wave

    # Setup LLVM func compilation configs.
    llvm_func_config = {}
    if options.denorm_fp_math_f32:
        llvm_func_config["denormal-fp-math-f32"] = options.denorm_fp_math_f32

    if options.waves_per_eu:
        llvm_func_config["amdgpu-waves-per-eu"] = options.waves_per_eu

    dispatch_entrypoint = exe.define_entrypoint(
        entrypoint_name,
        kernel_sig,
        launchable.grid_type,
        workgroup_size,
        subgroup_size,
        options.dynamic_symbols,
        llvm_func_config,
        trace.location,
    )

    # Only emit MLIR if we don't have a module yet.
    if not module_op:
        emitter = WaveEmitter(
            dispatch_entrypoint,
            trace,
            launchable.constraints,
            options,
            launchable.grid_type.dims,
            entrypoint_name,
        )
        with mb.module_op.context, Location.unknown():
            module_op = builtin_d.ModuleOp()

        with InsertionPoint(module_op.body), Location.unknown():
            func = emitter.emit(trace.get_root_graph())
            if options.use_water_backend:
                emitter.emit_host_func(func)

    # Otherwise, we need to iree-fy the existing module (that supposedly has
    # upstream MLIR ops only) in order for it to be executable in the wave
    # pipeline.
    # `dispatch_entrypoint` already has most of the setup, we'll just need
    # to move the ops from existing module to inside `dispatch_entrypoint`.
    # Also we'll need to update the uses of the memref arguments (from the
    # existing module) to be compatible with the new stream.binding arguments.

    if options.use_water_backend:
        mb.module_op = module_op
    else:
        assert not any(
            isinstance(op, stream_d.ExecutableOp)
            for op in module_op.operation.regions[0].blocks[0]
        ), "expected overriding module to contain only upstream MLIR ops"
        _rewrite_module_for_iree_stream_abi(module_op, dispatch_entrypoint, exe)

    if options.postprocess:
        apply_transform(mb.module_op, options.postprocess, options.subs)

    if options.canonicalize:
        canonicalize_module(mb.module_op)

    return mb, trace, exe, kernel_sig, entrypoint_name


def _trace_launchable_and_get_kernel_signature(
    launchable: "LaunchableWave",
    options: WaveCompileOptions,
    schedule: Optional[WaveSchedule] = None,
    context: Optional[Context] = None,
    module_op: Optional[Operation] = None,
) -> tuple[
    builder.ModuleBuilder,
    CapturedTrace,
    dispatch_codegen.StreamExecutable,
    kernel_codegen.KernelSignature,
    str,
    WaveCompileOptions,
    Sequence[DebugArgInfo],
    Grid,
]:
    # Issue a warning if IREE ver is too low.
    # Warning will only be issued if we are compiling the kernel and won't
    # if we are using cached kernel as we don't want to add any additional
    # overhead to 'happy' path.
    warn_iree_is_too_old()

    # Build wave runtime, if specified.
    if options.wave_runtime:
        # Remove any existing hsaco files in this directory.
        # If the kernel is being cached, then it will be referenced from the
        # cache directory. When kernels are not being cached, we remove them
        # to ensure that at any time there is only one hsaco file in this directory.
        remove_files_with_extension(get_temp_binary_dir(), ".hsaco")

    print_ir_after = options.print_ir_after
    print_ir_before = options.print_ir_before
    profile_pass = options.profile_pass
    if options.print_trace_begin:
        print(f"\n***Tracing kernel {launchable.name}***")

    debug_arg_info = []
    debug_handlers = []

    trace = launchable._trace(location_capture_config=options.location_capture_config)
    if (
        "all" in print_ir_after
        or "all" in print_ir_before
        or "trace" in print_ir_after
        or "first" in print_ir_before
    ):
        print(f"***After trace/Before first pass***\n")
        print_trace(trace)

    # Initial passes, pre-optimization.
    graph_passes = _build_initial_pass_pipeline(
        launchable,
        trace,
        options,
        debug_arg_info,
        debug_handlers,
        print_ir_before,
        print_ir_after,
    )

    graph_passes += [
        partial(decompose_vmma_ops, trace, launchable.constraints),
        partial(decompose_dot_mma, trace, launchable.constraints),
    ]

    # Optimizations.
    if options.optimization_level:
        graph_passes += [
            partial(hoist_loop_invariant_ops, trace, launchable.constraints),
            partial(tensor_load_to_shared, trace, launchable.constraints, options),
            partial(multicast, trace, launchable.constraints, options),
            partial(fuse_tensor_loads, trace, launchable.constraints, options),
            partial(in_thread_transpose, trace, launchable.constraints, options),
            partial(global_to_shared_gathers, trace, launchable.constraints),
            partial(minimize_global_loads, trace, launchable.constraints),
            # Wave specialization
            partial(specialize_kernel, trace, launchable.constraints, options),
            partial(gather_to_shared, trace, launchable.constraints, options),
            partial(gather_to_shared_swizzling, trace, launchable.constraints, options),
            partial(
                mark_hardware_transpose_candidates,
                trace,
                launchable.constraints,
                options,
            ),
        ]
    graph_passes += [
        partial(
            apply_shared_memory_indexing_corrections, trace, launchable.constraints
        ),
    ]

    # Partition strided operators.
    graph_passes += [
        partial(partition_ops_with_gpr_offsets, trace, launchable.constraints),
        partial(partition_strided_operators, trace, launchable.constraints),
        partial(remove_chained_extractslice, trace),
    ]

    graph_passes += [
        partial(decompose_reduce_ops, trace, launchable.constraints),
        partial(decompose_scan_ops, trace, launchable.constraints),
        partial(decompose_topk_ops, trace, launchable.constraints),
    ]

    # Schedule the iterate ops.
    scheduling_type = options.schedule
    use_scheduling_barriers = options.use_scheduling_barriers
    if options.schedule == SchedulingType.MANUAL:
        graph_passes.append(
            partial(
                launchable.run_manual_schedule,
                trace,
                launchable.constraints,
                schedule,
                use_scheduling_barriers,
            ),
        )
    else:
        graph_passes.append(
            partial(
                schedule_graph,
                trace,
                launchable.constraints,
                use_scheduling_barriers,
                scheduling_type,
                options.override_schedule,
                options.dump_schedule,
                options.multi_buffer_count,
            )
        )

    if options.optimization_level:
        graph_passes += [
            partial(
                schedule_reordering,
                trace,
                launchable.constraints,
                scheduling_type,
                options.use_global_to_shared,
            ),
            partial(
                minimize_shared_allocs,
                trace,
                options.minimize_shared_allocs,
            ),
        ]
    graph_passes += [
        partial(
            add_shared_memory_barriers,
            trace,
            target=options.target,
            is_specialized=options.specialize,
        ),
        partial(add_cluster_barriers, trace, launchable.constraints, options),
        partial(compute_shared_memory_usage, trace, options.kernel_launch_info),
        partial(
            partition_gather_like_ops, trace, launchable.constraints, options.target
        ),
        partial(generate_bounds_exprs, trace, launchable.constraints),
    ]

    if options.use_bound_check:
        graph_passes += [
            partial(generate_bound_checks, trace),
        ]

    graph_passes.append(
        partial(
            location_check_pass,
            trace,
            "enforce-locations",
            log=False,
            enforce_locations=options.enforce_locations,
        )
    )

    pass_times = {}
    for p in graph_passes:
        try_apply_pass(
            p, trace, print_ir_before, print_ir_after, profile_pass, pass_times
        )

    if options.print_pass_times:
        pass_times_list = sorted(pass_times.items(), key=lambda x: x[1], reverse=True)

        print(f"Pass times:")
        for k, v in pass_times_list:
            print(f"    {k}: {v:.4f}s")

    if "all" in print_ir_after or "last" in print_ir_after:
        # Take advantage of Python leaking loop variables
        print(f"***After final pass {p.__name__}***\n")
        print_trace(trace)

    # Determine grid shape.
    launchable.infer_grid_shape(IndexingContext.current())
    launchable.infer_device_layout(IndexingContext.current())

    if options.print_grid:
        print(f"Grid Dimensions: {launchable.grid_type.dims}")
        print(f"Device layout: {launchable.device_layout}")

    # Add grid and block dims to kernel launch info.
    # Convert the grid into a lambda that we can use to compute the grid dimension.
    hw_constraint = get_hardware_constraint(launchable.constraints)
    grid_symbols = list(launchable.bound_scalar_symbols.keys()) + list(
        options.dynamic_symbols
    )
    options.kernel_launch_info.grid = sympy.lambdify(
        [grid_symbols], launchable.grid_type.dims
    )
    options.kernel_launch_info.grid_str = lambdastr(
        [grid_symbols], launchable.grid_type.dims
    )
    options.kernel_launch_info.blocks = [
        int(x) for x in hw_constraint.threads_per_block
    ]
    options.kernel_launch_info.cluster_dims = (
        [int(x) for x in hw_constraint.workgroups_per_cluster]
        if hw_constraint.workgroups_per_cluster
        else [0, 0, 0]
    )
    options.kernel_launch_info.func_name = launchable.name

    idxc = IndexingContext.current()
    for sym, val in zip(
        [THREAD_0, THREAD_1, THREAD_2, WORKGROUP_0, WORKGROUP_1, WORKGROUP_2],
        chain(hw_constraint.threads_per_block, launchable.grid_type.dims),
    ):
        if safe_subs(val, idxc.subs) == 1:
            idxc.bind_constant(sym, 0)

    return (
        *compile_launchable_to_mlir(
            launchable, trace, context, module_op, options=options
        ),
        options,
        debug_arg_info,
        debug_handlers,
        launchable.device_layout,
    )


def wave_compile(
    options: WaveCompileOptions,
    kernel: "LaunchableWave",
    schedule: Optional["WaveSchedule"] = None,
) -> WaveKernel | WaveKernelExecutionEngine:
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
        ) = _trace_launchable_and_get_kernel_signature(kernel, options, schedule)

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
                ) = compile_launchable_to_mlir(
                    launchable=kernel,
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

        if not options.use_water_backend:
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

        kernel_usages = [
            binding.kernel_buffer_type.usage
            for binding in kernel_sig.kernel_buffer_bindings
        ]
        options.kernel_usages = kernel_usages

        if options.compile_to_asm or options.backend == "asm":
            # ASM flow: generate AMDGCN assembly; optionally build a binary
            asm = _generate_asm_code(mb, options)
            if options.backend == "asm" and not options.compile_to_asm:
                _compile_asm_to_binary(asm, options)
        elif options.use_water_backend:
            module = water_lowering_pipeline(mb.module_op, options)
            return WaveKernelExecutionEngine(
                options,
                module,
                asm,
                create_execution_engine=not options.compile_to_mlir,
            )
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
