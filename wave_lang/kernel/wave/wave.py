# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Lang, compiler, ops, constraints
import inspect
import logging

# Others
from typing import Any, Callable, Optional, get_type_hints

import torch.fx as fx

from .wave_schedule import WaveSchedule

from .._support.indexing import IndexExpr, IndexingContext, index_symbol
from ...support.location_config import LocationCaptureConfig, LocationCaptureLevel
from .._support.location import capture_function_location
from .._support.tracing import (
    CapturedTrace,
    CompiledContext,
    KernelRegionGraph,
    Launchable,
)
from ..lang import Grid, Memory, SymbolBind
from ..lang.global_symbols import *
from ..ops import wave_ops
from ..ops.wave_ops import CustomOp, Iterate, get_custom, Placeholder, Output

from .constraints import (
    Constraint,
    GridConstraint,
    HardwareConstraint,
    ReorderingConstraint,
    TilingConstraint,
    WaveConstraint,
    WorkgroupConstraint,
    DeviceConstraint,
    get_grid_shape,
    get_device_layout,
)

from .symbolic_constraints import SymbolicAlias
from .utils.general_utils import (
    delinearize_index,
)

# Utils
from .utils.symbol_utils import safe_subs, subs_idxc, get_induction_symbol

logger = logging.getLogger(__name__)


__all__ = ["wave", "wave_trace_only"]


def wave(constraints: Optional[list[Constraint]] = None):
    def decorator(f: Callable[..., Any]) -> "LaunchableWave":
        return LaunchableWave(constraints, f.__name__, f)

    return decorator


def wave_trace_only(
    constraints: Optional[list[Constraint]] = None,
    *,
    location_capture_config: Optional[LocationCaptureConfig] = None,
):
    def decorator(f: Callable[..., Any]) -> "Callable[[], CapturedTrace]":
        wave = LaunchableWave(constraints, f.__name__, f)
        return lambda: wave._trace(location_capture_config=location_capture_config)  # type: ignore

    return decorator


def _is_symbol_bind(a: Any) -> bool:
    return inspect.isclass(a) and issubclass(a, SymbolBind)


def _is_memory_arg(a: Any) -> bool:
    return inspect.isclass(a) and issubclass(a, Memory)


def add_placeholder_locations(
    trace: "CapturedTrace", kernel_func: Callable
) -> "CapturedTrace":
    """
    Add location info to placeholder nodes.
    The location used is the kernel function definition location.
    Also adds the location to the output node of the graph, which is maybe
    unnecessary, but it's convenient to have a location for the output as well
    so that 100% of nodes have locations at the beginning of the pipeline.

    Args:
        trace: The captured trace to analyze
        kernel_func:
            The original kernel function being traced -- IE the function
            definition with the @tkw.wave annotation.

    Returns:
        The modified trace with location information added
    """

    root_graph = trace.get_root_graph()
    region_graph = getattr(trace, "region_graph", None)
    location_capture_config = getattr(region_graph, "location_capture_config", None)

    if (
        location_capture_config is None
        or location_capture_config.level == LocationCaptureLevel.NONE
    ):
        return trace

    kernel_location = capture_function_location(kernel_func, location_capture_config)
    if kernel_location is None:
        return trace

    for node in root_graph.nodes:
        custom_op = get_custom(node)
        if hasattr(custom_op, "location") and custom_op.location is not None:
            continue

        if isinstance(custom_op, (Placeholder, Output)):
            custom_op.fx_node.location = kernel_location

    return trace


class LaunchableWave(Launchable):
    def __init__(
        self,
        constraints: Optional[list[Constraint]],
        name: str,
        eager_function: Callable[[Any], Any],
    ):
        super().__init__(eager_function)

        self.constraints = constraints if constraints else []
        self.induction_vars: dict[CustomOp, IndexExpr] = {}
        self._name = name
        self._f = eager_function
        self._sig = inspect.signature(eager_function)
        self.grid_type = Grid[
            tuple(get_grid_shape(self.workgroup_constraints, self.device_constraints))
        ]
        self.device_layout = Grid[tuple(get_device_layout(self.device_constraints))]

        # TODO: needed for the wave_runtime grid calculations, we should really
        # just generate host wrapper suitable for wave_runtime instead of doing
        # it in python (and it will be faster as well).
        hints = get_type_hints(eager_function)
        self.bound_scalar_symbols = {
            index_symbol(name): i
            for i, (name, arg) in enumerate(hints.items())
            if _is_symbol_bind(arg)
        }

        # Build a mapping between symbol and tensor arg (index, dim) so we can
        # use it to extract dynamic symbols from the tensor args.
        symbols_args_map = {}
        for arg_idx, arg in enumerate(hints.values()):
            if not _is_memory_arg(arg):
                continue

            for dim, symbol in enumerate(arg.symbolic_shape):
                if symbol in symbols_args_map:
                    continue

                symbols_args_map[symbol] = (arg_idx, dim)
        self.symbols_args_map = symbols_args_map

    @property
    def device_constraints(self) -> list[DeviceConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, DeviceConstraint)
        ]

    @property
    def workgroup_constraints(self) -> list[WorkgroupConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, WorkgroupConstraint)
        ]

    @property
    def tiling_constraints(self) -> list[TilingConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, TilingConstraint)
        ]

    @property
    def wave_constraints(self) -> list[WaveConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, WaveConstraint)
        ]

    @property
    def hardware_constraints(self) -> list[HardwareConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, HardwareConstraint)
        ]

    @property
    def reordering_constraints(self) -> list[ReorderingConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, ReorderingConstraint)
        ]

    @property
    def symbolic_constraints(self) -> list[HardwareConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, SymbolicAlias)
        ]

    def _validate_constraints(self):
        wave_map = {
            constraint.dim: subs_idxc(constraint.tile_size)
            for constraint in self.wave_constraints
        }

        workgroup_map = {
            constraint.dim: subs_idxc(constraint.tile_size)
            for constraint in self.workgroup_constraints
        }

        for dim in set(wave_map.keys()) | set(workgroup_map.keys()):
            wave_size = wave_map[dim] if dim in wave_map else None
            workgroup_size = workgroup_map[dim] if dim in workgroup_map else None

            assert (
                workgroup_size is not None
            ), f"expected non-empty tile size in `WorkgroupConstraint` for dimension {dim}"

            if wave_size is None:
                continue

            assert (
                wave_size > 0
            ), f"expected non-zero tile in `WaveConstraint` for dimension {dim}"

            assert (
                workgroup_size > 0
            ), f"expected non-zero tile in `WorkgroupConstraint` for dimension {dim}"

            assert (
                workgroup_size >= wave_size
            ), f"expected workgroup tile size to be the same or larger than wavefront tile size for dimension {dim}"

            assert (
                workgroup_size % wave_size == 0
            ), f"expected workgroup tile size to be an integral multiple of wavefront tile size for dimension {dim}"

        workgroup_dims = set(
            [cons.workgroup_dim for cons in self.workgroup_constraints]
        )

        min_dim = min(workgroup_dims)
        max_dim = max(workgroup_dims)
        assert max_dim - min_dim + 1 == len(
            workgroup_dims
        ), "expected contiguous indices for `workgroup_dim` field in workgroup constraints"

        return

    @property
    def name(self) -> str:
        return self._name

    def _trace(
        self, *, location_capture_config: Optional[LocationCaptureConfig] = None
    ) -> CapturedTrace:
        region_graph = KernelRegionGraph(
            location_capture_config=location_capture_config, func=self._f
        )
        with CompiledContext(region_graph, grid_type=self.grid_type) as context:
            # Get all explicitly defined custom ops
            custom_ops: dict[str, wave_ops.CustomOp] = {
                cls.tkw_op_name: cls
                for _, cls in inspect.getmembers(wave_ops, inspect.isclass)
                if issubclass(cls, wave_ops.CustomOp) and hasattr(cls, "tkw_op_name")
            }

            # Register custom ops
            for name, op in custom_ops.items():
                context.register_custom_op(name, op)

            with region_graph.subtracer() as subtracer:
                root_name, _ = subtracer.trace(self._f)
                kernel_location = capture_function_location(
                    self._f, location_capture_config
                )
                trace = CapturedTrace(region_graph, root_name, kernel_location)
                trace = add_placeholder_locations(trace, self._f)

        return trace

    def create_induction_vars(self, trace: CapturedTrace) -> None:
        """
        Creates induction variables for all the reductions in the graph
        and associates tiling constraints all the reduction dimensions
        with the appropriate induction variables.

        """

        def is_reduction(node: fx.Node):
            custom = get_custom(node)
            return isinstance(custom, Iterate)

        reduction_nodes = trace.walk(is_reduction)
        for node in reduction_nodes:
            custom = get_custom(node)
            self.induction_vars[custom] = get_induction_symbol(custom.axis)
            for tiling_constraint in self.tiling_constraints:
                if tiling_constraint.dim == custom.axis:
                    tiling_constraint.induction_var = self.induction_vars[custom]

    def initialize_wave_constraints(self) -> None:
        """
        For each wave constraint, determines the appropriate wave id by looking
        for workgroup constraints along the same dimension and using information
        from the hardware constraints.

        """

        self._validate_constraints()
        hardware_constraint = self.hardware_constraints[0]
        for wave_constraint in self.wave_constraints:
            for workgroup_constraint in self.workgroup_constraints:
                if wave_constraint.dim == workgroup_constraint.dim:
                    wave_constraint.set_wave_id_from_hardware_and_workgroup_constraint(
                        hardware_constraint, workgroup_constraint
                    )

        if hardware_constraint.waves_per_block is None:
            waves_per_block = [1, 1, 1]
            for wave_constraint in self.wave_constraints:
                count = subs_idxc(wave_constraint.waves_per_block)
                waves_per_block[wave_constraint.workgroup_dim] = count

            hardware_constraint.waves_per_block = tuple(waves_per_block)

    def initialize_reductions(self, trace: CapturedTrace) -> None:
        """
        For each reduction, initializes the reduction count by looking at the
        tiling constraints associated with the reduction.

        """
        is_reduction = lambda node: isinstance(get_custom(node), Iterate)
        for reduction in trace.walk(is_reduction):
            for tiling_constraint in self.tiling_constraints:
                if tiling_constraint.dim == get_custom(reduction).axis:
                    reduction.count = subs_idxc(tiling_constraint.count)

    def get_workgroup_dims(self) -> list[int]:
        """
        Returns the workgroup dimensions that are not aliased.
        """
        # Ignore aliased variables. They will be handled separately.
        aliased_dims = [
            x.source for x in self.constraints if isinstance(x, SymbolicAlias)
        ]
        workgroup_dims = [
            x for x in self.workgroup_constraints if x.dim not in aliased_dims
        ]
        return workgroup_dims

    def update_aliased_workgroup_constraints(
        self, workgroup_dims: dict[int, int]
    ) -> None:
        """
        This function updates the wg_dim for aliased workgroup constraints.
        """
        aliased_dims = [
            x.source for x in self.constraints if isinstance(x, SymbolicAlias)
        ]
        # Update the workgroup constraints for aliases sources.
        for constraint in self.workgroup_constraints:
            if constraint.dim in aliased_dims:
                constraint.wg_dim = workgroup_dims[constraint.workgroup_dim].wg_dim

    def initialize_workgroup_constraints(self) -> None:
        """
        For kernels that distribute more than three dimensions among workgroups,
        we need to update the workgroup constraints for dimensions >= 2
        with the appropriate workgroup index.
        """

        workgroup_dims = self.get_workgroup_dims()
        # Filter to WG2 and above.
        dims_to_delinearize = [x for x in workgroup_dims if x.workgroup_dim >= 2]
        if all(x.workgroup_dim <= 2 for x in dims_to_delinearize):
            return
        # Only take account primary dim for delinearize shape.
        shape = [subs_idxc(x.count) for x in dims_to_delinearize if x.primary]
        new_workgroup_dims = delinearize_index(WORKGROUP_2, shape)
        for delinearize_dim in dims_to_delinearize:
            delinearize_dim.wg_dim = new_workgroup_dims[
                delinearize_dim.workgroup_dim - 2
            ]
        self.update_aliased_workgroup_constraints(workgroup_dims)

    def initialize_symbolic_constraints(self) -> None:
        """
        For each symbolic constraint, create new constraints for the
        related symbolic values with appropriate substitutions.
        """
        new_wg_constraints, new_wave_constraints, new_tiling_constraints = [], [], []
        for symbolic_constraint in self.symbolic_constraints:
            new_wg_constraints += symbolic_constraint.create_new_constraints(
                self.workgroup_constraints
            )
            new_wave_constraints += symbolic_constraint.create_new_constraints(
                self.wave_constraints
            )
            new_tiling_constraints += symbolic_constraint.create_new_constraints(
                self.tiling_constraints
            )
        # Remove wave constraints with same tile size as workgroup constraints
        for wave_constraint in new_wave_constraints:
            for workgroup_constraint in new_wg_constraints:
                if (
                    wave_constraint.dim == workgroup_constraint.dim
                    and wave_constraint.tile_size == workgroup_constraint.tile_size
                ):
                    new_wave_constraints.remove(wave_constraint)
        self.constraints += (
            new_wg_constraints + new_wave_constraints + new_tiling_constraints
        )
        idxc = IndexingContext.current()
        for constraint in self.symbolic_constraints:
            if subs_idxc(constraint.target).is_number:
                idxc._bind_symbol(
                    constraint.source,
                    subs_idxc(constraint.source_to_target(constraint.target)),
                )

    def infer_grid_shape(self, idxc: IndexingContext):
        self.grid_type.dims = [1, 1, 1]
        max_workgroup_dim = 2

        for constraint in self.constraints:
            if isinstance(constraint, GridConstraint):
                grid = []
                for dim in constraint.grid_size:
                    dim_val = safe_subs(dim, idxc.subs)
                    grid.append(int(dim_val))
                self.grid_type.dims = grid
                return

        aliases = [x.source for x in self.constraints if isinstance(x, SymbolicAlias)]
        for constraint in self.workgroup_constraints:
            if constraint.dim in aliases:
                continue
            if not constraint.primary:
                continue
            dim = (
                constraint.workgroup_dim
                if constraint.workgroup_dim < max_workgroup_dim
                else max_workgroup_dim
            )
            self.grid_type.dims[dim] *= safe_subs(constraint.count, idxc.subs)

    def infer_device_layout(self, idxc: IndexingContext):
        self.device_layout.dims = [1, 1, 1]
        max_device_dim = 2
        aliases = [x.source for x in self.constraints if isinstance(x, SymbolicAlias)]
        for constraint in self.device_constraints:
            if constraint.dim in aliases:
                continue
            dim = (
                constraint.device_dim
                if constraint.device_dim < max_device_dim
                else max_device_dim
            )
            self.device_layout.dims[dim] *= safe_subs(constraint.count, idxc.subs)

    def aot_execute(self, args, kwargs):
        raise NotImplementedError("AOT execution for wave not implemented yet.")

    def eager_execute(self, args, kwargs):
        raise NotImplementedError("Eager execution for wave not implemented yet.")

    def __repr__(self):
        return f"tk.wave @{self._name}[{self.grid_type}]"

    def run_manual_schedule(
        self,
        trace: CapturedTrace,
        constraints: list[Constraint],
        schedule: WaveSchedule,
        use_scheduling_barriers: bool = False,
    ):
        """
        Runs the manual schedule provided by the user.
        """
        schedule.trace(
            kernel_trace=trace,
            constraints=constraints,
            use_scheduling_barriers=use_scheduling_barriers,
        )
