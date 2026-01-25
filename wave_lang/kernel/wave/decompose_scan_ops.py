import math
from operator import ge
from typing import Callable, Optional

import torch.fx as fx

import wave_lang.kernel.lang as tkl
from wave_lang.kernel._support.indexing import IndexSequence, IndexSymbol
from wave_lang.kernel.lang.global_symbols import SHARED_ADDRESS_SPACE
from wave_lang.kernel.wave.utils.general_utils import all_equal
from wave_lang.kernel.wave.utils.symbol_utils import subs_idxc

from .._support.dtype import i1
from .._support.location import CapturedLocation
from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    Add,
    Allocate,
    Broadcast,
    Conditional,
    Cumsum,
    CustomOp,
    Eq,
    Extract,
    NewRegister,
    NewScalar,
    Placeholder,
    Read,
    Reshape,
    ScanOp,
    SelectOp,
    ShuffleOp,
    Write,
    get_custom,
)
from .constraints import HardwareConstraint, WaveConstraint, WorkgroupConstraint
from .utils.classes import ShuffleMode
from .utils.general_utils import all_equal, delinearize_index
from .utils.graph_utils import DCE, get_outer_node


def get_graph_node(
    custom: CustomOp, graph: fx.Graph, location: Optional[CapturedLocation]
) -> fx.Node:
    custom.add_to_graph(graph)
    custom.location = location
    return custom.fx_node


def emit_local_inclusive_scan(
    binary_fn: Callable,
    scan_src: list[fx.Node],
    graph: fx.Graph,
    elements_per_thread: int,
    location: Optional[CapturedLocation],
) -> list[list[fx.Node]]:
    """
    Perform local inclusive scan for `n` elements per thread.
    """
    result = []
    for node in scan_src:
        values = [
            get_graph_node(Extract(node, [i]), graph, location)
            for i in range(elements_per_thread)
        ]
        values[0].index = node.index
        for i in range(1, elements_per_thread):
            values[i] = get_graph_node(
                binary_fn(values[i], values[i - 1]), graph, location
            )
            values[i].index = node.index

        result.append(values)
    return result


def emit_variable_scan(
    binary_fn: Callable,
    src: list[list[fx.Node]],
    graph: fx.Graph,
    elements_per_thread: int,
    location: Optional[CapturedLocation],
) -> list[list[fx.Node]]:
    result: list[list[fx.Node]] = []
    result.append(src[0])

    for scan_src_idx in range(1, len(src)):
        prev_thread_last_val = result[scan_src_idx - 1][-1]
        temp: list[fx.Node] = []
        for i in range(elements_per_thread):
            cur_val = src[scan_src_idx][i]
            offset_val = get_graph_node(
                binary_fn(prev_thread_last_val, cur_val), graph, location
            )
            offset_val.index = get_custom(src[scan_src_idx][i]).index
            temp.append(offset_val)
        result.append(temp)

    return result


def emit_global_scan(
    binary_fn: Callable,  # Supports only Add for now.
    src: fx.Node,
    local_scan: list[list[fx.Node]],
    graph: fx.Graph,
    subgroup_size: int,
    hardware_constraint: HardwareConstraint,
    local_scan_size: int,
    scan_dim: IndexSymbol,
    location: Optional[CapturedLocation],
) -> list[fx.Node]:
    """
    Emit an intra-warp inclusive scan using butterfly pattern scan and masking.
    In this algorithm, we are using two-pass approach.
    If the number of elements per thread is one, then we just need one shuffle pass.
    If there are more than one elements per thread, we need to calculate the exclusive
        offset for the next lane elements using the information from the previous lane.
        That is why we have another shuffle phase.

    TODO: Can we make it more efficient?
    """
    lane_id = (
        hardware_constraint.linearized_thread_id % hardware_constraint.threads_per_wave
    )

    scanop_result = local_scan[-1][-1]
    last_local_scan_node = get_custom(scanop_result)

    # Set the index to the scan_dim with size 1. The size 1 here is because
    # when running the shuffle_xor we are only picking one element per thread.
    scanop_result.index = {scan_dim: IndexSequence(0, 1, 1)}

    num_steps = int(math.log2(float(subgroup_size)))
    for idx in range(num_steps):
        offset_val = 1 << idx

        # shuffle operation to get value from another thread
        shuffle = ShuffleOp(scanop_result, offset_val, subgroup_size, ShuffleMode.UP)
        shuffle_val_node = get_graph_node(shuffle, graph, location)

        # we are explicitly adding index because this pass is being
        # applied after the indexing phase
        # TODO (xintin): check if we can replace register with scalar.
        # No point using register for a scalar. Applies to other objects too.
        zero_vec = get_graph_node(
            NewRegister(
                get_custom(shuffle_val_node).type.symbolic_shape,
                get_custom(shuffle_val_node).type.dtype,
                0.0,
            ),
            graph,
            location,
        )
        zero_vec.index = last_local_scan_node.index

        # condition node: thread ID >= offset
        cond_expr = ge(lane_id, offset_val)
        cond_node = get_graph_node(
            NewRegister(get_custom(scanop_result).type.symbolic_shape, i1, cond_expr),
            graph,
            location,
        )
        cond_node.index = last_local_scan_node.index

        # apply shuffle_val only if condition is true; else use 0
        masked = get_graph_node(
            SelectOp(cond=cond_node, if_true=shuffle_val_node, if_false=zero_vec),
            graph,
            location,
        )

        # perform binary scan op
        scanop_result = get_graph_node(
            binary_fn(scanop_result, masked), graph, location
        )
        final_scanop_result = [scanop_result]

    if local_scan_size > 1:
        # Phase 2 to calculate exclusive offset from previous lane id.
        scan_offset = ShuffleOp(scanop_result, 1, subgroup_size, ShuffleMode.UP)
        scan_offset_node = get_graph_node(scan_offset, graph, location)

        lane_id_ge_one = ge(lane_id, 1)
        lane_id_ge_one_node = get_graph_node(
            NewRegister(
                get_custom(scanop_result).type.symbolic_shape, i1, lane_id_ge_one
            ),
            graph,
            location,
        )
        # TODO: debug why last_local_scan_node.index is setting to None if assigned outside.
        lane_id_ge_one_node.index = last_local_scan_node.index

        identity_vec_node = get_graph_node(
            NewRegister(
                get_custom(scan_offset_node).type.symbolic_shape,
                get_custom(scan_offset_node).type.dtype,
                0.0,
            ),
            graph,
            location,
        )
        identity_vec_node.index = last_local_scan_node.index

        excl_offset = get_graph_node(
            SelectOp(
                cond=lane_id_ge_one_node,
                if_true=scan_offset_node,
                if_false=identity_vec_node,
            ),
            graph,
            location,
        )

        # Update the prefix sum for each locally scanned element based on the updated offset
        final_scanop_result = []
        for loc_scan in local_scan:
            custom = get_custom(loc_scan[0])

            final_scalars = [
                get_graph_node(binary_fn(lane_elem, excl_offset), graph, location)
                for lane_elem in loc_scan
            ]

            # pack the output in the expected order
            scanop_result = Reshape(
                args=final_scalars,
                target_vector_shape={scan_dim: local_scan_size},
            ).add_to_graph(graph)
            scanop_result.location = location

            # assign attributes
            scanop_result.index = custom.index
            scanop_result.expanded_dims = custom.expanded_dims
            scanop_result.vector_shapes = custom.vector_shapes

            final_scanop_result.append(scanop_result)

    return final_scanop_result


def emit_interwave_scan(
    binary_fn: Callable,
    orig_src: fx.Node,
    src: fx.Node,
    graph: fx.Graph,
    trace: CapturedTrace,
    scan_dim: IndexSymbol,
    num_scan_waves: int,
    local_scan_size: int,
    wg_constraint_map: dict,
    hardware_constraint: HardwareConstraint,
    location: Optional[CapturedLocation],
) -> list[fx.Node]:
    """
    Emit inter-wave scan operations for block-level scans.

    This function handles
    1. scanning across multiple waves within a workgroup
    2. storing individual wave results into shared memory
    3. propagating prefix sums across waves.
    It achieves the above by storing individual wave results into shared memory and propagating
    prefix sums across waves.

    Args
        orig_src: The original source node.
        src: The source node after the local scan.
        graph: The graph to add the nodes to.
        trace: The trace to add the subgraph to.
        scan_dim: The dimension along which to scan.
        num_scan_waves: The number of waves to scan across.
        local_scan_size: The number of elements each thread handles (per-thread element count).
        wg_constraint_map: The workgroup constraint map.
        hardware_constraint: The hardware constraint.
        location: The location of the node.
    """

    # Identify lane_id and wave_id-along-scan-dim
    # Here, lane_id: the id of the lane within the wave
    lane_id = (
        hardware_constraint.linearized_thread_id % hardware_constraint.threads_per_wave
    )
    # Here, wave_id: the id of the wave within the workgroup
    wave_id = delinearize_index(
        hardware_constraint.linearized_thread_id
        // hardware_constraint.threads_per_wave,
        hardware_constraint.waves_per_block,
    )
    scan_wg_dim = wg_constraint_map[scan_dim].workgroup_dim
    scan_wave_id = wave_id[scan_wg_dim]

    # tpw: the number of threads per wave
    tpw = hardware_constraint.threads_per_wave

    # Ensure src has the correct index for subsequent operations
    # Only broadcast if local_scan_size == 1 (scalar case); otherwise src already has correct shape
    if local_scan_size == 1:
        src = get_graph_node(
            Broadcast(src, target_shape=orig_src.type.symbolic_shape), graph, location
        )
    src.index = get_custom(orig_src).index.copy()

    # Use local_scan_size for the index size so broadcast/add have matching vector shapes
    src.index.update({scan_dim: IndexSequence(0, local_scan_size, 1)})

    #  Compute per-wave total: last element of the per-thread vector
    wave_total = get_graph_node(Extract(src, [local_scan_size - 1]), graph, location)
    wave_total.index = src.index

    #  Shared buffer for per-wave totals
    sums_buf = Allocate(
        (scan_dim,),
        (num_scan_waves,),
        get_custom(wave_total).type.dtype,
        SHARED_ADDRESS_SPACE,
    ).add_to_graph(graph)
    # Add location to the sums_buf node
    sums_buf.location = location

    # only last lane writes wave_total to sums_buf[scan_wave_id]
    lane_id_reg = get_graph_node(NewScalar(lane_id, tkl.i32), graph, location)
    last_lane_reg = get_graph_node(NewScalar(tpw - 1, tkl.i32), graph, location)
    is_last_lane = get_graph_node(Eq(lane_id_reg, last_lane_reg), graph, location)

    # Create a subgraph for the last lane to write the wave_total to the sums_buf
    exec_on_last = fx.Graph()
    sub_store = f"store_wave_total_{src.name}"

    # Create placeholders for the wave_total and sums_buf
    ph_total = get_graph_node(
        Placeholder.from_fx_node(wave_total), exec_on_last, location
    )
    ph_total.type = get_custom(wave_total).type

    ph_sums = get_graph_node(Placeholder.from_fx_node(sums_buf), exec_on_last, location)
    ph_sums.type = get_custom(sums_buf).type
    ph_sums.meta["lifted"] = sums_buf

    # Create a write node to write the wave_total to the sums_buf
    write_sum = Write(ph_total, ph_sums, 1).add_to_graph(exec_on_last)
    write_sum.location = location
    write_sum.index = {scan_dim: IndexSequence(scan_wave_id, 1, 1)}

    # conditional node to execute the store subgraph if the current lane is the last lane.
    # Add location to the conditional node
    cond_store = get_graph_node(
        Conditional(
            is_last_lane,
            subgraph_name=sub_store,
            implicit_captures=[get_outer_node(wave_total), sums_buf],
        ),
        graph,
        location,
    )
    # Set the parent op of the store subgraph to the conditional node.
    # Add the store subgraph to the trace and update the root graph.
    # If we do not do this, the conditional body wouldn't be emitted,
    # and code generation would crash because the Conditional references a subgraph
    # that the compiler can't find.
    exec_on_last.parent_op = cond_store
    trace.add_subgraph(sub_store, exec_on_last)
    trace.get_root_graph().subgraphs[sub_store] = exec_on_last

    # read all per-wave totals
    read_totals = Read(
        sums_buf,
        elements_per_thread=num_scan_waves,
        _write_dependency=[cond_store, write_sum],
    ).add_to_graph(graph)
    # Add location to the read_totals node
    read_totals.location = location
    # Set the index to the scan_dim with size 1. The size 1 here is because
    # when running the shuffle_xor we are only picking one element per thread.
    read_totals.index = {scan_dim: IndexSequence(0, 1, 1)}

    # Inclusive prefix of totals: incl[i] = sum_{k<=i} totals[k]
    incl = emit_local_inclusive_scan(
        binary_fn, [read_totals], graph, num_scan_waves, location
    )[0]

    # convert to exclusive offsets: excl[0]=0, excl[i]=incl[i-1]
    # because we need to calculate the exclusive offset for the next lane elements using
    # the information from the previous lane.
    zero = get_graph_node(
        NewRegister(
            get_custom(incl[0]).type.symbolic_shape, get_custom(incl[0]).type.dtype, 0.0
        ),
        graph,
        location,
    )
    zero.index = incl[0].index
    excl_scalars = [zero] + incl[:-1]

    # reshape the exclusive offsetts to a vector of size num_scan_waves
    # Reshape is needed to pack the individual scalar nodes into a single vector
    # so we can dynamically index into it.
    offsets_vec = Reshape(
        args=excl_scalars,
        target_vector_shape={scan_dim: num_scan_waves},
    ).add_to_graph(graph)
    offsets_vec.location = location

    # each wave picks its offset and adds to src
    off = get_graph_node(Extract(offsets_vec, [scan_wave_id]), graph, location)
    # use orig_src's symbolic shape for broadcast target (src's shape may differ after reshape)
    target_shape = get_custom(orig_src).type.symbolic_shape
    off = get_graph_node(Broadcast(off, target_shape), graph, location)
    off.index = src.index

    out = get_graph_node(binary_fn(off, src), graph, location)
    return [out]


def decompose_scan_ops(
    trace: CapturedTrace,
    constraints: list,
) -> None:
    """Decomposes high-level scan operations (ScanOp) in a captured FX trace
    into lower-level warp-level inclusive scan implementations.

    Currently only supports Cumsum operations using a butterfly-style scan.

    Args:
        trace (CapturedTrace): fx trace representing the computation graph.
        constraints (list): list of hardware-related constraints

    Raises:
        NotImplementedError: If the scan operation type is not yet supported.
    """
    scan_nodes = trace.walk(lambda node: isinstance(get_custom(node), ScanOp))
    if not scan_nodes:
        return

    hardware_constraint = next(
        c for c in constraints if isinstance(c, HardwareConstraint)
    )

    wave_constraint_map = {
        c.dim: c for c in constraints if isinstance(c, WaveConstraint)
    }

    workgroup_constraint_map = {
        c.dim: c for c in constraints if isinstance(c, WorkgroupConstraint)
    }

    subgroup_size = hardware_constraint.threads_per_wave

    for node in scan_nodes:
        custom = get_custom(node)
        if not isinstance(custom, Cumsum):
            raise NotImplementedError(f"ScanOp '{custom}' not supported")

        with custom.graph.inserting_before(custom.fx_node):
            scan_src, scan_acc, scan_dim, block_scan = node.args
            binary_fn = Add

            if scan_dim is None:
                raise ValueError("No scan dim specified, please specify a scan dim.")

            if not isinstance(scan_src, (list, tuple)):
                scan_src = [scan_src]

            # sanity checks and check for the fastest dim
            src_fastest_dims = [
                get_custom(arg).type.symbolic_shape[-1] for arg in scan_src
            ]
            if not all_equal(src_fastest_dims):
                raise NotImplementedError(
                    "NYI: Expect all reduce_src to have same fastest dim."
                )
            if scan_dim is not src_fastest_dims[0]:
                raise NotImplementedError(
                    f"Only implemented reduction on fastest dimension. Got {scan_dim} and {src_fastest_dims}."
                    f"\n{custom}"
                )

            get_thread_shape = lambda index: max(
                subs_idxc(x.size) for x in index.values()
            )

            local_scan_sizes = []
            for arg in scan_src:
                try:
                    op = get_custom(arg)
                    thread_shape = get_thread_shape(op.index)
                    local_scan_sizes.append(thread_shape)
                except Exception as e:
                    index_str = "\n".join(f"{k}: {v}" for k, v in op.index.items())
                    raise RuntimeError(
                        f"Error in decompose_scan_ops: {scan_src} with index\n"
                        f"{index_str}\n{scan_src=}\n{scan_acc=}\n{scan_dim=}"
                    ) from e

            if not all_equal(local_scan_sizes):
                raise NotImplementedError(
                    "NYI: Expect all reduce_src to have same local reduce size."
                )

            # Phase 1: Local scan
            local_scan = emit_local_inclusive_scan(
                binary_fn,
                scan_src,
                custom.graph,
                local_scan_sizes[0],
                custom.location,
            )

            # Phase 2: Src wide local scan
            local_scan = emit_variable_scan(
                binary_fn,
                local_scan,
                custom.graph,
                local_scan_sizes[0],
                custom.location,
            )

            final_scan = emit_global_scan(
                binary_fn,
                scan_src[0],
                local_scan,
                custom.graph,
                subgroup_size,
                hardware_constraint,
                local_scan_sizes[0],
                scan_dim,
                custom.location,
            )

            if block_scan:
                # compute num_warps to scan across
                num_scan_waves = int(
                    workgroup_constraint_map[scan_dim].tile_size
                    // wave_constraint_map[scan_dim].tile_size
                )

                if num_scan_waves > subgroup_size:
                    raise NotImplementedError(
                        "The 2nd stage butterfly shuffle reduces the"
                        "the reduction outputs from all the wave. Hence, can only handle at most "
                        "threads_per_wave number of warps."
                    )

                # Scan and update output between waves, by storing individual wave result into shared memory,
                # and then adding it to the rest waves to update it.
                final_scan = emit_interwave_scan(
                    binary_fn,
                    scan_src[0],
                    final_scan[0],
                    custom.graph,
                    trace,
                    scan_dim,
                    num_scan_waves,
                    local_scan_sizes[0],
                    workgroup_constraint_map,
                    hardware_constraint,
                    custom.location,
                )

            # Update the users based on the global scan `reshape` results.
            for user in custom.users:
                user.update_arg(
                    custom.fx_node, final_scan[user.expanded_dims[scan_dim]]
                )

    DCE(trace)
