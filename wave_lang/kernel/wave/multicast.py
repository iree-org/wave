# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Multicast optimization pass.

This pass optimizes tensor loads using cluster-based multicast operations.
Clusters are defined by the `workgroups_per_cluster` field in HardwareConstraint.

The pass analyzes tensor load patterns and determines if they can benefit from
multicast operations where a single load operation can be shared across multiple
workgroups within a cluster.
"""

import logging
import math
import sympy

from .._support.indexing import IndexExpr, IndexSymbol, IndexSequence, safe_subs
from .._support.tracing import CapturedTrace
from ..lang.global_symbols import WORKGROUP_0, WORKGROUP_1, WORKGROUP_2
from ..ops.wave_ops import TensorLoadToLDS, get_custom
from .compile_options import WaveCompileOptions
from .constraints import Constraint, WorkgroupConstraint
from .utils.general_utils import get_hardware_constraint

logger = logging.getLogger(__name__)


def compute_multicast_mask(
    global_tile_index: dict[IndexSymbol, IndexSequence],
    workgroups_per_cluster: tuple[int, int, int],
) -> IndexExpr:
    """
    Compute the multicast mask based on global tile access pattern.

    The mask is a 16-bit value where each bit corresponds to a workgroup in the cluster.
    Workgroups that load the same data will have their corresponding bits set in the mask.

    Args:
        global_tile_index: Symbolic access pattern for the tensor load
        workgroups_per_cluster: Cluster dimensions (X, Y, Z)

    Returns:
        Mask (symbolic expression) indicating which workgroups share this data.
        Returns 0 if all workgroups load different data (no multicast benefit).
    """

    logger.debug(f"Global tile index: {global_tile_index}")

    mask_expr = sympy.sympify(0)
    wg_sizes = workgroups_per_cluster

    wgs = [WORKGROUP_0, WORKGROUP_1, WORKGROUP_2]

    # Current WG index inside the cluster
    subs = {wg: wg % s for wg, s in zip(wgs, wg_sizes)}
    target_index = [
        safe_subs(v.start, subs, simultaneous=True) for v in global_tile_index.values()
    ]
    logger.debug(f"Target index: {target_index}")

    for wg_z in range(wg_sizes[2]):
        for wg_y in range(wg_sizes[1]):
            for wg_x in range(wg_sizes[0]):
                # Specific WG index inside the cluster
                subs = {wg: value for wg, value in zip(wgs, (wg_x, wg_y, wg_z))}
                new_index = [
                    safe_subs(v.start, subs, simultaneous=True)
                    for v in global_tile_index.values()
                ]
                logger.debug(f"New index: {new_index}")

                # Linear index: bit position for this workgroup
                bit_pos = wg_x + wg_y * wg_sizes[0] + wg_z * wg_sizes[0] * wg_sizes[1]

                # Bit is set when wg index matches the current wg index in the cluster
                condition = sympy.And(
                    *(
                        sympy.Eq(idx1, idx2)
                        for idx1, idx2 in zip(new_index, target_index)
                    )
                )
                mask_expr += sympy.Piecewise((1 << bit_pos, condition), (0, True))

    # Try to simplify
    mask_expr = sympy.simplify(mask_expr)

    logger.debug(f"Symbolic multicast mask: {mask_expr}")
    return mask_expr


def multicast(
    trace: CapturedTrace,
    constraints: list[Constraint],
    options: WaveCompileOptions,
):
    """
    Optimize tensor loads using cluster-based multicast operations.

    This pass should run after tensor_load_to_shared to analyze and optimize
    tensor load operations that can benefit from multicast within clusters.

    Args:
        trace: The captured trace to optimize
        constraints: List of constraints including HardwareConstraint with
                    workgroups_per_cluster specification
        options: Compilation options

    The pass performs the following:
    1. Check if multicast is supported (cluster dimensions specified)
    2. Identify tensor load operations that can benefit from multicast
    3. Add multicast_mask metadata to eligible loads
    """
    # Get hardware constraints
    hardware_constraint = get_hardware_constraint(constraints)

    workgroups_per_cluster = hardware_constraint.workgroups_per_cluster
    if "gfx1250" not in options.target:
        logger.info("Multicast optimization is not supported on this architecture")
        return

    if not workgroups_per_cluster:
        logger.info(
            "No workgroups_per_cluster specified, skipping multicast optimization"
        )
        return

    cluster_size = math.prod(workgroups_per_cluster)
    if cluster_size <= 1:
        logger.info(f"Cluster size is {cluster_size}, skipping multicast optimization")
        return

    num_workgroups = (
        max(
            wg.workgroup_dim
            for wg in constraints
            if isinstance(wg, WorkgroupConstraint)
        )
        + 1
    )
    if num_workgroups > 3:
        raise ValueError(f"Unsupported number of workgroups: {num_workgroups}")

    if cluster_size > 16 or len(workgroups_per_cluster) != 3:
        raise ValueError(f"Invalid cluster configuration: {workgroups_per_cluster}")

    logger.info(
        f"Cluster configuration: {workgroups_per_cluster} (total {cluster_size} workgroups)"
    )

    # Find all TensorLoadToLDS operations
    tensor_loads = []
    for node in trace.walk(lambda n: isinstance(get_custom(n), TensorLoadToLDS)):
        tensor_loads.append(get_custom(node))

    if not tensor_loads:
        logger.info("No TensorLoadToLDS operations found")
        return

    logger.info(f"Found {len(tensor_loads)} TensorLoadToLDS operations")

    # Analyze each tensor load and compute multicast mask
    for tensor_load in tensor_loads:
        mask = compute_multicast_mask(
            tensor_load.global_tile_index, workgroups_per_cluster
        )

        if mask != 0:
            # Update the tensor load with multicast mask
            tensor_load.update_arg("multicast_mask", mask)
            logger.info(
                f"TensorLoadToLDS at {tensor_load.fx_node.name}: multicast_mask = {mask}"
            )
