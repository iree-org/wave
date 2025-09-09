# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List
from wave_lang.support.logging import get_logger
from .._support.tracing import CapturedTrace
from ..ops.wave_ops import get_custom, CustomOp
from typing import Optional

logger = get_logger("wave.ops_location_check")


def location_check_pass(
    trace: CapturedTrace,
    pass_name: str = "unnamed",
    print_locations: Optional[str] = None,
) -> CapturedTrace:
    """
    Debugging pass for finding where we are dropping locations on ops.

    Check which operations in the graph have location information and which do not.
    Prints a summary of location coverage and lists operations without locations.

    Args:
        pass_name: Name of the pass (for logging context)
        print_locations: None to just print summary, "missing" to print a list
            of ops that are missing locations, "all" to print all op locations.

    Returns:
        The unmodified trace
    """
    ops_with_location: List[CustomOp] = []
    ops_without_location: List[CustomOp] = []
    log_messages = []

    for node in trace.get_root_graph().nodes:
        custom_op = get_custom(node)
        location = getattr(custom_op, "location", None)
        if print_locations == "all":
            log_messages.append(f"  op: {node} location: {location}")
        if location is not None:
            ops_with_location.append(custom_op)
        else:
            ops_without_location.append(custom_op)
            if print_locations == "missing":
                log_messages.append(f"  Missing loc: {node}")

    total_ops = len(ops_with_location) + len(ops_without_location)
    location_percentage = (len(ops_with_location) / total_ops) * 100

    logger.info(
        f"[{pass_name}] Location summary: {len(ops_with_location)}/{total_ops}: {location_percentage:.0f}%"
    )
    for message in log_messages:
        logger.info(message)

    return trace
