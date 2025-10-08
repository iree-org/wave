# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from wave_lang.support.logging import get_logger
from .._support.tracing import CapturedTrace
from ..ops.wave_ops import get_custom
from typing import Optional

logger = get_logger("wave.ops_location_check")


def location_check_pass(
    trace: CapturedTrace,
    pass_name: str = "unnamed",
    log_locations: Optional[str] = "missing",
    log: bool = True,
    enforce_locations: bool = False,
) -> CapturedTrace:
    """
    Debugging pass for finding where we are dropping locations on ops.

    Check which operations in the graph have location information and which do not.
    Logs a summary of location coverage and lists operations without locations.

    Args:
        pass_name: Name of the pass (for logging context)
        log_locations: None to just log summary, "missing" to log a list
            of ops that are missing locations, "all" to log all op locations.
        log: Whether to perform logging (default True)
        enforce_locations: Whether to raise an error if any ops do not have locations (default False)

    Returns:
        The unmodified trace

    Raises:
        RuntimeError: If enforce_locations is True and any operations are missing locations
    """
    # Early exit if neither logging nor enforcement is enabled
    if not log and not enforce_locations:
        return trace

    ops_with_location = 0
    ops_without_location = 0
    log_messages = []

    for node in trace.get_root_graph().nodes:
        location = get_custom(node).location
        if log_locations == "all":
            log_messages.append(f"  op: {node} location: {location}")
        if location is not None:
            ops_with_location += 1
        else:
            ops_without_location += 1
            if log_locations == "missing":
                log_messages.append(f"  Missing loc: {node}")

    total_ops = ops_with_location + ops_without_location
    location_percentage = (ops_with_location / total_ops) * 100

    if log or (enforce_locations and ops_without_location):
        logger.info(
            f"[{pass_name}] Location summary: {ops_with_location}/{total_ops}: {location_percentage:.0f}%"
        )
        for message in log_messages:
            logger.info(message)

    if enforce_locations and ops_without_location:
        raise RuntimeError(
            f"[{pass_name}] {ops_without_location} operations are missing locations, "
            f"but enforce_locations=True requires all operations to have locations"
        )

    return trace
