# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

import sqlite3
import pandas as pd
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def analyze_rpd_trace(
    output_filename: str, kernel_name: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Analyze RPD trace and return kernel info, top summary, and kernel avg time (us)."""
    with sqlite3.connect(output_filename) as conn:
        df_top = pd.read_sql_query("SELECT * FROM top", conn)
        if df_top.empty:
            logger.error("Empty 'top' dataframe from profiling output.")
            raise ValueError("'df_top' DataFrame is empty.")

        df_kernel_info = pd.read_sql_query(
            """
            SELECT s1.string AS api_name, k.stream, k.gridX, k.gridY, k.gridZ,
                   k.workgroupX, k.workgroupY, k.workgroupZ, s2.string AS kernel_name
            FROM rocpd_kernelapi k
            LEFT JOIN rocpd_string s1 ON k.api_ptr_id = s1.id
            LEFT JOIN rocpd_string s2 ON k.kernelName_id = s2.id;
            """,
            conn,
        ).drop_duplicates()

    logger.info("Kernel Info:\n%s", df_kernel_info)
    logger.info("Top Kernels:\n%s", df_top)

    avg_time_us = 0.0
    if kernel_name:
        kernel_row = df_top[df_top["Name"] == kernel_name]
        if not kernel_row.empty:
            avg_time_us = kernel_row["Ave_us"].iloc[0]
        else:
            logger.warning(f"No entry found for '{kernel_name}' kernel.")
    else:
        avg_time_us = df_top["Ave_us"].iloc[0]

    return avg_time_us, df_kernel_info, df_top
