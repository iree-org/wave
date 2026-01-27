# -*- Python -*-
# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import lit.formats
import lit.util

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

config.name = 'WaveASM'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.waveasm_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# Add tools to the path
tool_dirs = [config.waveasm_tools_dir, config.llvm_tools_dir]
tools = ['waveasm-translate', 'FileCheck', 'count', 'not']
llvm_config.add_tool_substitutions(tools, tool_dirs)

# ROCm toolchain detection for integration tests
import shutil

rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
rocm_clang = os.path.join(rocm_path, 'llvm', 'bin', 'clang')
rocm_lld = os.path.join(rocm_path, 'llvm', 'bin', 'ld.lld')

# Check if ROCm clang exists and supports AMDGCN
if os.path.isfile(rocm_clang) and os.path.isfile(rocm_lld):
    config.available_features.add('rocm-toolchain')
    config.substitutions.append(('%rocm_clang', rocm_clang))
    config.substitutions.append(('%rocm_lld', rocm_lld))
else:
    # Try fallback to system clang
    system_clang = shutil.which('clang')
    system_lld = shutil.which('ld.lld')
    if system_clang and system_lld:
        config.available_features.add('rocm-toolchain')
        config.substitutions.append(('%rocm_clang', system_clang))
        config.substitutions.append(('%rocm_lld', system_lld))
