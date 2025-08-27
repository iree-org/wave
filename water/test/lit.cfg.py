# -*- Python -*-

import os

import lit.formats
import lit.util

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "WATER"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".py"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.water_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "Inputs",
    "Examples",
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
    "lit.cfg.py",
]

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.water_obj_root, "test")
config.water_tools_dir = os.path.join(config.water_obj_root, "bin")
config.water_libs_dir = os.path.join(config.water_obj_root, "lib")

config.substitutions.append(("%water_libs", config.water_libs_dir))
py_root_base = os.path.dirname(os.path.dirname(config.water_obj_root))  # -> .../build
config.substitutions.append(
    ("%py_pkg_root", os.path.join(py_root_base, "python_packages"))
)

if os.path.isdir(os.path.join(py_root_base, "python_packages", "water_mlir")):
    config.available_features.add("water_python")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.water_tools_dir, config.llvm_tools_dir]
tools = [
    "mlir-opt",
    "water-opt",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
