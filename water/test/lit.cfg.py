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
config.suffixes = [".mlir"]

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
config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.water_obj_root, "test")
config.water_tools_dir = os.path.join(config.water_obj_root, "bin")
config.water_libs_dir = os.path.join(config.water_obj_root, "lib")

config.substitutions.append(("%water_libs", config.water_libs_dir))

# Locate the Python package root:
# 1) Prefer build-tree location produced by add_mlir_python_modules:
#    <build>/python_packages
# 2) Allow override to point at an install prefix via WATER_PYTHON_PACKAGE_ROOT
py_pkg_root = os.path.join(config.water_obj_root, "python_packages")
py_pkg_root = os.environ.get("WATER_PYTHON_PACKAGE_ROOT", py_pkg_root)

config.substitutions.append(("%py_pkg_root", py_pkg_root))
if config.python_exe:
    config.substitutions.append(("%python", config.python_exe))

if os.path.isdir(os.path.join(py_pkg_root, "water_mlir")):
    config.available_features.add("water_python")
    # Ensure spawned processes can import the package during tests
    llvm_config.with_environment("PYTHONPATH", py_pkg_root, append_path=True)

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.water_tools_dir, config.llvm_tools_dir]
tools = [
    "mlir-opt",
    "water-opt",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
