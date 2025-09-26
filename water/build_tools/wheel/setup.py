# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from setuptools import setup, Extension, find_packages
import subprocess
import os
import shutil
from setuptools.command.build_ext import build_ext
from pathlib import Path
import sys


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        sourcedir = os.fspath(Path(sourcedir).resolve())
        self.sourcedir = os.path.dirname(os.path.dirname(sourcedir))
        print(f"sourcedir: {self.sourcedir}")


def invoke_cmake(*args, cwd=None):
    subprocess.check_call(["cmake", *args], cwd=cwd)


def invoke_git(*args, cwd=None):
    subprocess.check_call(["git", *args], cwd=cwd)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        # Ensure CMake is available
        try:
            invoke_cmake("--version")
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        build_llvm = bool(int(os.environ.get("WATER_BUILD_LLVM", 0)))
        if not build_llvm:
            mlir_dir = os.environ.get("WATER_MLIR_DIR", None)
            if not mlir_dir:
                raise RuntimeError(
                    "Expected MLIR directory to be provided via the WATER_MLIR_DIR environment variable"
                )
            if not os.path.isdir(mlir_dir) or not os.path.exists(
                os.path.join(mlir_dir, "MLIRConfig.cmake")
            ):
                raise RuntimeError(
                    f"WATER_MLIR_DIR={mlir_dir} does not point to the MLIR cmake configuration"
                )

        build_type = os.environ.get("WATER_BUILD_TYPE", "Release")

        if self.dry_run:
            return

        # Create build directory
        build_dir = os.path.abspath(os.path.join(self.build_temp, ext.name))
        shutil.rmtree(build_dir, ignore_errors=True)
        os.makedirs(build_dir, exist_ok=True)

        # Get extension directory
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve() / "water_mlir"

        source_dir = Path(ext.sourcedir)
        if build_llvm:
            llvm_dir = source_dir / "llvm-project"
            llvm_install_dir = source_dir / "llvm-install"
            llvm_build_dir = os.path.abspath(os.path.join(self.build_temp, "llvm"))
            shutil.rmtree(llvm_build_dir, ignore_errors=True)
            os.makedirs(llvm_build_dir, exist_ok=True)
            llvm_sha = (source_dir / "llvm-sha.txt").read_text().strip()
            if not os.path.exists(llvm_dir):
                os.makedirs(llvm_dir, exist_ok=True)
                invoke_git(
                    "clone",
                    "https://github.com/llvm/llvm-project.git",
                    ".",
                    cwd=llvm_dir,
                )

            invoke_git("fetch", cwd=llvm_dir)
            invoke_git("checkout", llvm_sha, cwd=llvm_dir)
            cmake_args = [
                "-G Ninja",
                "-DLLVM_TARGETS_TO_BUILD=host;AMDGPU",
                "-DLLVM_ENABLE_PROJECTS=mlir",
                "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
                "-DBUILD_SHARED_LIBS=OFF",
                "-DLLVM_ENABLE_ASSERTIONS=ON",
                "-DLLVM_ENABLE_ZSTD=OFF",
                "-DLLVM_INSTALL_UTILS=ON",
                "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON",
                f"-DCMAKE_INSTALL_PREFIX={llvm_install_dir}{os.sep}",
                f"-DCMAKE_BUILD_TYPE={build_type}",
                f"-DPython3_EXECUTABLE={sys.executable}",
                "-DWATER_ENABLE_PYTHON=ON",
            ]

            invoke_cmake(llvm_dir / "llvm", *cmake_args, cwd=llvm_build_dir)
            invoke_cmake("--build", ".", "--target", "install", cwd=llvm_build_dir)
            mlir_dir = llvm_install_dir / "lib" / "cmake" / "mlir"

        cmake_args = [
            "-G Ninja",
            f"-DMLIR_DIR={mlir_dir}",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON",
            f"-DCMAKE_INSTALL_PREFIX={extdir}{os.sep}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DWATER_ENABLE_PYTHON=ON",
        ]
        # Configure CMake
        invoke_cmake(source_dir, *cmake_args, cwd=build_dir)

        # Build CMake project
        invoke_cmake(
            "--build", ".", "--target", "tools/water-opt/install", cwd=build_dir
        )
        invoke_cmake(
            "--build", ".", "--target", "install-WaterPythonModules", cwd=build_dir
        )


setup(
    name="water_mlir",
    version="0.1.0",
    packages=find_packages(include=["water_mlir", "water_mlir.*"]),
    ext_modules=[CMakeExtension("water_mlir")],
    cmdclass={"build_ext": CMakeBuild},
)
