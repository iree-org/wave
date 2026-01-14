# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import distutils.command.build
import os
import shutil
import subprocess
import sys
import ninja
from pathlib import Path
from typing import Optional

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

THIS_DIR = os.path.realpath(os.path.dirname(__file__))
REPO_ROOT = THIS_DIR
BUILD_TYPE = os.environ.get("WAVE_BUILD_TYPE", "Release")
BUILD_WATER = int(os.environ.get("WAVE_BUILD_WATER", "0"))
WATER_DIR = os.getenv("WAVE_WATER_DIR")
LLVM_DIR = os.getenv("WAVE_LLVM_DIR")
LLVM_REPO = os.getenv("WAVE_LLVM_REPO", "https://github.com/llvm/llvm-project.git")
BUILD_SHARED_LIBS = os.getenv("WAVE_LLVM_BUILD_SHARED_LIBS", "OFF")
NINJA_PATH = Path(ninja.BIN_DIR) / "ninja"


class CMakeExtension(Extension):
    def __init__(
        self,
        name: str,
        sourcedir: str,
        install_dir: Optional[str] = None,
        need_llvm: bool = False,
        cmake_args: Optional[list[str]] = None,
        external_build_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())
        self.install_dir = install_dir
        self.need_llvm = need_llvm
        self.cmake_args = cmake_args or []
        self.external_build_dir = external_build_dir


def invoke_cmake(*args, cwd=None, env=None):
    subprocess.check_call(["cmake", *args], cwd=cwd, env=env)


def invoke_git(*args, cwd=None):
    subprocess.check_call(["git", *args], cwd=cwd)


def check_water_install(build_dir: Path):
    # Validate build directory contains expected artifacts
    water_opt_path = build_dir / "bin" / "water-opt"
    if not water_opt_path.exists():
        raise RuntimeError(
            f"WAVE_WATER_DIR does not contain water-opt at {water_opt_path}. "
            "Make sure you have built water with 'ninja' or 'cmake --build .'."
        )

    python_packages_dir = build_dir / "python_packages" / "water_mlir"
    if not python_packages_dir.exists():
        raise RuntimeError(
            f"WAVE_WATER_DIR does not contain Python packages at {python_packages_dir}. "
            "Make sure you built water with -DWATER_ENABLE_PYTHON=ON -DMLIR_ENABLE_BINDINGS_PYTHON=ON."
        )


class CMakeBuild(build_ext):
    def run(self) -> None:
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: CMakeExtension) -> None:
        # Get extension directory
        if ext.install_dir:
            # Use custom install directory relative to package root
            extdir = Path.cwd() / ext.install_dir
        else:
            # Default behavior: install alongside the extension
            ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
            extdir = ext_fullpath.parent.resolve()

        # Ensure install directory exists
        os.makedirs(extdir, exist_ok=True)

        if ext.name == "water" and ext.external_build_dir:
            print(
                f"Installing water from external build directory: {ext.external_build_dir}"
            )
            print(f"  Install location: {extdir}")
            print(
                "  Using CMAKE_INSTALL_MODE=ABS_SYMLINK for symlink-based installation"
            )
            check_water_install(ext.external_build_dir)
            env = os.environ.copy()
            env["CMAKE_INSTALL_MODE"] = "ABS_SYMLINK"
            invoke_cmake(
                "--install",
                str(ext.external_build_dir),
                "--prefix",
                extdir,
                env=env,
            )
            return

        # Create build directory
        build_dir = os.path.abspath(os.path.join(self.build_temp, ext.name))
        os.makedirs(build_dir, exist_ok=True)

        # Configure CMake
        cmake_args = [
            "-G",
            "Ninja",
            f"-DCMAKE_MAKE_PROGRAM:FILEPATH={NINJA_PATH}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_INSTALL_PREFIX={extdir}{os.sep}",
            f"-DCMAKE_BUILD_TYPE={BUILD_TYPE}",
        ]

        if ext.need_llvm:
            # Configure LLVM if WAVE_LLVM_DIR is set, otherwise build from source
            wave_llvm_dir = LLVM_DIR
            if wave_llvm_dir:
                wave_llvm_dir = Path(wave_llvm_dir).resolve()
            else:
                # Build LLVM from source
                print("WAVE_LLVM_DIR not set, building LLVM from source...")
                wave_llvm_dir = self._build_llvm()

            llvm_dir = wave_llvm_dir / "lib" / "cmake" / "llvm"
            mlir_dir = wave_llvm_dir / "lib" / "cmake" / "mlir"
            lld_dir = wave_llvm_dir / "lib" / "cmake" / "lld"
            cmake_args += [
                f"-DLLVM_DIR={llvm_dir}",
                f"-DMLIR_DIR={mlir_dir}",
                f"-DLLD_DIR={lld_dir}",
            ]
            print(f"Using built LLVM from: {wave_llvm_dir}")
            print(f"  LLVM_DIR: {llvm_dir}")
            print(f"  MLIR_DIR: {mlir_dir}")
            print(f"  LLD_DIR:  {lld_dir}")

        # Clang is required on Windows, since Wave runtime uses variable-length
        # arrays (VLAs) which are not supported by MSVC
        if os.name == "nt":
            cmake_args += [
                "-DCMAKE_C_COMPILER=clang",
                "-DCMAKE_CXX_COMPILER=clang++",
            ]

        cmake_args += ext.cmake_args

        invoke_cmake(ext.sourcedir, *cmake_args, cwd=build_dir)

        # Build CMake project
        invoke_cmake("--build", ".", "--target", "install", cwd=build_dir)

    def _build_llvm(self):
        """Build LLVM from source using the commit hash from water/llvm-sha.txt."""

        # Read LLVM commit hash from water/llvm-sha.txt
        llvm_sha_file = Path(REPO_ROOT) / "water" / "llvm-sha.txt"
        if not llvm_sha_file.exists():
            raise RuntimeError(
                f"LLVM SHA file not found at {llvm_sha_file}. "
                "Cannot build LLVM from source."
            )
        llvm_sha = llvm_sha_file.read_text().strip()

        # Setup directories
        water_dir = Path(REPO_ROOT) / "water"
        llvm_dir = water_dir / "llvm-project"
        llvm_install_dir = water_dir / "llvm-install"
        llvm_build_dir = Path(self.build_temp) / "llvm"

        # Check if LLVM is already built with the correct SHA
        vcs_revision_file = (
            llvm_install_dir / "include" / "llvm" / "Support" / "VCSRevision.h"
        )
        if llvm_install_dir.exists() and vcs_revision_file.exists():
            # Read the installed LLVM SHA
            vcs_content = vcs_revision_file.read_text()
            installed_sha = None
            for line in vcs_content.split("\n"):
                if line.strip().startswith("#define LLVM_REVISION"):
                    # Extract SHA from: #define LLVM_REVISION R"(5c35af8...)"
                    installed_sha = line.split('"')[1].replace("(", "").replace(")", "")
                    break

            if installed_sha == llvm_sha:
                print(
                    f"LLVM already built with correct SHA {llvm_sha}, skipping rebuild"
                )
                return llvm_install_dir
            else:
                print(
                    f"LLVM SHA mismatch: installed={installed_sha}, expected={llvm_sha}"
                )

        print("Removing old LLVM installation...")
        shutil.rmtree(llvm_install_dir, ignore_errors=True)

        print(f"Building LLVM from commit: {llvm_sha}")

        # Clean and create build directory
        shutil.rmtree(llvm_build_dir, ignore_errors=True)
        os.makedirs(llvm_build_dir, exist_ok=True)

        # Clone llvm-project if it doesn't exist
        if not llvm_dir.exists():
            os.makedirs(llvm_dir, exist_ok=True)
            print(f"Cloning LLVM project to {llvm_dir}...")
            invoke_git(
                "clone",
                "--depth",
                "1",
                "--no-checkout",
                LLVM_REPO,
                ".",
                cwd=llvm_dir,
            )

        # Fetch and checkout specific commit
        print(f"Fetching and checking out LLVM commit {llvm_sha}...")
        invoke_git("fetch", "--depth", "1", "origin", llvm_sha, cwd=llvm_dir)
        invoke_git("checkout", llvm_sha, cwd=llvm_dir)

        # Configure LLVM build
        llvm_cmake_args = [
            "-G",
            "Ninja",
            f"-DCMAKE_MAKE_PROGRAM:FILEPATH={NINJA_PATH}",
            "-DLLVM_TARGETS_TO_BUILD=host;AMDGPU",
            "-DLLVM_ENABLE_PROJECTS=llvm;mlir;lld",
            "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
            f"-DBUILD_SHARED_LIBS={BUILD_SHARED_LIBS}",
            "-DLLVM_ENABLE_ASSERTIONS=ON",
            "-DLLVM_ENABLE_ZSTD=OFF",
            "-DLLVM_INSTALL_UTILS=ON",
            "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON",
            f"-DCMAKE_INSTALL_PREFIX={llvm_install_dir}",
            f"-DCMAKE_BUILD_TYPE={BUILD_TYPE}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]

        print(f"Configuring LLVM in {llvm_build_dir}...")
        invoke_cmake(str(llvm_dir / "llvm"), *llvm_cmake_args, cwd=llvm_build_dir)

        # Build and install LLVM
        print("Building and installing LLVM (this may take a while)...")
        invoke_cmake("--build", ".", "--target", "install", cwd=llvm_build_dir)

        return llvm_install_dir


# Override build command so that we can build into _python_build
# instead of the default "build". This avoids collisions with
# typical CMake incantations, which can produce all kinds of
# hilarity (like including the contents of the build/lib directory).
class BuildCommand(distutils.command.build.build):
    def initialize_options(self):
        distutils.command.build.build.initialize_options(self)
        self.build_base = "_python_build"


ext_modules = [
    CMakeExtension("wave_runtime", "wave_lang/kernel/wave/runtime"),
]

if BUILD_WATER and WATER_DIR:
    raise RuntimeError("WAVE_WATER_DIR and WAVE_BUILD_WATER are mutually exclusive")

if BUILD_WATER or WATER_DIR:
    ext_modules += [
        CMakeExtension(
            "wave_execution_engine",
            "wave_lang/kernel/wave/execution_engine",
            install_dir="wave_lang/kernel/wave/execution_engine",
            need_llvm=True,
        ),
        CMakeExtension(
            "water",
            "water",
            install_dir="wave_lang/kernel/wave/water_mlir",
            need_llvm=True,
            cmake_args=[
                f"-DBUILD_SHARED_LIBS={BUILD_SHARED_LIBS}",
                "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON",
                "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
                "-DWATER_ENABLE_PYTHON=ON",
            ],
            external_build_dir=Path(WATER_DIR).resolve() if WATER_DIR else None,
        ),
    ]

# Only build-related configuration here.
# All metadata and dependencies are in pyproject.toml.
# Rust extensions are configured in pyproject.toml under [tool.setuptools-rust].
setup(
    cmdclass={"build": BuildCommand, "build_ext": CMakeBuild},
    ext_modules=ext_modules,
)
