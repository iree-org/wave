# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import distutils.command.build
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from setuptools import Extension, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools_rust import RustExtension

THIS_DIR = os.path.realpath(os.path.dirname(__file__))
REPO_ROOT = THIS_DIR
WAVE_IS_STABLE_REL = int(os.getenv("WAVE_IS_STABLE_REL", "0"))
BUILD_TYPE = os.environ.get("WAVE_BUILD_TYPE", "Release")
BUILD_WATER = int(os.environ.get("WAVE_BUILD_WATER", "0"))
LLVM_DIR = os.getenv("WAVE_LLVM_DIR")
LLVM_REPO = os.getenv("WAVE_LLVM_REPO", "https://github.com/llvm/llvm-project.git")
BUILD_SHARED_LIBS = os.getenv("WAVE_LLVM_BUILD_SHARED_LIBS", "OFF")


class CMakeExtension(Extension):
    def __init__(
        self,
        name: str,
        sourcedir: str,
        install_dir: Optional[str] = None,
        need_llvm: bool = False,
        cmake_args: Optional[list[str]] = None,
    ) -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())
        self.install_dir = install_dir
        self.need_llvm = need_llvm
        self.cmake_args = cmake_args or []


def invoke_cmake(*args, cwd=None):
    subprocess.check_call(["cmake", *args], cwd=cwd)


def invoke_git(*args, cwd=None):
    subprocess.check_call(["git", *args], cwd=cwd)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        # Create build directory
        build_dir = os.path.abspath(os.path.join(self.build_temp, ext.name))
        os.makedirs(build_dir, exist_ok=True)

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

        # Configure CMake
        cmake_args = [
            "-G",
            "Ninja",
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
                    # Extract SHA from: #define LLVM_REVISION "478e45fb..."
                    installed_sha = line.split('"')[1]
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

        print(f"Removing old LLVM installation...")
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


VERSION_FILE = os.path.join(REPO_ROOT, "version.json")
VERSION_FILE_LOCAL = os.path.join(REPO_ROOT, "version_local.json")


def load_version_info(version_file):
    with open(version_file, "rt") as f:
        return json.load(f)


try:
    version_info = load_version_info(VERSION_FILE_LOCAL)
except FileNotFoundError:
    print("version_local.json not found. Default to dev build")
    version_info = load_version_info(VERSION_FILE)

PACKAGE_VERSION = version_info["package-version"]
print(f"Using PACKAGE_VERSION: '{PACKAGE_VERSION}'")

with open(os.path.join(REPO_ROOT, "README.md"), "rt") as f:
    README = f.read()

packages = find_namespace_packages(
    include=[
        "wave_lang",
        "wave_lang.*",
    ],
)

print("Found packages:", packages)

# Lookup version pins from requirements files.
requirement_pins = {}


def load_requirement_pins(requirements_file: str):
    with open(Path(THIS_DIR) / requirements_file, "rt") as f:
        lines = f.readlines()
    pin_pairs = [line.strip().split("==") for line in lines if "==" in line]
    requirement_pins.update(dict(pin_pairs))


if WAVE_IS_STABLE_REL:
    load_requirement_pins("requirements-iree-stable.txt")


def get_version_spec(dep: str):
    if dep in requirement_pins:
        return f"=={requirement_pins[dep]}"
    else:
        return ""


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

if BUILD_WATER:
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
        ),
    ]

setup(
    name="wave-lang",
    version=f"{PACKAGE_VERSION}",
    author="IREE Authors",
    author_email="iree-technical-discussion@lists.lfaidata.foundation",
    description="Wave Language for Machine Learning",
    long_description=README,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    project_urls={
        "homepage": "https://iree.dev/",
        "repository": "https://github.com/iree-org/wave/",
        "documentation": "https://wave.readthedocs.io/en/latest/",
    },
    packages=packages,
    include_package_data=True,
    package_data={},
    entry_points={
        "torch_dynamo_backends": [],
    },
    install_requires=[
        f"numpy{get_version_spec('numpy')}",
        f"iree-base-compiler{get_version_spec('iree-base-compiler')}",
        f"iree-base-runtime{get_version_spec('iree-base-runtime')}",
        f"Jinja2{get_version_spec('Jinja2')}",
        f"ml_dtypes{get_version_spec('ml_dtypes')}",
        f"typing_extensions{get_version_spec('typing_extensions')}",
    ],
    extras_require={
        "testing": [
            f"pytest{get_version_spec('pytest')}",
            f"pytest-xdist{get_version_spec('pytest-xdist')}",
        ],
    },
    cmdclass={"build": BuildCommand, "build_ext": CMakeBuild},
    ext_modules=ext_modules,
    rust_extensions=[
        RustExtension("aplp_lib", "wave_lang/kernel/wave/scheduling/aplp/Cargo.toml")
    ],
    zip_safe=False,
)
