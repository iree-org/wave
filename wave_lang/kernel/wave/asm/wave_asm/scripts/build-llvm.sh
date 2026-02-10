#!/bin/bash
# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build LLVM/MLIR at the pinned SHA for wave-asm
#
# Usage: ./scripts/build-llvm.sh [install_dir]
#
# This script clones LLVM at the pinned SHA and builds it with the
# configuration needed for wave-asm.

set -e

# Pinned LLVM SHA - must match CMakeLists.txt
LLVM_SHA="c75d371f57608b01bfee12092e707c99124b5341"

# Installation directory
INSTALL_DIR="${1:-$HOME/llvm-amdasm}"
BUILD_DIR="${INSTALL_DIR}/build"
SRC_DIR="${INSTALL_DIR}/llvm-project"

echo "Building LLVM at SHA: ${LLVM_SHA}"
echo "Install directory: ${INSTALL_DIR}"

# Create directories
mkdir -p "${INSTALL_DIR}"

# Clone or update LLVM
if [ -d "${SRC_DIR}" ]; then
    echo "LLVM source already exists, fetching updates..."
    cd "${SRC_DIR}"
    git fetch origin
else
    echo "Cloning LLVM..."
    git clone --depth 1 https://github.com/llvm/llvm-project.git "${SRC_DIR}"
    cd "${SRC_DIR}"
    git fetch --depth=1 origin "${LLVM_SHA}"
fi

# Checkout the pinned SHA
echo "Checking out SHA: ${LLVM_SHA}"
git checkout "${LLVM_SHA}"

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure LLVM with MLIR and necessary targets
echo "Configuring LLVM..."
cmake -G Ninja "${SRC_DIR}/llvm" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}/install" \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

# Build
echo "Building LLVM (this may take a while)..."
ninja

echo ""
echo "LLVM build complete!"
echo ""
echo "To build wave-asm with this LLVM, run:"
echo "  cd wave-asm/build"
echo "  cmake -G Ninja -DMLIR_DIR=${BUILD_DIR}/lib/cmake/mlir .."
echo "  ninja"
