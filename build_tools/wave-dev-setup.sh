#!/usr/bin/env bash
# wave-dev-setup.sh -- Wave development environment setup and build tool
#
# ============================================================================
# ASSUMPTIONS
# ============================================================================
#
# This script assumes the following default directory layout, where WAVE_REPO
# is the **main** Wave checkout (not a worktree):
#
#   WAVE_REPO/                        The main Wave git repository
#   WAVE_REPO/../llvm-wave/           LLVM base directory (WAVE_LLVM_DIR):
#     llvm-project/                     LLVM source (cloned from git)
#     build/                            LLVM CMake build directory
#     install/                          LLVM CMake install directory
#
# All paths to directories outside the Wave repo are resolved relative to
# the main repository root, NOT relative to the current worktree.  This
# means worktrees can live anywhere and still find shared dependencies.
#
# The script auto-detects whether it is running inside a git worktree and
# resolves the main repo root accordingly.
#
# The script uses a virtualenv at `$WAVE_DIR/.venv`.  Each worktree will have a
# different virtualenv.  The script uses `uv virtualenv` to reduce install
# footprint -- `uv pip install` uses hard links to `$UV_CACHE_DIR`, so a big
# download/install like pytorch is not duplicated, and worktree installs are
# fast.
#
# The script mainly uses default build setups, and is intended to streamline
# environment setups, especially for worktrees, and especially for agents, to
# have a single command to get running.  It builds water and waveasm always.
#
# ============================================================================
# ENVIRONMENT VARIABLES (all optional)
# ============================================================================
#
# Paths -- override these to use non-default locations:
#
#   WAVE_DIR           Current Wave working directory (repo root or worktree).
#                      Default: current working directory.
#
#   WAVE_LLVM_DIR      LLVM base directory containing source, build, and install.
#                      Default: <MAIN_REPO>/../llvm-wave
#
#   LLVM_SRC           LLVM source clone.
#                      Default: $WAVE_LLVM_DIR/llvm-project
#
#   LLVM_BUILD         LLVM CMake build directory.
#                      Default: $WAVE_LLVM_DIR/build
#
#   LLVM_INSTALL       LLVM CMake install prefix.
#                      Default: $WAVE_LLVM_DIR/install
#
#   WATER_SRC          Water dialect source directory.
#                      Default: $WAVE_DIR/water
#
#   WATER_BUILD        Water CMake build directory.
#                      Default: $WAVE_DIR/water/build
#
#   WAVEASM_SRC        WaveASM source directory.
#                      Default: $WAVE_DIR/waveasm
#
#   WAVEASM_BUILD      WaveASM CMake build directory.
#                      Default: $WAVE_DIR/waveasm/build
#
#   LLVM_SHA_FILE      File containing the pinned LLVM commit SHA.
#                      Default: $WAVE_DIR/water/llvm-sha.txt
#
# Build options:
#
#   WAVE_LLVM_REPO                Git URL for LLVM.
#                                 Default: https://github.com/llvm/llvm-project.git
#
#   WAVE_LLVM_BUILD_SHARED_LIBS   ON or OFF for CMake BUILD_SHARED_LIBS.
#                                 Default: OFF
#
#   WAVE_BUILD_TYPE               CMake build type (Release, Debug, etc.).
#                                 Default: Release
#
#   WAVE_LLVM_JOBS                Parallel build jobs.
#                                 Default: $(nproc)
#
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve WAVE_DIR and the main repo root
# ---------------------------------------------------------------------------

WAVE_DIR="${WAVE_DIR:-$PWD}"

# Resolve the main (non-worktree) repo root.  git rev-parse --git-common-dir
# gives the main repo's .git directory even when run from a worktree.
resolve_main_repo_root() {
    local git_common_dir
    git_common_dir="$(git -C "$WAVE_DIR" rev-parse --git-common-dir 2>/dev/null)" || {
        echo "error: $WAVE_DIR does not appear to be inside a git repository" >&2
        exit 1
    }
    # git-common-dir may be relative; resolve it.
    git_common_dir="$(cd "$WAVE_DIR" && cd "$git_common_dir" && pwd)"
    # The repo root is the parent of .git
    dirname "$git_common_dir"
}

MAIN_REPO_ROOT="$(resolve_main_repo_root)"

# ---------------------------------------------------------------------------
# Default paths -- external deps relative to MAIN_REPO_ROOT, not WAVE_DIR
# ---------------------------------------------------------------------------

WAVE_LLVM_DIR="${WAVE_LLVM_DIR:-$MAIN_REPO_ROOT/../llvm-wave}"
LLVM_SRC="${LLVM_SRC:-$WAVE_LLVM_DIR/llvm-project}"
LLVM_BUILD="${LLVM_BUILD:-$WAVE_LLVM_DIR/build}"
LLVM_INSTALL="${LLVM_INSTALL:-$WAVE_LLVM_DIR/install}"
WATER_SRC="${WATER_SRC:-$WAVE_DIR/water}"
WATER_BUILD="${WATER_BUILD:-$WAVE_DIR/water/build}"
WAVEASM_SRC="${WAVEASM_SRC:-$WAVE_DIR/waveasm}"
WAVEASM_BUILD="${WAVEASM_BUILD:-$WAVE_DIR/waveasm/build}"
LLVM_SHA_FILE="${LLVM_SHA_FILE:-$WAVE_DIR/water/llvm-sha.txt}"

LLVM_REPO="${WAVE_LLVM_REPO:-https://github.com/llvm/llvm-project.git}"
BUILD_SHARED_LIBS="${WAVE_LLVM_BUILD_SHARED_LIBS:-OFF}"
BUILD_TYPE="${WAVE_BUILD_TYPE:-Release}"
JOBS="${WAVE_LLVM_JOBS:-$(nproc)}"

LLVM_DIR="$LLVM_BUILD/lib/cmake/llvm"
MLIR_DIR="$LLVM_BUILD/lib/cmake/mlir"
LLD_DIR="$LLVM_BUILD/lib/cmake/lld"

git_clone_depth=10
VERBOSE=0

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
wave-dev-setup.sh -- Wave development environment setup and build tool

Usage:
  --worktree-initialize  Create a virtualenv in \$WAVE_DIR/.venv, install
                         all Python and C++ dependencies, and build Water
                         and WaveASM.  Suitable for both the main checkout
                         and worktrees.  Builds silently unless there are
                         errors.  Prints exports on completion.
  --print-wave-env       Print shell exports for Wave development.

  --build-llvm           Clone (if needed), configure, and build LLVM/MLIR.
  --build-water          Configure and build the Water MLIR dialect.
  --build-waveasm        Configure and build the WaveASM backend.
  --build-all            Build LLVM, Water, and WaveASM (in order).
  --clean-llvm           Remove LLVM build and install directories.
  --clean-water          Remove Water build directory.
  --clean-waveasm        Remove WaveASM build directory.
  --clean-all            Remove all build directories.
  --checkout-llvm-sha    Check out the pinned LLVM SHA (from
                         water/llvm-sha.txt) in the local LLVM clone.
                         Does not modify the pin file itself.
  --test-wave-lit        Run Wave lit tests.
  --test-wave-e2e        Run Wave Python e2e tests (pytest --run-e2e).
  --test-water-lit       Run Water lit tests (ninja check-water).
  --test-waveasm-lit     Run WaveASM lit tests (ninja check-waveasm).
  --test-waveasm-e2e     Run WaveASM Python e2e tests (pytest --run-e2e).
  --verbose              Show full build output in worktree-initialize
                         (default: quiet, log on error).
  --help                 Show this help message.

Arguments after -- are passed to pytest (for e2e test commands):
  wave-dev-setup.sh --test-wave-e2e -- -k test_gemm

Multiple flags can be combined, e.g.:
  wave-dev-setup.sh --build-water --test-waveasm-lit

Re-running build commands skips already-completed steps (clone, cmake configure).

Resolved directories:
  Main repo root: $MAIN_REPO_ROOT
  WAVE_DIR:       $WAVE_DIR
  WAVE_LLVM_DIR:  $WAVE_LLVM_DIR
  LLVM source:    $LLVM_SRC
  LLVM build:     $LLVM_BUILD
  LLVM install:   $LLVM_INSTALL
  Water source:   $WATER_SRC
  Water build:    $WATER_BUILD
  WaveASM source: $WAVEASM_SRC
  WaveASM build:  $WAVEASM_BUILD

See the header of this script for all environment variables.
EOF
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

read_pinned_sha() {
    if [[ ! -f "$LLVM_SHA_FILE" ]]; then
        echo "error: $LLVM_SHA_FILE not found" >&2
        exit 1
    fi
    tr -d '[:space:]' < "$LLVM_SHA_FILE"
}

do_checkout() {
    local sha="$1"
    echo "Checking out LLVM commit $sha ..."
    git -C "$LLVM_SRC" fetch --depth "$git_clone_depth" origin "$sha"
    git -C "$LLVM_SRC" checkout "$sha"
}

require_llvm_install() {
    if [[ ! -d "$LLVM_INSTALL/lib/cmake/mlir" ]]; then
        echo "error: LLVM install not found at $LLVM_INSTALL" >&2
        echo "Run --build-llvm first." >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

do_checkout_llvm_sha() {
    local sha
    sha="$(read_pinned_sha)"
    if [[ ! -d "$LLVM_SRC/.git" ]]; then
        echo "error: no existing clone at $LLVM_SRC" >&2
        echo "Run --build-llvm first to create the clone." >&2
        exit 1
    fi
    do_checkout "$sha"
    echo "Done.  LLVM source is now at $sha"
}

do_clean_llvm() {
    echo "Removing LLVM build directory: $LLVM_BUILD"
    rm -rf "$LLVM_BUILD"
    echo "Removing LLVM install directory: $LLVM_INSTALL"
    rm -rf "$LLVM_INSTALL"
}

do_clean_water() {
    echo "Removing Water build directory: $WATER_BUILD"
    rm -rf "$WATER_BUILD"
}

do_clean_waveasm() {
    echo "Removing WaveASM build directory: $WAVEASM_BUILD"
    rm -rf "$WAVEASM_BUILD"
}

do_build_llvm() {
    local sha
    sha="$(read_pinned_sha)"

    if [[ ! -d "$LLVM_SRC/.git" ]]; then
        echo "Cloning LLVM to $LLVM_SRC ..."
        mkdir -p "$LLVM_SRC"
        git clone --depth "$git_clone_depth" --no-checkout "$LLVM_REPO" "$LLVM_SRC"
        do_checkout "$sha"
    fi

    if [[ ! -f "$LLVM_BUILD/build.ninja" ]]; then
        echo "Configuring LLVM (build type: $BUILD_TYPE) ..."
        mkdir -p "$LLVM_BUILD"
        cmake -G Ninja \
            -S "$LLVM_SRC/llvm" \
            -B "$LLVM_BUILD" \
            -DLLVM_TARGETS_TO_BUILD="host;AMDGPU" \
            -DLLVM_ENABLE_PROJECTS="llvm;mlir;lld;clang" \
            -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
            -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS" \
            -DLLVM_ENABLE_ASSERTIONS=ON \
            -DLLVM_ENABLE_ZSTD=OFF \
            -DLLVM_INSTALL_UTILS=ON \
            -DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON \
            -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL" \
            -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
            -DPython3_EXECUTABLE="$(which python3)"
    else
        echo "LLVM build directory already configured, skipping cmake."
    fi

    echo "Building LLVM with $JOBS parallel jobs ..."
    cmake --build "$LLVM_BUILD" --target install -- -j"$JOBS"

    echo ""
    echo "LLVM installed to: $LLVM_INSTALL"
}

do_build_water() {
    require_llvm_install

    if [[ ! -f "$WATER_BUILD/build.ninja" ]]; then
        echo "Configuring Water (build type: $BUILD_TYPE) ..."
        mkdir -p "$WATER_BUILD"
        cmake -G Ninja \
            -S "$WATER_SRC" \
            -B "$WATER_BUILD" \
            -DLLVM_DIR="$LLVM_DIR" \
            -DMLIR_DIR="$MLIR_DIR" \
            -DLLD_DIR="$LLD_DIR" \
            -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS" \
            -DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON \
            -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
            -DWATER_ENABLE_PYTHON=ON \
            -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
            -DPython3_EXECUTABLE="$(which python3)"
    else
        echo "Water build directory already configured, skipping cmake."
    fi

    echo "Building Water with $JOBS parallel jobs ..."
    cmake --build "$WATER_BUILD" -- -j"$JOBS"

    echo ""
    echo "Water built in: $WATER_BUILD"
}

do_build_waveasm() {
    require_llvm_install

    if [[ ! -f "$WAVEASM_BUILD/build.ninja" ]]; then
        echo "Configuring WaveASM (build type: $BUILD_TYPE) ..."
        mkdir -p "$WAVEASM_BUILD"
        cmake -G Ninja \
            -S "$WAVEASM_SRC" \
            -B "$WAVEASM_BUILD" \
            -DLLVM_DIR="$LLVM_DIR" \
            -DMLIR_DIR="$MLIR_DIR" \
            -DLLD_DIR="$LLD_DIR" \
            -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS" \
            -DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON \
            -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
            -DPython3_EXECUTABLE="$(which python3)"
    else
        echo "WaveASM build directory already configured, skipping cmake."
    fi

    echo "Building WaveASM with $JOBS parallel jobs ..."
    cmake --build "$WAVEASM_BUILD" -- -j"$JOBS"

    echo ""
    echo "WaveASM built in: $WAVEASM_BUILD"
}

do_test_waveasm_lit() {
    if [[ ! -f "$WAVEASM_BUILD/build.ninja" ]]; then
        echo "error: WaveASM not built.  Run --build-waveasm first." >&2
        exit 1
    fi
    echo "Running WaveASM lit tests ..."
    cmake --build "$WAVEASM_BUILD" --target check-waveasm
}

do_test_waveasm_e2e() {
    echo "Running WaveASM Python e2e tests ..."
    WAVE_CACHE_ON=0 \
    PYTHONPATH="$WAVE_DIR" \
        python3 -m pytest --run-e2e "$WAVE_DIR/tests/kernel/wave/asm/test_waveasm_e2e.py" -v \
        "${PASSTHROUGH_ARGS[@]}"
}

do_test_water_lit() {
    if [[ ! -f "$WATER_BUILD/build.ninja" ]]; then
        echo "error: Water not built.  Run --build-water first." >&2
        exit 1
    fi
    echo "Running Water lit tests ..."
    cmake --build "$WATER_BUILD" --target check-water
}

do_test_wave_lit() {
    echo "Running Wave lit tests ..."
    WAVE_CACHE_ON=0 \
    PYTHONPATH="$WAVE_DIR" \
        lit -v "$WAVE_DIR/lit_tests" \
        "${PASSTHROUGH_ARGS[@]}"
}

do_test_wave_e2e() {
    echo "Running Wave Python e2e tests ..."
    WAVE_CACHE_ON=0 \
    PYTHONPATH="$WAVE_DIR" \
        python3 -m pytest --run-e2e "$WAVE_DIR/tests" -v \
        "${PASSTHROUGH_ARGS[@]}"
}

do_print_wave_env() {
    cat <<EOF
# Copy this into your shell to use the wave environment.
source $WAVE_DIR/.venv/bin/activate
export WAVE_LLVM_DIR="$LLVM_INSTALL"
export WAVE_WATER_DIR="$WATER_BUILD"
export WAVE_WAVEASM_DIR="$WAVEASM_BUILD"
export WAVE_CACHE_ON=0
export PYTHONPATH="$WAVE_DIR"
EOF
}

do_worktree_initialize() {
    local venv_dir="$WAVE_DIR/.venv"

    run_init() {
        set -euo pipefail
        pytorch_req="$(mktemp /tmp/pytorch-rocm-requirements.XXXXXX.txt)"
        trap "rm -f '$pytorch_req'" EXIT
        uv venv "$venv_dir"
        source "$venv_dir/bin/activate"
        "$WAVE_DIR/gen-pytorch-rocm-requirements.py" -o "$pytorch_req"
        uv pip install -r "$pytorch_req"
        uv pip install -r "$WAVE_DIR/requirements-iree-pinned.txt"
        uv pip install nanobind
        do_build_waveasm
        do_build_water
        WAVE_LLVM_DIR="$LLVM_INSTALL" \
        WAVE_WAVEASM_DIR="$WAVEASM_BUILD" \
        WAVE_WATER_DIR="$WATER_BUILD" \
            uv pip install -e "$WAVE_DIR[dev]"
    }

    if [[ "$VERBOSE" -eq 1 ]]; then
        (run_init)
    else
        local log_file
        log_file="$(mktemp)"
        if ! (run_init) > "$log_file" 2>&1; then
            echo "error: worktree initialization failed. Output:" >&2
            cat "$log_file" >&2
            rm -f "$log_file"
            exit 1
        fi
        rm -f "$log_file"
    fi

    do_print_wave_env
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

if [[ $# -eq 0 ]]; then
    usage
    exit 0
fi

error=0
ACTIONS=()
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --)
            shift
            PASSTHROUGH_ARGS=("$@")
            break
            ;;
        --worktree-initialize)
            ACTIONS+=(worktree-initialize)
            shift
            ;;
        --build-llvm)
            ACTIONS+=(build-llvm)
            shift
            ;;
        --build-water)
            ACTIONS+=(build-water)
            shift
            ;;
        --build-waveasm)
            ACTIONS+=(build-waveasm)
            shift
            ;;
        --build-all)
            ACTIONS+=(build-llvm build-water build-waveasm)
            shift
            ;;
        --clean-llvm)
            ACTIONS+=(clean-llvm)
            shift
            ;;
        --clean-water)
            ACTIONS+=(clean-water)
            shift
            ;;
        --clean-waveasm)
            ACTIONS+=(clean-waveasm)
            shift
            ;;
        --clean-all)
            ACTIONS+=(clean-llvm clean-water clean-waveasm)
            shift
            ;;
        --checkout-llvm-sha)
            ACTIONS+=(checkout-llvm-sha)
            shift
            ;;
        --test-wave-lit)
            ACTIONS+=(test-wave-lit)
            shift
            ;;
        --test-wave-e2e)
            ACTIONS+=(test-wave-e2e)
            shift
            ;;
        --test-water-lit)
            ACTIONS+=(test-water-lit)
            shift
            ;;
        --test-waveasm-lit)
            ACTIONS+=(test-waveasm-lit)
            shift
            ;;
        --test-waveasm-e2e)
            ACTIONS+=(test-waveasm-e2e)
            shift
            ;;
        --print-wave-env)
            ACTIONS+=(print-wave-env)
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            echo "Run with --help for usage." >&2
            error=1
            shift
            ;;
    esac
done

if [[ "$error" -eq 1 ]]; then
    exit 1
fi

if [[ ${#ACTIONS[@]} -eq 0 ]]; then
    usage
    exit 0
fi

for action in "${ACTIONS[@]}"; do
    case "$action" in
        worktree-initialize) do_worktree_initialize ;;
        clean-llvm)          do_clean_llvm ;;
        clean-water)         do_clean_water ;;
        clean-waveasm)       do_clean_waveasm ;;
        build-llvm)          do_build_llvm ;;
        build-water)         do_build_water ;;
        build-waveasm)       do_build_waveasm ;;
        checkout-llvm-sha)   do_checkout_llvm_sha ;;
        test-wave-lit)       do_test_wave_lit ;;
        test-wave-e2e)       do_test_wave_e2e ;;
        test-water-lit)      do_test_water_lit ;;
        test-waveasm-lit)    do_test_waveasm_lit ;;
        test-waveasm-e2e)    do_test_waveasm_e2e ;;
        print-wave-env)      do_print_wave_env ;;
    esac
done
