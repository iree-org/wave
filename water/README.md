# Water 💧

Water forms waves 🌊, and hosts sharks 🦈 and other eerie creatures 👻.

Water is a collection of [MLIR](https://mlir.llvm.org) components including
passes, dialects and interfaces usable in other projects.

## Building

### Component Build

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR`.

```sh
cmake -G Ninja \
      -B build \
      -DMLIR_DIR=$BUILD_DIR/lib/cmake/mlir \
      -DBUILD_SHARED_LIBS=On \
      -DPython3_EXECUTABLE="$(which python)" \
      -DWATER_ENABLE_PYTHON=ON
cmake --build build --target check-water
```

### Monolithic Build

This setup assumes that you build the project as part of a monolithic LLVM
build via the `LLVM_EXTERNAL_PROJECTS` mechanism.  To build LLVM, MLIR, the
example and launch the tests run

```sh
cmake -G Ninja \
      -B build \
      $LLVM_SRC_DIR/llvm \
      -DLLVM_TARGETS_TO_BUILD=host \
      -DLLVM_ENABLE_PROJECTS=mlir \
      -DLLVM_EXTERNAL_PROJECTS=water \
      -DLLVM_EXTERNAL_WATER_SOURCE_DIR=$PWD \
      -DBUILD_SHARED_LIBS=On \
      -DPython3_EXECUTABLE="$(which python)" \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DWATER_ENABLE_PYTHON=ON
cmake --build build --target check-water
```

Here, `$LLVM_SRC_DIR` needs to point to the root of the monorepo.

### Useful Cmake Flags

* `-DCMAKE_BUILD_TYPE=RelWithDebInfo` - build with optimizations but preserve
  debug information.
* `-DBUILD_SHARED_LIBS=Off` - build static libraries, increases build time and
  **may lead to missing library dependencies**, may be necessary to avoid LLVM
  version conflicts in binary distributions.
* `-DCMAKE_C_COMPILER_LAUNCHER=ccache` - use ccache for C files.
* `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache` - use ccache for C++ files.
* `-DLLVM_PARALLEL_LINK_JOBS=1` - disable parallel linking, helps with build
  times and out-of-memory errors on low-memory machines.
* `-DCMAKE_EXPORT_COMPILE_COMMANDS=On` - generate `compile_commands.json` used
  by IDEs to discover the source file tree.
* `-DWATER_ENABLE_PYTHON=ON` - enable building Python bindings.

### Python Wheel

A Python wheel containing the `water-opt` binary can be produced using the
following commands:

```sh
cd build_tools/wheel
WATER_MLIR_DIR=$BUILD_DIR/lib/cmake/mlir python -m pip wheel .
```

This will produce a `.whl` file containing the `water-opt` and metadata. It can
be installed using:

```sh
pip install --force-reinstall *.whl
```

Note that force-reinstallation flag without which the wheel may not be
reinstalled without bumping the version.

The package build can be configured to checkout and build LLVM from source by
setting the `WATER_BUILD_LLVM` environment variable to `1`:

```sh
cd build_tools/wheel
WATER_BUILD_LLVM=1 python -m pip wheel .
```

The build mode is controlled by the `WATER_BUILD_TYPE` environment variable
whose content is forward to cmake.

An editable build allows you to modify the python code and have the changes
immediately reflected without reinstalling the package. This is useful during
development. To install an editable build:

```sh
cd build_tools/wheel
WATER_MLIR_DIR=$BUILD_DIR/lib/cmake/mlir python -m pip install -e .
```

Note that editable builds still require rebuilding native parts when they are
modified.

## Developing

### Pre-commit

Please use pre-commit by installing it from
[pre-commit.com](https://pre-commit.com) or system repository and running

```sh
pre-commit install
pre-commit install --hook-type commit-msg
```

in the repository once. After that, every further commit will run through the
pre-commit checks such as formatters and linters. If any problems are found,
please fix them and amend the commit before pushing.

Pre-commit hooks require `black` and `gem` to be available on the system.

### Certificate of Origin

Please follow the [Developer Certificate of
Origin](https://wiki.linuxfoundation.org/dco) policy by signing off the
commits, e.g., use `git commit -s` to automatically add the required field.
