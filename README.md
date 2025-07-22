# Wave: Python Domain-Specific Language for High Performance Machine Learning

> [!IMPORTANT]
> This repository is a fork of https://github.com/iree-org/iree-turbine. Code is
> currently being migrated from the
> [`iree/turbine/kernel/wave/`](./iree/turbine/kernel/wave/) directory.

## Motivation

Wave is a high-level programming language designed to accelerate the development and optimization of machine learning kernels. It aims to dramatically improve kernel author velocity in two critical dimensions:

### Implementation Velocity
Wave enables rapid prototyping of new optimization ideas and algorithms through its high-level abstractions and symbolic programming model. Kernel authors can quickly express complex tensor operations and experiment with different optimization strategies without getting bogged down in low-level implementation details.

### Performance Velocity
The language is designed to achieve high performance with minimal tuning effort. Through its declarative constraints and hardware-aware abstractions, Wave automatically generates optimized GPU code, allowing kernel authors to focus on algorithmic innovation rather than manual optimization.

Wave is particularly focused on the machine learning domain, with deep integration into the PyTorch ecosystem. This integration enables:

1. Seamless transition from PyTorch models to optimized GPU kernels
2. Easy experimentation with new ML algorithms and optimizations
3. Rapid deployment of custom kernels in production ML pipelines
4. Direct reuse of PyTorch’s tensor abstractions and operator semantics

The language bridges the gap between high-level ML programming and low-level GPU performance by providing a flexible and expressive programming model that maintains close control over hardware resources while enabling rapid innovation in the ML space.

## Core Design Principles

Wave is built around several key design principles that guide its architecture and implementation:

- Symbolic Programming Model
- Heavy use of symbolic variables to represent tensor dimensions
- Symbolic expressions for memory access patterns and layouts
- Enables compile-time optimization and analysis
- Provides flexibility in expressing complex tensor operations
- Separation of Distribution and Computation
- Clear separation between distribution strategy and computation graph
- Distribution strategy defined through declarative constraints
- Computation graph expressed independently of parallelization
- Enables better optimization and code reuse
- Dual Programming Models
- Support for both tile-based and SIMT-based programming models
- Tile-based model for coarse-grained parallelism
- SIMT model for fine-grained vector operations
- Flexible mapping to different GPU architectures
- Hardware-Aware Abstractions
- Direct mapping to GPU hardware concepts (workgroups, waves, etc.)
- Explicit control over memory hierarchy
- Hardware-specific optimizations (MMA operations, memory layouts)
- Performance portability across different GPU architectures

For more information, visit the documentation here: https://wave-lang.readthedocs.io/en/latest/wave/wave.html

## Quickstart

Wave supports both end users (who want to use Wave kernels in their ML workflows) and developers (who want to contribute to or extend Wave). Follow the relevant guide below to get started quickly.

### For Users

1. **Install ROCm PyTorch Dependencies**

   Before installing Wave, ensure you have the appropriate ROCm-enabled PyTorch dependencies:

   ```bash
   pip install -r pytorch-rocm-requirements.txt
   ```

2. **Install Wave**

   You can then install Wave and its dependencies using pip:

   ```bash
   pip install wave-lang
   ```

3. **Try the Example Notebook**

   To get started, try the [examples/jupyter/wave_gemm_example.ipynb](examples/jupyter/wave_gemm_example.ipynb) notebook.


### For Developers

1. **Clone the Repository**

   ```bash
   git clone https://github.com/iree-org/wave.git
   cd wave
   ```

2. **Install Development Dependencies**

   It's recommended to use a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements-iree-pinned.txt
   pip install -r pytorch-rocm-requirements.txt
   pip install -r requirements.txt -e .
   ```

   > **Note:** If you do not have access to Instinct GPUs, you can still install Wave as above but with the CPU version of PyTorch:
   >
   > ```bash
   > pip install -r pytorch-cpu-requirements.txt
   > ```
   >
   > Currently, you can only run lit tests in this mode.


3. **Run Tests**

   To verify your setup:

   ```bash
   # Python tests
   pytest -s tests/
   # To run end to end tests, add --run-e2e flag

   # Lit tests
   lit lit_tests/ -v
   ```

4. **Contributing**

   Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to Wave.
