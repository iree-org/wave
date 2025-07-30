Getting Started with Wave
=========================

This guide will help you get up and running with Wave, a high-performance machine learning programming language designed for accelerating ML kernel development.

Prerequisites
--------------

Before installing Wave, ensure you have the following prerequisites:

1. Python 3.10 or later
2. PyTorch
3. ROCm (for AMD GPU support)
4. A compatible AMD GPU with ROCm support (MI250, MI300, etc.)

Quickstart
--------------

Wave supports both end users (who want to use Wave kernels in their ML workflows) and developers (who want to contribute to or extend Wave). Follow the relevant guide below to get started quickly.

For Users
~~~~~~~~~

1. **Install ROCm PyTorch Dependencies**

   Before installing Wave, ensure you have the appropriate ROCm-enabled PyTorch dependencies:

   .. code-block:: bash

      pip install -r pytorch-rocm-requirements.txt

2. **Install Wave**

   You can then install Wave and its dependencies using pip:

   .. code-block:: bash

      pip install wave-lang

For Developers
~~~~~~~~~

1. **Clone the Repository**

   .. code-block:: bash

      git clone https://github.com/iree-org/wave.git
      cd wave

2. **Install Development Dependencies**

   If you don't have Rust already installed on your system, install Rust using the following command:

   .. code-block:: bash

      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

   Next, install the python dependencies. It is recommended to use a virtual environment.

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate
      pip install --upgrade pip
      pip install -r requirements-iree-pinned.txt
      pip install -r pytorch-rocm-requirements.txt
      pip install -r requirements.txt -e .

   *Note:*

      If you do not have access to Instinct GPUs, you can still install Wave as above but with the CPU version of PyTorch:

      .. code-block:: bash

        pip install -r pytorch-cpu-requirements.txt

      Currently, you can only run lit tests in this mode.

3. **Run Tests**

   To verify your setup:

   .. code-block:: bash

      # Python tests
      pytest -s tests/
      # To run end to end tests, add --run-e2e flag
      # Lit tests
      lit lit_tests/ -v

Next Steps
-----------

- Read the :doc:`system_architecture` guide to understand Wave's compilation pipeline
- Check out the :doc:`gemm_tutorial` for a more complex example
- Explore :doc:`shared_memory` for optimization techniques
- Learn about the :doc:`runtime` for advanced usage

For more detailed information about Wave's architecture and optimization passes, see the :doc:`system_architecture` documentation.
