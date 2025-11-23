# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import torch
import operator
import torch.fx as fx
import wave_lang.kernel.lang as tkl

from torch._dynamo import register_backend
from torch.fx.node import Node
from typing import List, Dict, Any, Callable, Optional

from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.templates.gemm import get_gemm_kernel
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.wave.utils.torch_utils import device_zeros, device_tensor
from wave_lang.kernel.wave.utils.run_utils import get_default_arch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomBackendConfig:
    """Configuration for the custom backend"""

    def __init__(
        self,
        enable_custom_kernels: bool = True,
        debug_mode: bool = False,
        fallback_to_eager: bool = True,
    ):
        self.enable_custom_kernels = enable_custom_kernels
        self.debug_mode = debug_mode
        self.fallback_to_eager = fallback_to_eager


class GraphAnalyzer:
    """Analyzes the FX graph to identify optimization opportunities"""

    def __init__(self, gm: fx.GraphModule):
        self.gm = gm
        self.graph = gm.graph

    def analyze(self) -> Dict[str, Any]:
        """Perform graph analysis"""
        analysis: Dict[str, Any] = {
            "total_nodes": 0,
            "op_counts": {},
            "optimizable_ops": [],
        }

        for node in self.graph.nodes:
            analysis["total_nodes"] += 1

            # Count operation types
            if node.op == "call_function":
                op_name = self._get_op_name(node)
                op_counts: Dict[str, int] = analysis["op_counts"]
                op_counts[op_name] = op_counts.get(op_name, 0) + 1

                # Identify operations that can be optimized
                if self._is_optimizable(node):
                    optimizable_ops: List[Node] = analysis["optimizable_ops"]
                    optimizable_ops.append(node)

        return analysis

    def _get_op_name(self, node: Node) -> str:
        """Extract operation name from node"""
        if hasattr(node.target, "__name__"):
            return node.target.__name__
        return str(node.target)

    def _is_optimizable(self, node: Node) -> bool:
        """Check if a node can be optimized with custom kernels"""
        # Add your custom logic here
        optimizable_ops = [
            torch.matmul,
            operator.matmul,
        ]

        if node.op == "call_function" and node.target in optimizable_ops:
            return True
        return False


class KernelRegistry:
    """Registry for custom kernel implementations"""

    def __init__(self):
        self._kernels = {}

    def register(self, op_name: str, kernel_fn: Callable):
        """Register a custom kernel for an operation"""
        self._kernels[op_name] = kernel_fn
        logger.info(f"Registered custom kernel for {op_name}")

    def get_kernel(self, op_name: str) -> Optional[Callable]:
        """Get custom kernel for an operation"""
        return self._kernels.get(op_name)

    def has_kernel(self, op_name: str) -> bool:
        """Check if custom kernel exists for an operation"""
        return op_name in self._kernels


# Global kernel registry
kernel_registry = KernelRegistry()


class GraphOptimizer:
    """Optimizes the FX graph by replacing operations with custom kernels"""

    def __init__(self, gm: fx.GraphModule, config: CustomBackendConfig):
        self.gm = gm
        self.graph = gm.graph
        self.config = config
        self.analyzer = GraphAnalyzer(gm)

    def optimize(self) -> fx.GraphModule:
        """Apply optimizations to the graph"""
        logger.info("Starting graph optimization...")

        # Analyze the graph
        analysis = self.analyzer.analyze()
        logger.info(f"Graph analysis: {analysis['op_counts']}")

        if self.config.enable_custom_kernels:
            self._replace_with_custom_kernels(analysis["optimizable_ops"])

        # Recompile the graph after modifications
        self.gm.recompile()
        logger.info("Graph optimization complete")

        return self.gm

    def _replace_with_custom_kernels(self, optimizable_ops: List[Node]):
        """Replace operations with custom kernel implementations"""
        for node in optimizable_ops:
            op_name = self._get_op_name(node)
            custom_kernel = kernel_registry.get_kernel(op_name)

            if custom_kernel is not None:
                logger.info(f"Replacing {op_name} with custom kernel")
                self._replace_node_with_kernel(node, custom_kernel)

    def _replace_node_with_kernel(self, node: Node, kernel_fn: Callable):
        """Replace a node with a custom kernel function"""
        with self.graph.inserting_after(node):
            # Create new node with custom kernel
            new_node = self.graph.call_function(
                kernel_fn, args=node.args, kwargs=node.kwargs
            )

            # Replace all uses of old node with new node
            node.replace_all_uses_with(new_node)

            # Erase the old node
            self.graph.erase_node(node)

    def _get_op_name(self, node: Node) -> str:
        """Extract operation name from node"""
        if hasattr(node.target, "__name__"):
            return node.target.__name__
        return str(node.target)


# ============================================================================
# Wave Kernel Implementations
# ============================================================================


def wave_gemm_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Wave matrix multiplication kernel.
    """
    logger.debug("Executing Wave GEMM kernel")

    shape = (a.shape[0], b.shape[1], a.shape[1])
    dynamic_dims = False
    datatype = torch.float16
    mfma_variant = MMAType.F32_16x16x16_F16
    threads_per_wave = 64

    # TODO(vinayakdsci): Add a branch for `gfx1250` once we add support for it.
    if "gfx120" in get_default_arch():
        mfma_variant = MMAType.RDNA4_WAVE32_F32_16x16x16_F16
        threads_per_wave = 32

    gemm, hyperparams, dynamic_symbols = get_gemm_kernel(
        shape, dynamic_dims, mfma_variant, datatype, threads_per_wave=threads_per_wave
    )

    multibuffer = False
    UNROLL_FACTOR = tkl.sym.UNROLL_FACTOR
    hyperparams[UNROLL_FACTOR] = 2 if multibuffer else 1

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        dynamic_symbols=dynamic_symbols,
    )
    options.postprocess = """
    module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
            %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
            transform.loop.unroll %0 { factor = %%UNROLL_FACTOR%% } : !transform.any_op
            transform.yield
        }
    }
    """
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)

    a_tensor_wave = device_tensor(a.detach().clone(), dtype=datatype)
    b_tensor_wave = device_tensor(b.detach().clone().t(), dtype=datatype)
    c = device_zeros(shape[0], shape[1], dtype=torch.float32)
    gemm(a_tensor_wave, b_tensor_wave, c)

    return c


# Register custom kernels
kernel_registry.register("matmul", wave_gemm_kernel)


# ============================================================================
# Backend Registration
# ============================================================================


def create_backend(config: Optional[CustomBackendConfig] = None):
    """
    Factory function to create a backend with custom configuration.

    Args:
        config: Configuration for the backend

    Returns:
        Backend function that can be registered with torch.compile
    """
    if config is None:
        config = CustomBackendConfig()

    def backend_impl(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
        """
        The actual backend implementation.

        Args:
            gm: FX GraphModule representing the traced computation graph
            example_inputs: Sample inputs for shape/dtype inference

        Returns:
            Callable that executes the optimized graph
        """
        try:
            logger.info(
                f"Custom backend invoked with {len(example_inputs)} example inputs"
            )

            if config.debug_mode:
                logger.info("Graph before optimization:")
                logger.info(gm.graph)

            # Create optimizer and optimize the graph
            optimizer = GraphOptimizer(gm, config)
            optimized_gm = optimizer.optimize()

            if config.debug_mode:
                logger.info("Graph after optimization:")
                logger.info(optimized_gm.graph)

            # Return the forward method of the optimized graph
            return optimized_gm.forward

        except Exception as e:
            logger.error(f"Backend optimization failed: {e}")

            if config.fallback_to_eager:
                logger.warning("Falling back to eager execution")
                return gm.forward
            else:
                raise

    return backend_impl


# Register the Wave backend
@register_backend
def wave(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    Wave custom backend with standard configuration.

    Usage:
        model = torch.compile(my_model, backend="wave")
    """
    config = CustomBackendConfig()
    backend = create_backend(config)
    return backend(gm, example_inputs)
