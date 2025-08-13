# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import torch
from torch import Tensor

from iree.runtime import (
    HalBufferView,
    HalDevice,
    HalElementType,
    HalFence,
    VmContext,
    VmFunction,
    VmInstance,
    VmModule,
    VmRef,
    VmVariantList,
    create_hal_module,
    get_driver,
)

from .device import (
    Device,
    get_device_from_torch,
)

__all__ = [
    "MultiDeviceLaunchable",
]


class MultiDeviceLaunchable:
    """
    A simplified launchable that supports multi-device execution similar to iree-run-module.

    Key differences from original Launchable:
    - Simpler API focused on core functionality
    - Supports multiple devices from the start
    """

    def __init__(
        self,
        vmfb_path_or_bytes: Union[str, Path, bytes],
        devices: Optional[List[Union[str, Device]]] = None,
        function_name: str = "main",
        runtime_flags: Optional[List[str]] = None,
        is_async: bool = True,
    ):
        """
        Initialize a multi-device launchable.

        Args:
            vmfb_path_or_bytes: Path to VMFB file or raw VMFB bytes
            devices: List of devices (e.g., ["hip://0", "hip://1"]) or Device objects
            function_name: Name of function to execute (default: "main")
            runtime_flags: Runtime flags (e.g., ["--hip_use_streams=true"])
        """
        self.function_name = function_name
        self.runtime_flags = runtime_flags or []

        # Load VMFB data
        if isinstance(vmfb_path_or_bytes, bytes):
            self.vmfb_data = vmfb_path_or_bytes
        else:
            with open(vmfb_path_or_bytes, "rb") as f:
                self.vmfb_data = f.read()

        # Initialize devices if provided
        self._devices = devices
        self._vm_context: Optional[VmContext] = None
        self._vm_function: Optional[VmFunction] = None
        self._hal_devices: Optional[List[HalDevice]] = None
        self._is_async = is_async

    def _initialize_runtime(self, inferred_device: Optional[Device] = None):
        """Lazily initialize the runtime with devices."""
        if self._vm_context is not None:
            return

        # Determine devices to use
        if self._devices:
            devices = self._devices
        elif inferred_device:
            devices = [inferred_device]
        else:
            raise RuntimeError(
                "No devices specified and none could be inferred from inputs"
            )

        # Create VM instance
        vm_instance = VmInstance()

        # Create HAL devices
        self._hal_devices = []
        hal_drivers = {}

        for device in devices:
            if isinstance(device, str):
                # Parse device string (e.g., "hip://0")
                driver_name, device_id = device.split("://")
                if driver_name not in hal_drivers:
                    hal_drivers[driver_name] = get_driver(driver_name)

                hal_device = hal_drivers[driver_name].create_device_by_uri(
                    device, allocators=[]
                )
            else:
                # Assume it's a Device object
                hal_device = device.hal_device

            self._hal_devices.append(hal_device)

        # Create HAL module with all devices
        hal_module = create_hal_module(vm_instance, devices=self._hal_devices)

        # Load main module
        main_module = VmModule.copy_buffer(vm_instance, self.vmfb_data)

        # Create context with both modules
        modules = [hal_module, main_module]
        self._vm_context = VmContext(vm_instance, modules)
        self._vm_function = main_module.lookup_function(self.function_name)

    def __call__(
        self, *args, device: Optional[torch.device] = None, outputs: Sequence[Any] = ()
    ) -> Union[Tensor, List[Tensor], None]:
        """
        Execute the function with given arguments.

        Args:
            *args: Input tensors and scalars
            device: Optional device hint (inferred from tensors if not provided)
            outputs: Optional pre-allocated output tensors

        Returns:
            Output tensor(s) or None
        """

        # Infer device from inputs
        inferred_device = None
        if device is not None:
            inferred_device = get_device_from_torch(device)
        else:
            for arg in args:
                if isinstance(arg, Tensor):
                    inferred_device = get_device_from_torch(arg.device)
                    break

        # Initialize runtime if needed
        self._initialize_runtime(inferred_device)

        # Ensure we have a device and context after initialization
        if inferred_device is None:
            raise RuntimeError("No device could be inferred from inputs")

        if self._vm_context is None:
            raise RuntimeError("VM context not initialized")

        # Prepare arguments
        arg_list = VmVariantList(len(args))
        for arg in args:
            if isinstance(arg, Tensor):
                # Import tensor to HAL
                # For multi-device, the runtime will handle placement
                if not arg.is_contiguous():
                    arg = arg.contiguous()

                device_obj = get_device_from_torch(arg.device)
                arg_list.push_ref(device_obj.import_torch_tensor(arg))
            elif isinstance(arg, int):
                arg_list.push_int(arg)
            elif isinstance(arg, float):
                arg_list.push_float(arg)

        # Prepare outputs
        ret_list = VmVariantList(len(outputs))
        for output in outputs:
            if isinstance(output, Tensor):
                assert output.is_contiguous(), "Output tensors must be contiguous"
                device_obj = get_device_from_torch(output.device)
                ret_list.push_ref(device_obj.import_torch_tensor(output))
            else:
                raise ValueError(
                    f"Unsupported output type: {type(output)}. " "Expected Tensor."
                )

        # Execute
        if self._is_async:
            # in async mode, we use the simplest implementation from the launch.py
            # launch.py supports a device.sync mode which uses timepoints, not implemented here
            external_timepoint = inferred_device.setup_iree_action()
            wait_fence = HalFence(0)
            signal_fence = HalFence(0)
            arg_list.push_ref(wait_fence)
            arg_list.push_ref(signal_fence)

            # asynchronous execution
            self._vm_context.invoke(self._vm_function, arg_list, ret_list)

            inferred_device.finalize_iree_action(external_timepoint)
        else:
            # Synchronous execution
            self._vm_context.invoke(self._vm_function, arg_list, ret_list)

        # Extract results
        results = []
        for i in range(len(ret_list)):
            result = ret_list.get_variant(i)
            if isinstance(result, VmRef):
                buffer_view = result.deref(HalBufferView, True)
                if buffer_view is not None:
                    torch_tensor = self._export_torch_tensor(
                        buffer_view, inferred_device
                    )
                    results.append(torch_tensor)

        if len(results) == 0:
            return None
        elif len(results) == 1:
            return results[0]
        else:
            return results

    def _export_torch_tensor(self, bv: HalBufferView, device: Device) -> Tensor:
        """Export HAL buffer view to PyTorch tensor."""
        dtype_map = {
            int(HalElementType.FLOAT_32): torch.float32,
            int(HalElementType.FLOAT_16): torch.float16,
            int(HalElementType.INT_32): torch.int32,
            int(HalElementType.INT_64): torch.int64,
            # Add more mappings as needed
        }

        dtype = dtype_map.get(int(bv.element_type))
        if dtype is None:
            raise NotImplementedError(
                f"HalBufferView.element_type({bv.element_type}) has no mapping to dtype"
            )

        meta_tensor = torch.empty(bv.shape, dtype=dtype, device="meta")
        return device.export_torch_tensor(bv, meta_tensor)

    @classmethod
    def from_vmfb(
        cls,
        vmfb_path: Union[str, Path],
        devices: Optional[List[Union[str, Device]]] = None,
        function_name: str = "main",
        runtime_flags: Optional[List[str]] = None,
    ) -> "MultiDeviceLaunchable":
        """
        Create a launchable from a VMFB file.

        Example:
            launchable = MultiDeviceLaunchable.from_vmfb(
                "model.vmfb",
                devices=["hip://0", "hip://1"],
                runtime_flags=["--hip_use_streams=true"]
            )
        """
        return cls(
            vmfb_path,
            devices,
            function_name,
            runtime_flags,
            is_async=(function_name.endswith("$async")),
        )
