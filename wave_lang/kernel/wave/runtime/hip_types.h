// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

// We don't want to depend on the HIP headers, so we just hardcode the necessary
// constants and types here.

#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void *)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE ((void *)0x02)
#define HIP_LAUNCH_PARAM_END ((void *)0x03)

using hipError_t = int;
using hipStream_t = void *;
using hipFunction_t = void *;
using hipModule_t = void *;

typedef enum hipLaunchAttributeID {
  hipLaunchAttributeAccessPolicyWindow =
      1, ///< Valid for Streams, graph nodes, launches
  hipLaunchAttributeCooperative = 2, ///< Valid for graph nodes, launches
  hipLaunchAttributeSynchronizationPolicy = 3, ///< Valid for streams
  hipLaunchAttributePriority = 8, ///< Valid for graph node, streams, launches
  hipLaunchAttributeMemSyncDomainMap =
      9, ///< Valid for streams, graph nodes, launches
  hipLaunchAttributeMemSyncDomain =
      10, ///< Valid for streams, graph nodes, launches
  hipLaunchAttributeMax
} hipLaunchAttributeID;

typedef union hipLaunchAttributeValue {
  char pad[64]; ///< 64 byte padding
  // hipAccessPolicyWindow
  //     accessPolicyWindow;  ///< Value of launch attribute
  //     ::hipLaunchAttributeAccessPolicyWindow.
  // int cooperative;         ///< Value of launch attribute
  // ::hipLaunchAttributeCooperative. Indicates
  //                          ///< whether the kernel is cooperative.
  // int priority;  ///< Value of launch attribute ::
  // hipLaunchAttributePriority. Execution priority of
  //                ///< kernel
  // hipSynchronizationPolicy
  //     syncPolicy;  ///< Value of launch attribute ::
  //     hipLaunchAttributeSynchronizationPolicy. Used
  //                  ///< to work queued up in stream
  // hipLaunchMemSyncDomainMap
  //     memSyncDomainMap;  ///< Value of launch attribute
  //     hipLaunchAttributeMemSyncDomainMap
  // hipLaunchMemSyncDomain
  //     memSyncDomain;  ///< Value of launch attribute
  //     hipLaunchAttributeMemSyncDomain
} hipLaunchAttributeValue;

typedef struct hipLaunchAttribute_st {
  hipLaunchAttributeID id; ///< Identifier of the launch attribute
  char pad[8 - sizeof(hipLaunchAttributeID)]; ///< Padding to align the
                                              ///< structure to 8 bytes
  union {
    hipLaunchAttributeValue val; ///< Value associated with the launch attribute
    hipLaunchAttributeValue
        value; ///< Value associated with the launch attribute
  };
} hipLaunchAttribute;

typedef struct HIP_LAUNCH_CONFIG_st {
  unsigned int gridDimX;  ///< Grid width in blocks
  unsigned int gridDimY;  ///< Grid height in blocks
  unsigned int gridDimZ;  ///< Grid depth in blocks
  unsigned int blockDimX; ///< Thread block dimension in X
  unsigned int blockDimY; ///< Thread block dimension in Y
  unsigned int blockDimZ; ///< Thread block dimension in Z
  unsigned int
      sharedMemBytes;        ///< Dynamic shared-memory size in bytes per block
  hipStream_t hStream;       ///< HIP stream identifier
  hipLaunchAttribute *attrs; ///< Attribute list
  unsigned int numAttrs;     ///< Number of attributes
} HIP_LAUNCH_CONFIG;

using hipModuleLaunchKernel_t = hipError_t (*)(hipFunction_t, unsigned int,
                                               unsigned int, unsigned int,
                                               unsigned int, unsigned int,
                                               unsigned int, unsigned int,
                                               hipStream_t, void **, void **);

using hipDrvLaunchKernelEx_t = hipError_t (*)(const HIP_LAUNCH_CONFIG *,
                                              hipFunction_t, void **, void **);

using hipGetErrorName_t = const char *(*)(hipError_t);
using hipGetErrorString_t = const char *(*)(hipError_t);
using hipModuleUnload_t = hipError_t (*)(hipModule_t);
using hipModuleLoad_t = hipError_t (*)(hipModule_t *, const char *);
using hipModuleLoadData_t = hipError_t (*)(hipModule_t *, const void *);
using hipModuleGetFunction_t = hipError_t (*)(hipFunction_t *, hipModule_t,
                                              const char *);
