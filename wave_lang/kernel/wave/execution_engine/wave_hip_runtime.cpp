// Copyright 2025 The IREE Authors
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "wave_hip_runtime.h"

#include <sstream>
#include <stdexcept>
#include <string>

#if defined(__linux__)
#include <dlfcn.h> // dlopen, dlsym, dlerror
using module_handle_t = void *;
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h> // LoadLibrary, GetProcAddress, GetLastError
using module_handle_t = HMODULE;
#else
#error "Unsupported platform"
#endif

// HIP constants and types
#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void *)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE ((void *)0x02)
#define HIP_LAUNCH_PARAM_END ((void *)0x03)

using hipError_t = int;
using hipStream_t = void *;
using hipFunction_t = void *;
using hipModule_t = void *;

using hipModuleLaunchKernel_t = hipError_t (*)(hipFunction_t, unsigned int,
                                               unsigned int, unsigned int,
                                               unsigned int, unsigned int,
                                               unsigned int, unsigned int,
                                               hipStream_t, void **, void **);

using hipGetErrorName_t = const char *(*)(hipError_t);
using hipGetErrorString_t = const char *(*)(hipError_t);
using hipModuleUnload_t = hipError_t (*)(hipModule_t);
using hipModuleLoadData_t = hipError_t (*)(hipModule_t *, const void *);
using hipModuleGetFunction_t = hipError_t (*)(hipFunction_t *, hipModule_t,
                                              const char *);

// Global function pointers
static hipModuleLaunchKernel_t hipModuleLaunchKernel = nullptr;
static hipGetErrorName_t hipGetErrorName = nullptr;
static hipGetErrorString_t hipGetErrorString = nullptr;
static hipModuleUnload_t hipModuleUnload = nullptr;
static hipModuleLoadData_t hipModuleLoadData = nullptr;
static hipModuleGetFunction_t hipModuleGetFunction = nullptr;

static void *get_symbol_address(module_handle_t module,
                                const char *symbol_name) {
#if defined(__linux__)
  return dlsym(module, symbol_name);
#elif defined(_WIN32)
  return reinterpret_cast<void *>(GetProcAddress(module, symbol_name));
#endif
}

#define GET_FUNC(module, name)                                                 \
  do {                                                                         \
    name =                                                                     \
        reinterpret_cast<decltype(name)>(get_symbol_address(module, #name));   \
    if (!name) {                                                               \
      throw std::runtime_error("Failed to load symbol: " +                     \
                               std::string(#name));                            \
    }                                                                          \
  } while (0)

extern "C" void load_functions() {
  // Return early if already loaded
  if (hipModuleLaunchKernel && hipGetErrorName && hipGetErrorString &&
      hipModuleUnload && hipModuleLoadData && hipModuleGetFunction)
    return;

  module_handle_t module = nullptr;

#if defined(__linux__)
  module = dlopen("libamdhip64.so", RTLD_NOW);
  if (!module) {
    throw std::runtime_error("Failed to load libamdhip64.so: " +
                             std::string(dlerror()));
  }
#elif defined(_WIN32)
  module = LoadLibrary("amdhip64.dll");
  if (!module) {
    DWORD error_code = GetLastError();
    throw std::runtime_error("Failed to load amdhip64.dll: error code " +
                             std::to_string(error_code));
  }
#endif

  GET_FUNC(module, hipModuleLaunchKernel);
  GET_FUNC(module, hipGetErrorName);
  GET_FUNC(module, hipGetErrorString);
  GET_FUNC(module, hipModuleUnload);
  GET_FUNC(module, hipModuleLoadData);
  GET_FUNC(module, hipModuleGetFunction);
}

#undef GET_FUNC

#define HIP_CHECK_EXC(expr)                                                    \
  do {                                                                         \
    hipError_t e = (expr);                                                     \
    if (e) {                                                                   \
      const char *errName = hipGetErrorName(e);                                \
      const char *errMsg = hipGetErrorString(e);                               \
      std::ostringstream msg;                                                  \
      msg << "Error " << e << "(" << errName << ") " << __FILE__ << ":"        \
          << __LINE__ << ": " << std::endl                                     \
          << #expr << std::endl                                                \
          << errMsg << std::endl;                                              \
      throw std::runtime_error(msg.str());                                     \
    }                                                                          \
  } while (0)

extern "C" void *wave_load_kernel(void *stream, void **cached_kernel_handle,
                                  const void *binary_pointer,
                                  size_t /*binary_size*/,
                                  const char *kernel_name) {
  hipFunction_t function = *cached_kernel_handle;
  if (function)
    return function;

  hipModule_t mod = nullptr;
  HIP_CHECK_EXC(hipModuleLoadData(&mod, binary_pointer));
  HIP_CHECK_EXC(hipModuleGetFunction(&function, mod, kernel_name));
  *cached_kernel_handle = function;

  return function;
}

extern "C" void wave_launch_kernel(void *stream, void *function,
                                   int shared_memory_bytes, int grid_x,
                                   int grid_y, int grid_z, int block_x,
                                   int block_y, int block_z, void **args,
                                   int /*num_args*/) {
  HIP_CHECK_EXC(hipModuleLaunchKernel(function, grid_x, grid_y, grid_z, block_x,
                                      block_y, block_z, shared_memory_bytes,
                                      stream, args, nullptr));
}
