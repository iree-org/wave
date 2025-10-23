// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "water-gpu-to-gpu-runtime"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace mlir::water {
#define GEN_PASS_DEF_WATERGPUTOGPURUNTIME
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

using namespace mlir;
using namespace mlir::water;

namespace {
static std::string getUniqueLLVMGlobalName(ModuleOp mod, llvm::Twine srcName) {
  for (int i = 0;; ++i) {
    auto name =
        (i == 0 ? srcName.str() : (srcName + "_" + llvm::Twine(i)).str());
    if (!mod.lookupSymbol(name))
      return name;
  }
}

struct FunctionCallBuilder {
  FunctionCallBuilder(StringRef functionName, Type returnType,
                      ArrayRef<Type> argumentTypes)
      : functionName(functionName),
        functionType(LLVM::LLVMFunctionType::get(returnType, argumentTypes)) {}
  LLVM::CallOp create(Location loc, OpBuilder &builder,
                      ArrayRef<Value> arguments) const {
    auto module = builder.getBlock()->getParent()->getParentOfType<ModuleOp>();
    auto function = [&] {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(module.getBody());
      if (auto function = module.lookupSymbol<LLVM::LLVMFuncOp>(functionName))
        return function;
      return builder.create<LLVM::LLVMFuncOp>(loc, functionName, functionType);
    }();
    return builder.create<LLVM::CallOp>(loc, function, arguments);
  }

  StringRef functionName;
  LLVM::LLVMFunctionType functionType;
};

static Value createKernelHandle(OpBuilder &builder, Type globaType,
                                ModuleOp mod, StringRef name) {
  Type ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  Location loc = builder.getUnknownLoc();
  LLVM::GlobalOp handle;
  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(mod.getBody());
    auto handleName = getUniqueLLVMGlobalName(mod, "kernel_handle");
    handle = builder.create<LLVM::GlobalOp>(
        loc, globaType, /*isConstant*/ false, LLVM::Linkage::Internal,
        handleName, Attribute());
  }
  return builder.create<LLVM::AddressOfOp>(loc, ptrType, handle.getSymName());
}

static gpu::ObjectAttr getSelectedObject(gpu::BinaryOp op) {
  ArrayRef<Attribute> objects = op.getObjectsAttr().getValue();

  // Obtain the index of the object to select.
  int64_t index = -1;
  if (Attribute target =
          cast<gpu::SelectObjectAttr>(op.getOffloadingHandlerAttr())
              .getTarget()) {
    // If the target attribute is a number it is the index. Otherwise compare
    // the attribute to every target inside the object array to find the index.
    if (auto indexAttr = mlir::dyn_cast<IntegerAttr>(target)) {
      index = indexAttr.getInt();
    } else {
      for (auto &&[i, attr] : llvm::enumerate(objects)) {
        auto obj = mlir::dyn_cast<gpu::ObjectAttr>(attr);
        if (obj.getTarget() == target) {
          index = i;
        }
      }
    }
  } else {
    // If the target attribute is null then it's selecting the first object in
    // the object array.
    index = 0;
  }

  if (index < 0 || index >= static_cast<int64_t>(objects.size())) {
    op->emitError("the requested target object couldn't be found");
    return nullptr;
  }
  return dyn_cast<gpu::ObjectAttr>(objects[index]);
}

static gpu::ObjectAttr getBinary(OpBuilder &builder, Location loc,
                                 gpu::LaunchFuncOp op) {
  auto kernelBinary = SymbolTable::lookupNearestSymbolFrom<gpu::BinaryOp>(
      op, op.getKernelModuleName());
  if (!kernelBinary) {
    op.emitError("Couldn't find the binary holding the kernel: " +
                 op.getKernelModuleName().getValue());
    return nullptr;
  }

  gpu::ObjectAttr object = getSelectedObject(kernelBinary);
  if (!object)
    op.emitError("Cannot get gpu object");

  return object;
}

static Value allocArray(OpBuilder &builder, Location loc, Type elemType,
                        ValueRange values) {
  auto arrayType = LLVM::LLVMArrayType::get(elemType, values.size());
  Value array = builder.create<LLVM::PoisonOp>(loc, arrayType);
  for (auto &&[i, val] : llvm::enumerate(values))
    array = builder.create<LLVM::InsertValueOp>(loc, array, val, i);

  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  Value size = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                values.size());
  Value res = builder.create<LLVM::AllocaOp>(loc, ptrType, elemType, size, 0);
  builder.create<LLVM::StoreOp>(loc, array, res);
  return res;
}

struct WaterGPUtoGPURuntimePass
    : public water::impl::WaterGPUtoGPURuntimeBase<WaterGPUtoGPURuntimePass> {
  using WaterGPUtoGPURuntimeBase::WaterGPUtoGPURuntimeBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *context = &getContext();
    OpBuilder builder(context);

    Type i32Type = builder.getI32Type();
    Type i64Type = builder.getI64Type();
    Type ptrType = LLVM::LLVMPointerType::get(context);
    Type voidType = LLVM::LLVMVoidType::get(context);
    FunctionCallBuilder loadFuncBuilder("wave_load_kernel", ptrType,
                                        {
                                            ptrType, // stream
                                            ptrType, // cached kernel handle
                                            ptrType, // binary pointer
                                            i64Type, // binary size
                                            ptrType  // function name
                                        });
    FunctionCallBuilder launchFuncBuilder("wave_launch_kernel", voidType,
                                          {
                                              ptrType, // stream
                                              ptrType, // function
                                              i32Type, // shared memory bytes
                                              i64Type, // gridX
                                              i64Type, // gridY
                                              i64Type, // gridZ
                                              i64Type, // blockX
                                              i64Type, // blockY
                                              i64Type, // blockZ
                                              ptrType, // kernel operands
                                              i32Type  // kernel operands count
                                          });

    auto visitor = [&](gpu::LaunchFuncOp op) -> WalkResult {
      auto func = op->getParentOfType<FunctionOpInterface>();
      if (!func) {
        op.emitError("launch func op must have a func op parent");
        return WalkResult::interrupt();
      }
      ValueRange blockArgs = func.getFunctionBody().front().getArguments();
      if (blockArgs.empty()) {
        op.emitError("func op must have at least one argument");
        return WalkResult::interrupt();
      }
      // First argument is stream pointer
      Value stream = blockArgs.front();
      if (!isa<LLVM::LLVMPointerType>(stream.getType())) {
        op.emitError("stream argument must be a pointer");
        return WalkResult::interrupt();
      }

      gpu::ObjectAttr object = getBinary(builder, op.getLoc(), op);
      if (!object)
        return WalkResult::interrupt();

      StringRef objData = object.getObject();

      Location loc = op.getLoc();
      auto getStr = [&](StringRef varName, StringRef str) -> Value {
        std::string strVal = str.str();
        strVal.append(std::string_view("\0", 1));
        return LLVM::createGlobalString(loc, builder,
                                        getUniqueLLVMGlobalName(mod, varName),
                                        strVal, LLVM::Linkage::Internal);
      };

      auto createConst = [&](int64_t val, Type type) -> Value {
        return builder.create<LLVM::ConstantOp>(
            loc, type, builder.getIntegerAttr(type, val));
      };

      auto createAlloca = [&](Type elemType, int64_t size) -> Value {
        Value sizeVal = createConst(size, i64Type);
        return builder.create<LLVM::AllocaOp>(loc, ptrType, elemType, sizeVal,
                                              0);
      };

      builder.setInsertionPoint(op);
      StringRef kernelName = op.getKernelName();
      Value kernelHandle = createKernelHandle(builder, ptrType, mod,
                                              (kernelName + "_handle").str());
      Value kernelNameStr = getStr(kernelName, kernelName);

      Value dataPtr = LLVM::createGlobalString(
          loc, builder, getUniqueLLVMGlobalName(mod, "kernel_data"), objData,
          LLVM::Linkage::Internal);
      Value dataSize = createConst(objData.size(), i64Type);

      Value funcObject =
          loadFuncBuilder
              .create(loc, builder,
                      {stream, kernelHandle, dataPtr, dataSize, kernelNameStr})
              ->getResult(0);

      Value sharedMemoryBytes = createConst(0, i32Type);
      ValueRange args = op.getKernelOperands();
      auto argsPtrArrayType = LLVM::LLVMArrayType::get(ptrType, args.size());
      Value argsArray = builder.create<LLVM::PoisonOp>(loc, argsPtrArrayType);

      for (auto &&[i, arg] : llvm::enumerate(args)) {
        Value argData = createAlloca(arg.getType(), 1);
        builder.create<LLVM::StoreOp>(loc, arg, argData);
        argsArray =
            builder.create<LLVM::InsertValueOp>(loc, argsArray, argData, i);
      }
      Value argsArrayPtr = createAlloca(ptrType, args.size());
      builder.create<LLVM::StoreOp>(loc, argsArray, argsArrayPtr);
      Value argsCount = createConst(args.size(), i32Type);

      launchFuncBuilder.create(
          loc, builder,
          {stream, funcObject, sharedMemoryBytes, op.getGridSizeX(),
           op.getGridSizeY(), op.getGridSizeZ(), op.getBlockSizeX(),
           op.getBlockSizeY(), op.getBlockSizeZ(), argsArrayPtr, argsCount});
      op->erase();
      return WalkResult::advance();
    };
    if (mod.walk(visitor).wasInterrupted())
      return signalPassFailure();

    mod->walk([](gpu::BinaryOp op) { op->erase(); });
  }
};
} // namespace
