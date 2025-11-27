// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace mlir::water {
#define GEN_PASS_DEF_WATERGPUTOGPURUNTIME
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

using namespace mlir;
using namespace mlir::water;

namespace {
/// Generate a unique LLVM global name for a given source name.
static SmallString<128> getUniqueLLVMGlobalName(ModuleOp mod,
                                                SymbolTable &table,
                                                const llvm::Twine &srcName) {
  unsigned counter = 0;
  return SymbolTable::generateSymbolName<128>(
      srcName.str(),
      [&](StringRef candidate) { return table.lookupSymbolIn(mod, candidate); },
      counter);
}

/// Helper to build a function call to a given function name with the given
/// return type and argument types.
struct FunctionCallBuilder {
  // TODO: cannot use TypeRange as `LLVM::LLVMFunctionType::get` refuses to
  // accept it.
  FunctionCallBuilder(StringRef functionName, Type returnType,
                      ArrayRef<Type> argumentTypes)
      : functionName(functionName),
        functionType(LLVM::LLVMFunctionType::get(returnType, argumentTypes)) {}
  LLVM::CallOp create(Location loc, OpBuilder &builder,
                      ValueRange arguments) const {
    Operation *module = builder.getBlock()
                            ->getParentOp()
                            ->getParentWithTrait<OpTrait::SymbolTable>();
    assert(module && "module not found");
    SymbolTable symbolTable(module);
    auto function = [&] {
      if (auto function = symbolTable.lookup<LLVM::LLVMFuncOp>(functionName))
        return function;

      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(&module->getRegion(0).front());
      return LLVM::LLVMFuncOp::create(builder, loc, functionName, functionType);
    }();
    return LLVM::CallOp::create(builder, loc, function, arguments);
  }

  StringRef functionName;
  LLVM::LLVMFunctionType functionType;
};

/// Create a unique LLVM global for a kernel handle.
static Value createKernelHandle(OpBuilder &builder, SymbolTable &symbolTable,
                                Type globalType, ModuleOp mod,
                                const llvm::Twine &name) {
  Type ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  Location loc = builder.getUnknownLoc();
  LLVM::GlobalOp handle;
  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(mod.getBody());
    SmallString<128> handleName =
        getUniqueLLVMGlobalName(mod, symbolTable, name);
    handle = LLVM::GlobalOp::create(
        builder, loc, globalType, /*isConstant*/ false, LLVM::Linkage::Internal,
        handleName, LLVM::ZeroAttr::get(builder.getContext()));
  }
  return LLVM::AddressOfOp::create(builder, loc, ptrType, handle.getSymName());
}

/// Get the object from a gpu binary op.
static gpu::ObjectAttr getSelectedObject(gpu::BinaryOp op) {
  ArrayRef<Attribute> objects = op.getObjectsAttr().getValue();

  // Obtain the index of the object to select.
  int64_t index = -1;
  if (Attribute target =
          cast<gpu::SelectObjectAttr>(op.getOffloadingHandlerAttr())
              .getTarget()) {
    // If the target attribute is a number it is the index. Otherwise compare
    // the attribute to every target inside the object array to find the index.
    if (auto indexAttr = dyn_cast<IntegerAttr>(target)) {
      index = indexAttr.getInt();
    } else {
      for (auto &&[i, attr] : llvm::enumerate(objects)) {
        auto obj = dyn_cast<gpu::ObjectAttr>(attr);
        if (obj && obj.getTarget() == target) {
          index = i;
          break;
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
  auto result = dyn_cast<gpu::ObjectAttr>(objects[index]);
  if (!result)
    op->emitError("invalid object type");

  return result;
}

/// Lookup the binary holding the kernel from enclosing module.
static gpu::ObjectAttr getBinary(gpu::LaunchFuncOp op) {
  auto kernelBinary = SymbolTable::lookupNearestSymbolFrom<gpu::BinaryOp>(
      op, op.getKernelModuleName());
  if (!kernelBinary) {
    op.emitError("couldn't find the binary holding the kernel: " +
                 op.getKernelModuleName().getValue());
    return nullptr;
  }

  return getSelectedObject(kernelBinary);
}

struct WaterGPUToGPURuntimePass final
    : public water::impl::WaterGPUToGPURuntimeBase<WaterGPUToGPURuntimePass> {
  using WaterGPUToGPURuntimeBase::WaterGPUToGPURuntimeBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *context = &getContext();
    IRRewriter builder(context);

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

    SymbolTable symbolTable(mod);

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

      gpu::ObjectAttr object = getBinary(op);
      if (!object)
        return WalkResult::interrupt();

      StringRef objData = object.getObject();

      Location loc = op.getLoc();
      auto getStr = [&](StringRef varName, StringRef str) -> Value {
        Twine strVal = str + StringRef("\0", 1);
        return LLVM::createGlobalString(
            loc, builder, getUniqueLLVMGlobalName(mod, symbolTable, varName),
            strVal.str(), LLVM::Linkage::Internal);
      };

      auto createConst = [&](Type type, int64_t val) -> Value {
        return LLVM::ConstantOp::create(builder, loc, type,
                                        builder.getIntegerAttr(type, val));
      };

      auto createAlloca = [&](Type elemType, int64_t size) -> Value {
        Value sizeVal = createConst(i64Type, size);
        return LLVM::AllocaOp::create(builder, loc, ptrType, elemType, sizeVal,
                                      0);
      };

      builder.setInsertionPoint(op);
      StringRef kernelName = op.getKernelName();
      Value kernelHandle = createKernelHandle(builder, symbolTable, ptrType,
                                              mod, kernelName + "_handle");
      Value kernelNameStr = getStr(kernelName, kernelName);

      Value dataPtr = LLVM::createGlobalString(
          loc, builder,
          getUniqueLLVMGlobalName(mod, symbolTable, kernelName + "_data"),
          objData, LLVM::Linkage::Internal);
      Value dataSize = createConst(i64Type, objData.size());

      Value funcObject =
          loadFuncBuilder
              .create(loc, builder,
                      {stream, kernelHandle, dataPtr, dataSize, kernelNameStr})
              ->getResult(0);

      Value sharedMemoryBytes = createConst(i32Type, 0);
      ValueRange args = op.getKernelOperands();
      auto argsPtrArrayType = LLVM::LLVMArrayType::get(ptrType, args.size());
      Value argsArray = LLVM::PoisonOp::create(builder, loc, argsPtrArrayType);

      for (auto &&[i, arg] : llvm::enumerate(args)) {
        Value argData = createAlloca(arg.getType(), 1);
        LLVM::StoreOp::create(builder, loc, arg, argData);
        argsArray =
            LLVM::InsertValueOp::create(builder, loc, argsArray, argData, i);
      }
      Value argsArrayPtr = createAlloca(argsPtrArrayType, 1);
      LLVM::StoreOp::create(builder, loc, argsArray, argsArrayPtr);
      Value argsCount = createConst(i32Type, args.size());

      launchFuncBuilder.create(
          loc, builder,
          {stream, funcObject, sharedMemoryBytes, op.getGridSizeX(),
           op.getGridSizeY(), op.getGridSizeZ(), op.getBlockSizeX(),
           op.getBlockSizeY(), op.getBlockSizeZ(), argsArrayPtr, argsCount});
      builder.eraseOp(op);
      return WalkResult::advance();
    };
    if (mod.walk(visitor).wasInterrupted())
      return signalPassFailure();

    mod->walk([&](gpu::BinaryOp op) { builder.eraseOp(op); });
  }
};
} // namespace
