# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unifies all imports of iree.compiler.ir into one place."""

# ruff: noqa: F401

from iree.compiler.dialects import (
    affine as affine_d,
)
from iree.compiler.dialects import (
    amdgpu as amdgpu_d,
)
from iree.compiler.dialects import (
    arith as arith_d,
)
from iree.compiler.dialects import (
    builtin as builtin_d,
)
from iree.compiler.dialects import (
    flow as flow_d,
)
from iree.compiler.dialects import (
    func as func_d,
)
from iree.compiler.dialects import (
    gpu as gpu_d,
)
from iree.compiler.dialects import (
    iree_codegen as iree_codegen_d,
)
from iree.compiler.dialects import (
    llvm as llvm_d,
)
from iree.compiler.dialects import (
    math as math_d,
)
from iree.compiler.dialects import (
    memref as memref_d,
)
from iree.compiler.dialects import (
    rocdl as rocdl_d,
)
from iree.compiler.dialects import (
    scf as scf_d,
)
from iree.compiler.dialects import (
    stream as stream_d,
)
from iree.compiler.dialects import (
    tensor as tensor_d,
)
from iree.compiler.dialects import (
    transform as transform_d,
)
from iree.compiler.dialects import (
    util as util_d,
)
from iree.compiler.dialects import (
    vector as vector_d,
)
from iree.compiler.dialects import (
    hal as hal_d,
)
from iree.compiler.dialects import (
    tensor as tensor_d,
)
from iree.compiler.ir import (
    AffineConstantExpr,
    AffineExpr,
    AffineMap,
    AffineMapAttr,
    ArrayAttr,
    AsmState,
    Attribute,
    BF16Type,
    Block,
    BlockArgument,
    # Types.
    ComplexType,
    Context,
    DenseElementsAttr,
    DenseResourceElementsAttr,
    DictAttr,
    F16Type,
    F32Type,
    F64Type,
    FlatSymbolRefAttr,
    Float4E2M1FNType,
    Float6E2M3FNType,
    Float8E4M3FNType,
    Float8E4M3FNUZType,
    Float8E5M2FNUZType,
    Float8E5M2Type,
    Float8E8M0FNUType,
    FloatAttr,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    MemRefType,
    MLIRError,
    Module,
    Operation,
    OpResult,
    RankedTensorType,
    ShapedType,
    StringAttr,
    SymbolRefAttr,
    SymbolTable,
    TypeAttr,
    UnitAttr,
    Value,
    VectorType,
    WalkResult,
)
from iree.compiler.ir import (
    Type as IrType,
)
from iree.compiler.passmanager import (
    PassManager,
)
