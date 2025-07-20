# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unifies all imports of iree.compiler.ir into one place."""

# ruff: noqa: F401

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
    tensor as tensor_d,
)
from iree.compiler.dialects import (
    util as util_d,
)
from iree.compiler.ir import (
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
    MLIRError,
    Module,
    Operation,
    OpResult,
    RankedTensorType,
    ShapedType,
    StringAttr,
    SymbolTable,
    TypeAttr,
    UnitAttr,
    Value,
)
from iree.compiler.ir import (
    Type as IrType,
)
from iree.compiler.passmanager import (
    PassManager,
)
