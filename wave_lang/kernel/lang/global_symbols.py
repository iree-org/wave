import sympy

from .._support.indexing import (
    index_symbol,
    MMA_ACC_SYMBOL_NAME,
    THREAD_SYMBOL_NAMES,
    WORKGROUP_SYMBOL_NAMES,
    DEVICE_SYMBOL_NAMES,
    GPR_SYMBOL_NAME,
)

# Global symbols used throughout the code.

# Address spaces.
GLOBAL_ADDRESS_SPACE = index_symbol("$GLOBAL_ADDRESS_SPACE")
SHARED_ADDRESS_SPACE = index_symbol("$SHARED_ADDRESS_SPACE")

# Device Distribution symbols.
# TODO: Can only do three dimensions for now.
DEVICE_DIM_0 = index_symbol(DEVICE_SYMBOL_NAMES[0])
DEVICE_DIM_1 = index_symbol(DEVICE_SYMBOL_NAMES[1])
DEVICE_DIM_2 = index_symbol(DEVICE_SYMBOL_NAMES[2])


# Distribution symbols.
WORKGROUP_0 = index_symbol(WORKGROUP_SYMBOL_NAMES[0])
WORKGROUP_1 = index_symbol(WORKGROUP_SYMBOL_NAMES[1])
WORKGROUP_2 = index_symbol(WORKGROUP_SYMBOL_NAMES[2])


def get_workgroup_symbol(i: int):
    assert i >= 0, "Workgroup index must be non-negative."
    symbol_name = f"WORKGROUP_{i}"
    symbol = index_symbol(WORKGROUP_SYMBOL_NAMES[i] if i < 3 else "$WG" + str(i))
    if symbol_name not in globals():
        globals()[symbol_name] = symbol
    return symbol


THREAD_0 = index_symbol(THREAD_SYMBOL_NAMES[0])
THREAD_1 = index_symbol(THREAD_SYMBOL_NAMES[1])
THREAD_2 = index_symbol(THREAD_SYMBOL_NAMES[2])

# Input selector symbol for selecting input from different tensors.
INPUT_SELECTOR = index_symbol("$INPUT_SELECTOR")

# MMA symbols.
MMA_LHS = index_symbol("$MMA_LHS")
MMA_RHS = index_symbol("$MMA_RHS")
MMA_ACC = index_symbol(MMA_ACC_SYMBOL_NAME)
MMA_LHS_SCALE = index_symbol("$MMA_LHS_SCALE")
MMA_RHS_SCALE = index_symbol("$MMA_RHS_SCALE")
MMA_SCALE_FP4 = index_symbol("$MMA_SCALE_FP4")
GPR_NUM = index_symbol(GPR_SYMBOL_NAME)

# Scheduling symbols.
READ_SHARED_DELAY = index_symbol("$READ_SHARED_DELAY")
WRITE_SHARED_DELAY = index_symbol("$WRITE_SHARED_DELAY")
READ_GLOBAL_DELAY = index_symbol("$READ_GLOBAL_DELAY")
WRITE_GLOBAL_DELAY = index_symbol("$WRITE_GLOBAL_DELAY")
GLOBAL_TO_SHARED_DELAY = index_symbol("$GLOBAL_TO_SHARED_DELAY")
MMA_DELAY = index_symbol("$MMA_DELAY")
VALU_DELAY = index_symbol("$VALU_DELAY")
SHUFFLE_DELAY = index_symbol("$SHUFFLE_DELAY")
SHARED_MEMORY_UNITS = index_symbol("$SHARED_MEMORY_UNITS")
GLOBAL_MEMORY_UNITS = index_symbol("$GLOBAL_MEMORY_UNITS")
MMA_UNITS = index_symbol("$MMA_UNITS")
VALU_UNITS = index_symbol("$VALU_UNITS")
SHUFFLE_UNITS = index_symbol("$SHUFFLE_UNITS")


# Iteration symbols.
def GET_ITER_ARG(i: int):
    return sympy.Symbol(f"$GET_ITER_ARG_{i}")
