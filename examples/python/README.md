# Wave Lang Python Examples

This directory contains organized examples demonstrating various features of Wave Lang.

## Running Examples

Each example file contains multiple tests. You can:

1. **List available tests:**
   ```bash
   python 1_dynamic_mapping.py --list_tests
   ```

2. **Run a specific test:**
   ```bash
   python 1_dynamic_mapping.py --test test_read_write_dynamic_mapping_broadcast
   ```

3. **Run with debug output:**
   ```bash
   python 1_dynamic_mapping.py --test test_read_write_dynamic_mapping_broadcast --debug
   ```

4. **Repeat a test multiple times:**
   ```bash
   python 1_dynamic_mapping.py --test test_read_write_dynamic_mapping_broadcast --repeat 5
   ```

## Example Categories

### 1. Dynamic Mapping (`1_dynamic_mapping.py`)
Demonstrates reading and writing with dynamic index mappings and offset-based access patterns.
- `test_read_write_dynamic_mapping_broadcast` - 2D read with dynamic offset and broadcast
- `test_one_read_write_dynamic_mapping_broadcast` - 1D read with dynamic offset
- `test_one_nooffset_dynamic_mapping_broadcast` - Read/write with constant offset mapping

### 2. Control Flow (`2_control_flow.py`)
Demonstrates unstructured loops with dynamic conditions and iteration patterns.
- `test_iteration_with_condition` - Unstructured loop with runtime-determined exit condition

### 3. Atomic Operations (`3_atomics.py`)
Demonstrates atomic memory operations.
- `test_atomic_add_return_value` - Basic atomic add with return value
- `test_read_back_scalar` - Atomic operation followed by scalar readback

### 4. Reduction Operations (`4_reductions.py`)
Demonstrates various reduction patterns.
- `test_reduce_sum` - Basic sum reduction
- `test_broadcast_reduce_sum` - Broadcast then reduce pattern
- `test_moe_weighted_sum` - 3D weighted sum (MOE-style pattern)
