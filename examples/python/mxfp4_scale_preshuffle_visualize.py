"""
Helper functions for e8m0_shuffle coordinate mapping.  Taken from Aiter kernel shuffle.
Specifically from: https://github.com/ROCm/rocm-libraries/blob/4348901528fe100a84975b89c247eece553a2a2d/shared/mxdatagenerator/lib/include/mxDataGenerator/PreSwizzle.hpp#L403

The e8m0_shuffle operation transforms a matrix with shape (m, n) as follows:
1. Pads to shape ((m+255)//256*256, (n+7)//8*8) - let's call this (sm, sn)
2. Reshapes to (sm//32, 2, 16, sn//8, 2, 4)
3. Permutes dimensions: (0, 3, 5, 2, 4, 1)
4. Flattens back to (sm, sn)

This file provides:
- e8m0_shuffle: The original shuffle function
- e8m0_shuffle_coords: Maps (i, j) in original -> (i', j') in shuffled
- e8m0_unshuffle_coords: Maps (i', j') in shuffled -> (i, j) in original
- e8m0_shuffle_coords_alt: Flattened expression version of e8m0_shuffle_coords
- e8m0_unshuffle_coords_alt: Flattened expression version of e8m0_unshuffle_coords
"""

import torch


def e8m0_shuffle(scale):
    """
    Shuffle the scale tensor for e8m0 format.

    Args:
        scale: A 2D tensor to be shuffled

    Returns:
        Shuffled tensor with the same padded shape
    """
    if scale is None:
        return scale
    if scale.dtype == torch.float32:
        return scale
    assert scale.ndim == 2, "scale must be a 2D tensor"
    m, n = scale.shape
    scale_padded = torch.zeros(
        (m + 255) // 256 * 256,
        (n + 7) // 8 * 8,
        dtype=scale.dtype,
        device=scale.device,
    )

    scale_padded[:m, :n] = scale
    scale = scale_padded
    sm, sn = scale.shape
    scale = scale.view(sm // 32, 2, 16, sn // 8, 2, 4)
    scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous()
    scale = scale.view(sm, sn)
    return scale


def e8m0_shuffle_coords(i, j, m, n):
    """
    Convert coordinates from original (unshuffled) matrix to shuffled matrix.

    Given a coordinate (i, j) in the original matrix (after padding),
    returns the corresponding coordinate (i', j') in the shuffled matrix.

    Args:
        i: Row index in original padded matrix (0-indexed)
        j: Column index in original padded matrix (0-indexed)
        m: Original row dimension (before padding)
        n: Original column dimension (before padding)

    Returns:
        Tuple (i_shuffled, j_shuffled) representing the position in shuffled matrix
    """
    # Calculate padded dimensions
    sm = ((m + 255) // 256) * 256
    sn = ((n + 7) // 8) * 8

    # Decompose original coordinates into 6D indices
    # Original view: [sm//32, 2, 16, sn//8, 2, 4]
    idx0 = i // 32
    remainder_i = i % 32
    idx1 = remainder_i // 16
    idx2 = remainder_i % 16

    idx3 = j // 8
    remainder_j = j % 8
    idx4 = remainder_j // 4
    idx5 = remainder_j % 4

    # After permute(0, 3, 5, 2, 4, 1), the order is: [idx0, idx3, idx5, idx2, idx4, idx1]
    # New shape: (sm//32, sn//8, 4, 16, 2, 2)

    # Compute linear index in permuted 6D array
    strides_permuted = [
        (sn // 8) * 4 * 16 * 2 * 2,  # stride for dim 0
        4 * 16 * 2 * 2,  # stride for dim 1
        16 * 2 * 2,  # stride for dim 2
        2 * 2,  # stride for dim 3
        2,  # stride for dim 4
        1,  # stride for dim 5
    ]

    linear = (
        idx0 * strides_permuted[0]
        + idx3 * strides_permuted[1]
        + idx5 * strides_permuted[2]
        + idx2 * strides_permuted[3]
        + idx4 * strides_permuted[4]
        + idx1 * strides_permuted[5]
    )

    # Convert linear index to 2D coordinates
    i_shuffled = linear // sn
    j_shuffled = linear % sn

    return (i_shuffled, j_shuffled)


def e8m0_unshuffle_coords(i_shuffled, j_shuffled, m, n):
    """
    Convert coordinates from shuffled matrix back to original (unshuffled) matrix.

    Given a coordinate (i', j') in the shuffled matrix,
    returns the corresponding coordinate (i, j) in the original matrix.

    Args:
        i_shuffled: Row index in shuffled matrix (0-indexed)
        j_shuffled: Column index in shuffled matrix (0-indexed)
        m: Original row dimension (before padding)
        n: Original column dimension (before padding)

    Returns:
        Tuple (i, j) representing the position in original unshuffled matrix
    """
    # Calculate padded dimensions
    sm = ((m + 255) // 256) * 256
    sn = ((n + 7) // 8) * 8

    # Convert 2D shuffled coordinates to linear index
    linear = i_shuffled * sn + j_shuffled

    # Decompose linear index into 6D indices in permuted space
    # Permuted shape: (sm//32, sn//8, 4, 16, 2, 2)
    strides_permuted = [
        (sn // 8) * 4 * 16 * 2 * 2,  # stride for dim 0
        4 * 16 * 2 * 2,  # stride for dim 1
        16 * 2 * 2,  # stride for dim 2
        2 * 2,  # stride for dim 3
        2,  # stride for dim 4
        1,  # stride for dim 5
    ]

    # Extract indices from linear position (in permuted order)
    idx0 = linear // strides_permuted[0]
    linear = linear % strides_permuted[0]

    idx3 = linear // strides_permuted[1]
    linear = linear % strides_permuted[1]

    idx5 = linear // strides_permuted[2]
    linear = linear % strides_permuted[2]

    idx2 = linear // strides_permuted[3]
    linear = linear % strides_permuted[3]

    idx4 = linear // strides_permuted[4]
    idx1 = linear % strides_permuted[4]

    # Reconstruct original coordinates from 6D indices
    # Original view: [sm//32, 2, 16, sn//8, 2, 4]
    # i = idx0 * 32 + idx1 * 16 + idx2
    # j = idx3 * 8 + idx4 * 4 + idx5

    i = idx0 * 32 + idx1 * 16 + idx2
    j = idx3 * 8 + idx4 * 4 + idx5

    return (i, j)


def e8m0_shuffle_coords_alt(i, j, m, n):
    """
    Simplified version of e8m0_shuffle_coords with expressions flattened.

    This computes the same transformation but with a single expression for each coordinate, similar to Wave kernel mappings.

    Args:
        i: Row index in original padded matrix (0-indexed)
        j: Column index in original padded matrix (0-indexed)
        m: Original row dimension (before padding)
        n: Original column dimension (before padding)

    Returns:
        Tuple (i_shuffled, j_shuffled) representing the position in shuffled matrix
    """
    # Calculate padded dimensions
    sm = ((m + 255) // 256) * 256
    sn = ((n + 7) // 8) * 8

    # Flattened linear index computation:
    # linear = idx0*stride0 + idx3*stride1 + idx5*stride2 + idx2*stride3 + idx4*stride4 + idx1*stride5
    # where strides = [(sn//8)*256, 256, 64, 4, 2, 1]
    # and indices are: idx0=i//32, idx1=(i%32)//16, idx2=i%16, idx3=j//8, idx4=(j%8)//4, idx5=j%4
    linear = (
        (i // 32) * ((sn // 8) * 256)
        + (j // 8) * 256
        + ((j % 8) % 4) * 64
        + ((i % 32) % 16) * 4
        + (((j % 8) // 4) * 2)
        + ((i % 32) // 16)
    )

    # Convert linear index to 2D coordinates
    i_shuffled = linear // sn
    j_shuffled = linear % sn

    return (i_shuffled, j_shuffled)


def e8m0_unshuffle_coords_alt(i_shuffled, j_shuffled, m, n):
    """
    Simplified version of e8m0_unshuffle_coords with fully inlined expressions.

    This computes the same transformation but with all intermediate variables inlined into single expressions.

    Args:
        i_shuffled: Row index in shuffled matrix (0-indexed)
        j_shuffled: Column index in shuffled matrix (0-indexed)
        m: Original row dimension (before padding)
        n: Original column dimension (before padding)

    Returns:
        Tuple (i, j) representing the position in original unshuffled matrix
    """
    # Calculate padded dimensions
    sm = ((m + 255) // 256) * 256
    sn = ((n + 7) // 8) * 8

    # Convert 2D shuffled coordinates to linear index
    linear = i_shuffled * sn + j_shuffled

    i = (
        (linear // ((sn // 8) * 256)) * 32
        + ((linear % ((sn // 8) * 256) % 256 % 64 % 4) % 2) * 16
        + (linear % ((sn // 8) * 256) % 256 % 64) // 4
    )

    j = (
        ((linear % ((sn // 8) * 256)) // 256) * 8
        + ((linear % ((sn // 8) * 256) % 256 % 64 % 4) // 2) * 4
        + (linear % ((sn // 8) * 256) % 256) // 64
    )

    return (i, j)


def test_round_trip(m, n, verbose=False):
    """
    Exhaustively test round-trip coordinate mapping for a matrix of size (m, n).

    For each coordinate in the padded matrix:
    1. Convert from original to shuffled coordinates
    2. Convert back from shuffled to original coordinates
    3. Verify the result matches the original coordinates
    4. Verify that accessing data through the coordinate mapping matches the shuffle operation

    Args:
        m: Row dimension of the matrix (before padding)
        n: Column dimension of the matrix (before padding)
        verbose: If True, print detailed progress and error messages

    Returns:
        True if all tests pass, False otherwise
    """
    # Calculate padded dimensions
    sm = ((m + 255) // 256) * 256
    sn = ((n + 7) // 8) * 8

    if verbose:
        print(f"Testing round-trip for matrix size ({m}, {n})")
        print(f"Padded dimensions: ({sm}, {sn})")
        print(f"Total coordinates to test: {sm * sn}")

    # Create test matrix with unique values (incrementing integers)
    # Use int32 to ensure we have enough unique values for large matrices
    original_matrix = torch.arange(sm * sn, dtype=torch.int32).reshape(sm, sn)

    # Create a copy and shuffle it
    shuffled_matrix = e8m0_shuffle(original_matrix.clone())

    if verbose:
        print(f"Created test matrices with unique values 0 to {sm * sn - 1}")

    coord_errors = []
    data_errors = []
    simplified_shuffle_errors = []
    simplified_unshuffle_errors = []
    total_tests = sm * sn

    for i in range(sm):
        if verbose and i % 32 == 0:
            print(f"  Progress: {i}/{sm} rows tested ({100.0 * i / sm:.1f}%)")

        for j in range(sn):
            # Forward: original -> shuffled
            i_shuf, j_shuf = e8m0_shuffle_coords(i, j, m, n)

            # Test simplified shuffle function
            i_shuf_simp, j_shuf_simp = e8m0_shuffle_coords_alt(i, j, m, n)
            if i_shuf_simp != i_shuf or j_shuf_simp != j_shuf:
                error_msg = f"Simplified shuffle mismatch at ({i}, {j}): got ({i_shuf_simp}, {j_shuf_simp}), expected ({i_shuf}, {j_shuf})"
                simplified_shuffle_errors.append(error_msg)
                if verbose and len(simplified_shuffle_errors) <= 10:
                    print(f"  ERROR (simplified shuffle): {error_msg}")

            # Reverse: shuffled -> original
            i_back, j_back = e8m0_unshuffle_coords(i_shuf, j_shuf, m, n)

            # Test simplified unshuffle function
            i_back_simp, j_back_simp = e8m0_unshuffle_coords_alt(i_shuf, j_shuf, m, n)
            if i_back_simp != i_back or j_back_simp != j_back:
                error_msg = f"Simplified unshuffle mismatch at ({i_shuf}, {j_shuf}): got ({i_back_simp}, {j_back_simp}), expected ({i_back}, {j_back})"
                simplified_unshuffle_errors.append(error_msg)
                if verbose and len(simplified_unshuffle_errors) <= 10:
                    print(f"  ERROR (simplified unshuffle): {error_msg}")

            # Verify coordinate round-trip
            if i_back != i or j_back != j:
                error_msg = f"Round-trip failed: ({i}, {j}) -> ({i_shuf}, {j_shuf}) -> ({i_back}, {j_back})"
                coord_errors.append(error_msg)

                if verbose and len(coord_errors) <= 10:
                    print(f"  ERROR (coord): {error_msg}")

            # Verify data mapping: original[i,j] should equal shuffled[i_shuf, j_shuf]
            orig_val = original_matrix[i, j].item()
            shuf_val = shuffled_matrix[i_shuf, j_shuf].item()

            if orig_val != shuf_val:
                error_msg = f"Data mismatch: original[{i},{j}]={orig_val} != shuffled[{i_shuf},{j_shuf}]={shuf_val}"
                data_errors.append(error_msg)

                if verbose and len(data_errors) <= 10:
                    print(f"  ERROR (data): {error_msg}")

    if verbose:
        print(f"  Progress: {sm}/{sm} rows tested (100.0%)")

    # Report results
    total_errors = (
        len(coord_errors)
        + len(data_errors)
        + len(simplified_shuffle_errors)
        + len(simplified_unshuffle_errors)
    )

    if total_errors == 0:
        if verbose:
            print(f"\n✓ SUCCESS: All {total_tests} round-trip tests passed!")
            print(f"  - Coordinate mapping: {total_tests} tests passed")
            print(f"  - Data mapping: {total_tests} tests passed")
            print(f"  - Simplified shuffle: {total_tests} tests passed")
            print(f"  - Simplified unshuffle: {total_tests} tests passed")
        return True
    else:
        print(f"\n✗ FAILURE: {total_errors} out of {total_tests * 4} tests failed")

        if len(simplified_shuffle_errors) > 0:
            print(f"\n  Simplified shuffle errors: {len(simplified_shuffle_errors)}")
            if not verbose and len(simplified_shuffle_errors) <= 10:
                print("  First few errors:")
                for error_msg in simplified_shuffle_errors[:10]:
                    print(f"    {error_msg}")
            elif not verbose:
                print(f"  Showing first 10 of {len(simplified_shuffle_errors)} errors:")
                for error_msg in simplified_shuffle_errors[:10]:
                    print(f"    {error_msg}")

        if len(simplified_unshuffle_errors) > 0:
            print(
                f"\n  Simplified unshuffle errors: {len(simplified_unshuffle_errors)}"
            )
            if not verbose and len(simplified_unshuffle_errors) <= 10:
                print("  First few errors:")
                for error_msg in simplified_unshuffle_errors[:10]:
                    print(f"    {error_msg}")
            elif not verbose:
                print(
                    f"  Showing first 10 of {len(simplified_unshuffle_errors)} errors:"
                )
                for error_msg in simplified_unshuffle_errors[:10]:
                    print(f"    {error_msg}")

        if len(coord_errors) > 0:
            print(f"\n  Coordinate mapping errors: {len(coord_errors)}")
            if not verbose and len(coord_errors) <= 10:
                print("  First few coordinate errors:")
                for error_msg in coord_errors[:10]:
                    print(f"    {error_msg}")
            elif not verbose:
                print(f"  Showing first 10 of {len(coord_errors)} coordinate errors:")
                for error_msg in coord_errors[:10]:
                    print(f"    {error_msg}")

        if len(data_errors) > 0:
            print(f"\n  Data mapping errors: {len(data_errors)}")
            if not verbose and len(data_errors) <= 10:
                print("  First few data errors:")
                for error_msg in data_errors[:10]:
                    print(f"    {error_msg}")
            elif not verbose:
                print(f"  Showing first 10 of {len(data_errors)} data errors:")
                for error_msg in data_errors[:10]:
                    print(f"    {error_msg}")

        return False


def parse_range(range_str):
    """
    Parse a range string like "0-8" into a list of integers.

    Args:
        range_str: String in format "start-end" (e.g., "0-8", "20-25")

    Returns:
        List of integers in the range [start, end]
    """
    start, end = map(int, range_str.split("-"))
    return list(range(start, end + 1))


def visualize_shuffle(m, n, row_range=None, col_range=None, use_unshuffle=False):
    """
    Visualize the shuffle mapping by printing a matrix of tuples.

    For each coordinate (i, j) in the shuffled matrix, shows which original
    coordinate it maps to via e8m0_unshuffle_coords.

    Args:
        m: Row dimension of the matrix (before padding)
        n: Column dimension of the matrix (before padding)
        row_range: List of row indices to display (required)
        col_range: List of column indices to display (required)
        use_unshuffle: If True, use e8m0_unshuffle_coords; otherwise use e8m0_shuffle_coords
    """
    # Calculate padded dimensions
    sm = ((m + 255) // 256) * 256
    sn = ((n + 7) // 8) * 8

    translate_func = e8m0_unshuffle_coords if use_unshuffle else e8m0_shuffle_coords
    forward_shuffle = translate_func == e8m0_shuffle_coords

    print(
        f"Visualizing {'shuffle' if forward_shuffle else 'unshuffle'} mapping for size ({m}, {n})"
    )
    if forward_shuffle:
        print(
            f"IE the grid has the logical grid, with each cell showing the coordinates of where it will be shuffled to."
        )
    else:
        print(
            f"IE the grid has the physical shuffled grid, with each cell showing the unshuffled coordinates."
        )
    print(f"Padded dimensions: ({sm}, {sn})")

    if row_range is None or col_range is None:
        print("Error: --visualize-range is required for visualization")
        print('Usage: --visualize-range "ROW_START-ROW_END" "COL_START-COL_END"')
        print('Example: --visualize-range "0-8" "0-15"')
        return

    # First pass: determine the maximum width needed for formatting
    max_width = 0
    for i in row_range:
        for j in col_range:
            orig_i, orig_j = translate_func(i, j, m, n)
            tuple_str = f"({orig_i},{orig_j})"
            max_width = max(max_width, len(tuple_str))

    rows_to_print = row_range
    cols_to_print = col_range

    # Print column headers
    max_row_num_width = len(str(max(rows_to_print)))

    # Print column header
    col_header = " " * (max_row_num_width + 2)  # Space for row numbers
    for j in cols_to_print:
        col_str = str(j).ljust(max_width)
        col_header += col_str + " "
    print(col_header)
    print()  # Empty line after header

    for i in rows_to_print:
        # Print row number
        row_prefix = str(i).rjust(max_row_num_width) + ": "

        row_strs = []
        for j in cols_to_print:
            orig_i, orig_j = translate_func(i, j, m, n)
            tuple_str = f"({orig_i},{orig_j})"
            row_strs.append(tuple_str.ljust(max_width))
        print(row_prefix + " ".join(row_strs))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test e8m0_shuffle coordinate mapping functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show help (default when no options given)
  python mxfp4_scale_preshuffle_visualize.py

  # Run exhaustive round-trip test with default size (256x64)
  python mxfp4_scale_preshuffle_visualize.py --test

  # Run test with custom matrix size
  python mxfp4_scale_preshuffle_visualize.py --test --size 128 32

  # Run test with verbose output
  python mxfp4_scale_preshuffle_visualize.py --test --verbose

  # Visualize shuffle mapping (requires --visualize-range)
  python mxfp4_scale_preshuffle_visualize.py --visualize-shuffle --visualize-range "0-8" "0-15"

  # Visualize unshuffle mapping (requires --visualize-range)
  python mxfp4_scale_preshuffle_visualize.py --visualize-shuffle --visualize-unshuffle --visualize-range "0-8" "0-15"

  # Visualize with custom matrix size
  python mxfp4_scale_preshuffle_visualize.py --visualize-shuffle --size 256 64 --visualize-range "0-31" "0-15"
        """,
    )

    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("M", "N"),
        default=[256, 64],
        help="Matrix size (M rows, N columns) for testing (default: 256 64)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with progress information",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run exhaustive round-trip coordinate mapping tests",
    )

    parser.add_argument(
        "--visualize-shuffle",
        action="store_true",
        help="Visualize the shuffle mapping as a matrix of coordinate tuples",
    )

    parser.add_argument(
        "--visualize-unshuffle",
        action="store_true",
        help="Use unshuffle coordinates for visualization (shows physical shuffled grid with unshuffled coordinates)",
    )

    parser.add_argument(
        "--visualize-range",
        type=str,
        nargs=2,
        metavar=("ROW_RANGE", "COL_RANGE"),
        help='Range to visualize (e.g., "0-8" "0-15"). Required for --visualize-shuffle.',
    )

    args = parser.parse_args()

    # If no action is specified, print help and exit
    if not args.test and not args.visualize_shuffle:
        parser.print_help()
        exit(0)

    m, n = args.size

    # Validate dimensions
    if m <= 0 or n <= 0:
        print(f"Error: Matrix dimensions must be positive (got {m}x{n})")
        exit(1)

    # If visualize-shuffle is requested, run visualization and exit
    if args.visualize_shuffle:
        row_range = None
        col_range = None

        if args.visualize_range:
            # Parse row and column ranges
            row_range = parse_range(args.visualize_range[0])
            col_range = parse_range(args.visualize_range[1])

        visualize_shuffle(
            m,
            n,
            row_range=row_range,
            col_range=col_range,
            use_unshuffle=args.visualize_unshuffle,
        )
        exit(0)

    # Run round-trip test if --test is specified
    if args.test:
        print("=" * 70)
        print("ROUND-TRIP COORDINATE MAPPING TEST")
        print("=" * 70)
        passed = test_round_trip(m, n, verbose=args.verbose)
        print()

        # Final summary
        print("=" * 70)
        if passed:
            print("ALL TESTS PASSED ✓")
        else:
            print("SOME TESTS FAILED ✗")
        print("=" * 70)

        exit(0 if passed else 1)
