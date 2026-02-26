"""
Utilities for running wave examples.
"""

import argparse
import traceback


def parse_args():
    """Parse command line arguments for example scripts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", type=str, required=False, help="Name of the test to run"
    )
    parser.add_argument(
        "--list_tests", action="store_true", help="List all available tests"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--repeat", type=int, default=1, help="Number of times to repeat the test"
    )
    # shape of the test
    parser.add_argument(
        "--shape", type=str, default=None, help="Shape of the test, e.g. 1024,1024,8192"
    )
    # block size of the test
    parser.add_argument(
        "--block",
        type=str,
        default=None,
        help="Block size of the test, e.g. 256,256,256",
    )

    args = parser.parse_args()

    # Convert shape and block arguments to tuples of ints
    if isinstance(args.shape, str):
        args.shape = tuple(map(int, args.shape.split(",")))
    if isinstance(args.block, str):
        args.block = tuple(map(int, args.block.split(",")))

    return args


def list_tests(module_globals):
    """List all test functions in the module."""
    tests = [f for f in module_globals if f.startswith("test_")]
    print("Available tests:")
    for test in tests:
        print(f"  {test}")


def run_test(test_name, module_globals, debug=False, repeat=1, shape=None, block=None):
    """Run a test function multiple times."""
    if test_name not in module_globals:
        print(f"Error: Test '{test_name}' not found")
        return False

    test_func = module_globals[test_name]
    kwargs = {"is_debug": debug}
    if shape is not None:
        kwargs["shape"] = shape
    if block is not None:
        kwargs["block"] = block

    for i in range(repeat):
        try:
            test_func(**kwargs)
            if repeat > 1:
                print(f"Test {i+1}/{repeat} passed")
        except Exception as e:
            traceback.print_exc()
            if repeat > 1:
                print(f"Test {i+1}/{repeat} failed")
            else:
                print(f"Test failed")
            return False
    return True
