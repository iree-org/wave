"""
Utilities for running wave examples.
"""

import argparse


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
    return parser.parse_args()


def list_tests(module_globals):
    """List all test functions in the module."""
    tests = [f for f in module_globals if f.startswith("test_")]
    print("Available tests:")
    for test in tests:
        print(f"  {test}")


def run_test(test_name, module_globals, debug=False, repeat=1):
    """Run a test function multiple times."""
    if test_name not in module_globals:
        print(f"Error: Test '{test_name}' not found")
        return False

    test_func = module_globals[test_name]
    for i in range(repeat):
        try:
            test_func(debug)
            if repeat > 1:
                print(f"Test {i+1}/{repeat} passed")
        except Exception as e:
            print(f"Error: {e}")
            if repeat > 1:
                print(f"Test {i+1}/{repeat} failed")
            else:
                print(f"Test failed")
            return False
    return True
