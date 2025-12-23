# RUN: python %s
# Just check that we exit with no error.

import sys

# Importing main wave package should not import IREE.


def testIreeNotImported():
    for imported in sys.modules:
        assert "iree" not in imported, f"iree is imported: {sys.modules[imported]}"


# This cannot run under pytest as that will explore the whole package and transitively import IREE.
if __name__ == "__main__":
    testIreeNotImported()
