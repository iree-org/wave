# Copyright 2026 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Thin wrappers around dill that temporarily raise the recursion limit.

Large Wave object graphs can exceed Python's default 1000-frame recursion
limit during serialization. These helpers bump the limit for the
duration of the call and restore it afterwards.
"""

import sys

import dill

# Large kernel object graphs (e.g. attention) can exceed the default
# 1000-frame Python recursion limit during dill serialization.
DILL_RECURSION_LIMIT = 10000


def dumps(obj: object) -> bytes:
    """Serialize `obj` with dill, temporarily raising the recursion limit."""
    saved = sys.getrecursionlimit()
    sys.setrecursionlimit(DILL_RECURSION_LIMIT)
    try:
        return dill.dumps(obj)
    finally:
        sys.setrecursionlimit(saved)


def loads(data: bytes) -> object:
    """Deserialize `data` with dill, temporarily raising the recursion limit."""
    saved = sys.getrecursionlimit()
    sys.setrecursionlimit(DILL_RECURSION_LIMIT)
    try:
        return dill.loads(data)
    finally:
        sys.setrecursionlimit(saved)
