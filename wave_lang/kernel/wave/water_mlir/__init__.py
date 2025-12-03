# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from importlib import import_module as _import_module
import sys

# Re-export core modules from the nested relocated package.
ir = _import_module(".water_mlir.ir", __name__)
passmanager = _import_module(".water_mlir.passmanager", __name__)
rewrite = _import_module(".water_mlir.rewrite", __name__)
dialects = _import_module(".water_mlir.dialects", __name__)
extras = _import_module(".water_mlir.extras", __name__)
runtime = _import_module(".water_mlir.runtime", __name__)

# Ensure import machinery can resolve submodules like
# `from water_mlir.dialects import wave` by aliasing the nested modules
# into top-level qualified names in sys.modules.
sys.modules.setdefault(__name__ + ".ir", ir)
sys.modules.setdefault(__name__ + ".passmanager", passmanager)
sys.modules.setdefault(__name__ + ".rewrite", rewrite)
sys.modules.setdefault(__name__ + ".dialects", dialects)
sys.modules.setdefault(__name__ + ".extras", extras)
sys.modules.setdefault(__name__ + ".runtime", runtime)

__all__ = [
    "ir",
    "passmanager",
    "rewrite",
    "dialects",
    "extras",
    "runtime",
]
