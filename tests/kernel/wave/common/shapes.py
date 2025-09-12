# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import pytest
from typing import Sequence
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.templates.attention_common import (
    AttentionShape,
)

# List of all test shapes for end to end tests.
_e2e_test_shapes = {}

# List of all test shapes for perf tests.
_perf_test_shapes = {}

# Order of shapes: (B, M, N, K1, K2)
_e2e_test_shapes["attention"] = [
    (8, 128, 128, 64, 256),
    (40, 1024, 64, 64, 1024),
]
_e2e_test_shapes["bhsd_attention"] = [
    (4, 32, 128, 128, 128, 128),
]
_e2e_test_shapes["cdna4_bshd_attention"] = [
    (64, 16384, 128, 128, 16384),
]
_e2e_test_shapes["chained_gemm"] = _e2e_test_shapes["attention"]
_e2e_test_shapes["decode_attention"] = _e2e_test_shapes["attention"]
_e2e_test_shapes["quantized_attention"] = [(1, 4096, 64, 64, 4096)]
_e2e_test_shapes["unaligned_attention"] = [
    (32, 1024, 128, 128, 1357),
    (48, 1024, 128, 128, 1357),
]
_e2e_test_shapes["all_attention"] = (
    _e2e_test_shapes["unaligned_attention"] + _e2e_test_shapes["attention"]
)

# Order of shapes: (B, BN, K2, H, K1, M, N)
_e2e_test_shapes["evoformer"] = [
    (1, 256, 256, 4, 32, 256, 32),
    (1, 512, 256, 8, 8, 256, 8),
]

_e2e_test_shapes["extend"] = [
    AttentionShape(
        num_seqs=2,
        context_len=1024,
        num_query_heads=16,
        num_kv_heads=1,
        head_size=128,
        head_size_kv=128,
        block_size=64,
    ),
    AttentionShape(
        num_seqs=1,
        context_len=1024,
        num_query_heads=16,
        num_kv_heads=2,
        head_size=64,
        head_size_kv=64,
        block_size=128,
    ),
    AttentionShape(
        num_seqs=4,
        context_len=1024,
        num_query_heads=4,
        num_kv_heads=4,
        head_size=256,
        head_size_kv=256,
        block_size=64,
    ),
]

_e2e_test_shapes["gqa_bshd_attention"] = [
    AttentionShape(
        num_seqs=1,
        num_query_heads=32,
        num_kv_heads=1,
        query_seq_len=8000,
        kv_seq_len=8000,
        head_size_kv=256,
        head_size=256,
    ),
]
_e2e_test_shapes["gqa_bshd_decode_attention"] = [
    AttentionShape(
        num_seqs=1,
        num_query_heads=32,
        num_kv_heads=1,
        query_seq_len=1,
        kv_seq_len=8000,
        head_size_kv=256,
        head_size=256,
    ),
]

_e2e_test_shapes["test_block_reduce"] = [
    (1, 37),
    (1, 312),
    (256, 256),
    (256, 1024),
]

test_names = [
    "attention",
    "bhsd_attention",
    "cdna4_bshd_attention",
    "chained_gemm",
    "decode_attention",
    "unaligned_attention",
    "quantized_attention",
    "all_attention",
    "evoformer",
    "gqa_bshd_attention",
    "gqa_bshd_decode_attention",
    "test_block_reduce",
]
for test in test_names:
    _perf_test_shapes[test] = _e2e_test_shapes[test]

_perf_test_shapes["extend"] = [
    AttentionShape(
        num_seqs=32,
        context_len=1024,
        num_query_heads=6,
        num_kv_heads=1,
        head_size=128,
        head_size_kv=128,
        block_size=64,
        fixed_seq_len_prefix=512,
        fixed_seq_len_extend=128,
    )
]


def make_shape_param(shape: Sequence[int], is_perf: bool):
    name = "x".join(map(str, shape))
    if is_perf:
        return pytest.param(shape, id=name + "-perf", marks=pytest.mark.perf_only)
    else:
        return pytest.param(shape, id=name)


def get_test_shapes(test_name: str):
    assert test_name in _e2e_test_shapes, f"Unknown test name: {test_name}"
    shapes = [make_shape_param(s, False) for s in _e2e_test_shapes[test_name]]
    shapes += [make_shape_param(s, True) for s in _perf_test_shapes[test_name]]
    return shapes
