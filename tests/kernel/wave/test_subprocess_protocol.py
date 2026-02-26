# Copyright 2026 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the length-prefixed subprocess protocol.

The protocol has three guarantees:
  1. Round-trip identity: recv(send(data)) == data for any bytes payload.
  2. Multiplexing: multiple messages through the same pipe stay distinct.
  3. Clean failure: truncated or missing data raises EOFError, never
     returns corrupt output.
"""

import io
import struct
import pytest

from wave_lang.kernel.wave.mlir_converter.protocol import (
    recv_message,
    send_message,
    _HEADER_FMT,
    _HEADER_SIZE,
)


# Test Round-trip identity
@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\x00",
        b"hello",
        b"\xff" * 1024,
        b"hello\x00world",
    ],
    ids=["empty", "null-byte", "ascii", "1K-binary", "embedded-null"],
)
def test_roundtrip(payload: bytes):
    """send then recv on the same pipe recovers the original bytes."""
    pipe = io.BytesIO()
    send_message(pipe, payload)
    pipe.seek(0)
    assert recv_message(pipe) == payload


def test_multiple_messages_stay_distinct():
    """Several messages written sequentially are read back in order."""
    messages = [b"first", b"", b"third\x00with\x00nulls", b"\xff" * 512]
    pipe = io.BytesIO()
    for msg in messages:
        send_message(pipe, msg)
    pipe.seek(0)
    for expected in messages:
        assert recv_message(pipe) == expected


# Test clean failure on truncation
def test_eof_on_empty_pipe():
    """recv from an empty pipe raises EOFError."""
    pipe = io.BytesIO(b"")
    with pytest.raises(EOFError):
        recv_message(pipe)


def test_eof_on_partial_header():
    """recv with fewer than 4 header bytes raises ConnectionError."""
    pipe = io.BytesIO(b"\x00\x00")
    with pytest.raises(ConnectionError):
        recv_message(pipe)


def test_eof_on_truncated_payload():
    """Header promises N bytes but pipe has fewer -> ConnectionError."""
    header = struct.pack(_HEADER_FMT, 100)
    pipe = io.BytesIO(header + b"short")
    with pytest.raises(ConnectionError):
        recv_message(pipe)


def test_wire_format():
    """The on-wire bytes are [4-byte LE/native length][payload]."""
    payload = b"ABC"
    pipe = io.BytesIO()
    send_message(pipe, payload)
    raw = pipe.getvalue()
    expected_header = struct.pack(_HEADER_FMT, len(payload))
    assert raw == expected_header + payload
