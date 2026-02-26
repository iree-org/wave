# Copyright 2026 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Length-prefixed message protocol for emitter subprocess communication.

Wire format (same in both directions):
[4-byte native uint32 length][length bytes of payload]

We use native byte order (`=`) because both ends run on the same machine.
`uint32` (`I`) supports payloads up to ~4 GB which is more than enough.

This module only depends on the stdlib `struct` module, so it can be
imported from any Python environment without mlir/iree dependencies.
"""

import struct

_HEADER_FMT = "=I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def send_message(pipe, data: bytes) -> None:
    """Write a length-prefixed message to *pipe*."""
    pipe.write(struct.pack(_HEADER_FMT, len(data)))
    pipe.write(data)
    pipe.flush()


def recv_message(pipe) -> bytes:
    """Read a length-prefixed message from *pipe*.

    Raises `EOFError` on clean shutdown (peer closed the connection
    before a new message) and `ConnectionError` if the stream is
    truncated mid-message (partial header or incomplete payload).
    """
    header = pipe.read(_HEADER_SIZE)
    if len(header) == 0:
        raise EOFError("Peer closed the connection")
    if len(header) < _HEADER_SIZE:
        raise ConnectionError(
            f"Truncated header: expected {_HEADER_SIZE} bytes, got {len(header)}"
        )
    (length,) = struct.unpack(_HEADER_FMT, header)
    data = pipe.read(length)
    if len(data) < length:
        raise ConnectionError(
            f"Truncated payload: expected {length} bytes, got {len(data)}"
        )
    return data
