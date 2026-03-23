"""WebSocket RFC 6455 low-level codec for Aize peer connections.

Supports text frames, ping/pong, and close frames.
Binary frames and fragmentation are accepted but not used by the peer protocol.
"""
from __future__ import annotations

import base64
import hashlib
import struct
from typing import IO

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONTINUATION = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA


def compute_accept_key(client_key: str) -> str:
    """Return Sec-WebSocket-Accept value for the given Sec-WebSocket-Key."""
    raw = (client_key.strip() + WS_GUID).encode("utf-8")
    return base64.b64encode(hashlib.sha1(raw).digest()).decode("ascii")


def read_frame(rfile: IO[bytes]) -> tuple[int, bytes] | None:
    """Read one WebSocket frame from *rfile*.

    Returns ``(opcode, payload)`` or ``None`` on EOF / incomplete read.
    Client-to-server masking is decoded automatically.
    Fragmented messages are returned as individual frames; callers that need
    reassembly should handle ``OP_CONTINUATION`` frames themselves.
    """
    header = rfile.read(2)
    if len(header) < 2:
        return None
    b0, b1 = header[0], header[1]
    opcode = b0 & 0x0F
    masked = (b1 & 0x80) != 0
    payload_len = b1 & 0x7F

    if payload_len == 126:
        ext = rfile.read(2)
        if len(ext) < 2:
            return None
        payload_len = struct.unpack("!H", ext)[0]
    elif payload_len == 127:
        ext = rfile.read(8)
        if len(ext) < 8:
            return None
        payload_len = struct.unpack("!Q", ext)[0]

    mask_key = b""
    if masked:
        mask_key = rfile.read(4)
        if len(mask_key) < 4:
            return None

    data = rfile.read(payload_len)
    if len(data) < payload_len:
        return None

    if masked:
        data = bytes(b ^ mask_key[i % 4] for i, b in enumerate(data))

    return opcode, data


def write_text_frame(wfile: IO[bytes], text: str) -> None:
    """Write a WebSocket text (UTF-8) frame to *wfile*."""
    _write_frame(wfile, OP_TEXT, text.encode("utf-8"))


def write_close_frame(wfile: IO[bytes], code: int = 1000) -> None:
    """Write a WebSocket close frame with the given status code."""
    _write_frame(wfile, OP_CLOSE, struct.pack("!H", code))


def write_pong_frame(wfile: IO[bytes], payload: bytes = b"") -> None:
    """Write a WebSocket pong frame (echo of ping payload)."""
    _write_frame(wfile, OP_PONG, payload)


def _write_frame(wfile: IO[bytes], opcode: int, payload: bytes) -> None:
    b0 = 0x80 | (opcode & 0x0F)  # FIN=1, RSV=0
    length = len(payload)
    header = bytearray([b0])
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    wfile.write(bytes(header))
    wfile.write(payload)
    wfile.flush()
