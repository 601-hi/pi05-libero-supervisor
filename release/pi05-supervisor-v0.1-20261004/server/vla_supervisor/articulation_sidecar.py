"""Small fail-closed TCP client for an isolated GPU perception process."""
from __future__ import annotations

import base64
from io import BytesIO
import json
import socket
import struct
from typing import Mapping

import numpy as np
from PIL import Image


_HEADER = struct.Struct("!I")
_MAX_REPLY_BYTES = 1_000_000


def _receive_exact(stream: socket.socket, size: int) -> bytes:
    parts = []
    remaining = size
    while remaining:
        part = stream.recv(remaining)
        if not part:
            raise ConnectionError("articulation sidecar closed the connection")
        parts.append(part)
        remaining -= len(part)
    return b"".join(parts)


def _encode_image(image) -> str:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("articulation sidecar expects an HxWx3 RGB image")
    buffer = BytesIO()
    # Lossless transport: the vision sidecar applies the same single JPEG-95
    # conversion used by the frozen offline SAM2 pipeline.
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class ArticulationSidecarClient:
    """Callable backend used by :class:`ValidatedArticulationProvider`.

    Communication errors return ``None`` so the validation adapter emits an
    unobservable measurement.  No retry is performed in the real-time action
    path: a delayed visual process must not delay or duplicate robot control.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8766,
                 timeout_seconds: float = 0.25):
        self.host = str(host)
        self.port = int(port)
        self.timeout_seconds = float(timeout_seconds)
        if not 0 < self.port < 65536 or self.timeout_seconds <= 0:
            raise ValueError("invalid sidecar connection configuration")

    def reset(self) -> None:
        self._request({"operation": "reset"})

    def __call__(self, *, images_before, images_after, intended_action,
                 history, action_index):
        if not isinstance(images_after, Mapping) or "agent" not in images_after:
            return None
        payload = {
            "operation": "observe",
            "action_index": int(action_index),
            "agent_image_base64": _encode_image(images_after["agent"]),
            "intended_action": np.asarray(intended_action, dtype=float).tolist(),
        }
        return self._request(payload)

    def _request(self, payload):
        raw = json.dumps(payload, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
        try:
            with socket.create_connection(
                    (self.host, self.port), timeout=self.timeout_seconds) as stream:
                stream.settimeout(self.timeout_seconds)
                stream.sendall(_HEADER.pack(len(raw)) + raw)
                size = _HEADER.unpack(_receive_exact(stream, _HEADER.size))[0]
                if size > _MAX_REPLY_BYTES:
                    raise ValueError("articulation sidecar reply is too large")
                reply = json.loads(_receive_exact(stream, size).decode("utf-8"))
        except (OSError, ConnectionError, TimeoutError, UnicodeError,
                json.JSONDecodeError, ValueError):
            return None
        return reply if isinstance(reply, Mapping) else None
