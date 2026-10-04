import json
import socket
import struct
import threading

import numpy as np

from vla_supervisor.articulation_sidecar import ArticulationSidecarClient


HEADER = struct.Struct("!I")


def serve_once(reply):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    captured = {}
    def run():
        with listener, listener.accept()[0] as stream:
            size = HEADER.unpack(stream.recv(4))[0]
            body = bytearray()
            while len(body) < size:
                body.extend(stream.recv(size - len(body)))
            captured.update(json.loads(body.decode("utf-8")))
            raw = json.dumps(reply).encode("utf-8")
            stream.sendall(HEADER.pack(len(raw)) + raw)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return port, captured, thread


def test_observe_round_trip_uses_fixed_camera_and_action_index():
    reply = {"area_fraction": .4, "confidence": .9, "observable": True,
             "identity_reliable": True, "calibrated": True}
    port, captured, thread = serve_once(reply)
    client = ArticulationSidecarClient(port=port, timeout_seconds=1)
    result = client(images_before=None,
                    images_after={"agent": np.zeros((16, 16, 3), np.uint8)},
                    intended_action=np.zeros(7), history=(), action_index=12)
    thread.join(1)
    assert result == reply
    assert captured["operation"] == "observe"
    assert captured["action_index"] == 12
    assert captured["agent_image_base64"]


def test_unavailable_sidecar_fails_closed_without_retry():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    client = ArticulationSidecarClient(port=port, timeout_seconds=.05)
    assert client(images_before=None,
                  images_after={"agent": np.zeros((4, 4, 3), np.uint8)},
                  intended_action=np.zeros(7), history=(), action_index=0) is None


def test_missing_image_abstains_before_network_access():
    client = ArticulationSidecarClient(port=8766, timeout_seconds=.05)
    assert client(images_before=None, images_after=None,
                  intended_action=np.zeros(7), history=(), action_index=0) is None
