#!/usr/bin/env python3
"""Serve DINO-initialized SAM2 articulation measurements over localhost TCP.

Run this script from the isolated vision environment.  ``--calibrated`` is
deliberately opt-in and may only be used after saved-frame parity validation.
"""
from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import socketserver
import struct
import tempfile

import numpy as np
from PIL import Image
import torch


HEADER = struct.Struct("!I")
MAX_REQUEST_BYTES = 2_000_000


def receive_exact(stream, size):
    parts, remaining = [], size
    while remaining:
        part = stream.recv(remaining)
        if not part:
            raise ConnectionError("client closed connection")
        parts.append(part); remaining -= len(part)
    return b"".join(parts)


def mask_box(mask):
    points = np.argwhere(mask)
    if not len(points):
        return None
    y0, x0 = points.min(axis=0); y1, x1 = points.max(axis=0)
    return np.asarray([x0, y0, x1 + 1, y1 + 1], dtype=np.float32)


class DinoSam2Backend:
    def __init__(self, *, query, dino_model, sam_model, calibrated,
                 dino_threshold=.20, minimum_mask_score=.70):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.query = query
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(dino_model)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_model).to(self.device).eval()
        self.predictor = SAM2ImagePredictor.from_pretrained(sam_model)
        self.calibrated = bool(calibrated)
        self.dino_threshold = float(dino_threshold)
        self.minimum_mask_score = float(minimum_mask_score)
        self.reset()

    def reset(self):
        self.box = None
        self.low_res_logits = None
        self.initial_area = None
        self.initial_detection_score = 0.0
        self.initial_box = None

    def _initialize(self, image):
        inputs = self.processor(images=image, text=self.query + ".",
                                return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.detector(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            output, inputs.input_ids, threshold=.12, text_threshold=.10,
            target_sizes=[image.size[::-1]])[0]
        if not len(result["scores"]):
            return False
        index = int(torch.argmax(result["scores"]).item())
        score = float(result["scores"][index])
        if score < self.dino_threshold:
            return False
        self.box = result["boxes"][index].detach().cpu().numpy().astype(np.float32)
        self.initial_detection_score = score
        return True

    def observe(self, request):
        encoded = request.get("agent_image_base64", request.get("agent_jpeg_base64"))
        image = Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
        if self.box is None and not self._initialize(image):
            return self._unknown("grounding_failed")
        self.predictor.set_image(np.asarray(image))
        masks, scores, logits = self.predictor.predict(
            box=self.box,
            mask_input=self.low_res_logits,
            multimask_output=False,
            return_logits=False,
        )
        if not len(scores):
            return self._unknown("segmentation_failed")
        index = int(np.argmax(scores))
        mask = np.asarray(masks[index], dtype=bool)
        score = float(scores[index])
        box = mask_box(mask)
        area = float(mask.sum())
        if box is None or area <= 0:
            return self._unknown("empty_mask")
        self.box = box
        self.low_res_logits = np.asarray(logits[index:index + 1])
        if self.initial_area is None:
            self.initial_area = area
        confidence = float(min(self.initial_detection_score, score))
        reliable = score >= self.minimum_mask_score
        return {
            "area_fraction": area / float(image.width * image.height),
            "confidence": confidence,
            "observable": True,
            "identity_reliable": reliable,
            "calibrated": self.calibrated,
            "provenance": "grounding_dino_open_state_plus_sam2_image_prompt",
            "diagnostic": {"mask_score": score,
                           "initial_detection_score": self.initial_detection_score,
                           "relative_area": area / self.initial_area},
        }

    def _unknown(self, reason):
        return {"area_fraction": None, "confidence": 0.0,
                "observable": False, "identity_reliable": False,
                "calibrated": self.calibrated, "diagnostic": {"reason": reason}}


class StreamingDinoSam2Backend:
    """Incremental wrapper around SAM2's video-memory predictor.

    SAM2 exposes batch-video loading but no public live-camera API.  We use its
    ordinary video state for frame zero, append identically normalized tensors,
    and propagate exactly one newly arrived frame at a time.  This preserves
    temporal memory without looking at future frames.
    """

    def __init__(self, *, query, dino_model, sam_model, calibrated,
                 dino_threshold=.20, minimum_area_ratio=.10,
                 maximum_area_ratio=10.0, maximum_frames=600):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        self.query = query
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(dino_model)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_model).to(self.device).eval()
        self.predictor = SAM2VideoPredictor.from_pretrained(sam_model)
        self.calibrated = bool(calibrated)
        self.dino_threshold = float(dino_threshold)
        self.minimum_area_ratio = float(minimum_area_ratio)
        self.maximum_area_ratio = float(maximum_area_ratio)
        self.maximum_frames = int(maximum_frames)
        self._temporary_frames = None
        self.reset()

    def reset(self):
        if self._temporary_frames is not None:
            self._temporary_frames.cleanup()
        self._temporary_frames = tempfile.TemporaryDirectory(
            prefix="articulation-sam2-stream-")
        self.state = None
        self.initial_area = None
        self.previous_area = None
        self.initial_detection_score = 0.0
        self.frame_index = -1

    def _detect(self, image):
        inputs = self.processor(images=image, text=self.query + ".",
                                return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.detector(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            output, inputs.input_ids, threshold=.12, text_threshold=.10,
            target_sizes=[image.size[::-1]])[0]
        if not len(result["scores"]):
            return None
        index = int(torch.argmax(result["scores"]).item())
        score = float(result["scores"][index])
        if score < self.dino_threshold:
            return None
        self.initial_detection_score = score
        self.initial_box = result["boxes"][index].detach().cpu().numpy().astype(np.float32)
        return self.initial_box

    def _append_frame(self, image):
        image_size = self.predictor.image_size
        array = np.asarray(image.resize((image_size, image_size)), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor((.485, .456, .406), dtype=torch.float32)[:, None, None]
        std = torch.tensor((.229, .224, .225), dtype=torch.float32)[:, None, None]
        tensor = ((tensor - mean) / std).unsqueeze(0)
        tensor = tensor.to(self.state["device"])
        if self.state["num_frames"] >= self.maximum_frames:
            raise RuntimeError("stream exceeds preallocated frame budget")
        self.state["images"][self.state["num_frames"]].copy_(tensor[0])
        self.state["num_frames"] += 1

    def observe(self, request):
        encoded = request.get("agent_image_base64", request.get("agent_jpeg_base64"))
        image = Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
        self.frame_index += 1
        if self.state is None:
            box = request.get("anchor_box_xyxy")
            if box is not None:
                box = np.asarray(box, dtype=np.float32)
                self.initial_detection_score = 1.0
                self.initial_box = box.copy()
            else:
                box = self._detect(image)
            if box is None:
                return self._unknown("grounding_failed")
            first_path = f"{self._temporary_frames.name}/00000.jpg"
            image.save(first_path, format="JPEG", quality=90)
            self.state = self.predictor.init_state(
                self._temporary_frames.name, offload_video_to_cpu=False,
                offload_state_to_cpu=False)
            original = self.state["images"]
            allocated = torch.empty(
                (self.maximum_frames, *original.shape[1:]), dtype=original.dtype,
                device=original.device)
            allocated[0].copy_(original[0])
            self.state["images"] = allocated
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                self.predictor.add_new_points_or_box(
                    self.state, frame_idx=0, obj_id=1, box=box)
        else:
            self._append_frame(image)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = list(self.predictor.propagate_in_video(
                self.state, start_frame_idx=self.frame_index,
                max_frame_num_to_track=1, reverse=False))
        if not outputs:
            return self._unknown("video_propagation_failed")
        _, _, mask_logits = outputs[-1]
        mask = np.asarray(mask_logits[0].detach().cpu() > 0, dtype=bool)
        area = float(mask.sum())
        if area <= 0:
            return self._unknown("empty_mask")
        if self.initial_area is None:
            self.initial_area = area
        ratio = 1.0 if self.previous_area is None else area / self.previous_area
        self.previous_area = area
        reliable = self.minimum_area_ratio <= ratio <= self.maximum_area_ratio
        support = float(np.clip(
            self.initial_detection_score / max(self.dino_threshold, 1e-6), 0.0, 1.0))
        return {
            "area_fraction": area / float(image.width * image.height),
            # Calibrated support relative to the frozen DINO acceptance gate;
            # this is explicitly not a probability of semantic correctness.
            "confidence": support if reliable else 0.0,
            "observable": True,
            "identity_reliable": reliable,
            "calibrated": self.calibrated,
            "provenance": "grounding_dino_plus_incremental_sam2_video_memory",
            "diagnostic": {"initial_detection_score": self.initial_detection_score,
                           "initial_box_xyxy": self.initial_box.tolist(),
                           "relative_area": area / self.initial_area,
                           "frame_area_ratio": ratio},
        }

    def _unknown(self, reason):
        return {"area_fraction": None, "confidence": 0.0,
                "observable": False, "identity_reliable": False,
                "calibrated": self.calibrated, "diagnostic": {"reason": reason}}


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            size = HEADER.unpack(receive_exact(self.request, HEADER.size))[0]
            if size > MAX_REQUEST_BYTES:
                raise ValueError("request too large")
            request = json.loads(receive_exact(self.request, size).decode("utf-8"))
            if request.get("operation") == "reset":
                self.server.backend.reset(); reply = {"reset": True}
            elif request.get("operation") == "observe":
                reply = self.server.backend.observe(request)
            else:
                reply = {"observable": False, "identity_reliable": False,
                         "calibrated": False, "confidence": 0.0,
                         "area_fraction": None, "diagnostic": {"reason": "bad_operation"}}
        except Exception as error:
            reply = {"observable": False, "identity_reliable": False,
                     "calibrated": False, "confidence": 0.0,
                     "area_fraction": None,
                     "diagnostic": {"reason": type(error).__name__}}
        raw = json.dumps(reply, separators=(",", ":")).encode("utf-8")
        self.request.sendall(HEADER.pack(len(raw)) + raw)


class Server(socketserver.TCPServer):
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--query", default="open top drawer")
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam-model", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--tracking-backend", choices=("streaming_video", "image_prompt"),
                        default="streaming_video")
    parser.add_argument("--calibrated", action="store_true")
    args = parser.parse_args()
    backend_class = (StreamingDinoSam2Backend if args.tracking_backend == "streaming_video"
                     else DinoSam2Backend)
    backend = backend_class(
        query=args.query, dino_model=args.dino_model, sam_model=args.sam_model,
        calibrated=args.calibrated)
    with Server((args.host, args.port), Handler) as server:
        server.backend = backend
        print(json.dumps({"event": "ready", "host": args.host, "port": args.port,
                          "device": backend.device, "calibrated": args.calibrated,
                          "tracking_backend": args.tracking_backend}), flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
