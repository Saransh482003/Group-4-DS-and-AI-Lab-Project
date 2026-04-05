import time
import cv2
from typing import Any
from app.pipeline.frame_context import FrameContext
from app.mechanics.object_detection import ObjectDetector


class DetectorComponent:
    def __init__(self, yolo_weights_path: str):
        self.detector = ObjectDetector(yolo_weights_path)
        self.detector.load_model()

    def run(self, ctx: FrameContext) -> None:
        start_time = time.perf_counter()

        _, bboxes = self.detector.predict(ctx.frame_bgr)
        ctx.detections = bboxes

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        ctx.metrics["yolo_latency_ms"] = latency_ms
        ctx.metrics["detection_count"] = len(bboxes)
