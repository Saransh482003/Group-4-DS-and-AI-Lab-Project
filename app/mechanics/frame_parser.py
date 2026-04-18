import time

import cv2
import torch

from mechanics.depth_estimation import estimate_distance_from_depth
from mechanics.depth_estimation import scan_depth_hazards
from mechanics.tts_command_utils import build_short_tts_command


class SharedFrameParser:
    def __init__(
        self,
        object_detector,
        depth_estimator,
        nav_logic_factory,
        *,
        device,
        depth_hazard_enabled=True,
        danger_threshold_m=1.2,
        warning_threshold_m=2.0,
        stateful_navigation=True,
    ):
        self.object_detector = object_detector
        self.depth_estimator = depth_estimator
        self.nav_logic_factory = nav_logic_factory
        self.device = device
        self.depth_hazard_enabled = depth_hazard_enabled
        self.danger_threshold_m = float(danger_threshold_m)
        self.warning_threshold_m = float(warning_threshold_m)
        self.stateful_navigation = stateful_navigation
        self._navigation_logic = None

    def _sync_cuda_if_needed(self, enabled):
        if enabled and self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _get_navigation_logic(self, frame_width):
        if self.stateful_navigation:
            if self._navigation_logic is None or int(getattr(self._navigation_logic, "frame_width", -1)) != int(frame_width):
                self._navigation_logic = self.nav_logic_factory(frame_width)
            return self._navigation_logic
        return self.nav_logic_factory(frame_width)

    def get_current_navigation_logic(self):
        return self._navigation_logic

    def parse_frame(
        self,
        frame_bgr,
        *,
        nav_logic_mode=0,
        slm_navigation_logic=None,
        shorten_tts_commands=True,
        sync_after_yolo=True,
        sync_after_depth=False,
        confidence_threshold=None,
        timestamp=None,
        hazard_return_coords=False,
    ):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        latencies = {}

        yolo_start = time.perf_counter()
        _, detections_raw = self.object_detector.predict(frame_bgr)
        self._sync_cuda_if_needed(sync_after_yolo)
        latencies["yolo_latency_ms"] = (time.perf_counter() - yolo_start) * 1000.0

        frame_boxes = []
        for det in detections_raw:
            confidence = float(det["confidence"])
            if confidence_threshold is not None and confidence < float(confidence_threshold):
                continue

            frame_boxes.append(
                {
                    "class_id": int(det.get("class_id", -1)),
                    "class_name": str(det["class_name"]),
                    "confidence": confidence,
                    "x1": int(det["x1"]),
                    "y1": int(det["y1"]),
                    "x2": int(det["x2"]),
                    "y2": int(det["y2"]),
                }
            )

        depth_start = time.perf_counter()
        depth_float, depth_color = self.depth_estimator.predict(frame_rgb)
        self._sync_cuda_if_needed(sync_after_depth)
        latencies["depth_latency_ms"] = (time.perf_counter() - depth_start) * 1000.0

        hazard_result = None
        hazard_start = time.perf_counter()
        if self.depth_hazard_enabled:
            hazard_result = scan_depth_hazards(
                depth_float,
                danger_threshold_m=self.danger_threshold_m,
                warning_threshold_m=self.warning_threshold_m,
                return_masks=True,
                return_coords=hazard_return_coords,
            )
        latencies["hazard_scan_latency_ms"] = (time.perf_counter() - hazard_start) * 1000.0

        spatial_start = time.perf_counter()
        for det in frame_boxes:
            bbox = [det["x1"], det["y1"], det["x2"], det["y2"]]
            object_depth, relative_distance = estimate_distance_from_depth(depth_float, bbox)
            det["depth_relative"] = object_depth
            det["distance_relative"] = relative_distance
            det["depth"] = object_depth
            det["distance"] = relative_distance
        latencies["spatial_latency_ms"] = (time.perf_counter() - spatial_start) * 1000.0

        nav_detections = [
            {
                "class": det["class_name"],
                "bbox": [det["x1"], det["y1"], det["x2"], det["y2"]],
                "depth": det.get("depth_relative"),
                "distance": det.get("distance_relative"),
            }
            for det in frame_boxes
        ]

        navigation_logic = self._get_navigation_logic(frame_bgr.shape[1])

        navigation_start = time.perf_counter()
        deterministic_start = time.perf_counter()
        zone_risks, nav_command = navigation_logic.process_detections(
            nav_detections,
            timestamp=timestamp,
            depth_hazard=hazard_result,
        )
        latencies["deterministic_nav_latency_ms"] = (time.perf_counter() - deterministic_start) * 1000.0

        if nav_logic_mode == 1 and slm_navigation_logic is not None:
            slm_start = time.perf_counter()
            slm_result = slm_navigation_logic.decide_instruction(
                zone_risks,
                nav_detections,
                frame_width=frame_bgr.shape[1],
            )
            nav_command = slm_result["instruction"]
            latencies["slm_nav_latency_ms"] = (time.perf_counter() - slm_start) * 1000.0

        latencies["navigation_latency_ms"] = (time.perf_counter() - navigation_start) * 1000.0

        tts_text = build_short_tts_command(nav_command) if shorten_tts_commands else nav_command

        return {
            "frame_boxes": frame_boxes,
            "depth_float": depth_float,
            "depth_color": depth_color,
            "hazard_result": hazard_result,
            "zone_risks": zone_risks,
            "nav_command": nav_command,
            "tts_text": tts_text,
            "nav_detections": nav_detections,
            "navigation_logic": navigation_logic,
            "latencies": latencies,
        }
