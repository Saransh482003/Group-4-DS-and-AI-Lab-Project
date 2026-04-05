import time
import numpy as np
from typing import Dict, Any

from app.pipeline.frame_context import FrameContext
from app.mechanics.depth_estimation import estimate_distance_from_depth

def fuse_detections_and_depth(ctx: FrameContext) -> None:
    """
    Enrich detections with depth information and compute scene-level depth statistics.
    """
    start_time = time.perf_counter()

    depth_map = ctx.depth_map
    nav_detections = []

    min_depth = float('inf')
    total_depth = 0.0
    valid_depth_count = 0

    for det in ctx.detections:
        bbox = [det["x1"], det["y1"], det["x2"], det["y2"]]

        if depth_map is not None:
            object_depth, relative_distance = estimate_distance_from_depth(depth_map, bbox)
        else:
            object_depth, relative_distance = None, None

        det["depth_relative"] = object_depth
        det["distance_relative"] = relative_distance

        nav_detections.append({
            "class": det["class_name"],
            "bbox": bbox,
            "depth": object_depth,
            "distance": relative_distance,
        })

        if object_depth is not None:
            if object_depth < min_depth:
                min_depth = object_depth
            total_depth += object_depth
            valid_depth_count += 1

    ctx.nav_detections = nav_detections

    if valid_depth_count > 0:
        ctx.metrics["min_depth_m"] = min_depth
        ctx.metrics["mean_depth_m"] = total_depth / valid_depth_count
        ctx.metrics["nearest_object_depth_m"] = min_depth
    else:
        ctx.metrics["min_depth_m"] = None
        ctx.metrics["mean_depth_m"] = None
        ctx.metrics["nearest_object_depth_m"] = None

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    ctx.metrics["fusion_latency_ms"] = latency_ms
