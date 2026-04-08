import time
import cv2
from app.pipeline.frame_context import FrameContext
from app.mechanics.depth_estimation import DepthEstimator


class DepthComponent:
    """
    Component responsible for managing the DepthEstimator and computing depth maps for frames in the pipeline.
    """
    def __init__(self, depth_model_path: str, device: str = "cpu"):
        self.estimator = DepthEstimator(depth_model_path, device=device)
        self.estimator.load_model()

    def run(self, ctx: FrameContext) -> None:
        start_time = time.perf_counter()

        # Ensure RGB frame is available
        if ctx.frame_rgb is None:
            ctx.frame_rgb = cv2.cvtColor(ctx.frame_bgr, cv2.COLOR_BGR2RGB)

        depth_float, depth_color = self.estimator.predict(ctx.frame_rgb)

        ctx.depth_map = depth_float
        ctx.depth_color = depth_color

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        ctx.metrics["depth_latency_ms"] = latency_ms
