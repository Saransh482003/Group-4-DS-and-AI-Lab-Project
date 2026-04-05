import time
from app.pipeline.frame_context import FrameContext
from app.mechanics.navigation_logic import NavigationLogic

class NavigationComponent:
    def __init__(self, frame_width: int):
        self.logic = NavigationLogic(frame_width=frame_width)

    def run(self, ctx: FrameContext) -> None:
        start_time = time.perf_counter()

        # We rely on fusion having populated ctx.nav_detections
        zone_risks, nav_command = self.logic.process_detections(ctx.nav_detections)

        ctx.zone_risks = zone_risks
        ctx.nav_command = nav_command

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Attach to deterministic navigation for now.
        # Future-proofed for SLM separation as well.
        ctx.metrics["deterministic_nav_latency_ms"] = latency_ms
        ctx.metrics["navigation_latency_ms"] = latency_ms

        ctx.metrics["nav_command"] = nav_command
        for k, v in zone_risks.items():
            ctx.metrics[f"{k}_risk"] = v
