import time
import cv2
from app.pipeline.frame_context import FrameContext
from app.mechanics.object_detection import draw_centered_label

class VisualizationComponent:
    def __init__(
        self,
        plot_stream_mode: int = 0, # 0 = RGB, 1 = Depth
        zone_shade_opacity: float = 0.1,
        left_end_ratio: float = 0.30,
        center_end_ratio: float = 0.70
    ):
        self.plot_stream_mode = plot_stream_mode
        self.zone_shade_opacity = zone_shade_opacity
        self.left_end_ratio = left_end_ratio
        self.center_end_ratio = center_end_ratio

    def run(self, ctx: FrameContext) -> None:
        start_time = time.perf_counter()

        # Base frame to plot on
        if self.plot_stream_mode == 0:
            plot_frame = ctx.frame_bgr.copy()
        else:
            if ctx.depth_color is not None:
                plot_frame = ctx.depth_color.copy()
            else:
                plot_frame = ctx.frame_bgr.copy()

        h, w = plot_frame.shape[:2]

        # 1. Apply zone shading
        self._apply_zone_shading(plot_frame, w, h)

        # 2. Draw bounding boxes and labels
        for det in ctx.detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Box
            cv2.rectangle(plot_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Class & Conf
            cv2.putText(
                plot_frame,
                f"{det['class_name']} {det['confidence']:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                2,
            )

            # Distance
            distance = det.get("distance_relative")
            dist_str = f"dist: {distance:.3f}" if distance is not None else "dist: N/A"
            draw_centered_label(plot_frame, dist_str, center_x, center_y)

        # 3. Draw navigation overlay (risks and commands)
        self._draw_nav_overlays(plot_frame, ctx.zone_risks, ctx.nav_command, w)

        ctx.annotated_frame = plot_frame

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        ctx.metrics["visualization_latency_ms"] = latency_ms

    def _apply_zone_shading(self, frame, w, h):
        left_end = int(self.left_end_ratio * w)
        center_end = int(self.center_end_ratio * w)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (left_end, h), (0, 0, 255), -1)
        cv2.rectangle(overlay, (left_end, 0), (center_end, h), (0, 255, 0), -1)
        cv2.rectangle(overlay, (center_end, 0), (w, h), (0, 0, 255), -1)

        cv2.addWeighted(overlay, self.zone_shade_opacity, frame, 1.0 - self.zone_shade_opacity, 0, frame)

    def _draw_nav_overlays(self, frame, zone_risks, nav_command, w):
        """Draw text summaries of risks and the final command."""
        # Zone risks text
        risk_text = f"L: {zone_risks.get('left', 'None')} | C: {zone_risks.get('center', 'None')} | R: {zone_risks.get('right', 'None')}"
        cv2.putText(frame, risk_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Command text
        if nav_command:
            cmd_color = (0, 0, 255) if "Stop" in nav_command else (0, 255, 0)
            text_size = cv2.getTextSize(nav_command, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, nav_command, (text_x, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, cmd_color, 2)
