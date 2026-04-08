import time
import cv2
from typing import Optional

from app.config.pipeline_schema import AppConfig
from app.sources.base import FrameSource
from app.pipeline.frame_context import FrameContext
from app.pipeline.orchestrator import PipelineOrchestrator
from app.metrics.tracker import MetricsTracker

class LiveRunner:
    """
    Main execution loop for real-time processing of camera or video stream inputs.
    """

    def __init__(self, config: AppConfig, orchestrator: PipelineOrchestrator, metrics_tracker: MetricsTracker, source: FrameSource):
        self.config = config
        self.orchestrator = orchestrator
        self.tracker = metrics_tracker
        self.source = source

    def run(self):
        if not self.source.open():
            raise RuntimeError(f"Cannot open live source.")

        print("Press 'q' to stop.")

        frame_idx = 0
        run_start_time = time.time()

        try:
            while True:
                loop_start = time.perf_counter()

                capture_start = time.perf_counter()
                ok, frame_bgr, source_id = self.source.read()
                capture_ms = (time.perf_counter() - capture_start) * 1000.0

                if not ok:
                    break

                frame_idx += 1

                ctx = FrameContext(frame_idx=frame_idx, source_id=source_id, frame_bgr=frame_bgr)

                # Execute pipeline
                ctx = self.orchestrator.process_frame(ctx)

                display_ms = 0.0
                if self.config.pipeline.show_windows:
                    display_start = time.perf_counter()

                    if ctx.annotated_frame is not None:
                        try:
                            cv2.imshow("Depth + BBoxes + Navigation", ctx.annotated_frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                display_ms = (time.perf_counter() - display_start) * 1000.0
                                # Add timings
                                ctx.metrics["capture_latency_ms"] = capture_ms
                                ctx.metrics["display_latency_ms"] = display_ms
                                ctx.metrics["loop_total_latency_ms"] = (time.perf_counter() - loop_start) * 1000.0
                                ctx.metrics["fps_instant"] = 1000.0 / ctx.metrics["loop_total_latency_ms"]

                                self.tracker.log_frame(ctx)
                                break
                        except cv2.error:
                            print("OpenCV GUI is not available. Continuing without preview windows.")

                    display_ms = (time.perf_counter() - display_start) * 1000.0

                # Add loop timing
                ctx.metrics["capture_latency_ms"] = capture_ms
                ctx.metrics["display_latency_ms"] = display_ms
                ctx.metrics["loop_total_latency_ms"] = (time.perf_counter() - loop_start) * 1000.0
                if ctx.metrics["loop_total_latency_ms"] > 0:
                    ctx.metrics["fps_instant"] = 1000.0 / ctx.metrics["loop_total_latency_ms"]

                # Hand over to metrics tracker
                self.tracker.log_frame(ctx)

        except KeyboardInterrupt:
            print("Stopped by user (Ctrl+C).")
        finally:
            self.source.close()
            cv2.destroyAllWindows()
            self.tracker.finalize()
            print(f"Saved run outputs to: {self.tracker.run_dir}")
