import time
from typing import Dict, Any, Optional

from app.pipeline.frame_context import FrameContext
from app.pipeline.executor import PipelineExecutor, SequentialExecutor
from app.components.detector import DetectorComponent
from app.components.depth import DepthComponent
from app.components.navigation import NavigationComponent
from app.components.visualization import VisualizationComponent
from app.components.tts import TTSComponent

class PipelineOrchestrator:
    """Coordinates the execution of components per frame according to the chosen executor."""

    def __init__(
        self,
        executor: PipelineExecutor,
        components: Dict[str, Any]
    ):
        self.executor = executor
        self.components = components

    def process_frame(self, ctx: FrameContext) -> FrameContext:
        start_time = time.perf_counter()

        # 1. Detection + Depth + Fusion (handled by executor)
        self.executor.execute(ctx, self.components)

        # 2. Navigation
        navigation = self.components.get("navigation")
        if navigation:
            navigation.run(ctx)

        # 3. Visualization
        visualization = self.components.get("visualization")
        if visualization:
            visualization.run(ctx)

        # 4. TTS (Optional)
        tts = self.components.get("tts")
        if tts:
            tts.run(ctx)

        ctx.metrics["frame_total_latency_ms"] = (time.perf_counter() - start_time) * 1000.0

        return ctx
