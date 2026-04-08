import time
from typing import Optional, Any
from app.pipeline.frame_context import FrameContext
from app.mechanics.navigation_logic import NavigationLogic
from app.mechanics.decision_engine import DecisionEngine

class NavigationComponent:
    """
    Advanced navigation component that orchestrates decision-making and 
    determines if TTS feedback should be triggered.
    """
    def __init__(self, frame_width: int, tts_component: Optional[Any] = None):
        self.logic = NavigationLogic(frame_width=frame_width)
        self.tts_comp = tts_component
        
        # Initialize the advanced decision engine
        self.decision_engine = DecisionEngine(
            navigation_logic=self.logic,
            tts_controller=self.tts_comp.controller if self.tts_comp else None
        )

    def run(self, ctx: FrameContext) -> None:
        start_time = time.perf_counter()

        # Execute decision engine with enriched detections from fusion
        decision = self.decision_engine.process(ctx.nav_detections)

        # Update context with decision results
        ctx.zone_risks = decision.zone_risks
        ctx.nav_command = decision.command
        
        # Store detailed assessment in context for visualization/logging
        ctx.metrics["max_urgency"] = decision.max_urgency.value
        ctx.metrics["critical_count"] = decision.critical_count
        ctx.metrics["decision_latency_ms"] = decision.latency_ms
        
        # Orchestrate TTS - if decision engine says we should speak, 
        # we can flag it here. Note: the TTSComponent in the pipeline 
        # will actually perform the synthesis if it remains in the chain, 
        # OR we can let this component handle the async dispatch if needed.
        ctx.metrics["should_speak"] = decision.should_speak
        
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        ctx.metrics["navigation_latency_ms"] = latency_ms
        ctx.metrics["nav_command"] = decision.command
        
        for k, v in decision.zone_risks.items():
            ctx.metrics[f"{k}_risk"] = v
