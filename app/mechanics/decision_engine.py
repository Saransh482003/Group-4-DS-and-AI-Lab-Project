import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

class RiskLevel(Enum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

@dataclass
class DetectionRisk:
    """Detailed risk assessment for a single detection."""
    class_name: str
    bbox: List[int]
    distance: Optional[float]
    urgency_level: RiskLevel
    risk_score: float

@dataclass
class DecisionResult:
    """Final output of the DecisionEngine."""
    command: str
    zone_risks: Dict[str, float]
    max_urgency: RiskLevel
    critical_count: int
    detections_with_risk: List[DetectionRisk] = field(default_factory=list)
    should_speak: bool = False
    tts_message: Optional[str] = None
    latency_ms: float = 0.0

class DecisionEngine:
    """
    Advanced decision-making engine that combines navigation logic with 
    TTS orchestration and urgency-based assessment.
    """

    def __init__(
        self, 
        navigation_logic, # Reference to NavigationLogic or similar
        tts_controller=None, # Reference to TTSRuntimeController
        critical_distance: float = 1.2,
        warning_distance: float = 2.5
    ):
        self.nav_logic = navigation_logic
        self.tts_controller = tts_controller
        self.critical_distance = critical_distance
        self.warning_distance = warning_distance
        
        self._last_decision_time = 0.0
        self._last_command = None
        self._frame_count = 0

    def process(self, nav_detections: List[Dict[str, Any]], timestamp: Optional[float] = None) -> DecisionResult:
        """
        Processes detections to make a navigation decision and determine TTS needs.
        """
        start_time = time.perf_counter()
        if timestamp is None:
            timestamp = time.time()
        
        self._frame_count += 1
        
        # 1. Assess individual risks (TTC-like distance scoring)
        detailed_risks = []
        critical_count = 0
        max_risk_level = RiskLevel.OK
        
        for det in nav_detections:
            dist = det.get("distance")
            level = RiskLevel.OK
            score = 0.0
            
            if dist is not None:
                if dist < self.critical_distance:
                    level = RiskLevel.CRITICAL
                    critical_count += 1
                    score = 10.0 / max(0.1, dist)
                elif dist < self.warning_distance:
                    level = RiskLevel.WARNING
                    score = 5.0 / max(0.1, dist)
                
                # Update max_risk_level safely
                if level == RiskLevel.CRITICAL:
                    max_risk_level = RiskLevel.CRITICAL
                elif level == RiskLevel.WARNING and max_risk_level == RiskLevel.OK:
                    max_risk_level = RiskLevel.WARNING
            
            detailed_risks.append(DetectionRisk(
                class_name=det["class"],
                bbox=det["bbox"],
                distance=dist,
                urgency_level=level,
                risk_score=score
            ))

        # 2. Delegate to primary navigation logic for zone analysis and command generation
        zone_risks, command = self.nav_logic.process_detections(nav_detections, timestamp)

        # 3. Decision for TTS - smarter orchestration
        should_speak = False
        tts_message = None
        
        if self.tts_controller:
            # We check the controller's internal logic for whether it SHOULD speak
            # given previous frame history and current command.
            tts_res = self.tts_controller.handle_command(command)
            should_speak = tts_res.get("tts_should_speak", False)
            if should_speak:
                tts_message = command # Usually speak the command

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        return DecisionResult(
            command=command,
            zone_risks=zone_risks,
            max_urgency=max_risk_level,
            critical_count=critical_count,
            detections_with_risk=detailed_risks,
            should_speak=should_speak,
            tts_message=tts_message,
            latency_ms=latency_ms
        )
