from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class FrameContext:
    """
    Data container that holds the state, outputs, and metrics of a single frame as it passes through the pipeline.
    """

    # Core identifying info
    frame_idx: int
    source_id: str

    # Input frames
    frame_bgr: np.ndarray
    frame_rgb: Optional[np.ndarray] = None

    # Output annotations/rendering
    annotated_frame: Optional[np.ndarray] = None

    # Component outputs
    detections: List[Dict[str, Any]] = field(default_factory=list)

    depth_map: Optional[np.ndarray] = None
    depth_color: Optional[np.ndarray] = None

    nav_detections: List[Dict[str, Any]] = field(default_factory=list)
    zone_risks: Dict[str, str] = field(default_factory=dict)
    nav_command: str = ""

    tts_result: Dict[str, Any] = field(default_factory=dict)

    # Metrics attached throughout the pipeline
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Always initialize metrics dictionary to avoid NoneType errors
        if self.metrics is None:
            self.metrics = {}
