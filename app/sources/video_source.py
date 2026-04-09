import cv2
import os
from typing import Tuple, Any, Optional
from app.sources.base import FrameSource

class VideoFileFrameSource(FrameSource):
    """Reads frames from a video file."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.cap = None
        self.source_id = os.path.basename(filepath)

    def open(self) -> bool:
        if not os.path.isfile(self.filepath):
            return False

        self.cap = cv2.VideoCapture(self.filepath)
        return self.cap.isOpened()

    def read(self) -> Tuple[bool, Any, Optional[str]]:
        if self.cap is None:
            return False, None, None

        ok, frame_bgr = self.cap.read()
        return ok, frame_bgr, self.source_id

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
